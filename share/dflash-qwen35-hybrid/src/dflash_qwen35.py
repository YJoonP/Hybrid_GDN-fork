"""DFlash speculative decoding for hybrid Qwen3.5 targets (Gated DeltaNet + full attention) on Transformers.

The official Transformers backend (third_party/dflash) rolls the target cache back with `DynamicCache.crop`, which
restores attention KV and GDN conv states but NOT the GDN recurrent states, so rejected draft tokens would stay
written in the state. Here the verify pass runs GDN with a recurrent kernel that keeps the state after every
position of the block, and after acceptance the recurrent state is reset to the one after the last kept token.
"""
import os
import sys
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party/dflash"))
from dflash.model import (  # noqa: E402
    DFlash2DraftModel,
    _crop_to,
    _cuda_time,
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    _rejection_sample,
    _sample_probs,
    _sampling_probs,
    _validate_sampling,
    extract_context_feature,
    sample,
)


# ---------------------------------------------------------------- kernel
@triton.jit(do_not_specialize=["T"])
def _gdn_fwd_states_kernel(q, k, v, g, beta, o, h0, hs, scale, T,
                           HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                           BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """Gated delta rule, token by token (as fla fused_recurrent), also storing the state after each token."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    bos = i_n * T
    p_q = q + (bos * HV + i_hv) * K + o_k
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_h = hs + (bos * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mask_h, other=0).to(tl.float32)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_b).to(tl.float32)
        b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_h, b_h, mask=mask_h)
        p_q += HV * K
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
        p_o += HV * V
        p_h += HV * K * V


def gdn_recurrent_with_states(q, k, v, g, beta, h0, scale=None, use_qk_l2norm=True):
    """q, k: [B, T, HV, K]; v: [B, T, HV, V]; g, beta: [B, T, HV]; h0: [B, HV, K, V] fp32.
    Returns o [B, T, HV, V] and hs [B, T, HV, K, V] (fp32 state after each token)."""
    q, k, v, g, beta, h0 = (x.contiguous() for x in (q, k, v, g, beta, h0))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    o = torch.empty_like(v)
    hs = torch.empty(B, T, HV, K, V, device=v.device, dtype=torch.float32)
    BK, BV = triton.next_power_of_2(K), 8
    grid = (triton.cdiv(V, BV), B * HV)
    _gdn_fwd_states_kernel[grid](q, k, v, g, beta, o, h0, hs, scale, T, HV=HV, K=K, V=V, BK=BK, BV=BV,
                                 USE_QK_L2NORM=use_qk_l2norm, num_warps=1, num_stages=3)
    return o, hs


# ---------------------------------------------------------------- verify-mode patch
class _Verify:
    active = False
    layer = None
    states = {}   # layer_idx -> hs [B, T, HV, K, V]


VERIFY = _Verify()
_orig_chunk = M.torch_chunk_gated_delta_rule


def _patched_chunk(query, key, value, g=None, beta=None, initial_state=None, output_final_state=False,
                   use_qk_l2norm_in_kernel=False, **kw):
    if VERIFY.active and initial_state is not None:
        o, hs = gdn_recurrent_with_states(query, key, value, g, beta, initial_state,
                                          use_qk_l2norm=use_qk_l2norm_in_kernel)
        VERIFY.states[VERIFY.layer] = hs
        return o, hs[:, -1]
    return _orig_chunk(query, key, value, g=g, beta=beta, initial_state=initial_state,
                       output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)


def install(target):
    M.torch_chunk_gated_delta_rule = _patched_chunk
    layers = []
    for mod in target.modules():
        if isinstance(mod, M.Qwen3_5GatedDeltaNet):
            mod.register_forward_pre_hook(lambda m, a: setattr(VERIFY, "layer", m.layer_idx))
            layers.append(mod.layer_idx)
    return layers


def rollback_gdn(cache, kept):
    """Set every GDN recurrent state to the state after `kept` tokens of the last verify block."""
    for i, hs in VERIFY.states.items():
        cache.layers[i].recurrent_states[0].copy_(hs[:, kept - 1])
    VERIFY.states = {}


# ---------------------------------------------------------------- generation (official loop + GDN rollback)
@torch.inference_mode()
def dflash_generate_hybrid(model, target, input_ids, max_new_tokens, stop_token_ids, temperature=0.0, top_p=1.0,
                           top_k=0, block_size=None, return_stats=False, rollback=True):
    """Same as dflash.model.dflash_generate, plus GDN recurrent-state rollback after each verify step.
    rollback=False reproduces the official Transformers behaviour (recurrent state not rolled back)."""
    _validate_sampling(temperature, top_p, top_k)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size

    output_ids = torch.full((1, max_length + 1), model.mask_token_id, dtype=torch.long, device=target.device)
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = _make_cache(target.config)
    past_key_values_draft = _make_cache(model.config)

    prefill_start = _cuda_time() if return_stats else None
    output = target(input_ids, position_ids=position_ids[:, :num_input_tokens], past_key_values=past_key_values_target,
                    use_cache=True, logits_to_keep=1, output_hidden_states=block_size > 1)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature, top_p, top_k)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    _crop_to(past_key_values_target, num_input_tokens)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths, draft_time, verify_time = [], 0.0, 0.0
    start = num_input_tokens
    stop_tokens = (torch.tensor(stop_token_ids, dtype=output_ids.dtype, device=output_ids.device)
                   if stop_token_ids is not None else None)
    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()
    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start:start + verify_size].clone()
        block_position_ids = position_ids[:, start:start + verify_size]
        t0 = _cuda_time() if return_stats else None
        if verify_size > 1:
            noise_embedding = _raw_input_embeddings(target, block_output_ids,
                                                    float(_draft_value(model.config, "input_embedding_scale", 1.0)))
            draft_hidden = model(target_hidden=target_hidden, noise_embedding=noise_embedding,
                                 position_ids=position_ids[:, start - target_hidden.shape[1]:start + verify_size],
                                 past_key_values=past_key_values_draft, use_cache=True)[:, 1 - verify_size:, :]
            _crop_to(past_key_values_draft, start)
            if isinstance(model, DFlash2DraftModel):
                draft_tokens, draft_indices, draft_probs = model.propose(
                    draft_hidden, block_output_ids[:, 0], _output_head(target), temperature)
                block_output_ids[:, 1:] = draft_tokens
            else:
                draft_logits = model.compute_logits(draft_hidden, _output_head(target))
                if temperature > 0:
                    draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)
                    block_output_ids[:, 1:] = _sample_probs(draft_probs)
                    draft_indices = None
                else:
                    block_output_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
        t1 = _cuda_time() if return_stats else None
        VERIFY.active = rollback and verify_size > 1
        output = target(block_output_ids, position_ids=block_position_ids, past_key_values=past_key_values_target,
                        use_cache=True, output_hidden_states=verify_size > 1)
        VERIFY.active = False

        if temperature > 0:
            target_probs = _sampling_probs(output.logits, temperature, top_p, top_k)
            if verify_size > 1:
                acceptance_length, bonus = _rejection_sample(block_output_ids[:, 1:], target_probs, draft_probs,
                                                             draft_indices)
            else:
                acceptance_length = 0
                bonus = _sample_probs(target_probs[:, -1])[0]
        else:
            posterior = torch.argmax(output.logits, dim=-1)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            bonus = posterior[:, acceptance_length][0]
        output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = bonus
        produced = min(acceptance_length + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(output_ids[0, start + 1:start + produced + 1], stop_tokens).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True
        start += produced
        _crop_to(past_key_values_target, start)
        if rollback and verify_size > 1:
            rollback_gdn(past_key_values_target, produced)
        VERIFY.states = {}
        acceptance_lengths.append(produced)
        if verify_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :produced, :]
        if return_stats:
            t2 = _cuda_time()
            draft_time += t1 - t0
            verify_time += t2 - t1

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(output_ids=output_ids, num_input_tokens=num_input_tokens,
                           num_output_tokens=num_output_tokens, time_to_first_token=time_to_first_token,
                           decode_time=total_decode_time, time_per_output_token=total_decode_time / num_output_tokens,
                           acceptance_lengths=acceptance_lengths, draft_time=draft_time, verify_time=verify_time)
