"""Static-batch greedy DFlash speculative decoding for hybrid Qwen3.5 targets (Gated DeltaNet + full attention).

Per-sequence state lives in batch buffers:
  attention layers : static KV buffers [B, Hkv, Lmax, D] + per-sequence committed length `start`
  GDN layers       : conv state [B, C, k-1] (bf16) and recurrent state [B, HV, K, V] (fp32)
  draft layers     : static context-KV buffers indexed by absolute position (same length as the target)
Rejected tokens are handled per sequence: attention KV beyond `start` is masked and later overwritten; GDN layers
run the verify block from the committed state without updating it, and after acceptance a commit kernel replays
only the accepted tokens of each sequence (memory O(B * block) instead of one state per draft position).
Prefill runs the stock HF forward one prompt at a time and copies its cache into the batch buffers.
block_size=1 runs plain autoregressive decoding on the same engine (the baseline).
"""
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from dflash_qwen35 import extract_context_feature  # also puts third_party/dflash on sys.path


@triton.jit(do_not_specialize=["T"])
def _gdn_commit_kernel(k, v, g, beta, h, lens, T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """In place: h[n] <- state after the first lens[n] tokens of the block (same math as fla fused_recurrent)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    n_t = tl.load(lens + i_n)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    bos = i_n * T
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    for _ in range(0, n_t):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_beta = tl.load(p_b).to(tl.float32)
        b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
    tl.store(p_h, b_h, mask=mask_h)


def gdn_commit(k, v, g, beta, h, lens):
    k, v, g, beta = (x.contiguous() for x in (k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    grid = (triton.cdiv(V, 8), B * HV)
    _gdn_commit_kernel[grid](k, v, g, beta, h, lens.to(torch.int32), T, HV=HV, K=K, V=V,
                             BK=triton.next_power_of_2(K), BV=8, USE_QK_L2NORM=True, num_warps=1, num_stages=3)


def _rope(x, cos, sin):
    """x [B, H, L, D]; cos/sin [B, L, rot]. Rotates the first `rot` dims (as HF apply_rotary_pos_emb)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + M.rotate_half(xr) * sin, xp], dim=-1)


class BatchDFlash:
    def __init__(self, target, draft):
        self.t, self.d = target, draft
        self.cfg = target.config
        self.dev = target.device
        self.layers = target.model.layers
        self.gdn = [i for i, l in enumerate(self.layers) if l.block_type == "linear_attention"]
        self.att = [i for i, l in enumerate(self.layers) if l.block_type == "full_attention"]
        self.feat_ids = set(draft.target_layer_ids)
        self.mask_id = draft.mask_token_id

    # ------------------------------------------------------------ state
    def _alloc(self, B, with_draft):
        c, dev, L = self.cfg, self.dev, self.max_len
        self.K = {i: torch.zeros(B, c.num_key_value_heads, L, c.head_dim, device=dev, dtype=torch.bfloat16)
                  for i in self.att}
        self.V = {i: torch.zeros_like(self.K[i]) for i in self.att}
        conv_dim = c.linear_key_head_dim * c.linear_num_key_heads * 2 + c.linear_value_head_dim * c.linear_num_value_heads
        self.conv = {i: torch.zeros(B, conv_dim, c.linear_conv_kernel_dim - 1, device=dev, dtype=torch.bfloat16)
                     for i in self.gdn}
        self.rec = {i: torch.zeros(B, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                                   device=dev, dtype=torch.float32) for i in self.gdn}
        dc = self.d.config
        n = len(self.d.layers) if with_draft else 0
        self.DK = {j: torch.zeros(B, dc.num_key_value_heads, L, dc.head_dim, device=dev, dtype=torch.bfloat16)
                   for j in range(n)}
        self.DV = {j: torch.zeros_like(self.DK[j]) for j in range(n)}

    def _select(self, keep):
        # one tensor at a time so the old buffer is freed before the next copy
        for dct in (self.K, self.V, self.conv, self.rec, self.DK, self.DV):
            for i in list(dct):
                dct[i] = dct[i][keep]

    def _free(self):
        self.K = self.V = self.conv = self.rec = self.DK = self.DV = None
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _prefill(self, prompts, need_feat):
        """Stock HF forward per prompt; copy caches into the batch buffers. Returns first tokens and features."""
        first, feats = [], []
        for b, ids in enumerate(prompts):
            cache = DynamicCache(config=self.cfg)
            out = self.t(ids[None].to(self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1,
                         output_hidden_states=need_feat)
            n = ids.shape[0]
            for i in self.att:
                self.K[i][b, :, :n] = cache.layers[i].keys[0]
                self.V[i][b, :, :n] = cache.layers[i].values[0]
            for i in self.gdn:
                self.conv[i][b] = cache.layers[i].conv_states[0][0, :, -self.conv[i].shape[-1]:]
                self.rec[i][b] = cache.layers[i].recurrent_states[0][0]
            first.append(out.logits[0, -1].argmax())
            if need_feat:
                feats.append(extract_context_feature(out.hidden_states, self.d.target_layer_ids)[0])
            del cache, out
        return torch.stack(first), feats

    # ------------------------------------------------------------ target step
    def _gdn(self, i, m, x, verify, save):
        B, T, _ = x.shape
        mixed = m.in_proj_qkv(x).transpose(1, 2)
        z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
        beta = m.in_proj_b(x).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
        full = torch.cat([self.conv[i], mixed], dim=-1)                   # [B, C, k-1+T] conv inputs
        co = F.silu(F.conv1d(full.to(m.conv1d.weight.dtype), m.conv1d.weight, None, groups=full.shape[1]))
        q, k, v = torch.split(co.transpose(1, 2), [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, T, -1, m.head_k_dim)
        k = k.reshape(B, T, -1, m.head_k_dim)
        v = v.reshape(B, T, -1, m.head_v_dim)
        rep = m.num_v_heads // m.num_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        o, h_new = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=self.rec[i],
                                                    output_final_state=not verify, use_qk_l2norm_in_kernel=True)
        if verify:
            save[i] = (k, v, g, beta, full)
        else:
            self.rec[i].copy_(h_new)
            self.conv[i] = full[..., -self.conv[i].shape[-1]:].contiguous()
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
        return m.out_proj(o)

    def _attn(self, i, a, x, pos, cos, sin, L):
        B, T, _ = x.shape
        hd = a.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        bi = torch.arange(B, device=x.device)[:, None]
        self.K[i][bi, :, pos] = k.transpose(1, 2)
        self.V[i][bi, :, pos] = v.transpose(1, 2)
        mask = torch.arange(L, device=x.device)[None, None, :] <= pos[:, :, None]          # [B, T, L]
        o = F.scaled_dot_product_attention(q, self.K[i][:, :, :L], self.V[i][:, :, :L], attn_mask=mask[:, None],
                                           scale=a.scaling, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def _target_step(self, ids, pos, verify):
        """ids, pos: [B, T]. Returns greedy tokens [B, T], draft features [B, T, F] (verify only), saved GDN inputs."""
        h = self.t.model.embed_tokens(ids)
        cos, sin = self.t.model.rotary_emb(h, pos[None].expand(3, -1, -1))
        L = int(pos.max()) + 1
        save, feats = {}, []
        for i, layer in enumerate(self.layers):
            r = h
            x = layer.input_layernorm(h)
            if layer.block_type == "linear_attention":
                x = self._gdn(i, layer.linear_attn, x, verify, save)
            else:
                x = self._attn(i, layer.self_attn, x, pos, cos, sin, L)
            h = r + x
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if verify and i in self.feat_ids:
                feats.append(h)
        tokens = self.t.lm_head(self.t.model.norm(h)).argmax(-1)
        return tokens, (torch.cat(feats, -1) if verify else None), save

    def _commit(self, save, produced):
        """Advance every GDN state by the `produced` accepted tokens of its sequence."""
        for i, (k, v, g, beta, full) in save.items():
            gdn_commit(k, v, g, beta, self.rec[i], produced)
            km1 = self.conv[i].shape[-1]
            idx = (produced[:, None] + torch.arange(km1, device=self.dev)[None])[:, None, :]
            self.conv[i] = torch.gather(full, 2, idx.expand(-1, full.shape[1], -1))

    # ------------------------------------------------------------ draft step
    def _draft_step(self, ctx_feat, ctx_cnt, start, block_ids):
        """ctx_feat [B, C, F]: target features of the newest context tokens (first ctx_cnt valid), positions
        [start - ctx_cnt, start). block_ids [B, S] at positions [start, start + S). Returns draft tokens [B, S-1]."""
        d = self.d
        B, C, _ = ctx_feat.shape
        S = block_ids.shape[1]
        dev = self.dev
        th = d.hidden_norm(d.fc(ctx_feat))
        x = F.embedding(block_ids, self.t.get_input_embeddings().weight)
        pos_c = (start - ctx_cnt)[:, None] + torch.arange(C, device=dev)[None]
        pos_b = start[:, None] + torch.arange(S, device=dev)[None]
        cos_c, sin_c = d.rotary_emb(x, pos_c)
        cos_b, sin_b = d.rotary_emb(x, pos_b)
        Lc = int(start.max())
        key_pos = torch.arange(Lc, device=dev)[None, None, :]
        valid_c = (key_pos < start[:, None, None]).expand(B, S, Lc)               # committed context only
        near = pos_b[:, :, None] - key_pos                                        # query pos - key pos
        tri = torch.ones(S, S, dtype=torch.bool, device=dev).tril()[None].expand(B, S, S)
        full_blk = torch.ones(B, S, S, dtype=torch.bool, device=dev)
        bi = torch.arange(B, device=dev)[:, None]
        # padded context entries (j >= ctx_cnt) land at positions >= start: masked now, overwritten next step
        pos_w = pos_c.clamp(max=self.max_len - 1)
        for li, layer in enumerate(d.layers):
            a = layer.self_attn
            hd = a.head_dim
            hs = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(hs).view(B, S, -1, hd)).transpose(1, 2)
            kc = a.k_norm(a.k_proj(th).view(B, C, -1, hd)).transpose(1, 2)
            vc = a.v_proj(th).view(B, C, -1, hd).transpose(1, 2)
            kb = a.k_norm(a.k_proj(hs).view(B, S, -1, hd)).transpose(1, 2)
            vb = a.v_proj(hs).view(B, S, -1, hd).transpose(1, 2)
            q, kb, kc = _rope(q, cos_b, sin_b), _rope(kb, cos_b, sin_b), _rope(kc, cos_c, sin_c)
            self.DK[li][bi, :, pos_w] = kc.transpose(1, 2)
            self.DV[li][bi, :, pos_w] = vc.transpose(1, 2)
            mc = valid_c if a.sliding_window is None else valid_c & (near < a.sliding_window)
            mask = torch.cat([mc, tri if a.is_causal else full_blk], dim=-1)[:, None]
            keys = torch.cat([self.DK[li][:, :, :Lc], kb], dim=2)
            vals = torch.cat([self.DV[li][:, :, :Lc], vb], dim=2)
            o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=a.scaling, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, S, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return d.compute_logits(d.norm(x)[:, 1:], self.t.lm_head).argmax(-1)

    # ------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens, stop_ids, block_size):
        """Greedy decoding of a static batch. prompts: list of 1-D LongTensors.
        Returns per-prompt generated ids (incl. the stop token, as dflash_generate) and timing / acceptance stats."""
        B = len(prompts)
        dev = self.dev
        dflash = block_size > 1
        S = block_size
        n_prompt = torch.tensor([p.shape[0] for p in prompts], device=dev)
        self.max_len = int(n_prompt.max()) + max_new_tokens + 2 * S + 1
        torch.cuda.reset_peak_memory_stats()
        self._alloc(B, dflash)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first, feats = self._prefill(prompts, dflash)
        start = n_prompt.clone()                                   # tokens committed to the target cache
        max_total = n_prompt + max_new_tokens
        out = torch.full((B, self.max_len), self.mask_id, dtype=torch.long, device=dev)
        for b, p in enumerate(prompts):
            out[b, : p.shape[0]] = p.to(dev)
        out[torch.arange(B, device=dev), start] = first            # pending token, not yet in the cache
        if dflash:
            ctx_feat = torch.zeros(B, int(n_prompt.max()), feats[0].shape[-1], device=dev, dtype=feats[0].dtype)
            for b, f in enumerate(feats):
                ctx_feat[b, : f.shape[0]] = f
            ctx_cnt = n_prompt.clone()
            del feats
        stop_t = torch.tensor(stop_ids, device=dev)
        alive = torch.arange(B, device=dev)
        finished = torch.isin(first, stop_t) | (start + 1 >= max_total)
        results, accept_log = [None] * B, [[] for _ in range(B)]
        jS = torch.arange(S, device=dev)[None]
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        t_draft = t_verify = 0.0
        steps, batch_sizes = 0, []
        t1 = time.perf_counter()
        while True:
            if finished.any():
                for r in finished.nonzero().flatten().tolist():
                    results[alive[r].item()] = out[r, n_prompt[r].item(): start[r].item() + 1].tolist()
                keep = (~finished).nonzero().flatten()
                if keep.numel() == 0:
                    break
                self._select(keep)
                alive, out, start, n_prompt, max_total = (x[keep] for x in (alive, out, start, n_prompt, max_total))
                if dflash:
                    ctx_feat, ctx_cnt = ctx_feat[keep], ctx_cnt[keep]
            Bc = alive.numel()
            bi = torch.arange(Bc, device=dev)
            pos = start[:, None] + jS
            block = out[bi[:, None], pos].clone()                   # [pending token, mask, mask, ...]
            ts = time.perf_counter()
            if dflash:
                block[:, 1:] = self._draft_step(ctx_feat, ctx_cnt, start, block)
                torch.cuda.synchronize()
            tv = time.perf_counter()
            post, f_new, save = self._target_step(block, pos, verify=dflash)
            acc = (block[:, 1:] == post[:, :-1]).long().cumprod(1).sum(1) if dflash else torch.zeros_like(start)
            bonus = post[bi, acc]
            # new tokens at start+1 .. start+acc+1: accepted drafts then the target's bonus token
            new = torch.where(jS < acc[:, None], F.pad(block[:, 1:], (0, 1)), bonus[:, None])
            produced = torch.minimum(acc + 1, max_total - start - 1)
            is_stop = torch.isin(new, stop_t) & (jS < produced[:, None])
            has_stop = is_stop.any(1)
            produced = torch.where(has_stop, is_stop.long().argmax(1) + 1, produced)
            wmask = jS < produced[:, None]
            out[bi[:, None].expand(-1, S)[wmask], (start[:, None] + 1 + jS)[wmask]] = new[wmask]
            if dflash:
                self._commit(save, produced)
                ctx_feat, ctx_cnt = f_new, produced
            start = start + produced
            finished = has_stop | (start + 1 >= max_total)
            for r, p in zip(alive.tolist(), produced.tolist()):
                accept_log[r].append(p)
            steps += 1
            batch_sizes.append(Bc)
            t_draft += tv - ts
            t_verify += time.perf_counter() - tv
            del save, post, f_new
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t1
        self._free()
        return SimpleNamespace(outputs=results, accept=accept_log, n_out=sum(len(r) for r in results), steps=steps,
                               batch_sizes=batch_sizes, t_prefill=t_prefill, t_decode=t_decode, t_draft=t_draft,
                               t_verify=t_verify, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
