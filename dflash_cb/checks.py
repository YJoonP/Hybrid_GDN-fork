"""Phase 3 correctness checks for the full-state rollback.

State check (batch 1, random prompts): at every DFlash step, compare the restored durable GDN state of each layer
  (a) with the same recurrence replayed on the very inputs the verify pass used     -> exactness of save/restore
  (b) with autoregressive decoding of the committed tokens one at a time            -> spec reference
  (c) noise floor: (b) vs the same committed tokens decoded in one multi-token AR forward (different GEMM shapes)
Pass (fixed before running): (a) max abs error == 0 for recurrent and conv states, and for every layer the relative
error of (b) <= 2 x the largest relative error of (c) over all steps ("dtype level" = measured bf16 noise floor).

Output check: first 50 questions, batch 1, DFlash vs AR, token-exact.
"""
import random

import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule


def _rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item(), ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


@torch.inference_mode()
def state_check(eng, prompts, n_prompts=10, seed=0):
    """eng: DFlash Engine with bs >= 1 (row 0 used). Returns a JSON-able dict."""
    picks = random.Random(seed).sample(range(len(prompts)), n_prompts)
    keys = ["exact_rec", "ar_rec", "ar_conv", "floor_rec", "floor_conv"]
    per_layer = {i: {k: [0.0, 0.0] for k in keys} for i in eng.gdn}   # [max abs, max rel]
    steps = 0

    def upd(i, k, err):
        per_layer[i][k][0] = max(per_layer[i][k][0], err[0])
        per_layer[i][k][1] = max(per_layer[i][k][1], err[1])

    def snap():
        return dict(rec={i: eng.rec[i][0].clone() for i in eng.gdn}, conv={i: eng.conv[i][0].clone() for i in eng.gdn},
                    K={i: eng.K[i][0].clone() for i in eng.att}, V={i: eng.V[i][0].clone() for i in eng.att})

    def load(s):
        for i in eng.gdn:
            eng.rec[i][0].copy_(s["rec"][i])
            eng.conv[i][0].copy_(s["conv"][i])
        for i in eng.att:
            eng.K[i][0].copy_(s["K"][i])
            eng.V[i][0].copy_(s["V"][i])

    for pi in picks:
        eng.reset()
        eng.insert(pi, torch.tensor(prompts[pi]))
        while eng.n > 0:
            start0 = eng.start[0].item()
            pre = snap()
            eng.rb.capture = {}
            _, prod, finished, _ = eng.step()
            cap, eng.rb.capture = eng.rb.capture, None
            p = prod[0]
            toks = eng.out[0, start0:start0 + p].clone()                 # pending token + accepted drafts
            restored = snap()
            # (a) same inputs, same recurrence (fla fused_recurrent), conv inputs from the verify pass
            for i in eng.gdn:
                q, k, v, g, beta = cap[i]
                _, h = fused_recurrent_gated_delta_rule(q[:, :p], k[:, :p], v[:, :p], g=g[:, :p], beta=beta[:, :p],
                                                        initial_state=pre["rec"][i][None], output_final_state=True,
                                                        use_qk_l2norm_in_kernel=True)
                upd(i, "exact_rec", _rel(restored["rec"][i], h[0]))
            # (b) AR, one token at a time from the pre-verify state
            load(pre)
            for j in range(p):
                eng.target_forward(toks[None, j:j + 1], torch.tensor([[start0 + j]], device=eng.dev), verify=False)
            ar = snap()
            for i in eng.gdn:
                upd(i, "ar_rec", _rel(restored["rec"][i], ar["rec"][i]))
                upd(i, "ar_conv", _rel(restored["conv"][i], ar["conv"][i]))
            # (c) noise floor: the same tokens in one multi-token AR forward vs (b)
            load(pre)
            eng.target_forward(toks[None], (start0 + torch.arange(p, device=eng.dev))[None], verify=False)
            for i in eng.gdn:
                upd(i, "floor_rec", _rel(eng.rec[i][0], ar["rec"][i]))
                upd(i, "floor_conv", _rel(eng.conv[i][0], ar["conv"][i]))
            load(restored)
            steps += 1
            if finished:
                eng.remove(0)
    res = dict(prompts=picks, steps=steps, per_layer={str(i): v for i, v in per_layer.items()})
    worst = {k: [max(v[k][0] for v in per_layer.values()), max(v[k][1] for v in per_layer.values())] for k in keys}
    res["worst"] = worst
    return res


@torch.inference_mode()
def conv_exact(eng, prompt):
    """Restored conv state must be bitwise the conv-input window ending at the last committed token."""
    eng.reset()
    eng.insert(0, torch.tensor(prompt))
    worst = 0.0
    full_cap = {}
    orig_verify = eng.rb.verify_gdn

    def spy(i, q, k, v, g, beta, conv_full, n):
        full_cap[i] = conv_full.clone()
        return orig_verify(i, q, k, v, g, beta, conv_full, n)

    eng.rb.verify_gdn = spy
    try:
        while eng.n > 0:
            _, prod, finished, _ = eng.step()
            p = prod[0]
            for i in eng.gdn:
                km1 = eng.conv[i].shape[-1]
                ref = full_cap[i][0, :, p:p + km1]
                worst = max(worst, (eng.conv[i][0].float() - ref.float()).abs().max().item())
            if finished:
                eng.remove(0)
    finally:
        eng.rb.verify_gdn = orig_verify
    return worst


def state_pass(res, conv_exact_err):
    w = res["worst"]
    ok_exact = w["exact_rec"][0] == 0.0 and conv_exact_err == 0.0
    ok_rec = all(v["ar_rec"][1] <= 2 * res["worst"]["floor_rec"][1] for v in res["per_layer"].values())
    ok_conv = all(v["ar_conv"][1] <= 2 * res["worst"]["floor_conv"][1] for v in res["per_layer"].values())
    return dict(exact=ok_exact, ar_rec_within_floor=ok_rec, ar_conv_within_floor=ok_conv,
                passed=ok_exact and ok_rec and ok_conv)


@torch.inference_mode()
def state_check_graph(eng, prompts, n_prompts=10, seed=0):
    """CUDA-graph engine (debug=True, bs=1): after every replayed step, the restored durable GDN state of row 0 must
    equal the recurrence replayed (fla fused_recurrent) on the verify pass's own inputs from the pre-step state, and the
    restored conv state must equal the conv-input window ending at the last committed token. Pass: both exactly 0."""
    picks = random.Random(seed).sample(range(len(prompts)), n_prompts)
    per_layer = {i: dict(rec=0.0, conv=0.0) for i in eng.gdn}
    steps = 0
    km1 = eng.conv[eng.gdn[0]].shape[-1]
    d = eng.dbg
    for pi in picks:
        eng.reset()
        eng.rb.reset_stats()
        eng.insert(pi, torch.tensor(prompts[pi]))
        eng.pop_ingest()
        while eng.n > 0:
            _, prod, finished, _ = eng.step()
            p = prod[0]
            for i in eng.gdn:
                k, v, g, beta = d["k"][i][:, :p], d["v"][i][:, :p], d["g"][i][:, :p], d["beta"][i][:, :p]
                _, h = fused_recurrent_gated_delta_rule(k, k, v, g=g, beta=beta, initial_state=d["pre_rec"][i],
                                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
                per_layer[i]["rec"] = max(per_layer[i]["rec"], (eng.rec[i][0] - h[0]).abs().max().item())
                ref = d["full"][i][0, :, p:p + km1]
                per_layer[i]["conv"] = max(per_layer[i]["conv"], (eng.conv[i][0].float() - ref.float()).abs().max().item())
            steps += 1
            if finished:
                eng.remove(0)
    worst = dict(rec=max(v["rec"] for v in per_layer.values()), conv=max(v["conv"] for v in per_layer.values()))
    return dict(engine="graph", prompts=picks, steps=steps, worst=worst, per_layer={str(i): v for i, v in per_layer.items()},
                verdict=dict(exact=worst["rec"] == 0.0 and worst["conv"] == 0.0,
                             passed=worst["rec"] == 0.0 and worst["conv"] == 0.0))
