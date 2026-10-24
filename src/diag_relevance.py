"""How well does question->context q.k similarity identify gold tokens, per layer?

For each GDN layer (the proposal's estimator: per-head cosine(question q, context k), max over question tokens)
and each full-attention layer (pre-RoPE q.k), report token-level AUROC (gold vs other context tokens) for
  - mean over heads of per-head AUROC
  - best single head
  - head-mean score
and recall of gold tokens inside the top-`frac` mask after forward dilation (what the intervention keeps).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_ctl
from gdn_ctl import CTL
from run_eval import ROOT, Runner, build_samples, token_masks


def auroc(score, label):
    """score [H, N], label bool [N] -> [H]"""
    ranks = score.argsort(-1).argsort(-1).float() + 1
    n1 = label.sum().float()
    n0 = label.numel() - n1
    return (ranks[:, label].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ctx", type=int, default=16000)
    ap.add_argument("--fmt", default="both")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--needles", type=int, default=16)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R = Runner(os.path.join(ROOT, "models/Qwen3.5-4B"), "cuda:0")
    cap = gdn_ctl.AttnCapture(R.model.model, R.attn)
    samples = build_samples(R.tok, args)
    res = []
    for si, s in enumerate(samples):
        ids, masks, R.split = token_masks(R.tok, s)
        R._set_ctl(None, masks)
        CTL.capture, CTL.cap_layers = {}, None
        R._fwd(ids[:, : R.split])
        gold = masks["gold"][: R.split][masks["context"][: R.split]].to(R.dev)
        row = {}
        scores = {f"gdn{L}": CTL.capture[L].float().to(R.dev) for L in R.gdn}
        for L in R.attn:
            scores[f"attn{L}"] = cap.relevance(L, CTL.qidx, CTL.ctx)
        for name, r in scores.items():
            a = auroc(r, gold)
            am = auroc(r.mean(0, keepdim=True), gold)[0]
            mk = gdn_ctl.topk_mask(r, args.frac, args.win)          # per head
            mkm = gdn_ctl.topk_mask(r.mean(0, keepdim=True), args.frac, args.win)[0]
            row[name] = dict(auc_mean=a.mean().item(), auc_best=a.max().item(), auc_headmean=am.item(),
                             rec_head=mk[:, gold].mean().item(), kept_head=mk.mean().item(),
                             rec_hm=mkm[gold].mean().item(), kept_hm=mkm.mean().item())
        res.append(row)
        cap.clear()
        CTL.reset()
        print(si, "done", flush=True)
    json.dump(res, open(args.out, "w"))
    keys = list(res[0].keys())
    print(f"{'layer':<8}{'aucMean':>8}{'aucBest':>8}{'aucHM':>8}{'recHead':>8}{'keptHead':>9}{'recHM':>8}{'keptHM':>8}")
    for k in keys:
        v = {m: sum(r[k][m] for r in res) / len(res) for m in res[0][k]}
        print(f"{k:<8}{v['auc_mean']:8.3f}{v['auc_best']:8.3f}{v['auc_headmean']:8.3f}{v['rec_head']:8.3f}"
              f"{v['kept_head']:9.3f}{v['rec_hm']:8.3f}{v['kept_hm']:8.3f}")


if __name__ == "__main__":
    main()
