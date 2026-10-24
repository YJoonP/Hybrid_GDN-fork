"""Evaluate Qwen3.5-4B on long-context tasks under GDN-state interventions.

Prefill is always split at the end of the context: part 1 = [template + question-before + context],
part 2 = [question-after + answer slot]. Interventions act on part 1 only; mechanism probes (state reset,
state/KV swap) operate on the cache between the two parts. All conditions share this path.

Condition syntax (';' separated): name[,key=val...]
  base                          no intervention
  oracle|anti|random,s=,d=,L=   precomputed mask: gold tokens kept / gold suppressed / random span of gold size
  gdn_qk,s=,d=,frac=,win=,L=    proposal: same-layer per-head question-q . context-k relevance (single pass)
  randfrac,frac=,win=,...       control: random per-head token mask with the same keep fraction / window
  betatop,frac=,win=,...        control (query-agnostic): keep tokens with the largest native beta
  uniform,s=,d=                 control: scale beta/g of every context token
  attn,src=<layer>,...          relevance from full-attention layer <layer> q.k (pass 1), head-max, shared mask
  gdnL,src=<layer>,...          relevance from GDN layer <layer> q.k (pass 1), head-mean, shared mask
  reset                         zero all GDN states (recurrent + conv) after the context
  swapS:<cond>                  GDN state from <cond>, attention KV from base  (state-only effect)
  swapK:<cond>                  attention KV from <cond>, GDN state from base   (KV-only effect)
L: all | early (layer<16) | late (layer>=16) | a-b (inclusive layer index range)
"""
import argparse
import copy
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_ctl
import tasks
from gdn_ctl import CTL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- scoring
def norm_ans(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm_ans(pred).split(), norm_ans(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def score(task, out, answers):
    if task == "hotpot":
        return max(float(norm_ans(a) in norm_ans(out)) for a in answers), max(f1(out, a) for a in answers)
    pat = r"\d+" if task == "niah" else r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    found = re.findall(pat, out)
    c = float(any(a in found for a in answers))
    return c, c


# ---------------------------------------------------------------- conditions
def parse_cond(s):
    if s.startswith(("swapS:", "swapK:")):
        return dict(name=s, kind=s[:5], inner=parse_cond(s[6:]))
    parts = s.split(",")
    c = dict(name=s, kind=parts[0], s=0.0, d=0.0, L="all", frac=0.1, win=0, src=None)
    for p in parts[1:]:
        k, v = p.split("=")
        c[k] = v if k in ("L",) else float(v) if k in ("s", "d", "frac") else int(v)
    return c


def layer_set(spec, gdn):
    if spec == "all":
        return set(gdn)
    if spec == "early":
        return {i for i in gdn if i < 16}
    if spec == "late":
        return {i for i in gdn if i >= 16}
    a, b = map(int, spec.split("-"))
    return {i for i in gdn if a <= i <= b}


# ---------------------------------------------------------------- sample construction
def token_masks(tok, sample):
    enc = tok(sample["prompt"], return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc.input_ids, enc.offset_mapping
    T = len(ids)
    starts = torch.tensor([o[0] for o in offs])
    ends = torch.tensor([o[1] for o in offs])

    def m(name):
        out = torch.zeros(T, dtype=torch.bool)
        for a, b in sample["spans"].get(name, []):
            out |= (ends > a) & (starts < b)
        return out

    masks = {k: m(k) for k in ("context", "gold", "q_before", "q_after")}
    ctx_end = int(torch.nonzero(masks["context"])[-1]) + 1
    return torch.tensor([ids]), masks, ctx_end


def random_like_gold(masks, rng):
    """Random spans inside the context, same lengths as the gold spans, not overlapping gold."""
    ctx, gold = masks["context"], masks["gold"]
    cidx = torch.nonzero(ctx).flatten()
    lo, hi = int(cidx[0]), int(cidx[-1]) + 1
    g = gold.int().tolist()
    runs, i = [], 0
    while i < len(g):
        if g[i]:
            j = i
            while j < len(g) and g[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    out = torch.zeros_like(gold)
    for n in runs:
        for _ in range(100):
            a = rng.randrange(lo, max(lo + 1, hi - n))
            if not gold[a:a + n].any() and not out[a:a + n].any():
                out[a:a + n] = True
                break
    return out


# ---------------------------------------------------------------- model runner
class Runner:
    def __init__(self, path, device):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        self.dev = self.model.device
        gdn_ctl.install(self.model)
        self.gdn = gdn_ctl.gdn_layers(self.model.model)
        self.attn = gdn_ctl.attn_layers(self.model.model)
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}

    @torch.no_grad()
    def _fwd(self, ids, cache=None):
        out = self.model(input_ids=ids.to(self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.past_key_values

    def _set_ctl(self, cond, masks, mask_vec=None):
        CTL.reset()
        CTL.ctx = masks["context"][: self.split].to(self.dev)
        CTL.qidx = torch.nonzero(masks["q_before"][: self.split]).flatten().to(self.dev)
        if cond is None:
            CTL.cfg = None
            return
        CTL.cfg = dict(src=cond["kind"], s=cond["s"], d=cond["d"], layers=layer_set(cond["L"], self.gdn),
                       frac=cond["frac"], win=cond["win"])
        CTL.mask = None if mask_vec is None else mask_vec.to(self.dev)
        CTL.seed = self.seed

    def _mask_for(self, cond, ids, masks, rng):
        k = cond["kind"]
        ctx = masks["context"][: self.split]
        if k == "oracle":
            return (~ctx | masks["gold"][: self.split]).float()
        if k == "anti":
            return (~masks["gold"][: self.split]).float()
        if k == "uniform":
            return (~ctx).float()
        if k == "random":
            return (~ctx | random_like_gold(masks, rng)[: self.split]).float()
        if k in ("attn", "gdnL"):
            L = cond["src"]
            if k == "attn":
                cap = gdn_ctl.AttnCapture(self.model.model, [L])
                self._set_ctl(None, masks)
                self._fwd(ids[:, : self.split])
                r = cap.relevance(L, CTL.qidx, CTL.ctx).amax(0)
                cap.remove()
            else:
                self._set_ctl(None, masks)
                CTL.capture, CTL.cap_layers = {}, {L}
                self._fwd(ids[:, : self.split])
                r = CTL.capture[L].float().to(self.dev).mean(0)
            m = torch.ones(self.split, device=self.dev)
            m[CTL.ctx] = gdn_ctl.topk_mask(r[None], cond["frac"], cond["win"])[0]
            return m
        return None

    def prefill_part1(self, cond, ids, masks, rng):
        if cond is None or cond["kind"] in ("base", "reset"):
            self._set_ctl(None, masks)
        else:
            mv = self._mask_for(cond, ids, masks, rng)
            self._set_ctl(cond, masks, mv)
        _, cache = self._fwd(ids[:, : self.split])
        kept = CTL.stats.get("kept_frac")
        CTL.reset()
        return cache, (sum(kept) / len(kept) if kept else 1.0)

    def run(self, cond, ids, masks, answers, max_new, rng):
        T = ids.shape[1]
        kind = cond["kind"]
        if kind in ("swapS", "swapK"):
            cache_b, _ = self.prefill_part1(None, ids, masks, rng)
            cache_i, kept = self.prefill_part1(cond["inner"], ids, masks, rng)
            src_state, src_kv = (cache_i, cache_b) if kind == "swapS" else (cache_b, cache_i)
            for i in self.gdn:
                src_kv.layers[i].conv_states[0].copy_(src_state.layers[i].conv_states[0])
                src_kv.layers[i].recurrent_states[0].copy_(src_state.layers[i].recurrent_states[0])
            cache = src_kv
            del cache_b, cache_i
        else:
            cache, kept = self.prefill_part1(cond, ids, masks, rng)
            if kind == "reset":
                for i in self.gdn:
                    cache.layers[i].conv_states[0].zero_()
                    cache.layers[i].recurrent_states[0].zero_()
        CTL.reset()
        logits, cache = self._fwd(ids[:, self.split:], cache)

        # gold-answer log-prob (teacher forced on a copy of the cache)
        a_ids = self.tok(answers[0], add_special_tokens=False).input_ids
        lp0 = torch.log_softmax(logits, -1)
        ans_lp = lp0[a_ids[0]].item()
        first_lp = ans_lp
        if len(a_ids) > 1:
            c2 = copy.deepcopy(cache)
            out = self.model(input_ids=torch.tensor([a_ids[:-1]], device=self.dev), past_key_values=c2, use_cache=True)
            lps = torch.log_softmax(out.logits[0].float(), -1)
            ans_lp += lps[torch.arange(len(a_ids) - 1), torch.tensor(a_ids[1:])].sum().item()
            del c2, out
        # greedy decode
        gen = []
        nxt = int(logits.argmax())
        for _ in range(max_new):
            if nxt in self.eos:
                break
            gen.append(nxt)
            logits, cache = self._fwd(torch.tensor([[nxt]]), cache)
            nxt = int(logits.argmax())
        del cache
        return self.tok.decode(gen), ans_lp, first_lp, len(a_ids), kept


def build_samples(tok, args):
    pool = tasks.load_paragraphs(os.path.join(ROOT, "data/para_pool.json")) if args.task != "kv" else None
    hot = None
    if args.task == "hotpot":
        with open(os.path.join(ROOT, "data/hotpot_val.jsonl")) as f:
            hot = [json.loads(l) for l in f]
        random.Random(0).shuffle(hot)
    out = []
    for i in range(args.start, args.start + args.n):
        rng = random.Random(args.seed * 1_000_003 + i * 7919 + args.ctx)
        if args.task == "niah":
            s = tasks.make_niah(tok, pool, args.ctx, args.needles, args.fmt, rng)
        elif args.task == "hotpot":
            s = tasks.make_hotpot(tok, pool, hot[i], args.ctx, args.fmt, rng)
        else:
            s = tasks.make_kv(tok, args.ctx, args.fmt, rng)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["niah", "hotpot", "kv"])
    ap.add_argument("--ctx", type=int, default=16000)
    ap.add_argument("--fmt", default="both", choices=["after", "before", "both"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--needles", type=int, default=16)
    ap.add_argument("--conds", default="base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=0)
    args = ap.parse_args()

    R = Runner(os.path.join(ROOT, "models/Qwen3.5-4B"), args.device)
    conds = [parse_cond(c) for c in args.conds.split(";")]
    max_new = args.max_new or {"niah": 16, "kv": 40, "hotpot": 32}[args.task]
    samples = build_samples(R.tok, args)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for l in f:
                r = json.loads(l)
                done.add((r["idx"], r["cond"]))
    fout = open(args.out, "a")
    for si, s in enumerate(samples):
        idx = args.start + si
        ids, masks, R.split = token_masks(R.tok, s)
        for c in conds:
            if (idx, c["name"]) in done:
                continue
            t0 = time.time()
            rng = random.Random(idx * 31 + 5)
            R.seed = idx
            out, alp, flp, alen, kept = R.run(c, ids, masks, s["answers"], max_new, rng)
            acc, f = score(args.task, out, s["answers"])
            row = dict(task=args.task, ctx=args.ctx, fmt=args.fmt, idx=idx, cond=c["name"], acc=acc, f1=f,
                       ans_lp=alp, first_lp=flp, ans_len=alen, kept=kept, T=ids.shape[1],
                       gold_frac=(masks["gold"] & masks["context"]).sum().item() / masks["context"].sum().item(),
                       out=out, ans=s["answers"][0], meta=s["meta"], sec=round(time.time() - t0, 2))
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"[{idx}] {c['name']:<32} acc={acc:.0f} lp={alp:7.2f} kept={kept:.3f} "
                  f"{row['sec']:5.1f}s out={out[:40]!r} ans={s['answers'][0][:30]!r}", flush=True)


if __name__ == "__main__":
    main()
