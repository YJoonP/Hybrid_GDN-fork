"""Spec-Bench (static batch, greedy) benchmark: autoregressive vs DFlash on the same batch engine.

Questions are split into batches of --batch-size; a short last batch is topped up with earlier questions
(marked filler: decoded and timed, excluded from per-question metrics) so every batch runs at full size.
Optional --check N: teacher-force the first N real outputs of each mode through the target and report how often the
emitted token is not the target's argmax, and the largest logit gap at such positions (exact decoding: ~0).
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dflash_batch import BatchDFlash  # noqa: E402
from dflash.benchmark import apply_chat_template  # noqa: E402
from dflash.model import DFlashDraftModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.inference_mode()
def teacher_forced(target, prompt, gen, dev):
    ids = torch.tensor([prompt + gen], device=dev)
    logits = target(ids, use_cache=False, logits_to_keep=len(gen) + 1).logits[0, :-1].float()
    g = torch.tensor(gen, device=dev)
    top = logits.max(-1)
    gap = top.values - logits.gather(-1, g[:, None])[:, 0]
    bad = top.indices != g
    return int(bad.sum()), float(gap[bad].max()) if bad.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.5-4B"))
    ap.add_argument("--draft", default=os.path.join(ROOT, "models/Qwen3.5-4B-DFlash"))
    ap.add_argument("--category", default="math_reasoning", help="Spec-Bench category")
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=None, help="default: draft config (16)")
    ap.add_argument("--reasoning", default=None, help="None = chat-template default (Qwen3.5: thinking on)")
    ap.add_argument("--modes", default="ar,dflash")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=torch.bfloat16)
    target = target.to(dev).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    eng = BatchDFlash(target, draft)
    bs_draft = args.block_size or draft.block_size
    eos = target.generation_config.eos_token_id or tok.eos_token_id
    stop = [eos] if isinstance(eos, int) else list(eos)

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/spec_bench_question.jsonl"))]
    qs = [r for r in rows if r["category"] == args.category][: args.max_samples]
    prompts = [tok.encode(apply_chat_template(tok, [{"role": "user", "content": r["turns"][0]}], args.reasoning),
                          add_special_tokens=False) for r in qs]
    B = args.batch_size
    batches = []
    for s in range(0, len(qs), B):
        idx = list(range(s, min(s + B, len(qs))))
        fill = [i % len(qs) for i in range(B - len(idx))]
        batches.append((idx, fill))

    # warm-up (Triton autotune / compile) on a small batch
    for m in args.modes.split(","):
        eng.generate([torch.tensor(p) for p in prompts[:4]], 32, stop, 1 if m == "ar" else bs_draft)

    per_q = open(args.out + "_questions.jsonl", "w")
    summary = []
    for bi, (idx, fill) in enumerate(batches):
        batch_prompts = [torch.tensor(prompts[i]) for i in idx + fill]
        for m in args.modes.split(","):
            blk = 1 if m == "ar" else bs_draft
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            r = eng.generate(batch_prompts, args.max_new_tokens, stop, blk)
            wall = time.perf_counter() - t0
            steps_all = [a for acc in r.accept for a in acc]
            s = dict(batch=bi, mode=m, block=blk, batch_size=len(batch_prompts), real=len(idx), fill=len(fill),
                     tokens=r.n_out, steps=r.steps, prefill_s=r.t_prefill, decode_s=r.t_decode, wall_s=wall,
                     draft_s=r.t_draft, verify_s=r.t_verify, decode_tok_s=r.n_out / r.t_decode,
                     e2e_tok_s=r.n_out / (r.t_prefill + r.t_decode),
                     mat=sum(steps_all) / len(steps_all), mean_active=sum(r.batch_sizes) / len(r.batch_sizes),
                     peak_gb=r.peak_gb)
            if args.check:
                chk = [teacher_forced(target, prompts[q], r.outputs[k], dev) for k, q in enumerate(idx[: args.check])]
                s["check_n"] = len(chk)
                s["check_disagree_tokens"] = sum(c[0] for c in chk)
                s["check_tokens"] = sum(len(r.outputs[k]) for k in range(len(chk)))
                s["check_max_gap"] = max(c[1] for c in chk)
            summary.append(s)
            print(json.dumps(s), flush=True)
            for k, q in enumerate(idx):
                acc = r.accept[k]
                per_q.write(json.dumps(dict(batch=bi, mode=m, qid=qs[q]["question_id"], prompt_len=len(prompts[q]),
                                            n_out=len(r.outputs[k]), tau=sum(acc) / len(acc), accept=acc,
                                            text=tok.decode(r.outputs[k]))) + "\n")
            per_q.flush()
    json.dump(dict(args=vars(args), batches=summary), open(args.out + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
