"""Correctness check for greedy DFlash on a hybrid (GDN) target.

For each prompt, generate with AR, DFlash+rollback, and DFlash without rollback (official Transformers behaviour).
Then prefill prompt+output in one pass and check, at every generated position, whether the generated token is the
target's argmax. Exact speculative decoding can only disagree where the top-2 logits are (numerically) tied.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_qwen35 as DQ  # noqa: E402
from dflash.benchmark import _select_dataset, apply_chat_template, load_and_process_dataset  # noqa: E402
from dflash.model import DFlashDraftModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.inference_mode()
def teacher_forced(target, ids, n_prompt):
    logits = target(ids, use_cache=False).logits[0, n_prompt - 1:-1].float()
    top2 = logits.topk(2, dim=-1)
    gen = ids[0, n_prompt:]
    agree = top2.indices[:, 0] == gen
    gap_top = top2.values[:, 0] - top2.values[:, 1]
    gap_gen = top2.values[:, 0] - logits.gather(-1, gen[:, None])[:, 0]   # how far the emitted token is from top-1
    return agree, gap_top, gap_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.5-4B"))
    ap.add_argument("--draft", default=os.path.join(ROOT, "models/Qwen3.5-4B-DFlash"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa",
                                                  dtype=torch.bfloat16).to(dev).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    DQ.install(target)
    stop = [target.generation_config.eos_token_id] if isinstance(target.generation_config.eos_token_id, int) \
        else list(target.generation_config.eos_token_id)
    data = _select_dataset(load_and_process_dataset("gsm8k"), args.n)
    rows = []
    for i, inst in enumerate(data):
        ids = tok.encode(apply_chat_template(tok, [{"role": "user", "content": inst["turns"][0]}], None),
                         return_tensors="pt", add_special_tokens=False).to(dev)
        n = ids.shape[1]
        for name, bs, rb in [("ar", 1, True), ("dflash_rollback", 16, True), ("dflash_official", 16, False)]:
            out = DQ.dflash_generate_hybrid(draft, target, ids, args.max_new_tokens, stop, block_size=bs, rollback=rb)
            agree, gap_top, gap_gen = teacher_forced(target, out, n)
            bad = (~agree).nonzero().flatten()
            row = dict(i=i, mode=name, len=int(out.shape[1] - n), disagree=int(bad.numel()),
                       max_gap_at_disagree=float(gap_gen[bad].max()) if bad.numel() else 0.0,
                       first_disagree=int(bad[0]) if bad.numel() else -1,
                       median_gap_top=float(gap_top.median()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
    json.dump(rows, open(args.out, "w"))


if __name__ == "__main__":
    main()
