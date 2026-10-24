"""Batch-1 DFlash baseline for Qwen3.5 targets on Transformers (mirrors dflash.benchmark._run_transformers).

Per prompt it runs autoregressive decoding (block_size=1) and DFlash (block_size=B) with the same loop, and records
throughput, acceptance lengths, and whether the greedy DFlash output equals the autoregressive one.
--mode official reproduces the upstream Transformers backend (no GDN recurrent-state rollback) for comparison.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_qwen35 as DQ  # noqa: E402
from dflash.benchmark import _select_dataset, apply_chat_template, load_and_process_dataset  # noqa: E402
from dflash.model import DFlash2DraftModel, DFlashDraftModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.5-4B"))
    ap.add_argument("--draft", default=os.path.join(ROOT, "models/Qwen3.5-4B-DFlash"))
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--reasoning", default=None, help="None = chat-template default (Qwen3.5: thinking on)")
    ap.add_argument("--mode", default="rollback", choices=["rollback", "official"])
    ap.add_argument("--skip-ar", action="store_true", help="skip the autoregressive reference run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=torch.bfloat16)
    target = target.to(dev).eval()
    cfg = AutoConfig.from_pretrained(args.draft)
    cls = DFlash2DraftModel if "DFlash2DraftModel" in (cfg.architectures or []) else DFlashDraftModel
    draft = cls.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    DQ.install(target)
    rollback = args.mode == "rollback"
    bs = args.block_size or draft.block_size
    eos = target.generation_config.eos_token_id or tok.eos_token_id
    stop = [eos] if isinstance(eos, int) else list(eos)
    print(f"target {type(target).__name__} draft {cls.__name__} block {bs} mode {args.mode} stop {stop} "
          f"mem {torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

    if args.dataset.startswith("specbench_"):
        # Spec-Bench (Xia et al., 2024) category, in question_id order; raw question used as the user turn
        cat = {"specbench_math": "math_reasoning"}.get(args.dataset, args.dataset[len("specbench_"):])
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/spec_bench_question.jsonl"))]
        data = [dict(turns=r["turns"], qid=r["question_id"]) for r in rows if r["category"] == cat]
        data = data[: args.max_samples]
    else:
        data = _select_dataset(load_and_process_dataset(args.dataset), args.max_samples)
    warm = tok.encode(apply_chat_template(tok, [{"role": "user", "content": data[0]["turns"][0]}], args.reasoning),
                      return_tensors="pt", add_special_tokens=False).to(dev)
    for b in (1, bs):
        DQ.dflash_generate_hybrid(draft, target, warm, 64, None, block_size=b, rollback=rollback)

    done = set()
    if os.path.exists(args.out):
        done = {(json.loads(l)["idx"], json.loads(l)["turn"]) for l in open(args.out)}
    fout = open(args.out, "a")
    for idx, inst in enumerate(data):
        if all((idx, t) in done for t in range(len(inst["turns"]))):
            continue
        msgs = []
        for turn, user in enumerate(inst["turns"]):
            msgs.append({"role": "user", "content": user})
            text = apply_chat_template(tok, msgs, args.reasoning)
            ids = tok.encode(text, return_tensors="pt", add_special_tokens=False).to(dev)
            res = {}
            for b in ([bs] if args.skip_ar else [1, bs]):
                res[b] = DQ.dflash_generate_hybrid(draft, target, ids, args.max_new_tokens, stop, block_size=b,
                                                   return_stats=True, rollback=rollback)
            sp = res[bs]
            gen = sp.output_ids[0, sp.num_input_tokens:]
            msgs.append({"role": "assistant", "content": tok.decode(gen, skip_special_tokens=True)})
            if (idx, turn) in done:
                continue
            row = dict(dataset=args.dataset, idx=idx, qid=inst.get("qid"), turn=turn, mode=args.mode, block=bs, prompt_len=ids.shape[1],
                       df_tokens=sp.num_output_tokens, df_decode_s=sp.decode_time, df_ttft=sp.time_to_first_token,
                       df_draft_s=sp.draft_time, df_verify_s=sp.verify_time, accept=sp.acceptance_lengths,
                       tau=sum(sp.acceptance_lengths) / len(sp.acceptance_lengths))
            if 1 in res:
                ar = res[1]
                ag = ar.output_ids[0, ar.num_input_tokens:]
                n = min(len(ag), len(gen))
                mism = (ag[:n] != gen[:n]).nonzero()
                row.update(ar_tokens=ar.num_output_tokens, ar_decode_s=ar.decode_time,
                           speedup=ar.time_per_output_token / sp.time_per_output_token,
                           same_output=bool(len(ag) == len(gen) and len(mism) == 0),
                           first_mismatch=int(mism[0]) if len(mism) else (-1 if len(ag) == len(gen) else n))
            row["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"[{args.dataset} {idx}.{turn}] tau={row['tau']:.2f} df={sp.num_output_tokens / sp.decode_time:6.1f}tok/s "
                  + (f"ar={ar.num_output_tokens / ar.decode_time:6.1f}tok/s x{row['speedup']:.2f} "
                     f"same={row['same_output']} mism@{row['first_mismatch']} " if 1 in res else "")
                  + f"len={sp.num_output_tokens} mem={row['mem_gb']:.1f}GB", flush=True)
            del res
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
