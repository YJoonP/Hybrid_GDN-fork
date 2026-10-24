"""Summarize dflash_bench.py output (Spec-Bench and DFlash-style metrics).

usage: dflash_summary.py results/dflash/<file>.jsonl [...]
"""
import json
import statistics as st
import sys
from collections import Counter

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        continue
    bs = rows[0]["block"]
    steps = [a for r in rows for a in r["accept"]]
    df_tok, df_s = sum(r["df_tokens"] for r in rows), sum(r["df_decode_s"] for r in rows)
    print(f"== {path}  n={len(rows)}  block={bs}  mode={rows[0]['mode']}")
    print(f"  #MAT (Spec-Bench: total tokens / verify steps) : {len(steps) and sum(steps) / len(steps):.3f}")
    print(f"  tau  (DFlash: mean of per-request mean)        : {st.mean(r['tau'] for r in rows):.3f}")
    print(f"  DFlash decode throughput                       : {df_tok / df_s:.1f} tok/s")
    if "ar_tokens" in rows[0]:
        ar_tok, ar_s = sum(r["ar_tokens"] for r in rows), sum(r["ar_decode_s"] for r in rows)
        print(f"  AR decode throughput                           : {ar_tok / ar_s:.1f} tok/s")
        print(f"  Speedup (Spec-Bench: overall tok/s ratio)      : {(df_tok / df_s) / (ar_tok / ar_s):.2f}x")
        ar_tpot = st.mean(r["ar_decode_s"] / r["ar_tokens"] for r in rows)
        df_tpot = st.mean(r["df_decode_s"] / r["df_tokens"] for r in rows)
        print(f"  Speedup (DFlash: mean-TPOT ratio)              : {ar_tpot / df_tpot:.2f}x")
        same = sum(r["same_output"] for r in rows)
        print(f"  Greedy output identical to AR                  : {same}/{len(rows)}")
    print(f"  draft / verify share of decode time            : "
          f"{sum(r['df_draft_s'] for r in rows) / df_s:.1%} / {sum(r['df_verify_s'] for r in rows) / df_s:.1%}")
    print(f"  per-step time (draft, verify)                  : "
          f"{1e3 * sum(r['df_draft_s'] for r in rows) / len(steps):.1f} ms, "
          f"{1e3 * sum(r['df_verify_s'] for r in rows) / len(steps):.1f} ms")
    print(f"  TTFT (mean)                                    : {st.mean(r['df_ttft'] for r in rows) * 1e3:.0f} ms")
    print(f"  output length (mean), hit max                  : {st.mean(r['df_tokens'] for r in rows):.0f}, "
          f"{sum(r['df_tokens'] >= 1024 for r in rows)}/{len(rows)}")
    h = Counter(steps)
    print("  acceptance histogram (tokens/step): " + " ".join(f"{k}:{h[k] / len(steps):.1%}" for k in range(1, bs + 1)))
    print(f"  peak memory                                    : {max(r['mem_gb'] for r in rows):.1f} GB")
