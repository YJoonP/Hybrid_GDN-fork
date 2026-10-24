#!/bin/bash
# Batch-1 exactness check: AR vs DFlash with GDN rollback vs DFlash without rollback (upstream Transformers path).
cd "$(dirname "$0")/.."
mkdir -p results
CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python src/dflash_check.py --n 5 --max-new-tokens 512 \
  --out results/check_batch1_gsm8k5.json "$@"
