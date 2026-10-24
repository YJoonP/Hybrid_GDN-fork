#!/bin/bash
# Spec-Bench Math (80 GSM8K questions), static batch 64, greedy (temperature 0), max_new_tokens 1024.
# Runs AR and DFlash (block 16) on the same engine; teacher-forced exactness check on 8 outputs per batch/mode.
cd "$(dirname "$0")/.."
mkdir -p results
CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python src/dflash_batch_bench.py \
  --batch-size 64 --max-new-tokens 1024 --check 8 \
  --out results/batch64_specbench_math_4b "$@"
