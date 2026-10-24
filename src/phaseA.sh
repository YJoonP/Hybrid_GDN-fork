#!/bin/bash
# Phase A: baseline headroom across tasks / question placement
cd /home/sungfire/Hybrid_GDN
export CUDA_VISIBLE_DEVICES=0
CTX=${1:-16000}
for task in niah kv hotpot; do
  for fmt in after before both; do
    .venv/bin/python src/run_eval.py --task $task --ctx $CTX --fmt $fmt --n 50 --conds base \
      --out results/A_${task}_${CTX}_${fmt}.jsonl 2>&1 | grep "^\[" > logs/A_${task}_${CTX}_${fmt}.log
  done
done
