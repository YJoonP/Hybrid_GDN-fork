#!/bin/bash
# Phase B: oracle go/no-go. If even perfect relevance cannot improve accuracy, a training-free estimator cannot.
cd /home/sungfire/Hybrid_GDN
export CUDA_VISIBLE_DEVICES=0
CTX=$1; N=$2; shift 2
CONDS="base;oracle,s=0,d=0;oracle,s=0.5,d=0.5;oracle,s=0,d=1;oracle,s=1,d=0;anti,s=0,d=0;random,s=0,d=0;reset"
for task in "$@"; do
  .venv/bin/python src/run_eval.py --task $task --ctx $CTX --fmt both --n $N --conds "$CONDS" \
    --out results/B_${task}_${CTX}_both.jsonl > logs/B_${task}_${CTX}.log 2>&1
done
