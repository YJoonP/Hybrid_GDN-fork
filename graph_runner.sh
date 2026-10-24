#!/bin/bash
# CUDA-graph engine matrix: pull rollback specs from graph_queue.txt and run them on one GPU.
# usage: graph_runner.sh <gpu index>
cd "$(dirname "$0")"
GPUID=$1
mkdir -p logs/graph
while true; do
  LINE=$(flock graph_queue.lock bash -c 'L=$(head -n 1 graph_queue.txt); [ -n "$L" ] && sed -i 1d graph_queue.txt; echo "$L"')
  [ -z "$LINE" ] && { echo "[queue] GPU $GPUID: queue empty"; break; }
  NAME=$(echo "$LINE" | sed 's/--select topk --k /top/; s/--select thr --thr /thr/; s/ //g')
  echo "[queue] GPU $GPUID: $NAME"
  REUSE=""; [ "$NAME" != "fullstate" ] && REUSE="--reuse-ar-from results_graph_fullstate"
  GPU=$GPUID src/run_gpu.sh logs/graph/$NAME.log .venv/bin/python run_all.py --engine graph --rollback $LINE \
    --phases 0,3,4 --accept-tie-mismatch $REUSE || { echo "[queue] GPU $GPUID: $NAME failed, stopping"; exit 1; }
done
