#!/bin/bash
# Pull confk variants from confk_queue.txt (shared by both GPUs) once the given PID has exited.
# usage: queue_runner.sh <gpu index> <pid to wait for>
cd "$(dirname "$0")"
GPUID=$1; WAIT=$2
while [ -n "$WAIT" ] && kill -0 "$WAIT" 2>/dev/null; do sleep 15; done
while true; do
  LINE=$(flock confk_queue.lock bash -c 'L=$(head -n 1 confk_queue.txt); [ -n "$L" ] && sed -i 1d confk_queue.txt; echo "$L"')
  [ -z "$LINE" ] && { echo "[queue] GPU $GPUID: queue empty"; break; }
  set -- $LINE
  if [ "$1" = "top" ]; then ARGS="--select topk --k $2"; DIR=results_confk_top$2
  else ARGS="--select thr --thr $2"; DIR=results_confk_thr$2; fi
  mkdir -p $DIR
  echo "[queue] GPU $GPUID: $DIR"
  GPU=$GPUID src/run_gpu.sh $DIR/run.log .venv/bin/python run_all.py --rollback confk $ARGS --phases 0,3,4 \
    --methods dflash --accept-tie-mismatch --reuse-ar-from results_fullstate || { echo "[queue] GPU $GPUID: $DIR failed, stopping"; exit 1; }
done
