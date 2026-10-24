#!/bin/bash
# Run a resumable GPU job on GPU 0; re-run it when CUDA init fails transiently (cuInit error 3 on this machine).
# usage: run_gpu.sh <logfile> <cmd...>
LOG=$1; shift
export CUDA_VISIBLE_DEVICES=${GPU:-0}
for try in $(seq 1 20); do
  "$@" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "[run_gpu] done (try $try)" >> "$LOG"; exit 0; fi
  if tail -50 "$LOG" | grep -q -E "CUDA driver initialization failed|cuInit|no CUDA GPUs"; then
    echo "[run_gpu] CUDA init failure, retry $try in 15s" >> "$LOG"; sleep 15; continue
  fi
  echo "[run_gpu] failed rc=$rc (not a CUDA init error), giving up" >> "$LOG"; exit $rc
done
