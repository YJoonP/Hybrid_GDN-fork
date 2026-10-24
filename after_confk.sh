#!/bin/bash
# Same-GPU baselines once the confk queue runners exit:
#   GPU 1: AR on GPU 1 (results_ar_gpu1) so GPU-1 cells get a same-GPU speedup reference.
#   GPU 0: mode-AL and SpecLA replicated on GPU 0 (results_{modeal,specla}_gpu0) for a same-GPU comparison with
#          full state. Same code that passed Phase 3, so Phase 3 is not repeated there.
cd "$(dirname "$0")"
GPUID=$1; WAIT=$2
while kill -0 "$WAIT" 2>/dev/null; do sleep 15; done
if [ "$GPUID" = "1" ]; then
  mkdir -p results_ar_gpu1
  GPU=1 src/run_gpu.sh results_ar_gpu1/run.log .venv/bin/python run_all.py --rollback fullstate --phases 0,4 \
    --methods ar --out results_ar_gpu1
else
  for rb in modeal specla; do
    mkdir -p results_${rb}_gpu0
    GPU=0 src/run_gpu.sh results_${rb}_gpu0/run.log .venv/bin/python run_all.py --rollback $rb --phases 0,4 \
      --methods dflash --out results_${rb}_gpu0
  done
fi
