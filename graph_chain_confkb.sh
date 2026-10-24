#!/bin/bash
# After the conf-K (method A) graph queue on GPU 0 drains: run the conf-K method B integration check, then queue and
# run the method-B variants (mlp head) on the same GPU.
cd "$(dirname "$0")"
while kill -0 2736924 2>/dev/null; do sleep 20; done
echo "[chain] method-A queue finished"
GPU=0 src/run_gpu.sh logs/graph/confkb_check.log .venv/bin/python check_confkb.py \
  || { echo "[chain] confkb integration check FAILED (see logs/graph/confkb_check.log)"; exit 1; }
echo "[chain] confkb integration check passed; queueing method B"
flock graph_queue.lock bash -c 'cat >> graph_queue.txt <<EOF
confkb --select topk --k 1
confkb --select topk --k 2
confkb --select topk --k 4
confkb --select thr --thr 0.1
confkb --select thr --thr 0.3
confkb --select thr --thr 0.5
confkb --select thr --thr 0.7
confkb --select thr --thr 0.9
EOF'
./graph_runner.sh 0 >> graph_queue_gpu0.log 2>&1
echo "[chain] method-B queue finished rc=$?"
