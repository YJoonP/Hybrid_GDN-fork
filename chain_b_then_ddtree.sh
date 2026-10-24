#!/bin/bash
# GPU 0: conf-K method B on the CUDA-graph engine (8 variants), then the dflash-qwen35-ddtree package
# (DDTree + tree_fullstate, then chain DFlash + SpecLA with AR) in this repo's venv. The DDTree smoke test passed
# (logs/ddtree/smoke.log); the method-B integration check is in results_graph_confkb_check.json.
ROOT=/home/sungfire/Hybrid_GDN
cd $ROOT
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
./graph_runner.sh 0 >> graph_queue_gpu0.log 2>&1 || echo "[chain] method-B queue stopped on a failure"
echo "[chain] method-B queue finished"
PKG=$ROOT/ext/ddtree/dflash-qwen35-ddtree
PY=$ROOT/.venv/bin/python
M="--target $ROOT/models/Qwen3.5-4B --draft $ROOT/models/Qwen3.5-4B-DFlash"
cd $PKG
GPU=0 $ROOT/src/run_gpu.sh $ROOT/logs/ddtree/ddtree_tree_fullstate.log $PY src/run_all.py --method ddtree \
  --rollback tree_fullstate --accept-phase3-ties $M --out $ROOT/results_ext_ddtree_tree_fullstate \
  || { echo "[ddtree] DDTree tree_fullstate FAILED"; exit 1; }
echo "[ddtree] DDTree tree_fullstate done"
GPU=0 $ROOT/src/run_gpu.sh $ROOT/logs/ddtree/chain_specla.log $PY src/run_all.py --method dflash --rollback specla --ar \
  --accept-phase3-ties $M --out $ROOT/results_ext_chain_specla \
  || { echo "[ddtree] chain SpecLA FAILED"; exit 1; }
echo "[ddtree] chain SpecLA + AR done"
