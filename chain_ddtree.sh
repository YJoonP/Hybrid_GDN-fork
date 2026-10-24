#!/bin/bash
# After the conf-K method-B chain on GPU 0 finishes: run the dflash-qwen35-ddtree package (its own eager engine) in
# this repo's venv on GPU 0 -- smoke test, DDTree + tree_fullstate (budget 15), then chain DFlash + SpecLA with the AR
# baseline on the same engine. Same 256 GSM8K questions / prompts / 128 new tokens / batch 1,8,64,128 x 3 runs.
ROOT=/home/sungfire/Hybrid_GDN
PKG=$ROOT/ext/ddtree/dflash-qwen35-ddtree
PY=$ROOT/.venv/bin/python
M="--target $ROOT/models/Qwen3.5-4B --draft $ROOT/models/Qwen3.5-4B-DFlash"
mkdir -p $ROOT/logs/ddtree
while kill -0 2793353 2>/dev/null; do sleep 20; done
echo "[ddtree] method-B chain finished"
cd $PKG
GPU=0 $ROOT/src/run_gpu.sh $ROOT/logs/ddtree/smoke.log $PY src/smoke_test.py $M \
  || { echo "[ddtree] smoke test FAILED (logs/ddtree/smoke.log)"; exit 1; }
echo "[ddtree] smoke test passed"
GPU=0 $ROOT/src/run_gpu.sh $ROOT/logs/ddtree/ddtree_tree_fullstate.log $PY src/run_all.py --method ddtree \
  --rollback tree_fullstate --accept-phase3-ties $M --out $ROOT/results_ext_ddtree_tree_fullstate \
  || { echo "[ddtree] DDTree tree_fullstate FAILED"; exit 1; }
echo "[ddtree] DDTree tree_fullstate done"
GPU=0 $ROOT/src/run_gpu.sh $ROOT/logs/ddtree/chain_specla.log $PY src/run_all.py --method dflash --rollback specla --ar \
  --accept-phase3-ties $M --out $ROOT/results_ext_chain_specla \
  || { echo "[ddtree] chain SpecLA FAILED"; exit 1; }
echo "[ddtree] chain SpecLA + AR done"
