#!/bin/bash
# Oracle AL prediction (one snapshot slot at the true committed row) on GPU 0: CUDA-graph engine, then eager engine.
# AL traces come from the SpecLA runs of the same engine and batch size (outputs are identical across strategies).
cd "$(dirname "$0")"
GPU=0 src/run_gpu.sh logs/graph/oracle.log .venv/bin/python run_all.py --engine graph --rollback oracle \
  --oracle-from results_graph_specla --phases 0,3,4 --accept-tie-mismatch --reuse-ar-from results_graph_fullstate \
  || { echo "[oracle] graph FAILED"; exit 1; }
echo "[oracle] graph done"
GPU=0 src/run_gpu.sh logs/oracle_eager.log .venv/bin/python run_all.py --engine eager --rollback oracle \
  --oracle-from results_specla_gpu0 --phases 0,3,4 --accept-tie-mismatch --reuse-ar-from results_fullstate \
  || { echo "[oracle] eager FAILED"; exit 1; }
echo "[oracle] eager done"
