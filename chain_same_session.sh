#!/bin/bash
# Same-session re-measurement on GPU 0 (conditions drift: GPU 1 co-tenant, power/thermal): after the copy-only oracle
# run finishes, measure SpecLA, the original oracle, and SpecLA again back to back (batch 1, 8, 64; 3 runs each;
# phase 3 skipped -- same code as the validated runs). SpecLA at both ends brackets drift.
cd "$(dirname "$0")"
while kill -0 3336247 2>/dev/null; do sleep 15; done
echo "[session] oracle_copy finished $(date +%H:%M:%S)"
run() {  # name rollback extra...
  local name=$1; shift
  GPU=0 src/run_gpu.sh logs/graph/$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --phases 0,4 --batch-sizes 1,8,64 --reuse-ar-from results_graph_fullstate --out results_session2/$name \
    || { echo "[session] $name FAILED"; exit 1; }
  echo "[session] $name done $(date +%H:%M:%S)"
}
mkdir -p results_session2
run specla_a specla
run oracle oracle --oracle-from results_graph_specla
run specla_b specla
echo "[session] all done"
