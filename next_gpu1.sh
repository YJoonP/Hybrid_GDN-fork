#!/bin/bash
# After the GPU-1 mode-AL run (launched under the old name "specla") exits: rename its results to results_modeal,
# then run pure-recompute SpecLA on GPU 1 (Phase 0, 3, 4; DFlash cells only, AR reused later).
cd "$(dirname "$0")"
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
mv results_specla results_modeal
.venv/bin/python - <<'PY'
import glob, json
for p in glob.glob("results_modeal/cells/*.json"):
    c = json.load(open(p))
    if c.get("rollback") == "specla":
        c["rollback"] = "modeal"
        c["renamed_from"] = "specla (same mode-AL code; renamed when pure-recompute SpecLA was added)"
        json.dump(c, open(p, "w"), indent=1)
print("renamed results_specla -> results_modeal")
PY
mkdir -p results_specla
GPU=1 src/run_gpu.sh results_specla/run.log .venv/bin/python run_all.py --rollback specla --phases 0,3,4 --methods dflash --accept-tie-mismatch
