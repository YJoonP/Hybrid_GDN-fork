#!/bin/bash
# Phase 4 for both rollbacks: AR (measured once, in results_fullstate) -> DFlash full state -> DFlash SpecLA.
cd "$(dirname "$0")"
src/run_gpu.sh results_fullstate/phase4.log .venv/bin/python run_all.py --rollback fullstate --phases 3,4 --accept-tie-mismatch
src/run_gpu.sh results_specla/phase4.log .venv/bin/python run_all.py --rollback specla --phases 0,3,4 --accept-tie-mismatch --reuse-ar-from results_fullstate
