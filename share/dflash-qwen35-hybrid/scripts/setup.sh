#!/bin/bash
# Environment + models + data. Run from the package root: bash scripts/setup.sh
set -e
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -r requirements.txt

# DFlash reference code (draft model class), pinned to the commit used for these results
git clone -q https://github.com/z-lab/dflash.git third_party/dflash
git -C third_party/dflash checkout -q 07ebd93db9f472af339b644bb70221ad8428328a

# Target + draft (bf16). 4B: ~8.8 GB + 1.2 GB
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for rid in ["Qwen/Qwen3.5-4B", "z-lab/Qwen3.5-4B-DFlash"]:
    print(snapshot_download(rid, local_dir="models/" + rid.split("/")[1]))
PY

# Spec-Bench questions (Xia et al., 2024)
mkdir -p data
curl -sL https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench_question.jsonl \
  -o data/spec_bench_question.jsonl
echo "setup done"
