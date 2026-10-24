"""Record the software / hardware environment (env.json)."""
import importlib.metadata as md
import platform
import subprocess

import torch


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def collect(dflash_dir):
    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    return dict(
        gpu=torch.cuda.get_device_name(0),
        gpu_capability=list(torch.cuda.get_device_capability(0)),
        gpu_total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        driver=_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"]),
        cuda_runtime_torch=torch.version.cuda,
        nvcc=_cmd(["nvcc", "--version"]).splitlines()[-1] if _cmd(["nvcc", "--version"]) else None,
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=ver("transformers"),
        flash_linear_attention=ver("flash-linear-attention"),
        fla_core=ver("fla-core"),
        triton=ver("triton"),
        dflash_commit=_cmd(["git", "-C", dflash_dir, "rev-parse", "HEAD"]),
        attention_impl="torch SDPA (explicit boolean masks); prefill via HF sdpa",
    )
