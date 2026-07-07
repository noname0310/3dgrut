#!/usr/bin/env bash
set -euo pipefail

VENV=/workspace/.venv

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: expected Python environment at ${VENV}. Rebuild the Dev Container image." >&2
    exit 1
fi

git config --global --add safe.directory "$(pwd)"
git submodule update --init --recursive

source "${VENV}/bin/activate"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_PYTHON="${VENV}/bin/python"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-120}"

# Point the installed editable package at the mounted workspace, not the image's
# build-time copy under /workspace.
python -m pip install --no-deps -e .

python - <<'PY'
import importlib.util
import sys

import torch

print(f"Python: {sys.executable}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if importlib.util.find_spec("tinycudann") is None:
        raise SystemExit("tiny-cuda-nn is not installed")
    import tinycudann  # noqa: F401
    print("tiny-cuda-nn: ok")
else:
    print("tiny-cuda-nn: runtime import skipped because no CUDA device is visible")
PY
