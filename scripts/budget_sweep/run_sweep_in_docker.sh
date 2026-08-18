#!/usr/bin/env bash
# Host-side launcher (Git Bash on Windows): starts the budget sweep inside the
# 3dgrut:cuda12 image, mirroring the devcontainer setup (.devcontainer/).
# Usage:  bash scripts/budget_sweep/run_sweep_in_docker.sh
# Monitor: docker logs -f 3dgrut_budget_sweep
# Stop:    docker stop 3dgrut_budget_sweep
# The sweep script skips finished runs, so restarting the container resumes.

set -euo pipefail
cd "$(dirname "$0")/../.."

IMAGE=${IMAGE:-3dgrut:cuda12}
NAME=${NAME:-3dgrut_budget_sweep}

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Container $NAME already exists. Start it with: docker start $NAME"
  exit 1
fi

MSYS2_ARG_CONV_EXCL="*" docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --shm-size=16g --ipc=host \
  -v "D:/3dgrut:/workspaces/3dgrut" \
  -w /workspaces/3dgrut \
  -e VIRTUAL_ENV=/workspace/.venv \
  -e UV_PROJECT_ENVIRONMENT=/workspace/.venv \
  -e UV_PYTHON=/workspace/.venv/bin/python \
  -e UV_INDEX=pytorch=https://download.pytorch.org/whl/cu128 \
  -e PATH=/workspace/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
  -e TCNN_CUDA_ARCHITECTURES=89 \
  -e OPTIX_LIB_DIR=/workspaces/3dgrut/.local/optix-libs/610.43.02 \
  -e LD_LIBRARY_PATH=/workspaces/3dgrut/.local/optix-libs/610.43.02:/usr/local/cuda/lib64 \
  -e TORCH_EXTENSIONS_DIR=/workspaces/3dgrut/data/.cache/torch_extensions \
  "$IMAGE" bash -c "uv pip install --no-deps -e . && bash scripts/budget_sweep/run_budget_sweep.sh"

echo "Started container: $NAME"
echo "  docker logs -f $NAME"
