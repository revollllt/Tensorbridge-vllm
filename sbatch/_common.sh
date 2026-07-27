#!/usr/bin/env bash

set -euo pipefail

_TENSORBRIDGE_VLLM_SBATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${_TENSORBRIDGE_VLLM_SBATCH_DIR}/.." && pwd)}"
PYTHON="${TENSORBRIDGE_VLLM_PYTHON:-${PYTHON:-${REPO_DIR}/.venv/bin/python}}"
CUDA_MOD="${TENSORBRIDGE_CUDA_MOD:-cuda/12.8}"
GCC_MOD="${TENSORBRIDGE_GCC_MOD:-gcc/13.3}"

module purge
module load "${CUDA_MOD}"
module load "${GCC_MOD}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[sbatch] missing tensorbridge-vllm Python runtime: ${PYTHON}" >&2
    exit 2
fi

cd "${REPO_DIR}"
GCC_BIN="$(command -v gcc)"
GXX_BIN="$(command -v g++)"
export CC="${CC:-${GCC_BIN}}"
export CXX="${CXX:-${GXX_BIN}}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin ${GXX_BIN}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
# The v0.20.2 vendored namespace can exist without the DeepGEMM APIs needed
# by its warmup path. TensorBridge uses the selected CUTLASS FP8 kernels here.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

echo "[sbatch] host=$(hostname) cuda=${CUDA_MOD} gcc=${GCC_MOD}"
echo "[sbatch] python=${PYTHON} repo=${REPO_DIR}"
echo "[sbatch] vllm_use_deep_gemm=${VLLM_USE_DEEP_GEMM}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON}" - <<'PY'
import importlib.metadata
import tensorbridge
import torch
import vllm
from tensorbridge.api import v1

print(f"[sbatch] tensorbridge={tensorbridge.__file__}")
print(f"[sbatch] tensorbridge_version={importlib.metadata.version('tensorbridge-kernels')}")
print(f"[sbatch] tensorbridge_runtime_api={v1.RUNTIME_API_VERSION}")
print(f"[sbatch] vllm={vllm.__file__} version={vllm.__version__}")
print(f"[sbatch] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[sbatch] gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
