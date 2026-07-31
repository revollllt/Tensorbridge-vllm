#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_k_baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_k_baselines_%j.out
#SBATCH --error=sbatch/logs/tb_k_baselines_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"

# The kernel under test is whatever checkout this points at. Default is the
# clean worktree pinned to the wheel commit, so the numbers line up with what
# vLLM actually loads; point it elsewhere to measure in-progress kernel work.
TENSORBRIDGE_ROOT="${TENSORBRIDGE_ROOT:-/data/user/jzou521/codes/cuda/tensorbridge-pinned}"
SHAPE_FILE="${BENCH_SHAPE_FILE:-${TENSORBRIDGE_ROOT}/benchmarks/shapes/paper_main_shapes.csv}"
OUTPUT="${BENCH_OUTPUT:-${REPO_DIR}/benchmarks/results/kernel/nvfp4_kernel_baselines_${JOB_KEY}.json}"

mkdir -p "${REPO_DIR}/sbatch/logs" "$(dirname "${OUTPUT}")"
mkdir -p "${REPO_DIR}/benchmarks/tmp/kernel_baselines/${JOB_KEY}"

export TENSORBRIDGE_CACHE_DIR="${REPO_DIR}/benchmarks/tmp/kernel_baselines/${JOB_KEY}/tensorbridge"
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/benchmarks/tmp/kernel_baselines/${JOB_KEY}/torch"
mkdir -p "${TENSORBRIDGE_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

export TENSORBRIDGE_CLEAR_CACHE=0
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
source "${REPO_DIR}/sbatch/_common.sh"

# Production kernel contract, identical to the TensorBridge repo wrappers.
export TENSORBRIDGE_COMPILER=nvrtc
unset TENSORBRIDGE_EXTRA_NVRTC_FLAGS || true
export TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT="${TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT:-1}"
export BENCH_CURRENT_FLAGS_OVERRIDE="${BENCH_CURRENT_FLAGS_OVERRIDE:--DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1}"
# Where load_vllm_cutlass_ops() finds vllm/_C_stable_libtorch.abi3.so.
export VLLM_CUTLASS_ROOT="${VLLM_CUTLASS_ROOT:-${REPO_DIR}}"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CLOCK_PINNED=0
MAX_GR_CLK=$(nvidia-smi -i "${GPU_ID}" --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
if [[ -n "${MAX_GR_CLK}" ]] && nvidia-smi -i "${GPU_ID}" -lgc "${MAX_GR_CLK},${MAX_GR_CLK}" >/dev/null 2>&1; then
    CLOCK_PINNED=1
    echo "[baselines] GPU clocks pinned to ${MAX_GR_CLK} MHz"
else
    echo "[baselines] GPU clock pinning skipped (no permission or not available)"
fi
cleanup() {
    if [[ "${CLOCK_PINNED}" == "1" ]]; then
        nvidia-smi -i "${GPU_ID}" -rgc >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[baselines] tensorbridge_root=${TENSORBRIDGE_ROOT}"
echo "[baselines] shapes=${SHAPE_FILE}"
echo "[baselines] output=${OUTPUT}"

EXTRA_ARGS=()
if [[ -n "${BENCH_SHAPE_LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--shape-limit "${BENCH_SHAPE_LIMIT}")
fi
if [[ "${BENCH_CONTINUE_ON_ERROR:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--continue-on-error)
fi
if [[ "${BENCH_FLUSH_L2:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--flush-l2)
fi
if [[ "${BENCH_INCLUDE_ACTIVATION_QUANT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--include-activation-quant)
fi

"${PYTHON}" -u "${REPO_DIR}/benchmarks/bench_nvfp4_kernel_baselines.py" \
    --tensorbridge-root "${TENSORBRIDGE_ROOT}" \
    --shape-file "${SHAPE_FILE}" \
    --rounds "${BENCH_ROUNDS:-6}" \
    --prime "${BENCH_PRIME:-8}" \
    --warmup "${BENCH_WARMUP:-4}" \
    --iters "${BENCH_ITERS:-40}" \
    --dtype "${BENCH_DTYPE:-bfloat16}" \
    --output "${OUTPUT}" \
    "${EXTRA_ARGS[@]}"
