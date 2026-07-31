#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_e2e_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_e2e_sweep_%j.out
#SBATCH --error=sbatch/logs/tb_e2e_sweep_%j.err

# Every position sweeps the whole batch range inside one engine, so the run
# costs one engine start per palindrome position instead of one per (position,
# batch) pair. Staying on a single node also means the 21 GB checkpoint is read
# cold once and served from page cache for every later start.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
RESULT_DIR="${E2E_RESULT_DIR:-${REPO_DIR}/benchmarks/results/e2e/sweep_${JOB_KEY}}"
CACHE_ROOT="${REPO_DIR}/benchmarks/tmp/e2e_sweep/${JOB_KEY}"
mkdir -p "${REPO_DIR}/sbatch/logs" "${RESULT_DIR}" "${CACHE_ROOT}"

MODEL="${E2E_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
BATCH_SIZES="${E2E_BATCH_SIZES:-1,4,16,32,64,128}"
INPUT_LEN="${E2E_INPUT_LEN:-128}"
OUTPUT_LEN="${E2E_OUTPUT_LEN:-256}"
MAX_MODEL_LEN="${E2E_MAX_MODEL_LEN:-512}"
ITERS_WARMUP="${E2E_ITERS_WARMUP:-3}"
ITERS="${E2E_ITERS:-12}"
GPU_MEM_UTIL="${E2E_GPU_MEMORY_UTILIZATION:-0.85}"
TP_SIZE="${E2E_TENSOR_PARALLEL_SIZE:-1}"
read -r -a ORDER <<< "${E2E_ORDER:-official tensorbridge tensorbridge official}"

export VLLM_PLUGINS=tensorbridge
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
export TENSORBRIDGE_CLEAR_CACHE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source "${REPO_DIR}/sbatch/_common.sh"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CLOCK_PINNED=0
MAX_GR_CLK=$(nvidia-smi -i "${GPU_ID}" --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
if [[ -n "${MAX_GR_CLK}" ]] && nvidia-smi -i "${GPU_ID}" -lgc "${MAX_GR_CLK},${MAX_GR_CLK}" >/dev/null 2>&1; then
    CLOCK_PINNED=1
    export TENSORBRIDGE_GPU_CLOCK_PIN_STATUS="pinned_${MAX_GR_CLK}_mhz"
else
    export TENSORBRIDGE_GPU_CLOCK_PIN_STATUS=unavailable
fi
cleanup() {
    if [[ "${CLOCK_PINNED}" == "1" ]]; then
        nvidia-smi -i "${GPU_ID}" -rgc >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

run_arm() {
    local phase="$1" index="$2" arm="$3"
    local cache="${CACHE_ROOT}/${arm}"
    mkdir -p "${cache}"
    echo "[sweep] phase=${phase} index=${index} arm=${arm}"
    env \
        TENSORBRIDGE_VLLM_BACKEND="${arm}" \
        TENSORBRIDGE_CACHE_DIR="${cache}" \
        VLLM_CACHE_ROOT="${cache}/vllm" \
        "${PYTHON}" -u "${REPO_DIR}/benchmarks/bench_e2e_batch_sweep.py" \
            --model "${MODEL}" \
            --batch-sizes "${BATCH_SIZES}" \
            --input-len "${INPUT_LEN}" \
            --output-len "${OUTPUT_LEN}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --num-iters-warmup "${ITERS_WARMUP}" \
            --num-iters "${ITERS}" \
            --arm "${arm}" \
            --phase "${phase}" \
            --index "${index}" \
            --order "${ORDER[*]}" \
            --output "${RESULT_DIR}/${phase}_${index}_${arm}.json"
}

# One prime per unique arm: it pulls the checkpoint into page cache and pays the
# NVRTC JIT, so no measured position carries a cold start.
if [[ "${E2E_PRIME:-1}" == "1" ]]; then
    declare -A primed=()
    prime_index=0
    for arm in "${ORDER[@]}"; do
        if [[ -z "${primed[$arm]+x}" ]]; then
            run_arm prime "${prime_index}" "${arm}"
            primed["${arm}"]=1
            prime_index=$((prime_index + 1))
        fi
    done
fi

for index in "${!ORDER[@]}"; do
    run_arm measured "${index}" "${ORDER[$index]}"
done

echo "[sweep] results=${RESULT_DIR}"
