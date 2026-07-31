#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_e2e_lat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/tb_e2e_lat_%j.out
#SBATCH --error=sbatch/logs/tb_e2e_lat_%j.err

# End-to-end single-stream latency, TensorBridge vs the kernel vLLM would pick
# on its own. On Hopper `official` resolves to Marlin NVFP4, which is the same
# denominator the kernel-level benchmark uses, so the two layers compose.
#
# Backends must run in separate processes (the quant method is chosen at load
# time), so cross-process noise is what limits any claim. The order is a
# palindrome and every arm appears at mirrored positions, which makes the
# null control fall out of the same job rather than needing a separate one.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
RESULT_DIR="${E2E_RESULT_DIR:-${REPO_DIR}/benchmarks/results/e2e/latency_${JOB_KEY}}"
CACHE_ROOT="${REPO_DIR}/benchmarks/tmp/e2e_latency/${JOB_KEY}"
mkdir -p "${REPO_DIR}/sbatch/logs" "${RESULT_DIR}" "${CACHE_ROOT}"

MODEL="${E2E_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
INPUT_LEN="${E2E_INPUT_LEN:-128}"
OUTPUT_LEN="${E2E_OUTPUT_LEN:-256}"
BATCH_SIZE="${E2E_BATCH_SIZE:-8}"
MAX_MODEL_LEN="${E2E_MAX_MODEL_LEN:-512}"
ITERS_WARMUP="${E2E_ITERS_WARMUP:-3}"
ITERS="${E2E_ITERS:-12}"
GPU_MEM_UTIL="${E2E_GPU_MEMORY_UTILIZATION:-0.85}"
TP_SIZE="${E2E_TENSOR_PARALLEL_SIZE:-1}"
read -r -a ORDER <<< "${E2E_ORDER:-official tensorbridge tensorbridge official official tensorbridge tensorbridge official official tensorbridge tensorbridge official}"

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

# FULL_DECODE_ONLY with no inductor compilation is the TensorBridge path that
# was validated for CUDA Graph equivalence. The capture size must equal the
# decode batch or replay silently falls back.
COMPILATION_CONFIG="{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_num_of_warmups\":1,\"cudagraph_capture_sizes\":[${BATCH_SIZE}]}"

run_arm() {
    local phase="$1" index="$2" arm="$3"
    local out="${RESULT_DIR}/${phase}_${index}_${arm}.json"
    local raw="${RESULT_DIR}/.raw_${phase}_${index}_${arm}.json"
    local cache="${CACHE_ROOT}/${arm}"
    mkdir -p "${cache}"

    echo "[e2e] phase=${phase} index=${index} arm=${arm}"
    local started ended
    started=$(date +%s.%N)
    env \
        TENSORBRIDGE_VLLM_BACKEND="${arm}" \
        TENSORBRIDGE_CACHE_DIR="${cache}" \
        VLLM_CACHE_ROOT="${cache}/vllm" \
        "${PYTHON}" -u -m vllm.entrypoints.cli.main bench latency \
            --model "${MODEL}" \
            --quantization modelopt_mixed \
            --dtype bfloat16 \
            --language-model-only \
            --max-model-len "${MAX_MODEL_LEN}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --max-num-seqs "${BATCH_SIZE}" \
            --input-len "${INPUT_LEN}" \
            --output-len "${OUTPUT_LEN}" \
            --batch-size "${BATCH_SIZE}" \
            --num-iters-warmup "${ITERS_WARMUP}" \
            --num-iters "${ITERS}" \
            --disable-detokenize \
            --compilation-config "${COMPILATION_CONFIG}" \
            --output-json "${raw}"
    ended=$(date +%s.%N)

    # `vllm bench latency` records only latencies, so provenance is attached
    # here. The shape matches what scripts/analyze_perf_baseline.py consumes.
    TB_PHASE="${phase}" TB_INDEX="${index}" TB_ARM="${arm}" \
    TB_ORDER="${ORDER[*]}" TB_RAW="${raw}" TB_OUT="${out}" \
    TB_STARTED="${started}" TB_ENDED="${ended}" \
    TB_PROBE_INPUT_LEN="${INPUT_LEN}" TB_PROBE_OUTPUT_LEN="${OUTPUT_LEN}" \
    TB_PROBE_BATCH="${BATCH_SIZE}" TB_PROBE_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    TB_PROBE_ITERS="${ITERS}" TB_PROBE_WARMUP="${ITERS_WARMUP}" \
    TB_TP="${TP_SIZE}" TB_GPU_MEM="${GPU_MEM_UTIL}" TB_MODEL="${MODEL}" \
    TB_COMPILATION="${COMPILATION_CONFIG}" \
        "${PYTHON}" -u "${REPO_DIR}/scripts/annotate_e2e_latency.py"
    rm -f "${raw}"
}

# Prime each unique backend once so no measured run pays NVRTC JIT or a cold
# checkpoint page cache. Primed results are written but excluded from analysis.
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

echo "[e2e] results=${RESULT_DIR}"
