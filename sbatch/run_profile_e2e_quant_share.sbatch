#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_prof_quant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/tb_prof_quant_%j.out
#SBATCH --error=sbatch/logs/tb_prof_quant_%j.err

# How much GPU time does activation quantisation actually cost once the decode
# step is a replayed CUDA graph?
#
# The kernel harness charges TensorBridge's Triton quantiser ~26 us per call,
# but that number is flat in both M and K, which means it is launch overhead
# rather than work. Graph replay does not re-pay host launch cost, so the
# kernel harness may be measuring something production never sees. This asks
# the GPU directly.
#
# --cuda-graph-trace=node is the load-bearing flag: the default graph-level
# tracing reports one entry per graph launch, which hides everything inside.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
NSYS="${NSYS_BIN:-/usr/local/cuda-12.8/bin/nsys}"
ARM="${PROF_ARM:-tensorbridge}"
BATCH="${PROF_BATCH:-16}"
OUTPUT_LEN="${PROF_OUTPUT_LEN:-256}"
ITERS="${PROF_ITERS:-1}"
WARMUP="${PROF_WARMUP:-1}"
MODEL="${E2E_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"

REPORT_DIR="${PROF_REPORT_DIR:-${REPO_DIR}/benchmarks/results/nsys}"
PREFIX="${REPORT_DIR}/e2e_quant_${ARM}_bs${BATCH}_${JOB_KEY}"
CACHE="${REPO_DIR}/benchmarks/tmp/prof_quant/${JOB_KEY}/${ARM}"
# Node-local and short on purpose: vLLM puts its ZMQ IPC socket under TMPDIR,
# and a Unix domain socket path is capped at 107 characters. The repo-relative
# path plus a UUID overruns that and the engine dies before profiling starts.
NSYS_TMPDIR="/tmp/tbq${JOB_KEY}"
mkdir -p "${REPO_DIR}/sbatch/logs" "${REPORT_DIR}" "${CACHE}" "${NSYS_TMPDIR}"
export TMPDIR="${NSYS_TMPDIR}"
trap 'rm -rf "${NSYS_TMPDIR}"' EXIT

if [[ ! -x "${NSYS}" ]]; then
    echo "[prof] missing nsys: ${NSYS}" >&2
    exit 2
fi

export VLLM_PLUGINS=tensorbridge
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
export TENSORBRIDGE_CLEAR_CACHE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source "${REPO_DIR}/sbatch/_common.sh"

export TENSORBRIDGE_VLLM_BACKEND="${ARM}"
export TENSORBRIDGE_CACHE_DIR="${CACHE}"
export VLLM_CACHE_ROOT="${CACHE}/vllm"
export TENSORBRIDGE_GPU_CLOCK_PIN_STATUS=unprofiled

echo "[prof] arm=${ARM} batch=${BATCH} output_len=${OUTPUT_LEN} report=${PREFIX}.nsys-rep"

set +e
"${NSYS}" profile \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace=node \
    --force-overwrite=true \
    --output="${PREFIX}" \
    "${PYTHON}" -u "${REPO_DIR}/benchmarks/bench_e2e_batch_sweep.py" \
        --model "${MODEL}" \
        --batch-sizes "${BATCH}" \
        --input-len "${PROF_INPUT_LEN:-128}" \
        --output-len "${OUTPUT_LEN}" \
        --max-model-len "${PROF_MAX_MODEL_LEN:-512}" \
        --num-iters-warmup "${WARMUP}" \
        --num-iters "${ITERS}" \
        --arm "${ARM}" \
        --phase profile \
        --index 0 \
        --output "${PREFIX}_latency.json"
APP_STATUS=$?
set -e

REPORT="${PREFIX}.nsys-rep"
if [[ ! -f "${REPORT}" ]]; then
    echo "[prof] report not created: ${REPORT}" >&2
    exit 3
fi

# Per-kernel GPU time is the whole point; the API summary would only show the
# host side, which is exactly what graph replay removes.
"${NSYS}" stats --report cuda_gpu_kern_sum --format csv \
    --output "${PREFIX}_kern" "${REPORT}"

echo "[prof] application_status=${APP_STATUS}"
ls -la "${PREFIX}"*
exit "${APP_STATUS}"
