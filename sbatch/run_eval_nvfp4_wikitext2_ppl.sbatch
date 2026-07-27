#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_nvfp4_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_nvfp4_ppl_%j.out
#SBATCH --error=sbatch/logs/tb_nvfp4_ppl_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
PYTHON="${TENSORBRIDGE_VLLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[ppl] missing vLLM runtime: ${PYTHON}" >&2
    exit 2
fi

mkdir -p "${REPO_DIR}/sbatch/logs" "${REPO_DIR}/benchmarks/results/ppl"
DEFAULT_TENSORBRIDGE_CACHE_DIR="${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/tensorbridge"
export TENSORBRIDGE_CACHE_DIR="${PPL_TENSORBRIDGE_CACHE_DIR:-${DEFAULT_TENSORBRIDGE_CACHE_DIR}}"
mkdir -p "${TENSORBRIDGE_CACHE_DIR}"
mkdir -p "${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/torch"
mkdir -p "${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/vllm"

export REPO_DIR
export TENSORBRIDGE_PYTHON="${PYTHON}"
export TENSORBRIDGE_CLEAR_CACHE=0
export TENSORBRIDGE_COMPILER=nvrtc
export TENSORBRIDGE_EXTRA_NVRTC_FLAGS=""
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/torch"
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
source "${REPO_DIR}/sbatch/_common.sh"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VLLM_CACHE_ROOT="${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=tensorbridge

BACKEND="${PPL_BACKEND:-tensorbridge}"
MODEL="${PPL_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
DEFAULT_DATASET="/data/user/jzou521/.cache/huggingface/datasets/wikitext/"
DEFAULT_DATASET+="wikitext-2-raw-v1/0.0.0/51aa13ed94b80c1a/wikitext-test.arrow"
DATASET="${PPL_DATASET_ARROW:-${DEFAULT_DATASET}}"
OUTPUT="${PPL_OUTPUT:-${REPO_DIR}/benchmarks/results/ppl/wikitext2_${BACKEND}_${JOB_KEY}.json}"
EXECUTION_MODE="${PPL_EXECUTION_MODE:-eager}"
CUDAGRAPH_MODE="${PPL_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
MAX_BLOCKS="${PPL_MAX_BLOCKS:-4}"
MAX_NUM_SEQS="${PPL_MAX_NUM_SEQS:-8}"
COMPILATION_MODE="${PPL_COMPILATION_MODE:-VLLM_COMPILE}"
FDO_PROFILE=0
if [[ "${CUDAGRAPH_MODE}" == "FULL_DECODE_ONLY" ]]; then
    FDO_PROFILE=1
    MAX_BLOCKS="${PPL_MAX_BLOCKS:-1}"
    MAX_NUM_SEQS="${PPL_MAX_NUM_SEQS:-1}"
    COMPILATION_MODE="${PPL_COMPILATION_MODE:-NONE}"
    REQUESTED_OUTPUT_TOKENS="${PPL_REQUESTED_OUTPUT_TOKENS:-8}"
    REPEAT_RUNS="${PPL_REPEAT_RUNS:-12}"
else
    REQUESTED_OUTPUT_TOKENS="${PPL_REQUESTED_OUTPUT_TOKENS:-1}"
    REPEAT_RUNS="${PPL_REPEAT_RUNS:-1}"
fi
if [[ "${EXECUTION_MODE}" == "cudagraph" ]]; then
    if [[ "${FDO_PROFILE}" == "1" ]]; then
        CUDAGRAPH_CAPTURE_SIZES="${PPL_CUDAGRAPH_CAPTURE_SIZES:-1}"
    else
        REQUESTED_OUTPUT_TOKENS="${PPL_REQUESTED_OUTPUT_TOKENS:-8}"
        REPEAT_RUNS="${PPL_REPEAT_RUNS:-12}"
        CUDAGRAPH_CAPTURE_SIZES="${PPL_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,32}"
    fi
    if [[ -z "${PPL_EAGER_REFERENCE_JSON:-}" ]]; then
        echo "[ppl] graph validation requires PPL_EAGER_REFERENCE_JSON" >&2
        exit 2
    fi
else
    CUDAGRAPH_CAPTURE_SIZES="${PPL_CUDAGRAPH_CAPTURE_SIZES:-}"
fi

EXTRA_ARGS=()
if [[ -n "${PPL_BLOCK_START:-}" ]]; then
    EXTRA_ARGS+=(--block-start "${PPL_BLOCK_START}")
fi
if [[ "${MAX_BLOCKS}" != "all" ]]; then
    EXTRA_ARGS+=(--max-blocks "${MAX_BLOCKS}")
fi
if [[ "${PPL_ALLOW_RUNTIME_VERSION_MISMATCH:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--allow-runtime-version-mismatch)
fi
if [[ "${PPL_FPMA_ULP_CORRECTION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--fpma-ulp-correction)
fi
if [[ "${PPL_DISABLE_ALLREDUCE_RMS_FUSION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-allreduce-rms-fusion)
fi
if [[ "${PPL_VALIDATION_DETAILS:-0}" == "1" || "${EXECUTION_MODE}" == "cudagraph" || "${FDO_PROFILE}" == "1" ]]; then
    EXTRA_ARGS+=(--validation-details)
fi
if [[ -n "${CUDAGRAPH_CAPTURE_SIZES}" ]]; then
    EXTRA_ARGS+=(--cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}")
fi
if [[ -n "${PPL_EAGER_REFERENCE_JSON:-}" ]]; then
    EXTRA_ARGS+=(--eager-reference-json "${PPL_EAGER_REFERENCE_JSON}")
fi
if [[ -n "${PPL_FPMA_ALPHA+x}" ]]; then
    if [[ -z "${PPL_FPMA_ALPHA}" ]]; then
        echo "[ppl] PPL_FPMA_ALPHA must not be empty when set" >&2
        exit 2
    fi
    EXTRA_ARGS+=(--fpma-alpha "${PPL_FPMA_ALPHA}")
fi

echo "[ppl] backend=${BACKEND} tp=${PPL_TENSOR_PARALLEL_SIZE:-1} " \
    "mode=${EXECUTION_MODE} model=${MODEL} dataset=${DATASET}"
echo "[ppl] output=${OUTPUT} max_blocks=${MAX_BLOCKS}"
echo "[ppl] requested_output_tokens=${REQUESTED_OUTPUT_TOKENS} repeat_runs=${REPEAT_RUNS}"
echo "[ppl] compilation_mode=${COMPILATION_MODE} cudagraph_mode=${CUDAGRAPH_MODE}"
echo "[ppl] tensorbridge_cache=${TENSORBRIDGE_CACHE_DIR}"
"${PYTHON}" -u "${REPO_DIR}/scripts/eval_nvfp4_wikitext2_ppl.py" \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --dataset-arrow "${DATASET}" \
    --max-model-len "${PPL_MAX_MODEL_LEN:-2048}" \
    --target-tokens-per-block "${PPL_TARGET_TOKENS_PER_BLOCK:-1024}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${PPL_GPU_MEMORY_UTILIZATION:-0.5}" \
    --tensor-parallel-size "${PPL_TENSOR_PARALLEL_SIZE:-1}" \
    --execution-mode "${EXECUTION_MODE}" \
    --compilation-mode "${COMPILATION_MODE}" \
    --cudagraph-mode "${CUDAGRAPH_MODE}" \
    --requested-output-tokens "${REQUESTED_OUTPUT_TOKENS}" \
    --repeat-runs "${REPEAT_RUNS}" \
    --normal-a8-chunk-rows "${PPL_NORMAL_A8_CHUNK_ROWS:-256}" \
    --fpma-prefold-selector "${PPL_FPMA_PREFOLD_SELECTOR:-none}" \
    --fpma-selector-chunk-rows "${PPL_FPMA_SELECTOR_CHUNK_ROWS:-256}" \
    --output-json "${OUTPUT}" \
    "${EXTRA_ARGS[@]}"
