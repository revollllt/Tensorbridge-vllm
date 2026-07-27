#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_lm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_lm_eval_%A_%a.out
#SBATCH --error=sbatch/logs/tb_lm_eval_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${TENSORBRIDGE_VLLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[lm-eval] missing vLLM runtime: ${PYTHON}" >&2
    exit 2
fi

ARMS=(official normal_a8 fpma_default selector_alpha1 ulp_v1 alpha_0960)
SUITE="${LM_EVAL_SUITE:-smoke_mc}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ARMS[@]})); then
        echo "[lm-eval] array index must be in 0..5" >&2
        exit 2
    fi
    if [[ -n "${LM_EVAL_ARM:-}" || -n "${LM_EVAL_OUTPUT:-}" \
        || -n "${LM_EVAL_SAMPLES_DIR:-}" \
        || -n "${LM_EVAL_TENSORBRIDGE_CACHE_DIR:-}" ]]; then
        echo "[lm-eval] array jobs reject shared arm/output/sample/cache overrides" >&2
        exit 2
    fi
    ARM="${ARMS[SLURM_ARRAY_TASK_ID]}"
elif [[ -n "${LM_EVAL_ARM:-}" ]]; then
    ARM="${LM_EVAL_ARM}"
else
    echo "[lm-eval] set LM_EVAL_ARM or submit array index 0..5" >&2
    exit 2
fi

MODEL="${LM_EVAL_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
DEFAULT_CHECKPOINT_MANIFEST="${REPO_DIR}/benchmarks/manifests/qwen3.6-27b-nvfp4.sha256.json"
CHECKPOINT_MANIFEST="${LM_EVAL_CHECKPOINT_MANIFEST:-${DEFAULT_CHECKPOINT_MANIFEST}}"
DEFAULT_MAX_MODEL_LEN=4096
if [[ "${SUITE}" == "mmlu_pro" || "${SUITE}" == "stage2_mmlu_pro" ]]; then
    DEFAULT_MAX_MODEL_LEN=16384
fi
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_$$}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-single}"
JOB_KEY="${RUN_ID}_${TASK_ID}_${SUITE}_${ARM}"
RESULT_DIR="${REPO_DIR}/benchmarks/results/lm_eval/${RUN_ID}/${SUITE}"
TMP_DIR="${REPO_DIR}/benchmarks/tmp/lm_eval/${JOB_KEY}"
OUTPUT="${LM_EVAL_OUTPUT:-${RESULT_DIR}/${ARM}_${SLURM_JOB_ID:-manual}.json}"
SAMPLES_DIR="${LM_EVAL_SAMPLES_DIR:-${RESULT_DIR}/${ARM}_${SLURM_JOB_ID:-manual}_samples}"
mkdir -p "${REPO_DIR}/sbatch/logs" "${RESULT_DIR}" \
    "${TMP_DIR}/tensorbridge" "${TMP_DIR}/torch" "${TMP_DIR}/vllm"

export REPO_DIR
export TENSORBRIDGE_PYTHON="${PYTHON}"
export TENSORBRIDGE_CLEAR_CACHE=0
export TENSORBRIDGE_CACHE_DIR="${LM_EVAL_TENSORBRIDGE_CACHE_DIR:-${TMP_DIR}/tensorbridge}"
mkdir -p "${TENSORBRIDGE_CACHE_DIR}"
unset TENSORBRIDGE_CACHE_SEED_SOURCE
unset TENSORBRIDGE_CACHE_SEED_SHA256
unset TENSORBRIDGE_CACHE_SEED_VERIFIED

cache_seed_digest() {
    local root="$1"
    local count
    count="$(
        find "${root}" -type f \
            \( -name kernel.cu -o -name signature.txt -o -name cmdline.json \
            -o -name kernel.cubin -o -name tensorbridge_launcher.so \) \
            -printf x | wc -c
    )"
    if [[ "${count}" -ne 341 ]]; then
        echo "[lm-eval] cache seed has ${count} audited files, expected 341" >&2
        return 1
    fi
    (
        export LC_ALL=C
        cd "${root}"
        find . -type f \
            \( -name kernel.cu -o -name signature.txt -o -name cmdline.json \
            -o -name kernel.cubin -o -name tensorbridge_launcher.so \) \
            -print0 | sort -z | xargs -0 -r sha256sum
    ) | sha256sum | awk '{print $1}'
}

CACHE_SEED=""
EXPECTED_CACHE_SEED_SHA256=""
DEFAULT_ABI_CACHE_SEED="${LM_EVAL_FPMA_DEFAULT_CACHE_SEED:-${LM_EVAL_TENSORBRIDGE_CACHE_SEED:-}}"
case "${ARM}" in
    fpma_default)
        CACHE_SEED="${DEFAULT_ABI_CACHE_SEED}"
        EXPECTED_CACHE_SEED_SHA256="f4dfa648fb1c00544671c33780496aac5c8b4372ad36784f63f36361cb02d5b2"
        ;;
    selector_alpha1)
        CACHE_SEED="${LM_EVAL_SELECTOR_ALPHA1_CACHE_SEED:-${DEFAULT_ABI_CACHE_SEED}}"
        EXPECTED_CACHE_SEED_SHA256="f4dfa648fb1c00544671c33780496aac5c8b4372ad36784f63f36361cb02d5b2"
        ;;
    alpha_0960)
        CACHE_SEED="${LM_EVAL_ALPHA_0960_CACHE_SEED:-${DEFAULT_ABI_CACHE_SEED}}"
        EXPECTED_CACHE_SEED_SHA256="f4dfa648fb1c00544671c33780496aac5c8b4372ad36784f63f36361cb02d5b2"
        ;;
    ulp_v1)
        CACHE_SEED="${LM_EVAL_ULP_V1_CACHE_SEED:-${LM_EVAL_TENSORBRIDGE_CACHE_SEED:-}}"
        EXPECTED_CACHE_SEED_SHA256="da6b312e7ee12d4921f0e1178a27816d81065393fbdeb6770376f6ba391bbbbb"
        ;;
esac
if [[ -n "${CACHE_SEED}" ]]; then
    if [[ ! -d "${CACHE_SEED}" ]]; then
        echo "[lm-eval] cache seed directory does not exist: ${CACHE_SEED}" >&2
        exit 2
    fi
    if [[ -n "$(find "${TENSORBRIDGE_CACHE_DIR}" -mindepth 1 -print -quit)" ]]; then
        echo "[lm-eval] destination cache must be empty before seeding" >&2
        exit 2
    fi
    CACHE_SEED_SHA256="$(cache_seed_digest "${CACHE_SEED}")"
    if [[ "${CACHE_SEED_SHA256}" != "${EXPECTED_CACHE_SEED_SHA256}" ]]; then
        echo "[lm-eval] source cache seed SHA256 mismatch" >&2
        exit 2
    fi
    cp -a -- "${CACHE_SEED}/." "${TENSORBRIDGE_CACHE_DIR}/"
    COPIED_CACHE_SEED_SHA256="$(cache_seed_digest "${TENSORBRIDGE_CACHE_DIR}")"
    if [[ "${COPIED_CACHE_SEED_SHA256}" != "${EXPECTED_CACHE_SEED_SHA256}" ]]; then
        echo "[lm-eval] copied cache seed SHA256 mismatch" >&2
        exit 2
    fi
    export TENSORBRIDGE_CACHE_SEED_SOURCE="${CACHE_SEED}"
    export TENSORBRIDGE_CACHE_SEED_SHA256="${COPIED_CACHE_SEED_SHA256}"
    export TENSORBRIDGE_CACHE_SEED_VERIFIED=1
fi

export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
export TENSORBRIDGE_EXTRA_NVRTC_FLAGS=""
export TORCH_EXTENSIONS_DIR="${TMP_DIR}/torch"
export VLLM_CACHE_ROOT="${TMP_DIR}/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=tensorbridge
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
source "${REPO_DIR}/sbatch/_common.sh"

EXTRA_ARGS=()
if [[ -n "${LM_EVAL_LIMIT_COUNT:-}" ]]; then
    EXTRA_ARGS+=(--limit-count "${LM_EVAL_LIMIT_COUNT}")
fi
if [[ -n "${LM_EVAL_LIMIT_FRACTION:-}" ]]; then
    EXTRA_ARGS+=(--limit-fraction "${LM_EVAL_LIMIT_FRACTION}")
fi
if [[ -n "${LM_EVAL_SAMPLE_MANIFEST:-}" ]]; then
    EXTRA_ARGS+=(--sample-manifest "${LM_EVAL_SAMPLE_MANIFEST}")
fi
if [[ -n "${LM_EVAL_NUM_FEWSHOT:-}" ]]; then
    EXTRA_ARGS+=(--num-fewshot "${LM_EVAL_NUM_FEWSHOT}")
fi
if [[ -n "${LM_EVAL_MAX_GEN_TOKS:-}" ]]; then
    EXTRA_ARGS+=(--max-gen-toks "${LM_EVAL_MAX_GEN_TOKS}")
fi
if [[ -n "${LM_EVAL_BOOTSTRAP_ITERS:-}" ]]; then
    EXTRA_ARGS+=(--bootstrap-iters "${LM_EVAL_BOOTSTRAP_ITERS}")
fi
if [[ "${LM_EVAL_ENABLE_THINKING:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-thinking --think-end-token "${LM_EVAL_THINK_END_TOKEN:-</think>}")
fi
if [[ "${LM_EVAL_ALLOW_RUNTIME_VERSION_MISMATCH:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--allow-runtime-version-mismatch)
fi
if [[ "${LM_EVAL_OVERWRITE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "[lm-eval] arm=${ARM} suite=${SUITE} model=${MODEL}"
echo "[lm-eval] checkpoint_manifest=${CHECKPOINT_MANIFEST}"
echo "[lm-eval] output=${OUTPUT} samples=${SAMPLES_DIR}"
echo "[lm-eval] tensorbridge_cache=${TENSORBRIDGE_CACHE_DIR}"
if [[ "${TENSORBRIDGE_CACHE_SEED_VERIFIED:-0}" == "1" ]]; then
    echo "[lm-eval] cache_seed=${TENSORBRIDGE_CACHE_SEED_SOURCE}"
    echo "[lm-eval] cache_seed_sha256=${TENSORBRIDGE_CACHE_SEED_SHA256}"
fi
"${PYTHON}" -u "${REPO_DIR}/scripts/eval_nvfp4_lm_harness.py" \
    --model "${MODEL}" \
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
    --arm "${ARM}" \
    --suite "${SUITE}" \
    --max-model-len "${LM_EVAL_MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}" \
    --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.5}" \
    --max-num-seqs "${LM_EVAL_MAX_NUM_SEQS:-8}" \
    --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
    --output-json "${OUTPUT}" \
    --samples-dir "${SAMPLES_DIR}" \
    "${EXTRA_ARGS[@]}"
