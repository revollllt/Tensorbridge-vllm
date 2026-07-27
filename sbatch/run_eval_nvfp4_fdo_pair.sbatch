#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_nvfp4_fdo_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_nvfp4_fdo_pair_%j.out
#SBATCH --error=sbatch/logs/tb_nvfp4_fdo_pair_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
TP_SIZE="${PPL_TENSOR_PARALLEL_SIZE:-1}"
PAIR_REPEAT_RUNS="${PPL_PAIR_REPEAT_RUNS:-12}"
DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION=0
if (( TP_SIZE > 1 )); then
    DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION=1
fi
RESULT_DIR="${REPO_DIR}/benchmarks/results/ppl"
PAIR_CACHE_DIR="${PPL_TENSORBRIDGE_CACHE_DIR:-${REPO_DIR}/benchmarks/tmp/ppl/${JOB_KEY}/tensorbridge}"
EAGER_OUTPUT="${PPL_PAIR_EAGER_OUTPUT:-${RESULT_DIR}/wikitext2_tensorbridge_fdo_tp${TP_SIZE}_eager_${JOB_KEY}.json}"
GRAPH_OUTPUT="${PPL_PAIR_GRAPH_OUTPUT:-${RESULT_DIR}/wikitext2_tensorbridge_fdo_tp${TP_SIZE}_graph_${JOB_KEY}.json}"

mkdir -p "${RESULT_DIR}" "${PAIR_CACHE_DIR}"

COMMON_ENV=(
    "REPO_DIR=${REPO_DIR}"
    "PPL_BACKEND=tensorbridge"
    "PPL_CUDAGRAPH_MODE=FULL_DECODE_ONLY"
    "PPL_COMPILATION_MODE=NONE"
    "PPL_MAX_MODEL_LEN=${PPL_MAX_MODEL_LEN:-256}"
    "PPL_TARGET_TOKENS_PER_BLOCK=${PPL_TARGET_TOKENS_PER_BLOCK:-31}"
    "PPL_MAX_BLOCKS=1"
    "PPL_MAX_NUM_SEQS=1"
    "PPL_REQUESTED_OUTPUT_TOKENS=8"
    "PPL_REPEAT_RUNS=${PAIR_REPEAT_RUNS}"
    "PPL_TENSOR_PARALLEL_SIZE=${TP_SIZE}"
    "PPL_DISABLE_ALLREDUCE_RMS_FUSION=${PPL_DISABLE_ALLREDUCE_RMS_FUSION:-${DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION}}"
    "PPL_GPU_MEMORY_UTILIZATION=${PPL_GPU_MEMORY_UTILIZATION:-0.5}"
    "PPL_VALIDATION_DETAILS=1"
    "PPL_TENSORBRIDGE_CACHE_DIR=${PAIR_CACHE_DIR}"
)

echo "[fdo-pair] tp=${TP_SIZE} repeats=${PAIR_REPEAT_RUNS} cache=${PAIR_CACHE_DIR}"
echo "[fdo-pair] eager=${EAGER_OUTPUT}"
env -u PPL_EAGER_REFERENCE_JSON -u PPL_CUDAGRAPH_CAPTURE_SIZES \
    "${COMMON_ENV[@]}" \
    PPL_EXECUTION_MODE=eager \
    PPL_OUTPUT="${EAGER_OUTPUT}" \
    bash "${REPO_DIR}/sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch"

echo "[fdo-pair] graph=${GRAPH_OUTPUT}"
env "${COMMON_ENV[@]}" \
    PPL_EXECUTION_MODE=cudagraph \
    PPL_CUDAGRAPH_CAPTURE_SIZES=1 \
    PPL_EAGER_REFERENCE_JSON="${EAGER_OUTPUT}" \
    PPL_OUTPUT="${GRAPH_OUTPUT}" \
    bash "${REPO_DIR}/sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch"
