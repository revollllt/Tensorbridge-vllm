#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_nvfp4_fdo_nsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/tb_nvfp4_fdo_nsys_%j.out
#SBATCH --error=sbatch/logs/tb_nvfp4_fdo_nsys_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
TP_SIZE="${PPL_TENSOR_PARALLEL_SIZE:-1}"
NSYS="${NSYS_BIN:-/usr/local/cuda-12.8/bin/nsys}"
EAGER_REFERENCE="${PPL_EAGER_REFERENCE_JSON:?set PPL_EAGER_REFERENCE_JSON}"
REPORT_DIR="${NSYS_REPORT_DIR:-${REPO_DIR}/benchmarks/results/nsys}"
REPORT_PREFIX="${NSYS_REPORT_PREFIX:-${REPORT_DIR}/tensorbridge_fdo_tp${TP_SIZE}_${JOB_KEY}}"
PPL_OUTPUT="${PPL_OUTPUT:-${REPO_DIR}/benchmarks/results/ppl/wikitext2_tensorbridge_fdo_tp${TP_SIZE}_nsys_${JOB_KEY}.json}"
NSYS_TMPDIR="${NSYS_TMPDIR:-${REPO_DIR}/benchmarks/tmp/nsys/${JOB_KEY}}"

if [[ ! -x "${NSYS}" ]]; then
    echo "[nsys] missing executable: ${NSYS}" >&2
    exit 2
fi
mkdir -p "${REPORT_DIR}" "$(dirname "${PPL_OUTPUT}")" "${NSYS_TMPDIR}"
export TMPDIR="${NSYS_TMPDIR}"

DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION=0
if (( TP_SIZE > 1 )); then
    DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION=1
fi

export PPL_BACKEND=tensorbridge
export PPL_EXECUTION_MODE=cudagraph
export PPL_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export PPL_COMPILATION_MODE=NONE
export PPL_CUDAGRAPH_CAPTURE_SIZES=1
export PPL_MAX_MODEL_LEN="${PPL_MAX_MODEL_LEN:-256}"
export PPL_TARGET_TOKENS_PER_BLOCK="${PPL_TARGET_TOKENS_PER_BLOCK:-31}"
export PPL_BLOCK_START="${PPL_BLOCK_START:-26}"
export PPL_MAX_BLOCKS=1
export PPL_MAX_NUM_SEQS=1
export PPL_REQUESTED_OUTPUT_TOKENS=8
export PPL_REPEAT_RUNS="${PPL_REPEAT_RUNS:-12}"
export PPL_TENSOR_PARALLEL_SIZE="${TP_SIZE}"
export PPL_GPU_MEMORY_UTILIZATION="${PPL_GPU_MEMORY_UTILIZATION:-0.5}"
export PPL_VALIDATION_DETAILS=1
export PPL_DISABLE_ALLREDUCE_RMS_FUSION="${PPL_DISABLE_ALLREDUCE_RMS_FUSION:-${DEFAULT_DISABLE_ALLREDUCE_RMS_FUSION}}"
export PPL_EAGER_REFERENCE_JSON="${EAGER_REFERENCE}"
export PPL_OUTPUT

echo "[nsys] tp=${TP_SIZE} repeats=${PPL_REPEAT_RUNS} report=${REPORT_PREFIX}.nsys-rep"
set +e
"${NSYS}" profile \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace=graph:host-only \
    --force-overwrite=true \
    --output="${REPORT_PREFIX}" \
    bash "${REPO_DIR}/sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch"
APP_STATUS=$?
set -e

REPORT="${REPORT_PREFIX}.nsys-rep"
if [[ ! -f "${REPORT}" ]]; then
    echo "[nsys] report was not created: ${REPORT}" >&2
    exit 3
fi
"${NSYS}" stats --report cuda_api_sum "${REPORT}" \
    > "${REPORT_PREFIX}_cuda_api_sum.txt"
echo "[nsys] application_status=${APP_STATUS}"
exit "${APP_STATUS}"
