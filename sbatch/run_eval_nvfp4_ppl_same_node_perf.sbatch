#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_nvfp4_perf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/logs/tb_nvfp4_perf_%j.out
#SBATCH --error=sbatch/logs/tb_nvfp4_perf_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
RESULT_DIR="${REPO_DIR}/benchmarks/results/ppl/same_node_${JOB_KEY}"
CACHE_ROOT="${REPO_DIR}/benchmarks/tmp/ppl/same_node_${JOB_KEY}"
MAX_BLOCKS="${PPL_PERF_MAX_BLOCKS:-64}"
PRIME_BLOCKS="${PPL_PERF_PRIME_BLOCKS:-${MAX_BLOCKS}}"
mkdir -p "${REPO_DIR}/sbatch/logs" "${RESULT_DIR}" "${CACHE_ROOT}"

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
    local phase="$1"
    local index="$2"
    local arm="$3"
    local blocks="$4"
    local alpha selector ulp

    case "${arm}" in
        default)  alpha=1.0;   selector=none;          ulp=0 ;;
        alpha)    alpha=0.960; selector=none;          ulp=0 ;;
        selector) alpha=0.986; selector=normal_b8_sse; ulp=0 ;;
        ulp)      alpha=1.0;   selector=none;          ulp=1 ;;
        *) echo "unknown performance arm: ${arm}" >&2; return 2 ;;
    esac

    local output="${RESULT_DIR}/${phase}_${index}_${arm}.json"
    local cache="${CACHE_ROOT}/${arm}"
    echo "[same-node-perf] phase=${phase} index=${index} arm=${arm} blocks=${blocks}"
    env \
        REPO_DIR="${REPO_DIR}" \
        PPL_BACKEND=tensorbridge \
        PPL_MAX_BLOCKS="${blocks}" \
        PPL_FPMA_ALPHA="${alpha}" \
        PPL_FPMA_PREFOLD_SELECTOR="${selector}" \
        PPL_FPMA_ULP_CORRECTION="${ulp}" \
        PPL_TENSORBRIDGE_CACHE_DIR="${cache}" \
        PPL_OUTPUT="${output}" \
        TENSORBRIDGE_PERF_PHASE="${phase}" \
        TENSORBRIDGE_PERF_INDEX="${index}" \
        TENSORBRIDGE_PERF_ARM="${arm}" \
        TENSORBRIDGE_PERF_ORDER="${order[*]}" \
        TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP=0 \
        bash "${REPO_DIR}/sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch"
}

read -r -a order <<< "${PPL_PERF_ORDER:-default alpha selector ulp ulp selector alpha default}"
if [[ "${PPL_PERF_PRIME:-1}" == "1" ]]; then
    declare -A primed=()
    prime_index=0
    for arm in "${order[@]}"; do
        if [[ -z "${primed[$arm]+x}" ]]; then
            run_arm prime "${prime_index}" "${arm}" "${PRIME_BLOCKS}"
            primed["${arm}"]=1
            prime_index=$((prime_index + 1))
        fi
    done
fi

for index in "${!order[@]}"; do
    run_arm measured "${index}" "${order[$index]}" "${MAX_BLOCKS}"
done

echo "[same-node-perf] results=${RESULT_DIR}"
