#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --job-name=tb_vllm_pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/tb_vllm_pytest_%j.out
#SBATCH --error=sbatch/logs/tb_vllm_pytest_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
PYTHON="${TENSORBRIDGE_VLLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[pytest] missing vLLM runtime: ${PYTHON}" >&2
    exit 2
fi

mkdir -p "${REPO_DIR}/sbatch/logs"
mkdir -p "${REPO_DIR}/benchmarks/tmp/vllm_pytest/${JOB_KEY}/tensorbridge"
mkdir -p "${REPO_DIR}/benchmarks/tmp/vllm_pytest/${JOB_KEY}/torch"

export REPO_DIR
export TENSORBRIDGE_PYTHON="${PYTHON}"
export TENSORBRIDGE_CLEAR_CACHE=0
export TENSORBRIDGE_CACHE_DIR="${REPO_DIR}/benchmarks/tmp/vllm_pytest/${JOB_KEY}/tensorbridge"
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/benchmarks/tmp/vllm_pytest/${JOB_KEY}/torch"
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
source "${REPO_DIR}/sbatch/_common.sh"

TEST_TARGETS=(
    tests/test_vllm_modelopt_adapter.py
    tests/test_ppl_accounting.py
    tests/test_lm_harness_contract.py
    tests/test_lm_confirmation_analysis.py
    tests/test_lm_stage_analysis.py
    tests/test_tensorbridge_dependency_boundary.py
)

echo "[pytest] targets=${TEST_TARGETS[*]}"
"${PYTHON}" -m pytest -p no:cacheprovider -q --maxfail=1 "${TEST_TARGETS[@]}"

if [[ "${TENSORBRIDGE_BUILD_CHECKPOINT_MANIFEST:-0}" == "1" ]]; then
    MODEL="${TENSORBRIDGE_CHECKPOINT_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
    DEFAULT_MANIFEST="${REPO_DIR}/benchmarks/manifests/qwen3.6-27b-nvfp4.sha256.json"
    MANIFEST="${TENSORBRIDGE_CHECKPOINT_MANIFEST:-${DEFAULT_MANIFEST}}"
    mkdir -p "$(dirname "${MANIFEST}")"
    echo "[checkpoint] model=${MODEL} manifest=${MANIFEST}"
    "${PYTHON}" -u "${REPO_DIR}/scripts/build_checkpoint_manifest.py" \
        --model "${MODEL}" \
        --output "${MANIFEST}" \
        --workers "${TENSORBRIDGE_CHECKPOINT_HASH_WORKERS:-3}" \
        --overwrite
fi
