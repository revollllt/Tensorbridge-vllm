#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/tb_quant_%j.out
#SBATCH --error=sbatch/logs/tb_quant_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JOB_KEY="${SLURM_JOB_ID:-manual_$$}"
export TENSORBRIDGE_CACHE_DIR="${REPO_DIR}/benchmarks/tmp/quant/${JOB_KEY}"
export TENSORBRIDGE_CLEAR_CACHE=0
export TENSORBRIDGE_DISABLE_PARALLEL_BUILD=1
mkdir -p "${TENSORBRIDGE_CACHE_DIR}" "${REPO_DIR}/sbatch/logs"
source "${REPO_DIR}/sbatch/_common.sh"
export TENSORBRIDGE_COMPILER=nvrtc
export VLLM_CUTLASS_ROOT="${REPO_DIR}"
"${PYTHON}" -u "${REPO_DIR}/benchmarks/bench_activation_quant.py" \
  --output "${REPO_DIR}/benchmarks/results/kernel/activation_quant_${JOB_KEY}.json"
