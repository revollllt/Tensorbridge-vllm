#!/usr/bin/env bash
# Slurm wrapper around run.sh: load this cluster's toolchain, then hand over.
# Everything about how the benchmarks run lives in run.sh so the two stay in step.
#
#   MODEL=/path/to/ckpt sbatch --export=ALL perf_simple/run_all.sbatch
#   MODEL=... sbatch --export=ALL perf_simple/run_all.sbatch gemm

#SBATCH -p acd_u
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=160G --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/perf_simple_%j.out
#SBATCH --error=sbatch/logs/perf_simple_%j.err

# One node, one job: the 21 GB checkpoint is read cold once and served from page
# cache for every later engine start. Spreading arms across nodes made startup
# an order of magnitude worse in an earlier attempt.
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
mkdir -p "${REPO_DIR}/sbatch/logs"

module purge
module load "${TENSORBRIDGE_CUDA_MOD:-cuda/12.8}"
module load "${TENSORBRIDGE_GCC_MOD:-gcc/13.3}"

# Short and node-local; run.sh explains the 107-character socket path limit.
export PERF_TMPDIR="/tmp/tbps${SLURM_JOB_ID}"

exec "${REPO_DIR}/perf_simple/run.sh" "${1:-all}"
