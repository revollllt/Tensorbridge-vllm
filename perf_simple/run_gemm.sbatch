#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=160G --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=../sbatch/logs/perf_gemm_%j.out
#SBATCH --error=../sbatch/logs/perf_gemm_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge && module load cuda/12.8 gcc/13.3
export CC=$(command -v gcc) CXX=$(command -v g++)
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export TMPDIR="/tmp/tbg${SLURM_JOB_ID}"
export TENSORBRIDGE_CACHE_DIR="${TMPDIR}/cache"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch"
mkdir -p "${TENSORBRIDGE_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
../.venv/bin/python bench_gemm.py --output results/gemm.json --dist-port 29613
