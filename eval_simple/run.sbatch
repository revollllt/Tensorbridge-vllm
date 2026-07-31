#!/usr/bin/env bash
# Slurm wrapper around run.sh: load this cluster's toolchain, then hand one arm
# to the portable script. Everything about how the evaluation runs lives in
# run.sh, so the two stay in step.
#
#   sbatch --array=0-2 eval_simple/run.sbatch          # three arms in parallel
#   RUN_ARMS="official" RUN_ID=457238 \
#     sbatch eval_simple/run.sbatch                    # rejoin an existing run
#
# Arms of one comparison must share RUN_ID. Submitting as an array does that
# automatically; a re-run of a single failed arm needs RUN_ID set by hand.

#SBATCH -p acd_u
#SBATCH --job-name=tb_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ACD1-1
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/tb_eval_%j.out
#SBATCH --error=sbatch/logs/tb_eval_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
read -r -a ARMS <<< "${RUN_ARMS:-official normal_a8 alpha_0961}"

# One arm per array task. Without an array, run whatever RUN_ARMS holds.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    (( SLURM_ARRAY_TASK_ID < ${#ARMS[@]} )) \
        || { echo "array index must be in 0..$(( ${#ARMS[@]} - 1 ))" >&2; exit 2; }
    SELECTED=("${ARMS[SLURM_ARRAY_TASK_ID]}")
else
    SELECTED=("${ARMS[@]}")
fi

mkdir -p "${REPO_DIR}/sbatch/logs"

module purge
module load "${TENSORBRIDGE_CUDA_MOD:-cuda/12.8}"
module load "${TENSORBRIDGE_GCC_MOD:-gcc/13.3}"

# Arms of one comparison share a results directory, so the default has to be the
# array job id rather than each task's own id.
export RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
export CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/tb_eval/${SLURM_JOB_ID}"

# Compute nodes here have no outbound network; datasets and tokenizer must come
# from the local cache. Drop these on a machine that needs to download them.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec "${REPO_DIR}/eval_simple/run.sh" "${SELECTED[@]}"
