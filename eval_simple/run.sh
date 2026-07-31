#!/usr/bin/env bash
# Run the accuracy evaluation for one or more arms. Plain shell, no scheduler.
#
#   ./run.sh                        # official, normal_a8, alpha_0961 in sequence
#   ./run.sh alpha_0961             # one arm
#   MODEL=/path/to/ckpt ./run.sh    # different checkpoint
#   SMOKE=1 ./run.sh alpha_0961     # 8 blocks and 32 documents, a few minutes
#
# Results land in results/<RUN_ID>/. Arms of one comparison must share that
# directory: compare.py pairs them by document and refuses to mix runs.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${HERE}/.." && pwd)}"

PYTHON="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
MODEL="${MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${HERE}/results/${RUN_ID}}"
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=(official normal_a8 alpha_0961)

die() { echo "[run] $*" >&2; exit 1; }

[[ -x "${PYTHON}" ]] || die "no Python at ${PYTHON} (set PYTHON=...)"
[[ -d "${MODEL}"  ]] || die "no checkpoint at ${MODEL} (set MODEL=...)"

# TensorBridge compiles its kernels through NVRTC at runtime and PyTorch's
# extension loader needs a modern host compiler. RHEL-family systems default to
# GCC 8, which fails; load a newer one before running this.
CC="${CC:-$(command -v gcc || true)}"
CXX="${CXX:-$(command -v g++ || true)}"
[[ -n "${CXX}" ]] || die "no g++ on PATH"
GCC_MAJOR="$("${CXX}" -dumpversion | cut -d. -f1)"
(( GCC_MAJOR >= 9 )) || die "g++ ${GCC_MAJOR} is too old, need >= 9 (module load a newer gcc)"
export CC CXX
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CXX}}"

# Keep compile caches off shared storage. Triton's compile-write-rename-read
# cycle loses races on NFS once inductor is active, surfacing as FileNotFoundError
# on a .cubin that was just produced. Per-run paths also stop concurrent arms
# from corrupting each other.
CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/tb_eval_$$}"
export TENSORBRIDGE_CACHE_DIR="${CACHE_ROOT}/tensorbridge"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export VLLM_TRITON_CACHE_BASE="${CACHE_ROOT}/triton"
mkdir -p "${TENSORBRIDGE_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
         "${VLLM_CACHE_ROOT}" "${VLLM_TRITON_CACHE_BASE}" "${OUT}"
[[ "${KEEP_CACHE:-0}" == "1" ]] || trap 'rm -rf "${CACHE_ROOT}"' EXIT

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[run] host=$(hostname) python=${PYTHON}"
echo "[run] model=${MODEL}"
echo "[run] arms=${ARMS[*]} out=${OUT}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Inductor-generated Triton kernels fault on some older drivers -- "an illegal
# instruction was encountered", or engine init dies outright -- and only once
# CUDA graphs are on, since eager never invokes inductor. Fail in seconds with a
# reason rather than after minutes of engine startup.
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
for bad in ${BAD_DRIVERS:-570.86.10}; do
    [[ "${DRIVER}" != "${bad}" ]] || die "driver ${DRIVER} faults under CUDA graphs; use another machine or set BAD_DRIVERS="
done

PPL_ARGS=() ; GSM_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    PPL_ARGS=(--max-blocks 8) ; GSM_ARGS=(--limit 32)
    echo "[run] SMOKE: 8 blocks and 32 documents; not comparable to the reference tables"
fi

cd "${HERE}"
for arm in "${ARMS[@]}"; do
    echo "[run] === ${arm} ==="
    "${PYTHON}" -u eval_ppl.py   --arm "${arm}" --model "${MODEL}" \
        --output "${OUT}/${arm}_ppl.json" "${PPL_ARGS[@]}"
    "${PYTHON}" -u eval_gsm8k.py --arm "${arm}" --model "${MODEL}" \
        --output "${OUT}/${arm}_gsm8k.json" \
        --samples-dir "${OUT}/${arm}_gsm8k_samples" "${GSM_ARGS[@]}"
done

echo "[run] done. Summarise with:"
echo "[run]   ${PYTHON} ${HERE}/compare.py ${OUT}"
