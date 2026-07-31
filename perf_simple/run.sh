#!/usr/bin/env bash
# Run the performance benchmarks. Plain shell, no scheduler.
#
#   MODEL=/path/to/ckpt ./run.sh          # GEMM probe, then the latency arms
#   MODEL=... ./run.sh gemm               # GEMM probe only (no model needed)
#   MODEL=... ./run.sh latency            # latency arms only
#
# The arm order is deliberately ABBA -- official, normal_a8, tensorbridge,
# tensorbridge, normal_a8, official. Running each arm twice at opposite ends
# separates a real difference between arms from drift over the session.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${HERE}/.." && pwd)}"

PYTHON="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
MODEL="${MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
OUT="${OUT:-${HERE}/results}"
WHAT="${1:-all}"

die() { echo "[perf] $*" >&2; exit 1; }

[[ -x "${PYTHON}" ]] || die "no Python at ${PYTHON} (set PYTHON=...)"
[[ -d "${MODEL}"  ]] || die "no checkpoint at ${MODEL} (set MODEL=...)"

CC="${CC:-$(command -v gcc || true)}"
CXX="${CXX:-$(command -v g++ || true)}"
[[ -n "${CXX}" ]] || die "no g++ on PATH"
GCC_MAJOR="$("${CXX}" -dumpversion | cut -d. -f1)"
(( GCC_MAJOR >= 9 )) || die "g++ ${GCC_MAJOR} is too old, need >= 9"
export CC CXX
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CXX}}"

# vLLM puts its ZMQ IPC socket under TMPDIR, and a Unix domain socket path is
# capped at 107 characters. A long TMPDIR fails engine startup with an error
# that does not mention path length, so keep it short and node-local.
export TMPDIR="${PERF_TMPDIR:-/tmp/tbperf$$}"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
export VLLM_TRITON_CACHE_BASE="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${VLLM_CACHE_ROOT}" \
         "${VLLM_TRITON_CACHE_BASE}" "${OUT}"
[[ "${KEEP_CACHE:-0}" == "1" ]] || trap 'rm -rf "${TMPDIR}"' EXIT

export TOKENIZERS_PARALLELISM=false

echo "[perf] host=$(hostname) model=${MODEL} out=${OUT}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.graphics --format=csv,noheader

# Inductor-generated Triton kernels fault on some older drivers once CUDA graphs
# are active. Fail immediately rather than after minutes of engine startup.
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
for bad in ${BAD_DRIVERS:-570.86.10}; do
    [[ "${DRIVER}" != "${bad}" ]] || die "driver ${DRIVER} faults under CUDA graphs; use another machine or set BAD_DRIVERS="
done

cd "${HERE}"

if [[ "${WHAT}" == "all" || "${WHAT}" == "gemm" ]]; then
    echo "[perf] === GEMM probe ==="
    # bench_gemm brings up a one-rank torch.distributed group; override the port
    # if another run on this machine already holds the default.
    "${PYTHON}" -u bench_gemm.py --model "${MODEL}" --output "${OUT}/gemm.json" \
        ${PERF_DIST_PORT:+--dist-port "${PERF_DIST_PORT}"}
fi

if [[ "${WHAT}" == "all" || "${WHAT}" == "latency" ]]; then
    i=0
    for arm in official normal_a8 tensorbridge tensorbridge normal_a8 official; do
        echo "[perf] === latency ${i}: ${arm} ==="
        # Per-arm compile cache: repeated runs of one arm must not race on it.
        export TENSORBRIDGE_CACHE_DIR="${TMPDIR}/cache_${arm}"
        mkdir -p "${TENSORBRIDGE_CACHE_DIR}"
        "${PYTHON}" -u bench_latency.py --arm "${arm}" --model "${MODEL}" \
            --output "${OUT}/$(printf %02d $i)_${arm}.json"
        i=$((i + 1))
    done
fi

echo "[perf] done. Summarise with:"
echo "[perf]   ${PYTHON} ${HERE}/compare.py ${OUT}"
