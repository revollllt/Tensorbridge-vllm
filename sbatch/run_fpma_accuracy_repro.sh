#!/usr/bin/env bash
# Reproduce the FPMA accuracy comparison: WikiText-2 PPL and full GSM8K across
# official / normal_a8 / alpha_0961.
#
# This is a submission driver, not a Slurm job. Run it from a login node.
#
#   ./sbatch/run_fpma_accuracy_repro.sh preflight
#   ./sbatch/run_fpma_accuracy_repro.sh ppl
#   ./sbatch/run_fpma_accuracy_repro.sh gsm8k
#   ./sbatch/run_fpma_accuracy_repro.sh analyze <lm_eval_run_id>
#
# See docs/FPMA_ACCURACY_REPRO.md for expected values and the working-tree
# constraint that GSM8K enforces.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${TENSORBRIDGE_VLLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
MODEL="${TENSORBRIDGE_REPRO_MODEL:-/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4}"
MANIFEST="${TENSORBRIDGE_REPRO_MANIFEST:-${REPO_DIR}/benchmarks/manifests/qwen3.6-27b-nvfp4.sha256.json}"

cd "${REPO_DIR}"

die() { echo "[repro] $*" >&2; exit 1; }
note() { echo "[repro] $*"; }

[[ -x "${PYTHON}" ]] || die "missing vLLM runtime: ${PYTHON}"

# GSM8K records the source-tree state before and after lm-eval and fails the run
# if it changed. `git status --porcelain=v1` feeds that hash, so an untracked
# file appearing mid-run kills an hour of GPU time. Snapshot it up front and let
# the caller confirm the tree will stay still.
tree_state() { git status --porcelain=v1 | sha256sum | awk '{print $1}'; }

require_quiet_tree() {
    note "working-tree state hash: $(tree_state)"
    note "GSM8K fails if this changes mid-run (untracked files count)."
    note "Do not add, remove, or edit files in this repo until the arms finish."
}

cmd_preflight() {
    note "verifying the pinned TensorBridge wheel against constraints/tensorbridge.json"
    "${PYTHON}" scripts/verify_tensorbridge_constraint.py

    note "checking the alpha_0961 arm is registered"
    "${PYTHON}" - <<'PY'
from vllm.plugins.tensorbridge_evaluation.lm_harness import resolve_arm
arm = resolve_arm("alpha_0961")
assert (arm.backend, arm.alpha, arm.selector, arm.ulp_correction) == (
    "tensorbridge", 0.961, "none", False
), arm
print(f"[repro] arm ok: {arm}")
PY

    if [[ -f "${MANIFEST}" ]]; then
        note "checkpoint manifest present: ${MANIFEST}"
        "${PYTHON}" scripts/build_checkpoint_manifest.py \
            --model "${MODEL}" --output "${MANIFEST}" --verify
    else
        note "checkpoint manifest missing; build it with:"
        note "  TENSORBRIDGE_BUILD_CHECKPOINT_MANIFEST=1 \\"
        note "    sbatch --time=01:30:00 --export=ALL sbatch/run_tensorbridge_vllm_pytest.sbatch"
        die "manifest required before GSM8K (lm-eval verifies it before and after each run)"
    fi
    note "preflight passed"
}

# Full WikiText-2: 297,192 scored tokens over 291 blocks per arm.
# The tensorbridge arm deliberately leaves PPL_FPMA_ALPHA unset so alpha resolves
# through the implicit analytic_v1 default (0.961) and keeps its scale-domain check.
cmd_ppl() {
    local jid
    for backend in tensorbridge official normal_a8; do
        jid=$(PPL_BACKEND="${backend}" PPL_MAX_BLOCKS=all \
            sbatch --parsable --export=ALL sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch)
        note "ppl ${backend}: job ${jid} -> benchmarks/results/ppl/wikitext2_${backend}_${jid}.json"
    done

    if [[ "${TENSORBRIDGE_REPRO_WITH_ALPHA1:-0}" == "1" ]]; then
        # Uncompensated FPMA. Also the clean discriminator against the frozen
        # record: alpha plays no part at 1.0, so a match proves kernel identity.
        jid=$(PPL_BACKEND=tensorbridge PPL_MAX_BLOCKS=all PPL_FPMA_ALPHA=1.0 \
            PPL_OUTPUT="${REPO_DIR}/benchmarks/results/ppl/wikitext2_tensorbridge_alpha1.0_fpma_default.json" \
            sbatch --parsable --export=ALL sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch)
        note "ppl fpma_default (alpha=1.0): job ${jid}"
    fi
}

# ARMS index in run_eval_nvfp4_lm_harness.sbatch:
#   0 official  1 normal_a8  2 fpma_default  3 selector_alpha1
#   4 ulp_v1    5 alpha_0960 6 alpha_0961
# The array form shares one RUN_ID so all arms land in the same result dir,
# which is what makes the paired analysis possible.
cmd_gsm8k() {
    require_quiet_tree
    local jid
    jid=$(LM_EVAL_SUITE=generation_core \
        sbatch --parsable --array="${TENSORBRIDGE_REPRO_ARMS:-0,1,6}" --time=08:00:00 \
        --export=ALL sbatch/run_eval_nvfp4_lm_harness.sbatch)
    note "gsm8k array job ${jid}"
    note "results: benchmarks/results/lm_eval/${jid}/generation_core/"
    note "when all arms finish: $0 analyze ${jid}"
}

cmd_analyze() {
    local run_id="${1:-}"
    [[ -n "${run_id}" ]] || die "usage: $0 analyze <lm_eval_run_id>"
    local dir="${REPO_DIR}/benchmarks/results/lm_eval/${run_id}/generation_core"
    [[ -d "${dir}" ]] || die "no such result dir: ${dir}"
    "${PYTHON}" scripts/analyze_gsm8k_paired.py "${dir}" \
        --json "${dir}/paired_analysis.json"
}

case "${1:-}" in
    preflight) cmd_preflight ;;
    ppl)       cmd_ppl ;;
    gsm8k)     cmd_gsm8k ;;
    analyze)   shift; cmd_analyze "$@" ;;
    all)       cmd_preflight; cmd_ppl; cmd_gsm8k ;;
    *)
        cat >&2 <<EOF
usage: $0 {preflight|ppl|gsm8k|analyze <run_id>|all}

env overrides:
  TENSORBRIDGE_REPRO_WITH_ALPHA1=1   also run the uncompensated alpha=1.0 PPL arm
  TENSORBRIDGE_REPRO_ARMS=0,1,2,6    override the GSM8K array indices
  TENSORBRIDGE_REPRO_MODEL=<path>    override the checkpoint
EOF
        exit 2
        ;;
esac
