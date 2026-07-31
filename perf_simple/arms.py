"""Arm definitions: the environment that selects which linear kernel runs.

The TensorBridge vLLM plugin reads these while loading weights, so they must be
set before `vllm` is imported. Every script here calls `apply(arm)` first.

Only the 192 transformer MLP projections differ between arms. `lm_head` stays
NVFP4 Marlin W4A16 in every arm, and the 208 FP8 layers are identical, so an arm
difference isolates the MLP kernel — and also means no arm can win by more than
the MLP's share of the forward pass.
"""

from __future__ import annotations

import os

ARMS = {
    # What vLLM picks on its own for this checkpoint. On Hopper that is Marlin
    # W4A16: the CUTLASS and FlashInfer NVFP4 kernels both require sm_100. This
    # is the baseline any "TensorBridge is faster" claim has to beat.
    "official": "official",
    # B8 expanded exactly at load, then vLLM's Cutlass FP8 kernel. Same
    # activation path as TensorBridge, so it separates "the FPMA mainloop" from
    # "using FP8 activations at all".
    "normal_a8": "normal_a8",
    # FPMA-SNC generating B8 in the mainloop.
    "tensorbridge": "tensorbridge",
}


def apply(arm: str) -> dict[str, str]:
    """Set the arm's environment and return what was set, for the result record."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")

    env = {
        "VLLM_PLUGINS": "tensorbridge",
        "TENSORBRIDGE_VLLM_BACKEND": ARMS[arm],
        # Production layout contract; the plugin raises if the kernel does not
        # come up in the swizzle64 + SNC form.
        "TENSORBRIDGE_COMPILER": "nvrtc",
        "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR": "none",
        "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION": "0",
        "TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP": "0",
        # A background compile process would contend with the measurement.
        "TENSORBRIDGE_DISABLE_PARALLEL_BUILD": "1",
        # vLLM 0.20.2 can import its vendored DeepGEMM namespace while missing
        # the APIs its FP8 warmup path calls; this checkpoint selects Cutlass.
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(env)
    return env


def versions() -> dict[str, str]:
    """Package versions that change results if they drift."""
    import importlib.metadata as md

    out = {}
    for name in ("vllm", "tensorbridge-kernels", "torch", "transformers"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def gpu_info() -> dict[str, str]:
    """Device and clock state. Timings are meaningless without both."""
    import subprocess

    try:
        query = "name,driver_version,clocks.max.graphics,clocks.current.graphics"
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()[0]
        name, driver, max_clock, current_clock = (f.strip() for f in raw.split(","))
    except Exception as error:  # noqa: BLE001 - provenance must not break a run
        return {"error": repr(error)}
    return {
        "name": name,
        "driver": driver,
        "max_graphics_clock": max_clock,
        "current_graphics_clock": current_clock,
    }
