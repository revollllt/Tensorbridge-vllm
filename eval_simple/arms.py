"""Arm definitions: the environment that selects which linear kernel runs.

The TensorBridge vLLM plugin reads these variables while loading weights, so
they must be set before `vllm` is imported. Every script here calls
`apply(arm)` as its first action for that reason.

Only the 192 transformer MLP projections differ between arms. `lm_head` stays
NVFP4 Marlin W4A16 with a BF16 activation everywhere, and the 208 FP8 layers are
identical, so an arm difference isolates the MLP kernel.
"""

from __future__ import annotations

import os

# backend, alpha
ARMS = {
    # NVFP4 weights through vLLM Marlin W4A16, BF16 activation.
    "official": ("official", None),
    # NVFP4 weights expanded to exact E4M3 at load, vLLM Cutlass FP8 kernel,
    # dynamic per-token FP8 activation. This is the exact-arithmetic counterpart
    # of the FPMA path and therefore the right baseline for an FPMA question.
    "normal_a8": ("normal_a8", None),
    # FPMA-SNC generating B8 in the mainloop, same activation path as normal_a8,
    # with the analytic one-ULP bias compensation 123/128 -> 0.961.
    "alpha_0961": ("tensorbridge", "0.961"),
    # Same kernel with the compensation switched off. Included because it is what
    # makes the compensation's effect visible, and because alpha plays no part at
    # 1.0, which makes it a clean check against a reference value.
    "fpma_default": ("tensorbridge", "1.0"),
}


def apply(arm: str) -> dict[str, str]:
    """Set the arm's environment and return what was set, for the result record."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    backend, alpha = ARMS[arm]

    env = {
        "VLLM_PLUGINS": "tensorbridge",
        "TENSORBRIDGE_VLLM_BACKEND": backend,
        # Production layout contract. The plugin asserts these after loading and
        # raises if the kernel did not come up in the swizzle64 + SNC form.
        "TENSORBRIDGE_COMPILER": "nvrtc",
        "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR": "none",
        "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION": "0",
        "TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP": "0",
        "TENSORBRIDGE_STRICT_QWEN36_LAYOUT": "1",
        # vLLM 0.20.2 can import its vendored DeepGEMM namespace while missing the
        # APIs its FP8 warmup path calls; this checkpoint selects Cutlass anyway.
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if alpha is not None:
        # Set explicitly rather than relying on the plugin's implicit default, so
        # the value that produced a number is always visible in the record.
        env["TENSORBRIDGE_NVFP4_FPMA_ALPHA"] = alpha

    os.environ.update(env)
    return env


def confirm_active() -> str:
    """Prove the plugin is in charge, not merely requested. Returns its config class.

    Setting TENSORBRIDGE_VLLM_BACKEND states an intent. If the plugin were not
    registered — entry point missing, a stale or non-editable install, VLLM_PLUGINS
    dropped by a job script — vLLM would fall back to its own `modelopt_mixed`
    implementation, no TensorBridge code would run, and every arm would quietly
    produce roughly the `official` numbers.

    That is the one failure mode here that yields plausible results instead of an
    error, so it gets an explicit check. The rest fail loudly on their own: an
    unknown backend raises, a checkpoint whose layout is not {NVFP4: 193, FP8: 208}
    raises under TENSORBRIDGE_STRICT_QWEN36_LAYOUT=1, and a layer that comes up
    without SNC and the swizzle64 layout raises "production contract is inactive".
    """
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    cls = get_quantization_config("modelopt_mixed")
    qualified = f"{cls.__module__}.{cls.__name__}"
    if cls.__name__ != "TensorBridgeModelOptMixedConfig":
        raise RuntimeError(
            f"the TensorBridge plugin is not active: modelopt_mixed resolves to "
            f"{qualified}. Results would come from vLLM's own kernels. Check that "
            f"VLLM_PLUGINS=tensorbridge and that this repository is pip-installed."
        )
    return qualified


def versions() -> dict[str, str]:
    """Package versions that change results if they drift."""
    import importlib.metadata as md

    out = {}
    for name in (
        "vllm",
        "tensorbridge-kernels",
        "torch",
        "transformers",
        "datasets",
        "lm_eval",
    ):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not-installed"
    return out
