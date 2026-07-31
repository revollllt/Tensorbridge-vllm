"""C++ fast path for the current NVFP4 swizzle64_raw scheduler."""
from __future__ import annotations

import os
import threading
import warnings
from typing import Any, Mapping


_NVFP4_SWIZZLE64_FIELDS = (
    "block_m",
    "block_n",
    "block_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "use_stream_k",
    "use_f16_accum",
    "num_stages",
    "use_warp_spec",
    "use_tma",
    "use_tma_b",
    "use_tma_c",
    "use_tma_bs",
    "use_tma_bzp",
    "use_mbarrier",
    "num_ctas_per_sm",
    "multi_cast_size_a",
    "multi_cast_size_b",
    "use_tma_b_swizzle_64",
    "use_m_fast_tile_order",
    "use_nvfp4_prefetch_raw_wait3_issue_auto",
    "warp_spec_producer_regs",
    "nvfp4_swz64_prebcast_prmt_const_variant",
)
_NVFP4_SWIZZLE64_CONFIG_KEYS = (
    "block_shape",
    "warp_shape",
    "use_stream_k",
    "use_f16_accum",
    "num_stages",
    "use_warp_spec",
    "use_tma",
    "use_tma_b",
    "use_tma_c",
    "use_tma_bs",
    "use_tma_bzp",
    "use_mbarrier",
    "num_ctas_per_sm",
    "multi_cast_size_a",
    "multi_cast_size_b",
    "use_tma_b_swizzle_64",
)
_NVFP4_SWIZZLE64_OPTIONAL_CONFIG_KEYS = (
    "use_m_fast_tile_order",
    "use_nvfp4_prefetch_raw_wait3_issue_auto",
    "warp_spec_producer_regs",
    "nvfp4_swz64_prebcast_prmt_const_variant",
)

_OP_UNINITIALIZED = "uninitialized"
_OP_READY = "ready"
_OP_FAILED = "failed"
_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE = _OP_UNINITIALIZED
_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP = None
_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_ERROR: Exception | None = None
_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_LOCK = threading.Lock()
_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED = False


def _override_to_int(use_stream_k: bool | None) -> int:
    if use_stream_k is None:
        return -1
    return 1 if use_stream_k else 0


def _canonicalize_nvfp4_swizzle64_raw_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the stable raw-config/JSON contract shared with Python routing."""
    canonical: dict[str, Any] = {
        "block_shape": tuple(int(item) for item in config["block_shape"]),
        "warp_shape": tuple(int(item) for item in config["warp_shape"]),
        "use_stream_k": bool(config.get("use_stream_k", False)),
        "use_f16_accum": bool(config.get("use_f16_accum", False)),
        "num_stages": int(config.get("num_stages", 4)),
        "use_warp_spec": bool(config.get("use_warp_spec", True)),
        "use_tma": bool(config.get("use_tma", True)),
        "use_tma_b": bool(config.get("use_tma_b", True)),
        "use_tma_c": bool(config.get("use_tma_c", True)),
        "use_tma_bs": bool(config.get("use_tma_bs", False)),
        "use_tma_bzp": bool(config.get("use_tma_bzp", False)),
        "use_mbarrier": bool(config.get("use_mbarrier", True)),
        "num_ctas_per_sm": int(config.get("num_ctas_per_sm", 1)),
        "multi_cast_size_a": int(config.get("multi_cast_size_a", 1)),
        "multi_cast_size_b": int(config.get("multi_cast_size_b", 1)),
        "use_tma_b_swizzle_64": bool(config.get("use_tma_b_swizzle_64", True)),
    }
    if bool(config.get("use_m_fast_tile_order", False)):
        canonical["use_m_fast_tile_order"] = True
    if bool(config.get("use_nvfp4_prefetch_raw_wait3_issue_auto", False)):
        canonical["use_nvfp4_prefetch_raw_wait3_issue_auto"] = True
    if int(config.get("warp_spec_producer_regs", 0)):
        canonical["warp_spec_producer_regs"] = int(config["warp_spec_producer_regs"])
    if bool(config.get("nvfp4_swz64_prebcast_prmt_const_variant", False)):
        canonical["nvfp4_swz64_prebcast_prmt_const_variant"] = True
    known = set(_NVFP4_SWIZZLE64_CONFIG_KEYS) | set(
        _NVFP4_SWIZZLE64_OPTIONAL_CONFIG_KEYS
    )
    for name in sorted(set(config) - known):
        canonical[name] = config[name]
    return canonical


def _packed_to_config(values: list[int]) -> dict[str, Any]:
    if len(values) != len(_NVFP4_SWIZZLE64_FIELDS):
        raise RuntimeError(
            "unexpected C++ NVFP4 router result length: "
            f"{len(values)} != {len(_NVFP4_SWIZZLE64_FIELDS)}"
        )
    data = dict(zip(_NVFP4_SWIZZLE64_FIELDS, values))
    config = {
        "block_shape": (int(data["block_m"]), int(data["block_n"]), int(data["block_k"])),
        "warp_shape": (int(data["warp_m"]), int(data["warp_n"]), int(data["warp_k"])),
        "use_stream_k": bool(data["use_stream_k"]),
        "use_f16_accum": bool(data["use_f16_accum"]),
        "num_stages": int(data["num_stages"]),
        "use_warp_spec": bool(data["use_warp_spec"]),
        "use_tma": bool(data["use_tma"]),
        "use_tma_b": bool(data["use_tma_b"]),
        "use_tma_c": bool(data["use_tma_c"]),
        "use_tma_bs": bool(data["use_tma_bs"]),
        "use_tma_bzp": bool(data["use_tma_bzp"]),
        "use_mbarrier": bool(data["use_mbarrier"]),
        "num_ctas_per_sm": int(data["num_ctas_per_sm"]),
        "multi_cast_size_a": int(data["multi_cast_size_a"]),
        "multi_cast_size_b": int(data["multi_cast_size_b"]),
        "use_tma_b_swizzle_64": bool(data["use_tma_b_swizzle_64"]),
        "use_m_fast_tile_order": bool(data["use_m_fast_tile_order"]),
        "use_nvfp4_prefetch_raw_wait3_issue_auto": bool(
            data["use_nvfp4_prefetch_raw_wait3_issue_auto"]
        ),
        "warp_spec_producer_regs": int(data["warp_spec_producer_regs"]),
        "nvfp4_swz64_prebcast_prmt_const_variant": bool(
            data["nvfp4_swz64_prebcast_prmt_const_variant"]
        ),
    }
    return _canonicalize_nvfp4_swizzle64_raw_config(config)


def _warn_cpp_router_unavailable_once(error: Exception) -> None:
    global _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED
    with _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_LOCK:
        if _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED:
            return
        _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED = True
    warnings.warn(
        "NVFP4 C++ router unavailable; using the Python router fallback: "
        f"{error!r}. Set TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT=1 to fail instead.",
        RuntimeWarning,
        stacklevel=3,
    )


def _get_select_nvfp4_swizzle64_raw_config_op():
    global _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE
    global _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP
    global _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_ERROR

    with _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_LOCK:
        if _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE == _OP_READY:
            return _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP
        if _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE == _OP_FAILED:
            assert _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_ERROR is not None
            raise _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_ERROR

        try:
            from tensorbridge.ops.utils import init_tensorbridge_launcher
            import torch

            init_tensorbridge_launcher()
            op = torch.ops.tensorbridge.select_nvfp4_swizzle64_raw_config
        except Exception as exc:
            _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_ERROR = exc
            _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE = _OP_FAILED
            raise

        _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP = op
        _SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_OP_STATE = _OP_READY
        return op


def select_nvfp4_swizzle64_raw_config_cpp(
    shape_m: int,
    shape_n: int,
    shape_k: int,
    *,
    use_stream_k: bool | None = None,
    num_sms: int = 132,
) -> dict[str, Any] | None:
    """Return C++-selected config, or None when disabled/unavailable."""
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if os.environ.get("TENSORBRIDGE_NVFP4_CPP_ROUTER", "1") == "0":
        return None
    try:
        op = _get_select_nvfp4_swizzle64_raw_config_op()
        packed = op(
            int(shape_m),
            int(shape_n),
            int(shape_k),
            _override_to_int(use_stream_k),
            num_sms,
        )
        return _packed_to_config([int(item) for item in packed])
    except Exception as exc:
        if os.environ.get("TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT", "0") == "1":
            raise
        _warn_cpp_router_unavailable_once(exc)
        return None
