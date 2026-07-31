from typing import TYPE_CHECKING

import torch

from tensorbridge.config import GemmType
from tensorbridge.tune.base import DeviceHeuristics
from tensorbridge.tune.sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from tensorbridge.tune.sm75 import Sm75Heuristics
from tensorbridge.tune.sm90 import Sm90Heuristics
from tensorbridge.tune.sm90_h20 import Sm90H20Heuristics
from tensorbridge.tune.sm100 import Sm100Heuristics
from tensorbridge.tune.plan_cache import TensorBridgePlanCache, TensorBridgePlanTable, TensorBridgeTuningPlan

if TYPE_CHECKING:
    from tensorbridge.layer import TensorBridgeLayerMeta

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
}


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)
    name = torch.cuda.get_device_name(device)
    if "H20" in name and "H200" not in name:
        return Sm90H20Heuristics

    return heuristics_map[sm_version]


_DEFAULT_HEURISTICS_PLAN_CACHE = TensorBridgePlanCache(max_entries=4096)


def get_heuristics_plan(
    meta: "TensorBridgeLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    use_stream_k: bool | None = None,
) -> TensorBridgeTuningPlan:
    """Return the cached shape plan for a TensorBridge layer/config pair."""
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if isinstance(meta, dict):
        from tensorbridge.layer import TensorBridgeLayerMeta
        meta = TensorBridgeLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    resolved_shape_m = shape_m if isinstance(shape_m, int) else None
    return _DEFAULT_HEURISTICS_PLAN_CACHE.get_or_create(
        meta=meta,
        shape_m=resolved_shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
        use_stream_k=use_stream_k,
        heuristics_cls=heuristics_cls,
    )


def get_heuristics_config(
    meta: "TensorBridgeLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    use_stream_k: bool | None = None,
):
    """Return a caller-owned tuning config from the cached shape plan."""
    return get_heuristics_plan(
        meta=meta,
        shape_m=shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
        use_stream_k=use_stream_k,
    ).config_copy()


def get_heuristics_config_json(
    meta: "TensorBridgeLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    use_stream_k: bool | None = None,
) -> str:
    """Return the cached tuning config JSON for launch/capture setup."""
    return get_heuristics_plan(
        meta=meta,
        shape_m=shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
        use_stream_k=use_stream_k,
    ).tuning_config_json


def warmup_heuristics_plan_cache(
    meta: "TensorBridgeLayerMeta | dict",
    shape_ms: list[int] | tuple[int, ...],
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    use_stream_k: bool | None = None,
) -> list[TensorBridgeTuningPlan]:
    """Pre-resolve common M buckets before CUDA graph capture/replay."""
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)
    if isinstance(meta, dict):
        from tensorbridge.layer import TensorBridgeLayerMeta
        meta = TensorBridgeLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    return _DEFAULT_HEURISTICS_PLAN_CACHE.warmup(
        meta=meta,
        shape_ms=shape_ms,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
        use_stream_k=use_stream_k,
        heuristics_cls=heuristics_cls,
    )


def build_heuristics_plan_table(
    meta: "TensorBridgeLayerMeta | dict",
    shape_ms: list[int] | tuple[int, ...],
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    use_stream_k: bool | None = None,
) -> TensorBridgePlanTable:
    """Build a per-layer M-bucket table for graph replay hot paths."""
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)
    if isinstance(meta, dict):
        from tensorbridge.layer import TensorBridgeLayerMeta
        meta = TensorBridgeLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    return _DEFAULT_HEURISTICS_PLAN_CACHE.build_table(
        meta=meta,
        shape_ms=shape_ms,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
        use_stream_k=use_stream_k,
        heuristics_cls=heuristics_cls,
    )


def clear_heuristics_plan_cache() -> None:
    _DEFAULT_HEURISTICS_PLAN_CACHE.clear()


def get_heuristics_plan_cache_info() -> dict[str, int]:
    return _DEFAULT_HEURISTICS_PLAN_CACHE.info()


# Preserve the functools.lru_cache surface used by existing callers.
setattr(get_heuristics_config, "cache_clear", clear_heuristics_plan_cache)
setattr(
    get_heuristics_config, "cache_info", _DEFAULT_HEURISTICS_PLAN_CACHE.cache_info
)
