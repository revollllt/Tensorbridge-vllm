"""Low-overhead cached tuning-plan lookup for runtime GEMM shapes.

The cache stores the resolved Python heuristic result as an immutable plan and
returns fresh deep copies to callers.  This keeps the hot path to a stable
shape-key lookup while avoiding accidental mutation of cached routing decisions.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
import json
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from tensorbridge.config import GemmType

if TYPE_CHECKING:
    from tensorbridge.layer import TensorBridgeLayerMeta
    from tensorbridge.tune.base import DeviceHeuristics


class _FrozenList(tuple):
    """Tuple-backed marker that thaws back to a list."""


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_value(nested) for nested in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(nested) for nested in value)
    return value


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(nested) for key, nested in value.items()}
    if isinstance(value, _FrozenList):
        return [_copy_value(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(nested) for nested in value)
    return value


def _safe_num_sms(heuristics_cls: type["DeviceHeuristics"]) -> int | None:
    """Return the hardware-dependent SM count without requiring CUDA."""
    getter = getattr(heuristics_cls, "get_num_sms", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def _is_interval_plan_item(item: Any) -> bool:
    return (
        isinstance(item, Sequence)
        and not isinstance(item, (str, bytes, bytearray))
        and len(item) == 3
        and isinstance(item[2], Mapping)
    )


def _freeze_plan_item(item: Any) -> Mapping[str, Any] | tuple[Any, Any, Mapping[str, Any]]:
    if isinstance(item, Mapping):
        return _freeze_value(item)
    if _is_interval_plan_item(item):
        return (_freeze_value(item[0]), _freeze_value(item[1]), _freeze_value(item[2]))
    raise TypeError(f"unsupported tuning-plan item: {item!r}")


def _copy_plan_item(item: Any) -> dict[str, Any] | list[Any]:
    if isinstance(item, Mapping):
        return _copy_value(item)
    if _is_interval_plan_item(item):
        return [_copy_value(item[0]), _copy_value(item[1]), _copy_value(item[2])]
    raise TypeError(f"unsupported cached tuning-plan item: {item!r}")


def _freeze_tuning_config(
    config: Mapping[str, Any] | Sequence[Any],
) -> Mapping[str, Any] | tuple[Any, ...]:
    if isinstance(config, Mapping):
        return _freeze_value(config)
    if isinstance(config, Sequence) and not isinstance(
        config, (str, bytes, bytearray)
    ):
        return tuple(_freeze_plan_item(item) for item in config)
    raise TypeError(f"unsupported tuning config: {config!r}")


def _copy_tuning_config(
    config: Mapping[str, Any] | tuple[Any, ...],
) -> dict[str, Any] | list[Any]:
    if isinstance(config, Mapping):
        return _copy_value(config)
    return [_copy_plan_item(item) for item in config]


@dataclass(frozen=True)
class TensorBridgePlanKey:
    """Complete heuristic-input key for a resolved tuning plan."""

    heuristics_name: str
    num_sms: int | None
    meta_str: str
    shape_m: int | None
    use_f16_accum: bool
    use_batch_invariant: bool
    gemm_type: str
    use_stream_k: bool | None


@dataclass(frozen=True)
class TensorBridgeTuningPlan:
    """Resolved tuning config plus its JSON form for graph-capture setup."""

    key: TensorBridgePlanKey
    tuning_config: Mapping[str, Any] | tuple[Any, ...]
    tuning_config_json: str = ""

    def __post_init__(self) -> None:
        frozen_config = _freeze_tuning_config(self.tuning_config)
        config_copy = _copy_tuning_config(frozen_config)
        object.__setattr__(self, "tuning_config", frozen_config)
        object.__setattr__(
            self,
            "tuning_config_json",
            json.dumps(config_copy, separators=(",", ":")),
        )

    def config_copy(self) -> dict[str, Any] | list[Any]:
        """Return a caller-owned config object with value types preserved."""
        return _copy_tuning_config(self.tuning_config)


@dataclass(frozen=True)
class TensorBridgePlanTable:
    """Per-layer M-bucket table for the CUDA graph hot path."""

    plans: Mapping[int, TensorBridgeTuningPlan]
    _resolver: Callable[[int], TensorBridgeTuningPlan] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plans",
            MappingProxyType({int(shape_m): plan for shape_m, plan in self.plans.items()}),
        )

    def get_plan(self, shape_m: int) -> TensorBridgeTuningPlan:
        normalized_shape_m = int(shape_m)
        try:
            return self.plans[normalized_shape_m]
        except KeyError:
            if self._resolver is None:
                raise

            return self._resolver(normalized_shape_m)

    def get_config(self, shape_m: int) -> dict[str, Any] | list[Any]:
        return self.get_plan(shape_m).config_copy()

    def get_config_json(self, shape_m: int) -> str:
        return self.get_plan(shape_m).tuning_config_json

    def shape_ms(self) -> tuple[int, ...]:
        return tuple(self.plans.keys())


class TensorBridgePlanCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int


class TensorBridgePlanCache:
    """LRU cache for workload-aware tuning plans.

    A plan is keyed by immutable layer metadata, runtime M, compute flags, the
    manual StreamK-tail override, and the selected architecture heuristic class.
    Cache misses run the normal heuristic once; hits do not re-run the router.
    """

    def __init__(self, max_entries: int = 4096):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.max_entries = int(max_entries)
        self._plans: OrderedDict[TensorBridgePlanKey, TensorBridgeTuningPlan] = OrderedDict()
        self._lock = RLock()
        self._inflight: dict[
            TensorBridgePlanKey, tuple[int, Future[TensorBridgeTuningPlan]]
        ] = {}
        self._generation = 0
        self._hits = self._misses = 0

    @staticmethod
    def make_key(
        *,
        meta: "TensorBridgeLayerMeta | Mapping[str, Any] | Any",
        shape_m: int | None,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType | str,
        use_stream_k: bool | None,
        heuristics_cls: type["DeviceHeuristics"],
    ) -> TensorBridgePlanKey:
        if hasattr(meta, "to_str"):
            meta_str = meta.to_str()
        elif isinstance(meta, Mapping):
            meta_str = json.dumps(meta, sort_keys=True, default=str, separators=(",", ":"))
        else:
            meta_str = repr(meta)

        if isinstance(gemm_type, GemmType):
            gemm_type_str = gemm_type.value
        elif hasattr(gemm_type, "value"):
            gemm_type_str = str(gemm_type.value)
        else:
            gemm_type_str = str(gemm_type)

        heuristics_name = f"{heuristics_cls.__module__}.{heuristics_cls.__qualname__}"
        num_sms = _safe_num_sms(heuristics_cls)
        return TensorBridgePlanKey(
            heuristics_name=heuristics_name,
            num_sms=num_sms,
            meta_str=meta_str,
            shape_m=shape_m,
            use_f16_accum=bool(use_f16_accum),
            use_batch_invariant=bool(use_batch_invariant),
            gemm_type=gemm_type_str,
            use_stream_k=None if use_stream_k is None else bool(use_stream_k),
        )

    def get(self, key: TensorBridgePlanKey) -> TensorBridgeTuningPlan | None:
        with self._lock:
            plan = self._plans.get(key)
            if plan is not None:
                self._plans.move_to_end(key)
            return plan

    def put(self, plan: TensorBridgeTuningPlan) -> TensorBridgeTuningPlan:
        with self._lock:
            self._put_locked(plan)
        return plan

    def _put_locked(self, plan: TensorBridgeTuningPlan) -> None:
        self._plans[plan.key] = plan
        self._plans.move_to_end(plan.key)
        while len(self._plans) > self.max_entries:
            self._plans.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._plans.clear()
            # Old leaders still resolve their own waiters, but cannot repopulate
            # this generation or capture post-clear callers.
            self._inflight.clear()
            self._hits = self._misses = 0

    def cache_info(self) -> TensorBridgePlanCacheInfo:
        with self._lock:
            return TensorBridgePlanCacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self.max_entries,
                currsize=len(self._plans),
            )

    def info(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._plans), "max_entries": self.max_entries}

    def get_or_create(
        self,
        *,
        meta: "TensorBridgeLayerMeta",
        shape_m: int | None,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
        heuristics_cls: type["DeviceHeuristics"],
    ) -> TensorBridgeTuningPlan:
        key = self.make_key(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
            use_stream_k=use_stream_k,
            heuristics_cls=heuristics_cls,
        )
        with self._lock:
            cached = self._plans.get(key)
            if cached is not None:
                self._plans.move_to_end(key)
                self._hits += 1
                return cached

            flight = self._inflight.get(key)
            if flight is None:
                generation = self._generation
                future: Future[TensorBridgeTuningPlan] = Future()
                flight = (generation, future)
                self._inflight[key] = flight
                self._misses += 1
                is_leader = True
            else:
                generation, future = flight
                self._hits += 1
                is_leader = False

        if not is_leader:
            return future.result()

        try:
            plan = self._create_plan(
                key=key,
                meta=meta,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
                heuristics_cls=heuristics_cls,
            )
        except BaseException as error:
            with self._lock:
                if self._inflight.get(key) == flight:
                    del self._inflight[key]
                future.set_exception(error)
            raise

        with self._lock:
            if self._generation == generation:
                self._put_locked(plan)
            if self._inflight.get(key) == flight:
                del self._inflight[key]
            future.set_result(plan)
        return plan

    def _create_plan(
        self,
        *,
        meta: "TensorBridgeLayerMeta",
        key: TensorBridgePlanKey,
        shape_m: int | None,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
        heuristics_cls: type["DeviceHeuristics"],
    ) -> TensorBridgeTuningPlan:
        if shape_m is None:
            config = heuristics_cls.get_configs(
                meta=meta,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
            )
        else:
            config = heuristics_cls.get_config(
                meta=meta,
                shape_m=int(shape_m),
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
            )

        return TensorBridgeTuningPlan(key=key, tuning_config=config)

    def warmup(
        self,
        *,
        meta: "TensorBridgeLayerMeta",
        shape_ms: list[int] | tuple[int, ...],
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
        heuristics_cls: type["DeviceHeuristics"],
    ) -> list[TensorBridgeTuningPlan]:
        return [
            self.get_or_create(
                meta=meta,
                shape_m=int(shape_m),
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
                heuristics_cls=heuristics_cls,
            )
            for shape_m in shape_ms
        ]

    def build_table(
        self,
        *,
        meta: "TensorBridgeLayerMeta",
        shape_ms: list[int] | tuple[int, ...],
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
        heuristics_cls: type["DeviceHeuristics"],
    ) -> TensorBridgePlanTable:
        plans = {
            int(plan.key.shape_m): plan
            for plan in self.warmup(
                meta=meta,
                shape_ms=shape_ms,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
                heuristics_cls=heuristics_cls,
            )
            if plan.key.shape_m is not None
        }

        def resolve(shape_m: int) -> TensorBridgeTuningPlan:
            return self.get_or_create(
                meta=meta,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_stream_k=use_stream_k,
                heuristics_cls=heuristics_cls,
            )

        return TensorBridgePlanTable(plans=plans, _resolver=resolve)
