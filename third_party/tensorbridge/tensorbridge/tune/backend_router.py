"""Workload-aware scheduling helpers for NVFP4 W4A8 interleave kernels.

The helpers are intentionally pure Python and tensor-free: callers provide the
runtime GEMM shape plus the already-selected BN128 tile config, and the router
returns whether the unified kernel should split only the residual tail with
StreamK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

NONSTREAMK_INTERLEAVE_BN128 = "nonstreamk_interleave_BN128"
STREAMK_INTERLEAVE_BN128 = "streamk_interleave_BN128"

# Historical V1 boundary-grid rule fitted from:
# benchmarks/results/backend_router_summary_interleave_snc_v1_fixed_327185.json
V1_DP_WAVE_FILL_STREAMK_MAX = 0.8409090909090909

# V2 model-weight SNC rule fitted from:
# benchmarks/results/backend_router_rows_model_weight_snc_v1_330003.csv
# Use rational thresholds so the runtime decision can stay integer-only.
SEVERE_UNDERFILL_DP_WAVE_FILL_MAX_NUM = 3
SEVERE_UNDERFILL_DP_WAVE_FILL_MAX_DEN = 4
MID_UNDERFILL_DP_WAVE_FILL_MAX_NUM = 112
MID_UNDERFILL_DP_WAVE_FILL_MAX_DEN = 132
STREAMK_REDUCE_ELEMS_MAX = 1_500_000
DEFAULT_NUM_SMS_H100 = 132


@dataclass(frozen=True)
class Nvfp4InterleaveFeatures:
    m_blocks: int
    n_blocks: int
    k_blocks: int
    mn_tiles: int
    mn_tiles_per_sm: float
    cta_groups: int
    dp_waves: int
    dp_wave_fill_num: int
    dp_wave_fill_den: int
    dp_wave_fill: float
    dp_underfill: float
    streamk_mn_tiles: int
    streamk_mnk_iters: int
    streamk_total_iters_per_cta: int
    streamk_wave_fill: float
    streamk_underfill: float
    estimated_slice_count: int
    estimated_reduce_tiles: int
    estimated_reduce_elems: int
    output_elems: int


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"divisor must be positive, got {b}")
    return (a + b - 1) // b


def _config_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    return int(value if value is not None else default)


def _shape3(value: Sequence[int] | Any, *, name: str) -> tuple[int, int, int]:
    try:
        x, y, z = value
    except Exception as exc:  # noqa: BLE001 - keep helper robust for config dicts.
        raise ValueError(f"{name} must be a 3-element shape, got {value!r}") from exc
    return int(x), int(y), int(z)


def _ratio_leq(num: int, den: int, limit_num: int, limit_den: int) -> bool:
    if den <= 0 or limit_den <= 0:
        return False
    return num * limit_den <= limit_num * den


def nvfp4_interleave_features(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
    *,
    num_sms: int = DEFAULT_NUM_SMS_H100,
    weight_scale_group_size: int = 16,
    input_scale_group_size: int = 0,
) -> Nvfp4InterleaveFeatures:
    """Compute scheduling features for a BN128 interleave config.

    The formulas mirror `scripts/bench_backend_router_grid.py::backend_features`.
    StreamK estimates are computed regardless of `config["use_stream_k"]`, so the
    router can decide whether to split the tail without launching or benchmarking it.
    """
    bm, bn, bk = _shape3(config["block_shape"], name="block_shape")
    mc_a = _config_int(config, "multi_cast_size_a", 1)
    mc_b = _config_int(config, "multi_cast_size_b", 1)
    num_ctas_per_sm = _config_int(config, "num_ctas_per_sm", 1)
    grid_ctas = max(1, int(num_sms)) * max(1, num_ctas_per_sm)
    cta_groups = max(1, grid_ctas // max(1, mc_a * mc_b))

    m_blocks = ceil_div(int(m), bm * max(1, mc_b))
    n_blocks = ceil_div(int(n), bn * max(1, mc_a))
    k_blocks = ceil_div(int(k), bk)
    mn_tiles = m_blocks * n_blocks
    dp_waves = ceil_div(mn_tiles, cta_groups) if mn_tiles else 0
    dp_wave_fill_den = dp_waves * cta_groups if dp_waves else 0
    dp_wave_fill = mn_tiles / dp_wave_fill_den if dp_wave_fill_den else 0.0

    streamk_mn_tiles = mn_tiles
    if mn_tiles > cta_groups:
        streamk_mn_tiles = mn_tiles % cta_groups
        if streamk_mn_tiles and streamk_mn_tiles * 10 <= cta_groups:
            streamk_mn_tiles += cta_groups
    streamk_mnk_iters = streamk_mn_tiles * k_blocks
    streamk_total_iters_per_cta = (
        ceil_div(streamk_mnk_iters, cta_groups) if streamk_mnk_iters else 0
    )
    max_group = max(input_scale_group_size or 1, weight_scale_group_size or 1)
    blocks_per_group = max_group // bk
    if blocks_per_group > 1 and streamk_total_iters_per_cta:
        streamk_total_iters_per_cta = blocks_per_group * ceil_div(
            streamk_total_iters_per_cta, blocks_per_group
        )
    streamk_waves = ceil_div(streamk_mnk_iters, cta_groups) if streamk_mnk_iters else 0
    streamk_wave_fill = (
        streamk_mnk_iters / (streamk_waves * cta_groups) if streamk_waves else 0.0
    )
    estimated_slice_count = 1
    if streamk_total_iters_per_cta:
        estimated_slice_count = max(1, ceil_div(k_blocks, streamk_total_iters_per_cta))
    estimated_reduce_tiles = streamk_mn_tiles * max(0, estimated_slice_count - 1)
    estimated_reduce_elems = estimated_reduce_tiles * bm * bn

    return Nvfp4InterleaveFeatures(
        m_blocks=m_blocks,
        n_blocks=n_blocks,
        k_blocks=k_blocks,
        mn_tiles=mn_tiles,
        mn_tiles_per_sm=mn_tiles / cta_groups,
        cta_groups=cta_groups,
        dp_waves=dp_waves,
        dp_wave_fill_num=mn_tiles,
        dp_wave_fill_den=dp_wave_fill_den,
        dp_wave_fill=dp_wave_fill,
        dp_underfill=1.0 - dp_wave_fill,
        streamk_mn_tiles=streamk_mn_tiles,
        streamk_mnk_iters=streamk_mnk_iters,
        streamk_total_iters_per_cta=streamk_total_iters_per_cta,
        streamk_wave_fill=streamk_wave_fill,
        streamk_underfill=1.0 - streamk_wave_fill,
        estimated_slice_count=estimated_slice_count,
        estimated_reduce_tiles=estimated_reduce_tiles,
        estimated_reduce_elems=estimated_reduce_elems,
        output_elems=int(m) * int(n),
    )


def _v2_enable_streamk_tail(
    dp_features: Nvfp4InterleaveFeatures,
) -> bool:
    if _ratio_leq(
        dp_features.dp_wave_fill_num,
        dp_features.dp_wave_fill_den,
        SEVERE_UNDERFILL_DP_WAVE_FILL_MAX_NUM,
        SEVERE_UNDERFILL_DP_WAVE_FILL_MAX_DEN,
    ):
        return True
    if not _ratio_leq(
        dp_features.dp_wave_fill_num,
        dp_features.dp_wave_fill_den,
        MID_UNDERFILL_DP_WAVE_FILL_MAX_NUM,
        MID_UNDERFILL_DP_WAVE_FILL_MAX_DEN,
    ):
        return False
    return dp_features.estimated_reduce_elems <= STREAMK_REDUCE_ELEMS_MAX


def _is_measured_m256_token256_island(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
) -> bool:
    """Return true for the SNC-on M=256 wide-N island where BM256 beats StreamK."""
    bm, bn, bk = _shape3(config["block_shape"], name="block_shape")
    low_k_exact_wins = {
        (32768, 512),
        (24576, 1536),
    }
    return (
        int(m) == 256
        and (
            (int(n) > 8192 and int(k) >= 4096)
            or (int(n), int(k)) in low_k_exact_wins
        )
        and (bm, bn, bk) == (256, 128, 128)
    )


def use_stream_k_tail_for_nvfp4_interleave(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
    *,
    num_sms: int = DEFAULT_NUM_SMS_H100,
) -> bool:
    """Return true when the unified NVFP4 interleave kernel should split the tail.

    The rule is deliberately shallow and shape-only:

    0. no residual suffix means tail-on is a no-op, so keep the DP-only kernel;
    1. severe DP wave underfill enables StreamK tail;
    2. medium underfill enables StreamK tail only when estimated reduction
       traffic is small;
    3. otherwise keep all MN tiles data-parallel to avoid reduce/lock overhead.
    """
    # Focused SNC-on ABBA jobs 330176/330177 showed that the CUTLASS-like
    # token256 data-parallel tile wins over both BM128 DP and StreamK for this
    # M=256 wide-N model-weight island. Keep this as a shape-and-tile guard so
    # the generic underfill rule still governs all other regimes.
    if _is_measured_m256_token256_island(m, n, k, config):
        return False

    dp_features = nvfp4_interleave_features(m, n, k, config, num_sms=num_sms)
    if dp_features.streamk_mn_tiles == 0:
        return False
    if _v2_enable_streamk_tail(dp_features):
        return True
    return False


def has_stream_k_tail_work_for_nvfp4_interleave(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
    *,
    num_sms: int = DEFAULT_NUM_SMS_H100,
) -> bool:
    """Return true when a StreamK-tail launch would have a non-empty tail suffix."""
    features = nvfp4_interleave_features(m, n, k, config, num_sms=num_sms)
    return features.streamk_mn_tiles != 0


def choose_nvfp4_interleave_backend(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
    *,
    streamk_config: Mapping[str, Any] | None = None,
    num_sms: int = DEFAULT_NUM_SMS_H100,
) -> str:
    """Legacy wrapper mapping the unified tail decision to old backend names."""
    del streamk_config
    if use_stream_k_tail_for_nvfp4_interleave(m, n, k, config, num_sms=num_sms):
        return STREAMK_INTERLEAVE_BN128
    return NONSTREAMK_INTERLEAVE_BN128


def use_stream_k_for_nvfp4_interleave(
    m: int,
    n: int,
    k: int,
    config: Mapping[str, Any],
    *,
    streamk_config: Mapping[str, Any] | None = None,
    num_sms: int = DEFAULT_NUM_SMS_H100,
) -> bool:
    """Return the `use_stream_k` tail flag implied by the V2 interleave router."""
    del streamk_config
    return use_stream_k_tail_for_nvfp4_interleave(m, n, k, config, num_sms=num_sms)
