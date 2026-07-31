import math
import os
import threading
import warnings
from typing import TYPE_CHECKING

import numpy as np

from tensorbridge import dtypes
from tensorbridge.config import GemmType
from tensorbridge.tune.backend_router import (
    has_stream_k_tail_work_for_nvfp4_interleave,
    use_stream_k_tail_for_nvfp4_interleave,
)
from tensorbridge.tune.base import DeviceHeuristics

if TYPE_CHECKING:
    from tensorbridge.layer import TensorBridgeLayerMeta

_CPP_ROUTER_UNINITIALIZED = "uninitialized"
_CPP_ROUTER_READY = "ready"
_CPP_ROUTER_FAILED = "failed"
_CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE = _CPP_ROUTER_UNINITIALIZED
_CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG = None
_CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_ERROR: Exception | None = None
_CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_LOCK = threading.Lock()
_CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED = False


def _cpp_router_strict() -> bool:
    return os.environ.get("TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT", "0") == "1"


def _warn_cpp_router_wrapper_unavailable_once(error: Exception) -> None:
    global _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED
    with _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_LOCK:
        if _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED:
            return
        _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_WARNING_EMITTED = True
    warnings.warn(
        "NVFP4 C++ router wrapper unavailable; using the Python router fallback: "
        f"{error!r}. Set TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT=1 to fail instead.",
        RuntimeWarning,
        stacklevel=3,
    )


def _get_cpp_select_nvfp4_swizzle64_raw_config():
    global _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE
    global _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG
    global _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_ERROR

    with _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_LOCK:
        if _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE == _CPP_ROUTER_READY:
            return _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG
        if _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE == _CPP_ROUTER_FAILED:
            assert _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_ERROR is not None
            raise _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_ERROR

        try:
            from tensorbridge.tune.cpp_router import select_nvfp4_swizzle64_raw_config_cpp
        except Exception as exc:
            _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_ERROR = exc
            _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE = _CPP_ROUTER_FAILED
            raise

        _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG = select_nvfp4_swizzle64_raw_config_cpp
        _CPP_SELECT_NVFP4_SWIZZLE64_RAW_CONFIG_STATE = _CPP_ROUTER_READY
        return select_nvfp4_swizzle64_raw_config_cpp


def _canonicalize_nvfp4_swizzle64_raw_config(config: dict) -> dict:
    """Keep raw dicts and cache JSON identical across C++ and Python routing."""
    canonical = {
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
    optional_names = {
        "use_m_fast_tile_order",
        "use_nvfp4_prefetch_raw_wait3_issue_auto",
        "warp_spec_producer_regs",
        "nvfp4_swz64_prebcast_prmt_const_variant",
    }
    if bool(config.get("use_m_fast_tile_order", False)):
        canonical["use_m_fast_tile_order"] = True
    if bool(config.get("use_nvfp4_prefetch_raw_wait3_issue_auto", False)):
        canonical["use_nvfp4_prefetch_raw_wait3_issue_auto"] = True
    if int(config.get("warp_spec_producer_regs", 0)):
        canonical["warp_spec_producer_regs"] = int(config["warp_spec_producer_regs"])
    if bool(config.get("nvfp4_swz64_prebcast_prmt_const_variant", False)):
        canonical["nvfp4_swz64_prebcast_prmt_const_variant"] = True
    for name in sorted(set(config) - set(canonical) - optional_names):
        canonical[name] = config[name]
    return canonical


class Sm90Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @staticmethod
    def _use_nvfp4_m_fast_tile_order(
        shape_m: int,
        shape_n: int,
        shape_k: int,
        block_shape: tuple[int, int, int],
    ) -> bool:
        # Full-grid tile-order sweep 322359: m-fast is only enabled in the
        # stable token256 island; all other raw-deint shapes default to n-fast.
        return (
            block_shape == (256, 128, 128)
            and 512 <= shape_m <= 2048
            and shape_n >= 14336
            and shape_k >= 3072
        )

    @classmethod
    def get_config1(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
    ):
        is_nvfp4_large_mlp_shape = (
            (meta.shape_n >= 8192 and meta.shape_k >= 3584)
            or (meta.shape_n >= 3584 and meta.shape_k >= 8192)
        )
        use_nvfp4_target_path = (
            meta.use_fused_e4m3_scale
            and meta.weight_scale_group_size == 16
            and gemm_type == GemmType.DENSE
            and not use_f16_accum
            and not use_batch_invariant
            # Legacy non-interleave callers still use use_stream_k as an input to
            # tile selection. The swizzle64 interleave wrapper below always
            # resolves the unified DP/full-K tile first, then applies the tail flag.
            and use_stream_k is not True
            and shape_m >= 1024
            and is_nvfp4_large_mlp_shape
            and meta.shape_n % 128 == 0
            and meta.shape_k % 128 == 0
        )
        if use_nvfp4_target_path:
            config = {
                "block_shape": (256, 128, 128),
                "warp_shape": (256, 16, 128),
                "use_stream_k": False,
                "use_f16_accum": False,
                "num_stages": 4,
                "use_warp_spec": True,
                "use_tma": True,
                "use_tma_b": False,
                "use_tma_c": True,
                "use_tma_bs": False,
                "use_mbarrier": True,
                "num_ctas_per_sm": 1,
                "multi_cast_size_a": 1,
                "multi_cast_size_b": 1,
            }
            if cls._use_nvfp4_m_fast_tile_order(
                    shape_m, meta.shape_n, meta.shape_k, (256, 128, 128)):
                config["use_m_fast_tile_order"] = True
            if (
                    shape_m == 2048
                    and meta.shape_n == 18944
                    and meta.shape_k == 3584
                    and meta.weight_scale_group_size == 16):
                config["use_nvfp4_prefetch_raw_wait3_issue_auto"] = True
            return config

        if use_f16_accum:
            max_block_m = 256
        else:
            max_block_m = 176
            # NVFP4 fused-E4M3 only has a stable bm184 win in dense 8k+ wide-N/K
            # very-large-M prefill. M=1024/2048/4096 should keep bm176; bm192+
            # hits a scheduler/local-memory cliff in the current layout.
            if (meta.use_fused_e4m3_scale
                    and gemm_type == GemmType.DENSE
                    and not use_batch_invariant
                    and shape_m >= 8192
                    and meta.shape_n >= 8192
                    and meta.shape_k >= 8192):
                max_block_m = 184

        num_blocks_list = cls.calc_num_block_list(meta, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        warp_shape_n = 32
        warp_shape_k = 1024 // meta.a_dtype.num_bits

        if meta.shape_n <= 4096 and not use_batch_invariant and block_shape_m <= 64:
            block_shape_n = 128
            block_shape_k = warp_shape_k * 2
            if block_shape_m <= 32:
                block_shape_k = block_shape_k * 2
            if block_shape_k > 256:
                block_shape_k = block_shape_k // 2
                warp_shape_k = warp_shape_k // 2
        else:
            block_shape_n = 256
            block_shape_k = warp_shape_k
            if block_shape_m <= 32 and meta.b_dtype.num_bits <= 6:
                block_shape_k = block_shape_k * 2
            elif block_shape_m <= 32:
                warp_shape_k = warp_shape_k // 2

        # NVFP4 fused: switch to (128, 128) tile with warp_shape_n=16 when the
        # default tile gives < 1 wave on 132 H100 SMs. Two regimes: wide-N
        # decode (shape_n > 8192, M <= 128) and mid-N small batch (4096 <=
        # shape_n < 8192, M <= 512). 8k-square and M >= 1024 already fill
        # at default and regress under the smaller tile. warp_shape_n MUST
        # halve with block_shape_n to keep math warp count constant.
        num_stages = 4
        if meta.use_fused_e4m3_scale:
            wide_n_decode = (meta.shape_n > 8192) and (shape_m <= 128)
            mid_n_smallbatch = (
                4096 <= meta.shape_n < 8192
            ) and (shape_m <= 512)
            if wide_n_decode or mid_n_smallbatch:
                block_shape_n = 128
                block_shape_k = warp_shape_k
                warp_shape_n = max(16, block_shape_n // 8)
                num_stages = 6 if wide_n_decode else 4

        # Legacy NVFP4 stream-K gate for non-interleave paths. The swizzle64 path
        # below rewrites use_stream_k into a StreamK-tail flag after tile selection.
        # Current evidence supports disabling StreamK on wide-N large-M prefill,
        # where K-split atomic-reduce is pure overhead.
        # Keep narrower-N shapes on StreamK until they have separate coverage.
        if use_stream_k is None:
            use_stream_k = not use_batch_invariant
            if (meta.use_fused_e4m3_scale and shape_m >= 2048
                    and meta.shape_n >= 8192
                    and meta.shape_k >= 4096
                    and not use_batch_invariant):
                use_stream_k = False

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": use_stream_k,
            "use_f16_accum": use_f16_accum,
            "num_stages": num_stages,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_tma_b"] = False
            config["use_mbarrier"] = True

            # multi_cast_size_a=2 amortizes weight broadcast at small-M decode,
            # but at M >= 1024 it halves the effective CTA count and worsens
            # wave fill for NVFP4 fused. Skip for large-M NVFP4.
            mc_disable_nvfp4_largeM = (
                meta.use_fused_e4m3_scale and shape_m >= 1024
            )
            if (meta.shape_n % (block_shape_n * 2) == 0
                    and shape_m / block_shape_m >= 4
                    and not mc_disable_nvfp4_largeM):
                if gemm_type == GemmType.DENSE:
                    config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def get_config2(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
    ):
        if use_f16_accum:
            max_block_m = 256
        elif meta.input_scale_group_size > 0:
            max_block_m = 160
        elif meta.weight_scale_group_size < 128:
            max_block_m = 192
        else:
            max_block_m = 200

        num_blocks_list = cls.calc_num_block_list(meta, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8

        block_shape_k = 256 if block_shape_m <= 32 else 128

        config = {
            "block_shape": (block_shape_m, 128, block_shape_k),
            "warp_shape": (block_shape_m, 16, 128),
            "use_stream_k": (use_stream_k if use_stream_k is not None
                             else not use_batch_invariant),
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_tma_b"] = False
            config["use_mbarrier"] = True

            if shape_m / block_shape_m >= 4 and gemm_type == GemmType.DENSE:
                config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def calc_num_block_list(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        max_block_m: int,
    ):
        num_blocks_list = []
        if not meta.num_experts:
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, meta.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        for i in range(max_block_m // 8):
            num_blocks = num_blocks_list[i]
            block_m = i * 8 + 8
            if meta.a_dtype == dtypes.int8 and num_blocks % 16 == 8 and block_m > 32:
                num_blocks_list[i] = 10000

        return num_blocks_list

    @classmethod
    def _apply_nvfp4_swizzle64_raw_config(cls, config: dict) -> dict:
        """Enable the phase-1 interleave path on the safe BN128 tile family."""
        config = dict(config)
        block_m = int(config["block_shape"][0])
        config.update({
            "block_shape": (block_m, 128, 128),
            "warp_shape": (block_m, 16, 128),
            "use_warp_spec": True,
            "use_tma": True,
            "use_tma_b": True,
            "use_tma_b_swizzle_64": True,
            "use_tma_bs": False,
            "use_tma_bzp": False,
            "use_mbarrier": True,
            "num_ctas_per_sm": 1,
            "multi_cast_size_b": 1,
        })
        return config

    @classmethod
    def _sanitize_nvfp4_swizzle64_raw_config(
        cls,
        config: dict,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
    ) -> dict:
        """Compatibility hook for the phase-1 swizzle64 selector.

        BM64 correctness is handled in-kernel by the TMA-C async-proxy fence.
        The tile-M overrides below are measured CUTLASS token256 islands for
        non-StreamK M=256 wide-N shapes.
        """
        config = dict(config)
        if (
                not bool(config.get("use_stream_k", False))
                and shape_m == 256
                and (
                    (meta.shape_n > 8192 and meta.shape_k >= 4096)
                    or (meta.shape_n, meta.shape_k) in {
                        (32768, 512),
                        (24576, 1536),
                    }
                )):
            # Match CUTLASS's channel128 x token256 schedule for M<=256,
            # wide-N W4A8 shapes. BM128 duplicates B/scale traffic over two
            # token tiles; BM256 keeps one CTA per N tile. The low-K rows are
            # exact measured wins from the Iter-13 SNC-on residual-tail probe.
            config["block_shape"] = (256, 128, 128)
            config["warp_shape"] = (256, 16, 128)
        if (
                not bool(config.get("use_stream_k", False))
                and shape_m == 128
                and (meta.shape_n, meta.shape_k) in {
                    (57344, 8192),
                    (28672, 8192),
                    (13824, 5120),
                    (15360, 5120),
                }):
            # Iter-14 stage-depth probe: these exact non-StreamK BM128 rows
            # were bitwise-correct and faster with five TMA stages. Nearby
            # M128 rows either regressed or failed strict correctness, so keep
            # this as an exact rescue rather than a broad wide-N rule.
            config["num_stages"] = 5
        return config

    @classmethod
    def _post_route_nvfp4_swizzle64_raw_config(
        cls,
        config: dict,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
    ) -> dict:
        """Apply measured tile upgrades after the StreamK-tail decision."""
        config = dict(config)
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (176, 128, 128)
                and (
                    (shape_m == 512 and meta.shape_n >= 13824)
                    or shape_m >= 2048
                )):
            # Focused SNC-on probes 330206/330283: among rows where the router
            # already disabled the StreamK tail, BM256 was bitwise-identical and faster
            # for M=512 wide-N and for all measured M>=2048 BM176 rows. Do this
            # after routing so StreamK-favored rows are not pulled into a
            # numerically different data-parallel path.
            config["block_shape"] = (256, 128, 128)
            config["warp_shape"] = (256, 16, 128)
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and shape_m in (4096, 8192)
                and meta.shape_n == 32768
                and meta.shape_k == 512):
            # Iter-29 accepted M8192 and Iter157 re-confirmed M4096 under the
            # current swizzle64/SNC/preint path. M<=2048 still regresses, so the
            # K512 B/L2 reuse rescue stays exact.
            config["use_m_fast_tile_order"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and shape_m == 1024
                and meta.shape_n == 4096
                and meta.shape_k == 7168):
            # Iter-47/50 N4096 sweep: only this exact BM256 row repeatedly won
            # with m-fast. Neighboring K/M controls were noise or regressions.
            config["use_m_fast_tile_order"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and (shape_m, meta.shape_n, meta.shape_k) in {
                    (1024, 8192, 28672),
                    (2048, 8192, 28672),
                    (4096, 24576, 4096),
                }):
            # Iter-55/56 and Iter85/86: m-fast improves B/L2 reuse only on
            # these exact BM256 islands. K512, low-K, and most M8192 controls
            # regress, so do not broaden by N/K alone.
            config["use_m_fast_tile_order"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and getattr(meta, "use_nvfp4_swizzle64_raw", False)
                and getattr(meta, "use_nvfp4_snc", False)
                and meta.weight_scale_group_size == 16
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and tuple(config.get("warp_shape", ())) == (256, 16, 128)
                and shape_m == 512
                and 4096 <= meta.shape_k <= 18432
                and (meta.shape_n >= 8192 or meta.shape_k >= 8192)):
            # Iter156/157: for M512 non-StreamK BM256 rows, m-fast shortens
            # B-tile reuse distance. NCU on (512,27648,5120) showed DRAM read
            # nearly halves with unchanged shared-load/TMA instruction counts.
            # Keep low-K controls and the N8192/K28672 high-K control n-fast.
            config["use_m_fast_tile_order"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and getattr(meta, "use_nvfp4_swizzle64_raw", False)
                and getattr(meta, "use_nvfp4_snc", False)
                and meta.weight_scale_group_size == 16
                and tuple(config.get("block_shape", ())) == (128, 128, 128)
                and tuple(config.get("warp_shape", ())) == (128, 16, 128)
                and shape_m == 256
                and meta.shape_n in (7168, 8192)
                and meta.shape_k >= 5120):
            # Iter171: M256/BM128 has two M tiles per B tile. m-fast shortens
            # the B/L2 reuse distance like the accepted M512/BM256 route.
            # NCU on (256,8192,28672) showed lower L2 read sectors and
            # scoreboard with unchanged shared-load/TMA instruction counts.
            config["use_m_fast_tile_order"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and shape_m in (16, 32, 64, 128)
                and meta.shape_n == 32768
                and meta.shape_k == 512):
            # Iter-32 K512 small-M sweep: scalar C stores beat the TMA-C/STSM
            # path for these exact rows, while M=256 regressed.
            config["use_tma_c"] = False
        if (
                not bool(config.get("use_stream_k", False))
                and shape_m == 128
                and (meta.shape_n, meta.shape_k) in {
                    (57344, 8192),
                    (28672, 8192),
                    (13824, 5120),
                    (15360, 5120),
                }):
            # Iter-44/45: these exact M128 stage5 rows also win with scalar C
            # stores. Nearby rows either regress or fail strict correctness.
            config["use_tma_c"] = False
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and (shape_m, meta.shape_n, meta.shape_k) in {
                    (512, 8192, 28672),
                    (2048, 8192, 28672),
                    (2048, 4096, 12288),
                }):
            # Iter-52/53: these BM256 residual rows consistently win with
            # scalar C stores; neighboring BM256 controls regress sharply.
            config["use_tma_c"] = False
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and (shape_m, meta.shape_n, meta.shape_k) in {
                    (4096, 8192, 5120),
                    (1024, 24576, 4096),
                    (2048, 22016, 4096),
                    (2048, 8192, 5120),
                    (4096, 8192, 8192),
                    (2048, 12288, 4096),
                    (1024, 4096, 7168),
                    (1024, 8192, 28672),
                    (1024, 15360, 5120),
                    (1024, 17408, 5120),
                    (1024, 18432, 7168),
                    (1024, 22016, 4096),
                    (1024, 25600, 5120),
                    (1024, 27648, 5120),
                    (1024, 28672, 8192),
                    (1024, 34816, 5120),
                    (1024, 36864, 5120),
                    (1024, 36864, 7168),
                    (1024, 51200, 5120),
                    (2048, 7168, 18432),
                    (2048, 25600, 5120),
                    (2048, 28672, 8192),
                    (2048, 34816, 5120),
                    (2048, 51200, 5120),
                }):
            # Iter-72/73, Iter85-87, and Iter159: giving the producer warp group 56
            # registers reduces scoreboard pressure only on these exact BM256
            # residual rows. K512, some high-K N8192, B-multicast, and low-K
            # controls regress, so keep this as an exact workload route.
            config["warp_spec_producer_regs"] = 56
        if (
                not bool(config.get("use_stream_k", False))
                and getattr(meta, "use_nvfp4_swizzle64_raw", False)
                and getattr(meta, "use_nvfp4_snc", False)
                and meta.weight_scale_group_size == 16
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and tuple(config.get("warp_shape", ())) == (256, 16, 128)
                and meta.shape_k > 512):
            # Iter153/154: aggregate the scale-prebroadcast + PRMT SNC-mask +
            # constant fused-scaleD composition as one graph-safe compile-time
            # variant. K512 is excluded because profiling showed the extra
            # instructions are not amortized and tensor activity falls.
            config["nvfp4_swz64_prebcast_prmt_const_variant"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and getattr(meta, "use_nvfp4_swizzle64_raw", False)
                and getattr(meta, "use_nvfp4_snc", False)
                and meta.weight_scale_group_size == 16
                and tuple(config.get("block_shape", ())) == (128, 128, 128)
                and tuple(config.get("warp_shape", ())) == (128, 16, 128)
                and shape_m == 256
                and meta.shape_n in (7168, 8192)
                and meta.shape_k >= 5120):
            # Iter154/166: the aggregate scale-prebroadcast + PRMT-mask +
            # constant-scaleD dependency route helps the non-StreamK M256
            # BM128 N7168/N8192 rows with K>=5120. Keep StreamK/small-N and
            # K2048 controls off because forcing the bit outside this routed
            # non-StreamK island can fail strict A/B correctness or is below
            # the residual target.
            config["nvfp4_swz64_prebcast_prmt_const_variant"] = True
        if (
                not bool(config.get("use_stream_k", False))
                and tuple(config.get("block_shape", ())) == (256, 128, 128)
                and (shape_m, meta.shape_n, meta.shape_k) in {
                    (4096, 36864, 7168),
                    (8192, 36864, 7168),
                    (4096, 51200, 5120),
                    (8192, 51200, 5120),
                    (4096, 57344, 8192),
                    (8192, 57344, 8192),
                }):
            # Iter-58/59: B multicast is a DRAM/L2 reuse win only on these
            # large-M very-wide-N BM256 rows. K512, N8192, and N24576 controls
            # regress, so keep the cluster route exact.
            config["multi_cast_size_b"] = 2
        return config

    @classmethod
    def _bm256_dp_wave_fill_num_den(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        *,
        num_sms: int = 132,
    ) -> tuple[int, int]:
        bm256_mn_tiles = math.ceil(shape_m / 256) * math.ceil(meta.shape_n / 128)
        waves = math.ceil(bm256_mn_tiles / num_sms) if bm256_mn_tiles else 0
        return bm256_mn_tiles, waves * num_sms

    @classmethod
    def _bm256_rescues_bm176_unified_tail(
        cls,
        config: dict,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
    ) -> bool:
        """Return true when a measured BM176 row should use the BM256 unified tile."""
        if (
                bool(config.get("use_stream_k", False))
                or tuple(config.get("block_shape", ())) != (176, 128, 128)
                or shape_m < 512):
            return False
        fill_num, fill_den = cls._bm256_dp_wave_fill_num_den(meta, shape_m)
        # SNC-on probes 330320/330348: BM256 wins for measured BM176 rows when
        # its DP wave fill is at least 96/132. Below that boundary keep BM176 and
        # let the unified scheduler decide whether to split the residual tail.
        return fill_den > 0 and fill_num * 132 >= 96 * fill_den

    @classmethod
    def _promote_to_bm256_nonstreamk(cls, config: dict) -> dict:
        config = dict(config)
        config["block_shape"] = (256, 128, 128)
        config["warp_shape"] = (256, 16, 128)
        config["use_stream_k"] = False
        return config

    @classmethod
    def _try_nvfp4_swizzle64_raw_cpp_config(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
        use_stream_k: bool | None,
    ) -> dict | None:
        if (
                use_f16_accum
                or use_batch_invariant
                or gemm_type != GemmType.DENSE
                or getattr(meta, "num_experts", 0)
                or not getattr(meta, "use_nvfp4_snc", False)
                or not getattr(meta, "use_fused_e4m3_scale", False)
                or getattr(meta, "use_nvfp4_raw_s2r_deint", False)
                or meta.input_scale_group_size != 0
                or meta.weight_scale_group_size != 16
                or meta.shape_n % 128 != 0
                or meta.shape_k % 128 != 0):
            return None
        if os.environ.get("TENSORBRIDGE_NVFP4_CPP_ROUTER", "1") == "0":
            return None
        try:
            select_nvfp4_swizzle64_raw_config_cpp = _get_cpp_select_nvfp4_swizzle64_raw_config()
            return select_nvfp4_swizzle64_raw_config_cpp(
                shape_m,
                meta.shape_n,
                meta.shape_k,
                use_stream_k=use_stream_k,
            )
        except Exception as exc:  # noqa: BLE001 - non-strict mode keeps Python fallback.
            if _cpp_router_strict():
                raise
            _warn_cpp_router_wrapper_unavailable_once(exc)
            return None

    @classmethod
    def get_config(
        cls,
        meta: "TensorBridgeLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_stream_k: bool | None = None,
    ):
        if getattr(meta, "use_nvfp4_raw_s2r_deint", False):
            # CUTLASS channel x token maps to TensorBridge (BlockShape.M=token,
            # BlockShape.N=channel).  The raw-deint data-parallel path currently
            # keeps channel=128 and selects token tiles {256,128,64,32,16}.
            #
            # Mirror CUTLASS's narrow-token threshold tree, then use a CTA-count
            # comparison for the token64/token128 wave-fill islands.  Full-grid scans
            # showed that the old score heuristic over-selected token128 just past one
            # H100 wave and missed the M=128/M=256 token64 islands.
            num_sms = 132
            channel_tiles = math.ceil(meta.shape_n / 128)
            token64_ctas = math.ceil(shape_m / 64) * channel_tiles
            token128_ctas = math.ceil(shape_m / 128) * channel_tiles
            data_parallel = not bool(use_stream_k)
            use_16_token_tile = data_parallel and shape_m <= 16
            use_32_token_tile = data_parallel and shape_m <= 32
            use_64_token_tile = (
                data_parallel
                and (
                    shape_m <= 64
                    or (shape_m <= 256 and token64_ctas <= num_sms)
                )
            )
            use_128_token_tile = (
                data_parallel
                and (
                    shape_m <= 128
                    or token128_ctas <= num_sms
                )
            )
            if use_16_token_tile:
                block_shape = (16, 128, 128)
                warp_shape = (16, 16, 128)
            elif use_32_token_tile:
                block_shape = (32, 128, 128)
                warp_shape = (32, 16, 128)
            elif use_64_token_tile:
                block_shape = (64, 128, 128)
                warp_shape = (64, 16, 128)
            elif use_128_token_tile:
                block_shape = (128, 128, 128)
                warp_shape = (128, 16, 128)
            else:
                block_shape = (256, 128, 128)
                warp_shape = (256, 16, 128)
            config = {
                "block_shape": block_shape,
                "warp_shape": warp_shape,
                "use_stream_k": bool(use_stream_k),
                "use_f16_accum": False,
                "num_stages": 4,
                "use_warp_spec": True,
                "use_tma": True,
                "use_tma_b": False,
                "use_tma_c": True,
                "use_tma_bs": False,
                "use_mbarrier": True,
                "num_ctas_per_sm": 1,
                "multi_cast_size_a": 1,
                "multi_cast_size_b": 1,
            }
            if cls._use_nvfp4_m_fast_tile_order(
                    shape_m, meta.shape_n, meta.shape_k, block_shape):
                config["use_m_fast_tile_order"] = True
            if (
                    shape_m == 2048
                    and meta.shape_n == 18944
                    and meta.shape_k == 3584
                    and meta.weight_scale_group_size == 16):
                config["use_nvfp4_prefetch_raw_wait3_issue_auto"] = True
            return config

        if meta.a_dtype.num_bits == 16:
            func = cls.get_config1
        elif meta.input_scale_group_size == 0 and meta.weight_scale_group_size == 0:
            func = cls.get_config1
        elif meta.use_fused_e8m0_scale or meta.use_fused_e4m3_scale:
            func = cls.get_config1
        else:
            func = cls.get_config2

        if getattr(meta, "use_nvfp4_swizzle64_raw", False):
            # Interleave uses one BN128 kernel family.  Resolve the DP/full-K
            # tile first, then let use_stream_k mean "enable StreamK tail" rather
            # than selecting a separate StreamK tile backend.
            cpp_config = cls._try_nvfp4_swizzle64_raw_cpp_config(
                meta,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
                use_stream_k,
            )
            if cpp_config is not None:
                return _canonicalize_nvfp4_swizzle64_raw_config(cpp_config)
            enable_tail_override = use_stream_k
            config = func(meta, shape_m, use_f16_accum, use_batch_invariant, gemm_type, False)
            config = cls._apply_nvfp4_swizzle64_raw_config(config)
            config = cls._sanitize_nvfp4_swizzle64_raw_config(config, meta, shape_m)
            force_tail_off = cls._bm256_rescues_bm176_unified_tail(config, meta, shape_m)
            if force_tail_off:
                config = cls._promote_to_bm256_nonstreamk(config)
            if enable_tail_override is None:
                config["use_stream_k"] = (
                    False if force_tail_off else
                    use_stream_k_tail_for_nvfp4_interleave(
                        shape_m, meta.shape_n, meta.shape_k, config)
                )
            else:
                config["use_stream_k"] = (
                    bool(enable_tail_override) and
                    has_stream_k_tail_work_for_nvfp4_interleave(
                        shape_m, meta.shape_n, meta.shape_k, config)
                )
            config = cls._post_route_nvfp4_swizzle64_raw_config(config, meta, shape_m)
            return _canonicalize_nvfp4_swizzle64_raw_config(config)

        config = func(meta, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_stream_k)
        return config
