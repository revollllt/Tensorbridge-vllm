import dataclasses
import json
import os
import shlex
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import jinja2

import tensorbridge.utils.jit as jit_utils
from tensorbridge import dtypes
from tensorbridge.config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    MmaOpClass,
    MmaType,
    TuningConfig,
)
from tensorbridge.jit.runtime import KernelRuntime
from tensorbridge.tune import get_heuristics_config

CODE_TEMPLATE = jinja2.Template("""

{{layer_config_macro}}

{{compute_config_macro}}

{{tuning_config_macro}}

// Graph-safe aggregate for the Iter153/154 swizzle64 raw SNC composition.
// It expands to device-only compile-time toggles; the raw-B preinterleaved
// loader remains a separate layer/storage contract.
#if TENSORBRIDGE_NVFP4_SWZ64_PREBCAST_PRMT_CONST_VARIANT
#ifndef TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD 1
#endif
#ifndef TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR
#define TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR 1
#endif
#ifndef TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED
#define TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED 1
#endif
#endif

#ifndef TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD 0
#endif

#if {{use_swapped_large_nvfp4}}
#include <tensorbridge/kernel/tensorbridge_swapped_nvfp4.cuh>
#elif {{use_warp_spec}}
#include <tensorbridge/kernel/tensorbridge_ws.cuh>
#else
#include <tensorbridge/kernel/tensorbridge.cuh>
#endif

class MmaOpClass {
public:
{{mma_op_class}}
};

class LayerConfig {
public:
{{layer_config}}
};

class ComputeConfig {
public:
{{compute_config}}
};

class TuningConfig {
public:
{{tuning_config}}
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<{{block_shape[0]}}, {{block_shape[1]}}, {{block_shape[2]}}>,
    Shape<{{warp_shape[0]}}, {{warp_shape[1]}}, {{warp_shape[2]}}>,
    {{a_dtype}},
    {{b_dtype}},
    {{bs_dtype}},
    LayerConfig,
    ComputeConfig,
    TuningConfig>;



#if {{use_swapped_large_nvfp4}}
extern "C" __constant__ uint32_t SMEM_SIZE = 128 * 1024;
extern "C" __constant__ uint32_t SMEM_SIZE_A = 0;
extern "C" __constant__ uint32_t SMEM_SIZE_B = 128 * 1024;
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = 0;
#else
extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A =
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeA * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_B =
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeB * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);
#endif

extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = {{problem_shape[1]}};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = {{problem_shape[2]}};

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = {{block_shape[0]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = {{block_shape[1]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = {{block_shape[2]}};

extern "C" __constant__ uint32_t WARP_SHAPE_M = {{warp_shape[0]}};
extern "C" __constant__ uint32_t WARP_SHAPE_N = {{warp_shape[1]}};
extern "C" __constant__ uint32_t WARP_SHAPE_K = {{warp_shape[2]}};

extern "C" __constant__ uint32_t A_DTYPE_ID = {{a_dtype}}::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = {{b_dtype}}::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = {{c_dtype}}::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = {{bs_dtype}}::kId;

// STSM epilogue is part of the optimal NVFP4 path (on unless the ablation baseline is
// selected via TENSORBRIDGE_NVFP4_PIPELINE_BASELINE). Derived from the single toggle so no
// Python config change is needed and the value stays consistent regardless of header
// include order. The host launcher reads USE_STSM_EPILOGUE (currently informational;
// make_tma_desc_c stages the 128B-swizzled tile unconditionally).
#ifndef TENSORBRIDGE_NVFP4_PIPELINE_BASELINE
#define TENSORBRIDGE_NVFP4_PIPELINE_BASELINE 0
#endif
#ifndef TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM
#define TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM (TENSORBRIDGE_NVFP4_PIPELINE_BASELINE == 0)
#endif
extern "C" __constant__ uint32_t USE_STSM_EPILOGUE = TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM;

{{layer_config_extern}}

{{compute_config_extern}}

{{tuning_config_extern}}

""")


@dataclasses.dataclass(kw_only=True)
class TensorBridgeKernel(KernelRuntime, LayerConfig, ComputeConfig, TuningConfig):
    name: ClassVar[str] = "tensorbridge"
    _str2kernel_cache: ClassVar[dict[tuple[str, str, str, tuple[tuple[str, Any], ...]], int | list[int]]] = {}
    _id2kernel: ClassVar[dict[int, "TensorBridgeKernel"]] = {}

    def __post_init__(self):
        LayerConfig.__post_init__(self)
        ComputeConfig.__post_init__(self)
        TuningConfig.__post_init__(self)
        KernelRuntime.__post_init__(self)

    def init_kernel(self) -> None:
        self.check_shape()
        self.check_dtype()
        self.check_scale()
        self.check_config()
        self.mma_op_class = self.select_mma_op_class()

        assert self.bs_dtype is not None
        self.code = CODE_TEMPLATE.render(
            use_warp_spec=int(self.use_warp_spec or False),
            use_swapped_large_nvfp4=int(self.use_swapped_large_nvfp4 or False),
            mma_op_class=self.mma_op_class.to_cpp_str(),
            problem_shape=self.problem_shape,
            pad_shape=self.pad_shape,
            block_shape=self.block_shape,
            warp_shape=self.warp_shape,
            layer_config=self.to_cpp_str(LayerConfig),
            compute_config=self.to_cpp_str(ComputeConfig),
            tuning_config=self.to_cpp_str(TuningConfig),
            layer_config_extern=self.to_extern_cpp_str(LayerConfig),
            compute_config_extern=self.to_extern_cpp_str(ComputeConfig),
            tuning_config_extern=self.to_extern_cpp_str(TuningConfig),
            layer_config_macro=self.to_macro_cpp_str(LayerConfig),
            compute_config_macro=self.to_macro_cpp_str(ComputeConfig),
            tuning_config_macro=self.to_macro_cpp_str(TuningConfig),
            a_dtype=self.a_dtype.to_cpp_str(),
            b_dtype=self.b_dtype.to_cpp_str(),
            c_dtype=self.c_dtype.to_cpp_str(),
            bs_dtype=self.bs_dtype.to_cpp_str(),
        )
        self.kernel_expr = (
            f"tensorbridge<\n"
            f"    MmaOpClass,\n"
            f"    Shape<0, {self.problem_shape[1]}, {self.problem_shape[2]}>,\n"
            f"    Shape<{self.block_shape[0]}, {self.block_shape[1]}, {self.block_shape[2]}>,\n"
            f"    Shape<{self.warp_shape[0]}, {self.warp_shape[1]}, {self.warp_shape[2]}>,\n"
            f"    Shape<0, {self.pad_shape[1]}, {self.pad_shape[2]}>,\n"
            f"    {self.a_dtype.to_cpp_str()},\n"
            f"    {self.b_dtype.to_cpp_str()},\n"
            f"    {self.c_dtype.to_cpp_str()},\n"
            f"    {self.bs_dtype.to_cpp_str()},\n"
            f"    LayerConfig,\n"
            f"    ComputeConfig,\n"
            f"    TuningConfig>"
        )

        self.prepare()

    def load_cubin(self):
        from tensorbridge import ops

        if self.cubin_loaded:
            return None
        if self.use_swapped_large_nvfp4 and os.environ.get("TENSORBRIDGE_ENABLE_SWAPPED_NVFP4_SKELETON_LAUNCH") != "1":
            raise RuntimeError("swapped NVFP4 one-tile probe is compile-only; refusing to register launchable kernel")
        kernel_filename = self.kernel_filename
        kernel_name = self.kernel_name
        self.kernel_id = ops.register_kernel(kernel_filename, kernel_name)
        self._id2kernel[self.kernel_id] = self
        self.kernel_dirname = os.path.dirname(kernel_filename)
        ref_kernel_id = zlib.crc32(kernel_filename.encode()) << 30
        ref_kernel_id += zlib.crc32(kernel_name.encode())
        assert ref_kernel_id == self.kernel_id
        module = jit_utils.make_tensorbridge_module("get_kernel_id", self.kernel_id)
        self.get_kernel_id = module.get_kernel_id

    def select_mma_op_class(self):
        if self.a_dtype in [dtypes.int4, dtypes.int8]:
            mma_cd_dtype = dtypes.int32
        elif self.use_f16_accum:
            mma_cd_dtype = self.c_dtype
        else:
            mma_cd_dtype = dtypes.float32

        if self.use_swapped_large_nvfp4:
            assert self.mma_type == MmaType.WGMMA
            assert self.a_dtype == dtypes.float8e4m3
            return MmaOpClass.from_config(
                self.mma_type,
                64,
                256,
                256 // self.a_dtype.num_bits,
                self.a_dtype,
                self.a_dtype,
                mma_cd_dtype,
            )

        mma_shape_m = 64 if self.mma_type == MmaType.WGMMA else 16
        mma_shape_n = self.warp_shape[0] if self.mma_type == MmaType.WGMMA else 8
        mma_shape_k = 256 // self.a_dtype.num_bits
        if self.sm_version == 75 and self.a_dtype == dtypes.int8:
            mma_shape_m = 8

        if self.mma_type == MmaType.MMA and self.warp_shape[0] % 16 == 8:
            mma_shape_m = 8

        if self.mma_type == MmaType.MMA and mma_shape_m == 8:
            mma_shape_k = mma_shape_k // 2

        if self.mma_type == MmaType.WGMMA:
            assert self.warp_shape[0] % mma_shape_n == 0
            assert self.warp_shape[1] % (mma_shape_m // 4) == 0
        else:
            assert self.warp_shape[0] % mma_shape_m == 0
            assert self.warp_shape[1] % mma_shape_n == 0
        assert self.warp_shape[2] % mma_shape_k == 0

        return MmaOpClass.from_config(
            self.mma_type,
            mma_shape_m,
            mma_shape_n,
            mma_shape_k,
            self.a_dtype,
            self.a_dtype,
            mma_cd_dtype,
        )

    def check_shape(self):
        if self.use_swapped_large_nvfp4:
            assert self.block_shape == (128, 256, 128)
            assert self.warp_shape == (128, 32, 128)
            assert self.problem_shape[1] <= self.block_shape[0]
            assert self.problem_shape[2] <= self.block_shape[2]
            assert self.problem_shape[1] > self.pad_shape[1]
            assert self.problem_shape[2] > self.pad_shape[2]
            assert self.pad_shape[2] % (128 // self.a_dtype.num_bits) == 0
            return

        assert self.problem_shape[1] % self.block_shape[1] == 0
        assert self.problem_shape[2] % self.block_shape[2] == 0
        assert self.block_shape[0] % self.warp_shape[0] == 0
        assert self.block_shape[1] % self.warp_shape[1] == 0
        assert self.block_shape[2] % self.warp_shape[2] == 0

        assert self.warp_shape[1] % 16 == 0
        assert jit_utils.is_power_of_two(self.block_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2])
        assert jit_utils.is_power_of_two(self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.warp_shape[2])
        assert jit_utils.is_power_of_two(self.block_shape[0] // self.warp_shape[0])
        assert jit_utils.is_power_of_two(self.block_shape[1] // self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2] // self.warp_shape[2])
        assert self.problem_shape[1] > self.pad_shape[1]
        assert self.problem_shape[2] > self.pad_shape[2]
        assert self.pad_shape[1] % 8 == 0
        assert self.pad_shape[2] % (128 // self.a_dtype.num_bits) == 0

        assert self.warp_shape[1] <= 64
        if self.a_dtype.num_bits == 16:
            assert self.warp_shape[1] >= 32
            assert self.warp_shape[2] >= 32
        elif self.a_dtype.num_bits == 8:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 64
        elif self.a_dtype.num_bits == 4:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 128

    def check_scale(self):
        # kPartMmaShapeK on the kernel side; the mainloop processes K in tiles
        # of this size.
        kp = 256 // self.a_dtype.num_bits
        if self.input_scale_group_size > 0:
            # input-scale path not yet relaxed for sub-kp groups (deferred:
            # MXFP4 W4A8 currently uses input_scale_group_size=0).
            assert self.input_scale_group_size >= kp
        if self.weight_scale_group_size > 0:
            # Weight scale groups must align cleanly with the kernel K tile.
            # NVFP4 W4A8 uses group_size=16 with fp8 activations (kp=32), so
            # sub-kp divisors are valid when they partition the tile exactly.
            assert (kp % self.weight_scale_group_size == 0
                    or self.weight_scale_group_size % kp == 0), (
                f"weight_scale_group_size={self.weight_scale_group_size} must "
                f"divide or be a multiple of kPartMmaShapeK={kp}"
            )
        if self.use_fused_e4m3_scale:
            assert self.input_scale_group_size == 0, (
                "NVFP4 W4A8 fused-E4M3 currently supports only unscaled fp8 "
                "activations"
            )
            assert self.weight_scale_group_size in (kp // 2, kp), (
                "NVFP4 W4A8 fused-E4M3 supports only group_size="
                f"{kp // 2} or {kp}; got {self.weight_scale_group_size}"
            )
        if self.use_nvfp4_snc:
            assert self.use_fused_e4m3_scale, "NVFP4 SNC requires the fused-E4M3 FPMA path"
            assert self.a_dtype == dtypes.float8e4m3
            assert self.b_dtype == dtypes.float4e2m1
            assert self.bs_dtype == dtypes.float8e4m3
        if self.weight_scale_group_size_n > 1:
            assert self.weight_scale_group_size_n >= 64

        if self.is_block_weight_scale:
            if self.input_scale_group_size > 0:
                assert self.input_scale_group_size == self.weight_scale_group_size
            assert self.weight_scale_group_size_n > 0
            assert not self.has_zero_point
        if self.is_tensor_weight_scale and not self.is_group_weight_scale:
            self.bs_dtype = self.c_dtype

    def check_dtype(self):
        dtype_map = {
            dtypes.int4: 80,
            dtypes.int8: 75,
            dtypes.float4e2m1: 120,
            dtypes.float8e4m3: 89,
            dtypes.float8e5m2: 89,
            dtypes.bfloat16: 80,
            dtypes.float16: 75,
        }
        assert self.a_dtype in dtype_map
        assert self.sm_version >= dtype_map[self.a_dtype]
        assert self.b_dtype.num_bits <= 8
        assert self.b_dtype.num_bits <= self.a_dtype.num_bits
        if self.b_dtype.is_integer_type and self.a_dtype.is_integer_type:
            if self.a_dtype.num_bits == self.b_dtype.num_bits:
                assert self.a_dtype == self.b_dtype
            else:
                assert not self.b_dtype.is_signed
        elif self.b_dtype.is_integer_type and self.a_dtype.is_floating_point_type:
            assert not self.b_dtype.is_signed
            if self.has_zero_point:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 1
            else:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 2
        elif self.b_dtype.is_floating_point_type and self.a_dtype.is_floating_point_type:
            assert self.b_dtype.is_signed
            assert self.b_dtype.exponent_bits <= self.a_dtype.exponent_bits
            assert self.b_dtype.mantissa_bits <= self.a_dtype.mantissa_bits
            assert self.b_dtype.exponent_bits >= 1
        elif self.b_dtype.is_floating_point_type and not self.a_dtype.is_integer_type:
            raise NotImplementedError

        if self.use_f16_accum:
            if self.a_dtype == dtypes.float8e4m3:
                assert self.b_dtype.is_integer_type or self.b_dtype.exponent_bits <= 4
            else:
                assert self.a_dtype == dtypes.float16

    def check_config(self):
        if self.use_warp_spec or self.use_tma:
            assert self.use_mbarrier
        if self.use_tma_b_swizzle_64:
            assert self.use_tma and self.use_tma_b, "use_tma_b_swizzle_64 requires TMA-B"
        if self.use_nvfp4_swizzle64_raw or self.use_tma_b_swizzle_64:
            def require_swizzle64(cond, msg):
                if not cond:
                    raise RuntimeError(msg)

            require_swizzle64(
                self.use_nvfp4_swizzle64_raw,
                "swizzle64 TMA-B requires nvfp4_swizzle64_raw weight layout",
            )
            require_swizzle64(
                not self.use_nvfp4_raw_s2r_deint,
                "nvfp4_swizzle64_raw and raw S2R deint are mutually exclusive",
            )
            require_swizzle64(
                self.use_tma and self.use_tma_b and self.use_tma_b_swizzle_64,
                "nvfp4_swizzle64_raw requires use_tma=True, use_tma_b=True, use_tma_b_swizzle_64=True",
            )
            require_swizzle64(self.mma_type == MmaType.WGMMA, "nvfp4_swizzle64_raw requires WGMMA")
            require_swizzle64(self.use_fused_e4m3_scale, "nvfp4_swizzle64_raw requires fused E4M3 NVFP4 scales")
            require_swizzle64(self.a_dtype == dtypes.float8e4m3, "nvfp4_swizzle64_raw requires FP8 E4M3 activations")
            require_swizzle64(self.b_dtype == dtypes.float4e2m1, "nvfp4_swizzle64_raw requires NVFP4 weights")
            require_swizzle64(self.bs_dtype == dtypes.float8e4m3, "nvfp4_swizzle64_raw requires FP8 E4M3 scales")
            require_swizzle64(not self.has_zero_point, "nvfp4_swizzle64_raw does not support zero points")
            require_swizzle64(self.is_group_weight_scale, "nvfp4_swizzle64_raw requires K-group weight scales")
            require_swizzle64(self.weight_scale_group_size == 16, "phase-1 nvfp4_swizzle64_raw supports only g16")
            require_swizzle64(self.weight_scale_group_size_n == 0, "nvfp4_swizzle64_raw requires pure K-group scales")
            require_swizzle64(self.problem_shape[1] % 128 == 0, "nvfp4_swizzle64_raw requires N to be 128-aligned")
            require_swizzle64(self.block_shape[1] == 128, "phase-1 nvfp4_swizzle64_raw supports only BlockShape.N=128")
            require_swizzle64(self.warp_shape[1] == 16, "phase-1 nvfp4_swizzle64_raw supports only WarpShape.N=16")
            require_swizzle64(
                self.block_shape[2] == 128 and self.warp_shape[2] == 128,
                "phase-1 nvfp4_swizzle64_raw requires BlockShape.K=WarpShape.K=128",
            )
            require_swizzle64(
                self.block_shape[0] == self.warp_shape[0],
                "phase-1 nvfp4_swizzle64_raw requires BlockShape.M=WarpShape.M",
            )
            preint_layout = os.environ.get("TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT", "0") == "1"
            try:
                nvrtc_flag_tokens = shlex.split(os.environ.get("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", ""))
            except ValueError:
                nvrtc_flag_tokens = os.environ.get("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", "").split()
            preint_device = any(
                token in (
                    "-DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD",
                    "-DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1",
                )
                for token in nvrtc_flag_tokens
            )
            require_swizzle64(
                preint_layout == preint_device,
                "dual-MMA preinterleaved swizzle64 layout requires matching "
                "TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT=1 and "
                "-DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1",
            )
        if self.use_nvfp4_raw_s2r_deint:
            def require_deint(cond, msg):
                if not cond:
                    raise RuntimeError(msg)

            require_deint(not self.use_swapped_large_nvfp4, "raw S2R deint is for the regular TensorBridge path")
            require_deint(self.mma_type == MmaType.WGMMA, "raw S2R deint requires WGMMA")
            require_deint(self.use_fused_e4m3_scale, "raw S2R deint requires fused E4M3 NVFP4 scales")
            require_deint(self.a_dtype == dtypes.float8e4m3, "raw S2R deint requires FP8 E4M3 activations")
            require_deint(self.b_dtype == dtypes.float4e2m1, "raw S2R deint requires NVFP4 weights")
            require_deint(self.bs_dtype == dtypes.float8e4m3, "raw S2R deint requires FP8 E4M3 weight scales")
            require_deint(not self.has_zero_point, "raw S2R deint does not support zero points")
            require_deint(self.is_group_weight_scale, "raw S2R deint requires K-group weight scales")
            require_deint(self.weight_scale_group_size in (16, 32), "raw S2R deint supports only g16/g32 NVFP4")
            require_deint(self.weight_scale_group_size_n == 0, "raw S2R deint requires pure K-group scales")
            require_deint(self.problem_shape[1] % 128 == 0, "raw S2R deint requires N to be 128-aligned")
            require_deint(self.block_shape[1] % 128 == 0, "raw S2R deint requires block_shape.N to be 128-aligned")
            require_deint(
                not (self.block_shape[1] == 256 and self.use_m_fast_tile_order),
                "channel256 raw S2R deint is validated only with n-fast tile order; disable use_m_fast_tile_order",
            )
            require_deint(self.warp_shape[1] == self.a_dtype.num_bits * 2, "raw S2R deint requires half-group WarpShape.N")
            # TMA-B loads the same GMEM bytes as cp.async — deint layout is
            # a GMEM-side property applied by layer.transform(), compatible
            # with both load paths.
        if self.use_swapped_large_nvfp4:
            def require(cond, msg):
                if not cond:
                    raise RuntimeError(msg)

            require(
                os.environ.get("TENSORBRIDGE_ENABLE_SWAPPED_NVFP4_SKELETON_PROBE") == "1",
                "use_swapped_large_nvfp4 currently requires the explicit one-tile probe guard",
            )
            require(self.mma_type == MmaType.WGMMA, "swapped NVFP4 skeleton requires WGMMA")
            require(self.use_fused_e4m3_scale, "swapped NVFP4 skeleton requires fused E4M3 scales")
            require(self.a_dtype == dtypes.float8e4m3, "swapped NVFP4 skeleton requires FP8 activations")
            require(self.b_dtype == dtypes.float4e2m1, "swapped NVFP4 skeleton requires NVFP4 weights")
            require(self.bs_dtype == dtypes.float8e4m3, "swapped NVFP4 skeleton requires FP8 weight scales")
            require(not self.has_zero_point, "swapped NVFP4 skeleton does not support zero points")
            require(self.is_group_weight_scale, "swapped NVFP4 skeleton requires group weight scales")
            require(not self.is_tensor_weight_scale, "swapped NVFP4 skeleton does not support tensor weight scales")
            require(not self.is_block_weight_scale, "swapped NVFP4 skeleton does not support block weight scales")
            require(self.weight_scale_group_size_n == 0, "swapped NVFP4 skeleton requires pure K-group scales")
            require(self.gemm_type in (None, GemmType.DENSE), "swapped NVFP4 skeleton is dense-only")
            require(not self.use_stream_k, "swapped NVFP4 skeleton does not support StreamK")
            require(self.num_stages >= 4, "swapped NVFP4 one-tile probe requires at least 4 stages of smem budget")
            require(self.num_ctas_per_sm == 1, "swapped NVFP4 skeleton is validated only for one CTA per SM")
            if self.use_warp_spec:
                require(self.use_mbarrier, "swapped NVFP4 warp-specialized skeleton expects mbarrier-enabled config")
            else:
                require(not self.use_mbarrier, "swapped NVFP4 consumer-only skeleton does not use mbarrier")
            require(not self.use_tma, "swapped NVFP4 skeleton uses synthetic smem, not TMA")
            require(not any((self.use_tma_a, self.use_tma_b, self.use_tma_c)), "swapped NVFP4 skeleton disables TMA tensors")
            require(not any((self.use_tma_bs, self.use_tma_bzp, self.use_tma_bias)), "swapped NVFP4 skeleton disables TMA extras")
            require(self.weight_scale_group_size == 16, "swapped NVFP4 skeleton is g16-only")
            require(self.block_shape == (128, 256, 128), "swapped NVFP4 skeleton uses 128x256x128 tile")
            require(self.warp_shape == (128, 32, 128), "swapped NVFP4 skeleton uses 2 consumer WGs")
        is_channel_weight_scale = self.is_channel_weight_scale
        is_group_weight_scale = self.is_group_weight_scale
        if not (is_channel_weight_scale or is_group_weight_scale):
            self.use_tma_bs = False
        if not self.has_zero_point:
            self.use_tma_bzp = False
        if not self.has_bias:
            self.use_tma_bias = False
        if self.gemm_type is None and self.num_experts == 0:
            self.gemm_type = GemmType.DENSE
        assert self.gemm_type is not None, "gemm_type must be specify for MoE GEMM"

    def __call__(self):
        msg = (
            "don't call TensorBridgeKernel object directly, "
            "please use tensorbridge.ops.launch_kernel([kernel.kernel_id], ...) instead."
        )
        raise NotImplementedError(msg)

    @classmethod
    def prepare_kernels(
        cls,
        layer_config: str | dict,
        compute_config: str | dict | None = None,
        tuning_config: str | dict | list | None = None,
    ) -> int | list[int]:
        def prepare_config_str(config: str | dict | list | None):
            if config is None:
                return "{}"
            elif isinstance(config, str):
                return config
            else:
                return str(config)

        def prepare_config_obj(config: str | dict | list | None):
            if config is None:
                return {}
            elif not isinstance(config, str):
                return config
            else:
                return json.loads(config)

        layer_config_str = prepare_config_str(layer_config)
        compute_config_str = prepare_config_str(compute_config)
        tuning_config_str = prepare_config_str(tuning_config)
        cache_key = (
            layer_config_str,
            compute_config_str,
            tuning_config_str,
            cls.compile_cache_signature(),
        )
        if cache_key in cls._str2kernel_cache:
            return cls._str2kernel_cache[cache_key]

        layer_config_obj = prepare_config_obj(layer_config)
        compute_config_obj = prepare_config_obj(compute_config)
        tuning_config_obj = prepare_config_obj(tuning_config)
        layer_config_obj.pop("sublayer_name", None)
        # Manual stream-k backend selector: an autotuner INPUT, not a kernel field.
        # Pop it so it drives heuristics but never reaches TensorBridgeKernel directly
        # (the resolved use_stream_k comes back in tuning_config_obj below).
        stream_k_override = compute_config_obj.pop("use_stream_k", None)

        if not tuning_config_obj:
            from tensorbridge.layer import TensorBridgeLayerMeta

            meta = TensorBridgeLayerMeta(**layer_config_obj)
            tuning_config_obj = get_heuristics_config(
                meta, use_stream_k=stream_k_override, **compute_config_obj
            )

        if isinstance(tuning_config_obj, dict):
            config = layer_config_obj | compute_config_obj | tuning_config_obj
            num_sms = config.pop("num_sms", 0)
            kernel = TensorBridgeKernel(**config)
            res = [0, 1 << 30, kernel.kernel_id, num_sms]
            cls._str2kernel_cache[cache_key] = res
            return res

        try:
            import torch

            build_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        except Exception:
            build_device = None

        def prepare_kernel(data):
            if build_device is not None:
                import torch

                torch.cuda.set_device(build_device)
            _, _, tuning_config_obj_single = data
            kernel_config = layer_config_obj | compute_config_obj | tuning_config_obj_single
            num_sms = kernel_config.pop("num_sms", 0)
            kernel = TensorBridgeKernel(**kernel_config)
            return data, kernel, num_sms

        res = []
        if os.environ.get("TENSORBRIDGE_DISABLE_PARALLEL_BUILD", "0") != "1":
            # Parallelize kernel compilation using multiple threads,
            # but ensure kernel loading occurs in the main thread to prevent CUDA context issues.
            # (KernelRuntime would skip loading when running in child thread).
            executor = ThreadPoolExecutor(max_workers=16)
            for config, kernel, num_sms in executor.map(prepare_kernel, tuning_config_obj):
                kernel.load_cubin()
                res += [config[0], config[1], kernel.kernel_id, num_sms]
            executor.shutdown(wait=False)
        else:
            for config in tuning_config_obj:
                data, kernel, num_sms = prepare_kernel(config)
                res += [data[0], data[1], kernel.kernel_id, num_sms]

        cls._str2kernel_cache[cache_key] = res
        return res
