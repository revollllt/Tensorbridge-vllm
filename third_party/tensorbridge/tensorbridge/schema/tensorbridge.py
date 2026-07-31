import dataclasses
from typing import Any, ClassVar

import torch

from tensorbridge import dtypes
from tensorbridge.config.enum import WeightScaleType
from tensorbridge.schema.base import BaseInputSchema, BaseWeightSchema
from tensorbridge.utils.weight import dequantize_weight, quantize_weight


@dataclasses.dataclass(kw_only=True)
class TensorBridgeWeightSchema(BaseWeightSchema):
    quant_method: str = "tensorbridge"
    b_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    use_nvfp4_snc: bool = False
    weight_layout: str = "default"
    use_nvfp4_raw_s2r_deint: bool = False
    use_nvfp4_swizzle64_raw: bool = False

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "b_dtype": ["weight_dtype", "dtype"],
        "weight_scale_group_size": ["group_size"],
        "weight_scale_group_size_n": ["group_size_n"],
        "scale_type": ["weight_scale_type"],
        "bs_dtype": ["weight_scale_dtype", "scale_dtype"],
        "use_nvfp4_snc": ["use_snc"],
        "weight_layout": ["layout"],
    }

    def __post_init__(self):
        assert self.quant_method == "tensorbridge"
        if isinstance(self.b_dtype, str):
            self.b_dtype = dtypes.DataType.from_str(str(self.b_dtype))
        if isinstance(self.b_dtype, dtypes.InergerType) and self.b_dtype.is_signed:
            self.b_dtype = dtypes.DataType.from_str("u" + str(self.b_dtype))
        if isinstance(self.bs_dtype, str):
            self.bs_dtype = dtypes.DataType.from_str(str(self.bs_dtype))

        valid_layouts = {
            "default",
            "current",
            "legacy",
            "nvfp4_raw_s2r_deint_v1",
            # Plain (N, K_int32) row-major int32 packed weight (no repack_weight
            # permutation). For the CUTLASS-aligned 2D + 64B-swizzle TMA-B path
            # where TMA reads weight directly and SMEM swizzle is per-row XOR.
            "nvfp4_swizzle64_raw",
        }
        if self.weight_layout not in valid_layouts:
            raise ValueError(
                f"unsupported TensorBridge weight_layout={self.weight_layout!r}; "
                f"expected one of {sorted(valid_layouts)}"
            )
        if self.weight_layout == "nvfp4_raw_s2r_deint_v1":
            self.use_nvfp4_raw_s2r_deint = True
        elif self.use_nvfp4_raw_s2r_deint:
            self.weight_layout = "nvfp4_raw_s2r_deint_v1"
        if self.weight_layout == "nvfp4_swizzle64_raw":
            self.use_nvfp4_swizzle64_raw = True
        elif self.use_nvfp4_swizzle64_raw:
            self.weight_layout = "nvfp4_swizzle64_raw"
        if self.use_nvfp4_raw_s2r_deint and self.use_nvfp4_swizzle64_raw:
            raise ValueError("nvfp4_raw_s2r_deint_v1 and nvfp4_swizzle64_raw are mutually exclusive")

        if self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP

            # NVFP4 W4A8 (fp4 weight + fp8e4m3 group scale) uses the FPMA bridge,
            # which requires per-group scales to be normalized by a per-tensor /
            # per-channel global_scale (else `code(s) - 0x1C` underflows for the
            # tiny absmax-derived scales). Upgrade to GROUP_TENSOR so that
            # load_from_unquantized produces a proper global_scale.
            if (
                self.weight_scale_type == WeightScaleType.GROUP
                and self.b_dtype == dtypes.float4e2m1
                and self.bs_dtype == dtypes.float8e4m3
            ):
                self.weight_scale_type = WeightScaleType.GROUP_TENSOR

        if self.weight_scale_type == WeightScaleType.BLOCK:
            self.bs_dtype = dtypes.float32

        if self.use_nvfp4_snc and (
            self.b_dtype != dtypes.float4e2m1 or self.bs_dtype != dtypes.float8e4m3
        ):
            raise ValueError("use_nvfp4_snc requires float4e2m1 weights and float8e4m3 scales")

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.b_dtype.num_bits
        group_size = self.weight_scale_group_size or shape_k

        scale_torch_dtype = param_dtype
        if self.bs_dtype == dtypes.float8e8m0:
            scale_torch_dtype = torch.float8_e8m0fnu
        elif self.bs_dtype == dtypes.float8e4m3:
            scale_torch_dtype = torch.float8_e4m3fn
        elif self.bs_dtype == dtypes.float8e5m2:
            scale_torch_dtype = torch.float8_e5m2

        tensor_meta: dict[str, Any] = {
            "weight": {
                "shape": (shape_n, shape_k * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }
        }

        if "GROUP" in str(self.weight_scale_type):
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": "group"},
            }
        elif self.weight_scale_type == WeightScaleType.CHANNEL:
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, 1),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"output_dim": 0, "scale_type": "channel"},
            }
        elif self.weight_scale_type == WeightScaleType.BLOCK:
            group_size_n = self.weight_scale_group_size_n
            tensor_meta["weight_scale"] = {
                "shape": (shape_n // group_size_n, shape_k // group_size),
                "dtype": torch.float32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": "block"},
            }

        if self.has_zero_point and not self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n * num_bits // 32, shape_k // group_size),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }

        if self.has_zero_point and self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": param_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }

        if "TENSOR" in str(self.weight_scale_type):
            tensor_meta["global_scale"] = {
                "shape": (1,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
            }

        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        num_bits = self.b_dtype.num_bits
        shape_n = tensors["weight"].size(-2)
        shape_k = tensors["weight"].size(-1) * 32 // num_bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    @classmethod
    def quant_tensor(
        self,
        tensor: torch.Tensor,
        schema: "TensorBridgeWeightSchema",
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        f16_dtype = dtypes.DataType.from_torch_dtype(param_dtype)
        shape_n = tensor.size(-2)
        shape_k = tensor.size(-1)
        num_experts = tensor.size(0) if tensor.ndim == 3 else None
        tensor_list = quantize_weight(
            weight=tensor,
            dtype=schema.b_dtype,
            scale_dtype=f16_dtype,
            group_size=schema.weight_scale_group_size,
            has_zero_point=schema.has_zero_point,
            has_global_scale="TENSOR" in str(schema.weight_scale_type),
            is_fp_zero_point=schema.is_fp_zero_point,
            pack=True,
        )

        keys = ["weight", "weight_scale", "zero_point", "global_scale"]
        tensors = {}
        for key, output_tensor in zip(keys, tensor_list, strict=True):
            if output_tensor is not None and output_tensor.nelement() > 0:
                tensors[key] = output_tensor

        schema.validate_tensors(tensors, shape_n, shape_k, param_dtype, num_experts)

        return tensors

    def dequant_tensors(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        zero_point = None if not self.has_zero_point else tensors["zero_point"]
        type_str = str(self.weight_scale_type)
        global_scale = None if "TENSOR" not in type_str else tensors["global_scale"]
        return dequantize_weight(
            tensors["weight"],
            weight_scale=tensors["weight_scale"],
            zero_point=zero_point,
            global_scale=global_scale,
            dtype=self.b_dtype,
            packed=True,
        )

    def requant_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        target_weight_schema: "TensorBridgeWeightSchema",
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        dequanted_weight = self.dequant_tensors(tensors)
        return self.quant_tensor(dequanted_weight, target_weight_schema, param_dtype)

    def convert_tensorbridge(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["TensorBridgeWeightSchema", dict[str, torch.Tensor]]:
        return self, tensors


@dataclasses.dataclass(kw_only=True)
class TensorBridgeInputSchema(BaseInputSchema):
    quant_method: str = "tensorbridge"
    a_dtype: dtypes.DataType | None = None
    input_scale_group_size: int = 0

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "a_dtype": ["input_dtype", "dtype"],
        "input_scale_group_size": ["group_size"],
    }

    def __post_init__(self):
        if isinstance(self.a_dtype, str):
            self.a_dtype = dtypes.DataType.from_str(str(self.a_dtype))

    def get_activation_bits(self):
        if self.a_dtype is None:
            return 16
        return self.a_dtype.num_bits

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        return {}

    def convert_tensorbridge(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple["TensorBridgeInputSchema", dict[str, torch.Tensor]]:
        return self, {}
