import dataclasses
import json
import math
import os
import re
import shlex
from typing import Any, Callable

import torch

from tensorbridge import dtypes
from tensorbridge.config import GemmType, LayerConfig, MmaType, WeightScaleType
from tensorbridge.schema import (
    BaseInputSchema,
    BaseWeightSchema,
    TensorBridgeInputSchema,
    TensorBridgeWeightSchema,
)
from tensorbridge.tune import get_heuristics_config
from tensorbridge.utils.device import estimate_compute_bound_threshold
from tensorbridge.utils.weight import (
    prepare_tensorbridge_bias,
    prepare_tensorbridge_weight,
    prepare_tensorbridge_weight_scale,
    prepare_tensorbridge_zero_point,
    select_nvfp4_prefold_scale,
)


_NVFP4_ULP_ENV = "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION"
_NVFP4_ULP_MACRO = "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION"
_NVFP4_ULP_SCALE_ABI_MACRO = "TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1"
_NVFP4_ULP_SCALE_ABI = "ulp_scale_msb_flag_v1"


def _nvfp4_ulp_scale_abi_from_environment() -> str | None:
    value = os.environ.get(_NVFP4_ULP_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_NVFP4_ULP_ENV} must be 0 or 1")

    tokens = shlex.split(os.environ.get("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", ""))
    required = {
        name: any(token in {f"-D{name}", f"-D{name}=1"} for token in tokens)
        for name in (_NVFP4_ULP_MACRO, _NVFP4_ULP_SCALE_ABI_MACRO)
    }
    conflicting = [
        token
        for token in tokens
        if any(token.startswith(f"-D{name}") for name in required)
        and token not in {
            f"-D{_NVFP4_ULP_MACRO}",
            f"-D{_NVFP4_ULP_MACRO}=1",
            f"-D{_NVFP4_ULP_SCALE_ABI_MACRO}",
            f"-D{_NVFP4_ULP_SCALE_ABI_MACRO}=1",
        }
    ]
    if conflicting:
        raise RuntimeError(f"conflicting NVFP4 ULP compile flags: {conflicting}")

    if value == "0":
        if any(required.values()):
            raise RuntimeError("NVFP4 ULP compile flags require the host ULP correction")
        return None
    if os.environ.get("TENSORBRIDGE_COMPILER", "").strip().lower() != "nvrtc":
        raise RuntimeError("NVFP4 ULP scale ABI requires TENSORBRIDGE_COMPILER=nvrtc")
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(f"NVFP4 ULP scale ABI is missing compile flags: {missing}")
    return _NVFP4_ULP_SCALE_ABI


def _nvfp4_ulp_environment_signature() -> tuple[str, str, str]:
    return (
        os.environ.get(_NVFP4_ULP_ENV, "0"),
        os.environ.get("TENSORBRIDGE_COMPILER", "").strip().lower(),
        os.environ.get("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", "").strip(),
    )


def get_default_f16_torch_dtype() -> torch.dtype:
    torch_dtype = torch.get_default_dtype()
    if torch_dtype not in [torch.float16, torch.bfloat16]:
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    return torch_dtype


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class TensorBridgeLayerMeta(LayerConfig):
    sublayer_name: str = ""

    @property
    def name_prefix(self):
        return self.sublayer_name + "_" if self.sublayer_name else ""

    @property
    def weight_name(self):
        return self.name_prefix + "weight"

    @property
    def zero_point_name(self):
        return self.name_prefix + "zero_point"

    @property
    def weight_scale_name(self):
        return self.name_prefix + "weight_scale"

    @property
    def global_scale_name(self):
        return self.name_prefix + "global_scale"

    @property
    def bias_name(self):
        return self.name_prefix + "bias"

    @property
    def param_dtype(self):
        if self.c_dtype == dtypes.float16:
            return torch.float16
        elif self.c_dtype == dtypes.bfloat16:
            return torch.bfloat16
        else:
            raise ValueError(f"unsupported c_dtype: {self.c_dtype}")

    @property
    def should_apply_bs_on_c(self):
        if self.use_fused_e8m0_scale:
            return False
        elif self.mma_type == MmaType.MMA:
            return self.weight_scale_group_size == 0 or self.a_dtype.num_bits != 16
        elif self.mma_type == MmaType.WGMMA:
            return self.weight_scale_group_size == 0
        else:
            raise ValueError(f"unsupported mma_type: {self.mma_type}")

    @property
    def weight_nbytes(self):
        nbytes1 = self.shape_n * self.shape_k * self.b_dtype.num_bits // 8
        num_groups = self.shape_k / (self.weight_scale_group_size or self.shape_k)
        assert self.bs_dtype is not None
        nbytes2 = self.shape_n * num_groups * self.bs_dtype.num_bits // 8
        nbytes3 = self.shape_n * num_groups * (math.ceil(self.b_dtype.num_bits / 4) * 4) // 8
        nbytes = nbytes1 + nbytes2
        if self.has_zero_point and self.is_fp_zero_point:
            nbytes = nbytes + nbytes2
        elif self.has_zero_point:
            nbytes = nbytes + nbytes3
        return nbytes * (self.num_experts or 1)

    def estimate_bound_min_shape_m(self, use_f16_accum: bool = False):
        return estimate_compute_bound_threshold(
            weight_nbytes=self.weight_nbytes // (self.num_experts or 1),
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            dtype=str(self.a_dtype),
            use_f16_accum=use_f16_accum,
        )

    def to_str(self) -> str:
        if hasattr(self, "_meta_str"):
            return self._meta_str
        return super().to_str()

    def __setattr__(self, name, value):
        if hasattr(self, "_meta_str"):
            raise AttributeError(f"Instance is frozen, cannot set {name}")
        super().__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.b_dtype, dtypes.InergerType):
            if isinstance(self.b_dtype, dtypes.FloatingPointType):
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)
            elif self.a_dtype.num_bits == self.b_dtype.num_bits:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=True)
            else:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)

        if not self.use_fused_e8m0_scale:
            self.use_fused_e8m0_scale = (
                self.a_dtype in [dtypes.float8e4m3, dtypes.int8]
                and self.weight_scale_group_size > 0
                and self.b_dtype in [dtypes.float4e2m1]
                and self.bs_dtype in [dtypes.float8e8m0]
            )

        if not self.use_fused_e4m3_scale:
            self.use_fused_e4m3_scale = (
                self.a_dtype == dtypes.float8e4m3
                and self.weight_scale_group_size > 0
                and self.b_dtype == dtypes.float4e2m1
                and self.bs_dtype == dtypes.float8e4m3
            )

        if not self.use_int_weight_scale and not self.use_fused_e8m0_scale:
            self.use_int_weight_scale = (
                self.a_dtype in [dtypes.int8, dtypes.int4]
                and self.input_scale_group_size == 0
                and self.weight_scale_group_size > 0
            )

        if self.use_int_weight_scale:
            self.weight_scale_type = WeightScaleType.GROUP_TENSOR
            self.bs_dtype = self.c_dtype

        if self.use_fused_e8m0_scale:
            self.weight_scale_type = WeightScaleType.GROUP_TENSOR

        if self.use_fused_e4m3_scale:
            self.weight_scale_type = WeightScaleType.GROUP_TENSOR
            if self.use_nvfp4_snc:
                assert self.b_dtype == dtypes.float4e2m1
                assert self.a_dtype == dtypes.float8e4m3
                assert self.bs_dtype == dtypes.float8e4m3
        else:
            assert not self.use_nvfp4_snc

        self._meta_str = self.to_str()


class TensorBridgeModule(torch.nn.Module):
    tensorbridge_block_size_configs: dict[str, list[int]]
    tensorbridge_kernel_config_modules: dict[str, Callable]
    tensorbridge_metas: dict[str, TensorBridgeLayerMeta]
    locks: torch.Tensor | None


class TensorBridgeLayerMethod:
    completed_layer_configs: set[tuple[TensorBridgeLayerMeta, tuple[str, ...]]] = set()

    @classmethod
    def may_set_param(cls, layer: torch.nn.Module, name: str, tensor: torch.Tensor | None):
        if tensor is None:
            return
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    @classmethod
    def prepare_layer_meta(
        cls,
        layer: TensorBridgeModule | torch.nn.Module,
        shape_n: int,
        shape_k: int,
        weight_schema: TensorBridgeWeightSchema,
        input_schema: TensorBridgeInputSchema | None = None,
        num_experts: int | None = None,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        has_bias: bool = False,
        torch_dtype: torch.dtype | None = None,
        sublayer_name: str = "",
    ):
        if torch_dtype is None:
            torch_dtype = get_default_f16_torch_dtype()
        f16_dtype = dtypes.DataType.from_torch_dtype(torch_dtype)
        pad_shape_n = math.ceil(shape_n / pad_n_to_multiple) * pad_n_to_multiple - shape_n
        pad_shape_k = math.ceil(shape_k / pad_k_to_multiple) * pad_k_to_multiple - shape_k

        if input_schema is None:
            input_schema = TensorBridgeInputSchema(a_dtype=f16_dtype)

        assert isinstance(input_schema, TensorBridgeInputSchema)
        assert isinstance(weight_schema, TensorBridgeWeightSchema)

        meta = TensorBridgeLayerMeta(
            a_dtype=input_schema.a_dtype or f16_dtype,
            b_dtype=weight_schema.b_dtype,
            bs_dtype=weight_schema.bs_dtype or f16_dtype,
            c_dtype=f16_dtype,
            shape_n=shape_n + pad_shape_n,
            shape_k=shape_k + pad_shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts or 0,
            has_bias=has_bias,
            input_scale_group_size=input_schema.input_scale_group_size,
            weight_scale_group_size=weight_schema.weight_scale_group_size,
            weight_scale_group_size_n=weight_schema.weight_scale_group_size_n,
            weight_scale_type=weight_schema.weight_scale_type,
            has_zero_point=weight_schema.has_zero_point,
            is_fp_zero_point=weight_schema.is_fp_zero_point,
            use_nvfp4_snc=getattr(weight_schema, "use_nvfp4_snc", False),
            use_nvfp4_raw_s2r_deint=weight_schema.use_nvfp4_raw_s2r_deint,
            use_nvfp4_swizzle64_raw=getattr(weight_schema, "use_nvfp4_swizzle64_raw", False),
            sublayer_name=sublayer_name,
        )

        if not hasattr(layer, "tensorbridge_metas"):
            layer.tensorbridge_metas = {}
        assert isinstance(layer.tensorbridge_metas, dict)
        layer.tensorbridge_metas[sublayer_name] = meta

        return meta

    @classmethod
    def check_and_pad_tensors(cls, tensors: dict[str, torch.Tensor], meta: TensorBridgeLayerMeta):
        tensors = tensors.copy()
        schema = TensorBridgeWeightSchema(
            b_dtype=meta.b_dtype,
            bs_dtype=meta.bs_dtype,
            weight_scale_group_size=meta.weight_scale_group_size,
            weight_scale_group_size_n=meta.weight_scale_group_size_n,
            weight_scale_type=meta.weight_scale_type,
            has_zero_point=meta.has_zero_point,
            is_fp_zero_point=meta.is_fp_zero_point,
        )

        if meta.use_int_weight_scale:
            dtype = dtypes.torch_dtype_map[meta.bs_dtype]
            tensors["weight_scale"] = tensors["weight_scale"].to(dtype)
            if "global_scale" not in tensors:
                tensors["global_scale"] = torch.ones(
                    (meta.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

        if meta.use_fused_e8m0_scale:
            if "global_scale" not in tensors:
                tensors["global_scale"] = torch.ones(
                    (meta.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

        if meta.use_fused_e4m3_scale:
            if "global_scale" not in tensors:
                tensors["global_scale"] = torch.ones(
                    (meta.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

        schema.validate_tensors(
            tensors,
            shape_n=meta.shape_n - meta.pad_shape_n,
            shape_k=meta.shape_k - meta.pad_shape_k,
            num_experts=meta.num_experts,
            param_dtype=meta.param_dtype,
            has_bias=meta.has_bias,
        )

        tensors_attrs = schema.get_tensors_attrs(
            shape_n=meta.shape_n,
            shape_k=meta.shape_k,
            num_experts=meta.num_experts,
            param_dtype=meta.param_dtype,
            has_bias=meta.has_bias,
        )

        for key, attrs in tensors_attrs.items():
            shape = attrs["shape"]
            tensor = tensors[key]
            padding: list[int] = []
            value = 0 if tensor.dtype != torch.float8_e8m0fnu else 1
            for i in range(1, len(shape) + 1):
                padding += (0, shape[-i] - tensor.shape[-i])

            tensors[key] = torch.nn.functional.pad(tensor, pad=padding, value=value)

        return tensors

    @classmethod
    def may_process_int_weight_scale(
        cls,
        meta: TensorBridgeLayerMeta,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if meta.bs_dtype is not None and meta.bs_dtype.num_bits == 8:
            assert weight_scale is not None
            torch_dtype = dtypes.torch_dtype_map[meta.c_dtype]
            weight_scale = weight_scale.to(torch_dtype)

        assert weight_scale is not None
        dtype = weight_scale.dtype
        assert dtype in [torch.float16, torch.bfloat16]
        scale_factor = weight_scale.float().abs().max() / 1024
        weight_scale = (weight_scale.float() / scale_factor).round().to(torch.int16)
        weight_scale = weight_scale.view(dtype)

        if global_scale is not None:
            assert global_scale is not None
            out_global_scale = global_scale * scale_factor
        else:
            meta.weight_scale_type = WeightScaleType.GROUP_TENSOR
            out_global_scale = torch.full(
                (meta.num_experts or 1,),
                fill_value=scale_factor.item(),
                device=weight_scale.device,
            )

        return weight_scale, out_global_scale

    @classmethod
    def may_process_fused_e8m0_scale(
        cls,
        meta: TensorBridgeLayerMeta,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin_dtype = weight_scale.dtype
        origin_shape = weight_scale.shape
        assert origin_dtype in [torch.uint8, torch.float8_e8m0fnu]
        weight_scale = weight_scale.view(torch.uint8).view(meta.num_experts or 1, -1)

        scale_max = weight_scale.max(1)[0].unsqueeze(-1)
        scale_min = weight_scale.min(1)[0].unsqueeze(-1)
        scale_range = scale_max - scale_min
        if meta.a_dtype == dtypes.int8:
            max_range_val = 3
        elif meta.a_dtype == dtypes.float8e4m3:
            max_range_val = 12

        max_range = torch.tensor(max_range_val, dtype=torch.uint8, device=scale_range.device)
        scale_range = scale_range.minimum(max_range)
        scale_min_new = scale_max - scale_range
        delta_scale_offsets = weight_scale.maximum(scale_min_new) - weight_scale
        weight_scale = weight_scale.maximum(scale_min_new) - scale_min_new
        weight_scale = weight_scale.view(origin_dtype).view(origin_shape)
        from tensorbridge import ops

        ops.process_mxfp4_w4a8_weight(weight, delta_scale_offsets, inplace=True)

        scale_factor = 2 ** (scale_min_new.view(-1).float() - 127)
        if meta.a_dtype == dtypes.int8:
            scale_factor = scale_factor / 2
        if global_scale is not None:
            assert global_scale is not None
            out_global_scale = global_scale * scale_factor
        else:
            meta.weight_scale_type = WeightScaleType.GROUP_TENSOR
            out_global_scale = scale_factor

        return weight, weight_scale, out_global_scale

    @classmethod
    def may_process_fused_e4m3_scale(
        cls,
        meta: TensorBridgeLayerMeta,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NVFP4 FPMA bridge convention: kernel reads `prefolded_scale = raw - 0x1C`
        # (= -kE4M3HalfMinusSixCodeDelta) and produces fp8(scale * fp4 / 6).
        # We bake the +28 prefold into the scale bytes once here, and multiply
        # global_scale by 6 in fp32 to compensate the bridge's /6.
        assert weight_scale.dtype == torch.float8_e4m3fn
        raw = weight_scale.view(torch.uint8).to(torch.int16)
        group_words = meta.weight_scale_group_size * meta.b_dtype.num_bits // 32
        assert group_words > 0
        assert meta.weight_scale_group_size * meta.b_dtype.num_bits % 32 == 0

        weight_words = weight.view(torch.int32)
        weight_view = weight_words.unsqueeze(0) if weight_words.ndim == 2 else weight_words
        raw_view = raw.unsqueeze(0) if raw.ndim == 2 else raw
        assert raw_view.shape[:-1] == weight_view.shape[:-1]
        assert weight_view.size(-1) == raw_view.size(-1) * group_words

        # raw E4M3 scale 0 must make the whole FP4 group zero. Otherwise the
        # bridge would still emit non-zero FP8 bytes from `0 + element_addend`.
        if (raw_view == 0).any().item():
            weight_words = weight_words.clone()
            weight_view = weight_words.unsqueeze(0) if weight_words.ndim == 2 else weight_words
            weight_view.view(*raw_view.shape, group_words)[raw_view == 0] = 0
            weight = weight_words.view(weight.dtype).view(weight.shape)

        try:
            cls.validate_custom_nvfp4_scale_domain(meta, weight, raw)
        except ValueError:
            if os.environ.get("TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP", "0") != "1":
                raise
            # System-level serving benchmarks do not check model quality. Clamp
            # non-zero groups into the FPMA bridge's representable scale domain
            # so checkpoints with wider scale spread can still exercise kernels.
            weight_words_for_clamp = weight.view(torch.int32)
            weight_view_for_clamp = (
                weight_words_for_clamp.unsqueeze(0)
                if weight_words_for_clamp.ndim == 2
                else weight_words_for_clamp
            )
            raw_view_for_clamp = raw.unsqueeze(0) if raw.ndim == 2 else raw
            group_nonzero = ((weight_view_for_clamp & 0x77777777) != 0).view(
                *raw_view_for_clamp.shape, group_words
            ).any(dim=-1)
            raw_view_for_clamp.clamp_(min=0x1C, max=0x7E)
            raw_view_for_clamp.masked_fill_(~group_nonzero, 0)
            cls.validate_custom_nvfp4_scale_domain(meta, weight, raw)
        ulp_scale_abi = _nvfp4_ulp_scale_abi_from_environment()
        ulp_correction = "1" if ulp_scale_abi is not None else "0"
        if ulp_correction == "1":
            alpha = float(os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "1.0"))
            selector = os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
            if os.environ.get("TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP", "0") == "1":
                raise ValueError("exact FPMA ULP correction forbids scale clamping")
            if (
                not meta.use_nvfp4_snc
                or meta.weight_scale_group_size != 16
                or not getattr(meta, "use_nvfp4_swizzle64_raw", False)
            ):
                raise ValueError(
                    "exact FPMA ULP correction requires SNC g16 NVFP4 swizzle64-raw"
                )
            if alpha != 1.0 or selector != "none":
                raise ValueError(
                    "exact FPMA ULP correction cannot be combined with alpha or prefold selection"
                )
            nonzero_raw = raw[raw != 0]
            if nonzero_raw.numel() and int(nonzero_raw.min().item()) < 0x39:
                raise ValueError("exact FPMA ULP correction requires raw E4M3 scales >= 0x39")
        prefold_selector = os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
        if prefold_selector == "none":
            prefolded = (raw - 0x1C).clamp_(min=0, max=255).to(torch.uint8)
        elif prefold_selector == "normal_b8_sse":
            if not meta.use_nvfp4_snc or meta.weight_scale_group_size != 16:
                raise ValueError("normal_b8_sse prefold selection requires SNC g16 NVFP4")
            chunk_rows = int(
                os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS", "256")
            )
            prefolded = select_nvfp4_prefold_scale(
                weight,
                weight_scale,
                group_size=meta.weight_scale_group_size,
                chunk_rows=chunk_rows,
            )
        else:
            raise ValueError(
                f"unknown TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR={prefold_selector!r}"
            )
        if ulp_correction == "1":
            # Bit 7 is unused by the verified prefold range [0x1d, 0x62].
            # Carry the residue decision into the mainloop without storing an
            # extra mask or recomputing it for every dequantized fragment.
            residue = prefolded & 0x07
            eligible = (residue >= 2) & (residue <= 6)
            prefolded = prefolded | (eligible.to(torch.uint8) << 7)
        out_weight_scale = prefolded.view(torch.float8_e4m3fn).view(weight_scale.shape)

        if global_scale is None:
            out_global_scale = torch.full(
                (meta.num_experts or 1,),
                fill_value=6.0,
                device=weight_scale.device,
                dtype=torch.float32,
            )
        else:
            out_global_scale = global_scale.float() * 6.0

        return weight, out_weight_scale, out_global_scale

    @classmethod
    def validate_custom_nvfp4_scale_domain(
        cls,
        meta: TensorBridgeLayerMeta,
        weight: torch.Tensor,
        raw_scale: torch.Tensor,
    ) -> None:
        assert meta.b_dtype == dtypes.float4e2m1
        assert meta.bs_dtype == dtypes.float8e4m3
        group_words = meta.weight_scale_group_size * meta.b_dtype.num_bits // 32
        assert group_words > 0
        assert meta.weight_scale_group_size * meta.b_dtype.num_bits % 32 == 0

        weight_words = weight.view(torch.int32)
        weight_words = weight_words.unsqueeze(0) if weight_words.ndim == 2 else weight_words
        raw = raw_scale.unsqueeze(0) if raw_scale.ndim == 2 else raw_scale
        assert raw.shape[:-1] == weight_words.shape[:-1]
        assert weight_words.size(-1) == raw.size(-1) * group_words

        # raw=0 is accepted only for true zero groups. FP4 -0 has mag=0 too,
        # so only the three magnitude bits in each nibble participate here.
        mag_nonzero_words = (weight_words & 0x77777777) != 0
        group_nonzero = mag_nonzero_words.view(*raw.shape, group_words).any(dim=-1)
        legal_raw = (raw >= 0x1C) & (raw <= 0x7E)
        legal_zero = (raw == 0) & ~group_nonzero
        invalid = ~(legal_raw | legal_zero)
        if invalid.any().item():
            bad_raw = int(raw[invalid][0].item())
            bad_count = int(invalid.sum().item())
            raise ValueError(
                "custom nvfp4a8 requires raw E4M3 scale bytes in "
                "[0x1C, 0x7E], except raw 0 for all-zero FP4 groups; "
                f"found 0x{bad_raw:02x} in {bad_count} group(s)"
            )

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer: TensorBridgeModule | torch.nn.Module,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType | str = GemmType.DENSE,
        sublayer_name: str = "",
        use_stream_k: bool | None = None,
    ) -> list[Any]:
        assert isinstance(layer.tensorbridge_metas, dict)
        meta = layer.tensorbridge_metas[sublayer_name]
        return get_heuristics_config(
            meta=meta,
            use_f16_accum=use_f16_accum,
            gemm_type=gemm_type,
            use_batch_invariant=use_batch_invariant,
            use_stream_k=use_stream_k,
        )

    @classmethod
    def transform_tensorbridge_layer(
        cls,
        layer: TensorBridgeModule | torch.nn.Module,
        sublayer_name: str = "",
        already_padded: bool = False,
    ):
        assert isinstance(layer.tensorbridge_metas, dict)
        meta = layer.tensorbridge_metas[sublayer_name]
        prefix = meta.name_prefix
        tensors = dict(
            (key.removeprefix(prefix), value)
            for key, value in layer.state_dict().items()
            if key.startswith(prefix)
        )

        if not already_padded:
            tensors = cls.check_and_pad_tensors(tensors, meta)

        weight = tensors["weight"]
        zero_point = tensors["zero_point"] if meta.has_zero_point else None
        weight_scale: torch.Tensor | None = None
        if meta.weight_scale_type != WeightScaleType.TENSOR:
            weight_scale = tensors["weight_scale"]
        bias = tensors["bias"] if meta.has_bias else None
        if "TENSOR" in str(meta.weight_scale_type):
            global_scale = tensors.get("global_scale", None)
        else:
            global_scale = None

        if meta.use_fused_e8m0_scale:
            assert weight_scale is not None
            weight, weight_scale, global_scale = cls.may_process_fused_e8m0_scale(
                meta,
                weight=weight,
                weight_scale=weight_scale,
                global_scale=global_scale,
            )
        elif meta.use_fused_e4m3_scale:
            assert weight_scale is not None
            weight, weight_scale, global_scale = cls.may_process_fused_e4m3_scale(
                meta,
                weight=weight,
                weight_scale=weight_scale,
                global_scale=global_scale,
            )
            scale_abi = _nvfp4_ulp_scale_abi_from_environment()
            layer._tensorbridge_nvfp4_scale_abi = scale_abi
            layer._tensorbridge_nvfp4_scale_abi_env = (
                _nvfp4_ulp_environment_signature() if scale_abi is not None else None
            )

        weight = prepare_tensorbridge_weight(
            weight=weight,
            b_dtype=meta.b_dtype,
            a_dtype=meta.a_dtype,
            zero_point=zero_point,
            use_wgmma=meta.mma_type == MmaType.WGMMA,
            use_fused_e8m0_scale=meta.use_fused_e8m0_scale,
            use_fused_e4m3_scale=meta.use_fused_e4m3_scale,
            use_nvfp4_snc=meta.use_nvfp4_snc,
            use_nvfp4_raw_s2r_deint=meta.use_nvfp4_raw_s2r_deint,
            use_nvfp4_swizzle64_raw=getattr(meta, "use_nvfp4_swizzle64_raw", False),
            packed=True,
        )

        if weight_scale is not None:
            weight_scale = prepare_tensorbridge_weight_scale(
                weight_scale,
                to_apply_on_c=meta.should_apply_bs_on_c,
                is_blockwise=meta.weight_scale_type == WeightScaleType.BLOCK,
                use_nvfp4_swizzle64_raw=getattr(meta, "use_nvfp4_swizzle64_raw", False),
            )

        if zero_point is not None:
            zero_point = prepare_tensorbridge_zero_point(zero_point, meta.b_dtype, packed=True)

        if bias is not None:
            bias = prepare_tensorbridge_bias(bias)

        if meta.use_int_weight_scale:
            assert weight_scale is not None
            weight_scale, global_scale = cls.may_process_int_weight_scale(
                meta,
                weight_scale=weight_scale,
                global_scale=global_scale,
            )

        cls.may_set_param(layer, meta.weight_name, weight)
        cls.may_set_param(layer, meta.weight_scale_name, weight_scale)
        cls.may_set_param(layer, meta.zero_point_name, zero_point)
        cls.may_set_param(layer, meta.global_scale_name, global_scale)
        cls.may_set_param(layer, meta.bias_name, bias)

    @classmethod
    def may_quant_input(
        cls,
        layer: TensorBridgeModule | torch.nn.Module,
        inputs: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        sublayer_name: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(layer.tensorbridge_metas, dict)
        meta = layer.tensorbridge_metas[sublayer_name]
        if meta.a_dtype.num_bits == 16:
            return inputs, None
        if input_scale is not None:
            return inputs, input_scale
        from tensorbridge import ops

        quanted_input, input_scale = ops.quant_input(
            inputs=inputs,
            outputs=quanted_input,
            dtype=str(meta.a_dtype),
            group_size=None,
        )
        return quanted_input, (input_scale if input_scale.size() else None)

    @classmethod
    def forward_layer(
        cls,
        layer: TensorBridgeModule | torch.nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
        sublayer_name: str = "",
    ):
        assert isinstance(layer.tensorbridge_metas, dict)
        meta = layer.tensorbridge_metas[sublayer_name]
        if meta.use_fused_e4m3_scale:
            stored_abi = getattr(layer, "_tensorbridge_nvfp4_scale_abi", None)
            current_signature = _nvfp4_ulp_environment_signature()
            relevant_flag_present = any(
                name in current_signature[2]
                for name in (_NVFP4_ULP_MACRO, _NVFP4_ULP_SCALE_ABI_MACRO)
            )
            if stored_abi is None:
                if current_signature[0] == "1" or relevant_flag_present:
                    raise RuntimeError("unflagged NVFP4 scales cannot use the ULP kernel ABI")
            elif (
                stored_abi != _NVFP4_ULP_SCALE_ABI
                or current_signature
                != getattr(layer, "_tensorbridge_nvfp4_scale_abi_env", None)
            ):
                raise RuntimeError("NVFP4 ULP scale ABI changed after weight transformation")
        inputs, input_scale = cls.may_quant_input(
            layer=layer,
            inputs=inputs,
            input_scale=input_scale,
            sublayer_name=sublayer_name,
        )

        if isinstance(compute_config, dict):
            compute_config = json.dumps(compute_config)

        if isinstance(tuning_config, (list, dict)):
            tuning_config = json.dumps(tuning_config)

        from tensorbridge import ops

        return ops.tensorbridge_gemm(
            layer_config=meta.to_str(),
            compute_config=compute_config,
            tuning_config=tuning_config,
            inputs=inputs,
            weight=getattr(layer, meta.weight_name),
            outputs=outputs,
            input_scale=input_scale,
            weight_scale=getattr(layer, meta.weight_scale_name, None),
            zero_point=getattr(layer, meta.zero_point_name, None),
            bias=getattr(layer, meta.bias_name, None),
            global_scale=getattr(layer, meta.global_scale_name, None),
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            locks=layer.locks,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
        )


class TensorBridgeMethod(TensorBridgeLayerMethod):
    pass


@dataclasses.dataclass(repr=False, eq=False)
class TensorBridgeLayer(TensorBridgeModule):
    shape_n: int
    shape_k: int
    weight_config: BaseWeightSchema | dict[str, Any]
    input_config: BaseInputSchema | dict[str, Any] | None = None
    pad_n_to_multiple: int = 1
    pad_k_to_multiple: int = 1
    num_experts: int | None = None
    has_bias: bool = False
    torch_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        super().__init__()

        if self.torch_dtype is None:
            self.torch_dtype = get_default_f16_torch_dtype()
        assert self.torch_dtype in [torch.float16, torch.bfloat16], self.torch_dtype

        self.input_config = self.input_config or {}

        if isinstance(self.input_config, dict):
            if "quant_method" not in self.input_config:
                self.input_config["quant_method"] = "tensorbridge"
            if "dtype" not in self.input_config:
                self.input_config["dtype"] = dtypes.DataType.from_torch_dtype(self.torch_dtype)
        if isinstance(self.weight_config, dict) and "quant_method" not in self.weight_config:
            self.weight_config["quant_method"] = "tensorbridge"

        self.input_schema: BaseInputSchema = (
            self.input_config
            if isinstance(self.input_config, BaseInputSchema)
            else BaseInputSchema.from_config(self.input_config)
        )

        self.weight_schema: BaseWeightSchema = (
            self.weight_config
            if isinstance(self.weight_config, BaseWeightSchema)
            else BaseWeightSchema.from_config(self.weight_config)
        )

        tensors_attrs = self.weight_schema.get_tensors_attrs(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            param_dtype=self.torch_dtype,
            num_experts=self.num_experts,
            has_bias=self.has_bias,
        )

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            param = torch.nn.Parameter(tensor, requires_grad=False)
            for key, value in attrs.items():
                if key not in ["shape", "dtype"]:
                    setattr(param, key, value)
            setattr(self, name, param)

        locks = torch.zeros((1024), dtype=torch.int32, device="cuda:0")
        self.register_buffer("locks", locks)

    @staticmethod
    def filter_tensors(
        tensors: dict[str, torch.Tensor], prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        tensors_new = {}
        for key in tensors:
            if key.startswith(prefix):
                key_new = key.removeprefix(prefix).lstrip(".")
                tensors_new[key_new] = tensors[key]
        return tensors_new

    def load_from_unquantized(self, tensor: torch.Tensor):
        assert isinstance(self.weight_schema, TensorBridgeWeightSchema)
        assert tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
        expected_shape: tuple[int, ...] = (self.shape_n, self.shape_k)
        if self.num_experts is not None and self.num_experts != 0:
            expected_shape = (self.num_experts,) + expected_shape
        assert tensor.shape == expected_shape

        from tensorbridge.utils.weight import quantize_weight

        f16_dtype = dtypes.DataType.from_torch_dtype(self.torch_dtype)
        weight, weight_scale, zero_point, global_scale = quantize_weight(
            weight=tensor,
            dtype=self.weight_schema.b_dtype,
            scale_dtype=self.weight_schema.bs_dtype or f16_dtype,
            group_size=self.weight_schema.weight_scale_group_size,
            has_zero_point=self.weight_schema.has_zero_point,
            has_global_scale="TENSOR" in str(self.weight_schema.weight_scale_type),
            is_fp_zero_point=self.weight_schema.is_fp_zero_point,
            pack=True,
        )

        tensors = {"weight": weight}
        if weight_scale is not None:
            tensors["weight_scale"] = weight_scale
        if zero_point is not None:
            tensors["zero_point"] = zero_point
        if global_scale is not None:
            if global_scale.dim() == 0:
                global_scale = global_scale.reshape(1)
            tensors["global_scale"] = global_scale

        self.load_from_tensors(tensors)

    def load_from_tensors(self, tensors: dict[str, torch.Tensor], prefix: str = ""):
        tensors = self.filter_tensors(tensors, prefix)
        self.load_state_dict(tensors, strict=False)

    def load_from_safetensors(self, name: str, prefix: str = ""):
        assert os.path.exists(name)
        import safetensors.torch

        if os.path.isfile(name):
            tensors = safetensors.torch.load_file(name)
            return self.load_from_tensors(tensors, prefix)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            return self.load_from_safetensors(filename, prefix)

        assert os.path.exists(index_filename)
        with open(index_filename, "r") as f:
            index_data = json.load(f)
        loaded_filenames = set()
        for key, filename in index_data["weight_map"].items():
            filename = os.path.join(name, filename)
            if filename in loaded_filenames:
                continue
            if key.startswith(prefix):
                self.load_from_safetensors(filename, prefix)
                loaded_filenames.add(filename)

    @classmethod
    def from_safetensors(
        cls,
        name: str,
        prefix: str = "",
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        torch_dtype: torch.dtype | None = None,
    ):
        assert os.path.isdir(name)
        import safetensors.torch

        config_filename = os.path.join(name, "config.json")
        with open(config_filename, "r") as f:
            config = json.load(f)
            if torch_dtype is None and config.get("torch_dtype", "") == "float16":
                torch_dtype = torch.float16

            assert "quantization_config" in config, "not a quantization model"
            config = config["quantization_config"]

        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        for key in keys:
            ignore_layers = config.get(key, []) or []
            assert not any(x in prefix for x in ignore_layers), f"layer {prefix} is unquantized"

        layer_config = config.copy()
        for regex in config.get("dynamic", {}):
            if regex[:1] != "-":
                assert not re.match(regex[2:], prefix), f"layer {prefix} is unquantized"
            elif re.match(regex[2:], prefix):
                layer_config.update(config["dynamic"][regex])
                break

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            target_group_config = None
            for group_config in config["config_groups"].values():
                if "Linear" in group_config["targets"]:
                    target_group_config = group_config["weights"].copy()
                    break
            assert target_group_config is not None, f"layer {prefix} is unquantized"
            target_group_config["quant_method"] = config["quant_method"]
            if "format" in config:
                target_group_config["format"] = config["format"]
            if "quant_algo" in config:
                target_group_config["quant_algo"] = config["quant_algo"]
            layer_config = target_group_config

        schema = BaseWeightSchema.from_config(layer_config)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            tensors = safetensors.torch.load_file(filename)
            tensors = cls.filter_tensors(tensors, prefix)
        else:
            assert os.path.exists(index_filename)
            with open(index_filename, "r") as f:
                index_data = json.load(f)
            loaded_filenames = set()
            tensors = {}
            for key, filename in index_data["weight_map"].items():
                filename = os.path.join(name, filename)
                if filename in loaded_filenames:
                    continue
                if key.startswith(prefix):
                    tensors2 = safetensors.torch.load_file(filename)
                    tensors.update(cls.filter_tensors(tensors2, prefix))
                    loaded_filenames.add(filename)

        shape_n, shape_k, num_experts, has_bias = schema.infer_shape(tensors)

        layer = cls(
            shape_n=shape_n,
            shape_k=shape_k,
            weight_config=schema,
            num_experts=num_experts or 0,
            pad_n_to_multiple=pad_n_to_multiple,
            pad_k_to_multiple=pad_k_to_multiple,
            has_bias=has_bias,
            torch_dtype=torch_dtype,
        )

        layer.load_from_tensors(tensors)
        return layer

    def transform(self):
        if not isinstance(self.weight_schema, TensorBridgeWeightSchema):
            assert self.torch_dtype is not None
            self.weight_schema, tensors = self.weight_schema.convert_tensorbridge(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            self.input_schema, _ = self.input_schema.convert_tensorbridge(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            for name, _ in list(self.named_parameters()):
                delattr(self, name)

            for name, tensor in tensors.items():
                param = torch.nn.Parameter(tensor, requires_grad=False)
                setattr(self, name, param)

        assert isinstance(self.input_schema, TensorBridgeInputSchema)
        TensorBridgeLayerMethod.prepare_layer_meta(
            layer=self,
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            num_experts=self.num_experts,
            pad_n_to_multiple=self.pad_n_to_multiple,
            pad_k_to_multiple=self.pad_k_to_multiple,
            torch_dtype=self.torch_dtype,
            has_bias=self.has_bias,
        )

        TensorBridgeLayerMethod.transform_tensorbridge_layer(self)

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
    ) -> torch.Tensor:
        return TensorBridgeLayerMethod.forward_layer(
            layer=self,
            inputs=inputs,
            outputs=outputs,
            input_scale=input_scale,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            top_k=top_k,
            compute_config=compute_config,
            tuning_config=tuning_config,
        )
