"""vLLM ModelOpt mixed-precision adapter for TensorBridge NVFP4A8."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import Parameter

from tensorbridge.api.v1 import (
    RUNTIME_API_VERSION,
    TensorBridgeInputSchema,
    TensorBridgeLayerMethod,
    TensorBridgeWeightSchema,
    WeightScaleType,
    build_normal_nvfp4_fp8_weight,
    default_fpma_alpha,
    float4e2m1,
    float8e4m3,
    validate_analytic_fpma_scale_domain,
)
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.parameter import PerTensorScaleParameter


_PREINT_MACRO = "TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD"
_PREINT_FLAG = f"-D{_PREINT_MACRO}=1"
_ULP_MACRO = "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION"
_ULP_FLAG = f"-D{_ULP_MACRO}=1"
_ULP_SCALE_ABI_MACRO = "TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1"
_ULP_SCALE_ABI_FLAG = f"-D{_ULP_SCALE_ABI_MACRO}=1"
_REGISTERED = False
logger = logging.getLogger(__name__)
_REQUIRED_TENSORBRIDGE_RUNTIME_API = 1
_ALGO_ALIASES = {
    "FP8": "FP8",
    "NVFP4": "NVFP4",
    "W4A16_NVFP4": "NVFP4",
}


def _normalize_algo(value: Any) -> str:
    raw = str(value).strip().upper()
    try:
        return _ALGO_ALIASES[raw]
    except KeyError as error:
        raise ValueError(f"unsupported ModelOpt per-layer quant_algo: {raw}") from error


def _fpma_alpha(default: float | None = None) -> float:
    if default is None:
        default = default_fpma_alpha(
            prefold_selector=os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none"),
            ulp_correction=(
                os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "0") == "1"
            ),
        )
    alpha = float(os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ALPHA", str(default)))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"invalid TENSORBRIDGE_NVFP4_FPMA_ALPHA={alpha!r}")
    return alpha


def _uses_analytic_alpha_v1_default() -> bool:
    return (
        "TENSORBRIDGE_NVFP4_FPMA_ALPHA" not in os.environ
        and os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none") == "none"
        and os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "0") == "0"
    )


def _prefold_selector() -> str:
    selector = os.environ.get("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
    if selector not in {"none", "normal_b8_sse"}:
        raise ValueError(f"unknown TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR={selector!r}")
    return selector


def _ulp_correction_enabled() -> bool:
    value = os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION must be 0 or 1, got {value!r}"
        )
    return value == "1"


def _enforce_production_environment(*, default_alpha: float | None = None) -> None:
    if os.environ.get("TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP", "0") == "1":
        raise RuntimeError("TensorBridge vLLM accuracy runs forbid NVFP4 scale clamping")
    compiler = os.environ.get("TENSORBRIDGE_COMPILER", "nvrtc").strip().lower()
    if compiler != "nvrtc":
        raise RuntimeError(
            "TensorBridge vLLM accuracy runs require TENSORBRIDGE_COMPILER=nvrtc; "
            "the production layout flags are NVRTC-only"
        )
    os.environ["TENSORBRIDGE_COMPILER"] = compiler
    os.environ["TENSORBRIDGE_NVFP4_CPP_ROUTER"] = "1"
    os.environ["TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT"] = "1"
    os.environ["TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT"] = "1"
    os.environ.setdefault("TENSORBRIDGE_DISABLE_PARALLEL_BUILD", "1")
    os.environ["VLLM_NVFP4_GEMM_BACKEND"] = "marlin"
    _fpma_alpha(default_alpha)
    _prefold_selector()
    ulp_correction = _ulp_correction_enabled()
    if ulp_correction and (
        _fpma_alpha(default_alpha) != 1.0 or _prefold_selector() != "none"
    ):
        raise ValueError(
            "exact FPMA ULP correction cannot be combined with alpha or prefold selection"
        )

    tokens = shlex.split(os.environ.get("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", ""))
    retained: list[str] = []
    for token in tokens:
        if token.startswith(f"-D{_ULP_MACRO}"):
            if not ulp_correction or token not in {f"-D{_ULP_MACRO}", _ULP_FLAG}:
                raise RuntimeError(f"conflicting TensorBridge ULP correction flag: {token}")
            continue
        if token.startswith(f"-D{_ULP_SCALE_ABI_MACRO}"):
            if not ulp_correction or token not in {
                f"-D{_ULP_SCALE_ABI_MACRO}",
                _ULP_SCALE_ABI_FLAG,
            }:
                raise RuntimeError(f"conflicting TensorBridge ULP scale ABI: {token}")
            continue
        if token.startswith(f"-D{_PREINT_MACRO}"):
            if token not in {f"-D{_PREINT_MACRO}", _PREINT_FLAG}:
                raise RuntimeError(f"conflicting TensorBridge preinterleave flag: {token}")
            continue
        retained.append(token)
    retained.append(_PREINT_FLAG)
    if ulp_correction:
        retained.extend((_ULP_FLAG, _ULP_SCALE_ABI_FLAG))
    os.environ["TENSORBRIDGE_EXTRA_NVRTC_FLAGS"] = shlex.join(retained)


def _isolate_triton_cache() -> None:
    """Keep TP workers from racing on one shared Triton cache directory."""
    base = os.environ.get("VLLM_TRITON_CACHE_BASE")
    if not base:
        tensorbridge_cache = os.environ.get("TENSORBRIDGE_CACHE_DIR")
        if tensorbridge_cache:
            base = str(Path(tensorbridge_cache) / "triton")
            os.environ["VLLM_TRITON_CACHE_BASE"] = base
    if base:
        process_cache = Path(base) / f"pid_{os.getpid()}"
        process_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(process_cache)


def _vllm_imports() -> dict[str, Any]:
    from vllm.model_executor.layers.attention import Attention, MLAAttention
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8KVCacheMethod,
        ModelOptFp8LinearMethod,
        ModelOptFp8MoEMethod,
        ModelOptMixedPrecisionConfig,
        ModelOptNvFp4LinearMethod,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PerTensorScaleParameter,
    )
    from vllm.model_executor.utils import set_weight_attrs

    return locals()


def _register_parameter(
    layer: nn.Module,
    name: str,
    parameter_cls: type,
    data: torch.Tensor,
    extra_weight_attrs: dict[str, Any],
    **parameter_kwargs: Any,
) -> None:
    imports = _vllm_imports()
    set_weight_attrs = imports["set_weight_attrs"]
    attrs = extra_weight_attrs.copy()
    weight_loader = attrs.pop("weight_loader", None)
    if weight_loader is not None:
        parameter_kwargs["weight_loader"] = weight_loader
    parameter = parameter_cls(data=data, **parameter_kwargs)
    set_weight_attrs(parameter, attrs)
    parameter.param_name = name
    parameter.ignore_warning = True
    layer.register_parameter(name, parameter)


class TensorBridgeMarlinNvfp4LmHeadMethod(ModelOptNvFp4LinearMethod):
    """Official W4A16 Marlin method with a stable ParallelLMHead checkpoint ABI."""

    backend = "marlin"

    def __init__(self, quant_config: Any) -> None:
        super().__init__(quant_config)
        if type(self.kernel).__name__ != "MarlinNvFp4LinearKernel":
            raise RuntimeError(
                f"NVFP4 lm_head requires Marlin, got {type(self.kernel).__name__}"
            )

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise TypeError(f"NVFP4 Marlin lm_head requires BF16 activations, got {params_dtype}")
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.params_dtype = params_dtype

        weight_loader = extra_weight_attrs.get("weight_loader")
        for name in ("input_scale", "weight_scale_2"):
            delattr(layer, name)
            parameter = PerTensorScaleParameter(
                data=torch.empty((), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, parameter)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4 Marlin lm_head expected BF16 input, got {x.dtype}")
        return super().apply(layer, x, bias)


class TensorBridgeNvfp4LinearMethod(LinearMethodBase):
    """LinearMethodBase-compatible ModelOpt NVFP4 loader and TensorBridge runner."""

    backend = "tensorbridge"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.compute_config = json.dumps(
            {"gemm_type": "dense", "use_batch_invariant": False, "use_f16_accum": False}
        )

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        imports = _vllm_imports()
        if input_size_per_partition % 16 != 0:
            raise ValueError("TensorBridge NVFP4 requires K divisible by 16")

        n = sum(output_partition_sizes)
        k = input_size_per_partition
        layer.input_size_per_partition = k
        layer.output_partition_sizes = list(output_partition_sizes)
        layer.output_size_per_partition = n
        layer.params_dtype = params_dtype

        _register_parameter(
            layer,
            "weight",
            imports["ModelWeightParameter"],
            torch.empty((n, k // 2), dtype=torch.uint8),
            extra_weight_attrs,
            input_dim=1,
            output_dim=0,
        )
        _register_parameter(
            layer,
            "weight_scale",
            imports["GroupQuantScaleParameter"],
            torch.empty((n, k // 16), dtype=torch.float8_e4m3fn),
            extra_weight_attrs,
            input_dim=1,
            output_dim=0,
        )

        scale_shape: tuple[int, ...]
        if isinstance(layer, imports["ParallelLMHead"]):
            scale_shape = ()
        else:
            scale_shape = (len(output_partition_sizes),)
        for name in ("weight_scale_2", "input_scale"):
            _register_parameter(
                layer,
                name,
                imports["PerTensorScaleParameter"],
                torch.empty(scale_shape, dtype=torch.float32),
                extra_weight_attrs,
            )
            if scale_shape:
                getattr(layer, name).needs_scalar_to_array = True

        layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        _enforce_production_environment()
        global_scales = layer.weight_scale_2.detach().float().reshape(-1)
        if global_scales.numel() == 0:
            raise ValueError(f"{self.prefix}: missing ModelOpt weight_scale_2")
        if not torch.equal(global_scales, global_scales[:1].expand_as(global_scales)):
            raise ValueError(
                f"{self.prefix}: fused NVFP4 global scales differ; "
                "TensorBridge requires one exact FP32 epilogue scale"
            )

        layer.weight = Parameter(
            layer.weight.detach().contiguous().view(torch.int32),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            layer.weight_scale.detach().contiguous(),
            requires_grad=False,
        )
        if _uses_analytic_alpha_v1_default():
            validate_analytic_fpma_scale_domain(layer.weight_scale)
        alpha = _fpma_alpha()
        layer.global_scale = Parameter(
            global_scales[:1].clone() * alpha,
            requires_grad=False,
        )
        del layer.weight_scale_2
        del layer.input_scale

        weight_schema = TensorBridgeWeightSchema(
            b_dtype=float4e2m1,
            bs_dtype=float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP_TENSOR,
            use_nvfp4_snc=True,
            weight_layout="nvfp4_swizzle64_raw",
        )
        input_schema = TensorBridgeInputSchema(
            a_dtype=float8e4m3,
            input_scale_group_size=0,
        )
        has_bias = getattr(layer, "bias", None) is not None
        TensorBridgeLayerMethod.prepare_layer_meta(
            layer=layer,
            shape_n=layer.output_size_per_partition,
            shape_k=layer.input_size_per_partition,
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            has_bias=has_bias,
            torch_dtype=layer.params_dtype,
        )
        TensorBridgeLayerMethod.transform_tensorbridge_layer(layer)
        meta = layer.tensorbridge_metas[""]
        if not (meta.use_nvfp4_snc and meta.use_nvfp4_swizzle64_raw):
            raise RuntimeError(f"{self.prefix}: TensorBridge production contract is inactive")
        layer.tensorbridge_vllm_prefix = self.prefix
        layer.tensorbridge_fpma_alpha = alpha

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        output = TensorBridgeLayerMethod.forward_layer(
            layer=layer,
            inputs=flat,
            compute_config=self.compute_config,
        )
        meta = layer.tensorbridge_metas[""]
        if bias is not None and not meta.has_bias:
            output = output + bias
        return output.reshape(*x.shape[:-1], output.shape[-1])


class TensorBridgeNormalA8LinearMethod(TensorBridgeNvfp4LinearMethod):
    """Normal NVFP4-to-E4M3 baseline executed by vLLM's Cutlass FP8 kernel."""

    backend = "normal_a8"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.fp8_linear = None

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.logical_widths = list(output_partition_sizes)
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticTensorSym,
            weight_shape=(layer.output_size_per_partition, layer.input_size_per_partition),
            input_dtype=params_dtype,
            out_dtype=params_dtype,
            force_kernel=CutlassFP8ScaledMMLinearKernel,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        _enforce_production_environment(default_alpha=1.0)
        global_scales = layer.weight_scale_2.detach().float().reshape(-1)
        if global_scales.numel() == 0:
            raise ValueError(f"{self.prefix}: missing ModelOpt weight_scale_2")
        if not torch.equal(global_scales, global_scales[:1].expand_as(global_scales)):
            raise ValueError(
                f"{self.prefix}: fused NVFP4 global scales differ; "
                "normal-A8 requires one exact FP32 epilogue scale"
            )
        if not torch.isfinite(global_scales).all().item() or (global_scales <= 0).any().item():
            raise ValueError(f"{self.prefix}: normal-A8 requires finite positive global scales")

        chunk_rows = int(os.environ.get("TENSORBRIDGE_NORMAL_A8_CHUNK_ROWS", "256"))
        normal_weight = build_normal_nvfp4_fp8_weight(
            layer.weight.detach().contiguous(),
            layer.weight_scale.detach().contiguous(),
            group_size=16,
            chunk_rows=chunk_rows,
        )
        layer.weight = Parameter(normal_weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(
            (global_scales[0] * 6.0).reshape(()).clone(), requires_grad=False
        )
        del layer.weight_scale_2
        del layer.input_scale
        del layer.locks
        if self.fp8_linear is None:
            raise RuntimeError(f"{self.prefix}: normal-A8 Cutlass kernel is not initialized")
        self.fp8_linear.process_weights_after_loading(layer)
        layer.tensorbridge_vllm_prefix = self.prefix

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fp8_linear is None:
            raise RuntimeError(f"{self.prefix}: normal-A8 Cutlass kernel is not initialized")
        return self.fp8_linear.apply_weights(layer, x, bias)


def _build_mixed_config_class():
    imports = _vllm_imports()
    base = imports["ModelOptMixedPrecisionConfig"]

    class TensorBridgeModelOptMixedConfig(base):
        @classmethod
        def get_min_capability(cls) -> int:
            return 90

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            counts = Counter(
                _normalize_algo(info["quant_algo"])
                for info in self.quantized_layers.values()
            )
            self.tensorbridge_checkpoint_counts = dict(counts)
            head_is_nvfp4 = (
                "lm_head" in self.quantized_layers
                and _normalize_algo(self.quantized_layers["lm_head"]["quant_algo"])
                == "NVFP4"
            )
            self.nvfp4_transformer_layers = counts.get("NVFP4", 0) - int(
                head_is_nvfp4
            )
            logger.info(
                "TensorBridge ModelOpt layout: checkpoint=%s, transformer NVFP4=%d, "
                "lm_head=NVFP4 Marlin W4A16",
                dict(counts),
                self.nvfp4_transformer_layers,
            )
            if os.environ.get("TENSORBRIDGE_STRICT_QWEN36_LAYOUT", "0") == "1":
                expected = {"NVFP4": 193, "FP8": 208}
                if dict(counts) != expected:
                    raise ValueError(
                        f"unexpected Qwen3.6 ModelOpt layout: {dict(counts)}, expected {expected}"
                    )

        def _resolve_quant_algo(self, prefix: str) -> str | None:
            if prefix.endswith(".lm_head") and "lm_head" in self.quantized_layers:
                return _normalize_algo(self.quantized_layers["lm_head"]["quant_algo"])
            resolved = super()._resolve_quant_algo(prefix)
            return None if resolved is None else _normalize_algo(resolved)

        def get_quant_method(self, layer: nn.Module, prefix: str):
            algo = self._resolve_quant_algo(prefix)
            is_linear = isinstance(layer, imports["LinearBase"])
            is_lm_head = isinstance(layer, imports["ParallelLMHead"])
            backend = os.environ.get("TENSORBRIDGE_VLLM_BACKEND", "tensorbridge")
            if backend not in {"tensorbridge", "normal_a8", "official"}:
                raise ValueError(f"unknown TENSORBRIDGE_VLLM_BACKEND={backend!r}")
            if backend != "tensorbridge" and (
                _fpma_alpha(1.0) != 1.0
                or _prefold_selector() != "none"
                or _ulp_correction_enabled()
            ):
                raise ValueError(f"FPMA compensation is incompatible with backend={backend!r}")
            if algo == "NVFP4" and is_lm_head:
                return TensorBridgeMarlinNvfp4LmHeadMethod(self.nvfp4_config)
            if backend == "tensorbridge" and algo == "NVFP4":
                if is_linear:
                    return TensorBridgeNvfp4LinearMethod(prefix)
                raise TypeError(
                    f"TensorBridge has no NVFP4 method for {type(layer).__name__}: {prefix}"
                )
            if backend == "normal_a8" and algo == "NVFP4":
                if is_linear:
                    return TensorBridgeNormalA8LinearMethod(prefix)
                raise TypeError(
                    f"TensorBridge has no normal-A8 method for {type(layer).__name__}: {prefix}"
                )
            if backend == "official" and algo == "NVFP4" and is_linear:
                return imports["ModelOptNvFp4LinearMethod"](self.nvfp4_config)
            return super().get_quant_method(layer, prefix)

    TensorBridgeModelOptMixedConfig.__name__ = "TensorBridgeModelOptMixedConfig"
    TensorBridgeModelOptMixedConfig.__qualname__ = "TensorBridgeModelOptMixedConfig"
    return TensorBridgeModelOptMixedConfig


TensorBridgeModelOptMixedConfig = _build_mixed_config_class()


def register() -> None:
    """Register the mixed ModelOpt override and Qwen3.5 lm_head compatibility model."""
    global _REGISTERED
    if _REGISTERED:
        return
    if RUNTIME_API_VERSION != _REQUIRED_TENSORBRIDGE_RUNTIME_API:
        raise RuntimeError(
            "tensorbridge-vllm requires TensorBridge runtime API "
            f"{_REQUIRED_TENSORBRIDGE_RUNTIME_API}, got {RUNTIME_API_VERSION}"
        )
    _isolate_triton_cache()
    _enforce_production_environment()
    imports = _vllm_imports()
    imports["register_quantization_config"]("modelopt_mixed")(
        TensorBridgeModelOptMixedConfig
    )
    imports["ModelRegistry"].register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm.plugins.tensorbridge_qwen35:TensorBridgeQwen3_5ForConditionalGeneration",
    )
    _REGISTERED = True
