"""Stable TensorBridge runtime API used by serving integrations."""

from tensorbridge import RUNTIME_API_VERSION, __version__
from tensorbridge.config import WeightScaleType
from tensorbridge.dtypes import float4e2m1, float8e4m3
from tensorbridge.layer import TensorBridgeLayerMethod
from tensorbridge.reference.nvfp4 import (
    build_normal_nvfp4_fp8_weight,
    default_fpma_alpha,
    validate_analytic_fpma_scale_domain,
)
from tensorbridge.schema.tensorbridge import TensorBridgeInputSchema, TensorBridgeWeightSchema


if RUNTIME_API_VERSION != 1:
    raise RuntimeError(f"tensorbridge.api.v1 cannot serve runtime API {RUNTIME_API_VERSION}")

__all__ = [
    "RUNTIME_API_VERSION",
    "TensorBridgeInputSchema",
    "TensorBridgeLayerMethod",
    "TensorBridgeWeightSchema",
    "WeightScaleType",
    "__version__",
    "build_normal_nvfp4_fp8_weight",
    "default_fpma_alpha",
    "float4e2m1",
    "float8e4m3",
    "validate_analytic_fpma_scale_domain",
]
