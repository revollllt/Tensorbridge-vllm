"""Pure PyTorch references for TensorBridge numerical experiments."""

from tensorbridge.reference.nvfp4 import (
    FPMA_ANALYTIC_ALPHA_V1,
    FPMA_ANALYTIC_ALPHA_V1_SCALE_MAX,
    FPMA_ANALYTIC_ALPHA_V1_SCALE_MIN,
    FPMA_ANALYTIC_ALPHA_V1_UNROUNDED,
    FPMA_PREFOLD_DELTA,
    build_nvfp4_reference_weights,
    decode_e2m1_codes,
    default_fpma_alpha,
    fpma_snc_fp8_bytes,
    normal_nvfp4_fp8,
    prefold_nvfp4_scale,
    unpack_nvfp4_weight,
    validate_analytic_fpma_scale_domain,
    validate_nvfp4_scale_domain,
)

__all__ = [
    "FPMA_ANALYTIC_ALPHA_V1",
    "FPMA_ANALYTIC_ALPHA_V1_SCALE_MAX",
    "FPMA_ANALYTIC_ALPHA_V1_SCALE_MIN",
    "FPMA_ANALYTIC_ALPHA_V1_UNROUNDED",
    "FPMA_PREFOLD_DELTA",
    "build_nvfp4_reference_weights",
    "decode_e2m1_codes",
    "default_fpma_alpha",
    "fpma_snc_fp8_bytes",
    "normal_nvfp4_fp8",
    "prefold_nvfp4_scale",
    "unpack_nvfp4_weight",
    "validate_analytic_fpma_scale_domain",
    "validate_nvfp4_scale_domain",
]
