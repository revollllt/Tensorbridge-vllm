from typing import TYPE_CHECKING

import torch

from tensorbridge.kernel.tensorbridge import TensorBridgeKernel
from tensorbridge.ops.bench import tops_bench  # noqa
from tensorbridge.ops.input import quant_input
from tensorbridge.ops.moe import moe_fused_mul_sum
from tensorbridge.ops.utils import init_tensorbridge_launcher, register_op
from tensorbridge.ops.weight import (
    dequant_weight,
    pack_weight,
    process_mxfp4_w4a8_weight,
    quant_weight,
    repack_weight,
    unpack_weight,
)


def register_kernel(cubin_path: str, func_name: str) -> int:
    init_tensorbridge_launcher()
    return torch.ops.tensorbridge.register_kernel(cubin_path, func_name)


def launch_kernel(
    configs: list[int],
    inputs: torch.Tensor,
    weight: torch.Tensor,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    return torch.ops.tensorbridge.launch_kernel(
        configs,
        inputs,
        weight,
        outputs,
        input_scale,
        weight_scale,
        zero_point,
        bias,
        global_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        expert_layout,
        locks,
        top_k,
        valid_shape_m,
    )


def tensorbridge_gemm(
    layer_config: str,
    compute_config: str | None,
    tuning_config: str | None,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    configs = TensorBridgeKernel.prepare_kernels(layer_config, compute_config, tuning_config)
    if isinstance(configs, int):
        configs = [configs]
    return torch.ops.tensorbridge.launch_kernel(
        configs,
        inputs,
        weight,
        outputs,
        input_scale,
        weight_scale,
        zero_point,
        bias,
        global_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        expert_layout,
        locks,
        top_k,
        valid_shape_m,
    )


register_op("tensorbridge::quant_input", quant_input, quant_input)
register_op("tensorbridge::quant_weight", quant_weight, quant_weight)
register_op("tensorbridge::dequant_weight", dequant_weight, dequant_weight)
register_op("tensorbridge::repack_weight", repack_weight, repack_weight)
register_op("tensorbridge::pack_weight", pack_weight, pack_weight)
register_op("tensorbridge::unpack_weight", unpack_weight, unpack_weight)
register_op("tensorbridge::tensorbridge_gemm", tensorbridge_gemm, tensorbridge_gemm)
register_op("tensorbridge::fused_moe_mul_sum", moe_fused_mul_sum, moe_fused_mul_sum)
register_op(
    "tensorbridge::process_mxfp4_w4a8_weight",
    process_mxfp4_w4a8_weight,
    process_mxfp4_w4a8_weight,
)


if not TYPE_CHECKING:
    quant_input = torch.ops.tensorbridge.quant_input
    quant_weight = torch.ops.tensorbridge.quant_weight
    dequant_weight = torch.ops.tensorbridge.dequant_weight
    repack_weight = torch.ops.tensorbridge.repack_weight
    pack_weight = torch.ops.tensorbridge.pack_weight
    process_mxfp4_w4a8_weight = torch.ops.tensorbridge.process_mxfp4_w4a8_weight
    unpack_weight = torch.ops.tensorbridge.unpack_weight
    tensorbridge_gemm = torch.ops.tensorbridge.tensorbridge_gemm
    fused_moe_mul_sum = torch.ops.tensorbridge.fused_moe_mul_sum


__all__ = [
    "quant_input",
    "quant_weight",
    "dequant_weight",
    "repack_weight",
    "pack_weight",
    "process_mxfp4_w4a8_weight",
    "unpack_weight",
    "tensorbridge_gemm",
    "tops_bench",
    "moe_fused_mul_sum",
]
