import importlib.metadata
import multiprocessing
from types import MethodType, SimpleNamespace

import pytest
import torch


try:
    _VLLM_VERSION = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    _VLLM_VERSION = "not-installed"

if not _VLLM_VERSION.startswith("0.20.2"):
    pytest.skip("TensorBridge vLLM adapter targets vLLM 0.20.2", allow_module_level=True)

from vllm.plugins import tensorbridge as integration
from vllm.plugins.tensorbridge_qwen35 import (
    TensorBridgeQwen3_5ForConditionalGeneration,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers import vocab_parallel_embedding as vocab_module
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration


def _assert_spawned_config_class(config_cls):
    assert config_cls.__name__ == "TensorBridgeModelOptMixedConfig"


def test_plugin_overrides_modelopt_mixed_registry(monkeypatch):
    monkeypatch.setenv("TENSORBRIDGE_STRICT_QWEN36_LAYOUT", "0")
    integration.register()

    config_cls = get_quantization_config("modelopt_mixed")
    assert config_cls.__name__ == "TensorBridgeModelOptMixedConfig"
    assert issubclass(integration.TensorBridgeNvfp4LinearMethod, LinearMethodBase)
    assert issubclass(integration.TensorBridgeNormalA8LinearMethod, LinearMethodBase)


def test_mixed_config_class_survives_spawn_import():
    integration.register()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_assert_spawned_config_class,
        args=(integration.TensorBridgeModelOptMixedConfig,),
    )
    process.start()
    process.join(timeout=600)
    if process.is_alive():
        process.terminate()
        process.join()
    assert process.exitcode == 0


def test_mixed_config_normalizes_w4a16_and_lm_head_alias(monkeypatch):
    monkeypatch.setenv("TENSORBRIDGE_STRICT_QWEN36_LAYOUT", "0")
    integration.register()
    config_cls = get_quantization_config("modelopt_mixed")
    config = config_cls(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers={
            "model.layers.0.mlp.gate_proj": {"quant_algo": "W4A16_NVFP4"},
            "lm_head": {"quant_algo": "W4A16_NVFP4"},
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        },
        fp8_config=object(),
        nvfp4_config=object(),
    )

    assert config._resolve_quant_algo("model.layers.0.mlp.gate_proj") == "NVFP4"
    assert config._resolve_quant_algo("model.language_model.lm_head") == "NVFP4"
    assert config.tensorbridge_checkpoint_counts == {"NVFP4": 2, "FP8": 1}
    assert config.nvfp4_transformer_layers == 1
    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    for backend in ("tensorbridge", "normal_a8", "official"):
        monkeypatch.setenv("TENSORBRIDGE_VLLM_BACKEND", backend)
        method = config.get_quant_method(lm_head, "model.language_model.lm_head")
        assert isinstance(method, integration.TensorBridgeMarlinNvfp4LmHeadMethod)
        assert method.backend == "marlin"
        assert type(method.kernel).__name__ == "MarlinNvFp4LinearKernel"

    linear = LinearBase.__new__(LinearBase)
    expected_methods = {
        "tensorbridge": integration.TensorBridgeNvfp4LinearMethod,
        "normal_a8": integration.TensorBridgeNormalA8LinearMethod,
        "official": integration._vllm_imports()["ModelOptNvFp4LinearMethod"],
    }
    for backend, expected_method in expected_methods.items():
        monkeypatch.setenv("TENSORBRIDGE_VLLM_BACKEND", backend)
        method = config.get_quant_method(linear, "model.layers.0.mlp.gate_proj")
        assert isinstance(method, expected_method)


def test_nvfp4_method_creates_fused_checkpoint_parameters(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = torch.nn.Module()

    def weight_loader(*args, **kwargs):
        del args, kwargs

    method = integration.TensorBridgeNvfp4LinearMethod("model.layers.0.mlp.gate_up_proj")
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[256, 256],
        input_size=128,
        output_size=512,
        params_dtype=torch.bfloat16,
        weight_loader=weight_loader,
    )

    assert layer.weight.shape == (512, 64)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (512, 8)
    assert layer.weight_scale.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_2.shape == (2,)
    assert layer.input_scale.shape == (2,)
    assert layer.weight_scale_2.needs_scalar_to_array is True
    assert layer.input_scale.needs_scalar_to_array is True
    assert layer.locks.shape == (1024,)


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_row_parallel_slices_packed_k_and_group_scales(monkeypatch, rank):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 2
    )
    owner = SimpleNamespace(tp_rank=rank, tp_size=2)
    weight_loader = MethodType(RowParallelLinear.weight_loader, owner)
    layer = torch.nn.Module()
    method = integration.TensorBridgeNvfp4LinearMethod(
        "model.layers.0.mlp.down_proj"
    )
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[4],
        input_size=128,
        output_size=4,
        params_dtype=torch.bfloat16,
        weight_loader=weight_loader,
    )

    full_weight = (torch.arange(4 * 64).reshape(4, 64) % 251).to(torch.uint8)
    full_scale_raw = (0x39 + torch.arange(4 * 8).reshape(4, 8) % 16).to(
        torch.uint8
    )
    full_scale = full_scale_raw.view(torch.float8_e4m3fn)
    layer.weight.weight_loader(layer.weight, full_weight)
    layer.weight_scale.weight_loader(layer.weight_scale, full_scale)
    layer.weight_scale_2.weight_loader(layer.weight_scale_2, torch.tensor(0.25))
    layer.input_scale.weight_loader(layer.input_scale, torch.tensor(0.5))

    assert torch.equal(layer.weight, full_weight[:, rank * 32 : (rank + 1) * 32])
    expected_scale = full_scale_raw[:, rank * 4 : (rank + 1) * 4]
    assert torch.equal(layer.weight_scale.view(torch.uint8), expected_scale)
    assert layer.weight_scale_2.tolist() == [0.25]
    assert layer.input_scale.tolist() == [0.5]


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_merged_gate_up_slices_each_output_shard(monkeypatch, rank):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 2
    )
    owner = SimpleNamespace(tp_rank=rank, tp_size=2, output_sizes=[8, 12])
    owner.validate_shard_id = MethodType(
        MergedColumnParallelLinear.validate_shard_id, owner
    )
    weight_loader = MethodType(MergedColumnParallelLinear.weight_loader, owner)
    layer = torch.nn.Module()
    method = integration.TensorBridgeNvfp4LinearMethod(
        "model.layers.0.mlp.gate_up_proj"
    )
    method.create_weights(
        layer,
        input_size_per_partition=32,
        output_partition_sizes=[4, 6],
        input_size=32,
        output_size=20,
        params_dtype=torch.bfloat16,
        weight_loader=weight_loader,
    )

    gate_weight = (torch.arange(8 * 16).reshape(8, 16) % 127).to(torch.uint8)
    up_weight = (128 + torch.arange(12 * 16).reshape(12, 16) % 127).to(
        torch.uint8
    )
    gate_scale_raw = (0x39 + torch.arange(8 * 2).reshape(8, 2) % 8).to(torch.uint8)
    up_scale_raw = (0x49 + torch.arange(12 * 2).reshape(12, 2) % 8).to(torch.uint8)
    layer.weight.weight_loader(layer.weight, gate_weight, 0)
    layer.weight.weight_loader(layer.weight, up_weight, 1)
    layer.weight_scale.weight_loader(
        layer.weight_scale,
        gate_scale_raw.view(torch.float8_e4m3fn),
        0,
    )
    layer.weight_scale.weight_loader(
        layer.weight_scale,
        up_scale_raw.view(torch.float8_e4m3fn),
        1,
    )
    for name in ("weight_scale_2", "input_scale"):
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, torch.tensor(0.25), 0)
        parameter.weight_loader(parameter, torch.tensor(0.25), 1)

    expected_weight = torch.cat(
        (gate_weight[rank * 4 : (rank + 1) * 4], up_weight[rank * 6 : (rank + 1) * 6])
    )
    expected_scale = torch.cat(
        (
            gate_scale_raw[rank * 4 : (rank + 1) * 4],
            up_scale_raw[rank * 6 : (rank + 1) * 6],
        )
    )
    assert torch.equal(layer.weight, expected_weight)
    assert torch.equal(layer.weight_scale.view(torch.uint8), expected_scale)
    assert layer.weight_scale_2.tolist() == [0.25, 0.25]
    assert layer.input_scale.tolist() == [0.25, 0.25]


def test_normal_a8_post_load_expands_weight_and_keeps_one_epilogue_scale(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "1.0")
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
    kernel_calls = {}

    class FakeFp8Kernel:
        def process_weights_after_loading(self, layer):
            kernel_calls["processed_layer"] = layer

        def apply_weights(self, layer, x, bias):
            del layer, bias
            return x

    fake_kernel = FakeFp8Kernel()

    def init_kernel(**kwargs):
        kernel_calls["init"] = kwargs
        return fake_kernel

    monkeypatch.setattr(integration, "init_fp8_linear_kernel", init_kernel)
    layer = torch.nn.Module()
    method = integration.TensorBridgeNormalA8LinearMethod(
        "model.layers.0.mlp.gate_up_proj"
    )
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[64, 128],
        input_size=128,
        output_size=192,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.weight.data.fill_(0x22)
    layer.weight_scale.data.view(torch.uint8).fill_(0x39)
    layer.weight_scale_2.data.fill_(0.25)
    layer.input_scale.data.fill_(0.5)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (128, 192)
    assert layer.weight_scale.shape == torch.Size([])
    assert layer.weight_scale.item() == 1.5
    assert not hasattr(layer, "weight_scale_2")
    assert not hasattr(layer, "input_scale")
    assert not hasattr(layer, "locks")
    assert kernel_calls["processed_layer"] is layer
    assert kernel_calls["init"]["activation_quant_key"] == integration.kFp8DynamicTokenSym
    assert kernel_calls["init"]["weight_quant_key"] == integration.kFp8StaticTensorSym
    assert kernel_calls["init"]["weight_shape"] == (192, 128)
    assert kernel_calls["init"]["force_kernel"] is integration.CutlassFP8ScaledMMLinearKernel


def test_normal_a8_rejects_different_fused_global_scales(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        integration,
        "init_fp8_linear_kernel",
        lambda **kwargs: object(),
    )
    layer = torch.nn.Module()
    method = integration.TensorBridgeNormalA8LinearMethod(
        "model.layers.0.mlp.gate_up_proj"
    )
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[64, 64],
        input_size=128,
        output_size=128,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.weight_scale_2.data.copy_(torch.tensor([0.25, 0.5]))

    with pytest.raises(ValueError, match="global scales differ"):
        method.process_weights_after_loading(layer)


def test_unknown_modelopt_layer_algo_is_rejected():
    with pytest.raises(ValueError, match="unsupported ModelOpt"):
        integration._normalize_algo("INT4")


def test_fpma_alpha_default_is_backend_aware(monkeypatch):
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", raising=False)
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", raising=False)
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", raising=False)

    assert integration._fpma_alpha() == 0.961
    assert integration._fpma_alpha(1.0) == 1.0

    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "0.974")
    assert integration._fpma_alpha() == 0.974
    assert integration._fpma_alpha(1.0) == 0.974


def test_tensorbridge_post_load_applies_analytic_alpha_once(monkeypatch):
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", raising=False)
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", raising=False)
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", raising=False)
    monkeypatch.setenv("TENSORBRIDGE_COMPILER", "nvrtc")
    monkeypatch.setenv("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", "")
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP", raising=False)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    layer = torch.nn.Module()
    method = integration.TensorBridgeNvfp4LinearMethod(
        "model.layers.0.mlp.gate_up_proj"
    )
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[256],
        input_size=128,
        output_size=256,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.weight.data.fill_(0x22)
    layer.weight_scale.data.view(torch.uint8).fill_(0x39)
    layer.weight_scale_2.data.fill_(0.25)
    layer.input_scale.data.fill_(0.5)

    method.process_weights_after_loading(layer)

    # Match production's two FP32 operations: apply alpha at load, then FPMA's x6.
    expected = torch.tensor([0.25], dtype=torch.float32) * 0.961
    expected = expected * 6.0
    torch.testing.assert_close(layer.global_scale.detach(), expected, rtol=0.0, atol=0.0)
    assert layer.tensorbridge_fpma_alpha == 0.961


def test_legacy_ulp_defaults_to_neutral_alpha(monkeypatch):
    monkeypatch.delenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", raising=False)
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "1")
    assert integration._fpma_alpha() == 1.0


def test_fpma_compensation_environment_is_strict_and_sets_compile_flag(monkeypatch):
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "1.0")
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none")
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "1")
    monkeypatch.setenv("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", "")

    integration._enforce_production_environment()

    flags = integration.shlex.split(
        integration.os.environ["TENSORBRIDGE_EXTRA_NVRTC_FLAGS"]
    )
    assert integration._PREINT_FLAG in flags
    assert integration._ULP_FLAG in flags
    assert integration._ULP_SCALE_ABI_FLAG in flags

    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "0.974")
    with pytest.raises(ValueError, match="cannot be combined"):
        integration._enforce_production_environment()


def test_fpma_production_environment_rejects_non_nvrtc_compiler(monkeypatch):
    monkeypatch.setenv("TENSORBRIDGE_COMPILER", "nvcc")

    with pytest.raises(RuntimeError, match="require TENSORBRIDGE_COMPILER=nvrtc"):
        integration._enforce_production_environment()


def test_marlin_lm_head_method_uses_scalar_checkpoint_scales(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )

    class Nvfp4Config:
        is_checkpoint_nvfp4_serialized = True
        group_size = 16

    method = integration.TensorBridgeMarlinNvfp4LmHeadMethod(Nvfp4Config())
    layer = torch.nn.Module()

    def weight_loader(*args, **kwargs):
        del args, kwargs

    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[256],
        input_size=128,
        output_size=256,
        params_dtype=torch.bfloat16,
        weight_loader=weight_loader,
    )

    assert layer.params_dtype == torch.bfloat16
    assert layer.weight.shape == (256, 64)
    assert layer.weight_scale.shape == (256, 8)
    assert layer.input_scale.shape == torch.Size([])
    assert layer.weight_scale_2.shape == torch.Size([])
    with pytest.raises(TypeError, match="expected BF16 input"):
        method.apply(layer, torch.empty((1, 128), dtype=torch.float16))


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_marlin_lm_head_slices_vocab_and_zero_pads(monkeypatch, rank):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: 2
    )

    class Nvfp4Config:
        is_checkpoint_nvfp4_serialized = True
        group_size = 16

    method = integration.TensorBridgeMarlinNvfp4LmHeadMethod(Nvfp4Config())

    class QuantConfig:
        def get_quant_method(self, layer, prefix):
            del layer, prefix
            return method

    head = ParallelLMHead(
        num_embeddings=130,
        embedding_dim=128,
        params_dtype=torch.bfloat16,
        org_num_embeddings=130,
        padding_size=64,
        quant_config=QuantConfig(),
        prefix="model.language_model.lm_head",
    )
    full_weight = (torch.arange(130 * 64).reshape(130, 64) % 251).to(torch.uint8)
    full_scale_raw = (0x39 + torch.arange(130 * 8).reshape(130, 8) % 16).to(
        torch.uint8
    )
    head.weight.weight_loader(head.weight, full_weight)
    head.weight_scale.weight_loader(
        head.weight_scale,
        full_scale_raw.view(torch.float8_e4m3fn),
    )
    head.weight_scale_2.weight_loader(head.weight_scale_2, torch.tensor(0.25))
    head.input_scale.weight_loader(head.input_scale, torch.tensor(0.5))

    start = rank * 96
    stop = min(start + 96, 130)
    loaded_rows = stop - start
    assert torch.equal(head.weight[:loaded_rows], full_weight[start:stop])
    assert torch.count_nonzero(head.weight[loaded_rows:]).item() == 0
    assert torch.equal(
        head.weight_scale[:loaded_rows].view(torch.uint8),
        full_scale_raw[start:stop],
    )
    assert torch.count_nonzero(
        head.weight_scale[loaded_rows:].view(torch.uint8)
    ).item() == 0
    assert head.weight_scale_2.shape == torch.Size([])
    assert head.input_scale.shape == torch.Size([])
    assert head.weight_scale_2.item() == 0.25
    assert head.input_scale.item() == 0.5


def test_qwen_adapter_streams_nvfp4_lm_head_to_marlin(monkeypatch):
    model = TensorBridgeQwen3_5ForConditionalGeneration.__new__(
        TensorBridgeQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.language_model = torch.nn.Module()
    model.language_model.lm_head = torch.nn.Module()

    head_tensors = {
        "weight": torch.tensor([[0x12, 0x34]], dtype=torch.uint8),
        "weight_scale": torch.tensor([[1.0]], dtype=torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
        "input_scale": torch.tensor(0.5, dtype=torch.float32),
    }
    loader_calls = {}

    def make_loader(parameter_name):
        def weight_loader(parameter, loaded_weight):
            assert parameter is getattr(model.language_model.lm_head, parameter_name)
            loader_calls[parameter_name] = loaded_weight.clone()

        return weight_loader

    for name, tensor in head_tensors.items():
        parameter = torch.nn.Parameter(torch.empty_like(tensor), requires_grad=False)
        parameter.weight_loader = make_loader(name)
        model.language_model.lm_head.register_parameter(name, parameter)

    passthrough = []

    def parent_load_weights(self, weights):
        assert self is model
        passthrough.extend(weights)
        return {name for name, _ in passthrough}

    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "load_weights",
        parent_load_weights,
    )
    ordinary = [
        ("model.visual.patch_embed.weight", torch.tensor([1.0])),
        ("model.language_model.layers.0.input_layernorm.weight", torch.tensor([2.0])),
    ]
    interleaved = [
        ordinary[0],
        ("lm_head.weight", head_tensors["weight"]),
        ("lm_head.weight_scale", head_tensors["weight_scale"]),
        ordinary[1],
        ("lm_head.weight_scale_2", head_tensors["weight_scale_2"]),
        ("lm_head.input_scale", head_tensors["input_scale"]),
    ]
    yielded = []

    def source():
        for item in interleaved:
            yielded.append(item[0])
            yield item

    loaded = model.load_weights(source())

    assert yielded == [name for name, _ in interleaved]
    assert [name for name, _ in passthrough] == [name for name, _ in ordinary]
    for name, expected in head_tensors.items():
        assert torch.equal(loader_calls[name], expected)
    assert loaded == {
        *(name for name, _ in ordinary),
        *(f"language_model.lm_head.{name}" for name in head_tensors),
    }


def test_triton_cache_is_process_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TRITON_CACHE_BASE", str(tmp_path))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    integration._isolate_triton_cache()

    resolved = tmp_path / f"pid_{integration.os.getpid()}"
    assert integration.os.environ["TRITON_CACHE_DIR"] == str(resolved)
    assert resolved.is_dir()


def test_zero_groups_below_prefold_floor_drops_group_and_scale():
    # Two groups of 16 nibble-weights: one legal scale, one under the 0x1C floor.
    scales = torch.tensor([0x40, 0x10], dtype=torch.uint8).view(torch.float8_e4m3fn)
    weights = torch.tensor([0x11111111, 0x22222222, 0x33333333, 0x44444444],
                           dtype=torch.int32)
    dropped = integration._zero_groups_below_prefold_floor(weights, scales)

    assert dropped == 1
    raw = scales.view(torch.uint8)
    assert raw.tolist() == [0x40, 0]
    # The legal group is untouched; the dropped group loses its weights too,
    # because a zero scale alone still emits `0 + addend` in the mainloop.
    assert weights.tolist() == [0x11111111, 0x22222222, 0, 0]


def test_zero_groups_below_prefold_floor_is_a_noop_in_domain():
    scales = torch.tensor([0x39, 0x7E, 0x1C], dtype=torch.uint8).view(torch.float8_e4m3fn)
    weights = torch.arange(1, 7, dtype=torch.int32)
    before = weights.clone()

    assert integration._zero_groups_below_prefold_floor(weights, scales) == 0
    assert torch.equal(weights, before)
    assert scales.view(torch.uint8).tolist() == [0x39, 0x7E, 0x1C]


def test_zero_groups_below_prefold_floor_keeps_true_zero_groups():
    # raw == 0 is the kernel's existing all-zero convention, not an underflow.
    scales = torch.tensor([0x00, 0x50], dtype=torch.uint8).view(torch.float8_e4m3fn)
    weights = torch.tensor([0, 0, 0x55555555, 0x66666666], dtype=torch.int32)

    assert integration._zero_groups_below_prefold_floor(weights, scales) == 0
    assert weights.tolist() == [0, 0, 0x55555555, 0x66666666]
