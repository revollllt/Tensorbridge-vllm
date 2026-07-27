"""Qwen3.5 compatibility model ensuring a quantized ParallelLMHead."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.utils import maybe_prefix


logger = logging.getLogger(__name__)
_LM_HEAD_CHECKPOINT_PARAMETERS = frozenset(
    {"weight", "weight_scale", "weight_scale_2", "input_scale"}
)


class TensorBridgeQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Keep the checkpoint's NVFP4 lm_head on the official W4A16 Marlin path."""

    def __init__(self, *, vllm_config: Any, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        language_model = self.language_model
        lm_head = language_model.lm_head
        if not isinstance(lm_head, ParallelLMHead):
            return

        config = vllm_config.model_config.hf_text_config
        if config.tie_word_embeddings:
            return
        if self._is_nvfp4_marlin_head(lm_head):
            return

        lm_head_prefix = maybe_prefix(maybe_prefix(prefix, "language_model"), "lm_head")
        language_model.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=vllm_config.quant_config,
            prefix=lm_head_prefix,
        )
        if not self._is_nvfp4_marlin_head(language_model.lm_head):
            raise RuntimeError("Qwen3.5 lm_head did not select NVFP4 Marlin W4A16")

    @staticmethod
    def _is_nvfp4_marlin_head(lm_head: ParallelLMHead) -> bool:
        parameters = {
            name: getattr(lm_head, name, None) for name in _LM_HEAD_CHECKPOINT_PARAMETERS
        }
        quant_method = getattr(lm_head, "quant_method", None)
        return (
            all(isinstance(parameter, nn.Parameter) for parameter in parameters.values())
            and parameters["weight"].dtype == torch.uint8
            and parameters["weight_scale"].dtype == torch.float8_e4m3fn
            and parameters["input_scale"].shape == torch.Size([])
            and parameters["weight_scale_2"].shape == torch.Size([])
            and getattr(quant_method, "backend", None) == "marlin"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        lm_head = self.language_model.lm_head
        expected = {
            name
            for name in _LM_HEAD_CHECKPOINT_PARAMETERS
            if isinstance(getattr(lm_head, name, None), nn.Parameter)
        }
        if expected != _LM_HEAD_CHECKPOINT_PARAMETERS:
            raise RuntimeError(f"NVFP4 Marlin lm_head parameters are incomplete: {expected}")
        loaded: set[str] = set()

        def without_lm_head():
            for name, weight in weights:
                if not name.startswith("lm_head."):
                    yield name, weight
                    continue

                parameter_name = name.removeprefix("lm_head.")
                if parameter_name not in _LM_HEAD_CHECKPOINT_PARAMETERS:
                    raise ValueError(f"unsupported Qwen3.5 lm_head tensor: {name}")
                if parameter_name in loaded:
                    raise ValueError(f"duplicate Qwen3.5 lm_head tensor: {name}")
                parameter = getattr(lm_head, parameter_name)
                weight_loader = getattr(parameter, "weight_loader", None)
                if not callable(weight_loader):
                    raise TypeError(f"lm_head parameter has no weight loader: {name}")
                weight_loader(parameter, weight)
                loaded.add(parameter_name)

        loaded_weights = set(super().load_weights(without_lm_head()))
        missing = expected - loaded
        if missing:
            raise ValueError(f"checkpoint is missing lm_head tensors: {sorted(missing)}")

        loaded_lm_head = {f"language_model.lm_head.{name}" for name in loaded}
        logger.info("TensorBridge loaded NVFP4 Marlin lm_head: %s", sorted(loaded_lm_head))
        return loaded_weights | loaded_lm_head
