from tensorbridge.schema.awq import AWQWeightSchema
from tensorbridge.schema.base import BaseInputSchema, BaseWeightSchema
from tensorbridge.schema.bitnet import BitnetWeightSchema
from tensorbridge.schema.compressed_tensors import (
    CompressedTensorsInputSchema,
    CompressedTensorsWeightSchema,
)
from tensorbridge.schema.fp8 import Fp8InputSchema, Fp8WeightSchema
from tensorbridge.schema.gpt_oss_mxfp4 import GptOssMxfp4WeightSchema
from tensorbridge.schema.gptq import GPTQWeightSchema
from tensorbridge.schema.tensorbridge import TensorBridgeInputSchema, TensorBridgeWeightSchema
from tensorbridge.schema.modelopt import ModeloptInputSchema, ModeloptWeightSchema
from tensorbridge.schema.mxfp4 import Mxfp4WeightSchema

WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "awq": AWQWeightSchema,
    "bitnet": BitnetWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
    "fp8": Fp8WeightSchema,
    "gptq": GPTQWeightSchema,
    "tensorbridge": TensorBridgeWeightSchema,
    "modelopt": ModeloptWeightSchema,
    "mxfp4": Mxfp4WeightSchema,
    "gpt_oss_mxfp4": GptOssMxfp4WeightSchema,
}

INPUT_SCHEMA_MAP: dict[str, type[BaseInputSchema]] = {
    "compressed-tensors": CompressedTensorsInputSchema,
    "fp8": Fp8InputSchema,
    "tensorbridge": TensorBridgeInputSchema,
    "modelopt": ModeloptInputSchema,
}

BaseWeightSchema.WEIGHT_SCHEMA_MAP = WEIGHT_SCHEMA_MAP
BaseInputSchema.INPUT_SCHEMA_MAP = INPUT_SCHEMA_MAP


__all__ = [
    "BaseInputSchema",
    "BaseWeightSchema",
    "TensorBridgeInputSchema",
    "TensorBridgeWeightSchema",
]
