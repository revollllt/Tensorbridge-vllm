# TensorBridge vLLM Integration

This repository owns the vLLM-facing part of TensorBridge: ModelOpt checkpoint
loading, Qwen3.6 compatibility, Marlin W4A16 `lm_head`, tensor parallelism,
CUDA Graph execution, and model-level evaluation. CUDA kernels remain in the
separate sibling `tensorbridge` repository.

The dependency is intentionally one-way. Integration code imports only
`tensorbridge.api.v1` and consumes the wheel pinned by
`constraints/tensorbridge.json`. It must not import TensorBridge compiler,
schema, router, or launcher internals.

The normal update flow is:

1. Build a versioned wheel in `../tensorbridge/dist/`.
2. Update the version, source digest, filename, and SHA256 in the constraint.
3. Install that wheel into this repository's `.venv`.
4. Run `scripts/verify_tensorbridge_constraint.py`.
5. Run adapter pytest and the H100 TP/CUDA Graph gates before accepting the pin.

An editable TensorBridge install is allowed only in a disposable joint-debug
environment. The checked integration environment always uses the pinned wheel,
so kernel feature branches cannot silently change vLLM results.

The supplied Slurm wrappers default `VLLM_USE_DEEP_GEMM=0`. The vLLM v0.20.2
vendored namespace can be importable while lacking APIs required by its FP8
warmup path, whereas this checkpoint's FP8 layers select the CUTLASS backend.
An environment with a verified compatible DeepGEMM installation may explicitly
override the default.
