# Third-Party Provenance

This document separates TensorBridge's project identity from the upstream and
third-party work present in the repository. Renaming an inherited interface does
not remove its provenance or change the license attached to third-party code.

## Humming

TensorBridge began as a derivative of the open-source Humming CUDA kernel stack.
The package, runtime, benchmark, and kernel interfaces maintained here use the
TensorBridge name. Humming is referenced only to acknowledge that origin; the
TensorBridge project is not Humming and is not represented as an upstream
continuation.

The current repository does not contain a verified upstream repository URL or
revision for Humming. This file therefore does not assert one. A verified source
locator should be added before public redistribution rather than substituting an
unverified URL or commit.

## NVIDIA CUTLASS And CuTe

The experimental `cute_cutlass_nvfp4a8/` tree contains copied or adapted NVIDIA
CUTLASS and CuTe reference sources. In particular, files under
`cute_cutlass_nvfp4a8/cutlass_examples/` and
`cute_cutlass_nvfp4a8/include/cutlass/` retain their original NVIDIA copyright
and license headers. Those notices must remain unchanged.

## vLLM

The benchmark harness can load the vLLM-provided CUTLASS W4A8 operator as an
external comparison target. That operator is not part of TensorBridge.

The repository-level `LICENSE` applies subject to all retained third-party
notices and license terms in the corresponding source files.
