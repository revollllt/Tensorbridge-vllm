# TensorBridge: NVFP4A8 on Hopper

TensorBridge is a CUDA kernel stack for serving new-generation NVFP4 checkpoints
on Hopper GPUs. It keeps the checkpoint's compact NVFP4
weight bandwidth while converting FP4 values and scales on the fly into the
FP8 WGMMA compute path that Hopper can saturate. The current target is LLM
serving with FP8 E4M3 activations and compact NVFP4 weights:

```text
A: fp8 e4m3 activations, optional per-token scale
B: fp4 e2m1 weights, group size 16
S: fp8 e4m3 weight scale, one byte per 16 weights
storage: 4 bits/weight + 8-bit scale/16 = 4.5 bits/weight
```

SNC is mandatory for the accepted benchmark path. The compact production layout is
`nvfp4_swizzle64_raw` with the dual-MMA preinterleaved host layout enabled.

## Status

This is a research implementation: the main kernel idea, correctness guards,
and paper-oriented performance paths are present, but packaging, cross-SKU
hardware profiles, serving-framework integration, and production CI still need
engineering work. The hand-written TensorBridge path is the production candidate; the
CuTe/CUTLASS rewrite remains isolated and experimental.

## Current Production Path

The active path is a single BN128 interleave kernel family, not two independent
non-StreamK and StreamK backends.

```text
layout                = nvfp4_swizzle64_raw
host layout env       = TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT=1
device layout flag    = -DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1
BlockN / WarpN        = 128 / 16
BlockK / WarpK        = 128 / 128
weight group size     = 16
SNC                   = enabled
```

`use_stream_k` means "enable StreamK split-K only for the residual tail". The
same launch first covers full data-parallel MN waves; when the shape underfills
the last wave enough, the router lets StreamK split the tail over K. The tail
choice is shape-only and CUDA-graph friendly.

The workload router and exact post-route islands live in:

- `tensorbridge/tune/backend_router.py`
- `tensorbridge/tune/sm90.py`
- `tensorbridge/tune/plan_cache.py`

For CUDA graph integration, resolve the route once per layer and M bucket with
`build_heuristics_plan_table(...)`, then use `plan_table.get_config_json(M)` on
the hot path.  On the paper 40-shape set this reduces route lookup from about
`18.6 us` for the full Python heuristic to about `0.25 us` for the plan table.
Random prefill shapes that miss the table use the C++ router by default on the
production NVFP4/SNC path (`TENSORBRIDGE_NVFP4_CPP_ROUTER=0` disables it); on the full
410-shape set it matches the Python reference scheduler with `0` mismatches and
cuts `Sm90Heuristics.get_config` from about `19.2 us` to about `7.1 us`.

The graph-safe aggregate dequant variant is controlled by one tuning bit:

```text
nvfp4_swz64_prebcast_prmt_const_variant
  -> TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
  -> TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR
  -> TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED
```

## Provenance

TensorBridge began as a derivative of the open-source Humming kernel stack.
Humming remains credited by its original name, but TensorBridge is a separate
research implementation and is not presented as Humming or as an upstream
continuation. TensorBridge-specific work includes the Hopper NVFP4A8 path, the
compact NVFP4-to-FP8 bridge, the large-shape mainloop, and the workload-aware
router described in this repository.

The current repository does not record a verified Humming upstream URL or
revision, so this documentation intentionally does not assert one. See
`THIRD_PARTY.md` for the provenance boundary and the notices that must remain
with copied third-party sources.

## Repository Layout

```text
tensorbridge/           production-candidate Python/JIT glue and CUDA kernels
cute_cutlass_nvfp4a8/   isolated experimental CuTe/CUTLASS rewrite
scripts/                benchmark, profiling, and shared helpers
sbatch/                 reproducible H100 build/test/benchmark wrappers
tests/                  CPU references and Hopper correctness guards
benchmarks/             shape lists, results, and profiler exports
docs/                   current design and benchmark notes
```

## Serving Integration Boundary

This repository is the framework-independent TensorBridge kernel runtime. Its
versioned consumer interface is `tensorbridge.api.v1`; serving integrations must
not import private compiler, router, schema, or launcher modules directly.

The vLLM ModelOpt adapter, Qwen3.6 model compatibility, quantized `lm_head`
routing, model-level evaluation, tensor parallel tests, and CUDA Graph tests live
in the separate `tensorbridge-vllm` repository. The dependency is one-way:
`tensorbridge-vllm` installs a pinned `tensorbridge-kernels` wheel, while this
repository does not import or install vLLM.

Build the kernel-only artifact from a committed revision with the dedicated
environment:

```bash
uv build --wheel --no-build-isolation \
  --python .venv-kernel/bin/python --out-dir dist .
```

The build metadata derives `0.2.0+g<sha>` from the current TensorBridge commit.
After a kernel update, publish a new wheel and explicitly advance the pin in the
separate integration repository; never make its `.venv` an editable install of
this source tree.

Important docs:

- `docs/BRANCH_INTEGRATION.md`: source-line provenance and promotion gates
- `docs/BENCHMARKING.md`: full 410-shape benchmark and ablation commands
- `docs/NVFP4_SNC.md`: current SNC contract
- `docs/workload-aware-backend/README.md`: router and production schedule summary
- `docs/workload-aware-backend/COST_MODEL.md`: shape-only StreamK-tail model
- `docs/workload-aware-backend/NONSTREAMK_COMPUTE_BOUND_GAP.md`: current CUTLASS gap evidence
- `cute_cutlass_nvfp4a8/README.md`: scope and build contract for the experimental rewrite
- `THIRD_PARTY.md`: upstream provenance and retained third-party notices

## Full 410-Shape Benchmark

The accepted model-weight benchmark uses the BN128-aligned shape file and excludes
DeepSeek-V3 `kv_a_proj_with_mqa` because its `N=576` is not BN128-compatible.

```bash
BENCH_SHAPE_FILE=benchmarks/shapes/model_weight_shapes_no_lm_router_bn128.csv \
BENCH_ROUNDS=6 BENCH_PRIME=8 BENCH_WARMUP=4 BENCH_ITERS=40 \
sbatch --array=0-7 sbatch/run_bench_nvfp4_optimal_vs_cutlass.sbatch
```

See `docs/BENCHMARKING.md` for result collection, summary generation, and
the separate ablation wrapper.
