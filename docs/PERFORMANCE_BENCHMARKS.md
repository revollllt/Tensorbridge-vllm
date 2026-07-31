# TensorBridge Performance Benchmarks

Last updated: 2026-07-31

This page records the performance harnesses added alongside the existing
accuracy evaluation, the exact workload each one drives, and the results
produced so far. It is deliberately separate from
`docs/ACCURACY_EVALUATION.md`: none of the probes here use real text or check
model output, so nothing on this page is accuracy evidence.

## Why these are separate from the PPL harness

`scripts/eval_nvfp4_wikitext2_ppl.py` is an accuracy tool. Its cross-repeat gate
raises `ExecutionValidationError: repeated execution changed greedy generated
token IDs` whenever two repeats diverge, and TensorBridge's own contract
declares `streamk_mode = auto_router_multi_slice_nondeterministic`. Over a long
generation that gate is structurally unsatisfiable, so a decode-dominated timing
probe cannot be built on it (job `455277` failed exactly this way at 512 output
tokens). The latency harnesses below use `vllm bench latency` semantics instead:
no output comparison, fixed length via `ignore_eos`.

## Which kernel is the baseline

On Hopper `init_nvfp4_linear_kernel()` resolves NVFP4 to **Marlin (W4A16)**:
`FlashInferCutlassNvFp4LinearKernel` requires `sm_100`, and
`CutlassNvFp4LinearKernel` requires `cutlass_scaled_mm_supports_fp4()`, also
`sm_100`. `vllm/plugins/tensorbridge.py:119` forces
`VLLM_NVFP4_GEMM_BACKEND=marlin`, which on this hardware equals what auto
selection would pick anyway — but it would silently downgrade the baseline on
Blackwell.

`TENSORBRIDGE_VLLM_BACKEND` (`vllm/plugins/tensorbridge.py:547`) selects
`tensorbridge` / `normal_a8` / `official` at weight-load time. Backends
therefore need separate processes; batch size does not.

`lm_head` stays Marlin W4A16 in every arm, because the `lm_head` branch in
`get_quant_method` precedes the backend branch. Comparisons below isolate the
transformer linear layers, not the whole model.

## Harnesses

### Kernel level

| File | What it measures |
| --- | --- |
| `benchmarks/bench_nvfp4_kernel_baselines.py` | Five labels timed in one process, palindrome-interleaved |
| `sbatch/run_bench_nvfp4_kernel_baselines.sbatch` | Slurm wrapper |
| `benchmarks/bench_activation_quant.py` | TensorBridge vs vLLM FP8 activation quantiser: equivalence and cost |
| `sbatch/run_bench_activation_quant.sbatch` | Slurm wrapper |
| `benchmarks/shapes/crossover_shapes.csv` | 10 `(N,K)` pairs x M in {16, 32, 64, 128} |

Labels: `tensorbridge_nvfp4a8` (W4A8), `tensorbridge_vllm_quant` (same GEMM, vLLM
quantiser), `cutlass_w4a8`, `marlin_nvfp4_w4a16` (W4A16), `cutlass_fp8_w8a8`.
They are **not numerically equivalent** — they are the precision points
reachable on this hardware, measured on identical shapes.

Timing primitives are imported from the TensorBridge kernel repo
(`palindrome_labels`, `time_event`, `summarize`), so numbers compose with
`bench_nvfp4_optimal_vs_cutlass.py`. `BENCH_INCLUDE_ACTIVATION_QUANT=1` moves
the BF16->FP8 activation quantisation inside the timed region for the A8 labels;
Marlin is W4A16 and has no such step, so it is unaffected and acts as a control.

The kernel under test is whatever `TENSORBRIDGE_ROOT` points at. It defaults to
`/data/user/jzou521/codes/cuda/tensorbridge-pinned`, a `git worktree` pinned to
`43cc2aa` (the wheel commit). A clean worktree is required, not optional:
`bench_nvfp4_common.py:29` does `sys.path.insert(0, REPO_ROOT)`, so running from
the sibling repo imports its working tree regardless of which venv is active.

### End to end

| File | What it measures |
| --- | --- |
| `benchmarks/bench_e2e_batch_sweep.py` | One engine, whole batch sweep, per-batch latency |
| `sbatch/run_bench_e2e_batch_sweep.sbatch` | Slurm wrapper (preferred) |
| `sbatch/run_bench_e2e_latency_backends.sbatch` | One `vllm bench latency` per position (superseded) |
| `sbatch/run_profile_e2e_quant_share.sbatch` | Nsight Systems over the decode graph, per-kernel GPU time |
| `scripts/annotate_e2e_latency.py` | Attaches provenance to `vllm bench latency` output |
| `scripts/analyze_perf_baseline.py` | Fairness gates, noise floor, position-paired speedup |

The profiling wrapper sets a short node-local `TMPDIR`. vLLM puts its ZMQ IPC
socket there, and a Unix domain socket path is capped at 107 characters; a
repo-relative temp directory plus a UUID overruns it and the engine dies before
profiling starts.

The sweep version exists because the per-position version paid one engine start
per `(position, batch)` pair. Six batch sizes across six nodes meant sixty cold
21 GB checkpoint reads contending on shared storage; observed engine start rose
from 78 s to 1588 s. One engine can capture a CUDA graph per batch size, so the
sweep costs one start per palindrome position and stays on one node where the
checkpoint is served from page cache.

`bench_e2e_batch_sweep.py` fails closed if any requested batch size is missing
from `resolved_cudagraph_capture_sizes`: an uncaptured size falls back to eager
and would read as an unexplained slowdown at that batch only.

## Workload

Both end-to-end harnesses drive the same probe.

```text
model            Qwen3.6-27B-NVFP4, TP=1, bfloat16, gpu_memory_utilization 0.85
prompts          batch_size x 128 random token ids (np.random.randint(10000))
generation       256 tokens, ignore_eos=True, temperature=1.0, detokenize off
max_model_len    512
compilation      mode=NONE, cudagraph_mode=FULL_DECODE_ONLY,
                 cudagraph_capture_sizes=[each batch size]
sampling         3 warmup iterations discarded, 12 measured
metric           wall clock of one llm.generate() = prefill + 255 decode steps
```

`--dtype bfloat16` is pinned rather than left to `auto` because this checkpoint
declares no `torch_dtype`, so `auto` would resolve from vLLM's fallback instead
of from the checkpoint. It also matches every existing accuracy config, and it
is the dtype Marlin dequantises into.

`ignore_eos` keeps all sequences the same length, which keeps the decode batch
pinned at the captured graph size.

Limitations that must travel with any number from these probes:

- Not a serving benchmark. No arrival process, no continuous batching, no
  queueing, no TTFT/ITL split.
- `max_model_len=512` makes KV cache negligible, so attention is a smaller
  share than in real serving. This flatters the linear-layer comparison.
- Prompts are random token ids: shape-correct, semantically meaningless.
- Prefill scales with batch, so the large-batch points are not pure decode.

## Results

### Kernel level, activation quantisation included (`nvfp4_crossover_withquant.json`)

Geometric-mean gap over CI95-resolvable shapes; negative means TensorBridge is
faster.

| Baseline | M=16 | M=32 | M=64 | M=128 |
| --- | ---: | ---: | ---: | ---: |
| Marlin (W4A16) | +55.1% | +36.1% | +2.8% | -32.4% |
| CUTLASS W4A8 | +20.1% | +18.5% | +16.3% | +12.9% |
| CUTLASS FP8 | +12.2% | +9.9% | +14.2% | +12.7% |

With the vLLM quantiser feeding the same TensorBridge GEMM:

| Baseline | M=16 | M=32 | M=64 | M=128 |
| --- | ---: | ---: | ---: | ---: |
| Marlin (W4A16) | +14.5% | +3.1% | -21.2% | -49.7% |
| CUTLASS W4A8 | -14.0% | -14.5% | -13.7% | -11.2% |
| CUTLASS FP8 | -17.2% | -18.6% | -13.7% | -10.3% |

Excluding activation quantisation — which the upstream kernel benchmark does,
since `bench_nvfp4_common.py:553` quantises once during setup — overstates the
A8 paths. Marlin needs no quantisation at all, so it is the only label whose
GEMM-only number was already complete. At M=16 the sign flips: -17.0% becomes
+55.1%.

### Activation quantiser comparison (`activation_quant_455676.json`)

Interchangeable: scale ratio 1.00000000-1.00000012 across 28 shapes, and the
maximum error against the unquantised input matches to three significant figures
on every shape. Codes differ bitwise on 23/28 shapes, but only by rounding
tie-breaks — neither is more accurate.

Cost: TensorBridge is flat at 26-30 us for M <= 512 regardless of M and K, while
vLLM is ~10 us; at M=4096 TensorBridge becomes faster (0.61x at K=28672). A cost
independent of problem size is overhead, not work — consistent with the Triton
launch path. `tensorbridge/ops/input.py` launches one program per token with
`BLOCK = next_power_of_2(K)`.

### End to end batch sweep (`sweep_455790`, ACD1-13, clocks unpinned)

| batch | official s | TensorBridge s | paired speedup | null floor | verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3.574 | 3.593 | 0.995x | 1.003 | reportable, negligible |
| 4 | 3.959 | 3.851 | 1.029x | 1.076 | inside noise floor |
| 16 | 4.653 | 4.680 | 0.995x | 1.060 | inside noise floor |
| 32 | 5.721 | 5.420 | 1.056x | 1.050 | reportable |
| 64 | 8.123 | 6.880 | 1.181x | 1.038 | reportable |
| 128 | 13.034 | 9.952 | 1.310x | 1.027 | reportable |

The null floor is the worst same-arm spread at that batch, taken from the two
mirrored positions of the palindrome. `official` reproduces to 0.03% at batch
128; TensorBridge's two runs differ by 3-7%, and that is what puts batch 4 and
16 below the reporting bar.

### What the two layers say together

The large-batch prediction holds: the kernel benchmark said -32.4% versus Marlin
at M=128 and the engine delivered 1.310x at batch 128. The small-batch
prediction does not: the kernel benchmark said TensorBridge is 55% slower at
M=16, and the engine measured parity at batch 16.

The difference is CUDA Graphs. The kernel harness launches every call from the
host, so it charges TensorBridge's Triton quantiser its full launch overhead.
`FULL_DECODE_ONLY` replay issues the captured graph instead, so that overhead
does not recur. The measured crossover is therefore between batch 16 and 32,
earlier than the M=64 the kernel numbers imply.

This downgrades the quantiser swap: most of the 20-26% it saves in the kernel
harness is launch overhead the production path does not pay. The profiling
section below measures the in-graph cost directly and closes the question.

### In-graph GPU time (`sbatch/run_profile_e2e_quant_share.sbatch`)

Nsight Systems over the decode graph, TensorBridge arm, `--cuda-graph-trace=node`
(graph-level tracing reports one entry per graph launch and hides everything
inside, so the flag is load-bearing).

| Group | bs=16 GPU % | us/call | bs=128 GPU % | us/call |
| --- | ---: | ---: | ---: | ---: |
| TensorBridge NVFP4 GEMM | 27.30 | 46.99 | 21.96 | 75.45 |
| CUTLASS (FP8 layers) | 17.50 | 23.92 | 14.05 | 38.31 |
| gated delta rule (linear attention) | 7.00 | 32.56 | **29.10** | 270.25 |
| elementwise | 31.43 | 3.54 | 21.59 | 4.86 |
| reduce | 7.49 | 7.81 | 4.24 | 8.81 |
| **TensorBridge activation quant** | **1.32** | **2.28** | **1.10** | **3.78** |
| vLLM FP8 quant (FP8 layers) | 1.57 | 2.40 | 1.02 | 3.10 |

**The quantiser swap is worthless and is not being pursued.** On the GPU
TensorBridge's Triton quantiser takes 2.28 us per call against vLLM's 2.34 us —
TensorBridge is marginally faster, and the whole step is 1.1-1.3% of GPU time.

| | kernel-harness wall | in-graph GPU | difference |
| --- | ---: | ---: | ---: |
| TensorBridge `quant_input` | 26.8 us | 2.28 us | 24.5 us launch overhead |
| vLLM `scaled_fp8_quant` | 10.4 us | 2.34 us | 8.0 us launch overhead |

This also bounds what `benchmarks/bench_activation_quant.py` can be used for: it
times wall clock, so it charges each quantiser its host launch path. That is the
right question for an eager forward and the wrong one for a captured graph. Its
equivalence check remains valid; its cost ranking does not transfer to
production.

### The large-batch bottleneck is not the linear layers

`fused_recurrent_gated_delta_rule_packed_decode` grows from 7.00% to 29.10% of
GPU time between batch 16 and 128, becoming the single largest item — larger
than the NVFP4 GEMM it is being compared against. Per-call time grows 8.3x for
an 8x batch increase, i.e. no batching amortisation at all.

That is inherent, not a bug. Qwen3.6 has 64 layers on a `full_attention_interval`
of 4, so 48 are linear attention, and each keeps a per-sequence recurrent state
of `(num_v_heads, head_v_dim, head_k_dim) = (48, 128, 128)`. With
`mamba_cache_dtype=auto` resolving to the model dtype, that is bfloat16:

```text
state per sequence per layer   48 * 128 * 128 * 2 B  = 1.57 MB
read + write per decode step   3.15 MB per sequence per layer
batch 16                       50.3 MB  -> 15.0 us at 3.35 TB/s   (measured 32.56 us)
batch 128                      403 MB   -> 120 us  at 3.35 TB/s   (measured 270.25 us)
```

Every decode step must read and write each sequence's state, so the traffic is
linear in batch and no amount of batching amortises it. The kernel sustains
about 45% of HBM peak at both batch sizes, which leaves roughly 2x of
implementation headroom (`grid = (NV, B * HV)` with `num_warps=1` and `BV=32`
gives each block one warp over 32 of the 128 V dims), but not more.

Consequences for this project:

- It is identical in both arms, so it dilutes every speedup measured end to end.
  At batch 128 it caps the achievable ratio near `1 / (1 - 0.22) ~ 1.28x` from
  the NVFP4 GEMM alone; the measured 1.310x already includes the FP8-layer
  difference, so the GEMM path has little headroom left.
- It lives in vLLM's model implementation, not in TensorBridge. Whether this
  project should own it is a scope question, not a technical one.

## Reproduction

Kernel, five labels with activation quantisation:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
BENCH_INCLUDE_ACTIVATION_QUANT=1 \
BENCH_SHAPE_FILE=$PWD/benchmarks/shapes/crossover_shapes.csv \
BENCH_OUTPUT=$PWD/benchmarks/results/kernel/crossover.json \
sbatch -w <node> --export=ALL sbatch/run_bench_nvfp4_kernel_baselines.sbatch
```

End-to-end batch sweep:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
E2E_BATCH_SIZES=1,4,16,32,64,128 \
sbatch -w <node> --export=ALL sbatch/run_bench_e2e_batch_sweep.sbatch
```

Analysis of a per-position latency cohort:

```bash
.venv/bin/python scripts/analyze_perf_baseline.py <result_dir> \
    --baseline official --warmup-repeats 0
```

## Open items

Ranked by expected value after the profiling above.

1. **The M>=512 router config cliff.** TensorBridge goes from -15.7% to +8.3%
   against CUTLASS W4A8 between M=128 and M=512, on nearly every `(N,K)` pair
   and in the same direction. That is real GPU work, so it transfers end to end,
   and it is TensorBridge's own code.
2. **Elementwise fragmentation.** 31% of GPU time at batch 16 across 400k+ calls
   averaging 3.5 us. Large in aggregate, but a vLLM fusion question.
3. **Gated delta rule occupancy.** ~2x of roofline headroom at the dominant
   large-batch cost, but it is vLLM's model code and the scaling itself is
   inherent.
4. ~~Swap TensorBridge's activation quantiser for vLLM's.~~ Closed: 1.1-1.3% of
   GPU time, and TensorBridge's kernel is already the faster of the two on the
   GPU.

## Standing caveats

- GPU clock pinning has never succeeded on these nodes (`nvidia-smi -lgc` is not
  permitted); every wrapper records `gpu_clock_pin_status` and continues.
- No node in this cluster is reliably idle, and `--exclusive` queues
  indefinitely. Runs pin a single low-load node instead.
- Cluster policy caps CPUs at 12 per GPU.
- The sibling kernel repo working tree carries uncommitted changes. All kernel
  results here come from the pinned worktree, not that tree.
