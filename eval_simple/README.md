# Simple FPMA accuracy evaluation

Four small scripts that measure whether TensorBridge's FPMA approximate
dequantization costs accuracy, on WikiText-2 perplexity and full GSM8K.

```
arms.py                 which kernel runs, expressed as environment variables
eval_ppl.py             WikiText-2 perplexity for one arm
eval_gsm8k.py           full GSM8K exact-match for one arm
compare.py              summarise a results directory, paired GSM8K test
gsm8k_final_answer.yaml lm-eval task: dataset revision and answer filter
```

Each script runs one arm and writes one JSON file. There is no scheduler, no
manifest, no protocol machinery — run them three times, then run `compare.py`.

## What the arms are

| Arm | MLP path |
| --- | --- |
| `official` | NVFP4 weights, Marlin W4A16, BF16 activation |
| `normal_a8` | Pre-deuqantize NVFP4 to FP8, and use 8-bit expanded weight exactly at load, Cutlass FP8, dynamic per-token FP8 activation |
| `alpha_0961` | FPMA-SNC generating B8 in the mainloop, same activation path, `alpha=0.961` |
| `fpma_default` | the same FPMA kernel with compensation off (`alpha=1.0`) |

**Compare `alpha_0961` against `normal_a8`, not against `official`.** Those two
share the B8 weights, the activation path, and the epilogue scale; they differ
only in whether B8 is expanded exactly at load or approximated in the mainloop.
That difference is FPMA. `official` also changes the activation dtype, so a
comparison against it answers a different question.

Only the 192 transformer MLP projections vary. `lm_head` stays NVFP4 Marlin
W4A16 with a BF16 activation in every arm, and the 208 FP8 layers are identical.

## Confirming TensorBridge actually ran

`TENSORBRIDGE_VLLM_BACKEND` states an intent. Four things turn it into evidence,
and three of them are automatic:

1. **The result JSON records `quant_config_class`.** It must read
   `vllm.plugins.tensorbridge.TensorBridgeModelOptMixedConfig`. Both scripts call
   `arms.confirm_active()` before building the engine and abort if `modelopt_mixed`
   resolves to vLLM's own class instead. This is the check that matters most,
   because an unregistered plugin is the only failure here that yields *plausible*
   numbers rather than an error — vLLM would fall back to its own kernels and every
   arm would land near `official`.
2. **An unknown backend raises.** `get_quant_method` validates the value.
3. **The checkpoint layout is asserted.** `TENSORBRIDGE_STRICT_QWEN36_LAYOUT=1`,
   which `arms.py` sets, requires exactly `{NVFP4: 193, FP8: 208}`.
4. **Each layer asserts the production contract.** A layer that comes up without
   SNC and the swizzle64 layout raises `TensorBridge production contract is
   inactive` during weight loading.

The engine log also prints, once per run:

```
TensorBridge ModelOpt layout: checkpoint={'FP8': 208, 'NVFP4': 193},
    transformer NVFP4=192, lm_head=NVFP4 Marlin W4A16
```

What none of this records is a per-layer observation of the kernel that ran; the
layer counts above come from the checkpoint, not from inspecting the loaded
modules. Given a confirmed-active plugin and a validated backend value, dispatch
is deterministic, so the gap is narrow — but the JSON asserts a configuration,
not a measurement.

## Installing on a fresh H100 machine

Requires an H100 (sm90), CUDA 12.8 toolkit, and GCC ≥ 9 — TensorBridge compiles
its kernels through NVRTC at runtime and PyTorch's extension loader needs a
modern host compiler. On RHEL-family systems the default `/usr/bin/gcc` is
version 8 and will fail; load a newer one and export it.

```bash
# 1. toolchain
export CC=$(command -v gcc) CXX=$(command -v g++)
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
gcc --version   # must be >= 9

# 2. this repository (it is a vLLM fork; the plugin lives inside it)
git clone <this-repo> tensorbridge-vllm && cd tensorbridge-vllm
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=bc150f50299199599673614f80d12a196f377655 \
  uv pip install -e . --torch-backend=auto
# Without the pin, setup.py asks upstream vLLM for today's main and fetches
# that nightly wheel, which drifts away from this fork's base over time.
# The commit is the `vllm.base_commit` in constraints/tensorbridge.json.

# 3. exact package set that produced the reference numbers
uv pip install -r constraints/runtime-requirements.txt
uv pip install lm-eval==0.4.11

# 4. the TensorBridge kernel wheel that produced the reference numbers
sha256sum third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
# expect 7b3aaa6199e22655a7d81a18bbdae695110aa05b9c98a0c05c516473e686ec09
uv pip install third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
```

Check that hash before installing. Every number below is a property of this
specific kernel build, so a different wheel silently invalidates the comparison
rather than failing loudly.

The kernel source for that wheel is vendored alongside it, verified against the
snapshot hash in `constraints/tensorbridge.json`. See `third_party/README.md` —
in particular, install the wheel and not the source: an editable TensorBridge
install lets a kernel branch move these numbers without any signal.

`scripts/verify_tensorbridge_constraint.py` performs the same check but resolves
the wheel through the constraint's `../tensorbridge/dist/` path, so it only runs
where that sibling checkout exists. The `sha256sum` above is the portable
equivalent.

The reference environment is Python 3.12.12, vLLM 0.20.2+cu128, torch
2.11.0+cu128, `tensorbridge-kernels` 0.2.0+g43cc2aa3d9a1, lm-eval 0.4.11,
transformers 5.9.0, datasets 4.8.5, CUDA 12.8, GCC 13.3. Each script records the
versions it ran under, so a later disagreement can be traced.

### Data and checkpoint

The checkpoint is `nvidia/Qwen3.6-27B-NVFP4` (~21 GB). Datasets are pulled by
`datasets`: WikiText-2 raw test, and GSM8K main test pinned to revision
`740312add88f781978c0658806c59bc2815b9866` in the task file. Pre-download them
if the compute node has no network, and set `HF_DATASETS_OFFLINE=1` there.

## Running

Roughly 25-35 min per arm on one H100. Engine startup is about 4 minutes once
the compile caches are warm on the node: vLLM JIT-compiles
a FlashInfer kernel for this model's linear-attention layers during its KV-cache
memory profiling, and TensorBridge compiles ~85 NVRTC kernels. Both are cached
after the first run on a machine.

```bash
cd eval_simple
MODEL=/path/to/Qwen3.6-27B-NVFP4 ./run.sh      # all three arms in sequence
python compare.py results/<run_id>
```

`results/` already holds the reference run whose numbers appear below, so
`python compare.py results` reproduces the tables without running anything. A new
run lands in `results/<run_id>/` beside it.

`run.sh` needs no scheduler. It checks the toolchain and driver, puts every
compile cache on node-local disk, runs both benchmarks per arm, and prints the
`compare.py` line to follow up with. `PYTHON`, `MODEL`, `RUN_ID`, `OUT`,
`CACHE_ROOT`, and `BAD_DRIVERS` override its defaults; `./run.sh alpha_0961`
runs a single arm.

`SMOKE=1 ./run.sh alpha_0961` cuts it to 8 blocks and 32 documents, a few
minutes, to prove a new machine works. Those numbers do not compare to the tables
below.

On Slurm, `sbatch --array=0-2 eval_simple/run.sbatch` runs the arms in parallel;
that wrapper only loads this cluster's modules and hands one arm to `run.sh`.

Arms of one comparison must share `RUN_ID` — `compare.py` pairs them by
directory. Sequential and array runs handle that themselves, but re-running a
single failed arm needs `RUN_ID=<original>` set by hand, or it lands in a
directory of its own and cannot be paired.

## Expected results

Measured on one H100 80GB in the environment above.

Produced by these scripts, run 457238.

### WikiText-2, 297,192 scored tokens over 291 blocks

| Arm | Mean NLL | PPL |
| --- | ---: | ---: |
| `alpha_0961` | 1.9482251711 | 7.016223927 |
| `official` | 1.9496641970 | 7.026327723 |
| `normal_a8` | 1.9525988881 | 7.046978110 |

### GSM8K, all 1319 documents

| Arm | exact_match | Correct |
| --- | ---: | ---: |
| `official` | 96.2851% | 1270 |
| `normal_a8` | 96.2092% | 1269 |
| `alpha_0961` | 96.1334% | 1268 |

Paired, exact McNemar over the documents where two arms disagree:

| Comparison | win/loss | Δ pp | 95% CI pp | p |
| --- | ---: | ---: | ---: | ---: |
| **`alpha_0961` − `normal_a8`** | 9/10 | **−0.08** | [−0.72, +0.57] | 1.00 |
| `official` − `alpha_0961` | 10/8 | +0.15 | [−0.48, +0.78] | 0.81 |
| `official` − `normal_a8` | 8/7 | +0.08 | [−0.50, +0.65] | 1.00 |

The three arms agree on 1293 of 1319 documents and land within one correct
answer of each other.

### The same measurement in eager mode

An earlier set of runs used `enforce_eager=True`, before these scripts moved to
CUDA graphs. Both are shown because the agreement between them is the useful
part, not either column alone.

| Arm | PPL eager | PPL graph | GSM8K eager | GSM8K graph |
| --- | ---: | ---: | ---: | ---: |
| `alpha_0961` | 7.007963 | 7.016224 | 96.0576% | 96.1334% |
| `official` | 7.036093 | 7.026328 | 96.2851% | 96.2851% |
| `normal_a8` | 7.054202 | 7.046978 | 95.9060% | 96.2092% |

The primary comparison holds either way: `alpha_0961` − `normal_a8` is −0.66% on
perplexity and +0.15 pp on GSM8K in eager, −0.43% and −0.08 pp under graphs. Both
GSM8K deltas sit well inside their intervals.

Individual arms move by up to 0.2% on perplexity between the modes, and not all
in the same direction. Disabling eager also enables torch.compile, whose fusions
(`fuse_norm_quant`, `fuse_act_quant`) apply differently to arms with different
linear methods. So a cross-mode comparison of a single arm is not meaningful,
while a same-mode comparison between arms is.

Repeating `alpha_0961` under graphs three times gave 7.014765, 7.016224, and
7.016990 — a spread of 0.0022, against an arm-to-arm gap of about 0.01. Treat a
single perplexity run as resolving differences of roughly a percent, not a
tenth of one.

### Reading this

FPMA with the analytic compensation is −0.08 pp on GSM8K against its exact
baseline, with a 95% interval inside ±0.72 pp, and 0.43% below it on perplexity.
Neither benchmark detects an accuracy cost, and the eager runs agree.

The compensation is what makes that true. `fpma_default` — the same kernel with
`alpha=1.0` — measured 7.224942 in eager mode, 2.42% worse than `normal_a8`.
FPMA's one-ULP bias is real and roughly a hundred times the residual left after
compensating it.

GSM8K is greedy, but its total still moves by a document or two across drivers,
batching, and execution mode; that is small against the ±0.7 pp interval.
Perplexity is more reproducible but not exact — see the spread noted above.

## Pitfalls

Every item here cost at least one wasted run, and none of them announced itself.

**Engine arguments that fail quietly.** `eval_gsm8k.py` must pass
`enable_thinking=False`; lm-eval defaults it to `True`, and Qwen3.6 then keeps
commenting after its final answer line, which the end-anchored filter rejects.
That scored 27.7% with correct arithmetic throughout. `language_model_only=True`
is also required, though that one at least crashes: without it vLLM builds the
vision tower and imports a module this install does not ship. `eval_gsm8k.py`
now exits non-zero on a full run scoring under 0.80, after writing its artifacts:
the first symptom of the `enable_thinking` bug was a short run at 25% that read
as small-sample noise and was argued away rather than investigated.

**Compile caches on network storage.** The repository is on NFS here. Triton's
compile-write-rename-read cycle loses races there once inductor is active, and
`run.sbatch` puts every compile cache on node-local disk for that reason. Eager
runs never hit it because inductor never ran.

**Mixed driver versions.** Nodes in this cluster carry 570.86.10, 595.58.03, and
610.43.02. Inductor-generated Triton kernels fault on 570.86.10 with "an illegal
instruction was encountered", or the engine simply fails to initialize — again
only once CUDA graphs are on. `run.sbatch` checks the driver before doing any
work. A cluster with one driver everywhere will not see this, but a shared JIT
cache across mixed drivers is worth ruling out before debugging anything else.

## Scope

These scripts measure accuracy. They say nothing about throughput: the arms have
different startup costs and were never run under clock pinning or on a quiet
node. Timing fields in the JSON are for spotting a pathological run, not for
performance claims.

Compared to the full harness in `scripts/` and
`vllm/plugins/tensorbridge_evaluation/`, this version drops the git provenance
hashing, the 21 GB checkpoint manifest, the JIT cache seeding, and the frozen
protocol files. Those exist to make a result auditable as formal evidence. What
is kept here is what makes a result correct and comparable: pinned dataset
revision, greedy decoding, fixed seeds, prefix caching off, a hard failure on
non-finite logprobs, an assertion on the scored-token count, and a verified
pairing before any GSM8K comparison.
