# Simple TensorBridge performance evaluation

Four small scripts that measure whether TensorBridge's NVFP4A8 kernel makes vLLM
faster, and where decode time actually goes.

```
arms.py               which kernel runs, expressed as environment variables
bench_gemm.py         per-layer GEMM timing for all three arms, inside a CUDA graph
bench_latency.py      decode latency across batch sizes for one arm
compare.py            summarise a results directory, paired per-batch speedup
summarize_profile.py  group an Nsight Systems kernel summary by GPU time
```

Each `bench_latency.py` invocation runs one arm and writes one JSON file. There
is no scheduler and no manifest — run them a few times in the prescribed order,
then run `compare.py`.

## What the arms are

| Arm | MLP path |
| --- | --- |
| `official` | NVFP4 weights, Marlin W4A16, BF16 activation |
| `normal_a8` | B8 expanded exactly at load, Cutlass FP8, per-token FP8 activation |
| `tensorbridge` | FPMA-SNC generating B8 in the mainloop, same activation path |

**`official` is the baseline.** On Hopper it is what vLLM picks by itself:
`FlashInferCutlassNvFp4LinearKernel` needs `sm_100` and `CutlassNvFp4LinearKernel`
needs `cutlass_scaled_mm_supports_fp4()`, also `sm_100`, so Marlin is the only
NVFP4 linear kernel available. A speedup claim has to beat it.

Only the 192 transformer MLP projections vary. `lm_head` stays NVFP4 Marlin
W4A16 in every arm — it shows up in the profile as ~2% of GPU time at 960 us per
call — and the 208 FP8 layers are identical. That bounds any arm difference to
the MLP's share of the forward pass.

## Installing on a fresh H100 machine

Requires an H100 (sm90), CUDA 12.8 toolkit, and GCC >= 9: TensorBridge compiles
its kernels through NVRTC at runtime and PyTorch's extension loader needs a
modern host compiler. On RHEL-family systems `/usr/bin/gcc` is version 8 and
will fail; load a newer one and export it.

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

# 4. the TensorBridge kernel wheel that produced the reference numbers
sha256sum third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
# expect 7b3aaa6199e22655a7d81a18bbdae695110aa05b9c98a0c05c516473e686ec09
uv pip install third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
```

Check that hash. Every number below is a property of this kernel build, and a
different wheel changes the results without failing. Install the wheel, not the
source: an editable TensorBridge install lets a kernel branch move these numbers
with no signal.

Reference environment: Python 3.12.12, vLLM 0.20.2+cu128, torch 2.11.0+cu128,
`tensorbridge-kernels` 0.2.0+g43cc2aa3d9a1, CUDA 12.8, GCC 13.3, driver
610.43.02, H100 80GB HBM3. Each result records its own versions and GPU, and
`compare.py` refuses to compare runs that disagree.

The checkpoint is `nvidia/Qwen3.6-27B-NVFP4` (~21 GB). No dataset is needed —
prompts are random token ids.

## Running

Run the arms interleaved as a palindrome and repeat, so that each arm appears at
mirrored positions in time:

```bash
cd perf_simple
M=/path/to/Qwen3.6-27B-NVFP4

# Per-layer GEMM, all three arms in one process. A few minutes.
python bench_gemm.py --model $M --output results/gemm.json

# End to end, one process per arm. Hours.
i=0
for arm in official normal_a8 tensorbridge tensorbridge normal_a8 official; do
  python bench_latency.py --arm $arm --model $M \
      --output results/$(printf %02d $i)_${arm}.json
  i=$((i+1))
done

python compare.py results/
```

The palindrome is what makes the comparison survive drift: `compare.py` pairs
the k-th run of one arm with the k-th run of the other, so a monotonic change in
machine state affects both halves of a pair equally. The repeat is what gives
each arm two runs at mirrored positions, which is the noise floor — without it
`compare.py` has nothing to test a ratio against and says so.

About 25 minutes per run: roughly 10 minutes of measurement and the rest engine
startup. Startup is dominated by the checkpoint read and, on the first run for
an arm, NVRTC compilation of ~85 kernels. Both are cached afterwards, so run
everything on one machine and do not spread arms across nodes — six parallel
cold reads of a 21 GB checkpoint on shared storage pushed startup from 78 s to
1588 s in an earlier attempt.

Smoke test first: `--batch-sizes 1,16 --iters 2 --warmup 1` finishes in a few
minutes and proves the environment works. Those numbers are not comparable to
the table below.

If a batch size is not in the captured CUDA graph list the script aborts rather
than running. An uncaptured size falls back to eager silently, which reads as a
kernel slowdown at that batch alone.

## Expected results

H100 80GB, clocks not pinned (`nvidia-smi -lgc` needs privileges this cluster
does not grant). The two tables were produced on different nodes — GEMM on
driver 595.58.03, end to end on 610.43.02 — so compare ratios between them, not
absolute microseconds.

### End to end

`compare.py` over six runs of `bench_latency.py`, ordered
`official normal_a8 tensorbridge tensorbridge normal_a8 official`:

```
normal_a8 vs official
   batch   baseline s    normal_a8    speedup    floor  verdict
       1        3.576        4.051     0.883x    1.002  reportable
       4        3.953        4.425     0.893x    1.002  reportable
      16        4.910        5.224     0.940x    1.004  reportable
      32        5.973        6.027     0.991x    1.001  reportable
      64        8.371        7.319     1.144x    1.004  reportable
     128       13.296       10.369     1.282x    1.001  reportable

tensorbridge vs official
   batch   baseline s tensorbridge    speedup    floor  verdict
       1        3.576        3.588     0.997x    1.002  reportable
       4        3.953        3.978     0.994x    1.002  reportable
      16        4.910        4.815     1.020x    1.004  reportable
      32        5.973        5.553     1.076x    1.001  reportable
      64        8.371        7.014     1.193x    1.001  reportable
     128       13.296       10.081     1.319x    1.001  reportable
```

TensorBridge breaks even around batch 16 and reaches 1.32x at batch 128. It
beats `normal_a8` at every batch size, which is the interesting comparison:
those two share the weights, the activation path and the epilogue scale, and
differ only in whether B8 is expanded exactly at load or approximated in the
mainloop. The FPMA mainloop is a net win, not just a memory saving.

`normal_a8` is 11% *slower* than Marlin at batch 1. Quantising activations to
FP8 does not pay for itself until the GEMM is large enough to care.

The noise floor is 1.001-1.004 because all six runs share one job on one node
with the checkpoint already in page cache. An earlier attempt that spread arms
across nodes produced floors of 1.03-1.08, wide enough to swallow the batch 4
and 16 results.

### Per-layer GEMM

`bench_gemm.py`, all three arms inside a captured CUDA graph. Gap against
Marlin; negative means faster.

```
     M       N       K      normal_a8   tensorbridge
     1   34816    5120         +38.5%          -3.5%
     4   34816    5120         +39.6%          -2.7%
    16   34816    5120         +33.1%          -1.7%
    32   34816    5120         +24.1%          -6.7%
    64   34816    5120         -10.1%         -27.1%
   128   34816    5120         -49.8%         -53.8%

     1    5120   17408         +49.6%          +7.4%
     4    5120   17408         +54.0%          +6.9%
    16    5120   17408         +45.3%          +3.4%
    32    5120   17408         +26.8%         -17.7%
    64    5120   17408         -23.0%         -40.7%
   128    5120   17408         -42.4%         -53.8%

     1    6144    4096          -4.7%          -2.6%
     4    6144    4096          -1.1%          -2.4%
    16    6144    4096         -16.8%          -4.3%
    32    6144    4096          -7.8%         -18.5%
    64    6144    4096         -50.4%         -39.9%
   128    6144    4096         -59.7%         -48.1%

     1    2048    6144         +10.5%          -9.6%
     4    2048    6144         +17.0%          -9.4%
    16    2048    6144          -0.7%         -13.5%
    32    2048    6144         +10.1%         -20.5%
    64    2048    6144         -33.6%         -38.9%
   128    2048    6144         -29.7%         -38.5%
```

The first two shapes are this checkpoint's MLP and predict the end-to-end table:
`tensorbridge` crosses zero between M=16 and M=32, and the engine crosses between
batch 16 and 32. End-to-end ratios are smaller than GEMM ratios because the MLP
is only part of the forward pass — see the profile below.

The other two shapes are not in this checkpoint. They are there so the ranking
is not read off two matrices that both share a hidden size of 5120 and a very
wide inner dimension, and they change two conclusions:

- **TensorBridge's small-M weakness is shape-specific, not intrinsic.** At
  `2048,6144` it already wins 9.6% at M=1. The near-parity at M=1 on the MLP
  shapes comes from those particular dimensions, so a model with a smaller
  hidden size would cross over earlier than batch 16.
- **`normal_a8` overtakes TensorBridge on `6144,4096` at large M** (-50.4% vs
  -39.9% at M=64). It loses on the other three shapes. Same weights, same
  activation path — so this is the FPMA mainloop underperforming on that tile
  configuration, not a quantisation cost, and it is worth chasing.

Timing eager launches instead of graph replay inverts most of this. The same
`tensorbridge` layer at M=1 on `34816,5120` reads +95.3% eager and -3.5% in a
graph: about 98 points of host launch cost that the engine never pays because
`FULL_DECODE_ONLY` replays a captured decode step.

### Where decode time goes

`summarize_profile.py` on a batch-128 profile of the `tensorbridge` arm:

```
group                               GPU %    us/call
gated delta rule (linear attn)      29.10     270.25
TensorBridge NVFP4 GEMM             21.96      75.45
elementwise                         21.59       4.86
CUTLASS (FP8 layers)                14.05      38.31
reduce                               4.24       8.81
Marlin (NVFP4 W4A16, lm_head)        2.18     960.17
TensorBridge activation quant        1.10       3.78
vLLM FP8 activation quant            1.02       3.10
```

Two things follow.

**Activation quantisation is not a problem.** It is 1.1% of GPU time, and
TensorBridge's Triton kernel takes 2.28 us per call against vLLM's hand-written
CUDA kernel at 2.34 us. A wall-clock benchmark disagrees loudly — 26.8 us versus
10.4 us — because it charges each kernel its host launch path, and TensorBridge's
Triton launch path is slower. Graph replay does not repeat that cost. This is the
single largest gap between what an eager kernel benchmark predicts and what the
engine does.

**The linear-attention state, not the linear layers, dominates at large batch.**
Qwen3.6 runs 48 gated-delta-rule layers, each keeping a per-sequence recurrent
state of `(48, 128, 128)` in bfloat16, 1.57 MB. Every decode step reads and
writes it, so traffic is linear in batch with no amortisation: 7.0% of GPU time
at batch 16 becomes 29.1% at batch 128. It is identical in both arms, so it caps
what any NVFP4 kernel change can deliver — near `1 / (1 - 0.22) ~ 1.28x` at batch
128 from the GEMM alone.

## Scope

These scripts measure single-batch decode latency. They are not a serving
benchmark: there is no arrival process, no continuous batching, no queueing, and
no TTFT/ITL split. `max_model_len=512` keeps the KV cache small, so attention is
a smaller share than in real serving, which flatters the linear-layer
comparison. Prefill grows with batch, so the large-batch points are not pure
decode.

They also say nothing about accuracy — prompts are random token ids and no
output is checked. Use `eval_simple/` for that.

`bench_gemm.py` builds each arm from random weights rather than the checkpoint,
so it measures kernel speed at a shape, not this model's numerics. It also skips
`lm_head` (`N=248320, K=5120`), the checkpoint's third NVFP4 shape: the plugin's
`lm_head` branch precedes the backend branch, so every arm runs Marlin there and
an arm comparison on that shape would describe a path none of them take.

Neither script pins GPU clocks, and no node here is reliably idle. The noise
floor `compare.py` reports is the guard against that; treat a result inside it
as no result.
