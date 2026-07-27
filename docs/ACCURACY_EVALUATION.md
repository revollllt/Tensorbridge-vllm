# TensorBridge Accuracy Evaluation

This document defines the first accuracy gate for running NVIDIA ModelOpt NVFP4
checkpoints on Hopper through TensorBridge. FPMA is an efficiency-oriented
approximation, so the gate records its numerical tradeoff rather than imposing a
bit-exact end-to-end threshold.

## Evaluation contract

The canonical experiment uses:

- checkpoint: `nvidia/Qwen3.6-27B-NVFP4`;
- dataset: WikiText-2 raw test split, joined with two newlines and tokenized without
  special tokens;
- context: `max_model_len=2048`, up to 2047 prompt tokens and up to 1024 scored
  target tokens per block;
- coverage: 291 overlapping blocks and 297,192 uniquely scored target tokens;
- runtime: vLLM 0.20.2, TP=1, BF16, eager mode, `max_num_seqs=8`, and
  `gpu_memory_utilization=0.5`;
- resources: one H100, eight CPUs, 80 GiB host memory, and all OpenMP/BLAS thread
  limits set to eight;
- compilation: NVRTC, strict C++ workload routing, production swizzle64 raw layout,
  host/device preinterleave enabled, and no scale clamp.

Pytest is run only inside a GPU allocation. Each compile-time ULP ABI uses a separate
physical TensorBridge cache.

## Numerical paths

For an E2M1 weight `q`, an E4M3 block scale `s` (one per 16 weights), and the
ModelOpt FP32 tensor scale `g`:

```text
W_exact = q * s * g
g       = weight_amax / (6 * 448)
```

The experiment distinguishes three baseline paths:

1. `W4A16 official`: BF16 activation with the checkpoint NVFP4 weight, executed by
   vLLM Marlin.
2. `normal-A8`: pre-expand `B8 = E4M3(BF16(q*s/6))`; dynamically quantize each
   activation row/token to E4M3; run vLLM `CutlassFP8ScaledMMLinearKernel`; apply one
   scalar `weight_scale = 6*g`.
3. `TensorBridge FPMA-SNC`: keep W4 plus g16 E4M3 scales in the production swizzle64
   layout, generate B8 fragments inside the mainloop, and apply the same external
   `6*g` scale. SNC is always enabled.

The normal-A8 whole-model baseline intentionally uses vLLM Cutlass rather than the
generic TensorBridge FP8 kernel. The latter has an unrelated large-M correctness
problem at the Qwen runtime shapes and is not used for whole-model PPL. The
controlled Linear harness below uses its verified shared-FP8 reference path only at
`M<=512`.

### Why `/6` is required

E2M1 reaches 6 and E4M3 block scales reach 448. Casting `q*s` directly to E4M3 can
therefore exceed its finite range. TensorBridge instead encodes:

```text
prefolded_scale = raw_e4m3(s) - 0x1c
B8              ~= E4M3(q*s/6)
epilogue_scale  = 6*g
```

This is range protection, not an SNC operation. SNC remains enabled independently.

## Compensation methods

| Method | Mechanism | Steady-state cost | Calibration status |
| --- | --- | --- | --- |
| Global alpha (`alpha_0960`, frozen evaluated arm) | Replace every external scale with `6*g*alpha` | Zero extra kernel instructions or storage | `alpha=0.960`; selected on WikiText-2 |
| `normal_b8_sse` selector | At load time choose `prefold` or `prefold-1` for each g16 group using the checkpoint's E2M1 histogram | Zero extra kernel instructions or storage; additional load-time work | Weight-only |
| Conditional ULP V1 | Subtract one B8 ULP for the exact scale-residue/magnitude cases | Extra integer mainloop instructions; no extra weight bytes | Weight-only, verified scale domain |

The evaluated `alpha_0960` arm is useful as a zero-cost model calibration, but a
value selected on WikiText-2 must be frozen and checked on held-out tasks before it
is treated as a general result. The selector and ULP rule depend only on checkpoint
bytes.

### Conditional ULP V1

For raw scales `0x39..0x7e`, exhaustive enumeration of all 70 scales and 16 E2M1
codes gives 1,120 combinations. Default FPMA differs from
`E4M3(BF16(q*s/6))` in 344 combinations. The conditional correction removes all
344 mismatches, so all 1,120 combinations match after canonicalizing signed zero.

The correction applies when:

```text
prefolded_scale modulo 8 (its low-three-bit residue) is in [2, 6]
and SNC-remapped E2M1 magnitude is even
```

The first implementation recomputed scale eligibility in every dequant fragment.
V1 now computes it once at load time and stores it in bit 7 of the prefolded scale;
the verified prefold range is only `0x1d..0x62`, so that bit is otherwise unused.
The device extracts the flag, clears bit 7, combines it with the existing magnitude
parity through one LOP3, and performs the conditional subtract.

The versioned compile contract is:

```text
TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION=1
TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1=1
```

The vLLM and Linear entrypoints inject both flags automatically and fail closed on
ABI mismatch. ULP is accepted only with all of the following:

- SNC, g16, and swizzle64 raw layout;
- `alpha == 1.0` and selector `none`;
- scale clamp disabled;
- every nonzero raw scale in `0x39..0x7e`;
- NVRTC and a matching V1 scale ABI cubin;
- an independent physical JIT cache for the compile variant.

This rule is exact for normal-B8 weight bytes in the verified scale domain. It does
not imply bit-exact model outputs across different GEMM reduction schedules.

## Linear evaluation

The Linear harness reports:

```text
MSE_A8    = mse(Y_normal,       Y_w4a16_exact)
MSE_FPMA  = mse(Y_tensorbridge, Y_normal)
MSE_total = mse(Y_tensorbridge, Y_w4a16_exact)
```

It also reports NMSE, `MSE_FPMA/MSE_A8`, a bit-level FPMA B8 reference, production
router configuration, finite counts, and canonical B8 mismatch counts. Separate
StreamK launches are not bit deterministic, so output-level reference residuals can
remain nonzero even when the B8 bytes match exactly.

This is a synthetic-input Linear check, not a trace of Qwen activations. For each
module, the harness generates one 512-row BF16 normal buffer with standard deviation
1.0 and seed 1234; `M=1,16,128,512` are prefixes of that buffer. The checkpoint input
scale is recorded but not applied. `Y_normal` uses the verified TensorBridge
shared-FP8 reference kernel for these controlled sizes, whereas whole-model
normal-A8 PPL uses vLLM Cutlass. The MSE ratios are therefore scoped to this input
distribution; held-out real-activation traces remain a later gate.

Final V1 validation used layer-0 gate and down projections at `M=1,16,128,512`:

| Module | `(N,K)` | Raw scale range | `MSE_FPMA/MSE_A8` range | Canonical B8 mismatches |
| --- | ---: | ---: | ---: | ---: |
| `gate_proj` | `(17408,5120)` | `0x49..0x7e` | 0.444%-1.405% | 0 / 89,128,960 |
| `down_proj` | `(5120,17408)` | `0x42..0x7e` | 1.812%-2.792% | 0 / 89,128,960 |

Every output was finite. Raw byte differences were entirely signed-zero encoding:
3,494,983 for gate and 3,524,502 for down. Job `416626` contains the final V1
artifacts.

Run the same gate/down matrix with:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
export TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION=1
export TENSORBRIDGE_NVFP4_FPMA_ALPHA=1.0
export TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR=none
export TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP=0
export TENSORBRIDGE_EVAL_MATRIX_M_VALUES=1,16,128,512
sbatch --array=0,2%2 --export=ALL sbatch/run_eval_nvfp4_linear_matrix.sbatch
```

The full diagnostic matrix covers layers 0, 31, and 63 (`gate_proj`, `up_proj`, and
`down_proj`). Its `lm_head` task is diagnostic only; production PPL uses Marlin for
the output head.

## Whole-model output head

The checkpoint contains 193 NVFP4 logical layers, including `lm_head`.

- Official: all 193 NVFP4 layers use Marlin W4A16.
- Normal-A8: 192 transformer MLP projections use dynamic-A8/Cutlass B8; `lm_head`
  remains NVFP4 Marlin W4A16.
- TensorBridge: 192 transformer MLP projections use FPMA-SNC; `lm_head` remains
  NVFP4 Marlin W4A16.

The output-head activation is BF16 in every arm. This keeps `lm_head` policy out of
the TensorBridge accuracy delta.

## WikiText-2 results

All rows below score the same 297,192 target tokens.

| Arm | Compensation | Mean NLL | PPL | Notes |
| --- | --- | ---: | ---: | --- |
| Official Marlin W4A16 | none | 1.951021393 | 7.035870298 | Reference checkpoint path |
| Normal-A8 Cutlass | normal B8 | 1.953623434 | 7.054201761 | Dynamic per-token A8 |
| TensorBridge default | none | 1.977539164 | 7.224941694 | FPMA-SNC baseline |
| TensorBridge selector | selector, alpha=1 | 1.957302064 | 7.080199351 | Weight-only, zero steady-state cost |
| TensorBridge ULP V1 | conditional ULP | 1.953321450 | 7.052071826 | Weight-only, no dataset calibration |
| TensorBridge global alpha | alpha=0.960 | 1.948419298 | 7.017586098 | Lowest tested boundary; WikiText-2 calibrated |
| TensorBridge selector+alpha | selector, alpha=0.986 | 1.949608748 | 7.025938130 | Lowest tested selector boundary; calibrated |

ULP V1 reduces default TensorBridge PPL by 0.172870 (2.393%) and lands within
0.002130 of normal-A8. It is 0.016202 (0.230%) above Official. Its PPL and NLL are
bit-for-bit identical to the earlier runtime-residue implementation, confirming that
the scale-MSB encoding changes cost rather than arithmetic.

Global alpha and selector+alpha produce lower PPL than Official on the tuning set.
That is not evidence of a more faithful dequantization: both alpha values were
selected on WikiText-2 and need held-out validation.

### Global alpha sweep

| Alpha | Mean NLL | PPL |
| ---: | ---: | ---: |
| 0.980 | 1.958388828 | 7.087898036 |
| 0.976 | 1.955800557 | 7.069576360 |
| 0.974 | 1.955440748 | 7.067033117 |
| 0.973 | 1.954952883 | 7.063586199 |
| 0.972 | 1.953844548 | 7.055761719 |
| 0.970 | 1.952923680 | 7.049267282 |
| 0.968 | 1.950443604 | 7.031806229 |
| 0.966 | 1.949624255 | 7.026047083 |
| 0.965 | 1.949535276 | 7.025421937 |
| 0.964 | 1.948989738 | 7.021590351 |
| 0.962 | 1.948527945 | 7.018348578 |
| 0.960 | 1.948419298 | 7.017586098 |

The sweep stops at 0.960 even though the lowest boundary still improves. Further
search on the same evaluation set would increase calibration bias without answering
whether the value generalizes.

### Selector sweep

| Alpha | Mean NLL | PPL |
| ---: | ---: | ---: |
| 1.004 | 1.961219188 | 7.107987760 |
| 1.002 | 1.957673088 | 7.082826757 |
| 1.000 | 1.957302064 | 7.080199351 |
| 0.998 | 1.956088241 | 7.071610453 |
| 0.996 | 1.955018716 | 7.064051236 |
| 0.994 | 1.952140200 | 7.043746489 |
| 0.990 | 1.950554389 | 7.032585286 |
| 0.986 | 1.949608748 | 7.025938130 |

## lm-evaluation-harness smoke

The vLLM integration now runs `lm-evaluation-harness` 0.4.11 in process and
records per-sample JSONL, prompt/target hashes, checkpoint identity, source hashes,
thread limits, and the selected precision ABI. The first gate is intentionally
small: the first 16 ARC-Challenge multiple-choice documents and the first 16 paired
GSM8K documents. It is a wiring and regression smoke, not a randomized sample or
a statistical accuracy claim.

The stock `gsm8k_cot_zeroshot` task was rejected for this purpose. Its strict
regex consumes the final digit when the model omits a trailing period, its
flexible filter can select a unit-conversion number instead of the answer, and
3 of 16 outputs reached the old 512-token cap. TensorBridge therefore uses the
zero-shot `tensorbridge_gsm8k_relative_smoke` task: concise reasoning, thinking
disabled, a final `The answer is N` line, one end-anchored filter, and a 1024-token
safety cap. In the retained run, outputs use only 67-473 tokens.

| Arm | ARC acc | ARC acc_norm | GSM8K final-answer EM | Format valid |
| --- | ---: | ---: | ---: | ---: |
| Official Marlin W4A16 | 0.5625 | 0.5000 | 1.0000 | 1.0000 |
| Normal-A8 Cutlass | not run | not run | 1.0000 | 1.0000 |
| TensorBridge default FPMA-SNC | 0.5625 | 0.4375 | 0.8750 | 1.0000 |
| TensorBridge ULP V1 | 0.5625 | 0.5000 | 0.8750 | 0.9375 |

The two FPMA arms miss the same two GSM8K documents in the 16-sample run.
Default omits the 40 minutes spent before a forced restart on one sample; both
FPMA arms choose break-even year 12 rather than the target's first-positive-profit
year 13 on another, semantically ambiguous sample. ULP also produces the isolated
single-token output `ModifiedDate` on the restart sample. That anomaly does not
reproduce in a cache-reused first-8 replay, which scores 8/8. With only two
discordant pairs, the exact two-sided paired p-value is 0.5; these observations
identify sensitive examples but do not establish a population accuracy loss.

The tracked machine-readable audit is
`docs/results/tensorbridge_lm_eval_smoke_20260717.json`. Raw artifacts remain under
`benchmarks/results/lm_eval/` and are ignored by Git.

## Held-out lm-evaluation-harness confirmation

This confirmation is separate from the WikiText-2 contract above. It follows the
pre-run `accuracy_confirm_v1.json` protocol (SHA256
`3a3ab3a617143b29eb94af5a75316b74f44d66841d2970a85206c4ac091350a2`) and uses the
same NVIDIA checkpoint revision and content hash in all four arms. Default FPMA-SNC
is the primary candidate. ULP V1 is exploratory. Both TensorBridge arms use
`alpha=1.0`, selector `none`, SNC enabled, and scale clamp disabled, so this run does
not validate either WikiText-2-tuned alpha or the selector.

### Confirmation protocol

- ARC-Challenge executes all 1,172 test documents, but the primary held-out analysis
  is doc IDs `16..1171` (`n=1156`). IDs `0..15`, used by the earlier smoke, are
  retained only in the descriptive full-split columns.
- GSM8K reuses the same task definition and final-answer scorer as the smoke, but it
  is not a smoke sample. A preregistered SHA256-rank manifest selects 128 documents
  from IDs 16 onward. The manifest SHA256 is
  `37bca36cf4be344ed07209b2caa76688969ac20664f3ad4e263409296455f2df`, and the
  selected ID-list SHA256 is
  `a43574fc29a99293c793b08c17a02b720cd4f9487e9fd33ef299515903924fc2`.
- The required primary comparison is Default FPMA-SNC minus Normal-A8. Its
  preregistered point-estimate margins are `-2 pp` for ARC `acc_norm` and `-5 pp` for
  GSM8K exact match. GSM8K additionally requires both candidate and baseline to
  produce at least 127 valid final-answer formats out of 128.
- ULP V1 minus Normal-A8 is evaluated against the same thresholds only as an
  exploratory check. Normal-A8 minus Official isolates activation quantization, and
  ULP V1 minus Default is descriptive.
- Every comparison is paired. Reports include loss/gain flips, a two-sided exact
  McNemar p-value, and a 10,000-resample paired percentile-bootstrap 95% CI. The CI is
  reported but is explicitly not a gate. Invalid or truncated generations count as
  incorrect, and no ambiguous or discordant sample is dropped.

### GSM8K-128 generation

| Arm | Final-answer exact match | Correct/total | Valid format |
| --- | ---: | ---: | ---: |
| Official Marlin W4A16 | 95.3125% | 122/128 | 128/128 |
| Normal-A8 Cutlass | 95.3125% | 122/128 | 128/128 |
| TensorBridge default FPMA-SNC | 96.8750% | 124/128 | 128/128 |
| TensorBridge ULP V1 | 94.5313% | 121/128 | 128/128 |

| Comparison (candidate - baseline) | Candidate loss/gain flips | Delta (pp) | 95% paired bootstrap CI (pp) | Exact McNemar p | Role and outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Default FPMA-SNC - Normal-A8 | 0/2 | +1.5625 | [0.0000, +3.9063] | 0.5000 | Primary; `-5 pp` point-estimate and format gates passed |
| Normal-A8 - Official | 1/1 | 0.0000 | [-2.3438, +2.3438] | 1.0000 | Activation-quantization control |
| ULP V1 - Default FPMA-SNC | 3/0 | -2.3438 | [-5.4688, 0.0000] | 0.2500 | Exploratory descriptive comparison |
| ULP V1 - Normal-A8 | 1/0 | -0.7813 | [-2.3438, 0.0000] | 1.0000 | Exploratory; `-5 pp` threshold and format checks passed |

The only ULP loss relative to Normal-A8 is doc 873. Normal-A8 and Default end with
`The answer is 12`, while ULP ends with `The answer is 12.00`. The gold answer is
`12`; the registered exact-string scorer therefore counts ULP as incorrect even
though the values are numerically equal. The primary result keeps this loss. As a
separate sensitivity observation only, accepting numeric equivalence would raise ULP
to 122/128 (95.3125%).

### ARC-Challenge

| Arm | Primary acc_norm, n=1156 | Primary acc, n=1156 | Full acc_norm, n=1172 | Full acc, n=1172 |
| --- | ---: | ---: | ---: | ---: |
| Official Marlin W4A16 | 62.9758% | 60.9862% | 62.7133% | 60.9215% |
| Normal-A8 Cutlass | 62.2837% | 61.1592% | 62.0307% | 61.0068% |
| TensorBridge default FPMA-SNC | 61.2457% | 60.5536% | 61.0068% | 60.4949% |
| TensorBridge ULP V1 | 61.8512% | 60.7266% | 61.6041% | 60.6655% |

| Comparison (candidate - baseline) | Candidate loss/gain flips | Delta (pp) | 95% paired bootstrap CI (pp) | Exact McNemar p | Role and outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Default FPMA-SNC - Normal-A8 | 37/25 | -1.0381 | [-2.3356, +0.3460] | 0.1619 | Primary; `-2 pp` point-estimate gate passed |
| Normal-A8 - Official | 20/12 | -0.6920 | [-1.6436, +0.2595] | 0.2153 | Activation-quantization control |
| ULP V1 - Default FPMA-SNC | 33/40 | +0.6055 | [-0.8651, +2.0761] | 0.4828 | Exploratory descriptive comparison |
| ULP V1 - Normal-A8 | 30/25 | -0.4325 | [-1.7301, +0.8651] | 0.5901 | Exploratory; `-2 pp` threshold check passed |

Default passes the preregistered `-2 pp` point-estimate gate, but its CI lower bound
is `-2.3356 pp`, below that margin. This result therefore does not establish
noninferiority under a CI-based rule. ULP scores `+0.6055 pp` above Default on
`acc_norm`, but that exploratory difference is not statistically significant.

The ULP array task ran on ACD1-8 with driver 610.43.02, whereas the other ARC arms ran
on ACD1-6 with driver 570.86.10. A cache audit found the 85 kernel source payloads
unchanged, but the CUDA include path in the cache signature differed between
`/usr/local/cuda/include` and `/usr/local/cuda-12.8/include`. ULP consequently paid a
one-time cold JIT and took 21:34. This is a runtime-environment provenance caveat, not
an accuracy failure or evidence about steady-state ULP cost.

### Confirmation decision

Default FPMA-SNC passes both required preregistered point-estimate gates, and ULP V1
passes both corresponding exploratory threshold checks. On this reduced confirmation
set, we did not observe a degradation exceeding the preregistered permissive
thresholds. This is not a claim of exact equivalence, and the reported CIs are not
used to declare noninferiority.

The v1 protocol sets `stop_after_confirmation=true`. At that decision point, the
planned HellaSwag, Winogrande, MMLU-Pro, and full-GSM8K expansions stopped rather
than selecting more tasks after seeing the confirmation results. The v1 protocol
and its conclusions remain unchanged. A separately authorized extension, frozen
after v1 but before its own prospective stages, is reported in the next section.
The original machine-readable analyses are:

- `docs/results/tensorbridge_lm_eval_confirm_generation_20260717.json`;
- `docs/results/tensorbridge_lm_eval_confirm_mc_20260717.json`.

## Six-arm task-level extension (2026-07-18)

The original `accuracy_confirm_v1.json` protocol was not retroactively edited, and
its four-arm reports remain the canonical v1 evidence. After those results were
observed, the user separately authorized a six-arm extension. It was frozen on
2026-07-18, before its prospective Stage 1, Stage 2, and Stage 3 runs, in
`accuracy_expand_v2.json` (raw SHA256
`77333f869b16d8f29c49f0338594adbc92ae788d02dc6759434d444a737b7b3d`).
Reanalysis of the original ARC and GSM8K tasks is explicitly post-confirmation
sensitivity analysis, not preregistered v1 evidence.

The extension adds the two methods that had favorable WikiText-2 PPL:

- `selector_alpha1`: weight-only `normal_b8_sse`, `alpha=1.0`, and 256-row
  load-time chunks;
- `alpha_0960`: default FPMA encoding with global `alpha=0.960`.

"Zero steady-state cost" has a narrow meaning: neither method adds steady-state
kernel instructions or weight bytes relative to default FPMA-SNC. It does not mean
zero initialization cost. The selector performs extra load-time computation and
may use temporary memory; alpha rewrites only the existing global output scale at
load time. The value 0.960 was selected at the lowest tested boundary of the
WikiText-2 sweep. Its WikiText-2 score is calibration-set evidence, not evidence of
a more faithful local dequantizer. Its results below test transfer away from that
calibration set, but the arm remains exploratory.

### Frozen sample and execution contract

All six arms run in all three prospective stages; intermediate results do not
select or remove arms. The common screen is candidate minus Normal-A8 greater than
or equal to `-5 pp`. The 10,000-resample paired 95% confidence intervals and exact
two-sided McNemar p-values are reported but are not gates.

- Stage 1 is the first 512 post-processed HellaSwag and Winogrande examples. Their
  selected-document SHA256 values are respectively
  `39af1ea86866600455e0543d95601d1e158b0b42c384667ddb9dd3346e09024a` and
  `2745f5df477b490dc9f3b1d02b4a1a0eac4a9c7794be959a44451835f6ebdd33`;
  the composite identity is
  `b8f345a9030494ab72895765d51cc312f0600043fc6f0eac489e516e451c310a`.
- Stage 2 is five-shot MMLU-Pro, first 64 examples in each of 14 fixed leaf tasks,
  with composite identity
  `8fc7f27b644625b3d0d6efa9b2d83523b7f7ff33c8cb91616acd266a32dda312`.
- Stage 3 is 256 SHA256-ranked GSM8K examples, excluding all 144 smoke and v1
  confirmation examples. Its manifest SHA256 is
  `37077471979c08a0b71ecdb384c0d4b5c7d9e37f59225ba724bb0ba3693915e4`,
  selected-ID SHA256 is
  `481e6b075c43ca12c490c460a38430aa57dd22dc83aa5915edc7a268af771cb4`,
  selected-document SHA256 is
  `ee105ca1a10576013a6708dba5da76d5330cecf4481e6acc02db9018a79bf263`,
  and task-source SHA256 is
  `47c193604c56a717778641320d09ed49cab619f00406ce3588cc235c447a36a5`.
  Invalid generation counts as incorrect and each arm must have at least 254/256
  valid final-answer formats.

Every prospective job used one H100 80GB GPU, eight CPUs, 80 GiB host memory, and
all four CPU thread limits set to eight. The frozen vLLM settings include TP=1,
BF16, `batch_size=auto`, `gpu_memory_utilization=0.5`, `max_num_seqs=8`, and exact
per-stage model lengths. GPU pytest job `419662` passed 97 tests in 116.52 seconds
on the same resource class with the same thread limits.

### Immutable reports and jobs

| Evidence | Repository report path | Report SHA256 |
| --- | --- | --- |
| Original v1 GSM8K confirmation | `docs/results/tensorbridge_lm_eval_confirm_generation_20260717.json` | `9446fe6dc96c2699033a97b11be51a9f9d19358570baf3a4f3ec3aea1a98c62a` |
| Original v1 ARC confirmation | `docs/results/tensorbridge_lm_eval_confirm_mc_20260717.json` | `1df537c28d60a32b39f394bf0d81bbee26376facbd0d3ff2e6a69681937b7022` |
| Post-confirm GSM8K sensitivity | `docs/results/tensorbridge_lm_eval_post_confirm_generation_20260718.json` | `bd7b7544f9d2bc457f2fb81a4fa7f504747de2f7f4bb0589630b2bd47451bf84` |
| Post-confirm ARC sensitivity | `docs/results/tensorbridge_lm_eval_post_confirm_mc_20260718.json` | `e472ff91ce026171186b36e582f2b768aebe1020149f1034101bf169855a4578` |
| Prospective Stage 1 | `docs/results/tensorbridge_lm_eval_stage1_mc_20260718.json` | `0ea0bdeacd162797a37b59b74b40684dd4b89849e5e3d77cbbf5393413163a96` |
| Prospective Stage 2 | `docs/results/tensorbridge_lm_eval_stage2_mmlu_pro_20260718.json` | `6c5ad1ade2687d0b577ec7edc138b6182eb1bcb8c28c95be57fb99ae9a052f6b` |
| Prospective Stage 3 | `docs/results/tensorbridge_lm_eval_stage3_generation_20260718.json` | `cef05667b8d6395a5d004dc6a36a6d1558b14a16f9bebaa5fd46995caa14c1c1` |

The v1 reports use bootstrap seed `20260717`; the v2 reports use the separately
frozen seed `20260718`. V2 intervals for reused arms do not replace their v1
counterparts. Each job-table entry is `run-directory/job-element-ID` as recorded in
the report input path. All Stage 1, Stage 2, and Stage 3 elements completed with
exit code `0:0`.

| Suite | Official | Normal-A8 | Default | Selector | ULP V1 | Alpha 0.960 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Post-confirm GSM8K | `417973/417973` | `417983/417984` | `417983/417985` | `419473/419475` | `417983/417983` | `419473/419473` |
| Post-confirm ARC | `417989/417990` | `417989/417991` | `417989/417992` | `419474/419476` | `417989/417989` | `419474/419474` |
| Stage 1 | `419692/419693` | `419692/419694` | `419692/419695` | `419692/419696` | `419692/419697` | `419692/419692` |
| Stage 2 | `419847/419848` | `419847/419849` | `419847/419850` | `419847/419851` | `419847/419852` | `419847/419847` |
| Stage 3 | `420078/420079` | `420078/420080` | `420078/420081` | `420078/420082` | `420078/420083` | `420078/420078` |

### Six-arm scores

The ARC and GSM8K-128 columns are post-confirmation sensitivity results. Stage 1,
Stage 2, and Stage 3 are prospective with respect to the frozen v2 protocol. The
Stage 1 and Stage 2 values are subset scores, not full-benchmark scores.

| Arm | ARC acc_norm, n=1156 | GSM8K EM, n=128 | Hella acc_norm, n=512 | Wino acc, n=512 | MMLU-Pro EM, n=896 | GSM8K EM, n=256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official Marlin W4A16 | 62.9758% (728) | 95.3125% (122) | 76.7578% (393) | 76.7578% (393) | 80.4688% (721) | 95.7031% (245) |
| Normal-A8 Cutlass | 62.2837% (720) | 95.3125% (122) | 76.1719% (390) | 77.7344% (398) | 80.1339% (718) | 94.9219% (243) |
| Default FPMA-SNC | 61.2457% (708) | 96.8750% (124) | 76.1719% (390) | 76.3672% (391) | 78.1250% (700) | 95.3125% (244) |
| Selector, alpha=1 | 62.0242% (717) | 93.7500% (120) | 75.9766% (389) | 78.7109% (403) | 79.0179% (708) | 95.3125% (244) |
| ULP V1 | 61.8512% (715) | 94.5313% (121) | 76.7578% (393) | 78.3203% (401) | 79.0179% (708) | 96.4844% (247) |
| Global alpha=0.960 | 62.1972% (719) | 96.0938% (123) | 76.5625% (392) | 76.7578% (393) | 80.1339% (718) | 96.0938% (246) |

All six arms produced valid final-answer formats on 128/128 post-confirm examples
and 256/256 Stage 3 examples, passing both format gates.

### Paired analysis

Every comparison uses `(leaf_task, doc_id, filter)` pairing. `L/G` is candidate
loss/gain flips. Intervals are 10,000-resample paired percentile-bootstrap 95% CIs;
Stage 2 resamples documents within each fixed category before recomputing the micro
average. P-values are exact two-sided McNemar values and are not adjusted for the
multiple comparisons.

#### Post-confirmation ARC sensitivity

| Candidate - baseline | L/G | Delta (pp) | 95% CI (pp) | p |
| --- | ---: | ---: | ---: | ---: |
| Normal-A8 - Official | 20/12 | -0.6920 | [-1.6436, +0.2595] | 0.2153 |
| Default - Normal-A8 | 37/25 | -1.0381 | [-2.4221, +0.2595] | 0.1619 |
| Selector - Default | 33/42 | +0.7785 | [-0.6920, +2.2491] | 0.3557 |
| Selector - Normal-A8 | 26/23 | -0.2595 | [-1.4706, +0.9516] | 0.7754 |
| ULP V1 - Default | 33/40 | +0.6055 | [-0.8651, +2.0761] | 0.4828 |
| ULP V1 - Normal-A8 | 30/25 | -0.4325 | [-1.7301, +0.8651] | 0.5901 |
| Alpha 0.960 - Default | 33/44 | +0.9516 | [-0.5190, +2.4221] | 0.2543 |
| Alpha 0.960 - Normal-A8 | 32/31 | -0.0865 | [-1.3841, +1.2976] | 1.0000 |

#### Post-confirmation GSM8K-128 sensitivity

| Candidate - baseline | L/G | Delta (pp) | 95% CI (pp) | p |
| --- | ---: | ---: | ---: | ---: |
| Normal-A8 - Official | 1/1 | 0.0000 | [-2.3438, +2.3438] | 1.0000 |
| Default - Normal-A8 | 0/2 | +1.5625 | [0.0000, +3.9063] | 0.5000 |
| Selector - Default | 4/0 | -3.1250 | [-6.2500, -0.7813] | 0.1250 |
| Selector - Normal-A8 | 2/0 | -1.5625 | [-3.9063, 0.0000] | 0.5000 |
| ULP V1 - Default | 3/0 | -2.3438 | [-5.4688, 0.0000] | 0.2500 |
| ULP V1 - Normal-A8 | 1/0 | -0.7813 | [-2.3438, 0.0000] | 1.0000 |
| Alpha 0.960 - Default | 2/1 | -0.7813 | [-3.9063, +1.5625] | 1.0000 |
| Alpha 0.960 - Normal-A8 | 1/2 | +0.7813 | [-1.5625, +3.9063] | 1.0000 |

The selector and alpha rows extend these old tasks only as sensitivity evidence.
The original v1 Default and exploratory ULP decisions retain their v1 task-specific
margins (`-2 pp` for ARC and `-5 pp` for GSM8K); the common v2 `-5 pp` screen does
not rewrite them.

#### Prospective Stage 1

Each cell is `delta pp [95% CI]; exact p`.

| Candidate - baseline | HellaSwag acc_norm | Winogrande acc |
| --- | --- | --- |
| Normal-A8 - Official | -0.5859 [-1.7578, +0.5859]; 0.5078 | +0.9766 [-0.3906, +2.5391]; 0.3018 |
| Default - Normal-A8 | 0.0000 [-1.3672, +1.3672]; 1.0000 | -1.3672 [-3.5156, +0.7813]; 0.2962 |
| Selector - Default | -0.1953 [-1.3672, +0.9766]; 1.0000 | +2.3438 [+0.1953, +4.4922]; 0.0428 |
| Selector - Normal-A8 | -0.1953 [-1.3672, +1.1719]; 1.0000 | +0.9766 [-0.5859, +2.5391]; 0.3323 |
| ULP V1 - Default | +0.5859 [-0.7813, +1.9531]; 0.5811 | +1.9531 [0.0000, +3.9063]; 0.0872 |
| ULP V1 - Normal-A8 | +0.5859 [-0.5859, +1.7578]; 0.5078 | +0.5859 [-0.9766, +2.1484]; 0.6291 |
| Alpha 0.960 - Default | +0.3906 [-0.9766, +1.7578]; 0.7905 | +0.3906 [-1.7578, +2.5391]; 0.8601 |
| Alpha 0.960 - Normal-A8 | +0.3906 [-0.9766, +1.7578]; 0.7744 | -0.9766 [-2.9297, +0.7813]; 0.4049 |

#### Prospective Stage 2

| Candidate - baseline | L/G | Delta (pp) | Stratified 95% CI (pp) | p |
| --- | ---: | ---: | ---: | ---: |
| Normal-A8 - Official | 21/18 | -0.3348 | [-1.6741, +1.0045] | 0.7493 |
| Default - Normal-A8 | 39/21 | -2.0089 | [-3.6830, -0.3348] | 0.0273 |
| Selector - Default | 27/35 | +0.8929 | [-0.7813, +2.5670] | 0.3742 |
| Selector - Normal-A8 | 31/21 | -1.1161 | [-2.6786, +0.4464] | 0.2116 |
| ULP V1 - Default | 25/33 | +0.8929 | [-0.7813, +2.5670] | 0.3581 |
| ULP V1 - Normal-A8 | 28/18 | -1.1161 | [-2.5670, +0.3348] | 0.1839 |
| Alpha 0.960 - Default | 19/37 | +2.0089 | [+0.4464, +3.6830] | 0.0222 |
| Alpha 0.960 - Normal-A8 | 24/24 | 0.0000 | [-1.4509, +1.4509] | 1.0000 |

Stage 2 shows Default below Normal-A8 by 2.0089 pp and alpha above Default by the
same point estimate; alpha and Normal-A8 have identical aggregate scores. This is
consistent with useful model-level calibration, but does not establish that alpha
is a more faithful local dequantizer.

#### Prospective Stage 3

| Candidate - baseline | L/G | Delta (pp) | 95% CI (pp) | p |
| --- | ---: | ---: | ---: | ---: |
| Normal-A8 - Official | 4/2 | -0.7813 | [-2.7344, +1.1719] | 0.6875 |
| Default - Normal-A8 | 2/3 | +0.3906 | [-1.1719, +2.3438] | 1.0000 |
| Selector - Default | 3/3 | 0.0000 | [-1.9531, +1.9531] | 1.0000 |
| Selector - Normal-A8 | 3/4 | +0.3906 | [-1.5625, +2.3438] | 1.0000 |
| ULP V1 - Default | 0/3 | +1.1719 | [0.0000, +2.7344] | 0.2500 |
| ULP V1 - Normal-A8 | 0/4 | +1.5625 | [+0.3906, +3.1250] | 0.1250 |
| Alpha 0.960 - Default | 0/2 | +0.7813 | [0.0000, +1.9531] | 0.5000 |
| Alpha 0.960 - Normal-A8 | 0/3 | +1.1719 | [0.0000, +2.7344] | 0.2500 |

ULP V1 has the highest Stage 3 point estimate, and its bootstrap interval against
Normal-A8 does not include zero. Only four pairs are discordant, however, and the
exact McNemar value is `p=0.125`. Neither statistic is a protocol gate, so this does
not establish a statistically significant universal improvement. The Stage 3
sample is disjoint from GSM8K-128: ULP and selector are below Normal-A8 on the old
sensitivity set but above it here, with nonsignificant exact paired tests in both
cases. Alpha is above Normal-A8 on both sets, but remains a WikiText-2-calibrated
exploratory arm.

### Extension decision and provenance

All four candidate arms pass the `-5 pp` point-estimate screen against Normal-A8
on both Stage 1 tasks, the Stage 2 micro average, and Stage 3. All generation format
gates pass. Confidence intervals are descriptive, and isolated p-values are not
decision rules. No report authorizes result-dependent arm dropping. The supported
conclusion is that no candidate shows a degradation exceeding the permissive margin
on these fixed subsets, not that the methods are exactly equivalent or universally
better.

Across the prospective stages, alpha equals Normal-A8 on Stage 2 and is 1.1719 pp
above it on Stage 3, while its Stage 1 deltas are small. Selector also passes every
screen and equals Default on Stage 3, but its post-confirm GSM8K-128 result is lower.
These observations support carrying both as engineering options; they do not remove
the WikiText-2 calibration-bias caveat for alpha or establish exact equivalence.

All reports use checkpoint revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`, content SHA256
`4ec0960247ca03fd10a9883d20de08d3795760ac1043fe7a9db6151b4074203f`,
and manifest SHA256
`e8ee68e23f8ed83d251e59a5bc3f9cef77fc7577d7377b14ee1ff0b15e2d0389`.
Each prospective report enforces the `all_six_arms_exact_source_identity` policy.
Its cross-arm paired-sample identity SHA256 is
`5c5e44bebd89dace747e26fce58edea695f961ad2a0e67fc560aa3d54a7c48ef`
for Stage 1,
`9ec1a7a5a3e14d951bcbc5d637c1141d402c5d97a73dd45ecefe35ea018d34bd`
for Stage 2, and
`c40c65277f1322e6725607e72226a617ffcc037929f805d6e3fb1ad72e73d494`
for Stage 3.

The post-confirm reports combine the original four-arm cohort with the later
selector/alpha cohort. Their analyzer requires matching TensorBridge tree, HEAD,
tracked diff, and vendored vLLM identity across cohorts, plus exact source identity
between the new arms. Only the Git dirty/status hash may differ because untracked
run artifacts changed. Their sample/source identity SHA256 values are
`5bf6902d0cb96c9fadd690d95547964b606dd7e802bcfae138525c6aa414ea41`
for GSM8K and
`f4148f9da7446377419609ee62b83042d08411137a4f7db49c7cc313b9757300`
for ARC. Both cohorts predate the selector chunk-row provenance field, so those two
reports explicitly permit its environment record to omit the otherwise required
value 256.

The earlier ARC driver caveat remains: reused ULP job `417989` ran on ACD1-8 with
driver 610.43.02, while the other reused ARC arms ran on ACD1-6 with driver
570.86.10. This does not invalidate the paired accuracy comparison, but it prevents
using those jobs as a same-node performance comparison.

The v2 protocol sets `stop_after_stage3=true` and authorizes no additional tasks.
The extension is complete here; further task selection would require a new protocol
and must not be presented as part of this prospective three-stage evaluation.

## Post-evaluation analytic FPMA default

After the WikiText-2 sweep, the frozen v2 protocol, and all v2 evaluations above
were completed, we adopted a first-order arithmetic interpretation of FPMA's
systematic positive one-ULP mismatch. Five of the eight prefolded-scale residues
are correction-eligible, and one half of the SNC-remapped E2M1 magnitudes have the
eligible parity. Under the balanced residue/parity model:

```text
p(+1 ULP)      = (5/8) * (1/2) = 5/16
E4M3 ULP       = 2^-3 = 1/8
estimated bias = (5/16) * (1/8) = 5/128
alpha_analytic = 1 - 5/128 = 123/128 = 0.9609375
```

TensorBridge therefore uses `alpha=0.961`, rounded to three decimal places, as the
forward `analytic_v1` default for plain FPMA-SNC with Selector and ULP correction
disabled. SNC remains enabled. Explicit Selector or ULP modes retain their neutral
`alpha=1.0` default. This changes only the existing global scale and adds no
steady-state kernel instructions or weight bytes.
The frozen `fpma_default` evaluation arm remains the uncompensated `alpha=1.0`
algorithmic baseline; it no longer denotes the forward runtime default.

This derivation was made after the evaluations above. It is a retrospective
mechanistic interpretation and a forward engineering-default decision, not a
preregistered prediction. The frozen `alpha_0960` arm used exactly `alpha=0.960`;
all tables, protocols, job records, and machine-readable reports continue to
describe that value. No `alpha=0.961` arm was run in this evidence set. The tested
WikiText-2 values 0.960 and 0.962 bracket the analytic value, and the held-out
`alpha=0.960` results are consistent with the predicted neighborhood, but they are
not an independent prospective validation of the exact 0.961 setting.

The `5/16` term is the balanced residue/parity model. It does not replace the
separately reported finite-domain enumeration of 344 mismatches among 1,120
scale/code combinations.

The implicit `analytic_v1` default is fail-closed to the verified nonzero raw E4M3
scale domain `0x39..0x7e`. Checkpoints outside that domain must select alpha
explicitly; TensorBridge does not silently generalize the first-order derivation to
lower-scale or subnormal regimes.

## Performance

Alpha changes only the existing FP32 global scale. The selector changes only
load-time prefolded-scale selection and encoding. They therefore use the same launch
configuration and device instructions as default TensorBridge; any steady-state
difference is measurement noise. Selector load-time work should be reported
separately.

The only compensation that changes the mainloop is ULP. Job `416652` used
interleaved default/ULP CUDA-event timing, 20 samples per label and shape:

| Shape | M | Default (us) | ULP V1 (us) | ULP overhead |
| --- | ---: | ---: | ---: | ---: |
| gate/up `(17408,5120)` | 1 | 25.067 | 30.015 | 19.74% |
| gate/up `(17408,5120)` | 16 | 28.295 | 36.534 | 29.12% |
| gate/up `(17408,5120)` | 128 | 35.903 | 41.573 | 15.79% |
| gate/up `(17408,5120)` | 512 | 104.776 | 123.249 | 17.63% |
| down `(5120,17408)` | 1 | 27.550 | 30.158 | 9.47% |
| down `(5120,17408)` | 16 | 28.563 | 31.530 | 10.39% |
| down `(5120,17408)` | 128 | 34.888 | 39.187 | 12.32% |
| down `(5120,17408)` | 512 | 94.688 | 113.825 | 20.21% |

The geometric-mean kernel overhead is 16.68% (median 16.71%). Preflagging reduces
the earlier implementation's 19.35% geometric-mean overhead by 2.67 percentage
points. SASS keeps 168 registers and reduces aggregate extra static instructions by
31.08% (6,280 to 4,328 across the eight cubins). The remaining measured cost is
consistent with the per-fragment flag/parity combine and subtract rather than an
occupancy loss.
Per-shape instruction counts, register usage, and source cubin hashes are recorded in
`docs/results/tensorbridge_ulp_sass_audit_20260716.json`.

Final-code job `416672` ran on one H100 at node `ACD1-15` with the order
`default -> ULP -> ULP -> default`. Each unique arm first received a separate
64-block prime, and every measured process then scored the same 64 blocks (65,536
unique target tokens). The four measured repetitions were:

| Index | Arm | Engine init (s) | Generation (s) | Prompt tok/s | Unique scored tok/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | default | 150.630 | 25.298 | 5,138.261 | 2,590.595 |
| 1 | ULP V1 | 104.139 | 18.104 | 7,180.132 | 3,620.060 |
| 2 | ULP V1 | 103.765 | 18.016 | 7,215.036 | 3,637.658 |
| 3 | default | 100.341 | 20.152 | 6,450.257 | 3,252.074 |

The ULP generation-time CV was 0.34%, but the default CV was 16.01%; default
engine-init CV was 28.34%. Arithmetic means would misleadingly show ULP throughput
24.22% above default, while the two position-paired ratios range from +11.86% to
+39.74%. This is inconsistent with the positive isolated-kernel cost and cannot be
attributed to ULP. With only two samples per arm and uncontrolled clocks/system
state, the defensible observation is only that this run did not show an end-to-end
slowdown. System overhead remains inconclusive; job `416652` is the primary
performance evidence. The two default repetitions also produced slightly different
64-block PPL values (6.702268 and 6.698444), so these short timing runs are not used
as model-accuracy evidence.

GPU clock locking was unavailable in the system jobs. The four-arm job `416484`
showed large differences that cannot be attributed to the zero-instruction-cost arms
(for example, the two alpha runs differed by 1.92x). System throughput must therefore
be reported with raw repetitions and an uncontrolled-clock/system-state caveat; the
interleaved CUDA-event Linear benchmark is the primary ULP overhead measurement.

## Tensor parallel and CUDA Graph validation

The production TensorBridge path was also validated with vLLM
`FULL_DECODE_ONLY` CUDA Graphs at TP=1 and TP=2. This is an execution-mode
equivalence probe, not another WikiText-2 PPL claim: it uses one 248-token prompt
block, generates eight tokens, and compares eager against graph replay over repeated
runs. Every graph run must contain exactly one eager `NONE` prefill followed by
seven `FULL` decode dispatches at capture size one, with no padding or fallback.

Block 0 was rejected as a correctness probe after one of 22 eager repeats changed
the fourth greedy token. A 32-block eager screen selected block 26 because all eight
chosen-token logprobs were at least `-0.435` (chosen probability at least about
0.647). This input selection only avoids an autoregressive decision boundary; it
does not select a favorable accuracy score. Both final arms use 2 warmups plus 50
measured repeats and retain the exact-token hard gate.

| Configuration | Job | Decode delta-NLL 90% CI | Chosen-probability ratio | RMS / max-T p | Median eager / graph | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TP=1, 1 H100, 8 CPU | `430015` | `[-0.002813, 0.000144]` | 1.001336 | 0.306 / 0.0538 | 0.6620 s / 0.2015 s | passed |
| TP=2, 2 H100, 16 CPU | `430103` | `[-0.002121, 0.000926]` | 1.000597 | 0.165 / 0.371 | 0.6645 s / 0.1857 s | passed |

All prompt, prefill-position-0, decode, scalar-noise, vector-RMS-noise,
permutation, token-identity, protocol, dispatch, and driver-memory gates pass.
TP=2 records two visible GPUs and leaves each worker's BLAS/OpenMP limits at eight.
The small-probe median generation speedups are about 3.29x (TP=1) and 3.58x
(TP=2); they demonstrate replay is active but are not serving-throughput numbers.

vLLM 0.20.2 automatically enables its optional FlashInfer fused allreduce+RMS path
for TP on Hopper. On nodes with the 570.86.10 driver, that path stalled during
CUDA symmetric-memory multicast initialization. TensorBridge's FDO TP launcher now
sets `pass_config.fuse_allreduce_rms=false`; ordinary TP all-reduce remains enabled.
The resolved setting is recorded and checked before generation. Job `430103`
passes after this change; job `430032` is the retained failure diagnostic.

Nsight Systems job `430132` independently profiles the TP=2 graph path. The vLLM
runtime records 84 graph decode dispatches (12 runs times seven decode forwards),
while the rank-0 CUDA API trace captures 49 `cudaGraphLaunch_v10000` calls and 49
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` rows. The captured count is a lower bound because
only the traced worker/window is represented; the important external check is that
real CUDA graph launch activity is present. The 7.9 MiB report SHA256 is
`0035d1d49dc25a499bac887176821cde89d589940b545b5a30cbe4b6c769a0af`.
Full paths, hashes, gates, and statistics are in
`docs/results/tensorbridge_tp_cudagraph_20260722.json`.

## Reproduction

Run the GPU test gate:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
sbatch sbatch/run_tensorbridge_vllm_pytest.sbatch
```

Run the same-node TP=2 eager/graph pair with:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
PPL_PAIR_REPEAT_RUNS=52 PPL_BLOCK_START=26 \
sbatch --gres=gpu:2 --cpus-per-task=16 --mem=160G \
  --export=ALL,PPL_TENSOR_PARALLEL_SIZE=2 \
  sbatch/run_eval_nvfp4_fdo_pair.sbatch
```

The paired launcher disables the optional allreduce-RMS fusion for TP>1 by default.
It does not disable vLLM's ordinary tensor-parallel collective path.

Run full ULP PPL with an independent cache:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
export PPL_BACKEND=tensorbridge
export PPL_MAX_BLOCKS=all
export PPL_FPMA_ULP_CORRECTION=1
export PPL_FPMA_ALPHA=1.0
export PPL_FPMA_PREFOLD_SELECTOR=none
export TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP=0
export PPL_TENSORBRIDGE_CACHE_DIR="$PWD/benchmarks/tmp/ppl/ulp_scale_msb_flag_v1"
export PPL_OUTPUT="$PWD/benchmarks/results/ppl/wikitext2_tensorbridge_ulp_v1.json"
sbatch --export=ALL sbatch/run_eval_nvfp4_wikitext2_ppl.sbatch
```

The adapter adds both ULP device flags. Do not reuse an unversioned or older ULP
cache with V1-marked scale bytes.

Run the controlled same-node performance wrapper with:

```bash
cd /data/user/jzou521/codes/cuda/tensorbridge-vllm
PPL_PERF_MAX_BLOCKS=64 \
PPL_PERF_ORDER='default ulp ulp default' \
sbatch --export=ALL sbatch/run_eval_nvfp4_ppl_same_node_perf.sbatch
```

The wrapper primes each unique arm, records engine-init and generation timing in
JSON, and saves the explicit phase/index/order metadata.

## Artifacts and limitations

The canonical Linear/PPL machine-readable summary is
`docs/results/tensorbridge_accuracy_20260716.json`. The immutable v1 confirmation
reports and all five v2 sensitivity/prospective reports are enumerated with their
SHA256 values above. Raw result and sample JSON artifacts remain under
`benchmarks/results/` and are intentionally ignored by Git.

The earlier PPL artifacts do not record an exact Git revision or source-bundle
hash. The lm-eval artifacts record TensorBridge HEAD, tracked diff state, a
TensorBridge source-tree hash, and vendored vLLM HEAD/diff state before and after
each run. The vendored vLLM dependency is still a dirty editable checkout, and its
untracked `sitecustomize.py` content is not yet bound by these reports. Clean
revisions or fully hashed source bundles for both components are still required for
publication-grade reproduction claims.

The v1 confirmation and the separately authorized v2 three-stage extension are both
complete under their respective stop rules. No further task is part of either
protocol. Work beyond this evidence should first freeze a new protocol and clean
source bundle. Independent engineering follow-up remains useful for controlled-clock
system timing and Linear MSE on held-out real-activation traces; the unrelated
generic FP8 large-M investigation remains separate from TensorBridge FPMA accuracy.
