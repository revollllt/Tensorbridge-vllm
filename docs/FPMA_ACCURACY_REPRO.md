# FPMA Accuracy Reproduction

How much does TensorBridge's FPMA approximate dequantization cost in accuracy,
once the analytic alpha compensation is applied? This document reproduces the
WikiText-2 perplexity and full GSM8K measurements that answer that.

These runs are engineering evidence. They are **not** part of the frozen v1
confirmation or v2 expansion protocols in `ACCURACY_EVALUATION.md`, and their
results must not be filed as protocol evidence.

## What is being compared

| Arm | Path | Isolates |
| --- | --- | --- |
| `official` | NVFP4 weights, Marlin W4A16, BF16 activation | reference checkpoint path |
| `normal_a8` | B8 expanded exactly at load, vLLM Cutlass FP8, dynamic per-token A8 | exact-arithmetic counterpart of FPMA |
| `alpha_0961` | FPMA-SNC in the mainloop, same A8 path, `alpha = 0.961` | FPMA plus analytic compensation |

`normal_a8` is the required baseline for the FPMA question, not `official`.
Both `normal_a8` and `alpha_0961` use the same B8 weights, the same dynamic
per-token FP8 activation, and the same `6*g` epilogue scale. The only difference
is how the B8 fragments are produced: an exact load-time expansion versus the
in-mainloop integer-add approximation. Comparing against `official` instead
would confound FPMA with the activation dtype change.

In all arms `lm_head` stays NVFP4 Marlin W4A16 with BF16 activation, and the 208
FP8 layers are identical. Only the 192 transformer MLP projections differ.

## Prerequisites

The `alpha_0961` arm is a local addition to `ARMS` in
`vllm/plugins/tensorbridge_evaluation/lm_harness.py`, with the matching entries in
`tests/test_lm_harness_contract.py::test_arm_contracts_are_fixed` and in the
`ARMS` array and cache-seed case of `sbatch/run_eval_nvfp4_lm_harness.sbatch`.
It is deliberately appended last so the frozen six-arm indices 0..5 are unchanged.

Build the checkpoint manifest once. lm-eval verifies it before and after every
run and refuses to start without it:

```bash
TENSORBRIDGE_BUILD_CHECKPOINT_MANIFEST=1 \
  sbatch --time=01:30:00 --export=ALL sbatch/run_tensorbridge_vllm_pytest.sbatch
```

That job also runs the pytest gate, which should report `110 passed`.

Then:

```bash
./sbatch/run_fpma_accuracy_repro.sh preflight
```

This verifies the pinned wheel against `constraints/tensorbridge.json`, checks
that `resolve_arm("alpha_0961")` yields `(tensorbridge, 0.961, none, False)`, and
re-verifies the checkpoint manifest.

## The working-tree constraint

lm-eval hashes `git status --porcelain=v1` before and after the run and raises
`RuntimeError: source state changed while lm-eval was running` on any change.
**Untracked files count.** Creating a new file anywhere in the repo while an arm
is running kills that arm after it has already finished evaluating, and the
results are lost: the guard fires before the result JSON is written, leaving only
a small error record and no samples directory.

Freeze the working tree before submitting GSM8K and leave it alone until every
arm reports `COMPLETED`. `run_fpma_accuracy_repro.sh gsm8k` prints the tree hash
it is committing to.

## Running

```bash
./sbatch/run_fpma_accuracy_repro.sh ppl      # 3 arms, full WikiText-2
./sbatch/run_fpma_accuracy_repro.sh gsm8k    # array 0,1,6 sharing one RUN_ID
./sbatch/run_fpma_accuracy_repro.sh analyze <run_id>
```

`TENSORBRIDGE_REPRO_WITH_ALPHA1=1` adds the uncompensated `alpha=1.0` PPL arm.
That arm doubles as the discriminator against the frozen record: alpha plays no
part at 1.0, so reproducing the recorded value proves the kernel is unchanged.

The GSM8K array form matters. All arms share one `RUN_ID`, so their results land
in the same directory, which is what allows the paired analysis.

## Expected results

Environment: Python 3.12.12, vLLM 0.20.2+cu128, torch 2.11.0+cu128,
`tensorbridge-kernels` 0.2.0+g43cc2aa3d9a1, lm-eval 0.4.11, transformers 5.9.0,
datasets 4.8.5, `cuda/12.8` + `gcc/13.3`, one H100 80GB (sm90) per arm.

### WikiText-2, 297,192 scored tokens over 291 blocks

| Arm | Mean NLL | PPL | alpha source | Job |
| --- | ---: | ---: | --- | ---: |
| `official` | 1.9510530810 | 7.036093257 | `neutral_default` | 455086 |
| `normal_a8` | 1.9536234338 | 7.054201761 | `neutral_default` | 455087 |
| `alpha_0961` | 1.9470471293 | 7.007963388 | `analytic_v1` | 455084 |
| `fpma_default` (alpha=1.0) | 1.9775391640 | 7.224941694 | `explicit_cli` | 455372 |

`normal_a8` reproduces the recorded `1.953623434 / 7.054201761` on every
published digit. `fpma_default` reproduces the recorded
`1.977539164 / 7.224941694` exactly, which is what establishes that this kernel
build is arithmetically identical to the one behind the frozen tables.
`official` lands 3.2e-5 NLL above its recorded value; the gap is three orders of
magnitude below the inter-arm differences and does not affect any comparison.

Uncompensated FPMA costs +2.42% PPL against `normal_a8`. With `alpha=0.961` the
arm is 0.66% *below* `normal_a8`. The compensation is doing real work; "FPMA is
free" is only true with it enabled.

### GSM8K `generation_core`, full 1319 documents

| Arm | exact_match | Correct |
| --- | ---: | ---: |
| `official` | 96.2851% | 1270 / 1319 |
| `alpha_0961` | 96.0576% | 1267 / 1319 |
| `normal_a8` | 95.9060% | 1265 / 1319 |

Paired (exact McNemar over discordant pairs; run id 455371):

| Comparison | win/loss | Δ pp | 95% CI pp | p |
| --- | ---: | ---: | ---: | ---: |
| **`alpha_0961` − `normal_a8`** | 10/8 | **+0.1516** | [−0.4788, +0.7821] | 0.8145 |
| `alpha_0961` − `official` | 10/13 | −0.2274 | [−0.9401, +0.4852] | 0.6776 |
| `normal_a8` − `official` | 8/13 | −0.3791 | [−1.0600, +0.3019] | 0.3833 |

Agreement: 1250 documents correct in all three arms, 38 wrong in all three, 31
disputed.

The arms score the same documents, so the paired form is the right one; the
analyzer hard-fails unless `doc_id` and `doc_hash` match across arms. The
unpaired two-proportion comparison discards most of the power and is not used.

### Conclusion

On the primary comparison, FPMA with analytic alpha compensation is +0.15 pp on
GSM8K with a 95% interval inside ±0.8 pp, and −0.66% on WikiText-2 PPL. Neither
benchmark detects an accuracy cost. The preregistered GSM8K margin in the frozen
protocol was −5 pp, far outside this interval.

## Known open item

The `alpha_0961` PPL value of 7.007963 sits 0.0096 below the bracket implied by
the recorded alpha sweep (0.960 gives 7.017586, 0.962 gives 7.018349), roughly
twelve times the spread of that bracket. Eliminated so far: kernel drift (the
alpha=1.0 arm reproduces exactly), cache or compile differences (the seed digest
mismatch traces to absolute paths in `signature.txt` and `cmdline.json`; no
`kernel.cu` contains a path), mutation inside
`validate_analytic_fpma_scale_domain` (it is read-only), an alpha value
difference between the implicit and explicit paths (both resolve to exactly
0.961), and a token-count difference in the sweep.

The remaining discriminator is an explicit `PPL_FPMA_ALPHA=0.960` run checked
against 7.017586098.

This does not affect the GSM8K numbers: lm-eval's `configure_environment` always
writes `TENSORBRIDGE_NVFP4_FPMA_ALPHA` explicitly, so no GSM8K arm uses the
implicit `analytic_v1` path.

## What these runs do not establish

Throughput. The arms ran concurrently on different nodes without clock pinning,
so the observed generation rates (139.5, 137.9, and 103.6 tok/s for `official`,
`normal_a8`, and `alpha_0961`) are not a controlled measurement. Use
`sbatch/run_eval_nvfp4_ppl_same_node_perf.sbatch`, which pins clocks and
alternates arm order on one node, for any performance claim.
