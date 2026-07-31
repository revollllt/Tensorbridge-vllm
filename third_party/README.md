# third_party/tensorbridge

The TensorBridge kernel dependency, vendored so this repository alone is enough
to reproduce the accuracy results on a fresh machine.

```
tensorbridge/                                              source at the pinned commit
tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl  the artifact to install
```

## Install the wheel, do not install the source

```bash
uv pip install third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
```

`docs/TENSORBRIDGE_INTEGRATION.md` makes the dependency one-way on purpose: the
integration imports only `tensorbridge.api.v1` and consumes the pinned wheel, so
a kernel feature branch cannot silently move vLLM's numbers. An editable install
(`pip install -e third_party/tensorbridge`) belongs only in a disposable
joint-debug environment, never in an environment that produces reported results.

The source here is for reading and rebuilding, not for importing.
`tests/test_tensorbridge_dependency_boundary.py` enforces the import boundary.

## Provenance

| Field | Value |
| --- | --- |
| Repository | `tensorbridge` (sibling checkout) |
| Commit | `43cc2aa3d9a13d1f568f9eeb9dd8fbcdeb9a83bc` |
| Tree state at build | clean |
| Wheel SHA256 | `7b3aaa6199e22655a7d81a18bbdae695110aa05b9c98a0c05c516473e686ec09` |
| Retained source files | 124, listed in `tensorbridge.manifest.sha256` |

`constraints/tensorbridge.json` is the authority for the commit and the wheel
hash; the table restates them so this directory is self-describing.

The source came from `git archive` at that commit, not from a working tree. The
distinction matters: the sibling checkout has since accumulated uncommitted
changes to the FPMA dequant path, the WGMMA layer, and the loaders, so a
working-tree copy would not correspond to this wheel.

## Verifying

```bash
sha256sum third_party/tensorbridge_kernels-0.2.0+g43cc2aa3d9a1-py3-none-any.whl
# 7b3aaa6199e22655a7d81a18bbdae695110aa05b9c98a0c05c516473e686ec09

( cd third_party/tensorbridge && LC_ALL=C sha256sum -c ../tensorbridge.manifest.sha256 ) \
  | grep -v ': OK$' || echo "all 124 files match"
```

Every retained file is listed individually, so each one still verifies against
the upstream commit even though the tree is a subset.

`constraints/tensorbridge.json` also carries a `snapshot_sha256` over the
*complete* tree at that commit
(`ba85df9808c08f7afbd9223004dc10871b98bd6c2fc8e23f3c73082d2a922b99`). It does not
apply here and cannot be recomputed from this directory — reproducing it needs a
full `git archive` of the commit. That whole-tree form is also why the manifest
uses null-delimited sorting: one upstream documentation filename contains spaces,
and a whitespace-splitting pipeline silently yields a different digest.

## What was removed

The upstream tree is 290 files; 124 are kept. Dropped: `ITERATIONS.md` (a 459 KB
development log), `HINTS.md`, `docs/` (including a ~1 MB third-party conference
paper PDF), `analysis/`, `benchmarks/`, `sbatch/`, `scripts/`, `tasks/`,
`tests/`, `cute_cutlass_nvfp4a8/`, and the linter and editor configs.

Nothing removed participates in the build: `pyproject.toml` already excludes
`tests`, `scripts`, and `cute_cutlass_nvfp4a8` from package discovery, and
`package-data` reaches only `tensorbridge/csrc/launcher/*.{cpp,h}` and
`tensorbridge/include/**/*.cuh`. All 7 launcher sources, 52 headers, and 59
Python modules are present.

What is kept is the kernel implementation, the build files, and the licences.
Read the FPMA dequantization at
`tensorbridge/include/tensorbridge/datatype/dequant_fused.cuh`.

## Rebuilding

```bash
TENSORBRIDGE_BUILD_VERSION=0.2.0+g43cc2aa3d9a1 \
  uv pip wheel third_party/tensorbridge -w /tmp/tb-rebuild
```

The version override is required here. `setup.py` derives the version from `git
rev-parse HEAD`, and this directory is a source export with no git history, so
without it the build produces `0.2.0+source` instead of the pinned version.

A rebuild should reproduce the wheel hash. If it does not, prefer the vendored
wheel and treat the difference as unexplained rather than benign — every accuracy
number in `eval_simple/README.md` and `docs/FPMA_ACCURACY_REPRO.md` is a property
of this specific build.
