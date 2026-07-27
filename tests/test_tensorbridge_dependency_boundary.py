from pathlib import Path

from tensorbridge.api import v1


def test_runtime_api_contract() -> None:
    assert v1.RUNTIME_API_VERSION == 1
    assert v1.__version__.startswith("0.2.0+g")
    assert v1.default_fpma_alpha() == 0.961


def test_plugin_uses_only_versioned_tensorbridge_api() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "vllm/plugins/tensorbridge.py"
    ).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith("from tensorbridge")]
    assert imports == ["from tensorbridge.api.v1 import ("]


def test_sbatch_defaults_deep_gemm_off_unless_explicitly_enabled() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "sbatch/_common.sh"
    ).read_text(encoding="utf-8")
    assert 'export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"' in source
