#!/usr/bin/env python3
"""Verify the installed TensorBridge runtime against the pinned wheel manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT = REPO_ROOT / "constraints/tensorbridge.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    constraint = json.loads(CONSTRAINT.read_text(encoding="utf-8"))
    wheel = (REPO_ROOT / constraint["wheel"]["relative_path"]).resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"pinned TensorBridge wheel is missing: {wheel}")
    if wheel.name != constraint["wheel"]["filename"]:
        raise RuntimeError(
            f"TensorBridge wheel filename mismatch: {wheel.name} != "
            f"{constraint['wheel']['filename']}"
        )
    wheel_size = wheel.stat().st_size
    if wheel_size != constraint["wheel"]["size_bytes"]:
        raise RuntimeError(
            f"TensorBridge wheel size mismatch: {wheel_size} != "
            f"{constraint['wheel']['size_bytes']}"
        )
    wheel_sha256 = _sha256(wheel)
    if wheel_sha256 != constraint["wheel"]["sha256"]:
        raise RuntimeError(
            f"TensorBridge wheel hash mismatch: {wheel_sha256} != "
            f"{constraint['wheel']['sha256']}"
        )

    from tensorbridge.api import v1

    installed_version = importlib.metadata.version("tensorbridge-kernels")
    if installed_version != constraint["package"]["version"]:
        raise RuntimeError(
            f"TensorBridge version mismatch: {installed_version} != "
            f"{constraint['package']['version']}"
        )
    if v1.__version__ != installed_version:
        raise RuntimeError(
            f"TensorBridge API version mismatch: {v1.__version__} != {installed_version}"
        )
    if v1.RUNTIME_API_VERSION != constraint["package"]["runtime_api"]:
        raise RuntimeError(
            f"TensorBridge runtime API mismatch: {v1.RUNTIME_API_VERSION} != "
            f"{constraint['package']['runtime_api']}"
        )

    print(
        json.dumps(
            {
                "constraint": str(CONSTRAINT),
                "installed_version": installed_version,
                "runtime_api": v1.RUNTIME_API_VERSION,
                "wheel": str(wheel),
                "wheel_sha256": wheel_sha256,
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
