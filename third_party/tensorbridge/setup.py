from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup


BASE_VERSION = "0.2.0"
REPO_ROOT = Path(__file__).resolve().parent


def _build_version() -> str:
    override = os.environ.get("TENSORBRIDGE_BUILD_VERSION")
    if override is not None:
        if not re.fullmatch(r"0\.2\.0\+g[0-9a-f]{7,40}", override):
            raise RuntimeError(f"invalid TENSORBRIDGE_BUILD_VERSION: {override!r}")
        return override

    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return f"{BASE_VERSION}+source"
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"git returned an invalid TensorBridge revision: {sha!r}")
    return f"{BASE_VERSION}+g{sha[:12]}"


setup(version=_build_version())
