"""Small, key-selective safetensors checkpoint helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def resolve_safetensor_key_files(
    checkpoint: str | Path,
    keys: list[str] | tuple[str, ...],
) -> dict[str, Path]:
    """Resolve requested tensor keys without opening unrelated checkpoint shards."""
    root = Path(checkpoint)
    if root.is_file():
        return {key: root for key in keys}
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {root}")

    single_file = root / "model.safetensors"
    if single_file.is_file():
        return {key: single_file for key in keys}

    index_file = root / "model.safetensors.index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"missing model.safetensors or index under {root}")
    with index_file.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})

    missing = [key for key in keys if key not in weight_map]
    if missing:
        raise KeyError(f"checkpoint index is missing requested keys: {missing}")
    resolved = {key: root / weight_map[key] for key in keys}
    missing_files = sorted({str(path) for path in resolved.values() if not path.is_file()})
    if missing_files:
        raise FileNotFoundError(f"checkpoint shard files are missing: {missing_files}")
    return resolved


def load_safetensor_keys(
    checkpoint: str | Path,
    keys: list[str] | tuple[str, ...],
    *,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load only requested keys and return each key's source shard."""
    key_files = resolve_safetensor_key_files(checkpoint, keys)
    by_file: dict[Path, list[str]] = defaultdict(list)
    for key, filename in key_files.items():
        by_file[filename].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for filename, file_keys in by_file.items():
        with safe_open(str(filename), framework="pt", device=device) as handle:
            available = set(handle.keys())
            missing = [key for key in file_keys if key not in available]
            if missing:
                raise KeyError(f"{filename} is missing requested keys: {missing}")
            for key in file_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors, {key: str(filename) for key, filename in key_files.items()}
