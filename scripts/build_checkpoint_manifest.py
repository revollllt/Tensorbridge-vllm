#!/usr/bin/env python3
"""Build a reusable SHA256 manifest for a TensorBridge evaluation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm.plugins.tensorbridge_evaluation.lm_harness import (
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)


DEFAULT_MODEL = Path("/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        identity = verify_checkpoint_manifest(args.model, args.output)
        print(json.dumps(identity, indent=2, sort_keys=True))
        return

    manifest = write_checkpoint_manifest(
        args.model,
        args.output,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    encoded = args.output.read_bytes()
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                "checkpoint_content_sha256": manifest[
                    "checkpoint_content_sha256"
                ],
                "source": manifest["source"],
                "expected_checkpoint_verified": manifest["source"] is not None,
                "weight_shards": manifest["weight_shards"],
                "weight_bytes": manifest["weight_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
