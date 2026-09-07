#!/usr/bin/env python3
"""Verify every shard referenced by a safetensors index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (args.model_dir / name).is_file()]
    empty = [
        name
        for name in shards
        if (args.model_dir / name).is_file() and (args.model_dir / name).stat().st_size == 0
    ]
    result = {
        "index_shards": len(shards),
        "expected_shards": args.expected_shards,
        "missing": missing,
        "empty": empty,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if len(shards) == args.expected_shards and not missing and not empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
