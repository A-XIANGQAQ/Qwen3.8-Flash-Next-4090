#!/usr/bin/env python3
"""Which GGUF shard holds the per-layer-embedding (PLE) table?

`iq_pack.py` leaves `per_layer_token_embd.weight` out of the pack and the engine
reads it from the GGUF named by `--ple-gguf`.  That table is not always in the
first shard, so check the tensor index before writing the run config.

Usage:
  python3 find_ple_shard.py --strata /path/to/Strata <shard1.gguf> [shard2.gguf ...]

`--strata` is the Strata source tree (the one containing `tools/_paths.py` and
`third_party/llama.cpp/gguf-py`).  Only tensor indexes are read; no weights.
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default=os.environ.get("STRATA_DIR"),
                    help="Strata source tree (contains tools/_paths.py)")
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    if not args.strata:
        ap.error("--strata (or env STRATA_DIR) is required")

    sys.path.insert(0, os.path.join(args.strata, "tools"))
    import _paths  # noqa: E402

    _paths.add_gguf_py()
    from gguf import GGUFReader  # noqa: E402

    want = ("per_layer_token_embd", "ple")
    for path in args.shards:
        try:
            reader = GGUFReader(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: cannot read index ({exc})")
            continue
        names = [t.name for t in reader.tensors]
        hits = [n for n in names if any(w in n for w in want)]
        total = sum(t.n_bytes for t in reader.tensors) / 1e9
        print(f"{path}\n  tensors={len(names)} bytes={total:.2f} GB")
        for name in hits[:6]:
            tensor = next(t for t in reader.tensors if t.name == name)
            print(f"  PLE: {name}  {tensor.n_bytes / 1e9:.2f} GB"
                  f"  shape={list(tensor.shape)}")
        if not hits:
            print("  no PLE tensor here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
