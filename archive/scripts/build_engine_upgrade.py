#!/usr/bin/env python3
"""Build the Strata 0.1.45 engine + GPU vision encoder from the v0.1.45 git worktree.

Why not ./update.sh or setup.py: setup's main() rewrites the run config
(carry_over() drops hand-added args) and, as of 0.1.44+, its shard check can
delete model shards on a false SHA-256 mismatch (#2902).  build_engine() is the
only piece we want: it compiles `strata`, `strata-vision` and, new in 0.1.44,
the K-quant prompt library `libstrata_kq.so`, then writes engine/BUILD.json.

Run with the installed tree's venv (setup.py imports nothing exotic, but that
interpreter is the one setup already validated):
  <installed>/.venv/bin/python scripts/build_engine_upgrade.py --src Strata-0.1.45
"""
import importlib.util
import os
import sys
import time

SRC = os.environ.get("STRATA_SRC") or (sys.argv[1] if len(sys.argv) > 1 else "Strata-0.1.45")
if not os.path.isfile(os.path.join(SRC, "setup.py")):
    sys.exit(f"no setup.py under {SRC!r}: pass the worktree path as the first argument")

spec = importlib.util.spec_from_file_location("s145", os.path.join(SRC, "setup.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)          # __name__ != "__main__", so main() does not run

print("worktree  :", m.ROOT, flush=True)
print("version   :", m.source_version(), flush=True)
gpu = m.gpu_info(0)
print("gpu       :", gpu.get("name"), "arch", gpu.get("arch"), flush=True)
llama = m.get_llama_cpp()
print("llama.cpp :", llama, flush=True)

t0 = time.time()
eng = m.build_engine(gpu, "gpu", True, llama, toolkit=13)
print(f"BUILD_DONE {eng} in {time.time() - t0:.1f}s", flush=True)
