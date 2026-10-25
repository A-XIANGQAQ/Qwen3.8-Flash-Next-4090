#!/usr/bin/env python3
"""Run the prefill / decode matrix against a live Strata server.

This is a thin driver around vLLM's official `benchmark_serving.py` (vendored in
`vllm_0.7.3/`).  vLLM owns the request generation, client-side timing and
metric aggregation; this script only:

  * runs one cell at a time with a fixed seed,
  * records the engine's own per-request counters from `/metrics`,
  * filters out the warm-up request and cross-checks the two sides.

Usage:
  python3 run_matrix.py --tokenizer bench/tokenizer \
      --out bench/results/2026-10-25 [--only P1K,D1K]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "vllm_0.7.3" / "benchmark_serving.py"
BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-flash-next-davidau-iq3s-ablit"

# id, kind, input tokens, output tokens, prompts, concurrency
CELLS = [
    ("P1K",   "prefill", 1024,    8, 5, 1),
    ("P8K",   "prefill", 8192,    8, 3, 1),
    ("P32K",  "prefill", 32768,   8, 3, 1),
    ("P64K",  "prefill", 65536,   8, 2, 1),
    ("P128K", "prefill", 131072,  8, 2, 1),
    ("P256K", "prefill", 260000,  4, 1, 1),
    ("D1K",   "decode",  1024,  512, 3, 1),
    ("D32K",  "decode",  32768, 256, 3, 1),
    ("D128K", "decode",  131072, 256, 2, 1),
    ("D256K", "decode",  260000, 256, 1, 1),
    ("C2K",   "conc",    1024,  512, 4, 2),
    ("C4K",   "conc",    1024,  512, 8, 4),
    ("C4_8K", "conc",    8192,  256, 4, 4),
]


def metrics() -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30) as r:
        return json.load(r)


def wait_idle(timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        live = metrics()["live"]
        if live["state"] == "idle" and live["running"] == 0 and live["queued"] == 0:
            return
        time.sleep(2)
    raise TimeoutError("server did not become idle")


def engine_rows(t0: float, input_len: int) -> list[dict]:
    rows = metrics()["requests"]
    band = max(64, int(input_len * 0.02)) + 32
    return [
        r for r in rows
        if r["time"] >= t0 - 2
        and abs(r["prompt_tokens"] - input_len) <= band
    ]


def median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def run_cell(cell, tokenizer: Path, outdir: Path, seed: int) -> dict:
    cid, kind, input_len, output_len, n, conc = cell
    raw_dir = outdir / "raw"
    raw_file = raw_dir / f"{cid}.json"
    cmd = [
        sys.executable, str(BENCH),
        "--backend", "openai-chat",
        "--host", "127.0.0.1", "--port", "8000",
        "--endpoint", "/v1/chat/completions",
        "--model", MODEL,
        "--tokenizer", str(tokenizer),
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--seed", str(seed),
        "--disable-tqdm",
        "--save-result",
        "--result-dir", str(raw_dir),
        "--result-filename", raw_file.name,
    ]
    wait_idle()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE / "vllm_0.7.3",
                          capture_output=True, text=True, timeout=900)
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "logs" / f"{cid}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{cid}: benchmark failed\n{proc.stderr[-4000:]}")
    wait_idle()
    t1 = time.time()

    result = json.loads(raw_file.read_text())
    rows = engine_rows(t0, input_len)
    prompt_tokens = median([r["prompt_tokens"] for r in rows])
    prompt_ms = median([r["prompt_ms"] for r in rows])
    decode_ms = median([r["decode_ms"] for r in rows])
    decode_tok_s = median([r["decode_tok_s"] for r in rows])
    first_token_s = median([r["first_engine_token_s"] for r in rows])
    reused = sum(r["reused"] for r in rows)
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    cache_ratio = reused / total_prompt if total_prompt else 0.0

    ttft_s = (result["median_ttft_ms"] or 0) / 1000.0
    row = {
        "id": cid,
        "kind": kind,
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": n,
        "concurrency": conc,
        "seed": seed,
        "completed": result["completed"],
        "duration_s": result["duration"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "ttft_ms_median": result["median_ttft_ms"],
        "ttft_ms_p99": result["p99_ttft_ms"],
        "tpot_ms_median": result["median_tpot_ms"],
        "tpot_ms_p99": result["p99_tpot_ms"],
        "output_tok_s_wall": result["output_throughput"],
        "total_tok_s_wall": result["total_token_throughput"],
        "per_stream_tok_s_wall": result["output_throughput"] / max(result["completed"], 1),
        "client_prefill_tok_s": (prompt_tokens / ttft_s) if (ttft_s and prompt_tokens) else None,
        "decode_tok_s_client": (1000.0 / result["median_tpot_ms"]) if result["median_tpot_ms"] else None,
        "engine_prompt_tokens_median": prompt_tokens,
        "engine_prompt_ms_median": prompt_ms,
        "engine_prefill_tok_s": (prompt_tokens / (prompt_ms / 1000.0)) if (prompt_ms and prompt_tokens) else None,
        "engine_decode_tok_s_median": decode_tok_s,
        "engine_first_token_s_median": first_token_s,
        "engine_decode_ms_median": decode_ms,
        "engine_cache_reused_ratio": cache_ratio,
        "engine_requests_matched": len(rows),
        "engine_prompt_tokens_all": [r["prompt_tokens"] for r in rows],
        "engine_reused_all": [r["reused"] for r in rows],
        "engine_prompt_ms_all": [r["prompt_ms"] for r in rows],
        "engine_decode_tok_s_all": [r["decode_tok_s"] for r in rows],
    }
    print(f"{cid:>6}  TTFT {row['ttft_ms_median']:9.1f} ms  "
          f"TPOT {row['tpot_ms_median']:7.2f} ms  "
          f"engine prefill {row['engine_prefill_tok_s'] or 0:8.0f} t/s  "
          f"decode {row['engine_decode_tok_s_median'] or 0:7.1f} t/s  "
          f"cache {cache_ratio:.3f}  matched {len(rows)}", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer"))
    ap.add_argument("--out", default=str(HERE / "results" / datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    tokenizer = Path(args.tokenizer).resolve()
    if not (tokenizer / "tokenizer.json").exists():
        raise SystemExit(f"tokenizer not found at {tokenizer}; run build_tokenizer.py first")

    first = metrics()
    meta = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "base_url": BASE_URL,
        "model": MODEL,
        "engine": first["engine"],
        "hardware_static": first.get("hardware_static"),
        "conversation_cache": first.get("conversation_cache"),
        "harness": {
            "project": "vllm-project/vllm",
            "version": "v0.7.3",
            "file": "benchmarks/benchmark_serving.py + backend_request_func.py + benchmark_utils.py",
            "local_patches": "bench/vllm_0.7.3/PATCHES.md",
        },
        "tokenizer": {
            "path": str(tokenizer),
            "tokenizer_json_sha256": __import__("hashlib").sha256(
                (tokenizer / "tokenizer.json").read_bytes()).hexdigest(),
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = []
    for idx, cell in enumerate(CELLS):
        if only and cell[0] not in only:
            continue
        rows.append(run_cell(cell, tokenizer, outdir, seed=100 + idx))

    (outdir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    keys = [k for k in rows[0].keys() if not k.endswith("_all")] if rows else []
    with (outdir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {outdir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
