#!/usr/bin/env python3
"""Single-variable A/B probe for the Strata 0.1.43.1 -> 0.1.45 upgrade.

Measures, from the client side over the OpenAI-compatible API:
  decode   : tok/s after the first token (6 runs, median).  Counts reasoning
             deltas too - Qwen3.8 thinks first, and a 256-token cap is spent
             entirely on thinking, which made an earlier version of this probe
             look like an empty reply.
  ttft     : time to first token at 1K / 32K / 100K prompts (fresh text, so the
             conversation cache cannot answer for us)
  batch    : wall time for 4 concurrent 512-token replies
  vision   : whether the three bars (red / green / blue) are named correctly

Usage:
  python ab_0145.py --arm A            # writes bench/ab_<arm>.json + appends jsonl
"""
import argparse
import base64
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
BASE = "http://127.0.0.1:8000"

# Fixed filler so both arms read byte-identical prompts.  ~4 english chars = 1 token.
FILLER = (
    "The harbour clock keeps its own time. Cranes lift containers in slow arcs while "
    "the tide works through its list of small errands: a rope tightened, a buoy turned, "
    "a gull persuaded to move. Nobody writes any of this down, and that is the point of "
    "a harbour. "
)


def prompt_of(approx_tokens: int, nonce: str) -> str:
    head = f"[probe {nonce}] Read the following notes, then answer in one short sentence.\n"
    need = max(0, approx_tokens * 4 - len(head) - 200)
    body = (FILLER * (need // len(FILLER) + 1))[:need]
    tail = "\n\nQuestion: which object in the notes was persuaded to move? Answer in one sentence."
    return head + body + tail


def chat(prompt, max_tokens=512, temperature=0.0, image_b64=None, timeout=1800):
    """One streaming request.  Returns (first_token_s, decode_tok_s, text, n_deltas)."""
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
        ]
    else:
        content = prompt
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    n = 0
    pieces = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    piece = d.get(key) or ""
                    if not piece:
                        continue
                    if first is None:
                        first = time.time()
                    n += 1
                    if key == "content":
                        pieces.append(piece)
    end = time.time()
    if first is None or n < 2:
        return None, None, "".join(pieces), n
    return first - t0, (n - 1) / max(end - first, 1e-9), "".join(pieces), n


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, unit="", nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def warmup(arm, runs=2):
    """Two short requests so the expert cache and draft head are warm before timing."""
    print(f"[{arm}] warmup {runs}x ...", flush=True)
    for i in range(runs):
        ttft, tps, _, n = chat(prompt_of(48, f"warm-{arm}-{i}"), max_tokens=192)
        print(f"   warm {i + 1}: ttft={fmt(ttft, 's', 2)} decode={fmt(tps, ' t/s')} deltas={n}", flush=True)


def run_decode(arm, runs=6):
    print(f"[{arm}] decode: {runs} runs ...", flush=True)
    out = []
    for i in range(runs):
        ttft, tps, text, n = chat(prompt_of(48, f"dec-{arm}-{i}"), max_tokens=512)
        out.append({"ttft": ttft, "tps": tps, "deltas": n})
        print(f"   run {i + 1}: ttft={fmt(ttft, 's', 3)} decode={fmt(tps, ' t/s')} deltas={n}",
              flush=True)
    return {"runs": out, "median_tps": median([r["tps"] for r in out]),
            "median_ttft": median([r["ttft"] for r in out])}


def run_context(arm, sizes):
    res = {}
    for size in sizes:
        print(f"[{arm}] context {size}: reading ...", flush=True)
        ttft, tps, text, n = chat(prompt_of(size, f"ctx-{arm}-{size}"), max_tokens=256)
        print(f"   ttft={fmt(ttft, 's', 2)} decode={fmt(tps, ' t/s')} deltas={n}", flush=True)
        res[str(size)] = {"ttft": ttft, "tps": tps, "deltas": n}
    return res


def run_batch(arm, width=4, max_tokens=512):
    print(f"[{arm}] {width}-way concurrent ...", flush=True)
    results = [None] * width
    t0 = time.time()

    def one(i):
        results[i] = chat(prompt_of(512, f"batch-{arm}-{i}"), max_tokens=max_tokens)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(width)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    tps = [r[1] for r in results if r]
    print(f"   wall={wall:.2f}s  per-stream={[round(x, 1) if x else None for x in tps]}", flush=True)
    return {"wall_s": wall, "width": width, "per_stream_tps": tps,
            "median_stream_tps": median(tps), "total_tps": sum(x for x in tps if x)}


def run_vision(arm):
    img = os.path.join(HERE, "vision-test.png")
    if not os.path.isfile(img):
        print(f"[{arm}] vision: {img} missing, skipped", flush=True)
        return None
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    print(f"[{arm}] vision: three colour bars (red / green / blue, top to bottom) ...", flush=True)
    ttft, tps, text, n = chat(
        "Name the coloured bars in this image in order, top to bottom, as three comma-separated words.",
        max_tokens=128, image_b64=b64)
    low = text.lower()
    hits = [c for c in ("red", "green", "blue") if c in low]
    print(f"   answer: {text.strip()[:200]!r}", flush=True)
    print(f"   bars: {len(hits)}/3  ttft={fmt(ttft, 's', 2)}", flush=True)
    return {"ttft": ttft, "answer": text.strip(), "bars": len(hits), "hit_names": hits}


def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--contexts", default="1024,32768,102400")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--model", default=MODEL, help="the served model id (A/B arms differ)")
    ap.add_argument("--skip-batch", action="store_true")
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"
    MODEL = a.model
    sizes = [int(x) for x in a.contexts.split(",") if x.strip()]

    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        models = json.load(r)
    ids = [m["id"] for m in models.get("data", [])]
    if MODEL not in ids:
        print(f"model not served: {ids}", file=sys.stderr)
        return 2
    print(f"[{a.arm}] serving {ids[0]} on :{a.port}", flush=True)

    warmup(a.arm)
    out = {"arm": a.arm, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "decode": run_decode(a.arm),
           "context": run_context(a.arm, sizes)}
    if not a.skip_batch:
        out["batch"] = run_batch(a.arm)
    if not a.skip_vision:
        out["vision"] = run_vision(a.arm)
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    json.dump(out, open(os.path.join(HERE, f"ab_{a.arm}.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "ab_results.jsonl"), "a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    d, c, b = out["decode"], out["context"], out.get("batch")
    print("\n=== SUMMARY arm", a.arm, "===", flush=True)
    print(f"decode median {fmt(d['median_tps'], ' t/s')}  ttft {fmt(d['median_ttft'], 's', 3)}")
    for k, v in c.items():
        print(f"ctx {k:>7}: ttft {fmt(v['ttft'], 's', 2):>7}  decode {fmt(v['tps'], ' t/s'):>8}")
    if b:
        print(f"batch {b['width']}-way: wall {b['wall_s']:.2f}s  sum {fmt(b['total_tps'], ' t/s')}")
    if out.get("vision"):
        print(f"vision: {out['vision']['bars']}/3 bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
