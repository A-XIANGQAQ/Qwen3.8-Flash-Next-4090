#!/usr/bin/env python3
"""Focused long-context decode check: N fresh 100K-token prompts, 512 generated.

The full probe's 100K point generated only ~48 tokens, and token-by-token decode
over 48 tokens is too noisy to separate the two engines.  This one keeps the
generation long enough for a stable rate, and runs nothing else, so the engine
state is the same for both arms.

  python scripts/long_ctx_probe.py --arm B --tokens 102400 --reps 3
"""
import argparse
import json
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ab_probe import chat, prompt_of, median, fmt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tokens", type=int, default=102400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=512)
    a = ap.parse_args()

    chat(prompt_of(48, f"warm-{a.arm}"), max_tokens=128)      # one short request: drafts and cache warm
    out = []
    for i in range(a.reps):
        # a long answer averages out draft-acceptance variance; the short answer this
        # prompt used to get (46 tokens) made decode +/-25% run to run
        prompt = prompt_of(a.tokens, f"long-{a.arm}-{i}").replace(
            "Answer in one sentence.", "Answer in about 400 words of plain prose.")
        ttft, tps, text, n = chat(prompt, max_tokens=a.gen)
        out.append({"ttft": ttft, "tps": tps, "deltas": n})
        print(f"   rep {i + 1}: ttft={fmt(ttft, 's', 2)} decode={fmt(tps, ' t/s')} deltas={n}", flush=True)
    res = {"arm": a.arm, "tokens": a.tokens,
           "median_ttft": median([r["ttft"] for r in out]),
           "median_tps": median([r["tps"] for r in out]),
           "reps": out}
    print(f"[{a.arm}] {a.tokens} tokens: ttft {fmt(res['median_ttft'], 's', 2)}  "
          f"decode {fmt(res['median_tps'], ' t/s')}", flush=True)
    with open(__import__("os").path.join(__import__("os").path.dirname(
            __import__("os").path.abspath(__file__)), "long_ctx.jsonl"), "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
