#!/usr/bin/env python3
"""Mine lexicon candidates from the paraphrase corpus (corpus.jsonl).

No segmenter is available, so candidates are character n-grams (2-4 chars) over CJK clauses plus latin tokens.
A candidate is scored by lift = P(seen | this mode) / P(seen | every other mode), which is what makes it
discriminative rather than merely frequent.

Usage:
  python3 08_autolex_mine.py --corpus corpus.jsonl [--top 25] [--json candidates.json]
  python3 08_autolex_mine.py --corpus corpus.jsonl --mode house --top 40
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

STOP = set("""帮我 我想 我想 我要 你能 能不 可以 怎么 这个 那个 什么 一下 我的 我们 你们 请问 如何 需要
没有 有没 现在 就是 的话 这样 那样 应该 是不是 为什么 然后 还是 或者 因为 所以 但是 其实 感觉 觉得
知道 想要 想要 能帮 怎么样 在哪里 有哪些 一点 多少 最近 总是 已经 还有 不会 不能 不是 这么 那么 时候
如果 只要 虽然 不过 而且 并且 以及 关于 对于 由于 通过 根据 一个 一些 这种 那种 各种 每个 所有
好像 可能 应该 必须 一定 真的 确实 直接 自己 别人 东西 问题 办法 方法 方面 情况 内容 部分 地方
""".split())


def load_lexicon():
    spec = importlib.util.spec_from_file_location('merge_tpl', HERE / '08_merge_template.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODE_LEX, mod.AUTO_ORDER


def grams(text: str) -> set:
    out = set()
    for clause in re.split(r'[^\u4e00-\u9fffA-Za-z0-9+#]+', text):
        for tok in re.findall(r'[A-Za-z][A-Za-z0-9+#.]{1,}', clause):
            out.add(tok.lower())
        cjk = re.sub(r'[^\u4e00-\u9fff]', '', clause)
        for n in (2, 3, 4):
            for i in range(len(cjk) - n + 1):
                g = cjk[i:i + n]
                if g in STOP or any(s in g for s in STOP):
                    continue
                out.add(g)
    return out


EDGE_STOP = set('的了是我你他她它这那就也都很太不没在有和与对被让给到去来说个些把从向往过得地着吗呢吧啊么')
EDGE_STOP |= set('吧呢吗啊呀哦嗯')


def boundary_entropy(texts: list[str], gram: str) -> tuple[float, float, int, int]:
    """Left/right boundary entropy: a real word shows up in many contexts, a fragment does not."""
    import math
    left, right = Counter(), Counter()
    for text in texts:
        start = 0
        while True:
            i = text.find(gram, start)
            if i < 0:
                break
            left[text[i - 1] if i > 0 else '^'] += 1
            j = i + len(gram)
            right[text[j] if j < len(text) else '$'] += 1
            start = i + 1

    def ent(c):
        total = sum(c.values()) or 1
        return -sum((v / total) * math.log(v / total) for v in c.values())

    return ent(left), ent(right), len(left), len(right)


def covered(word: str, existing: list[str]) -> str:
    for old in existing:
        if word in old or old in word:
            return old
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(HERE / 'corpus.jsonl'))
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--mode')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    lex, order = load_lexicon()
    docs = defaultdict(list)
    for line in open(args.corpus, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        docs[row['mode']].append(row['text'])
    if not docs:
        sys.exit('empty corpus')

    n_mode = {m: len(v) for m, v in docs.items()}
    n_other = {m: sum(n_mode[o] for o in n_mode if o != m) for m in n_mode}
    df = {m: Counter() for m in n_mode}
    for mode, texts in docs.items():
        for text in texts:
            for g in grams(text):
                df[mode][g] += 1

    picked = [args.mode] if args.mode else order
    result = {}
    for mode in picked:
        existing = [w for tier in lex[mode].values() for w in tier]
        rows = []
        for g, c in df[mode].items():
            if c < 3 or len(g) < 2:
                continue
            other_hits = sum(df[o].get(g, 0) for o in df if o != mode)
            p_this = c / n_mode[mode]
            p_other = (other_hits / n_other[mode]) if n_other[mode] else 0.0
            lift = p_this / max(p_other, 1.0 / max(n_other[mode], 1))
            if lift < 3 or p_other > 0.25:
                continue
            if g[0] in EDGE_STOP or g[-1] in EDGE_STOP:
                continue
            l_ent, r_ent, l_n, r_n = boundary_entropy(docs[mode], g)
            if min(l_ent, r_ent) < 0.6 or min(l_n, r_n) < 2:
                continue
            rows.append((g, c, other_hits, lift))
        # keep maximal phrases: drop an n-gram that is a substring of a longer candidate with the same coverage
        rows.sort(key=lambda r: (-r[3] * (r[1] ** 0.5), -len(r[0])))
        keep = []
        for row in rows:
            if any(row[0] in k[0] and k[1] >= row[1] * 0.8 for k in keep):
                continue
            keep.append(row)
        keep = keep[:args.top]
        result[mode] = [{'cand': g, 'df': c, 'other': o, 'lift': round(l, 1),
                         'old': covered(g, existing)} for g, c, o, l in keep]

    for mode in picked:
        print(f'\n=== {mode}  (docs={n_mode[mode]}) ===')
        for row in result[mode]:
            tag = f'  [已有: {row["old"]}]' if row['old'] else ''
            print(f'  {row["cand"]:12s} df={row["df"]:3d} other={row["other"]:3d} '
                  f'lift={row["lift"]:6.1f}{tag}')
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
