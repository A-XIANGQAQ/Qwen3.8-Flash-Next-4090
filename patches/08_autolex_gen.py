#!/usr/bin/env python3
"""Generate a paraphrase corpus for lexicon mining.

For each auto mode we ask the local model for user messages that would *belong* to that mode, so the label is
given by construction (no human labelling).  The request carries `{REASON:ioff}` so neither a mode body nor
thinking contaminates the generator: with a mode body injected the model stops following the "one question per
line" format (measured: it emitted "# DevOracle v1 ..." and "⎔⎔⎔ teacher ⎔").

Usage:
  python3 08_autolex_gen.py --out corpus.jsonl [--per-mode 40] [--batch 20] [--dry-run]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'Qwen3.8-Flash-Next'

# mode -> one-line description of the *user's goal* (never the mode's name or persona)
MODES = {
    'coder': '写代码、调试报错、部署上线、技术选型、性能优化',
    'matlock': '劳动合同、竞业限制、合同条款、打官司、侵权维权',
    'house': '身体不舒服、看化验单、问病因、用药与就医建议',
    'business': '创业开店、定价、营销获客、营收利润、融资、竞品',
    'spoon': '写调研报告、文献综述、行业白皮书、系统性梳理某个领域',
    'story': '写小说、短篇、剧本、人物设定与对白',
    'prompt': '设计或修改提示词、system prompt、让模型稳定输出某种格式',
    'comedy': '要段子、要笑话、要吐槽、脱口秀素材',
    'brainstorm': '头脑风暴、起名字、想点子、找灵感',
    'logic': '证明、反例、逻辑漏洞、推理是否成立、第一性原理',
    'teacher': '解释概念、讲原理、通俗科普、入门教学',
}

SYSTEM = ('你在帮人整理"用户会怎么问 AI 助手"的语料。只输出问题本身，不要解释、不要编号、不要引号。')


def ask(desc: str, n: int, temperature: float = 0.9) -> list[str]:
    prompt = (f'请写 {n} 条中文提问，都是用户发给 AI 助手的，内容属于这一类：{desc}。\n'
              '要求：每行一条；每条 8–40 字；口语和书面混着来；有的短、有的长；'
              '不要提到"模式""风格""角色扮演""助手设定"；不要出现大括号标签；不要编号、不要引号。')
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': '{REASON:ioff} ' + prompt}],
            'max_tokens': 1200, 'temperature': temperature,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    text = data['choices'][0]['message'].get('content') or ''
    out = []
    for line in text.split('\n'):
        line = line.strip().strip('-*').strip()
        for sep in ('、', '. ', '．'):
            if line[:2].isdigit() and sep in line[:4]:
                line = line.split(sep, 1)[-1].strip()
        line = line.strip('“”"\'，,。 ')
        if 4 <= len(line) <= 60 and '{REASON' not in line and '模式' not in line:
            out.append(line)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='corpus.jsonl')
    ap.add_argument('--per-mode', type=int, default=40)
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--mode', action='append', help='only these modes (repeatable)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    modes = args.mode or list(MODES)
    rows, seen = [], set()
    for mode in modes:
        got = 0
        rounds = 0
        while got < args.per_mode and rounds < 6:
            rounds += 1
            try:
                items = ask(MODES[mode], min(args.batch, args.per_mode - got))
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                print(f'  {mode}: request failed ({exc})', file=sys.stderr)
                continue
            new = 0
            for text in items:
                key = (mode, text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'mode': mode, 'text': text})
                new += 1
            got += new
            print(f'  {mode}: round {rounds} -> {new} new, {got}/{args.per_mode}', flush=True)
            if new == 0:
                break
    if args.dry_run:
        for row in rows[:10]:
            print(row)
        return 0
    with open(args.out, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'wrote {len(rows)} rows to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
