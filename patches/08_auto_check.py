#!/usr/bin/env python3
"""Evaluate the auto-mode rule against the REAL template (not a re-implementation).

The chosen mode is detected from the rendered prompt: the mode's body is emitted at the head of the system
message, so matching a body prefix identifies it; no system message at all means "no mode".

Sets:
  dev  - cases used while designing the rule/lexicon      (73)
  val  - fresh cases, checked once per optimization round (30)
  test - locked; run ONCE at the end                      (40)

Usage:
  python3 08_auto_check.py                      # dev
  python3 08_auto_check.py --set all            # dev + val
  python3 08_auto_check.py --set test           # only when the plan says so
  python3 08_auto_check.py --case "讲个笑话"     # one ad-hoc question, with scores
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / '08_qwen38_froggeric_v225_with_dau_modes.jinja'

DEV = [
    # ---- design set (41): every rule change has a case here ----
    ("I trust you, this pipeline is rapid — tell me the plan", ''),
    ("帮我把这段代码里的 bug 修一下", 'coder'),
    ("用代码证明这个数学命题，再写个 python 脚本", 'coder'),
    ("合同里竞业条款的有效性", 'matlock'),
    ("这个合同会不会违约，帮我看看诉讼风险", 'matlock'),
    ("竞业限制的判例有哪些，顺便这段代码合规吗", 'matlock'),
    ("最近头晕发烧，帮我诊断一下可能病因", 'house'),
    ("体检报告指标异常，需要用药建议", 'house'),
    ("我想创业，做个社区团购的商业模式", 'business'),
    ("帮我做个市场调研，看看竞品定价", 'business'),
    ("研究现状写个综述，要文献支撑", 'spoon'),
    ("帮我写个故事，主角是个程序员", 'story'),
    ("写一篇小说的开头，要有世界观设定", 'story'),
    ("帮我写个系统提示词，角色设定是律师", 'prompt'),
    ("来段脱口秀，讲程序员的梗", 'comedy'),
    ("头脑风暴一下，给这个产品起名", 'brainstorm'),
    ("帮我创意几个开店的点子", 'brainstorm'),
    ("用反例证明这个命题不成立", 'logic'),
    ("第一性原理推导一下这个方案是否成立", 'logic'),
    ("解释一下什么是傅里叶变换，通俗点", 'teacher'),
    ("讲清楚 attention 的原理，举例说明", 'teacher'),
    ("你好", ''),
    ("今天天气不错", ''),
    ("继续", ''),
    ("帮我翻译这段话成英文", ''),
    ("这段 python 代码报错 traceback", 'coder'),
    ("pip install 失败，依赖冲突", 'coder'),
    ("rust 的借用检查怎么过", 'coder'),
    ("docker 部署 nginx 反代", 'coder'),
    ("劳动法规定试用期多久", 'matlock'),
    ("症状是持续咳嗽，化验单怎么看", 'house'),
    ("营收利润下滑，怎么获客", 'business'),
    ("白皮书里的证据链有问题", 'spoon'),
    ("剧本对白怎么写才自然", 'story'),
    ("prompt 怎么写才能稳定输出 json", 'prompt'),
    ("讲个笑话", 'comedy'),
    ("逻辑漏洞在哪，自相矛盾吗", 'logic'),
    ("教我入门 cuda 编程", 'coder'),
    ("这个 bug 的法条依据是什么", ''),
    ("diagnose this segfault", 'coder'),
    ("trust the process, story time please", 'story'),  # out of scope (English sentence) - kept as a known miss
    # ---- held-out set (32): written after the rule was fixed, never used to tune it ----
    ("这段 TypeScript 泛型报错 TS2344 怎么解决", 'coder'),
    ("帮我把这个 sql 慢查询优化一下", 'coder'),
    ("公司要给我签竞业协议，补偿金标准是多少", 'matlock'),
    ("劳动仲裁申请时效是多久", 'matlock'),
    ("体检发现甲状腺结节三类，要紧吗", 'house'),
    ("最近总是失眠，需要吃药吗", 'house'),
    ("开一家奶茶店大概多久回本", 'business'),
    ("怎么给一个 SaaS 产品定价", 'business'),
    ("帮我梳理大模型推理优化的研究脉络", 'spoon'),
    ("写一份行业白皮书的提纲", 'spoon'),
    ("写个赛博朋克短篇的开头", 'story'),
    ("给这个角色写一段独白", 'story'),
    ("怎么设计 few-shot 提示词让输出稳定", 'prompt'),
    ("帮我改一下这段 system prompt", 'prompt'),
    ("来个冷笑话", 'comedy'),
    ("吐槽一下周一上班", 'comedy'),
    ("给新项目起十个名字", 'brainstorm'),
    ("头脑风暴几个降本的点子", 'brainstorm'),
    ("这个论证是不是犯了循环论证", 'logic'),
    ("帮我找这个定理证明的反例", 'logic'),
    ("什么是傅里叶变换，讲通俗点", 'teacher'),
    ("给我讲讲 transformer 为什么能并行", 'teacher'),
    ("今天几号", ''),
    ("帮我订个会议室", ''),
    ("推荐一部电影", ''),
    ("把这句话翻成日语", ''),
    ("这个 API 文档写得真烂", ''),
    ("周末去哪儿玩比较好", ''),
    ("帮我总结一下这份会议纪要", ''),
    ("这周的工作计划怎么排", ''),
    ("把这段录音转成文字", ''),
    ("帮我写个请假邮件", ''),
]

VAL = [
    ("帮我看看这个 java 异常的堆栈", 'coder'),
    ("这个 sql 索引怎么建才快", 'coder'),
    ("公司的竞业协议补偿标准是多少", 'matlock'),
    ("劳动仲裁要准备什么材料", 'matlock'),
    ("体检查出脂肪肝，需要治疗吗", 'house'),
    ("最近睡眠不好，吃什么药", 'house'),
    ("开连锁咖啡店怎么选址", 'business'),
    ("订阅制产品的定价策略", 'business'),
    ("帮我做一份行业研究报告的框架", 'spoon'),
    ("这个领域的文献综述怎么写", 'spoon'),
    ("写个短篇科幻小说的设定", 'story'),
    ("给这个人物写一段内心独白", 'story'),
    ("怎么让模型稳定输出 json 格式", 'prompt'),
    ("帮我优化这段 system prompt", 'prompt'),
    ("来个冷笑话逗逗我", 'comedy'),
    ("吐槽一下这周的加班", 'comedy'),
    ("给产品起几个名字", 'brainstorm'),
    ("头脑风暴几个节省成本的方案", 'brainstorm'),
    ("这个推理哪里不成立", 'logic'),
    ("帮我构造一个反例", 'logic'),
    ("通俗讲讲什么是区块链", 'teacher'),
    ("解释一下梯度下降的原理", 'teacher'),
    ("明天要开会，帮我准备下议程", ''),
    ("帮我查一下附近的餐厅", ''),
    ("把这段文字翻译成法语", ''),
    ("推荐几本入门的书", ''),
    ("今天心情不太好", ''),
    ("帮我设置个提醒", ''),
    ("这段代码为什么运行不起来", 'coder'),
    ("写一首关于秋天的诗", ''),
]

TEST = [
    ("我的 React 组件渲染死循环了", 'coder'),
    ("这个 nginx 配置 502 怎么排查", 'coder'),
    ("git rebase 冲突怎么解决", 'coder'),
    ("docker compose 起不来，日志报 port already in use", 'coder'),
    ("对方违约了，我能要求赔偿吗", 'matlock'),
    ("签合同要注意哪些条款", 'matlock'),
    ("被侵权了怎么起诉", 'matlock'),
    ("竞业限制的补偿金可以不发吗", 'matlock'),
    ("孩子发烧 38.5 度需要去医院吗", 'house'),
    ("体检报告上血脂偏高怎么办", 'house'),
    ("长期头痛是什么原因", 'house'),
    ("这个症状需要挂号哪个科", 'house'),
    ("小团队怎么找到第一批客户", 'business'),
    ("电商店铺怎么做营销", 'business'),
    ("融资计划书怎么写", 'business'),
    ("帮我写个综述的引言", 'spoon'),
    ("写个悬疑小说的开头", 'story'),
    ("帮我写个剧本分镜", 'story'),
    ("怎么写出好的提示词", 'prompt'),
    ("提示词里加角色设定有用吗", 'prompt'),
    ("讲个相声段子", 'comedy'),
    ("有没有搞笑的笑话", 'comedy'),
    ("帮我想几个创业点子", 'brainstorm'),
    ("给活动起个名字", 'brainstorm'),
    ("这个论证有逻辑漏洞吗", 'logic'),
    ("用数学归纳法证明这个命题", 'logic'),
    ("什么是梯度下降，讲通俗点", 'teacher'),
    ("教我怎么理解傅里叶变换", 'teacher'),
    ("帮我看看明天的日程", ''),
    ("把这段话润色一下", ''),
    ("附近有什么好吃的", ''),
    ("帮我订一张去北京的机票", ''),
    ("帮我写个周报", ''),
    ("这段代码有没有性能问题", 'coder'),
    ("合同里的违约金条款合理吗", 'matlock'),
    ("我最近总是失眠，怎么办", 'house'),
    ("开店的成本怎么算", 'business'),
    ("写个童话故事给孩子", 'story'),
    ("周末想出去玩，有什么建议", ''),
    ("帮我记一下这个想法", ''),
]

SETS = {'dev': DEV, 'val': VAL, 'test': TEST}


def make_env():
    import json as _json

    def raise_exception(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return _json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    return env


def mode_bodies(source: str):
    """mode -> first 80 chars of its body, parsed from the template's own instruction_map."""
    variables = {}
    for name, var in re.findall(r"'([A-Za-z0-9_]+)':\s*(_mode_[A-Za-z0-9_]+|system_instruction_[A-Za-z0-9_]+),",
                                source):
        variables[name] = var
    body_map = {}
    for var, body in re.findall(r'\{%-\s*set\s+(system_instruction_[A-Za-z0-9_]+)\s*-\%\}(.*?)\{%-\s*endset\s*-\%\}',
                                source, re.S):
        body_map[var] = body.strip()
    short = dict(re.findall(r"\{%-\s*set\s+(_mode_[A-Za-z0-9_]+)\s*=\s*'([^']*)'\s*%\}", source))
    out = {}
    # The 8 xhigh combinations come first: their body starts with the xhigh sentence, so their prefix has to be
    # longer than the plain `xhigh` body's (which is a prefix of theirs) or detect() would name the wrong mode.
    xmap = re.findall(r"'(x[a-z0-9_]+)':\s*_mode_xhigh \+ _xsep \+ (system_instruction_[A-Za-z0-9_]+),", source)
    msep = re.search(r"\{%-\s*set\s+_xsep\s*=\s*('(?:[^'\\]|\\.)*')\s*%\}", source)
    xhigh = short.get('_mode_xhigh', '')
    if xmap and msep and xhigh:
        sep = ast.literal_eval(msep.group(1))
        for name, var in xmap:
            text = body_map.get(var) or ''
            if len(text) >= 80:
                out[name] = xhigh + sep + text[:80]
    for name, var in variables.items():
        text = body_map.get(var) or short.get(var) or ''
        if len(text) >= 80:
            out[name] = text[:80]
    return out


def detect(rendered: str, bodies: dict) -> str:
    for mode, prefix in bodies.items():
        if prefix and prefix in rendered:
            return mode
    return ''


def mirror(text: str, lex: dict):
    """Python mirror of the template rule, only for --explain / drift checking."""
    import re as _re

    order, punct, floor, margin = lex['order'], lex['punct'], lex['floor'], lex['margin']
    low = text.lower()[-4000:]
    padded = low
    for ch in punct:
        padded = padded.replace(ch, ' ')
    padded = ' ' + padded + ' '
    scores = {}
    for mode in order:
        total = 0
        for weight, words in lex['lexicon'][mode].items():
            for word in words:
                hit = (' ' + word + ' ') in padded if _re.fullmatch(r'[a-z0-9]+', word) else word in low
                total += int(weight) * int(hit)
        scores[mode] = total
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_mode, top = ranked[0]
    second = ranked[1][1]
    chosen = top_mode if (top >= floor and top - second >= margin) else ''
    return chosen, ranked


def evaluate(template: Path, sets, explain=False):
    source = template.read_text(encoding='utf-8')
    lex_path = template.with_suffix('').with_suffix('.lexicon.json')
    lex = json.loads(lex_path.read_text(encoding='utf-8')) if lex_path.exists() else None
    env = make_env()
    tpl = env.from_string(source)
    bodies = mode_bodies(source)
    rows = []
    drift = 0
    for set_name in sets:
        for text, gold in SETS[set_name]:
            rendered = tpl.render(messages=[{'role': 'user', 'content': text}], tools=None,
                                  add_generation_prompt=True)
            got = detect(rendered, bodies)
            if lex:
                mirrored, _ = mirror(text, lex)
                if mirrored != got:
                    drift += 1
                    print(f'  [DRIFT] template={got!r} mirror={mirrored!r} :: {text[:40]}')
            rows.append((set_name, text, gold, got))
    print(f'{"set":6s}{"n":>4s}{"ok":>6s}{"wrong":>7s}{"abstain":>9s}{"acc":>8s}')
    for set_name in sets:
        rs = [r for r in rows if r[0] == set_name]
        ok = sum(1 for _, _, gold, got in rs if got == gold)
        wrong = sum(1 for _, _, gold, got in rs if got not in ('', gold))
        abst = sum(1 for _, _, gold, got in rs if got == '' and gold != '')
        print(f'{set_name:6s}{len(rs):>4d}{ok:>6d}{wrong:>7d}{abst:>9d}{ok / len(rs) * 100:>7.1f}%')
    ok = sum(1 for _, _, gold, got in rows if got == gold)
    print(f'{"ALL":6s}{len(rows):>4d}{ok:>6d}'
          f'{sum(1 for _, _, g, t in rows if t not in ("", g)):>7d}'
          f'{sum(1 for _, _, g, t in rows if t == "" and g):>9d}{ok / len(rows) * 100:>7.1f}%')
    if drift:
        print(f'\n!! mirror drifts from the template on {drift} case(s) - fix the mirror before trusting --explain')
    if explain:
        print('\nfailures:')
        for set_name, text, gold, got in rows:
            if got != gold:
                print(f'  [{set_name}] want {gold or "-":9s} got {got or "-":9s} | {text}')
                if lex:
                    _, ranked = mirror(text, lex)
                    print(f'      scores: {[(m, v) for m, v in ranked if v]}')
    return 0 if drift == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', default=str(DEFAULT_TEMPLATE))
    ap.add_argument('--set', default='dev', choices=['dev', 'val', 'test', 'dev+val', 'all'])
    ap.add_argument('--explain', action='store_true')
    ap.add_argument('--case')
    args = ap.parse_args()
    template = Path(args.template)
    if args.case:
        source = template.read_text(encoding='utf-8')
        env = make_env()
        rendered = env.from_string(source).render(messages=[{'role': 'user', 'content': args.case}],
                                                  tools=None, add_generation_prompt=True)
        bodies = mode_bodies(source)
        print('mode :', detect(rendered, bodies) or '(none)')
        lex_path = template.with_suffix('').with_suffix('.lexicon.json')
        if lex_path.exists():
            lex = json.loads(lex_path.read_text(encoding='utf-8'))
            _, ranked = mirror(args.case, lex)
            print('scores:', [(m, v) for m, v in ranked if v])
        return 0
    sets = {'dev': ['dev'], 'val': ['val'], 'test': ['test'],
            'dev+val': ['dev', 'val'], 'all': ['dev', 'val']}[args.set]
    return evaluate(template, sets, explain=args.explain)


if __name__ == '__main__':
    sys.exit(main())
