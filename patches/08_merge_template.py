#!/usr/bin/env python3
"""Merge the DavidAU mode system (41 {REASON:} modes) into froggeric's v22.5 Qwen3.8 template.

Template-only integration: no server patch.  The result keeps froggeric's logic (effort aliases, think markers,
tool calling, truncation, warnings) and adds, before the effort sentence is built:

  * the 29 big `system_instruction_*` texts from the model's own template, verbatim,
  * the `{REASON:<mode>}` tag parser (last tag wins, `i` prefix = no thinking),
  * automatic mode choice from the conversation's user messages (so nobody has to type tags):
    a linear score over the lexicon below, decided by `top1 >= 2 and top1 > top2` (margin),
    with ASCII single tokens matched on word boundaries,
  * `instruction_map`, the same 41 mode keys DavidAU's template has (33 + the 8 "xhigh" combinations).

Usage: python3 merge_template.py <froggeric.jinja> <davidau.jinja> <out.jinja>
Writes <out.jinja> plus <out>.lexicon.json (the lexicon + thresholds, for 08_auto_check.py).
"""
import json
import re
import sys

MODES = {
    # mode: davidau block variable (None = short inline text handled below)
    'omni': 'system_instruction_DAU', 'deeptree': 'system_instruction_DAU2',
    'hyper': 'system_instruction_DAU3', 'socrates': 'system_instruction_DAU4',
    'logic': 'system_instruction_DAU5', 'help': 'system_instruction_DAU_help',
    'einstein': 'system_instruction_DAU_EIN', 'spoon': 'system_instruction_DAU_SPOON',
    'ultra': 'system_instruction_xxhigh_DAU',
    'xhigh': None, 'medium': None, 'low': None, 'off': None,
    'holodeck': 'system_instruction_holodeck_DAU', 'prompt': 'system_instruction_prompt_DAU',
    'pkd': 'system_instruction_pkd_DAU', 'musk': 'system_instruction_musk_DAU',
    'joker': 'system_instruction_joker_DAU', 'vader': 'system_instruction_vader_DAU',
    'quinn': 'system_instruction_quinn_DAU', 'joe': 'system_instruction_joe_DAU',
    'neo': 'system_instruction_neo_DAU', 'curie': 'system_instruction_curie_DAU',
    'brainstorm': 'system_instruction_brainstorm_DAU', 'comedy': 'system_instruction_comedy_DAU',
    'business': 'system_instruction_business_DAU', 'story': 'system_instruction_story_DAU',
    'teacher': 'system_instruction_teacher_DAU', '007': 'system_instruction_007_DAU',
    'coder': 'system_instruction_coder_DAU', 'stark': 'system_instruction_stark_DAU',
    'matlock': 'system_instruction_matlock_DAU', 'house': 'system_instruction_house_DAU',
}

# ------------------------------------------------------------------------------------------------------------
# Auto-mode lexicon.  weight 3 = unambiguous, 2 = clear topic word, 1 = weak hint.  A mode needs total >= 2 and a
# strict lead over the runner-up (`margin`), otherwise no mode is chosen and froggeric's default applies.
# Matching: a word made only of [a-z0-9] is matched on word boundaries after punctuation is flattened to spaces;
# anything else (CJK, '```') is matched as a plain substring.
# Order matters only as a deterministic tie-break; ties never select (the margin rule rejects them).
AUTO_ORDER = ['coder', 'matlock', 'house', 'business', 'spoon', 'story',
              'prompt', 'comedy', 'brainstorm', 'logic', 'teacher']

MODE_LEX = {
    'coder': {
        2: ['```', 'traceback', 'segfault', '代码', '报错', '部署',
            'python', 'javascript', 'typescript', 'golang', 'rust', 'java', 'sql', 'docker', 'cargo',
            'cuda', 'pytorch', 'nginx', '日志', '内存', '前端'],
        1: ['函数', '编译', '重构', '单元测试', 'bug', 'api', '接口', '脚本', '算法', '数据结构', '依赖',
            'npm', 'pip', 'regex', '异常', '变量', '数组', '字符串', '编程', '程序', '配置', '端口', '重启'],
    },
    'matlock': {
        2: ['诉讼', '判例', '竞业', '仲裁', '离职', '赔偿', '辞退', '起诉', '法院', '索赔'],
        1: ['法律', '法规', '法条', '合同', '条款', '合规', '侵权', '违约', '劳动法', '律师',
            '工资', '补偿', '时效'],
    },
    'house': {
        2: ['诊断', '病历', '体检', '失眠', '结节', '睡眠'],
        1: ['症状', '病因', '化验', '检查报告', '指标异常', '用药', '临床', '头晕', '疼痛', '发烧',
            '感冒', '咳嗽', '头疼', '缓解'],
    },
    'business': {
        2: ['商业模式', '创业', '定价', '回本'],
        1: ['商业', '市场', '营销', '营收', '利润', '获客', '融资', 'bp', '竞品', '供应链', '加盟',
            '开店', '变现', '生意', '成本', '降价', '咖啡店', '奶茶店'],
    },
    'spoon': {
        2: ['研究现状', '证据链', '白皮书', '梳理', '脉络'],
        1: ['调研', '综述', '文献', '深度报告', '系统性', '横向对比', '行业', '研究'],
    },
    'story': {
        2: ['故事', '小说', '剧本', '短篇', '独白', '剧情', '反派', '女主'],
        1: ['散文', '叙事', '人物设定', '对白', '世界观', '情节', '写一篇', '写个',
            '主角', '台词', '角色', '描写', '大纲', '内心'],
    },
    'prompt': {
        2: ['提示词', 'prompt', '输出格式', '约束'],
        1: ['系统提示', '指令模板', '角色设定', '稳定', '固定', '输出', '指令'],
    },
    'comedy': {
        2: ['段子', '脱口秀', '笑话', '吐槽', '幽默', '谐音梗'],
        1: ['搞笑', '梗', '相声'],
    },
    'brainstorm': {
        2: ['头脑风暴', '点子', '构思'],
        1: ['创意', '想法', '灵感', '命名', '起名', '主打', '活动', '新开'],
    },
    'logic': {
        2: ['证明', '反例', '第一性原理', '形式化', '论证', '悖论', '谬误'],
        1: ['推导', '是否成立', '逻辑', '漏洞', '自相矛盾', '推理', '前提', '反驳', '因果'],
    },
    'teacher': {
        2: ['科普'],
        1: ['解释', '讲清楚', '通俗', '入门', '教我', '什么是', '原理', '怎么理解', '举例说明',
            '讲讲', '说说', '讲解', '定义', '区别'],
    },
}

FLOOR = 2      # a mode needs at least this many points
MARGIN = 1     # ... and must lead the runner-up by at least this much

# The 8 "xhigh" combinations the model card and the in-model help menu advertise:
#   {REASON:x<mode>} = Qwen xhigh directive + <mode> body, exactly as DavidAU's own instruction_map builds them.
# They are explicit-only: they are NOT in AUTO_ORDER / MODE_LEX, so auto-selection can never pick one.
XCOMBOS = {
    'xomni': 'system_instruction_DAU', 'xdeeptree': 'system_instruction_DAU2',
    'xhyper': 'system_instruction_DAU3', 'xsocrates': 'system_instruction_DAU4',
    'xlogic': 'system_instruction_DAU5', 'xhelp': 'system_instruction_DAU_help',
    'xeinstein': 'system_instruction_DAU_EIN', 'xspoon': 'system_instruction_DAU_SPOON',
}
XSEP = ('\n\nIn addition to following "xhigh" directives, use the following directives, instructions and '
        'guidelines to process the user prompt and output the final answer:\n')

SHORT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
             'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
    'medium': '',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion '
           'without unnecessary elaboration.',
    'off': '',
}

# Full-width punctuation has to be flattened too, or `，pipeline，` never matches the ASCII word boundary.
PUNCT = list(".,;:!?/\\|()[]{}<>\"'#@$%^&+=~-_*，。？！：；（）【】「」、·…“”‘’《》")


def is_ascii_word(word: str) -> bool:
    return re.fullmatch(r'[a-z0-9]+', word) is not None


def blocks_from(dau: str):
    found = {}
    for m in re.finditer(r'\{%-\s*set\s+(system_instruction_[A-Za-z0-9_]+)\s*-\%\}(.*?)\{%-\s*endset\s*-\%\}',
                         dau, re.S):
        found[m.group(1)] = m.group(2)
    return found


def check_lexicon():
    """Fail the build rather than ship a lexicon that cannot work."""
    seen = set()
    for mode, tiers in MODE_LEX.items():
        if mode not in AUTO_ORDER:
            raise SystemExit(f'lexicon mode not in AUTO_ORDER: {mode}')
        for weight, words in tiers.items():
            if weight not in (1, 2, 3):
                raise SystemExit(f'{mode}: bad weight {weight}')
            for word in words:
                if word != word.lower():
                    raise SystemExit(f'{mode}: word not lowercased: {word!r}')
                if word != word.strip():
                    raise SystemExit(f'{mode}: word has surrounding space: {word!r}')
                if word in seen:
                    raise SystemExit(f'word listed twice: {word!r}')
                seen.add(word)
    for mode in AUTO_ORDER:
        if mode not in MODE_LEX:
            raise SystemExit(f'AUTO_ORDER mode missing from lexicon: {mode}')


def emit_auto_block() -> str:
    """The Jinja that picks a mode when the request carries no {REASON:} tag."""
    out = ['{%- set reasoning_instructions = \'\' %}\n']
    out.append('{#- DavidAU modes: an explicit {REASON:<mode>} tag wins; otherwise score the conversation\'s '
               'user text -#}\n')
    out.append("{%- set _mode = namespace(name='', tag='') %}\n")
    out.append("""{%- for _m in messages %}
    {%- set _ts = namespace(text='') %}
    {%- if _m.content is string %}
        {%- set _ts.text = _m.content %}
    {%- elif _m.content is iterable and _m.content is not mapping %}
        {%- for _it in _m.content %}
            {%- if _it is string %}{%- set _ts.text = _ts.text ~ ' ' ~ _it %}
            {%- elif _it is mapping and 'text' in _it and _it.text is string %}{%- set _ts.text = _ts.text ~ ' ' ~ _it.text %}{%- endif %}
        {%- endfor %}
    {%- endif %}
    {%- if '{REASON:' in _ts.text %}
        {%- set _cand = _ts.text.split('{REASON:')[1].split('}')[0] %}
        {%- set _key = _cand[1:] if _cand.startswith('i') else _cand %}
        {%- if _key in instruction_map %}
            {%- set _mode.name = _key %}
            {%- set _mode.tag = '{REASON:' + _cand + '}' %}
            {%- if _m.role == 'user' %}
                {%- if _cand.startswith('i') %}{%- set ns_state.thinking = false %}{%- else %}{%- set ns_state.thinking = true %}{%- endif %}
            {%- endif %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if not _mode.name %}
    {%- set _u = namespace(text='') %}
    {%- for _m in messages %}
        {%- if _m.role == 'user' %}
            {%- if _m.content is string %}
                {%- set _u.text = _u.text ~ ' ' ~ _m.content %}
            {%- elif _m.content is iterable and _m.content is not mapping %}
                {%- for _it in _m.content %}
                    {%- if _it is string %}{%- set _u.text = _u.text ~ ' ' ~ _it %}
                    {%- elif _it is mapping and 'text' in _it and _it.text is string %}{%- set _u.text = _u.text ~ ' ' ~ _it.text %}{%- endif %}
                {%- endfor %}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- set _t = (_u.text | lower)[-4000:] %}
    {%- set _n = _t""")
    for ch in PUNCT:
        out.append(" | replace(%r, ' ')" % ch)
    out.append(" %}\n    {%- set _n = ' ' ~ _n ~ ' ' %}\n")
    out.append('    {%%- set _h = namespace(%s) %%}\n' % ', '.join(f'{m}=0' for m in AUTO_ORDER))
    for mode in AUTO_ORDER:
        tiers = MODE_LEX[mode]
        for weight in (3, 2, 1):
            words = tiers.get(weight, [])
            ascii_words = [w for w in words if is_ascii_word(w)]
            raw_words = [w for w in words if not is_ascii_word(w)]
            if ascii_words:
                out.append("    {%%- for _w in [%s] %%}{%%- if ' ' ~ _w ~ ' ' in _n %%}"
                           "{%%- set _h.%s = _h.%s + %d %%}{%%- endif %%}{%%- endfor %%}\n"
                           % (', '.join(repr(w) for w in ascii_words), mode, mode, weight))
            if raw_words:
                out.append("    {%%- for _w in [%s] %%}{%%- if _w in _t %%}"
                           "{%%- set _h.%s = _h.%s + %d %%}{%%- endif %%}{%%- endfor %%}\n"
                           % (', '.join(repr(w) for w in raw_words), mode, mode, weight))
    pairs = ', '.join("('%s', _h.%s)" % (m, m) for m in AUTO_ORDER)
    out.append("    {%%- set _p = [%s] | sort(attribute=1, reverse=true) %%}\n" % pairs)
    out.append("    {%%- if _p[0][1] >= %d and _p[0][1] - _p[1][1] >= %d %%}"
               "{%%- set _mode.name = _p[0][0] %%}{%%- endif %%}\n" % (FLOOR, MARGIN))
    out.append('{%- endif %}\n')
    out.append("""{%- if _mode.name %}
    {%- set reasoning_instructions = instruction_map.get(_mode.name, '') %}
{%- elif ns_state.thinking %}
{%- if ns_state.effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif ns_state.effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
""")
    return ''.join(out)


def main(frog_path, dau_path, out_path):
    frog = open(frog_path, encoding='utf-8').read()
    dau = open(dau_path, encoding='utf-8').read()
    blocks = blocks_from(dau)
    missing = [v for v in MODES.values() if v and v not in blocks]
    if missing:
        raise SystemExit(f'missing blocks: {missing}')
    check_lexicon()

    out = []
    out.append('{#- ------------------------------------------------------------------------------------------- -#}\n')
    out.append('{#- DavidAU mode system, merged into froggeric v22.5 (2026-10-22).  The 41 {REASON:} modes of   -#}\n')
    out.append('{#- DavidAU/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, kept verbatim; froggeric\'s effort/think/tool logic   -#}\n')
    out.append('{#- is untouched.  A mode is loaded by a {REASON:<mode>} tag (i prefix = no thinking) or chosen    -#}\n')
    out.append('{#- automatically from the conversation\'s user messages (v3: + 8 xhigh combos; margin rule).     -#}\n')
    out.append('{#- ------------------------------------------------------------------------------------------- -#}\n')
    for name, var in MODES.items():
        if var:
            out.append('{%%- set %s -%%}%s{%%- endset -%%}\n' % (var, blocks[var]))
    for name, text in SHORT.items():
        out.append("{%%- set _mode_%s = %r %%}\n" % (name, text))
    out.append('{%%- set _xsep = %r %%}\n' % XSEP)

    # instruction_map
    out.append('{%- set instruction_map = {\n')
    for name, var in MODES.items():
        val = var if var else '_mode_' + name
        out.append("    '%s': %s,\n" % (name, val))
    for name, var in XCOMBOS.items():
        out.append("    '%s': _mode_xhigh + _xsep + %s,\n" % (name, var))
    out.append('} %}\n')

    # --- the insertion point in froggeric: right before the effort sentence is built
    _a = frog.find("{%- set reasoning_instructions = '' %}")
    if _a < 0 or frog.find("{%- set reasoning_instructions = '' %}", _a + 1) >= 0:
        raise SystemExit('effort anchor not unique')
    _b = frog.find("{%- endif %}\n{%- endif %}\n", _a)
    if _b < 0:
        raise SystemExit('effort anchor end not found')
    _b += len("{%- endif %}\n{%- endif %}\n")

    merged = frog[:_a] + emit_auto_block() + frog[_b:]
    # the head-system copy loop and the message loop: strip the tag from the rendered text
    old_strip = "    {%- if '<|think_minimal|>' in _part %}{%- set _part = _part.split('<|think_minimal|>') | join('') | trim %}{%- endif %}\n"
    if merged.count(old_strip) != 1:
        raise SystemExit('head strip anchor not found once')
    merged = merged.replace(old_strip, old_strip +
                            "    {%- if _mode.tag and _mode.tag in _part %}{%- set _part = _part.split(_mode.tag) | join('') | trim %}{%- endif %}\n")
    old_strip2 = "        {%- if '<|think_minimal|>' in content %}{%- set content = content.split('<|think_minimal|>') | join('') | trim %}{%- endif %}\n"
    if merged.count(old_strip2) != 1:
        raise SystemExit('message strip anchor not found once')
    merged = merged.replace(old_strip2, old_strip2 +
                            "        {%- if _mode.tag and _mode.tag in content %}{%- set content = content.split(_mode.tag) | join('') | trim %}{%- endif %}\n")
    merged = merged.replace('{%- set template_version = "qwen3.8-froggeric-v22.5" %}',
                            '{%- set template_version = "qwen3.8-froggeric-v22.5+dau-modes-v3" %}')

    open(out_path, 'w', encoding='utf-8').write(''.join(out) + merged)

    sidecar = re.sub(r'\.jinja$', '', out_path) + '.lexicon.json'
    with open(sidecar, 'w', encoding='utf-8') as fh:
        json.dump({'version': 'dau-modes-v3', 'floor': FLOOR, 'margin': MARGIN,
                   'order': AUTO_ORDER,
                   'lexicon': {m: {str(w): ws for w, ws in MODE_LEX[m].items()} for m in AUTO_ORDER},
                   'punct': PUNCT}, fh, ensure_ascii=False, indent=1)
    print('wrote', out_path, len(merged), '+ header;', sidecar)


if __name__ == '__main__':
    main(*sys.argv[1:4])
