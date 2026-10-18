#!/usr/bin/env python3
"""Merge the DavidAU mode system (41 {REASON:} modes) into froggeric's v22.5 Qwen3.8 template.

Template-only integration: no server patch.  The result keeps froggeric's logic (effort aliases, think markers,
tool calling, truncation, warnings) and adds, before the effort sentence is built:

  * the 29 big `system_instruction_*` texts from the model's own template, verbatim,
  * the `{REASON:<mode>}` tag parser (last tag wins, `i` prefix = no thinking),
  * automatic mode choice by keyword over the conversation's user messages (so nobody has to type tags),
  * `instruction_map`, the same 41 keys DavidAU's template has.

Usage: python3 merge_template.py <froggeric.jinja> <davidau.jinja> <out.jinja>
"""
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

# auto mode: (mode, strong words counted twice, normal words counted once) -- two points pick it
AUTO = [
    ('coder', ['```', 'traceback', 'segfault', '代码', '报错'],
     ['函数', '编译', '重构', '单元测试', 'bug', 'api', '接口', '脚本', '算法', '数据结构', '部署', '依赖',
      'python', 'javascript', 'typescript', 'golang', 'rust', 'java', 'sql', 'regex', 'docker', 'npm', 'pip',
      'cargo', 'cuda', 'pytorch', '异常', '变量', '数组', '字符串']),
    ('matlock', ['诉讼', '判例', '竞业'],
     ['法律', '法规', '法条', '合同', '条款', '仲裁', '合规', '侵权', '违约', '劳动法', '律师']),
    ('house', ['诊断', '病历'],
     ['症状', '病因', '化验', '检查报告', '指标异常', '用药', '临床', '头晕', '疼痛', '发烧']),
    ('business', ['商业模式', '创业'],
     ['商业', '市场', '营销', '定价', '营收', '利润', '获客', '融资', 'bp', '竞品', '供应链', '加盟',
      '开店', '变现', '生意']),
    ('spoon', ['研究现状', '证据链', '白皮书'],
     ['调研', '综述', '文献', '深度报告', '系统性', '横向对比']),
    ('story', ['故事', '小说', '剧本'],
     ['散文', '叙事', '人物设定', '对白', '世界观', '情节', '写一篇', '写个']),
    ('prompt', ['提示词', 'prompt'], ['系统提示', '指令模板', '角色设定']),
    ('comedy', ['段子', '脱口秀'], ['笑话', '吐槽', '搞笑', '梗', '相声']),
    ('brainstorm', ['头脑风暴', '点子'], ['创意', '想法', '灵感', '命名', '起名']),
    ('logic', ['证明', '反例', '第一性原理', '形式化'], ['推导', '是否成立', '逻辑', '漏洞', '自相矛盾']),
    ('teacher', [], ['解释', '讲清楚', '通俗', '入门', '教我', '什么是', '原理', '怎么理解', '举例说明']),
]

SHORT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
             'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
    'medium': '',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion '
           'without unnecessary elaboration.',
    'off': '',
}


def blocks_from(dau: str):
    found = {}
    for m in re.finditer(r'\{%-\s*set\s+(system_instruction_[A-Za-z0-9_]+)\s*-\%\}(.*?)\{%-\s*endset\s*-\%\}',
                         dau, re.S):
        found[m.group(1)] = m.group(2)
    return found


def main(frog_path, dau_path, out_path):
    frog = open(frog_path, encoding='utf-8').read()
    dau = open(dau_path, encoding='utf-8').read()
    blocks = blocks_from(dau)
    missing = [v for v in MODES.values() if v and v not in blocks]
    if missing:
        raise SystemExit(f'missing blocks: {missing}')

    out = []
    out.append('{#- ------------------------------------------------------------------------------------------- -#}\n')
    out.append('{#- DavidAU mode system, merged into froggeric v22.5 (2026-10-18).  The 41 {REASON:} modes of    -#}\n')
    out.append('{#- DavidAU/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, kept verbatim; froggeric\'s effort/think/tool logic   -#}\n')
    out.append('{#- is untouched.  A mode is loaded by a {REASON:<mode>} tag (i prefix = no thinking) or chosen    -#}\n')
    out.append('{#- automatically from the conversation\'s user messages.                                          -#}\n')
    out.append('{#- ------------------------------------------------------------------------------------------- -#}\n')
    for name, var in MODES.items():
        if var:
            out.append('{%%- set %s -%%}%s{%%- endset -%%}\n' % (var, blocks[var]))
    for name, text in SHORT.items():
        out.append("{%%- set _mode_%s = %r %%}\n" % (name, text))

    # instruction_map
    out.append('{%- set instruction_map = {\n')
    for name, var in MODES.items():
        val = var if var else '_mode_' + name
        out.append("    '%s': %s,\n" % (name, val))
    out.append('} %}\n')

    # --- the insertion point in froggeric: right before the effort sentence is built
    _a = frog.find("{%- set reasoning_instructions = '' %}")
    if _a < 0 or frog.find("{%- set reasoning_instructions = '' %}", _a + 1) >= 0:
        raise SystemExit('effort anchor not unique')
    _b = frog.find("{%- endif %}\n{%- endif %}\n", _a)
    if _b < 0:
        raise SystemExit('effort anchor end not found')
    _b += len("{%- endif %}\n{%- endif %}\n")
    anchor = frog[_a:_b]

    new_effort = """{%- set reasoning_instructions = '' %}
{#- DavidAU modes: an explicit {REASON:<mode>} tag wins; otherwise pick one from the conversation (or nothing) -#}
{%- set _mode = namespace(name='', tag='') %}
{%- for _m in messages %}
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
    {%- set _h = namespace(coder=0, matlock=0, house=0, business=0, spoon=0, story=0, prompt=0, comedy=0, brainstorm=0, logic=0, teacher=0) %}
"""
    for name, strong, normal in AUTO:
        if strong:
            out_words = ', '.join(repr(w) for w in strong)
            new_effort += ("    {%%- for _w in [%s] %%}{%%- if _w in _t %%}{%%- set _h.%s = _h.%s + 2 %%}"
                           "{%%- endif %%}{%%- endfor %%}\n" % (out_words, name, name))
        new_effort += ("    {%%- for _w in [%s] %%}{%%- if _w in _t %%}{%%- set _h.%s = _h.%s + 1 %%}"
                       "{%%- endif %%}{%%- endfor %%}\n" % (', '.join(repr(w) for w in normal), name, name))
    new_effort += "    {%%- if _h.%s >= 2 %%}{%%- set _mode.name = '%s' %%}" % (AUTO[0][0], AUTO[0][0])
    for name, _s, _n in AUTO[1:]:
        new_effort += "{%%- elif _h.%s >= 2 %%}{%%- set _mode.name = '%s' %%}" % (name, name)
    new_effort += "{%- endif %}\n{%- endif %}\n"
    new_effort += """{%- if _mode.name %}
    {%- set reasoning_instructions = instruction_map.get(_mode.name, '') %}
{%- elif ns_state.thinking %}
{%- if ns_state.effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif ns_state.effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
"""

    merged = frog[:_a] + new_effort + frog[_b:]
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
                            '{%- set template_version = "qwen3.8-froggeric-v22.5+dau-modes" %}')

    open(out_path, 'w', encoding='utf-8').write(''.join(out) + merged)
    print('wrote', out_path, len(merged), '+ header')


if __name__ == '__main__':
    main(*sys.argv[1:4])
