# 合并模板（patch 08）：模式自动选择 v3

一份 `chat_template.jinja` 同时干两件事：保持 froggeric v22.5 的渲染修复（effort 别名、think 标记、工具调用、
截断、连续失败告警），并把模型自带的 29 段模式正文和 8 个 xhigh 组合键接进来、由模板自己挑模式。
**服务端代码一行没改。**

为什么把它单列一篇：这是本仓库唯一改"模型行为"而不是"部署"的部分，逻辑全在一份 Jinja 里，改它要跑评测。

- 模板：[`patches/08_qwen38_froggeric_v225_with_dau_modes.jinja`](../patches/08_qwen38_froggeric_v225_with_dau_modes.jinja)（249,545 字节）
- 生成器：[`patches/08_merge_template.py`](../patches/08_merge_template.py)（`MODE_LEX` 就是词表）
- 词表快照：[`patches/08_qwen38_froggeric_v225_with_dau_modes.lexicon.json`](../patches/08_qwen38_froggeric_v225_with_dau_modes.lexicon.json)
- 评测/挖词：[`08_auto_check.py`](../patches/08_auto_check.py)、[`08_autolex_gen.py`](../patches/08_autolex_gen.py)、[`08_autolex_mine.py`](../patches/08_autolex_mine.py)，用法见 [运维 §1a](OPERATIONS.md)

---

模型自带 41 个可用 `{REASON:xxx}` 键（29 段模式正文 + 8 个 xhigh 组合 + 4 个 effort 档）。DavidAU 原装模板要用户手写标签才加载——
没人愿意记这些名字，所以把两件事**在一份模板里**合成：

| | |
|---|---|
| 底子 | [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)（effort 别名、think 标记、工具调用、截断、连续失败告警） |
| 加上 | 模型自带的 29 段模式正文（逐字节照搬）+ 8 个 xhigh 组合键（`_mode_xhigh + _xsep + 正文`）+ `instruction_map` 41 键 + `{REASON:}` 解析 + 自动选择 |
| 产物 | [`patches/08_qwen38_froggeric_v225_with_dau_modes.jinja`](../patches/08_qwen38_froggeric_v225_with_dau_modes.jinja)（249,545 字节；`/props` 243,742 字符） |
| 词表 | 生成器 [`patches/08_merge_template.py`](../patches/08_merge_template.py) 里的 `MODE_LEX`；生成时同时写一份 `.lexicon.json` 供评测工具读 |
| 装机 | `cp ... <pack>/tokenizer/chat_template.jinja`，然后重启（约 40 秒就绪） |

**服务端代码一行没改**。判据：`diff -r strata/Strata-0.2.2/serve <官方 v0.2.2 解压包>/serve` 为空；本地没有任何 `serve/` 补丁。

## 判定规则（v2：三条同时满足才选）

1. **分数 ≥ 2**。词表分三层：`3` = 无歧义（`traceback`/`代码`/`竞业`/`体检`/`第一性原理`…），`2` = 主题词，`1` = 弱提示。
2. **边际胜出**：最高分必须**严格高于**第二名（平分 → 不选）。这一条取代了 v1 的固定优先级——v1 里 `matlock` 4 分
   会输给 `coder` 2 分，只要 coder 排在前面。
3. **判不准就不选**：回落到 froggeric 默认（`medium`），绝不硬猜。误套 persona 的代价远大于不套。

匹配细节：**ASCII 单词走词边界**（先把中英文标点拍平成空格，`trust` 不会命中 `rust`、`pipeline` 不会命中 `pip`）；
中文词与 ```` ``` ```` 走子串。只看 **user 消息**——不扫 system：一段真实的英文 agent 系统提示（3,421 字符）
单独就能打 `coder 4 / prompt 2`，扫进去等于把模式钉死在 coder，而且会随客户端提示词漂移。

判定看的是**整个会话的 user 文本**（最近 4000 字符），所以一个会话里一旦命中就粘住，不会来回翻、不吃前缀缓存。

## xhigh 组合键（v3 补回）

模型卡和模板内置的 help 菜单都列了 8 个组合模式：`xomni / xdeeptree / xhyper / xsocrates / xlogic / xhelp /
xeinstein / xspoon`，语义是「Qwen 的 xhigh 指令 + 该模式的正文」。v2 的 `instruction_map` 只搬了 29 段正文，
这 8 个键**被静默丢弃**——用户照着模型卡写 `{REASON:xomni}` 时模板不认识，等于没写（`08_auto_check --case` 会报 `(none)`）。

v3 按 DavidAU 原装的拼法补回（分隔句也逐字照搬，只把原文的 `guidelinges` 拼写修正为 `guidelines`）：

```jinja
{%- set _xsep = '\n\nIn addition to following "xhigh" directives, use the following directives, instructions and guidelines to process the user prompt and output the final answer:\n' %}
'xomni': _mode_xhigh + _xsep + system_instruction_DAU,
```

它们是**显式专用**：不进 `AUTO_ORDER` / `MODE_LEX`，自动选择永远不会挑组合键（否则会和基础模式的词表分数互相干扰）。
线上实测（同一句 `你好` 的 `prompt_tokens`）：

| 请求 | prompt tokens | 说明 |
|---|---:|---|
| `{REASON:omni} 你好` | 959 | 基础模式正文 |
| `{REASON:xomni} 你好` | **1,027** | +68 = xhigh 指令 + 分隔句 |
| `{REASON:ixspoon} 你好` | 908 | instruct 版（`i` 前缀） |

`08_auto_check.py` 的识别顺序也改了：先认「xhigh 句 + 分隔句 + 正文头 80 字符」，再认普通正文——否则
`xomni` 会被误报成 `xhigh` 或 `omni`。

## 实测（2026-10-19 定规则；2026-10-22 在 0.2.2/0.1.47 上复核，同一台 4090）

| 集合 | 规模 | 命中 | 错选 | 弃权 | 准确率 |
|---|---:|---:|---:|---:|---:|
| dev（调参集） | 73 | 70 | **0** | 3 | **95.9%** |
| val（每轮检查） | 30 | 27 | **0** | 3 | **90.0%** |
| test（锁定，只跑一次） | 40 | 29 | **0** | 11 | **72.5%** |

**没达到计划里「test ≥ 90%」的目标**。全部错例都是弃权，没有一条选错模式；差距来自词表仍不覆盖手写改写——
`git rebase`、`长期头痛`、`客户`、`挂号`、`营销`(只有 1 分)、`融资`(只有 1 分) 这类。要继续提升必须先写**新的**锁定
测试集：现有的 test 已经被看过，不能再当验收集用。

2026-10-22 加 xhigh 组合键后重跑 dev/val：命中数与错选**完全不变**（70/73、27/30，0 错选，`[DRIFT]` 0 条）。

线上 spot check（重启后，`max_tokens=1` 读 `prompt_tokens`）：

| 请求 | prompt tokens | 模式 |
|---|---:|---|
| `你好` | 11 | 不选（连系统提示都不发） |
| `讲个笑话` | **2,847** | comedy（v1 时是 13 = 不选） |
| `这个 bug 的法条依据是什么` | 17 | 不选（双主题，弃权） |
| `I trust you, this pipeline is rapid` | 18 | 不选（词边界生效） |
| `diagnose this segfault` | 2,717 | coder |
| `{REASON:xomni} 你好` | 1,027 | xomni（v3 补回的组合键） |

渲染成本：v1 0.375 ms → v2 **0.446 ms**（200 次均值，同一台机器）。贵的那 0.07 ms 就是多算 200 个词的边界匹配。

## 评测工具

| 工具 | 用途 |
|---|---|
| [`patches/08_auto_check.py`](../patches/08_auto_check.py) | **渲染真实模板**认模式（不写 Python 复刻）：`--set dev\|val\|test\|dev+val`、`--explain` 打印分数、`--case "…"` 单条排查；镜像与模板不一致会报 `[DRIFT]` |
| [`patches/08_autolex_gen.py`](../patches/08_autolex_gen.py) | 用本地模型按模式生成改写语料（每条请求带 `{REASON:ioff}`——不关模式的话，模式正文会把「每行一条问题」的输出格式污染成 `⎔⎔⎔ teacher ⎔`） |
| [`patches/08_autolex_mine.py`](../patches/08_autolex_mine.py) | 挖 2–4 字短语 + 英文词，按 `lift = P(词\|本模式)/P(词\|其他)` 排序；带**左右边界熵**过滤，去掉 `看看这段`/`该怎` 这类碎片 |

改词表的顺序：`08_autolex_gen.py` 生成语料 → `08_autolex_mine.py` 出候选 → 人工复核 → 改 `MODE_LEX` →
`08_merge_template.py` 重新生成 → `08_auto_check.py --set dev+val` 看分数 → 装机。

⚠️ 三个边界：
1. `{REASON:i<mode>}`（i = 不要思考）**管不住服务端的解析**——标签只改模板前缀，答案仍会落进 `reasoning_content`；
   要真关思考请用客户端开关（`enable_thinking:false`）；2026-10-22 实测 `{REASON:ixspoon}` 仍返回 `reasoning_content`。
2. `setup.sh` 每次都会从 GGUF **重新导出**原装模板，覆盖这一份，见 [TROUBLESHOOTING #49](TROUBLESHOOTING.md)。
3. 同一会话里换过模式时，旧标签会以字面量留在上下文里（只剥胜出的那一个），实测多 6 个 token；换模式建议新开会话。

---

## 怎么改它（顺序不要乱）

```bash
cd qwen4090-repo/patches
python3 08_autolex_gen.py --out /tmp/corpus.jsonl --per-mode 40   # 1. 生成改写语料（约 3 分钟 GPU）
python3 08_autolex_mine.py --corpus /tmp/corpus.jsonl --top 22    # 2. 挖候选词（lift + 边界熵）
# 3. 人工复核，改 08_merge_template.py 的 MODE_LEX（每个新词都要能追到一条语料证据）
python3 08_merge_template.py <froggeric> <原装> /tmp/new.jinja    # 4. 重新生成
python3 08_auto_check.py --template /tmp/new.jinja --set dev+val  # 5. 评测
# 6. 装机：cp <pack>/tokenizer/chat_template.jinja 之前先备份，然后重启
```

**规矩**：新增词/参数必须先有一条现在失败的用例；`test` 集只跑一次，跑过就要重新写才能再当验收集用。

## 历史

| 版本 | 日期 | 变化 | dev 准确率 |
|---|---|---|---:|
| v1 | 2026-10-18 | 裸子串 + 固定优先级（强 2 分 / 普通 1 分，第一个过线即选） | 82.2%（73 条） |
| v2 | 2026-10-19 | 三层权重 + ASCII 词边界 + 边际胜出（弃权）+ 词表从语料挖 | **95.9%（73 条）** |
| **v3** | 2026-10-22 | 补回 8 个 xhigh 组合键（显式专用），规则/词表**一字未动** | 95.9%（73 条，不变） |

v1 的两个硬伤：`trust` 命中 `rust`、`pipeline` 命中 `pip`（无边界）；`matlock` 4 分会被排在前面且刚过线的
`coder` 2 分抢走（固定优先级）。v2 的"边际胜出"就是修第二个：分数高者赢，平分则弃权。
