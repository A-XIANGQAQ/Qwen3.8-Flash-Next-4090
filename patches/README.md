# 补丁索引

**v5.9 起生产模板是 `froggeric_v22.5.jinja`（纯 froggeric，28,234 B / 28,226 字符）。**
`08` 系列是 v5.3–v5.8（DavidAU IQ3_S）的合并模板 + 自动选模式，已退役、仅供历史复现。
补丁只改模板/配置层，**服务端代码一行没改**。

| 文件 | 用途 |
|---|---|
| `froggeric_v22.5.jinja` | **当前生产模板**（28,234 B / 28,226 字符）：上游 froggeric v22.5 原版，未改一字 |
| `08_qwen38_froggeric_v225_with_dau_modes.jinja` | 退役（v5.3–v5.8 生产，249,545 B）：froggeric v22.5 + 29 段模式正文 + 8 个 xhigh 组合键 + 自动选模式 v3 |
| `08_merge_template.py` | 生成器；`MODE_LEX` 是自动选择的词表 |
| `08_qwen38_froggeric_v225_with_dau_modes.lexicon.json` | 词表快照，供评测工具读 |
| `08_auto_check.py` | 渲染真实模板认模式：`--set dev+val` / `--explain` / `--case` |
| `08_autolex_gen.py` / `08_autolex_mine.py` | 生成改写语料 / 按 lift + 边界熵挖候选词 |

规则与评测见 [docs/TEMPLATE.md](../docs/TEMPLATE.md)，运维命令见 [OPERATIONS §1a](../docs/OPERATIONS.md)。

## 历史

`archive/patches/` 保留已删除运行时的补丁：01–03（sglang v0.5.20）、04–06（FreeToken 0.1.3）、
07（Strata ESP 家族白名单，只对 UD-Q4_K_XL 回滚有意义）。**不再维护**；详情在 git 历史与
[docs/HISTORY.md](../docs/HISTORY.md)。
