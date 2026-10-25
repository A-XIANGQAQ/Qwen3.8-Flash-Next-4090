# 补丁索引

当前只有 **08 系列**在生产：合并模板 + 自动选模式。**服务端代码一行没改。**

| 文件 | 用途 |
|---|---|
| `08_qwen38_froggeric_v225_with_dau_modes.jinja` | **生产模板**（249,545 B）：froggeric v22.5 + 29 段模式正文 + 8 个 xhigh 组合键 + 自动选模式 v3 |
| `08_merge_template.py` | 生成器；`MODE_LEX` 是自动选择的词表 |
| `08_qwen38_froggeric_v225_with_dau_modes.lexicon.json` | 词表快照，供评测工具读 |
| `08_auto_check.py` | 渲染真实模板认模式：`--set dev+val` / `--explain` / `--case` |
| `08_autolex_gen.py` / `08_autolex_mine.py` | 生成改写语料 / 按 lift + 边界熵挖候选词 |

规则与评测见 [docs/TEMPLATE.md](../docs/TEMPLATE.md)，运维命令见 [OPERATIONS §1a](../docs/OPERATIONS.md)。

## 历史

`archive/patches/` 保留已删除运行时的补丁：01–03（sglang v0.5.20）、04–06（FreeToken 0.1.3）、
07（Strata ESP 家族白名单，只对 UD-Q4_K_XL 回滚有意义）。**不再维护**；详情在 git 历史与
[docs/HISTORY.md](../docs/HISTORY.md)。
