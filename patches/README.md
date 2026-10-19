# 补丁索引

## 现行

| 编号 | 文件 | 状态 | 说明 |
|---|---|---|---|
| **08** | `08_merge_template.py` + `08_qwen38_froggeric_v225_with_dau_modes.jinja` + `.lexicon.json` | **生产在用** | 合并模板（froggeric v22.5 + 模型自带 29 段模式正文 + 自动选模式 v2）。只改模板，不动服务端。见 [../docs/TEMPLATE.md](../docs/TEMPLATE.md) |
| 08 | `08_auto_check.py` / `08_autolex_gen.py` / `08_autolex_mine.py` | 工具 | 评测（渲染真实模板认模式）、语料生成、lift 挖词 |
| 07 | `07_strata_esp_family_whitelist.patch` | 可选 | 让 Strata 的 ESP 家族白名单放行 `unsloth`。**当前模型自带 abliteration，不需要它**；只在回滚到 UD-Q4_K_XL 或重跑 `setup.sh` 时用 |

## 历史（运行时已删除，仅存档）

| 编号 | 面向 | 说明 |
|---|---|---|
| 01–03 | sglang v0.5.20（v3） | lk_moe 集成 / PLE 表分配到 CPU / 冲突解决文件，打 wheel 用 |
| 04–06 | FreeToken 0.1.3（v4） | rendezvous 端口避让 / workload / decode-interleave（上游 PR #484 仍 OPEN） |

历史补丁不再维护，也没有对应引擎可打；保留只为让 [../docs/UPGRADE.md](../docs/UPGRADE.md) 里的记录可对照。
