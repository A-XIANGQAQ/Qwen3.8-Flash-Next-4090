# 文档索引

## 现在（照这个做）

| 文档 | 什么时候看 |
|---|---|
| [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) | **当前模型怎么来的**：UD-Q4_K_XL + huihui v2 消融、pack v4、froggeric、实测与破限 |
| [BENCHMARK.md](BENCHMARK.md) | **最新实测**：prefill / decode / 并发、方法学、原始数据位置 |
| [OPERATIONS.md](OPERATIONS.md) | 启停、观察、日志、sanity 检查、同代升级 |
| [STRATA.md](STRATA.md) | 为什么用 Strata、关键参数、KV/视觉/prefill 取舍、已验证边界 |
| [TEMPLATE.md](TEMPLATE.md) | 合并模板（patch 08，**v5.9 起已退役**）：判定规则、词表、评测集，仅供复现 v5.3–v5.8 |
| [REPRODUCTION.md](REPRODUCTION.md) | 从零复现：硬件、模型下载、引擎、模板、启动、成功判据 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 症状 → 判据 → 修复（现行编号 29–50 + 通用项） |
| [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md) | 上游 0.2.12 升级记录（当前 0.1.58 配对） |
| [STRATA-UPSTREAM-2026-11-06.md](STRATA-UPSTREAM-2026-11-06.md) | 上游 0.2.11：W16 cache 为何在本机 0 MiB |
| [STRATA-UPSTREAM-2026-10-25.md](STRATA-UPSTREAM-2026-10-25.md) | 上游 0.2.3/0.2.4 评估：为什么暂缓、升级阈值 |
| [GPU-UNLOCK.md](GPU-UNLOCK.md) | GPU 张量速率解锁实测：指令级 2×、本栈无收益、回滚 |

## 历史

| 文档 | 内容 |
|---|---|
| [HISTORY.md](HISTORY.md) | 压缩版版本表（v1–v5.9）与已删除项；详细过程在 git 历史 |

运行时（FreeToken / Lsglang / 官方 sglang）已于 2026-10-18 全部删除；对应章节不再保留。
