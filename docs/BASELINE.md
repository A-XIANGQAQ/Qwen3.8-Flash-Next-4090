# 实测基线 (Baseline)

> [!IMPORTANT]
> **勘误（2026-09 / 2026-10）**：本页历史数字经历过多次修正，引用前请看清口径。
> 最早的 prefill 绝对数字（8k ~1195 t/s、256K ~1277 t/s）来自 2026-08-30 参数扫描期，已被空闲窗口复测取代；
> 2026-09-23 的 sglang v3 复测给出 8k ≈ 1968–1994、254K ≈ 1710 t/s。
> **2026-09-24 → 10-13 引擎为 FreeToken（v4）**；**2026-10-13 起引擎为 Strata（v5）**，主表见下方「主结果 · v5」。
> v3（sglang）数据见 [升级文档 §5.4](UPGRADE.md)。
>
> ⚠️ **v5 只有短上下文数据**：长上下文（128K / 256K）配置与显存账支持，但**没在真实长文下跑过**；
> 8 路以上并发、多日稳定性同样未测。逐项列在 [STRATA.md §9](STRATA.md)。

## 测试身份 (Test Identity)

同一台主机：**单 RTX 4090 48GB（SM89）+ 38 核 CPU + 247GB RAM**。两代引擎跑同一个模型家族，
差别在**权重格式 + 引擎**：

| | **v5.1（当前，2026-10-18 起）** | v4（2026-09-24 起，回滚路径） |
|---|---|---|
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF` → **UD-Q4_K_XL**（ModelScope，111.33 GB / 4 shards） | `RadixArk/Qwen3.8-Flash-Next-NVFP4`（ModelScope master，126GB / 206 shards） |
| 引擎 | **Strata 0.1.45**（Linux 自建，仅需 [patch 07](../patches/07_strata_esp_family_whitelist.patch) 放行 ESP；`engine/` 含 `strata` + `strata-vision` + `libstrata_kq.so`） | **FreeToken 0.1.3** + 三个本地补丁（rendezvous 自动避让 / workload / decode-interleave） |
| 依赖锁定 | BUILD.json `archs:[89]`、CUDA 13.2 | [freeze-freetoken-0.1.3.txt](archive/freeze-freetoken-0.1.3.txt) |
| 配置定稿 | `--batch 4`、`--max-context 262144`、`--kv k8v4`、`--kv-resident 32768`、`--batch-mtp`、`--resident-budget-gib 71`、ESP 控制向量 on | `--max-running-requests 4`、`--num-tokens 265216`、`--moe-cache-rate 0.40`、`--moe-strategy offload`（显式）、`--decode-interleave-every 2`、`FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` |

- **口径：客户端计时**（发请求掐表 + `usage.prompt_tokens`/`completion_tokens`；用唯一随机文本保证无前缀缓存）
- **前提：窗口独占**——外部客户端不定时打服务，测速前查日志 `#running-req`/`#queue-req`（TROUBLESHOOTING #18）

## 主结果 · v5 (Main Results · Strata / 2026-10-13)

**同机、同 prompt（2336 token）、同一服务入口、客户端计时**，与 v4 同窗口 A/B：

| 指标 | v4（FreeToken 0.1.3） | **v5.0（Strata 0.1.43.1）** |
|---|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s**（2.0×） |
| 单流 TTFT | 2.73 s | **0.20 s**（14×） |
| 2 路 聚合 / 每路 | 68.8 / 56.7 | **124** / **65.2** |
| 4 路 聚合 / 每路 | 117.1 / **45.8** | **137** / 36.9 |
| 4 路 TTFT | 5.9 s | **0.81 s** |
| 4 路端到端（TTFT + 512 token） | 17.1 s | **14.7 s** |
| 并发扩展比（1→4 聚合） | **2.55×** | 1.16× |
| 数学（15 题） | 15/15 | 15/15 |
| 破限（强探针 14 条） | 0/14 | 0/14（ESP 关则 **14/14**） |

运行中实测资源占用（2026-10-13，常驻、无外部流量）：GPU **48,438 / 49,140 MiB**（引擎独占 48,318 MiB）；
引擎 RSS **129.5 GiB** + server 0.22 GiB；`free` 报 247 GiB 中 68 GiB used、198 GiB buff/cache（含 mmap 的 GGUF）。

> 📌 **本表的 v4 列 ≠ 下方 v4 主表**：本表同窗口 A/B 跑的是**生产配置**（`mrr=4`、`cache-rate 0.40`），
> 下方主表来自 2026-09-23/24（`mrr=2`、`cache-rate 0.35`）。**两批数字不可跨表混引。**
>
> 📌 **上表是 0.1.43.1 时代的基线，保留不动。** 当前引擎 **0.1.45**：0.1.43.1 → 0.1.45 的两臂实测
> （各自全新启动）在 [UPGRADE.md §0b](UPGRADE.md)——短上下文 decode 139.7 → **144.5 t/s**、
> 32K TTFT 9.50 → **6.99 s**、100K TTFT 24.78 → **21.82 s**、100K decode 判等、
> 常驻 RAM 专家 39.45 → **52.10 GiB**。**跨引擎（v4↔v5）与跨版本（0.1.43.1↔0.1.45）是两批独立实验，别混引。**
>
> ⚠️ **并发口径**：v5 的 `--batch N` 是**每 slot 独立 session**（VRAM 与主 session 同量，≈0.86 GiB/槽），
> **不是** llama.cpp 那套 `n_ctx ÷ N`。两列都按「聚合 = 总 token ÷ 墙钟」计，跨引擎可比。
>
> ⚠️ **数学 15/15 与破限列不构成质量结论**——样本量太小。质量结论需上 GSM8K-1319 之类
> （v4 侧有 `score: 0.9727` 的历史基线，v5 未跑）。另：Strata 走投机解码，**输出与 v4 非逐位一致**。

## 主结果 · v4 (Main Results · FreeToken / 2026-09-24)

| 指标 | 数值 | 条件 |
|---|---|---|
| decode 512（短上下文） | **64.4 t/s** | moe-cache-rate 0.40 |
| decode @73K | **62.6–63.6 t/s** | moe-cache-rate 0.40 |
| decode @145K | **60.2 t/s** | moe-cache-rate 0.35（未用 0.40 复测） |
| 并发**聚合**（1/2/4/8 路，256 tok/请求） | **35.6 / 54.0 / 64.2 / 63.7 t/s** | ⚠️ 墙钟口径，**含 TTFT**；mrr=2、cache-rate 0.35 |
| 并发**每流**（1/2/4/8 路，纯解码） | **58.1 / 48.0 / 49.5 / 49.0 t/s** | 同上，分母只取 token 间隔和 |
| ITL p50 | 15.6–16.2 ms | |
| TTFT（短请求） | **3.4s**（前缀全命中 ~4.0s） | 固定开销，非 prefill 耗时 |
| prefill 64K | **2140 t/s** | |
| prefill 254K | **2081–2086 t/s**（122–149s） | |
| prefill 8K | **371–1566 t/s（方差极大）** | ⚠️ 见下 |
| 模型加载 | **~39 秒** | 206 shards + 专家 banks（63.3G，约 25s） |
| GPU 显存 | 42.9–46.1GB / 48GB | 随 moe-cache-rate 与 mrr 变化 |
| KV 池 | 265216 tokens（= 6.26 GiB，固定） | 单一共享池 |

> ⚠️ **8K prefill 不要引用单点值**：同配置实测出现过 371 / 938 / 1216 / 1347 / 1566 t/s。
> 该指标受窗口占用、页缓存冷热、温度影响极大，公开引用请给区间并注明条件。
>
> ⚠️ **并发两个口径**：**聚合** = 总 token ÷ 墙钟（**含 TTFT**，所以 N=1 会低于单流速率）；
> **每流** = token ÷ 各 token 间隔之和（纯解码）。跨引擎对照时尤其注意——
> v3（sglang）发布的历史并发值不含 TTFT，**与本页聚合值不可直接比**。
>
> 📌 **数据出处**：主矩阵来自 2026-09-23 独占窗口实测（`moe-cache-rate 0.35`、`mrr=2`，原始日志可查）；
> decode 短上下文 64.4 与 73K 62.6–63.6 来自 2026-09-24 调优对照（`moe-cache-rate 0.40`），**单次测量**。
> 两批配置不同、时间不同，**不要跨行组合引用**。

## 与 v3（sglang）的取舍 (v4 vs v3)

同机、客户端计时。v4 **不是全面更快**，是一次明确的取舍：

| 维度 | v4（FreeToken） | v3（sglang v0.5.20） | 结果 |
|---|---|---|---|
| decode @73K | 62.6–63.6 t/s | 42.0 t/s | ✅ +49~51% |
| decode @145K | 60.2 t/s | 39.4 t/s | ✅ +53% |
| prefill 254K | 2081 t/s（122s） | 1710 t/s（149s） | ✅ +22% |
| prefill 64K | 2140 t/s | 1887 t/s | ✅ +13% |
| 模型加载 | ~39 秒 | ~5.5 分钟 | ✅ ~8× |
| Anthropic 接口 | ✅ 原生 | ❌ 无 | ✅ |
| 部署依赖 | 无 flash_attn wheel（走 flashinfer） | 需 SM89 专用 PR #2751 wheel | ✅ 更简单 |
| prefill 8K | 371–1566 t/s（方差大） | 1968–1994 t/s | ❌ ~0.2–0.8× |
| 短请求 TTFT | 3.4s | 0.42s | ❌ 8× |
| 并发 4 路 | 每流 49.5 t/s（聚合 64.2，含 TTFT） | 历史值 72.3（不含 TTFT） | ⚠️ 口径不同，见下 |
| 并发 8 路 | 63.7 t/s（聚合，含 TTFT） | 108 t/s | ❌ 高并发扩展弱 |
| 多流冻结 | 最长 17.7s → 7.5s（打 #484 后） | 切成 4s 块插队 | ❌ 仍劣 |

选型：**长上下文单流 / 需要 Anthropic 协议 / 在意加载时间** → v4；
**高频短请求 / 高并发批量 / 在意 8K prefill** → v3。

> 📌 **以上是 v4 时代的选型（2026-09）。** 2026-10-13 起的选型是 **v5 vs v4**：
> **1–2 路交互**（单流快 2×、首字快 14×）→ **v5**；**8 路以上密集并发 / 要最大聚合吞吐** → v4 仍有优势
> （扩展比 2.55× vs 1.16×）。见本节顶部对照表与 [STRATA.md §1](STRATA.md)。

## 第三方工具交叉验证 · v4 (Cross-check · llm_speedtest)

> 本节是对 **v4** 的独立复测（2026-09-24）。**v5 没有做过第三方交叉验证**——v5 的数字只有本仓库自测。

用 [gengchaogit/llm_speedtest](https://github.com/gengchaogit/llm_speedtest) v3 Python 后端版
（FastAPI + WebSocket，走它自己的测量代码）对 v4 栈独立复测。2026-09-24，独占窗口，3 次重复：

| 提示词长度 | Prefill t/s | Decode t/s | TTFT |
|---|---|---|---|
| 1,000 | 354.6 / 355.3 | 62.3 / 65.9 | 2822 / 2817 ms |
| 8,000 | 2226.1 / 2245.9 / 2231.6 | 64.5 / 65.8 / 63.6 | 3597 / 3565 / 3587 ms |

口径与本仓库 `bench_once.py` 不同，**不能直接对表**：它的 prefill 分母是 **TTFT**（首个内容 token），
且走流式；本仓库用总墙钟（含 32 token 解码）与非流式。

**两点结论**：

1. **decode 得到独立佐证**：62–66 t/s，与本页 v4 的 60–65 t/s 区间吻合（3 次波动 <4%）。
2. **它的 prefill 列不可当绝对吞吐用**：`prompt_tokens ÷ TTFT` 里含着固定开销——
   1,000 token 那档掉到 355 t/s，用两档做分解得
   `(8000−1000) ÷ (3.587−2.817) ≈ 9,100 t/s` 为边际 prefill 速率，其余 ~2.7s 是固定开销
   （与本页「短请求 3.4s 固定开销」互为印证）。

> 用该工具时注意：`timeout` 字段单位是**毫秒**（前端输入框 `min=1000`），
> 对 <1024 token 的提示词直接当基准值用——填成秒会得到几毫秒的超时、请求必然失败。

## 调优扫描结论 · v4 (Tuning Summary · FreeToken)

> 以下全部是 **v4（FreeToken）** 的参数扫描，**不适用于 v5**。v5 的关键配置项与取舍见 [STRATA.md §5](STRATA.md)。

单变量扫描（客户端计时，同窗口对照）：

| 参数 | 结论 |
|---|---|
| `--moe-cache-rate` 0.35 → **0.40** | 73K decode **+4%**（60.4/60.7 → 62.6/63.6）；短上下文 63.4 → 64.4；显存 42.9G → 46.1G。**再高收益递减且预算太紧**（prefill OOM 会崩服务） |
| `--moe-cache-rate` 不给（默认 auto） | 与 `--num-tokens` 同时缺失时 >8K prompt 直接 400；只给 num-tokens 会 prefill OOM。**两者必须同时给** |
| `--moe-strategy` **offload**（显式） | decode 62.6–63.6；对照 **hybrid 46.6（-27%）**、**cpu 35.6**。prefill 三者无差异。
⚠️ 跑过 `ft bench bw` 后 `auto` 会静默选 hybrid——bench 看带宽、实跑看单步延迟与小批量 CPU GEMV 效率（上游 #151/#436） |
| `FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` | mamba 池 1.3G → **675M**；decode 65.1（对照 64.4，不降反升） |
| `--max-running-requests` 2 → **4** | mamba 槽 12 → **24**（长上下文并发 3 路 → 6 路）；显存 44.4G；decode 65.0/65.3 @73K **不变** |
| `--decode-interleave-every` 无 → **2** | 多流最长冻结 **17.7s → 7.5s**（-58%）；总停顿不变（21.3 → 21.5s，切成 4 段）；decode 不变。
⚠️ N 要按自己的 prompt 长度定，作者推荐的 N=8 对 43K prompt 完全无效 |
| `--num-tokens 265216` | 默认 auto 只给 8256 token KV，>8K prompt 直接 400（上游 #150） |

`ft bench bw` 实测（本机带宽天花板，决定 `auto` 策略的依据）：
CPU STREAM 读 **159.4 GB/s**、PCIe H2D **25.3** / D2H **26.4 GB/s**，阈值 2.0×。
CPU 内核速率随专家几何变化：7.61MB → 135.6、**2.64MB（本模型）→ 112.7**、1.69MB → 94.5 GB/s。
→ 几何越小 CPU 优势越弱，但**即便按真实几何重测，bench 仍判 hybrid，而实跑 offload 快 27%**（见上表）。

## 缓存复用 · v4 (Prefix Cache · FreeToken)

> v5 有等价机制：Strata 自带 prefix / conversation cache（`--conversation-cache-mib 8192`），
> **但本站未对 v5 的缓存命中延迟做过测量**——下表是 v4 数字，别当作 v5 的。

| 场景 | 首次 | 二次 |
|---|---|---|
| 完全相同 prompt（39.8K） | 18.15s | **4.04s**（`cached_tokens=39808`） |
| 同系统提示 + 不同问题（9K 系统提示） | 15.28s | **3.88s**（`cached_tokens=32640`） |

⚠️ 即便**全命中仍要 ~4s**——这就是「短请求 3.4s 固定开销」的来源，与 sglang 的 0.42s 差距在此。

## 输出质量观察 · v4 (Quality Notes · FreeToken)

- thinking 关闭时 reasoning_tokens=0 ✓
- 中文长输出（512+ tokens）正常，无乱码
- 工具调用（qwen3_coder parser 默认启用）结构化正常
- effort 映射（由 chat template 提供）7 个枚举全通过
- 256K 单次请求（254k prompt）完成，无 OOM
- Anthropic 兼容接口 `/v1/messages` 可用（v4 新增；v5 的 serve 层同样提供）

**v5 已观测**（样本都很小，只说明"没坏"，**不是质量结论**）：

- 中文短请求与长输出可读、无乱码 ✓
- 数学 15 题 15/15（与 v4 并列）
- 破限强探针：**ESP 开 0/14 拒答、关 14/14** —— 同引擎同进程、逐请求切换的单变量证明
- 思考档经 froggeric v22.5 模板，7 个枚举全通
- ⚠️ **256K 单次请求、工具调用结构化、多日稳定性：v5 均未测**（见 [STRATA.md §9](STRATA.md)）

## 对比等级 (Comparison Levels)

- 本仓库同配置多次测量：Level A
- **v5 与 v4 的同机对照**：本节顶部表与 [STRATA.md §1](STRATA.md)——**同窗口、同一 prompt（2336 token）、客户端计时**，
  Level A 条件满足，但**每格为单次测量**，未做重复与方差估计
- v4 与 v3 的同机对照见 [升级文档 §4](UPGRADE.md)——**同窗口、客户端计时**，Level A
- 与 TomPython 双卡（decode 34.5 t/s @1TiB/64 核）不可直接对比（单卡 38 核 CPU 瓶颈场景不同）
- 官方 sglang Blackwell 数字（NVFP4 单 B200）：硬件/引擎不同，仅背景参考
