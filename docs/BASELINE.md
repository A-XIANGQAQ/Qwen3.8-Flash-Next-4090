# 实测基线 (Baseline)

> [!IMPORTANT]
> **勘误（2026-09）**：本页历史数字经历过两次修正，引用前请看清口径。
> 最早的 prefill 绝对数字（8k ~1195 t/s、256K ~1277 t/s）来自 2026-08-30 参数扫描期，已被空闲窗口复测取代；
> 2026-09-23 的 sglang v3 复测给出 8k ≈ 1968–1994、254K ≈ 1710 t/s。
> **2026-09-24 起引擎切换为 FreeToken（v4），下方「主结果」整体换为 v4 数据**；
> v3（sglang）数据见 [升级文档 §5.4](UPGRADE.md)。

## 测试身份 (Test Identity)

- 模型：RadixArk/Qwen3.8-Flash-Next-NVFP4（ModelScope master，126GB/206 shards）
- 引擎（当前 v4）：**FreeToken 0.1.3** + 三个本地补丁（rendezvous 自动避让 / qwen3.8 workload / decode-interleave）；
  依赖锁定见 [freeze-freetoken-0.1.3.txt](freeze-freetoken-0.1.3.txt)
- 主机：单 RTX 4090 48GB（SM89）+ 38 核 CPU + 247GB RAM
- 配置（v4 定稿）：`--max-running-requests 4`、`--num-tokens 265216`、`--moe-cache-rate 0.40`、
  `--moe-strategy offload`（显式）、`--decode-interleave-every 2`、`FREETOKEN_MAMBA_SSM_DTYPE=bfloat16`
- **口径：客户端计时**（发请求掐表 + `usage.prompt_tokens`/`completion_tokens`；用唯一随机文本保证无前缀缓存）
- **前提：窗口独占**——外部客户端不定时打服务，测速前查日志 `#running-req`/`#queue-req`（TROUBLESHOOTING #18）

## 主结果 (Main Results · v4 / 2026-09-24)

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

## 第三方工具交叉验证 (Cross-check · llm_speedtest)

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

## 调优扫描结论 (Tuning Summary)

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

## 缓存复用 (Prefix Cache)

| 场景 | 首次 | 二次 |
|---|---|---|
| 完全相同 prompt（39.8K） | 18.15s | **4.04s**（`cached_tokens=39808`） |
| 同系统提示 + 不同问题（9K 系统提示） | 15.28s | **3.88s**（`cached_tokens=32640`） |

⚠️ 即便**全命中仍要 ~4s**——这就是「短请求 3.4s 固定开销」的来源，与 sglang 的 0.42s 差距在此。

## 输出质量观察 (Quality Notes)

- thinking 关闭时 reasoning_tokens=0 ✓
- 中文长输出（512+ tokens）正常，无乱码
- 工具调用（qwen3_coder parser 默认启用）结构化正常
- effort 映射（由 chat template 提供）7 个枚举全通过
- 256K 单次请求（254k prompt）完成，无 OOM
- Anthropic 兼容接口 `/v1/messages` 可用（v4 新增）

## 对比等级 (Comparison Levels)

- 本仓库同配置多次测量：Level A
- v4 与 v3 的同机对照见 [升级文档 §4](UPGRADE.md)——**同窗口、客户端计时**，Level A
- 与 TomPython 双卡（decode 34.5 t/s @1TiB/64 核）不可直接对比（单卡 38 核 CPU 瓶颈场景不同）
- 官方 sglang Blackwell 数字（NVFP4 单 B200）：硬件/引擎不同，仅背景参考
