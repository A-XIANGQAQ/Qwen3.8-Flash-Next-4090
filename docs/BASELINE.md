# 实测基线 (Baseline)

## 测试身份 (Test Identity)

- 模型：RadixArk/Qwen3.8-Flash-Next-NVFP4（ModelScope master，126GB/206 shards）
- 引擎：Lsglang 1.4.13 + lk_moe 2.4.0 + flash_attn 2.8.4+pr2751
- 主机：单 RTX 4090 48GB（SM89）+ 38 核 CPU + 247GB RAM
- 配置：6 层 resident（0-5）、LK36、prefetch 关、省电关、mrr2、TP1、256K
- **口径：客户端计时**（请求掐表 ÷ usage.prompt_tokens；8k 用唯一随机文本保证无前缀缓存，日志确认 cached-token=0）

## 主结果 (Main Results)

| 指标 | 数值 | 备注 |
|---|---|---|
| 8k prefill（无缓存） | **1195 t/s**（6.7s/8k） | 5 次均值，稳定 |
| 256K prefill | **1277 t/s**（254k tokens / 199s） | 32 chunk 流水线不拖慢 |
| decode 512 | **36~40 t/s** | 温度漂移 ±7%（见 TROUBLESHOOTING #11） |
| 服务内部 gen throughput | 38~40 t/s | 仅参考 |
| GPU 显存 | ~43GB / 48GB | KV 265216 + Mamba 61 states |
| KV 页表 | 265216 tokens | ≥262144 才保 256K |

## resident 层数扫描 (Resident Layer Sweep)

同窗口客户端口径，prefill 无漂移：

| 层数 | 8k prefill | decode | 256K |
|---|---|---|---|
| 0 | 1093 t/s | ~35.6 | ✅ |
| **6（默认）** | **1195 t/s** | ~37.8 | ✅ |
| 10 | 1280 t/s | ~42.2 | ❌ KV 仅 70k |

结论：常驻层对 prefill 真实有效（+9%）；decode 提升小（CPU 瓶颈）；**6 层是保 256K 的天花板**（7 层 KV 255616 差口气，推算规律：Mamba 先让位、见底后 KV 暴缩）。

## 全参数扫描结论 (Parameter Sweep Summary)

单变量扫描（每配置 5+5 次，漂移校正后）：

| 参数 | 结论 |
|---|---|
| `--chunked-prefill-size` | **4096 是坑（-38%）**；8192/16384 无差异 → 用 8192 |
| `--cuda-graph-backend-prefill full` | 崩溃（QSA 冲突）→ disabled |
| `--disable-shared-experts-fusion` | **必须保留**（去掉 → 加载 OOM） |
| `--max-total-tokens` | **必须显式**（删除 → 加载 OOM）；265216 = 256K+3k 合理值 |
| `LVLLM_GPU_PREFETCH_WINDOW=0` | 单独关闭安全；decode 数据同窗口最好之一 |
| `LK_POWER_SAVING=0` | 单独关闭安全 |
| `LK_THREADS` 36 vs 38 | 无差异（留 2 核零代价） |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` 512/1024/2048 | 无差异 |
| MTP（NEXTN） | 可启动无收益（CPU 瓶颈） |
| 并发 mrr 1/2/4 | 单请求 decode 受漂移影响无法区分；2 为默认 |

⚠️ 注意：`PREFETCH_WINDOW=0`/`POWER_SAVING=0`/`max-running-requests 4` 与 **NUMA 注释组合**会触发 ngram OOM（TROUBLESHOOTING #1）——本仓库默认配置（NUMA 启用 + 上述单项改动）是实测安全的。

## 输出质量观察 (Quality Notes)

- thinking 关闭（enable_thinking=false）时 reasoning_tokens=0 ✓
- 中文长输出（512+ tokens）正常，无乱码
- 工具调用（qwen3_coder parser）结构化正常
- effort 映射（patches）后 7 个 API 枚举全通过
- 256K 单次请求（254k prompt + 16 输出）完成，无 OOM

## 对比等级 (Comparison Levels)

- 本仓库同配置多次测量：Level A
- 与 TomPython 双卡（decode 34.5 t/s @1TiB/64 核）不可直接对比（单卡 38 核 CPU 瓶颈场景不同）
- 官方 sglang Blackwell 数字（NVFP4 单 B200）：硬件/引擎不同，仅背景参考
