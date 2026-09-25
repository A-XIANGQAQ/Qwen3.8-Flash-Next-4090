# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next-NVFP4 混合推理部署指南。

> [!WARNING]
> 社区实验复现，非官方支持方案。请勿把结果表述为"SM89 已获上游官方支持"。
>
> 当前 v4 栈（FreeToken）依赖**三个未上游化的本地补丁**（见 [已验证组合](#已验证组合-verified-stack)），
> 其中 decode-interleave 对应的上游 PR #484 仍是 OPEN 状态。
> 历史 sglang 路线另需 SM89 专用的 FlashAttention PR #2751 补丁（由 Lsglang release 提供 prebuilt wheel），
> 该需求**不适用于 v4**（FreeToken 走 flashinfer，不用 flash_attn wheel）。

## 为什么有这个项目

| 方案 | 硬件 | 路线 | 覆盖 4090 单卡？ |
|---|---|---|---|
| 官方 sglang（Flash-Next 已随 **v0.5.20** 于 2026-09-18 正式发布） | H200/B200/B300/GB300/MI350X/MI355X；NVFP4 仅 Blackwell（B200/B300/GB300/RTX PRO 6000/DGX Spark） | 纯 GPU | ❌ 无 sm89 路径（v0.5.20 之前：上游 PLE 表需瞬时 47.7GB GPU 显存；该问题上游已由 #39928 修复，但晚于 v0.5.20） |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 RTX 4090 24GB + 1TiB | Docker 三层镜像 | 双卡专用 |
| **[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)** | 上游未给出本模型的单卡 4090 组合 | MoE offload 混合推理（CPU 算 experts） | 需本项目补丁与参数（见下） |
| **本项目** | **单 RTX 4090 48GB** | **uv venv + FreeToken MoE offload**（+ [三个本地补丁](patches/)） | ✅ |

要点：**126GB NVFP4 权重驻内存（CPU 计算 experts）+ GPU 算 attention/PLE/常驻层**——让单卡 48GB 显存跑 176B 模型（6B active）。

## 已验证组合 (Verified Stack)

**当前推荐：FreeToken 0.1.3 基线**（2026-09-24 起，切换记录见 [升级文档](docs/UPGRADE.md)）

| 层 | 固定身份 |
|---|---|
| 模型 | `RadixArk/Qwen3.8-Flash-Next-NVFP4`（ModelScope **master**，126GiB / 206 shards） |
| 引擎 | **FreeToken `0.1.3`** + [FT_DIST_PORT 补丁](patches/04_ft_dist_port.patch) + [qwen3.8 workload 补丁](patches/05_ft_workload_qwen38.patch) + [decode-interleave 补丁](patches/06_ft_decode_interleave.patch)（**三个补丁均未上游化**） |
| 关键依赖 | `freetoken[accel]` 拉入：torch 2.11.0（约束 `>=2.11,<2.12`）、transformers 5.16.1（`>=5.16,<5.17`）、triton 3.6.0、flashinfer-python 0.6.18.post1、sglang-kernel 0.4.5（`accel` extra） |
| Chat Template | [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)（官方兼容 Flash-Next，原生 effort 别名） |
| 环境 | python 3.12 / uv venv / CUDA 13 runtime（driver ≥ 580；**nvcc 需在 PATH**，Triton 要 JIT） |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1 |

<details>
<summary>旧组合（历史）：v3 = 官方 sglang v0.5.20；v2 = Lsglang 0.5.19-lkmoe + PR #37500；v1 = Lsglang 1.4.13</summary>

| 路线 | 引擎 |
|---|---|
| v3（2026-09-23） | 官方 sglang `v0.5.20` + lk_moe 2.4.1 + PLE 补丁（自建 wheel，[patches/01-03](patches/)） |
| v2（2026-09-14） | Lsglang `0.5.19-lkmoe` 分支合并上游 PR #37500 + PLE 补丁 |
| v1（原始，最省事） | Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内）+ lk_moe 2.4.0 |

三条历史路线的脚本与补丁**全部保留在仓库中**（`scripts/start_lsglang*.sh`、`scripts/build_upstream_flashnext.sh`、
`scripts/install.sh`、`patches/01-03`），仍可复现；各自验证数据见 [升级文档](docs/UPGRADE.md)。

</details>

> 单卡 24GB 原版未验证（sglang 路线下 cache+权重 GPU 部分 ~30GB 超出）。**启动需 memlock 放开**
> （`ulimit -l unlimited`）—— FreeToken 下这是为了**专家 host banks 的 mlock 常驻**（PLE 表默认走
> `--ple-backend disk`，io_uring + O_DIRECT，不占锁页内存）。仓库启动脚本已内置。

## 实测摘要 (Measured Results)

测试主机：单 4090 48GB + 38 核 CPU + 247GB RAM。**客户端计时口径**。以下为 2026-09-24 v4 栈（FreeToken 0.1.3）实测：

| 指标 | 数值 | 测量条件 |
|---|---|---|
| 模型加载 | **~39 秒** | 206 shards 权重 + 专家 banks |
| decode 512（短上下文） | **64.4 t/s** | moe-cache-rate 0.40 |
| decode @73K 上下文 | **62.6–63.6 t/s** | moe-cache-rate 0.40 |
| decode @145K 上下文 | **60.2 t/s** | moe-cache-rate 0.35（未复测 0.40） |
| prefill 64K | **2140 t/s** | |
| prefill 254K | **2081–2086 t/s**（122–149s） | |
| prefill 8K | **方差极大**（实测 371–1566 t/s） | ⚠️ 见下 |
| 并发聚合（1/2/4/8） | **35.6 / 54.0 / 64.2 / 63.7 t/s** | ⚠️ 墙钟口径（**含 TTFT**）；mrr=2、cache-rate 0.35 |
| 并发每流（1/2/4/8，纯解码） | **58.1 / 48.0 / 49.5 / 49.0 t/s** | 同上，分母只取 token 间隔，不含 TTFT |
| TTFT（短请求） | **3.4s**（前缀全命中也要 ~4.0s） | ⚠️ 见下 |
| 上下文 | 262144（256K）✅ | KV 单一共享池 265216 tokens |
| GPU 显存 | ~44–46GB / 48GB | 随 moe-cache-rate 与 mrr 变化 |
| host RAM | 峰值 ~200GB（126GiB 权重 + 63.3G 专家 banks） | |

> ⚠️ **8K prefill 方差极大**：同配置实测出现过 371 / 938 / 1216 / 1347 / 1566 t/s 多个值，
> 公开引用请给区间并注明窗口是否独占、缓存冷热。**不要引用单点值。**
> ⚠️ **短请求有 ~3.4s 固定开销**（sglang 路线是 0.42s）——本栈适合长上下文与吞吐场景，不适合高频短请求。
> ⚠️ **并发有两个口径，别混用**：**聚合** = 总 token ÷ 墙钟（**把 TTFT 算进分母**，所以 N=1 时反而低于单流速率）；
> **每流** = token ÷ 各 token 间隔之和（纯解码）。引用时必须写明是哪一个。
> 本页跨引擎对照里，v3 的历史值是**不含 TTFT** 的口径，与 v4 的聚合值**不可直接比**——已在表格中标注。

**数据出处**：主矩阵（8K/64K/145K、并发）来自 2026-09-23 的独占窗口实测（`moe-cache-rate 0.35`、`mrr=2`）；
decode 短上下文的 64.4 与 73K 的 62.6–63.6 来自 2026-09-24 的调优对照（`moe-cache-rate 0.40`），为单次测量。
两者配置不同、时间不同，**不要跨行组合引用**。调优的相对结论见下表。

### 与 v3（sglang）的取舍 (v4 vs v3 Trade-offs)

v4 **不是全面更快**，是一次明确的取舍。同一台机器、客户端计时：

| 维度 | v4（FreeToken） | v3（sglang v0.5.20） | 结果 |
|---|---|---|---|
| decode @73K | **62.6–63.6 t/s** | 42.0 t/s | ✅ **+49~51%** |
| decode @145K | **60.2 t/s** | 39.4 t/s | ✅ **+53%** |
| prefill 254K | **2081 t/s**（122s） | 1710 t/s（149s） | ✅ +22% |
| 模型加载 | **~39 秒** | ~5.5 分钟 | ✅ **~8×** |
| Anthropic 接口 `/v1/messages` | ✅ 原生 | ❌ 无 | ✅ |
| prefill 8K | 371–1566 t/s | 1968–1994 t/s | ❌ ~0.2–0.8× |
| 短请求固定开销 | **3.4s** | 0.42s | ❌ 8× |
| 并发 4 路 | 每流纯解码 **49.5 t/s**（聚合 64.2，含 TTFT） | 历史值 72.3（**口径不含 TTFT**） | ⚠️ **口径不同，不可直接比** |
| 多流冻结（长 prefill 插队） | 最长 17.7s → **7.5s**（打 #484 后） | 切成 4s 块插队 | ❌ 仍劣 |
| flash_attn wheel 依赖 | 无（走 flashinfer） | 需 SM89 专用 PR #2751 wheel | ✅ 部署更简单 |

> 选型建议：**长上下文单流 / 需要 Anthropic 协议 / 在意加载时间** → v4；
> **高频短请求 / 高并发批量 / 在意 8K prefill** → v3（脚本与补丁仍在本仓库，可直接回滚）。

> **测速前必须确认窗口独占**：外部客户端会不定时打服务端口，实测把 145K decode 从 39.4 压到 8.8 t/s（并发污染）。
> 查服务日志 `#running-req` 与 HTTP 行确认（[TROUBLESHOOTING #18](docs/TROUBLESHOOTING.md)）。**同时控制温度**（[#11](docs/TROUBLESHOOTING.md)）。

### 第三方工具交叉验证 (Cross-check with llm_speedtest)

用 [gengchaogit/llm_speedtest](https://github.com/gengchaogit/llm_speedtest) v3 Python 后端版（FastAPI + WebSocket，
走它自己的测量代码）对 v4 栈独立复测。2026-09-24，独占窗口，3 次重复：

| 提示词长度 | Prefill t/s | Decode t/s | TTFT |
|---|---|---|---|
| 1,000 | 354.6 / 355.3 | 62.3 / 65.9 | 2822 / 2817 ms |
| 8,000 | 2226.1 / 2245.9 / 2231.6 | 64.5 / 65.8 / 63.6 | 3597 / 3565 / 3587 ms |

> **口径不同于本仓库**，不能直接对表：
>
> | | 本仓库 `bench_once.py` | llm_speedtest |
> |---|---|---|
> | prefill 分母 | 总墙钟（含 32 token 解码） | **TTFT**（首个内容 token） |
> | 传输 | 非流式 | 流式 |
> | 并发聚合 | — | 总 token ÷ 墙钟（含 TTFT） |

**两点结论**：

1. **decode 得到独立佐证**：62–66 t/s，与本页 FreeToken 的 60–65 t/s 区间吻合（重复性好，3 次波动 <4%）。
2. **prefill 列不可当绝对吞吐用**：该列 = `prompt_tokens ÷ TTFT`，而 TTFT 里含着 ~2.8 秒的固定开销。
   证据是 1,000 token 那档掉到 **355 t/s** —— 用两次测量做分解：
   `(8000−1000) ÷ (3.587−2.817) ≈ 9,100 t/s` 是边际 prefill 速率，其余 ~2.7s 是固定开销。
   所以 8,000 那档的 2,226 t/s **既低估了边际算力、又因不含生成时间而高于墙钟口径**（对照本页 64K 的 2,140 t/s）。

> 用该工具时注意：它的 `timeout` 字段单位是**毫秒**（前端输入框 `min=1000`），
> 对 <1024 token 的提示词直接当基准值用——填成秒会得到几毫秒的超时、请求必然失败。

## 快速开始 (Quick Start)

```bash
# 1. 环境安装（uv + PyPI wheel，含版本约束说明）
uv venv ~/ft-venv --python 3.12
uv pip install --python ~/ft-venv "freetoken[accel]"
sudo bash scripts/install_freetoken_patches.sh   # 打三个本地补丁（或按 patches/ 头部命令手动打）

# 2. 模型下载（ModelScope，126GiB）
modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local_dir /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
python3 scripts/verify_model_index.py /path/to/.../RadixArk--Qwen3.8-Flash-Next-NVFP4 --expected-shards 206

# 3. 编辑 scripts/start_freetoken.sh（模型路径/FT_BIN/SITE_PACKAGES）后启动
bash scripts/start_freetoken.sh
# 等日志出现 "ready to serve"（约 40 秒；注意不能用 /health 判断——见 TROUBLESHOOTING）
```

**三端口分工**：`8000` = API，`8001` = nginx（既有反代不动），`8002` = FreeToken 内部 rendezvous
（默认取 API 端口+1 会撞 8001，`FT_DIST_PORT` 补丁解耦）。

详细步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档 (Docs)

- [完整复现](docs/REPRODUCTION.md)
- [升级记录：v4 切换 FreeToken / v3 sglang v0.5.20 / v2](docs/UPGRADE.md)
  （原文件名 `UPGRADE_UPSTREAM_FLASHNEXT.md`，已重命名）
- [运维手册](docs/OPERATIONS.md)
- [排障（含 FreeToken 专属坑：/health 提前 200、KV 默认过小、moe-strategy auto 陷阱等）](docs/TROUBLESHOOTING.md)
- [实测基线 + 参数扫描结论](docs/BASELINE.md)

## 与相关项目的关系 (Relation to other projects)

- **FlashML-org/FreeToken**（当前引擎）：MoE offload 推理运行时，同时提供 OpenAI 与 Anthropic 兼容 API。
  本项目在其 `0.1.3` 上验证 4090 单卡 NVFP4 路线，并携带三个本地补丁（端口解耦 / workload 注册 / decode-interleave）；
  三者均未上游化，`uv pip install -U` 后需重打（启动脚本内置自检）。
- **官方 sglang**：v3 路线（已转为历史）。官方 sglang **main 已含 qwen4_exp**（PR #36497 经 #37500 于 2026-09-08 合并）；
  **v0.5.20（2026-09-18）是首个含它的正式 tag**。官方硬件矩阵仍不含 4090/SM89。
- **Lsglang**（guqiong96）：v1/v2 路线的引擎与 lk_moe 混合推理来源。注意 Lsglang 1.4.14+ / 1.5.x 主线分别转向
  GLM-5.3 与 DeepSeek-V4.1，**其主线 release 均不含 Flash-Next**（v1.5.6 已核验）。
- **lovedheart/sglang feat/qwen38-flash-next**：v1 的模型支持层（qwen4_exp：PLE/ngram/QSA/GDN），2026-08-27 后冻结。
- **TomPython/Qwen-3.8-Flash-Next**：双卡参考项目；本项目的单卡路线与其互补（直跑 vs Docker）。

## License

Apache-2.0（补丁逻辑受上游 FlashAttention / FreeToken 许可证约束；原创脚本与文档 Apache-2.0）。
