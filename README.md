# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next-NVFP4 混合推理部署指南。

> [!WARNING]
> 社区实验复现，非官方支持方案。请勿表述为"SM89 已获上游官方支持"。
> v4 栈依赖**三个未上游化的本地补丁**（其中 decode-interleave 对应的上游 PR #484 仍是 OPEN）。

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
| 引擎 | **FreeToken `0.1.3`** + [rendezvous 自动避让补丁](patches/04_ft_auto_dist_port.patch) + [qwen3.8 workload 补丁](patches/05_ft_workload_qwen38.patch) + [decode-interleave 补丁](patches/06_ft_decode_interleave.patch)（**三个补丁均未上游化**） |
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

测试主机：单 4090 48GB + 38 核 CPU + 247GB RAM。同机、客户端计时、独占窗口：

| 指标 | **v4（FreeToken 0.1.3）** | v3（sglang v0.5.20） |
|---|---|---|
| 模型加载 | **~39 秒** | ~5.5 分钟 |
| decode（短 / 73K / 145K 上下文） | **64.4 / 62.6 / 60.2 t/s** | 42.5 / 42.0 / 39.4 |
| prefill（64K / 254K） | **2140 / 2081 t/s** | 1887 / 1710 |
| 短请求 TTFT | 3.4s | 0.42s |
| Anthropic `/v1/messages` | ✅ 原生 | ❌ |
| 上下文 / 显存 | 256K / 44–46GB | 256K / 41.7GB |

**取舍**：v4 在长上下文 decode（+50%）、超长 prefill（+22%）、加载（8×）和 Anthropic 协议支持上明显更好；
代价是 8K prefill、短请求延迟（**3.4s 固定开销** vs 0.42s）和高并发扩展性不如 v3，
多路并行时仍会被长 prefill 打断吐字（打上 #484 补丁后最长停顿 17.7s → 7.5s）。
两条路线的脚本与补丁都在仓库里，可双向回滚。

> 完整数据、测量条件、口径说明（并发有「聚合/每流」两种口径，跨引擎不可直接比）、
> 单变量调优结论，以及第三方工具 [llm_speedtest](https://github.com/gengchaogit/llm_speedtest)
> 的交叉验证（decode 62–66 t/s 独立佐证），见 [docs/BASELINE.md](docs/BASELINE.md)。
>
> 测速前务必确认窗口独占（[TROUBLESHOOTING #18](docs/TROUBLESHOOTING.md)）并控制温度（[#11](docs/TROUBLESHOOTING.md)）。

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

**端口**：`8000` = API，`8001` = nginx（既有反代不动）。FreeToken 内部的 rendezvous 端口
**自动挑空闲的**（从 API 端口+1 起找）——反向代理占着相邻端口、同机跑第二个实例、
上个进程没退干净，都不会再挡启动。需要固定时显式设 `FT_DIST_PORT` 即可。

详细步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档 (Docs)

- [完整复现](docs/REPRODUCTION.md)
- [升级记录（v4 FreeToken / v3 sglang / v2 / v1）](docs/UPGRADE.md)
- [运维手册](docs/OPERATIONS.md)
- [排障（含 FreeToken 专属坑：/health 提前 200、KV 默认过小、moe-strategy auto 陷阱等）](docs/TROUBLESHOOTING.md)
- [实测基线 + 调优结论 + 交叉验证](docs/BASELINE.md)

## 与相关项目的关系 (Relation to other projects)

- **FlashML-org/FreeToken**（当前引擎）：MoE offload 推理运行时，提供 OpenAI 与 Anthropic 兼容 API。
  本项目在其 `0.1.3` 上验证 4090 单卡 NVFP4 路线，并携带三个未上游化的本地补丁。
- **官方 sglang**：v3 路线（已转为历史）。v0.5.20 是首个含 qwen4_exp 的正式 tag；官方硬件矩阵仍不含 4090/SM89。
- **Lsglang**（guqiong96）：v1/v2 的引擎与 lk_moe 来源；主线已转向 GLM-5.3 / DeepSeek-V4.1，release 均不含 Flash-Next。
- **lovedheart/sglang**：v1 的模型支持层，2026-08-27 后冻结。
- **TomPython/Qwen-3.8-Flash-Next**：双卡参考项目；本项目为单卡路线，两者互补。

## License

Apache-2.0（补丁逻辑受上游 FlashAttention / FreeToken 许可证约束；原创脚本与文档 Apache-2.0）。
