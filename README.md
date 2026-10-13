# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next-NVFP4 混合推理部署指南。

> [!WARNING]
> 社区实验复现，非官方支持方案。请勿表述为"SM89 已获上游官方支持"。
> **v5（Strata）不需要任何本地补丁**；v4 栈依赖**三个未上游化的本地补丁**（其中 decode-interleave 对应的
> 上游 PR #484 仍是 OPEN）——这些补丁仅为 v4 回滚路径保留。
>
> ⚠️ **Strata 是 3 周龄的新引擎**（790 open issues），且走投机解码 ⇒ 输出与 v4 非逐位一致。
> 本仓库不宣称"质量等价"，只报告实测的速度与拒答率。

## 为什么有这个项目

| 方案 | 硬件 | 路线 | 覆盖 4090 单卡？ |
|---|---|---|---|
| 官方 sglang（Flash-Next 已随 **v0.5.20** 于 2026-09-18 正式发布） | H200/B200/B300/GB300/MI350X/MI355X；NVFP4 仅 Blackwell（B200/B300/GB300/RTX PRO 6000/DGX Spark） | 纯 GPU | ❌ 无 sm89 路径（v0.5.20 之前：上游 PLE 表需瞬时 47.7GB GPU 显存；该问题上游已由 #39928 修复，但晚于 v0.5.20） |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 RTX 4090 24GB + 1TiB | Docker 三层镜像 | 双卡专用 |
| **[Niko1221/Strata](https://github.com/Niko1221/Strata)** | 上游只发 **Windows** 二进制，未见单卡 4090 + 本模型的组合记录 | 自有引擎 + MoE 专家 offload + MTP | 需 Linux 自建与参数（[见 v5 路线](docs/STRATA.md)） |
| **[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)** | 上游未给出本模型的单卡 4090 组合 | MoE offload 混合推理（CPU 算 experts） | 需本项目补丁与参数（见下） |
| **本项目** | **单 RTX 4090 48GB** | **v5 = Strata**（自带 offload，**无补丁**）；v4 = FreeToken + [三个本地补丁](patches/)（保留作回滚） | ✅ |

要点：**111 GB 4-bit 权重驻内存（CPU 计算 experts）+ GPU 算 attention/PLE/常驻层**——让单卡 48GB 显存跑 176B 模型（6B active）。
v5 与 v4 的差别只在**引擎**，模型同为 Qwen3.8-Flash-Next。

## 已验证组合 (Verified Stack)

**当前推荐：v5 = Strata 0.1.43.1**（2026-10-13 起，**不需要任何本地补丁**；切换记录见 [升级文档](docs/UPGRADE.md)，
完整路线说明见 [v5 路线文档](docs/STRATA.md)）

| 层 | 固定身份 |
|---|---|
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF` → **UD-Q4_K_XL**（ModelScope，111.33 GB / 4 shards，**4-bit**） |
| 引擎 | **[Niko1221/Strata](https://github.com/Niko1221/Strata) `0.1.43.1`**（Linux 自建；CUDA 13.2 / sm89 / `BUILD.json archs:[89]`）——**无本地补丁** |
| 并发 / 上下文 | `--batch 4` × `--max-context 262144`（每 slot 独立 session） |
| KV | **`k8v4` + `--kv-resident 32768`**（816 B/cell；KV 流式到宿主内存——4×256K 全放显存会 OOM） |
| 破限 | **引擎自带 ESP 控制向量**（480 KB，层 4–44，逐请求可关）；强探针关=14/14 拒答、开=0/14 |
| Chat Template | [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)（需装进 pack 的 `tokenizer/`；**重装会被覆盖，故挂在启动器上**） |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1 |

<details>
<summary>旧组合（历史）：v4 = FreeToken 0.1.3 + 三个本地补丁；v3 = 官方 sglang v0.5.20；v2 = Lsglang 0.5.19-lkmoe + PR #37500；v1 = Lsglang 1.4.13</summary>

| 路线 | 引擎 |
|---|---|
| v4（2026-09-24） | **FreeToken `0.1.3`** + [rendezvous 避让](patches/04_ft_auto_dist_port.patch) + [workload](patches/05_ft_workload_qwen38.patch) + [decode-interleave](patches/06_ft_decode_interleave.patch)（**三个补丁均未上游化**）；模型 `RadixArk/…-NVFP4`（126 GiB / 206 shards） |
| v3（2026-09-23） | 官方 sglang `v0.5.20` + lk_moe 2.4.1 + PLE 补丁（自建 wheel，[patches/01-03](patches/)） |
| v2（2026-09-14） | Lsglang `0.5.19-lkmoe` 分支合并上游 PR #37500 + PLE 补丁 |
| v1（原始，最省事） | Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内）+ lk_moe 2.4.0 |

**v4 完整保留为回滚目标**（环境、脚本、补丁、模型全部未动），一条命令切回：
`bash local-tools/start_freetoken_noroot.sh`。
三条更早的历史路线脚本与补丁也全部保留（`scripts/start_lsglang*.sh`、`scripts/build_upstream_flashnext.sh`、
`scripts/install.sh`、`patches/01-03`），仍可复现；各自验证数据见 [升级文档](docs/UPGRADE.md)。

</details>

> 单卡 24GB 原版未验证（sglang 路线下 cache+权重 GPU 部分 ~30GB 超出）。**启动需 memlock 放开**
> （`ulimit -l unlimited`）—— FreeToken 下这是为了**专家 host banks 的 mlock 常驻**（PLE 表默认走
> `--ple-backend disk`，io_uring + O_DIRECT，不占锁页内存）。仓库启动脚本已内置。

## 实测摘要 (Measured Results)

测试主机：单 4090 48GB + 38 核 CPU + 247GB RAM。同机、客户端计时、独占窗口。

### v5（Strata） vs v4（FreeToken）—— 2026-10-13，短上下文

同一个模型（Qwen3.8-Flash-Next）、同一 prompt（2336 token）、同一服务入口：

| 指标 | v4（FreeToken 0.1.3） | **v5（Strata 0.1.43.1）** |
|---|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s**（2.0×） |
| 单流 TTFT | 2.73 s | **0.20 s**（14×） |
| 2 路 聚合 / 每路 | 68.8 / 56.7 | **124** / **65.2** |
| 4 路 聚合 / 每路 | 117.1 / **45.8** | **137** / 36.9 |
| 4 路 TTFT | 5.9 s | **0.81 s** |
| **4 路端到端**（TTFT + 512 token） | 17.1 s | **14.7 s** |
| 并发扩展比（1→4 聚合） | **2.55×** | 1.16× |
| 数学（15 题） | 15/15 | 15/15 |
| 破限（强探针 14 条） | 0/14 | 0/14（关则 14/14） |

> ⚠️ **并发口径**：v5 的 `--batch N` 是**每 slot 独立 session**（VRAM 与主 session 同量），不是
> llama.cpp 那套 `n_ctx ÷ N`；两个数字都按"聚合 = 总 token ÷ 墙钟"计，跨引擎可比。
> ⚠️ **只有短上下文数据**。v5 的长上下文（128K/256K）**未实测** —— 配置与显存账支持 4×256K，但没跑过。

<details>
<summary>v4 vs v3 对照（2026-09-24，历史）</summary>

| 指标 | **v4（FreeToken 0.1.3）** | v3（sglang v0.5.20） |
|---|---|---|
| 模型加载 | **~39 秒** | ~5.5 分钟 |
| decode（短 / 73K / 145K 上下文） | **64.4 / 62.6 / 60.2 t/s** | 42.5 / 42.0 / 39.4 |
| prefill（64K / 254K） | **2140 / 2081 t/s** | 1887 / 1710 |
| 短请求 TTFT | 3.4s | 0.42s |
| Anthropic `/v1/messages` | ✅ 原生 | ❌ |
| 上下文 / 显存 | 256K / 44–46GB | 256K / 41.7GB |

</details>

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

**端口**：只需要指定 API 端口（`--port`）。FreeToken 内部的 rendezvous 端口会**自动挑空闲的**
（从 API 端口+1 起向上找）——它是实现细节，不该成为部署约束：相邻端口被任何进程占用都不会挡启动。
需要固定时显式设 `FT_DIST_PORT`。

详细步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档 (Docs)

- [**v5 路线：Strata**（如何在 Linux 自建、模型选型、KV 格式、破限 ESP、踩坑）](docs/STRATA.md)
- [完整复现](docs/REPRODUCTION.md)
- [升级记录（v5 Strata / v4 FreeToken / v3 sglang / v2 / v1）](docs/UPGRADE.md)
- [运维手册](docs/OPERATIONS.md)
- [排障（含 FreeToken 专属坑：/health 提前 200、KV 默认过小、moe-strategy auto 陷阱等）](docs/TROUBLESHOOTING.md)
- [实测基线 + 调优结论 + 交叉验证](docs/BASELINE.md)

## 与相关项目的关系 (Relation to other projects)

- **[Niko1221/Strata](https://github.com/Niko1221/Strata)**（**当前引擎**）：自有 C++/CUDA/HIP/SYCL 推理引擎，
  复用 llama.cpp 的 ggml + gguf-py + mtmd；自带 MoE 专家 offload、prefix/conversation cache、
  以及 **ESP 控制向量破限**。本项目在其 `0.1.43.1` 上验证 4090 单卡路线（Linux 需自建，release 只有 Windows 二进制）。
- **FlashML-org/FreeToken**（v4，回滚路径）：MoE offload 推理运行时，提供 OpenAI 与 Anthropic 兼容 API。
  本项目在其 `0.1.3` 上验证 4090 单卡 NVFP4 路线，并携带三个未上游化的本地补丁。
- **官方 sglang**：v3 路线（已转为历史）。v0.5.20 是首个含 qwen4_exp 的正式 tag；官方硬件矩阵仍不含 4090/SM89。
- **Lsglang**（guqiong96）：v1/v2 的引擎与 lk_moe 来源；主线已转向 GLM-5.3 / DeepSeek-V4.1，release 均不含 Flash-Next。
- **lovedheart/sglang**：v1 的模型支持层，2026-08-27 后冻结。
- **TomPython/Qwen-3.8-Flash-Next**：双卡参考项目；本项目为单卡路线，两者互补。
- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)**：
  修复版 Jinja 模板，修复渲染错误、KV cache 失效、token 浪费与 agentic 死循环。
  **v4/v5 都用它**（注意 v5 的模板会被 `setup` 覆盖，见 [STRATA.md §7](docs/STRATA.md)）。

## License

Apache-2.0（补丁逻辑受上游 FlashAttention / FreeToken 许可证约束；原创脚本与文档 Apache-2.0）。
