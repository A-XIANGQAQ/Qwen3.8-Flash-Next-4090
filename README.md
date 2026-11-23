# Qwen3.8-Flash-Next on a single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next 部署与实测记录：让 111.33GB 的 MoE 权重
（6B active）跑在一张消费级卡上——**权重驻内存、CPU 算 experts、GPU 算 attention/PLE/常驻层**。

**当前组合 v6.3 = Strata 0.2.20（服务端）+ engine 0.1.66（预编译）+ unsloth UD-Q4_K_XL（4-bit，huihui v2 破限档 1）+ Sharp-naze v1.5 模板。**
0.1.66 在 sm_89 上有一个上游回归（首个请求崩 `layer 0 never rang`），本机用 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 规避；细节见 [STRATA-UPSTREAM-2026-11-23.md](docs/STRATA-UPSTREAM-2026-11-23.md)。
v5.3 时期（2026-10-25）用 **vLLM 官方 benchmark** 重测了 prefill / decode / 并发，原始数据在
[`bench/results/2026-10-25/`](bench/results/2026-10-25/)。

> [!WARNING]
> 社区实验复现，非官方支持方案：Strata 官方口径没有 sm_89。所有数字是本机实测，不是厂商承诺。

## 换模型后的实测（v5.9，2026-11-08，0.1.58 口径）

全部改用 unsloth **UD-Q4_K_XL**（111.33 GB）后，同机同协议：

| 项 | IQ3_S（v5.8） | UD-Q4_K_XL（v5.9） |
|---|---:|---:|
| prefill 6.2K | 5,177 t/s | 2,752 t/s |
| prefill 12.8K | 5,526 t/s | 4,479 t/s |
| 4 并发聚合 | 207.9 / 214.8 / 214.8 t/s | **186.2 / 184.2 / 188.4 t/s** |
| 单流 decode | 158–164 t/s（0.1.57；0.1.58 未重测） | 112.8（default）/ **133–135**（low） |
| 破限探针 | 8/8 不拒答 | **8/8 不拒答**（6 条对抗 6/6 + 2 条正常对照） |

模型大 33%，专家槽 17,283 → 10,902；单流下滑基本等于多搬的专家字节，4 并发只掉 13%。
质量侧：WorkBuddy 两道原失败题 0.1666→**0.8917**、0.0→**0.6923**。细节见
[STRATA-MODEL-2026-11-08.md](docs/STRATA-MODEL-2026-11-08.md)。

## 引擎升级实测（v6.0，2026-11-16，0.1.58 → 0.1.60）

同机同协议（prefill 每条约 6.2K / 12.8K token，破 prefix cache；并发 4 路 × 512 token × 3 轮）：

| 项 | 0.1.58 | 0.1.60 |
|---|---:|---:|
| prefill 6.2K | 2,770 t/s | **2,795 t/s**（+0.9%） |
| prefill 12.8K | 4,631 t/s | **4,783 t/s**（+3.3%） |
| prefill 20K / 30K | 未测 | **4,847 / 4,939 t/s** |
| 4 路聚合 | 102–105 t/s\* | **175.0 / 187.1 / 185.6 t/s** |
| 每路 decode 中位 | 41.8 / 43.5 / 44.2 t/s\* | 45.4 / 48.8 / 48.6 t/s |

\* 切前那组 conversation cache 里停着外部 140K 会话，每轮都要 park/restore，被拖慢；对照 11-08 的 0.1.58 **空缓存**口径（186–188 t/s、每路 54.9–57.3），0.1.60 的并发生效结果属于持平。prompt 侧的 +0.9%/+3.3% 是干净信号：0.1.60 把 Q4_K/Q5_K 专家的 prompt 从 llama.cpp MMQ 换到 Strata 自有内核，日志确认 **48/48 层**走融合入口。

## 0.2.15 实测（v6.1，2026-11-17，serve 层小版本）

0.2.15 只更新 app / Terminal / Control，**引擎与 0.1.60 逐字节相同**（官方仍从 0.2.14 release 下载）。我们直跑 `serve/server.py`，实际拿到的只有两处修复：`frontend.py` **#5979**（模型把声明的布尔型工具参数写成 Python 拼法 `True` / `False` 时正确解析）、`server.py` **#6055**（没传 `--shutdown-timeout` 时关闭期新请求也回 503 `server_draining`，不再被接进来切 0 token）。

同机同协议复测（只换 serve 源码树，引擎 0.1.60 未变）：

| 项 | 0.2.14（11-16） | 0.2.15（11-17） |
|---|---:|---:|
| prefill 6.2K | 2,795 t/s | **2,801 t/s** |
| prefill 12.8K | 4,783 t/s | **4,809 t/s** |
| 4 路聚合 | 175.0 / 187.1 / 185.6 t/s | **182.7 / 184.9 / 185.6 t/s** |

差异在 ±0.5% 噪声内，符合「引擎没变」的预期。切替 14:39:37 开始、14:40:12 就绪（35 s，页缓存命中），验收 **24/24**；模板仍是 Sharp-naze v1.5（34,386 字符）。

## 0.2.20 实测（v6.3，2026-11-23，sm_89 Q8-TC 规避）

0.2.20 的引擎 0.1.66 在 RTX 4090（sm_89）上默认配置会崩：`verify: layer 0 never rang (operation not supported on global/shared address space)`，根因是 `fused_gr.cu` 的 `q8_ld16()` 用 `__ldg()` 读 `__shared__`（上游 `#6742`/`#6700`/`#6745`，修复 PR `#6714` 仍 open）。本机在 config 的 `env` 里关掉两个 Q8 TC kernel（`STRATA_HC_Q8_TC=0`、`STRATA_HC_Q8_BF16TC=0`）后正常。模型、模板、30 个引擎参数未动。

同机同协议：

| 项 | 0.2.16 + 0.1.62 | 0.2.20 + 0.1.66 |
|---|---:|---:|
| prefill 6.2K | 2,760 t/s | **2,739 t/s**（−0.8%） |
| prefill 12.8K | 4,747 t/s | **4,986 t/s**（+5.0%） |
| 4 路聚合 | 174.3 / 188.5 / 185.9 t/s | **204.9 / 200.4 / 199.7 t/s** |
| 每路 decode 中位 | 45.5 / 49.4 / 48.2 t/s | 52.8 / 52.4 / 52.5 t/s |
| 单流 decode（low） | 133–135 t/s | **134.3 t/s** |

并发聚合 +6~18%（中位 +7~10%），单流持平，prefill 长 prompt +5%。切替第一次在 11:24:27 就绪但首个请求即崩，回滚后加规避变量，11:28:17 重切、**11:28:45 就绪（28 s）**，验收 **24/24**。细节见 [STRATA-UPSTREAM-2026-11-23.md](docs/STRATA-UPSTREAM-2026-11-23.md)。

## 0.2.16 实测（v6.2，2026-11-19，引擎换代 + 目录布局）

0.2.16 换的是**引擎**：0.1.62 在 NVIDIA/AMD/Intel Arc/Mac 上全部改跑 Strata 自有代码，不再用 ggml / llama.cpp 的矩阵内核；仓库顶层也重排成 `engine/`、`app/`、`terminal/`、`docs/`、`tools/`。模型、模板、30 个引擎参数一律没动。

同机同协议：

| 项 | 0.2.15 + 0.1.60 | 0.2.16 + 0.1.62 |
|---|---:|---:|
| prefill 6.2K | 2,801 t/s | **2,760 t/s**（−1.5%） |
| prefill 12.8K | 4,809 t/s | **4,747 t/s**（−1.3%） |
| 4 路聚合 | 182.7 / 184.9 / 185.6 t/s | **174.3 / 188.5 / 185.9 t/s** |
| 每路 decode 中位 | 47.3 / 48.2 / 48.3 t/s | 45.5 / 49.4 / 48.2 t/s |

1.3–1.5% 落在重复测量噪声带（同版本三次重复也能差 ±2%），并发持平，没有可见回退。切替 10:08:19 停、10:08:54 就绪（**28 s**），验收 **24/24**，工具调用冒烟 `finish_reason=tool_calls` 正常。细节见 [STRATA-UPSTREAM-2026-11-19.md](docs/STRATA-UPSTREAM-2026-11-19.md)。

## 基准实测（v5.3，2026-10-25，vLLM 口径）

工具：[vLLM `v0.7.3` 的 `benchmark_serving.py`](https://github.com/vllm-project/vllm)（vendored + 5 处 Strata 适配，
逐条见 [`bench/vllm_0.7.3/PATCHES.md`](bench/vllm_0.7.3/PATCHES.md)）。服务未重启，独占窗口，全部新 prompt
（`reused=0`）。完整矩阵、方法学与限制：[docs/BENCHMARK.md](docs/BENCHMARK.md)。

**单流 prefill**（输出 8 token、256K 档 4 token；TTFT 为客户端，t/s 为引擎 `prompt_ms`）：

| 输入 | 1K | 8K | 32K | 64K | 128K | **256K** |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 0.82 s | 2.36 s | 8.37 s | 16.25 s | 32.72 s | **66.24 s** |
| prefill | 1.29K t/s | 3.52K | 3.94K | 4.05K | 4.02K | **3.94K** |

> 8K 以上线性：边际 **3,942 t/s**（8K→256K）。1K 档 TTFT 里约 0.6s 是固定开销，别拿 1.29K 当硬件上限。

**单流 decode**（TPOT 换算；引擎侧交叉验证）：

| 上下文 | 1K | 32K | 128K | **256K** |
|---|---:|---:|---:|---:|
| 客户端 | 167.7 t/s | 167.4 | 177.3 | **168.2** |
| 引擎 | 167.5 t/s | 166.5 | 176.8 | **167.6** |

> decode 对上下文长度基本不敏感；256K 是**首次实测**，TTFT 66.2s + decode 168 t/s。

**并发**（`--batch 4`，4 个独立 slot，不是吞吐扩展）：

| 路数 | 每路 decode | 聚合 decode | 墙钟输出 | 相对单流 |
|---|---:|---:|---:|---:|
| 1 | 167 t/s | 167 t/s | 138 t/s | 1.00× |
| 2 | 93.9 | **188** | 148.5 | 1.12× |
| 4 | 50.4 | **202** | 176.5 | 1.28× |
| 4（8K prompt） | 32.7 | 131 | 65.3 | 0.78× |

> 结论：**1–2 路交互最优，4 路只增加会话数、不增加多少吞吐**；8K prompt 的 4 路反而低于单流。
> 高并发批量请另选引擎/多卡。

## 为什么有这个项目

| 方案 | 硬件 | 覆盖 4090 单卡？ |
|---|---|---|
| 官方 sglang（v0.5.20 起支持 Flash-Next） | H200/B200/B300/MI350X…；NVFP4 仅 Blackwell | ❌ 无 sm89 路径 |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 4090 24GB + 1TiB | 双卡专用 |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | 上游只发 Windows 二进制 | 需 Linux 自建/调参（本文即这份记录） |
| **本项目** | **单 4090 48GB** | ✅ |

## 当前组合（v6.3，2026-11-23 起）

| 层 | 固定身份 |
|---|---|
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF` → **UD-Q4_K_XL**（111.33 GB / 4 shards，4-bit；huihui-ai v2 消融 48 张量（档 1）） |
| 引擎 | [Niko1221/Strata](https://github.com/Niko1221/Strata) **0.2.20 服务端 + 0.1.66 预编译引擎**（CUDA 13，arch 含 sm_89，sha256 与官方 `SHA256SUMS.txt` 一致 `13c5bebe…5e059`） |
| 目录 | 上游 0.2.20 是 `terminal/serve`；本机平铺成 `Strata-0.2.20/{serve,tools,strata_control,strata_cli}`，启动 `cd Strata-0.2.20 && python ./serve/server.py` |
| 必需规避 | `env` 里 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0`（0.1.66 在 sm_89 上的 `layer 0 never rang` 回归，上游 `#6742`，修复 PR `#6714` 未并入） |
| 客户端模型名 | config 顶层 `aliases: ["Qwen3.8-Flash-Next", "qwen3.8-flash-next-q4kxl-ablit"]`——0.2.20 `#6621` 会拒未别名的 Strata 风格 id（DSH/Cherry 沿用旧名会 404），别名下两条路由都 200 |
| 并发 / 上下文 | `--batch 4` × `--max-context 262144`（每 slot 独立 session） |
| KV | **`k8v4` + `--kv-resident 32768`**（816 B/cell；KV 流式到宿主内存，4×256K 全放显存会 OOM） |
| 破限 | **huihui-ai v2 消融补丁档 1**（`ssm_out` / `attn_output`，48/48 命中，不动 routed experts 与 FFN）；强探针 8/8 不拒答，性暴力露骨特写保留软边界（已确认接受） |
| 图像 | 自带 vision 编码器（`mmproj-BF16.gguf` 0.91 GB，GPU 模式）；`--vram-reserve-mib 2560` |
| Chat 模板 | **Sharp-naze v1.5**（34,422 B / 34,386 字符，2026-11-09 起）：`reasoning_effort` 全档位 + 工具调用修复；froggeric v22.5 备份在 pack 的 `chat_template.jinja.bak-20261109-froggeric225`，`reasoning_effort` 通路仍由服务端解析 |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1；引擎报告 32,816 MiB 专家缓存 / 10,977 槽（空闲 2,651 MiB） |

## 快速开始（v6.3）

完整从零步骤（模型下载、pack、PLE 分片、模板、成功判据）见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。
最短路径（已有模型/pack 时）：

```bash
export https_proxy=http://127.0.0.1:7890

# 1. 服务端源码（0.2.20）+ 官方预编译引擎（0.1.66）
mkdir -p strata/Strata-0.2.20
# 源码 tarball 偶发截断；若 gzip 报 unexpected end of file，改用 GitHub tree API 逐文件取（见 docs/UPSTREAM 文档）
curl -L https://github.com/Niko1221/Strata/archive/refs/tags/v0.2.20.tar.gz \
  | tar -xz -C strata/Strata-0.2.20 --strip-components=1
# 0.2.20 的仓库顶层是 engine/ app/ terminal/ …；本机只把 terminal/{serve,tools,strata_control,strata_cli} 平铺到树根：
#   cd strata/Strata-0.2.20 && mv terminal/serve terminal/tools terminal/strata_control terminal/strata_cli .
gh release download v0.2.20 --repo Niko1221/Strata \
  -p 'strata-engine-0.1.66-linux-x86_64-cuda-cuda13.tar.gz' -D /tmp
mkdir -p strata/Strata-0.2.20/engine
tar -xzf /tmp/strata-engine-0.1.66-linux-x86_64-cuda-cuda13.tar.gz -C strata/Strata-0.2.20/engine
# sha256 期望 13c5bebeaf24574b02307857bce6abfd3f861f3682ca98dbe36c1c461fd5e059

# 2. 模板（Sharp-naze v1.5，34422 B / 34386 字符；froggeric v22.5 留作历史）
cp patches/sharp-naze_v1.5.jinja \
   <Strata-data>/packs/<pack>/tokenizer/chat_template.jinja

# 3. 启动（必须在你自己的终端里跑；页缓存冷 106 秒、命中 53 秒；0.1.62/0.1.66 实测就绪 28–35 秒）
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
```

⚠️ **sm_89 必须加规避**：`strata-unsloth-q4kxl-ablit-l1-0220.json` 的 `env` 里 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0`，否则首个请求就崩 `verify: layer 0 never rang`。
⚠️ **0.2.20 还要加 `aliases`**：`"aliases": ["Qwen3.8-Flash-Next", "qwen3.8-flash-next-q4kxl-ablit"]`，否则 DSH/Cherry 沿用旧模型 id 会 404 `model_not_found`（`#6621`）。
⚠️ **不要跑 `./setup.sh` / `update.sh`**：会重导原装模板（9,993 B）、重写 config（丢手调参数），
0.1.44+ 还可能因 #2902 删模型分片。升级只换「服务端源码 + 引擎目录」。
⚠️ 服务端 / 引擎 / vision 必须同代（0.2.20 ↔ 0.1.66；0.2.19 之后 engine 与 Control 共用版本号）；从 agent 的 bash 工具里起进程会被 systemd cgroup 连坐。
⚠️ 新建启动器后先 `chmod +x`——2026-11-16 的 7 分 24 秒停机就是漏了这一步（`systemd-run` 报「权限不够」）。

## 上游：0.2.3 → 0.2.20 的结论

**2026-11-23 已升级到 0.2.20 + 0.1.66**（见 [docs/STRATA-UPSTREAM-2026-11-23.md](docs/STRATA-UPSTREAM-2026-11-23.md)；上一版 [11-19](docs/STRATA-UPSTREAM-2026-11-19.md)、[11-17](docs/STRATA-UPSTREAM-2026-11-17.md)、[11-16](docs/STRATA-UPSTREAM-2026-11-16.md)、[11-08](docs/STRATA-UPSTREAM-2026-11-08.md)、[11-06](docs/STRATA-UPSTREAM-2026-11-06.md)、[11-03](docs/STRATA-UPSTREAM-2026-11-03.md)、[11-01](docs/STRATA-UPSTREAM-2026-11-01.md)、[10-28](docs/STRATA-UPSTREAM-2026-10-28.md)）：

- 0.2.5 / 0.2.6 对 Qwen 单卡零增量（0.2.6 的 Qwen 引擎与 0.2.5 逐字节相同）；0.2.7 是 0.2.2 之后第一个改到 sm_89 的版本。
- **0.2.9 是第一个"默认开性能开关"的版本**，两条都命中本机：`--batch 4` 触发的**并行请求栈**（9 个 `STRATA_*`，上游 8 并发 1.45×）与 Linux CUDA Qwen 的 **GDN 张量核 prompt 内核**。
- **0.2.10 性能零收益、纯修复**：DFlash2 与 4 并发互斥（Auto 在 `--batch ≥2` 时一律选 MTP），全卡切分引擎要两张 sm86/sm120 NVIDIA。拿到 `#4590` 惩罚窗口、`#4677` 停放会话、`#4663/#4700` Anthropic 路由工具、`#4838` 溢出措辞、`#4701/#4751` write 截断。
- **0.2.11 引入的 FP16 weight cache 仍未启用**（`FP16 weight cache 0 MiB`：要分配后仍留 3,072 MiB 空闲，本机只剩 ~2,900 MiB），所以拿不到上游 4 路 +5.7%。
- **0.2.12 速度持平、换的是工具调用与健壮性**：`#4828` 多工具 + 强制 `tool_choice` 从 0/24 修到 24/24，另有 `#5312`、`#5409`、`#5276`、`#4832`、`#4622`；引擎 `#5108` 长 CUDA prompt 行表越界 fault、`#5096/#4844` copy 顺序。头部 tuned DFlash2 不适用——它要求「专家全部驻留显卡」，我们 4 slots + 流式专家 ⇒ MTP，不会下载 drafter。
- 实测（同协议前后）：6.25K prompt **5,177 t/s**、12.7K prompt **5,526 t/s**、4 并发聚合 **215 t/s**，与 0.1.57 差在 ±1% 噪声内；停机 48 s、模板 243,742 chars 不变、`build_info "Strata 0.1.58"`、验收 24/24、`drafting = mtp`。
- **v5.9 换的是模型，不是引擎**：0.2.12 + 0.1.58 不动，权重换 unsloth UD-Q4_K_XL + huihui v2 消融（档 1）；质量升、速度降，见 [STRATA-MODEL-2026-11-08.md](docs/STRATA-MODEL-2026-11-08.md)。11-09 模板又换成 Sharp-naze v1.5（34,386 字符）。
- **0.2.13 没有单独升**（只把启动器默认值搬进引擎）；**0.2.14 一步到位**：Q4_K/Q5_K 专家的 prompt 从 llama.cpp MMQ 换到 Strata 自有融合内核（`-DSTRATA_MMQ=ON` 可回旧路径），日志 48/48 层；本机 prefill 6.2K +0.9%、12.8K **+3.3%**、20K 4,847 t/s、30K 4,939 t/s；并发与 0.1.58 空缓存口径持平。工具调用/思考泄漏（`#4751`/`#4077`/`#4949`）**一条没修**。见 [STRATA-UPSTREAM-2026-11-16.md](docs/STRATA-UPSTREAM-2026-11-16.md)。
- **0.2.15 只动 serve 层、引擎没变**：app/Terminal/Control 0.1.61 与我们无关；真正相关的是 `#5979` 布尔工具参数解析与 `#6055` 关闭期 503 排水。复测 prefill/并发在 ±0.5% 内，验收 24/24。见 [STRATA-UPSTREAM-2026-11-17.md](docs/STRATA-UPSTREAM-2026-11-17.md)。
- **0.2.16 = 引擎换代**：0.1.62 在 NVIDIA/AMD/Arc/Mac 上全部改跑 Strata 自有内核，不再用 ggml/llama.cpp 矩阵内核；`strata-vision` 从 53.8 MB 缩到 1.2 MB。本机 prefill 6.2K −1.5%、12.8K −1.3%（噪声内），4 路 174.3/188.5/185.9 持平；28 s 切替、24/24。仓库改 `terminal/` 布局（本机平铺）。见 [STRATA-UPSTREAM-2026-11-19.md](docs/STRATA-UPSTREAM-2026-11-19.md)。
- **0.2.20 = 并发变快，但 sm_89 要规避**：`EXP_MODE=8` 融合专家 + INT8 dense 让 4 路聚合 **+6~18%**（174–188 → 200–205 t/s）、12.8K prefill **+5.0%**，单流持平。代价是 0.1.66 的 `#6742` 回归：`q8_ld16()` 用 `__ldg()` 读 `__shared__`，sm_80+ 默认开的 Q8 TC kernel 首个请求就崩 `verify: layer 0 never rang`；本机用 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 绕开（修复 PR `#6714` 未并入）。另含 `#6431` 长 prompt NaN 修复。见 [STRATA-UPSTREAM-2026-11-23.md](docs/STRATA-UPSTREAM-2026-11-23.md)。
- 回滚：IQ3_S 模型、pack 与 0.2.2–0.2.11 旧树**已物理删除**，回滚要重下 84 GB + 重建 pack；生产入口是 `run_strata_q4l1_0220.sh`（`run_strata.sh` 已改为指向它），回 0.2.16 改指 `run_strata_q4l1_0216.sh`，回 0.2.15 改指 `run_strata_q4l1_0215.sh`，回 0.2.12 改指 `run_strata_q4l1.sh`；`run_strata_q4stock.sh` / `run_strata_q4kxl.sh` 保留作权重回退。

## 边界与未验证

- **8 路以上并发**、多日稳定性、质量（GSM8K 之类）未测；本页是单窗口 8 分钟。
- **视觉**在跑，但本次基准全是文本；图像编码器显存需求见 [docs/STRATA.md §7](docs/STRATA.md)。
- 256K 只测了单请求；4 路 × 256K 的并发未实测。
- 前缀缓存复用延迟未单独测（本页全用新 prompt）。
- **张量速率解锁**（gpu-unlock-ada）已在本卡实测：指令级 2×，但本栈无收益（8K prefill 0.983×）；
  单次冷启动、未做 matched 复测。见 [docs/GPU-UNLOCK.md](docs/GPU-UNLOCK.md)。
- 详细限制：[docs/BENCHMARK.md §7](docs/BENCHMARK.md)。

## 历史（压缩）

| 版本 | 运行时 / 权重 | 一句话 |
|---|---|---|
| **v6.3（当前）** | Strata 0.2.20 + 0.1.66 / **unsloth UD-Q4_K_XL + huihui v2 破限档 1** | `EXP_MODE=8` 融合专家 + INT8 dense：4 路聚合 174–188 → **200–205 t/s**（+6~18%）、12.8K prefill +5.0%、单流持平；0.1.66 在 sm_89 崩 `layer 0 never rang`（`#6742`），用 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 规避；28 s 切替、24/24 |
| v6.2 | Strata 0.2.16 + 0.1.62 / **unsloth UD-Q4_K_XL + huihui v2 破限档 1** | 引擎换代：0.1.62 改跑 Strata 自有内核、视觉读取器缩到 1.2 MB；仓库改 `terminal/` 布局（本机平铺）；prefill 2,760 / 4,747 t/s（−1.3~1.5%，噪声内）、4 路 174.3/188.5/185.9；28 s 切替、验收 24/24、工具调用正常 |
| v6.1 | Strata 0.2.15 + 0.1.60 / **unsloth UD-Q4_K_XL + huihui v2 破限档 1** | serve-only 小版本：引擎不变，拿 `#5979` 布尔工具参数 + `#6055` 关闭期 503；prefill 2,801 / 4,809 t/s、4 路 182.7/184.9/185.6，均在噪声内；验收 24/24 |
| v6.0 | Strata 0.2.14 + 0.1.60 / **unsloth UD-Q4_K_XL + huihui v2 破限档 1** | prompt 走 Strata 自有内核：12.8K prefill +3.3%、20K 4,847 t/s；并发持平；模板 Sharp-naze v1.5（34,386 字符）；停机 7m24s（启动器漏 +x）、验收 23/23 |
| v5.9 | Strata 0.2.12 + 0.1.58 / **unsloth UD-Q4_K_XL + huihui v2 破限** | 换模型：质量升（WorkBuddy 0.17→0.89、0→0.69），速度降（4 并发 215→186 t/s）；模板 11-09 换成 Sharp-naze v1.5；验收 31/31（旧脚本） |
| v5.8 | Strata 0.2.12 + 0.1.58 / DavidAU IQ3_S | 速度持平；7 条工具调用修复（`#4828` 0/24→24/24）+ `#5108`；停机 48 s、验收 24/24 |
| v5.7 | Strata 0.2.11 + 0.1.57 / DavidAU IQ3_S | W16 cache 在 48 GB 上未启用（0 MiB）；换 4 条 slot/工具流修复；停机 47 s、验收 24/24 |
| v5.3 | Strata 0.2.2 + 0.1.47 / DavidAU IQ3_S | 2026-10-25 vLLM 基准：decode 167–177 t/s、prefill ~3.9–4.05K t/s、256K 首测 |
| v5.2 | Strata 0.1.45 / DavidAU IQ3_S | 换模型：decode +10%、拒答 4.55%→0% |
| v5.1 | Strata 0.1.45 / UD-Q4_K_XL + ESP | 32K TTFT 9.50→6.99 s、短 decode 139.7→144.5 |
| v5.0 | Strata 0.1.43.1 / UD-Q4_K_XL + ESP | 切引擎：单流 59→118 t/s、TTFT 2.73→0.20 s |
| v4 | FreeToken 0.1.3 / RadixArk NVFP4 | 单流 59–64 t/s；运行时/权重已删，不可回滚 |
| v1–v3 | Lsglang / 官方 sglang | 只在历史里；运行时已删 |

逐次升级的 A/B、踩坑与已删除项见 [docs/HISTORY.md](docs/HISTORY.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [BENCHMARK.md](docs/BENCHMARK.md) | **最新实测**：prefill / decode / 并发、方法学、原始数据 |
| [OPERATIONS.md](docs/OPERATIONS.md) | 启停、观察、日志、sanity、同代升级 |
| [STRATA.md](docs/STRATA.md) | 为什么用 Strata、关键参数、KV/视觉/prefill 取舍 |
| [REPRODUCTION.md](docs/REPRODUCTION.md) | 从零复现（含成功判据） |
| [TEMPLATE.md](docs/TEMPLATE.md) | 合并模板：规则、词表、评测集、怎么改 |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 症状 → 判据 → 修复（现行 29–53 + 通用项） |
| [HISTORY.md](docs/HISTORY.md) | 压缩版版本表与已删除项 |
| [STRATA-MODEL-2026-11-08.md](docs/STRATA-MODEL-2026-11-08.md) | 换模型：UD-Q4_K_XL + huihui v2 消融、pack v4、速度/质量实测（模板沿革见文内 §5） |
| [STRATA-UPSTREAM-2026-11-23.md](docs/STRATA-UPSTREAM-2026-11-23.md) | **当前版本怎么来的**：0.2.20 + 0.1.66、0.1.66 在 sm_89 的 `layer 0 never rang` 回归与规避、EXP_MODE=8 并发收益、两次切替时间线、24/24、实测 |
| [STRATA-UPSTREAM-2026-11-19.md](docs/STRATA-UPSTREAM-2026-11-19.md) | 0.2.16 + 0.1.62 引擎换代、`terminal/` 布局与本机平铺、兼容性核对、28 s 切替、24/24、实测 |
| [STRATA-UPSTREAM-2026-11-17.md](docs/STRATA-UPSTREAM-2026-11-17.md) | 0.2.15 升级：引擎不变、serve 两处修复（#5979 / #6055）、30 秒切替、24/24 |
| [STRATA-UPSTREAM-2026-11-16.md](docs/STRATA-UPSTREAM-2026-11-16.md) | 0.2.14 升级：prompt 改走 Strata 内核、48/48 层、prefill/并发实测、7m24s 停机复盘、模板真相 |
| [STRATA-UPSTREAM-2026-11-08.md](docs/STRATA-UPSTREAM-2026-11-08.md) | 0.2.12 升级评估：速度持平、工具调用 7 条修复、官方 FP8 模板 commit 是 no-op |
| [STRATA-UPSTREAM-2026-11-06.md](docs/STRATA-UPSTREAM-2026-11-06.md) | 0.2.11 升级评估：W16 cache 为何在本机 0 MiB、4 条命中修复 |
| [STRATA-UPSTREAM-2026-10-25.md](docs/STRATA-UPSTREAM-2026-10-25.md) | 0.2.3/0.2.4 升级评估与阈值 |

## 与相关项目的关系

- **[Niko1221/Strata](https://github.com/Niko1221/Strata)**（当前引擎）：自有 C++/CUDA 引擎，复用 llama.cpp 的
  ggml/gguf-py/mtmd；自带 MoE 专家分层、conversation cache、MTP 投机解码。
- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)**：
  修复版 Jinja 模板；v5.3–v5.8 曾以 v22.5 为底合并 DavidAU 的模式系统，v5.9 前期用纯 v22.5，**2026-11-09 起生产是 Sharp-naze v1.5**（`patches/sharp-naze_v1.5.jinja`）。
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)**：本仓库的基准脚本来源（v0.7.3，vendored，
  Apache-2.0，5 处适配见 [PATCHES](bench/vllm_0.7.3/PATCHES.md)）。

## License

Apache-2.0（原创脚本与文档部分）。`bench/vllm_0.7.3/` 来自 vLLM（Apache-2.0），保留其 LICENSE；
`archive/patches/` 的旧补丁随其来源（sglang / FreeToken / Strata），仅存档。
