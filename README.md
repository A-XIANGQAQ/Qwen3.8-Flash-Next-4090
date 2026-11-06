# Qwen3.8-Flash-Next on a single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next 部署与实测记录：让 83.86GB 的 MoE 权重
（6B active）跑在一张消费级卡上——**权重驻内存、CPU 算 experts、GPU 算 attention/PLE/常驻层**。

**当前组合 v5.7 = Strata 0.2.11（服务端）+ engine 0.1.57（预编译）+ DavidAU abliterated IQ3_S（3.5 bpw）+ 合并模板 v3。**
2026-10-25 用 **vLLM 官方 benchmark** 重测了 prefill / decode / 并发，原始数据在
[`bench/results/2026-10-25/`](bench/results/2026-10-25/)。

> [!WARNING]
> 社区实验复现，非官方支持方案。不要表述为「SM89 已获上游官方支持」。Strata 走投机解码，
> 输出与 v4（FreeToken）**非逐位一致**；本仓库只报告实测速度与拒答率，不宣称「质量等价」。

## 最新实测（2026-10-25）

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

## 当前组合（v5.3，2026-10-22 起）

| 层 | 固定身份 |
|---|---|
| 模型 | `DavidAU/Qwen3.8-Flash-Next-Abliterated-…-GSQ-RCO-GGUF` → **IQ3_S**（83.86 GB / 2 shards，3.5 bpw，自带 abliteration） |
| 引擎 | [Niko1221/Strata](https://github.com/Niko1221/Strata) **0.2.11 服务端 + 0.1.57 预编译引擎**（CUDA 13，arch 含 sm_89，sha256 与签名 feed 一致 `cc58226b…228b`） |
| 并发 / 上下文 | `--batch 4` × `--max-context 262144`（每 slot 独立 session） |
| KV | **`k8v4` + `--kv-resident 32768`**（816 B/cell；KV 流式到宿主内存，4×256K 全放显存会 OOM） |
| 破限 | **模型自带**（强探针 0/14、温和 0/22 拒答）；不需要 ESP 控制向量 |
| 图像 | 自带 vision 编码器（`mmproj-…-BF16.gguf` 0.91 GB，GPU 模式）；`--vram-reserve-mib 2560` |
| Chat 模板 | **合并模板 249,545 B**：froggeric v22.5 + 29 段模式正文 + 8 个 xhigh 组合键 + 自动选模式 v3，**服务端代码一行没改** |
| 模式选择 | 自动（分数 ≥2 且领先第二名才选）+ `{REASON:xxx}` 手动覆盖 |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1；引擎报告 33,605 MiB 专家缓存 / 17,284 槽（空闲 2,483 MiB） |

## 快速开始（v5.7）

完整从零步骤（模型下载、pack、PLE 分片、模板、成功判据）见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。
最短路径（已有模型/pack 时）：

```bash
export https_proxy=http://127.0.0.1:7890

# 1. 服务端源码 + 官方预编译引擎（0.2.11 / 0.1.57）
mkdir -p strata/Strata-0.2.11
curl -L https://github.com/Niko1221/Strata/archive/refs/tags/v0.2.11.tar.gz \
  | tar -xz -C strata/Strata-0.2.11 --strip-components=1
gh release download v0.2.11 --repo Niko1221/Strata \
  -p 'strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz' -D /tmp
mkdir -p strata/Strata-0.2.11/engine
tar -xzf /tmp/strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz -C strata/Strata-0.2.11/engine
# sha256 期望 cc58226bffc12216a5a2e6019dfdbf4896042500df132cc3fa3c72a2388a228b

# 2. 合并模板（249545 B）
cp patches/08_qwen38_froggeric_v225_with_dau_modes.jinja \
   <Strata-data>/packs/<pack>/tokenizer/chat_template.jinja

# 3. 启动（必须在你自己的终端里跑；就绪约 40 秒）
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
```

⚠️ **不要跑 `./setup.sh` / `update.sh`**：会重导原装模板（226,745 B）、重写 config（丢手调参数），
0.1.44+ 还可能因 #2902 删模型分片。升级只换「服务端源码 + 引擎目录」。
⚠️ 服务端 / 引擎 / vision 必须同代（0.2.11 ↔ 0.1.57）；从 agent 的 bash 工具里起进程会被 systemd cgroup 连坐。

## 上游：0.2.3 → 0.2.11 的结论

**2026-11-06 已升级到 0.2.11 + 0.1.57**（见 [docs/STRATA-UPSTREAM-2026-11-06.md](docs/STRATA-UPSTREAM-2026-11-06.md)，前几篇 [11-03](docs/STRATA-UPSTREAM-2026-11-03.md)、[11-01](docs/STRATA-UPSTREAM-2026-11-01.md)、[10-28](docs/STRATA-UPSTREAM-2026-10-28.md)）：

- 0.2.5 / 0.2.6 对 Qwen 单卡零增量（0.2.6 的 Qwen 引擎与 0.2.5 逐字节相同）；0.2.7 是 0.2.2 之后第一个改到 sm_89 的版本。
- **0.2.9 是第一个"默认开性能开关"的版本**，两条都命中本机：`--batch 4` 触发的**并行请求栈**（9 个 `STRATA_*`，上游 8 并发 1.45×）与 Linux CUDA Qwen 的 **GDN 张量核 prompt 内核**。
- **0.2.10 性能零收益、纯修复**：DFlash2 与 4 并发互斥（Auto 在 `--batch ≥2` 时一律选 MTP），全卡切分引擎要两张 sm86/sm120 NVIDIA。拿到 `#4590` 惩罚窗口、`#4677` 停放会话、`#4663/#4700` Anthropic 路由工具、`#4838` 溢出措辞、`#4701/#4751` write 截断。
- **0.2.11 的 FP16 weight cache 在我们这没启用**（`FP16 weight cache 0 MiB`：要分配后仍留 3,072 MiB 空闲，本机只剩 ~2,900 MiB），所以拿不到上游 4 路 +5.7%；拿到的是 `#2518` 4 会话 slot 串台、`#5040/#4701` 流式工具参数尾段、`#4451` 重启后旧请求挂新 slot / 控制行被抢、`#4271` 断流占用 slot 这四条修复。
- 实测（同协议前后）：6.25K prompt **5,060 t/s**、12.7K prompt **5,326 t/s**、4 并发聚合 **212 t/s**，与 0.1.56 差在 ±2% 噪声内；停机 47 s、模板 243,742 chars 不变、`build_info "Strata 0.1.57"`、验收 24/24、`drafting = mtp`。
- 回滚：`run_strata_0210.sh`（0.2.10 + 0.1.56）、`run_strata_029.sh`（0.2.9 + 0.1.55）、`run_strata_027.sh`（0.2.7 + 0.1.52）都保留；只关 0.2.9 的新默认值可用 `STRATA_CKPT_POOL=0` / `STRATA_GDN_TC=0`。

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
| **v5.7（当前）** | Strata 0.2.11 + 0.1.57 / DavidAU IQ3_S | W16 cache 在 48 GB 上未启用（0 MiB）；换 4 条 slot/工具流修复；验收 24/24 |
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
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 症状 → 判据 → 修复（现行 29–50 + 通用项） |
| [HISTORY.md](docs/HISTORY.md) | 压缩版版本表与已删除项 |
| [STRATA-UPSTREAM-2026-11-06.md](docs/STRATA-UPSTREAM-2026-11-06.md) | 0.2.11 升级评估：W16 cache 为何在本机 0 MiB、4 条命中修复 |
| [STRATA-UPSTREAM-2026-10-25.md](docs/STRATA-UPSTREAM-2026-10-25.md) | 0.2.3/0.2.4 升级评估与阈值 |

## 与相关项目的关系

- **[Niko1221/Strata](https://github.com/Niko1221/Strata)**（当前引擎）：自有 C++/CUDA 引擎，复用 llama.cpp 的
  ggml/gguf-py/mtmd；自带 MoE 专家分层、conversation cache、MTP 投机解码。
- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)**：
  修复版 Jinja 模板；本项目以 v22.5 为底合并模型自带模式系统。
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)**：本仓库的基准脚本来源（v0.7.3，vendored，
  Apache-2.0，5 处适配见 [PATCHES](bench/vllm_0.7.3/PATCHES.md)）。

## License

Apache-2.0（原创脚本与文档部分）。`bench/vllm_0.7.3/` 来自 vLLM（Apache-2.0），保留其 LICENSE；
`archive/patches/` 的旧补丁随其来源（sglang / FreeToken / Strata），仅存档。
