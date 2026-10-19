# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next 部署记录：让 111 GB 的 MoE 权重（6B active）
跑在一张消费级卡上——**权重驻内存、CPU 算 experts、GPU 算 attention/PLE/常驻层**。

**当前组合 v5.2 = Strata 0.1.45 + DavidAU abliterated IQ3_S + 合并 chat 模板**。
旧路线（v4 FreeToken / v3 sglang / v2·v1 Lsglang）的运行时**已于 2026-10-18 全部删除**，
只保留记录与存档脚本，见 [docs/README.md](docs/README.md)。

> [!WARNING]
> 社区实验复现，非官方支持方案。请勿表述为"SM89 已获上游官方支持"。
> Strata 仍是新引擎（3 周龄、790 open issues），走投机解码 ⇒ 输出与 v4 **非逐位一致**；
> 本仓库只报告实测的速度与拒答率，不宣称"质量等价"。

## 为什么有这个项目

| 方案 | 硬件 | 覆盖 4090 单卡？ |
|---|---|---|
| 官方 sglang（v0.5.20 起支持 Flash-Next） | H200/B200/B300/MI350X…；NVFP4 仅 Blackwell | ❌ 无 sm89 路径 |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 4090 24GB + 1TiB | 双卡专用 |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | 上游只发 Windows 二进制 | 需 Linux 自建与调参（本文即这份记录） |
| [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | 上游未给出本模型组合 | 需本地补丁（v4，已退役） |
| **本项目** | **单 4090 48GB** | ✅ |

要点：v4 与 v5 跑**同一基座模型**，位宽都是 4-bit（刻意对齐，否则质量差异会混进速度结论），
差别只在**引擎 + 权重格式**：v5 = GGUF IQ3_S + Strata，v4 = NVFP4 + FreeToken。

## 当前组合（v5.2，2026-10-18 起）

| 层 | 固定身份 |
|---|---|
| 模型 | `DavidAU/Qwen3.8-Flash-Next-Abliterated-…-GGUF` → **IQ3_S**（83.86 GB / 2 shards，3.5 bpw，**自带 abliteration**） |
| 引擎 | **[Niko1221/Strata](https://github.com/Niko1221/Strata) `0.1.45`**（Linux 自建；CUDA 13.2 / sm89；产出 `strata` + `strata-vision` + `libstrata_kq.so`） |
| 并发 / 上下文 | `--batch 4` × `--max-context 262144`（每 slot 独立 session） |
| KV | **`k8v4` + `--kv-resident 32768`**（816 B/cell；KV 流式到宿主内存——4×256K 全放显存会 OOM） |
| 破限 | **模型自带**（强探针 0/14、温和 0/22 拒答）；原 UD-Q4_K_XL 路线靠引擎 ESP 控制向量，现已不需要 |
| 图像 | 自带 vision 编码器（`mmproj-…-BF16.gguf` 0.91 GB，GPU 模式）；图像 token 上限 1024，约占 1.4 GB 显存 |
| Chat 模板 | **合并模板 248,855 字节**：froggeric v22.5 + 模型自带 29 段模式正文 + 自动选模式 v2，**服务端代码一行没改**，见 [docs/TEMPLATE.md](docs/TEMPLATE.md) |
| 模式选择 | 自动（分数 ≥2 且领先第二名才选，否则不加任何提示）+ `{REASON:xxx}` 手动覆盖 |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1 |

## 实测摘要

测试主机：单 4090 48GB + 38 核 CPU + 247 GB RAM，同机、客户端计时、独占窗口。

**v5（Strata）vs v4（FreeToken）**，2026-10-13，同一模型、同一 prompt（2,336 token）：

| 指标 | v4（FreeToken 0.1.3） | **v5.0（Strata 0.1.43.1）** |
|---|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s**（2.0×） |
| 单流 TTFT | 2.73 s | **0.20 s**（14×） |
| 4 路聚合 / 每路 | 117.1 / **45.8** | **137** / 36.9 |
| 4 路端到端（TTFT + 512 token） | 17.1 s | **14.7 s** |
| 并发扩展比（1→4） | **2.55×** | 1.16× |

> 📌 这张表是 0.1.43.1 时代的历史基线。当前引擎 0.1.45 的两臂实测见 [UPGRADE §0b](docs/UPGRADE.md)：
> 短上下文 decode 139.7 → **144.5 t/s**、32K TTFT 9.50 → **6.99 s**、100K TTFT 24.78 → **21.82 s**。
> 换模型的实测见 [UPGRADE §0c](docs/UPGRADE.md)：decode +10%、1K TTFT −49%、32K/100K decode +34%/+31%。

**长上下文**（v5）：32K TTFT 7.0 s / 103 t/s；64K 13.7 s / 107；110K 20.6 s / 100；**143K 24.1 s / 100 t/s**。
143K 仍 100 t/s 是对照 v4 @145K 的 1.7×；**256K 未实测**。

**取舍**：v5 单流快一倍、首字快 14 倍，1–2 路全面占优；代价是高并发扩展性弱（1→4 只涨 1.16×），
8 路以上密集并发未实测。完整数据与口径说明见 [docs/BASELINE.md](docs/BASELINE.md)。

## 快速开始（v5）

```bash
# 1. 取源码（Linux；上游 release 只有 Windows 二进制）
gh repo clone Niko1221/Strata && cd Strata

# 2. 构建 + 下模型 + 生成配置（--gguf-dir 指向已有的 DavidAU IQ3_S 分片）
STRATA_UV=1 ./setup.sh --setup --family unsloth --model IQ3_S \
  --gguf-dir /path/to/DavidAU-abliterated-IQ3_S --source modelscope \
  --vision yes --context 262144 --kv k8v4 --parallel 4 \
  --host 127.0.0.1 --port 8000 --yes --no-start

# 3. 装合并模板（必须在 setup 之后——setup 每次都会从 GGUF 重导原装模板）
cp qwen4090-repo/patches/08_qwen38_froggeric_v225_with_dau_modes.jinja \
   Strata-data/packs/<pack>/tokenizer/chat_template.jinja
wc -c Strata-data/packs/<pack>/tokenizer/chat_template.jinja    # 期望 248855

# 4. 启动（就绪标志：日志出现 "ready:"，约 40–50 秒）
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
```

⚠️ 两个必须知道的点：

1. **不要用 `./setup.sh` 无参直接起服务**——它每次重导**原装**模板，会把模式系统冲掉（判据 `wc -c` = 226745）。
2. **必须在自己的终端里启动**，不要从 agent/CI 的工具调用里起：进程会落在 `dsh-subprocess-*.scope` 的 cgroup 里，
   dsh-web 重启时被整组 SIGTERM。细节见 [OPERATIONS §1](docs/OPERATIONS.md)。

`patches/07`（ESP 白名单）只在**回滚到 UD-Q4_K_XL** 或重跑 `setup.sh` 时需要，当前模型自带 abliteration 用不上。

## 文档

入口：[docs/README.md](docs/README.md)（按"现在 / 历史"分类）

| 文档 | 内容 |
|---|---|
| [TEMPLATE.md](docs/TEMPLATE.md) | 合并模板与自动选模式：规则、词表、评测集、怎么改 |
| [OPERATIONS.md](docs/OPERATIONS.md) | 启停、观察、日志、sanity、引擎互切/升级 |
| [STRATA.md](docs/STRATA.md) | 为什么用 Strata、Linux 自建、关键参数、破限、两个引擎不能共存 |
| [REPRODUCTION.md](docs/REPRODUCTION.md) | 从零复现（含成功判据） |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 症状 → 判据 → 修复（编号 1–50） |
| [BASELINE.md](docs/BASELINE.md) | 主结果、调优扫描、第三方交叉验证 |
| [UPGRADE.md](docs/UPGRADE.md) | 升级实测（§0–0c 现行）与 v1–v4 演进史 |
| [patches/](patches/README.md) · [scripts/](scripts/README.md) | 补丁与脚本索引（含"现行 / 历史"标注） |

## 与相关项目的关系

- **[Niko1221/Strata](https://github.com/Niko1221/Strata)**（当前引擎）：自有 C++/CUDA 引擎，复用 llama.cpp 的
  ggml/gguf-py/mtmd；自带 MoE 专家 offload、conversation cache、ESP 控制向量。本项目验证其 4090 单卡路线。
- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)**：
  修复版 Jinja 模板（渲染错误、KV cache 失效、token 浪费、agentic 死循环）。本项目以它的 v22.5 为底，
  把模型自带的模式系统并了进去（[TEMPLATE.md](docs/TEMPLATE.md)）。
- **FlashML-org/FreeToken**（v4，已退役）、**官方 sglang**（v3）、**Lsglang**（v1/v2）、
  **TomPython/Qwen-3.8-Flash-Next**（双卡参考）：历史与对照见 [UPGRADE.md](docs/UPGRADE.md)。

## License

Apache-2.0（原创脚本与文档部分）。补丁与代码片段随其来源：`patches/01–03` 受上游 FlashAttention / sglang
许可证约束，`patches/04–06` 受 FreeToken 许可证约束，`patches/07` 改的是 Strata 的 `setup.py`，
随 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的许可证。
