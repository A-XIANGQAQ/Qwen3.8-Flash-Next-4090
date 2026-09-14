# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next-NVFP4 混合推理部署指南。

> [!WARNING]
> 社区实验复现，非官方支持方案。需要 SM89 专用的 FlashAttention PR #2751 补丁
> （由 Lsglang release 提供 prebuilt wheel）。请勿把结果表述为"SM89 已获上游官方支持"。

## 为什么有这个项目

| 方案 | 硬件 | 路线 | 覆盖 4090 单卡？ |
|---|---|---|---|
| 官方 sglang（Flash-Next 已于 2026-09-08 经 PR #37500 合入 main） | H200/B200/B300/GB300/MI350X/MI355X；NVFP4 仅 Blackwell（B200/B300/GB300/RTX PRO 6000/DGX Spark） | 纯 GPU | ❌ 无 sm89 路径（上游 PLE 表需瞬时 47.7GB GPU 显存，96GB 级卡才行） |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 RTX 4090 24GB + 1TiB | Docker 三层镜像 | 双卡专用 |
| **本项目** | **单 RTX 4090 48GB** | **conda 直跑 + lk_moe 混合推理**（现基于上游实现 + [PLE 补丁](patches/01_ple_cpu_alloc.patch)） | ✅ |

要点：**126GB NVFP4 权重驻内存（CPU 计算 experts）+ GPU 算 attention/PLE/常驻层**——让单卡 48GB 显存跑 176B 模型（6B active）。

## 已验证组合 (Verified Stack)

**当前推荐：上游 Flash-Next 路线**（2026-09-14 起，切换记录见 [升级文档](docs/UPGRADE_UPSTREAM_FLASHNEXT.md)）

| 层 | 固定身份 |
|---|---|
| 模型 | `RadixArk/Qwen3.8-Flash-Next-NVFP4`（ModelScope **master**，126GiB / 206 shards） |
| 引擎 | Lsglang `0.5.19-lkmoe` + 上游 Flash-Next（PR #37500）+ [PLE CPU 分配补丁](patches/01_ple_cpu_alloc.patch)；自建 wheel 用 `scripts/build_upstream_flashnext.sh` |
| 关键依赖 | lk_moe 2.4.1；**tilelang 必须锁 0.1.11**（0.1.12 与本环境 nvcc/CCCL 组合编译不兼容）；flashinfer-python 0.6.18 |
| FlashAttention | `flash_attn-2.8.4+pr2751`（cp312 prebuilt wheel，SM89 必需） |
| Chat Template | [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)（官方兼容 Flash-Next，原生 effort 别名） |
| 环境 | python 3.12 / torch 2.13.0+cu130 / CUDA 13 runtime（driver ≥ 580） |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1 |

<details>
<summary>旧组合：Lsglang 1.4.13 路线（原始方案，仍然可用、最省事）</summary>

| 层 | 固定身份 |
|---|---|
| 引擎 | Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内） + lk_moe 2.4.0 |

</details>

> 单卡 24GB 原版未验证（cache+权重 GPU 部分 ~30GB 超出）。**启动需 memlock 放开**（`ulimit -l unlimited`，PLE 表为 47.7GiB pinned 内存）——仓库启动脚本已内置。

## 实测摘要 (Measured Results)

测试主机：单 4090 48GB + 38 核 CPU + 247GB RAM。**客户端计时口径**。

| 指标 | 数值 |
|---|---|
| 8k prompt prefill（无缓存） | **~1880–2000 t/s**（4.1–4.4s） |
| 254K prefill（254k tokens） | **1784 t/s**（142.5s） |
| decode 512（短上下文，冷却后） | **~41.5 t/s**（随温度漂移 ±7%：59°C 时实测掉到 ~38.8） |
| decode @128K 上下文 | 36.5–38.7 t/s |
| 上下文 | 262144（256K）✅ |
| 模型加载 | ~254s（新栈实测） |
| GPU 显存 | ~41–43GB / 48GB |
| host RAM | 峰值 ~200GB（模型 126GB resident + 47.7GB pinned PLE 表） |

> **勘误**：本表早期版本给出的 8K prefill ~1195 t/s、254K ~1277 t/s 来自 2026-08-30 参数扫描期，后续复测（同一机器、客户端口径、服务空闲）修正为约 1900 t/s 与 1784 t/s。**测速时务必控制温度**（见 [TROUBLESHOOTING #11](docs/TROUBLESHOOTING.md)）。

## 快速开始 (Quick Start)

```bash
# 1. 环境安装（conda + wheel，含 SHA 校验）
bash scripts/install.sh

# 2. 模型下载（ModelScope，126GiB）
modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local_dir /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
python3 scripts/verify_model_index.py /path/to/.../RadixArk--Qwen3.8-Flash-Next-NVFP4 --expected-shards 206

# 3. 编辑 scripts/start_lsglang.sh（模型路径/模型名/SGLANG_BIN）后启动
bash scripts/start_lsglang.sh
# 等待 "The server is fired up and ready to roll!"（首次 ~5 分钟，含 126GB 加载）
```

两条路线任选：
- **v1（1.4.13，最省事）**：如上，直接装 release wheel
- **v2（上游路线，当前推荐）**：`bash scripts/build_upstream_flashnext.sh` 自建 wheel + 装依赖，详见 [升级文档](docs/UPGRADE_UPSTREAM_FLASHNEXT.md)

详细步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档 (Docs)

- [完整复现](docs/REPRODUCTION.md)
- [升级到上游 Flash-Next 实现（2026-09）](docs/UPGRADE_UPSTREAM_FLASHNEXT.md)
- [运维手册](docs/OPERATIONS.md)
- [排障（OOM/枚举/工具调用/新版 PLE 补丁等实测坑）](docs/TROUBLESHOOTING.md)
- [实测基线 + 全参数扫描结论](docs/BASELINE.md)

## 与相关项目的关系 (Relation to other projects)

- **Lsglang**（guqiong96）：本指南使用的引擎。v1 路线 = lovedheart 分支（Flash-Next 模型支持）+ lk_moe；**v2 路线（当前）= Lsglang `0.5.19-lkmoe` 合入上游 Flash-Next + PLE 补丁**。注意 Lsglang 1.4.14+ / 1.5.x 主线分别转向 GLM-5.3 与 DeepSeek-V4.1，Flash-Next 需按 [升级文档](docs/UPGRADE_UPSTREAM_FLASHNEXT.md) 自建。
- **lovedheart/sglang feat/qwen38-flash-next**：v1 的模型支持层（qwen4_exp：PLE/ngram/QSA/GDN），2026-08-27 后冻结。官方 sglang **main 已含 qwen4_exp**（PR #36497 经 #37500 于 2026-09-08 合并；首个包含它的 tag 将是 v0.5.20+）。
- **TomPython/Qwen-3.8-Flash-Next**：双卡参考项目；本项目的单卡路线与其互补（conda 直跑 vs Docker，wheel 免编译）。

## License

Apache-2.0（补丁逻辑受上游 FlashAttention 许可证约束，见 THIRD_PARTY_NOTICES 精神——原创脚本与文档 Apache-2.0）。
