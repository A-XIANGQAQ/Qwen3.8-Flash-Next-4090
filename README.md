# Qwen3.8-Flash-Next on single RTX 4090 48GB

一份面向**单卡 RTX 4090（SM89，48GB）**的 Qwen3.8-Flash-Next-NVFP4 混合推理部署指南。

> [!WARNING]
> 社区实验复现，非官方支持方案。需要 SM89 专用的 FlashAttention PR #2751 补丁
> （由 Lsglang release 提供 prebuilt wheel）。请勿把结果表述为"SM89 已获上游官方支持"。

## 为什么有这个项目

| 方案 | 硬件 | 路线 | 覆盖 4090 单卡？ |
|---|---|---|---|
| 官方 sglang（PR #36497，未合并） | H200/B200/B300/GB300/MI350X | 纯 GPU，NVFP4 仅 Blackwell | ❌ 无 sm89 路径 |
| [TomPython/Qwen-3.8-Flash-Next](https://github.com/TomPython/Qwen-3.8-Flash-Next) | 双 RTX 4090 24GB + 1TiB | Docker 三层镜像 | 双卡专用 |
| **本项目** | **单 RTX 4090 48GB** | **conda 直跑 + lk_moe 混合推理** | ✅ |

要点：**126GB NVFP4 权重驻内存（CPU 计算 experts）+ GPU 算 attention/PLE/常驻层**——让单卡 48GB 显存跑 176B 模型（6B active）。

## 已验证组合 (Verified Stack)

| 层 | 固定身份 |
|---|---|
| 模型 | `RadixArk/Qwen3.8-Flash-Next-NVFP4`（ModelScope **master**，126GiB / 206 shards） |
| 引擎 | Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内） + lk_moe 2.4.0 |
| FlashAttention | `flash_attn-2.8.4+pr2751`（cp312 prebuilt wheel，SM89 必需） |
| 环境 | python 3.12 / torch 2.13.0+cu130 / CUDA 13 runtime（driver ≥ 580） |
| GPU | 单 RTX 4090 **48GB** / sm89 / TP1 |

> 单卡 24GB 原版未验证（cache+权重 GPU 部分 ~30GB 超出）。

## 实测摘要 (Measured Results)

测试主机：单 4090 48GB + 38 核 CPU + 247GB RAM。**客户端计时口径**。

| 指标 | 数值 |
|---|---|
| 8k prompt prefill（无缓存） | **~1195 t/s**（6.7s） |
| 256K prefill（254k tokens） | **1277 t/s**（199s，32 chunk 流水线不拖累） |
| decode 512 | **36~40 t/s**（随 CPU 温度漂移 ±7%） |
| 上下文 | 262144（256K）✅ |
| GPU 显存 | ~43GB / 48GB |
| host RAM | 峰值 ~200GB（模型 126GB resident） |

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

详细步骤见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档 (Docs)

- [完整复现](docs/REPRODUCTION.md)
- [运维手册](docs/OPERATIONS.md)
- [排障（OOM/枚举/工具调用等 10 个实测坑）](docs/TROUBLESHOOTING.md)
- [实测基线 + 全参数扫描结论](docs/BASELINE.md)

## 与相关项目的关系 (Relation to other projects)

- **Lsglang**（guqiong96）：本指南使用的引擎 = lovedheart 分支（Flash-Next 模型支持）+ lk_moe（CPU-GPU 混合推理）。Lsglang 1.4.14+ 转向 GLM-5.3 并移除 Flash-Next，**1.4.13 是 Flash-Next 的正确版本**。
- **lovedheart/sglang feat/qwen38-flash-next**：模型支持层（qwen4_exp：PLE/ngram/QSA/GDN）。官方 sglang main 尚无 qwen4_exp（PR #36497 未合并）。
- **TomPython/Qwen-3.8-Flash-Next**：双卡参考项目；本项目的单卡路线与其互补（conda 直跑 vs Docker，wheel 免编译）。

## License

Apache-2.0（补丁逻辑受上游 FlashAttention 许可证约束，见 THIRD_PARTY_NOTICES 精神——原创脚本与文档 Apache-2.0）。
