# 升级：切换到上游 Flash-Next 实现（2026-09-14）

从 **Lsglang 1.4.13（lovedheart fork 快照）** 迁移到 **上游 sglang 的 Flash-Next 实现 + lk_moe** 的完整记录：为什么升、怎么升、三个必踩的坑、验证数据与回滚方式。

## 1. 为什么要升级

- 上游 `sgl-project/sglang` 已于 **2026-09-08 合并 Qwen3.8-Flash-Next 支持**（原 PR #36497 经 **PR #37500** 落地：91 文件 / +16407 行）
- 本项目原来的路线（Lsglang 1.4.13）= lovedheart/sglang `feat/qwen38-flash-next` 分支快照（2026-08-27 后冻结）+ lk_moe，属于临时方案
- 升级收益：代码回到主线、可持续跟进上游修复与新特性（PLE 文件后端、mamba 调优参数、MTP 等）
- **注意**：上游官方硬件矩阵**不含 4090/SM89**（NVFP4 官方仅 B200/B300/GB300/RTX PRO 6000/DGX Spark）。本项目能在 4090 跑，靠的是 lk_moe 混合推理 + 本文的 PLE 补丁——**这是社区路线，不是官方支持**。

## 2. 升级路线（自建构建）

```
Lsglang 0.5.19-lkmoe 分支（0.5.19 + lk_moe 集成，官方 universal 分支）
  └─ 合并上游 PR #37500（Flash-Next 模型/QSA 注意力/kernels）
      └─ 应用 patches/01_ple_cpu_alloc.patch（一行，见 §3.1）
          └─ pip wheel 构建 → 装进新 conda 环境
```

- 复现脚本：`scripts/build_upstream_flashnext.sh`
- 合并冲突很小：PR 的 91 个文件里与 lk_moe 集成重叠的只有 3 个（`quantization/unquant.py`、`vocab_parallel_embedding.py`、`utils/common.py`），冲突共 9 处，全部取**上游侧**（上游版本已包含 fork 里 cherry-pick 的同类修复）

## 3. 三个拦路问题（缺一不可）

### 3.1 PLE n-gram 表 GPU 瞬态分配 OOM（不改必炸）

**现象**：加载开始 ~1 秒即 `torch.OutOfMemoryError: Tried to allocate 47.69 GiB`。

**根因**（上游代码设计假设）：
```python
# Qwen4ExpPLELayer.__init__
self.ple_embedding = Qwen4ExpNGramEmbedding(...)        # ← 先在 GPU 构造整张表（FP8 也是 47.7GiB）
if config.ple_offload_embedding:
    ... = Qwen4ExpPinnedHostEmbedding(...)              # ← 再搬到 pinned 主机内存
```
96GB 卡（RTX PRO 6000）能撑过这个瞬态分配；48GB 卡没有路径能过。而且该 GPU 占位随后就被 `del`（权重实际由 loader 事后填进 pinned 副本）——**纯浪费**。显式 `--ple-offload-embedding` 无效（构造阶段就会炸）。

**修复**：`patches/01_ple_cpu_alloc.patch`——在构造处加 `is_lk_embedding=bool(config.ple_offload_embedding)`，借 lk_moe 集成在 `unquant.create_weights` 里的 CPU 分配分支，让表从一开始就在 CPU 上分配。

### 3.2 pinned 内存的 memlock 限制

pinned PLE 表需要 47.7GiB 锁页内存，而系统默认 `ulimit -l` 往往只有几 MB（本项目主机为 8192KB），**pinned 分配会直接失败**。

**修复**：启动脚本内置 `ulimit -l unlimited`（脚本因此需要 root，已做自动 sudo 提权）。

### 3.3 tilelang 版本（编译期不兼容）

环境里 nvcc 来自 `nvidia/cu13` pip 包（13.3），CCCL 头文件来自 `cuda-toolkit`（13.0.3）。tilelang **0.1.12** 在 CUDA graph 捕获期 JIT 编译时报：

```
error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

**修复**：锁 **tilelang==0.1.11**（该版本不引外部 CCCL，编译正常）。注意合并版 pyproject 声明的是 0.1.12，安装后需手动降级。

## 4. 验证矩阵（12/12 通过）

对旧栈基线（2026-09-10 复测口径）要求 ≥95%：

| 类别 | 项目 | 新栈实测 | 基线/阈值 | 结果 |
|---|---|---|---|---|
| 功能 | /v1/models、对话、工具调用（qwen3_coder）、thinking、effort 别名、/v1/messages | — | — | ✅ 全过 |
| 性能 | decode 512（短上下文，冷却后） | **41.5-41.9 t/s** | 41.6 / 39.5 | ✅ |
| 性能 | decode @128K 上下文 | **38.7 t/s** | 36.5 / 34.7 | ✅ 略优 |
| 性能 | prefill 8K / 64K | 1999 / 1937 t/s | 1961 / 1968 | ✅ |
| 性能 | 254K 冒烟（255k tokens） | 144.9s | 142.5s | ✅ |
| 性能 | 并发 2 请求 | 23.2×2 t/s | 23.6×2 | ✅ |

- 加载：254 秒，显存 41GB（与旧栈同量级）
- **测速注意**：首测曾出现 38.8 t/s 的"退步"，实为 GPU 59°C（基线测量时 44°C）——冷却到 41°C 后复测即回到 41.5-41.9。**对比 decode 务必控制温度**（见 TROUBLESHOOTING #11）。

## 5. 运维方式（与旧栈的差异）

| | 旧栈（1.4.13） | 新栈（1.5.0+flashnext） |
|---|---|---|
| 启动 | `bash scripts/start_lsglang.sh` | `bash scripts/start_lsglang_upstream.sh`（自动 sudo 提权 + `ulimit -l unlimited`，模型仍以普通用户跑） |
| 开机自启 | 无（本项目一贯手动启动） | 无（保持手动） |
| tilelang | 0.1.11 | **必须锁 0.1.11**（见 §3.3） |
| 模型名/端口/参数 | — | 完全不变（`Qwen3.8-Flash-Next`、8000、同一套启动参数） |

> 若日后 `pip install -U` 升级了环境，务必确认 **tilelang 未被升到 0.1.12+**。

## 6. 回滚

保留旧环境与旧脚本即可零成本回滚（5 分钟）：

```bash
pkill -f 'lsglang-next/bin/sglang serve'     # 停新栈
bash scripts/start_lsglang.sh.bak            # 旧脚本（指向 lsglang 1.4.13 环境）
```

建议升级时同时保留：旧 conda 环境（`lsglang`）、旧 wheel、旧启动脚本副本。
