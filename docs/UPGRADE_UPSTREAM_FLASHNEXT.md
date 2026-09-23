# 升级：切换到上游 Flash-Next 实现

本页记录本项目从 Lsglang 分支快照迁移到**官方 sglang** 的两次升级：

| 版本 | 时间 | 路线 | 状态 |
|---|---|---|---|
| v3（当前）| **2026-09-23** | 官方 tag **v0.5.20** + lk_moe 补丁重放 + PLE 补丁 | ✅ 已上线 |
| v2 | 2026-09-14 | Lsglang `0.5.19-lkmoe` 分支合并 PR #37500 + PLE 补丁 | 历史（见 §5） |

---

## 1. 为什么升级到 v0.5.20

- **v0.5.20（2026-09-18）是首个正式收录 Qwen3.8-Flash-Next 的 release**（PR #37500 于 9/8 合入 main，v0.5.19 不含）
- 合入后 main 上又落了 4 个本项目直接受益的修复，v2 栈（PR 分支快照）里没有：
  - `#38851` QSA 分页 sparse-decode gather 内存安全（zero-fill scratch / int64 偏移 / gather 时反量化 FP8）
  - `#38855` sparse prefill 对 FP8 缓存前缀反量化
  - `#39446` compress gather 行数钳制（reland）
  - `#39474` 复用 alt_stream（修 decode overlap 路径反复建流的泄漏）
- **Lsglang v1.5.6 不可用**：其 `dsv4.1-lkmoe-sm80plus` 分支不含 `qwen4_exp.py`（主线转向 DeepSeek-V4.1），Flash-Next 无现成 wheel → 仍需自建
- 上游 PLE 修复 `#39928`（meta device 方案）于 **9/20** 合入 main，晚于 v0.5.20 → 本版本**仍需**本地 PLE 补丁；下次上游同步时自然收敛
- 官方硬件矩阵依旧**不含 4090/SM89**——本项目靠 lk_moe 混合推理 + PLE 补丁，属社区路线

## 2. 升级方法（v0.5.20 基线重放）

构建仓是**浅克隆**（与上游无共同祖先）→ 不能 `git merge`。改用「以官方快照为基线、重放补丁」：

```
git fetch --depth=1 https://github.com/sgl-project/sglang tag v0.5.20   # 只取快照
git checkout -b v0520-lkmoe v0.5.20
git apply patches/01_lk_moe_v0520.patch        # lk_moe 集成（18 文件，直接可打）
git apply patches/02_ple_cpu_alloc.patch       # PLE CPU 分配（48GB 卡必需）
git apply patches/03_resolved_files_v0520.patch  # 3 处人工裁决差异
python -m build --wheel --no-isolation         # SGLANG_BUILD_RUST_EXTS=none
```

全流程见 `scripts/build_upstream_flashnext.sh`。**三个补丁在干净 v0.5.20 上均已验证直接可打**（无 3-way、无冲突）。

`03_resolved_files_v0520.patch` 覆盖三处需要判断的差异（v2 路线是 merge 冲突）：
- `pyproject.toml`：取上游依赖版本（`sglang-kernel==0.4.7` 等）+ 追加 `lk_moe==2.4.1` + 包名 `lsglang` / 版本 `1.6.0+flashnext.v0520`
- `configs/qwen3_asr.py`：取上游版（`AutoConfig.register(..., exist_ok=True)`，比注释掉注册更干净）
- `quantization/modelopt_quant.py`：**保留 lk_moe 的 CPU 常驻层跳过**（`is_gpu_resident_layer` 判定）**并接入上游新增的 megamoe 分支**

## 3. 本次踩到并修掉的 4 个坑

| # | 现象 | 根因与修法 |
|---|---|---|
| 1 | 启动即 `Exception: sglang-kernel is installed with version 0.4.6.post1, which is less than the minimum required version 0.4.7` | v0.5.20 启动强校验。**只升这一个包**（`pip install sglang-kernel==0.4.7`，dry-run 确认无连带升级）；tilelang 等仍按原锁 |
| 2 | `RuntimeError: get_global_server_args() is retired` | v0.5.20 废弃该 API（值改由命名空间袋提供）。lk_moe 补丁里 1 处（`FusedMoE.get_max_num_group_batch_size`）→ 改 `get_schedule().chunked_prefill_size` |
| 3 | 从 root 外壳跑启动脚本时，子进程 `PermissionError: [Errno 13] Permission denied: '/root'` | Python multiprocessing spawn 的 worker 会 `os.chdir(父进程 cwd)`；root 外壳 cwd=`/root`，降权到普通用户的 worker 无法进入。**启动脚本加 `cd` 到公共目录**（已内置） |
| 4 | 构建产物异常 | ① 构建期需 `setuptools-rust`/`setuptools-scm`；② 必须 `SGLANG_BUILD_RUST_EXTS=none` 保持纯 Python 包（否则多出 4 个 rust `.so`）；③ 打包前清 `python/build`（残留会让 wheel 混入 1.4 万条 `build/` 垃圾，20MB→76MB） |

## 4. 验证结果（2026-09-23，独占窗口对照 v2 基线）

> ⚠️ **测性能前先确认窗口独占**：本机有外部客户端会不定时打 8000 口，实测能把 145K decode 从 39.4 压到 8.8 t/s。查日志中 `#running-req` 与 HTTP 行确认（见 TROUBLESHOOTING #18）。

| 类别 | 项目 | v3 实测 | v2 基线 | 结果 |
|---|---|---|---|---|
| 功能 | /v1/models、对话、工具调用、thinking、effort 别名、/v1/messages | — | — | ✅ 12/12 |
| 性能 | prefill 8K / 64K | **1968–1994 / 1887 t/s** | 1874 / 1923 | ✅ |
| 性能 | 254K 冒烟（255k tokens） | 149.2s | 142.5s | ✅ 96% |
| 性能 | decode 512（短） | **42.5 t/s** | 41.9 | ✅ |
| 性能 | decode @73K / @145K | **42.0 / 39.4 t/s**（ITL p50 24ms） | 40.1 / 36.5 | ✅ |
| 性能 | 并发聚合 1/2/4/8 | **42.5 / 47.7 / 72.3 / 108 t/s** | 39 / 44 / 65 / 97 | ✅ +8~11% |
| 正确性 | 98K 上下文针式检索（3 处不同深度） | 3/3 命中 | — | ✅ |
| 稳定性 | 20 轮 decode 显存曲线 + 20 分钟混合 soak | 零增长 | — | ✅ |
| 资源 | 加载 ~5.5 分钟、显存 41.7GB | 41.6GB | ✅ |

## 5. 历史：v2（2026-09-14，Lsglang 分支 + PR #37500）

<details>
<summary>展开 v2 路线记录（保留作参考）</summary>

**路线**：`Lsglang 0.5.19-lkmoe` 分支 merge 上游 PR #37500 → PLE 补丁 → 自建 wheel（`lsglang-1.5.0+flashnext.merge2/3`）。

**当时的三个拦路问题**（第 1、3 条至今仍适用）：

### 5.1 PLE n-gram 表 GPU 瞬态分配 OOM（不改必炸）

上游实现 `Qwen4ExpPLELayer.__init__` **先在 GPU 构造整张表（47.7GiB）再搬到 pinned 主机内存**——48GB 卡没有路径能过，且该 GPU 占位随后被 `del`（权重实际由 loader 填进 pinned 副本），纯浪费。显式 `--ple-offload-embedding` 无效（构造阶段就炸）。

**修复**：`patches/02_ple_cpu_alloc.patch` —— 构造处加 `is_lk_embedding=bool(config.ple_offload_embedding)`，借 lk_moe 集成在 `unquant.create_weights` 的 CPU 分配分支，让表从一开始就在 CPU 分配。
（上游 9/20 的 #39928 用 meta device 解决同一问题，晚于 v0.5.20。）

### 5.2 pinned 内存的 memlock 限制

pinned PLE 表需 47.7GiB 锁页内存，系统默认 `ulimit -l` 常仅 8MB → pinned 分配直接失败。**修复**：启动脚本内置 `ulimit -l unlimited`（因此需要 root，脚本自动 sudo 提权）。

### 5.3 tilelang 0.1.12 编译不兼容

环境里 nvcc 来自 `nvidia/cu13` pip 包（13.3）、CCCL 头来自 `cuda-toolkit`（13.0.3）。tilelang **0.1.12** 在 CUDA graph 捕获期 JIT 报 `CUDA compiler and CUDA toolkit headers are incompatible`。**修复**：锁 **tilelang==0.1.11**（合并版 pyproject 声明 0.1.12，安装后需手动降级）。

### 5.4 v2 验证（2026-09-14）

12/12 通过；decode 41.5–41.9 t/s、128K 38.7、prefill 8K/64K 1999/1937；加载 254s、显存 41GB。测速曾因 GPU 59°C 误判为退步，冷却后复原（见 TROUBLESHOOTING #11）。

</details>

## 6. 运维方式

| | v1（1.4.13） | v3（当前） |
|---|---|---|
| 启动 | `bash scripts/start_lsglang.sh` | `bash scripts/start_lsglang_upstream.sh`（自动 sudo 提权 + `ulimit -l unlimited` + cd + 就绪等待，模型仍以普通用户跑） |
| 开机自启 | 无 | 无（本项目一贯手动启动） |
| tilelang | 0.1.11 | **必须锁 0.1.11** |
| 模型名/端口/参数 | — | `Qwen3.8-Flash-Next`、8000、同一套参数（`--max-running-requests 8`，并发甜点见 BASELINE） |

> 若日后 `pip install -U` 升级环境，务必确认 **tilelang 未被升到 0.1.12+**、**sglang-kernel ≥ 0.4.7**。

## 7. 回滚

```bash
pkill -f 'sglang serve'                       # 停当前栈
pip install --no-deps /path/to/旧wheel.whl     # 换回旧 wheel（记录版本号）
bash scripts/start_lsglang_upstream.sh
```

建议保留：旧 wheel 文件 + `pip freeze` 快照 + 旧启动脚本副本。本项目实测回滚耗时约 5 分钟（含模型重载）。
