# 升级记录 (Upgrade Log)

本页记录本项目的引擎演进。当前为 **v4（FreeToken）**，v1–v3（Lsglang / 官方 sglang）全部折叠保留作参考。

| 版本 | 时间 | 路线 | 状态 |
|---|---|---|---|
| v4（当前）| **2026-09-24** | **FreeToken 0.1.3** + 三个本地补丁 | ✅ 已上线 |
| v3 | 2026-09-23 | 官方 sglang tag **v0.5.20** + lk_moe 补丁重放 + PLE 补丁 | 历史（见 §5） |
| v2 | 2026-09-14 | Lsglang `0.5.19-lkmoe` 分支合并 PR #37500 + PLE 补丁 | 历史（见 §6） |
| v1 | 原始 | Lsglang 1.4.13 + lk_moe 2.4.0 | 历史（见 §6） |

> 本文件名原为 `UPGRADE_UPSTREAM_FLASHNEXT.md`（标题写死 sglang 路线）。切到 v4 后改名为 `UPGRADE.md`，
> 旧链接会 404，请更新。

---

## 1. 为什么切到 FreeToken

v3（sglang + lk_moe）的瓶颈是**架构性**的：experts 在 CPU 上算，decode 速度受 CPU 算力与单步同步延迟限制，
长上下文 decode 卡在 ~42 t/s，加载要 ~5.5 分钟。FreeToken 恰好是为「权重驻内存 + CPU 算 experts」
这一形态设计的 MoE offload 运行时，同机实测：

- **长上下文 decode +49~53%**（73K：42.0 → 62.6~63.6 t/s；145K：39.4 → 60.2 t/s）
- **254K prefill +22%**（1710 → 2081 t/s）
- **加载时间 ~8×**（~5.5 分钟 → **~39 秒**）
- 原生提供 **Anthropic 兼容 API**（`/v1/messages`），sglang 路线没有

**代价也是明确的**（详见 §4 取舍表）：8K prefill 明显变差且方差极大、短请求有 ~3.4s 固定开销、
≥4 并发不再扩展、多流冻结问题更严重。**这是一次按场景的取舍，不是全面升级。**

## 2. 迁移方法

```bash
# 1. 独立 venv（uv 管，不碰既有的 conda 环境）
uv venv ~/ft-venv --python 3.12
uv pip install --python ~/ft-venv "freetoken[accel]"

# 2. 打三个本地补丁（未上游化；install 脚本幂等，升级后重跑即可）
bash scripts/install_freetoken_patches.sh ~/ft-venv/lib/python3.12/site-packages

# 3. 启动（脚本自带 sudo 提权 + memlock + 补丁自检 + 就绪等待）
bash scripts/start_freetoken.sh
```

`accel` extra 会拉入 flashinfer-python[cu13] 0.6.18 与 sglang-kernel 0.4.5（**注意比 v3 要求的 0.4.7 低**——
两者服务于不同的引擎路径，不冲突）。版本硬约束：`torch>=2.11,<2.12`、`transformers>=5.16,<5.17`、`triton==3.6.0`。

**三个补丁各解决一个问题**：

| 补丁 | 解决什么 | 不打的后果 |
|---|---|---|
| [04 FT_DIST_PORT](../patches/04_ft_dist_port.patch) | 内部 rendezvous 默认监听「API 端口+1」，会撞 nginx 的 8001 | 无法维持 8000/8001 的既有端口布局 |
| [05 qwen3.8 workload](../patches/05_ft_workload_qwen38.patch) | 注册本模型的专家几何（2.64MB，`ft bench bw` 默认按 7.61MB 测） | `ft bench bw --model qwen3.8-flash-next` 跑不了 |
| [06 decode-interleave](../patches/06_ft_decode_interleave.patch) | prefill 无条件优先导致长 prompt 整块占住 GPU | 正在解码的流最长冻结 17.7s「不吐字」 |

## 3. 本次踩到的坑

| # | 现象 | 根因与修法 |
|---|---|---|
| 1 | 就绪判断失效：`/health` 在模型加载完成前就返回 200 | 上游 **#537**。就绪判据改用日志行 `ready to serve`（加载期还会看到 `POST /v1/chat/completions` 被拒） |
| 2 | 首个 >8K 的 prompt 直接 `HTTP 400 Bad Request` | 默认 `auto` 只规划 8256 token 的 KV。必须显式 `--num-tokens 265216`（≈256K+3K） |
| 3 | 加完 `--num-tokens` 后 prefill OOM、worker 崩 | 专家缓存先占满，没给 KV 留空间。必须**同时**给 `--moe-cache-rate 0.40`（上游 **#150 / #401**） |
| 4 | 跑过 `ft bench bw` 后，`--moe-strategy auto` 静默变成 hybrid，实测慢 **27%** | `auto` 会采纳 benchbw profile，而该 profile 是在**错误几何**上测的（见 §2 补丁 05）。**显式固定 `--moe-strategy offload`**（上游 **#151 / #436**） |
| 5 | 并发上不去，mrr 调到 8 也没用 | 并发上限由 **mamba 槽**决定：`slots = 4*mrr + max(4, 2*mrr) + 1`。mrr=2 → 12 槽（3 路长上下文），mrr=4 → 24 槽（6 路）。「mrr>3 没用」的说法是错的 |
| 6 | `ft ctl cache rebuild` 热调后服务卡死 | 上游 **#526**：重建 OOM 会 wedge 服务器且无法回滚。**改参数一律重启**（只需 40 秒，热调不值得冒险） |
| 7 | 升级 FreeToken 后补丁静默失效 | 三个补丁都打在 site-packages 里，`uv pip install -U` 会覆盖；且 `RECORD` 不会因 patch 更新，`pip check` 看不出异常。启动脚本已内置自检，缺补丁直接拒绝启动 |

## 4. 验证结果（2026-09-24，客户端计时）

> ⚠️ **测性能前先确认窗口独占**：本机有外部客户端会不定时打 8000 口，实测能把 145K decode 从 39.4 压到 8.8 t/s。
> 查日志中 `#running-req` 与 HTTP 行确认（见 TROUBLESHOOTING #18）。**8K prefill 的方差尤其大，不要引用单点值。**

| 类别 | 项目 | v4（FreeToken） | v3（sglang v0.5.20） | 结果 |
|---|---|---|---|---|
| 性能 | decode @73K | **62.6–63.6 t/s** | 42.0 | ✅ +49~51% |
| 性能 | decode @145K | **60.2 t/s** | 39.4 | ✅ +53% |
| 性能 | prefill 254K | **2081 t/s**（122s） | 1710（149.2s） | ✅ +22% |
| 性能 | prefill 64K | **2140 t/s** | 1887 | ✅ +13% |
| 性能 | decode 短（512） | **64.4 t/s** | 42.5 | ✅ +52% |
| 性能 | prefill 8K | 371–1566 t/s（方差极大） | 1968–1994 | ❌ ~0.2–0.8× |
| 性能 | 并发 1/2/4/8（聚合，含 TTFT） | 35.6 / 54.0 / 64.2 / 63.7 | — | ⚠️ 见下 |
| 性能 | 并发 1/2/4/8（每流纯解码） | 58.1 / 48.0 / 49.5 / 49.0 | 单流 42.5（短上下文） | ✅ 单流远优；高并发扩展性 v4 弱 |
| 性能 | TTFT（短请求） | 3.4s（全命中 ~4.0s） | 0.42s | ❌ 8× |
| 性能 | 多流最长冻结 | 17.7s → **7.5s**（打 #484 后） | 切成 4s 块插队 | ❌ 仍劣 |
| 资源 | 模型加载 | **~39 秒** | ~5.5 分钟 | ✅ ~8× |
| 资源 | GPU 显存 | 44–46GB / 48GB | 41.7GB | 略高（moe-cache 让位） |
| 资源 | host RAM 峰值 | ~200GB（126GiB + 63.3G 专家 banks） | ~200GB | ≈ |
| 功能 | Anthropic `/v1/messages` | ✅ 原生 | ❌ | ✅ |
| 功能 | 256K 上下文 | ✅ 265216 token 池 | ✅ | ≈ |
| 部署 | SM89 专用 flash_attn wheel | 不需要（走 flashinfer） | 必需（PR #2751） | ✅ 更简单 |

> ⚠️ **并发数两个口径不可混用**：v4 的「聚合」= 总 token ÷ 墙钟（含 TTFT，故 N=1 反而低于单流速率），
> 「每流」= token ÷ 各 token 间隔之和（纯解码）。v3 的历史值（42.5/47.7/72.3/108）**不含 TTFT**——
> 因此本表**不把两边的并发聚合并排比**；可比的是单流纯解码：v4 58.1 t/s vs v3 42.5 t/s。
> 结论：v4 **单流更快，但并发扩展性更差**。

调优过程中确认的几个单变量结论（详见 [BASELINE.md](BASELINE.md)）：
`moe-cache-rate` 0.35→0.40 使 73K decode **+4%**；`FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` 使 mamba 池
1.3G→675M 且 decode 不降（65.1）；mrr 2→4 把并发上限从 3 路提到 6 路、decode 不变。

## 5. 历史：v3（2026-09-23，官方 sglang v0.5.20 基线）

<details>
<summary>展开 v3 路线记录（保留作参考；脚本与补丁仍在本仓库）</summary>

### 5.1 为什么升级到 v0.5.20

- **v0.5.20（2026-09-18）是首个正式收录 Qwen3.8-Flash-Next 的 release**（PR #37500 于 9/8 合入 main，v0.5.19 不含）
- 合入后 main 上又落了 4 个本项目直接受益的修复，v2 栈（PR 分支快照）里没有：
  - `#38851` QSA 分页 sparse-decode gather 内存安全（zero-fill scratch / int64 偏移 / gather 时反量化 FP8）
  - `#38855` sparse prefill 对 FP8 缓存前缀反量化
  - `#39446` compress gather 行数钳制（reland）
  - `#39474` 复用 alt_stream（修 decode overlap 路径反复建流的泄漏）
- **Lsglang v1.5.6 不可用**：其 `dsv4.1-lkmoe-sm80plus` 分支不含 `qwen4_exp.py`（主线转向 DeepSeek-V4.1），Flash-Next 无现成 wheel → 仍需自建
- 上游 PLE 修复 `#39928`（meta device 方案）于 **9/20** 合入 main，晚于 v0.5.20 → v3 **仍需**本地 PLE 补丁
- 官方硬件矩阵依旧**不含 4090/SM89**——v3 靠 lk_moe 混合推理 + PLE 补丁，属社区路线

### 5.2 升级方法（v0.5.20 基线重放）

构建仓是**浅克隆**（与上游无共同祖先）→ 不能 `git merge`。改用「以官方快照为基线、重放补丁」：

```
git fetch --depth=1 https://github.com/sgl-project/sglang tag v0.5.20   # 只取快照
git checkout -b v0520-lkmoe v0.5.20
git apply patches/01_lk_moe_v0520.patch          # lk_moe 集成（15 文件，直接可打）
git apply patches/02_ple_cpu_alloc.patch         # PLE CPU 分配（48GB 卡必需）
git apply patches/03_resolved_files_v0520.patch  # 2 处人工裁决差异
python -m build --wheel --no-isolation           # SGLANG_BUILD_RUST_EXTS=none
```

全流程见 [`scripts/build_upstream_flashnext.sh`](../scripts/build_upstream_flashnext.sh)。
**三个补丁在干净 v0.5.20 上均已验证直接可打**（无 3-way、无冲突）。

`03_resolved_files_v0520.patch` 覆盖两处需要判断的差异：
- `pyproject.toml`：取上游依赖版本（`sglang-kernel==0.4.7` 等）+ 追加 `lk_moe==2.4.1` + 包名 `lsglang` / 版本 `1.6.0+flashnext.v0520`
- `quantization/modelopt_quant.py`：**保留 lk_moe 的 CPU 常驻层跳过**（`is_gpu_resident_layer` 判定）**并接入上游新增的 megamoe 分支**

> 勘误（2026-09-24）：本文早期版本在此处声称 03 号补丁还含 `configs/qwen3_asr.py`，且称 01 号补丁为 18 个文件——
> 实际分别是 **2 个文件**和 **15 个文件**。

### 5.3 v3 踩到并修掉的 4 个坑

| # | 现象 | 根因与修法 |
|---|---|---|
| 1 | 启动即 `Exception: sglang-kernel is installed with version 0.4.6.post1, which is less than the minimum required version 0.4.7` | v0.5.20 启动强校验。**只升这一个包**（`pip install sglang-kernel==0.4.7`，dry-run 确认无连带升级） |
| 2 | `RuntimeError: get_global_server_args() is retired` | v0.5.20 废弃该 API。lk_moe 补丁里 1 处（`FusedMoE.get_max_num_group_batch_size`）→ 改 `get_schedule().chunked_prefill_size` |
| 3 | 从 root 外壳跑启动脚本时，子进程 `PermissionError: [Errno 13] Permission denied: '/root'` | Python multiprocessing spawn 的 worker 会 `os.chdir(父进程 cwd)`；root 外壳 cwd=`/root`。**启动脚本加 `cd` 到公共目录**（已内置） |
| 4 | 构建产物异常 | ① 需 `setuptools-rust`/`setuptools-scm`；② 必须 `SGLANG_BUILD_RUST_EXTS=none` 保持纯 Python 包；③ 打包前清 `python/build`（残留会让 wheel 从 20MB 涨到 76MB） |

### 5.4 v3 验证结果（2026-09-23，独占窗口对照 v2）

| 类别 | 项目 | v3 实测 | v2 基线 | 结果 |
|---|---|---|---|---|
| 功能 | /v1/models、对话、工具调用、thinking、effort 别名 | — | — | ✅ |
| 性能 | prefill 8K / 64K | 1968–1994 / 1887 t/s | 1874 / 1923 | ✅ |
| 性能 | 254K 冒烟（255k tokens） | 149.2s | 142.5s | ✅ 96% |
| 性能 | decode 512（短） | 42.5 t/s | 41.9 | ✅ |
| 性能 | decode @73K / @145K | 42.0 / 39.4 t/s（ITL p50 24ms） | 40.1 / 36.5 | ✅ |
| 性能 | 并发聚合 1/2/4/8 | 42.5 / 47.7 / 72.3 / 108 t/s | 39 / 44 / 65 / 97 | ✅ +8~11% |
| 正确性 | 98K 上下文针式检索（3 处不同深度） | 3/3 命中 | — | ✅ |
| 稳定性 | 20 轮 decode 显存曲线 + 20 分钟混合 soak | 零增长 | — | ✅ |
| 资源 | 加载 ~5.5 分钟、显存 41.7GB | 41.6GB | ✅ |

</details>

## 6. 历史：v2 / v1

<details>
<summary>展开 v2（2026-09-14，Lsglang 分支 + PR #37500）与 v1（Lsglang 1.4.13）</summary>

### 6.1 v2 路线

`Lsglang 0.5.19-lkmoe` 分支 merge 上游 PR #37500 → PLE 补丁 → 自建 wheel（`lsglang-1.5.0+flashnext.merge2/3`）。

**当时的三个拦路问题**（第 1、3 条对 v3 仍适用）：

- **PLE n-gram 表 GPU 瞬态分配 OOM（不改必炸）**：上游 `Qwen4ExpPLELayer.__init__` 先在 GPU 构造整张表
  （47.7GiB）再搬到 pinned 主机内存——48GB 卡没有路径能过，且该 GPU 占位随后被 `del`，纯浪费。
  显式 `--ple-offload-embedding` 无效（构造阶段就炸）。修复见 `patches/02_ple_cpu_alloc.patch`。
  （上游 9/20 的 #39928 用 meta device 解决同一问题，晚于 v0.5.20。）
- **pinned 内存的 memlock 限制**：pinned PLE 表需 47.7GiB 锁页内存，系统默认 `ulimit -l` 常仅 8MB。
  修复：启动脚本内置 `ulimit -l unlimited`。
- **tilelang 0.1.12 编译不兼容**：CUDA graph 捕获期报 `CUDA compiler and CUDA toolkit headers are incompatible`。
  修复：锁 **tilelang==0.1.11**。

v2 验证（2026-09-14）：12/12 通过；decode 41.5–41.9 t/s、128K 38.7、prefill 8K/64K 1999/1937；
加载 254s、显存 41GB。测速曾因 GPU 59°C 误判为退步，冷却后复原（见 TROUBLESHOOTING #11）。

### 6.2 v1 路线

Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内）+ lk_moe 2.4.0，直接装 release wheel
（`scripts/install.sh`），最省事但功能最少。启动脚本为 `scripts/start_lsglang.sh`。

</details>

## 7. 运维方式

| | v4（当前，FreeToken） | v3（sglang，历史） |
|---|---|---|
| 启动 | `bash scripts/start_freetoken.sh`（自动 sudo 提权 + memlock + 补丁自检 + 就绪等待 + 可选关 GUI） | `bash scripts/start_lsglang_upstream.sh` |
| 就绪判据 | 日志 `ready to serve`（**不能用 `/health`**） | `/health` 200 |
| 停止 | `pkill -f "ft-venv/bin/ft serve"` | `pkill -f 'sglang serve'` |
| 日志 | `/tmp/ft.log` | `/tmp/lsglang.log` |
| 启动耗时 | ~40 秒 | ~5 分钟 |
| 开机自启 | 无 | 无（本项目一贯手动启动） |
| 升级后必做 | **重打三个补丁**（`scripts/install_freetoken_patches.sh`） | 确认 tilelang 未被升到 0.1.12+、sglang-kernel ≥ 0.4.7 |

> 若日后 `uv pip install -U freetoken`，务必重跑补丁脚本——否则服务会在下次启动时被自检拦下。

## 8. 回滚

```bash
# v4 → v3（sglang 环境、wheel、脚本全部原样保留）
pkill -f "ft-venv/bin/ft serve"
bash scripts/start_lsglang_upstream.sh          # 约 4-5 分钟起来

# v3 → v4
pkill -f 'sglang serve'
bash scripts/start_freetoken.sh                 # 约 40 秒起来
```

双向回滚都很快（引擎侧无需重建环境）。本仓库已收录 v4 的完整版本快照
[`freeze-freetoken-0.1.3.txt`](freeze-freetoken-0.1.3.txt)（102 个包，可直接 `uv pip install -r` 复现）。
建议另外保留：旧启动脚本副本、以及各版本 wheel（sglang 侧）。
