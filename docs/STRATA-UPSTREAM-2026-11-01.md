# Strata 上游升级：0.2.8 / 0.2.9（engine 0.1.54 / 0.1.55）

- 调研时间：2026-11-01 09:28–09:55（Asia/Shanghai）
- 数据来源：GitHub Releases API（v0.2.8、v0.2.9 正文逐条读完）、`SHA256SUMS.txt`、`git trees/v0.2.9?recursive=1`（5319 blob）+ raw 逐文件抓取（`serve/ strata_control/ tools/ strata_cli/` 共 479 文件，**每个文件的 `git hash-object` 都等于 tree 给的 blob sha**，Python 文件全部 `py_compile` 通过）、engine 0.1.55 tarball 内 `BUILD.json`
- 升级前生产：Strata **0.2.7 服务端 + engine 0.1.52**，DavidAU Qwen3.8-Flash-Next abliterated **IQ3_S**，单 RTX 4090 48GB（**sm_89**），`--max-context 262144 --kv k8v4 --kv-resident 32768 --spec 4 --batch-mtp --prefill auto:32768 --resident-budget-gib 56 --conversation-cache-mib 8192 --vram-reserve-mib 2560 --vision`，配置 `"parallel": 4`（服务端补 `--batch 4`）
- 上一篇：`docs/STRATA-UPSTREAM-2026-10-28.md`（0.2.5→0.2.7）

---

## 0. 结论

**0.2.8 是"修复版"，0.2.9 是本机（Linux + NVIDIA + Qwen3.8 + `--batch 4`）第一次吃到"默认打开的性能开关"。两版都建议升，我们已升到 0.2.9 + 0.1.55。**

- **0.2.9 有两条默认开、且正好命中我们的改动**：
  1. **并行请求栈**（9 个 `STRATA_*` 开关）——`--batch` / `--parallel` ≥ 2 且 Linux CUDA Qwen 引擎时由服务端 `setdefault` 打开。上游实测：8 并发总吞吐 1.45×，**单请求 +5.3%**，"单张大卡 +40~65%"。
  2. **Gated DeltaNet 张量核 prompt 内核**（`STRATA_GDN_TC=1` + `STRATA_GDN_CONVL2=2`）——Linux CUDA Qwen3.8 默认开。上游实测 prompt +4.4%(4K) / +5.5%(16K) / +5.7%(32K)，KL 均值 0.0019，61 题 55 vs 54。
- **0.2.8 对我们唯一重要的默认改动**：**工具调用容错默认开**（1462 个 agent 轮次里"按意图读对"从 92.9% → 98.3%，只作用于请求已声明的工具）。我们走 DSH 的工具调用链，这条比任何速度数字都值钱。
- 其余 0.2.8/0.2.9 内容几乎全是 Strix Halo / Intel Arc / Radeon / DGX Spark / DeepSeek / Mac / Windows 的，**与我们无关**（逐条见表 §2）。
- 升级只换「`serve/` 源码 + `engine/` 二进制」，模型 / pack / 合并模板 / 启动参数一字未动。

## 1. 版本与引擎对照

| tag | 发布（UTC） | Control | Qwen 引擎 | 对我们（sm_89 单卡 Qwen） |
|---|---|---|---|---|
| v0.2.7（升级前生产） | 10-27 14:54 | 0.1.52 | **0.1.52** | 基线 |
| **v0.2.8** | 10-29 22:40 | 0.1.54 | **0.1.54** | 工具调用容错默认开；Arc/四卡 Radeon/Mac 的改动与我们无关；12 GB 卡的 residency 路由**回滚**（N/A） |
| **v0.2.9** | 10-31 08:02 | 0.1.55 | **0.1.55** | 并行栈 + GDN 张量核**默认开**；工具调用修复；引擎按整机 NVIDIA 卡编译 |

## 2. 逐条对照：哪些落在我们这台机器上

### 2.1 命中（我们已吃到）

| 条目 | 出处 | 说明 |
|---|---|---|
| 并行请求栈默认开（9 个开关） | 0.2.9 | `--batch 4` 命中。开关：`STRATA_CKPT_POOL=1`、`STRATA_CKPT_POOL_READY=4`、`STRATA_SLOT_CKPT_MOVE=1`、`STRATA_SLOT_COPY_DIRECT=1`、`STRATA_SLOT_COPY_BATCHSYNC=1`、`STRATA_BATCH_GDN_GROUPS=1`、`STRATA_BATCH_ATTN_ROWS=1`、`STRATA_BATCH_SELF_COMMIT=1`、`STRATA_QSA_SLOTS=1` |
| GDN prompt 内核走张量核 | 0.2.9 | `STRATA_GDN_TC=1`、`STRATA_GDN_CONVL2=2`；引擎运行期门控 CC≥8，sm_89 满足 |
| 工具调用：代码栅栏/缩进块里的调用不再被执行 | 0.2.9 (#4702/#4703) | 栅栏必须等长闭合才算闭合 |
| 工具调用：会话里用过的工具视为已声明 | 0.2.9 (#4663) | 后续调用回落成工具调用而不是裸文本 |
| 工具调用容错默认开（参数残缺/缺 `</function>`/多余 think 结束符） | 0.2.8 (#4427/#4077) | 只对已声明工具生效 |
| 空闲 `--batch` slot 停放时保留共享前缀 | 0.2.8 (#4420) | 第二轮首 token 0.66 s 而不是 3.3 s —— 多轮对话直接受益 |
| PLE 表被锁时会话缓存不再短少 | 0.2.8 (#4518) | 我们日志里每代都有 `PLE table mlock failed`，属于同一条链路 |
| 客户端在 `[DONE]` 后立刻断开不再报 "Request cancelled." | 0.2.9 (#4690) | |
| 引擎按整机所有可用 NVIDIA 卡编译 + 启动时打印架构不匹配提示 | 0.2.9 (#4662) | 0.1.55 `BUILD.json` 仍是 `archs [75,86,89,120]`、`ptx: true`（PTX 兜底） |

### 2.2 不命中（明确跳过）

- **Strix Halo（gfx1151）**：Windows 上 prompt 1.8×、Intel Arc B 系列、DGX Spark、Mac M5 —— 都不是本机形态。
- **四卡 Radeon 的 32 槽位 profile**、**双卡 Arc 流水线**、**DeepSeek 多卡**：单卡无关。
- **DeepSeek 的 UD-IQ1_S / UD-IQ1_M 1-bit 集**：我们是 Qwen3.8，不是 DeepSeek。
- **12 GB 及以下 RTX 的 residency 路由**：0.2.7 打开、0.2.8 又默认关掉（与长会话重复文本有关）——本机 48 GB，从来不走这条。
- **App / Monitor / 聊天页渲染**（0.2.9 长答案主线程 15.7 s→9.0 s）：那是桌面 App 的 UI，我们只跑 `serve/server.py`。

## 3. 引擎包事实（v0.2.9，Linux CUDA 13）

- 资产：`strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz`，647,149,266 B，`sha256 ce86b23b26a4521a1871113057534b8a34c01df0ac7b916adccf9b1c8c068f92`（与 `SHA256SUMS.txt` 及签名 feed 一致）。
- `BUILD.json`：`{"version": "0.1.55", "source": "f-029-rel 99866a47a2b8", "archs": [75,86,89,120], "vision_archs": "75;86;89;120", "ptx": true, "cuda": "13.0", "portable": true, "vision": "cuda13.0"}` —— **sm_89 在列**，与 0.1.52 的 arch 集合一致。
- 包内：`strata`（286 MB）、`strata-vision`（125 MB）、`libstrata_kq.so`、`libcublas*.so.13`、`libcudart.so.13`。
- 无引擎版本门控：`serve/server.py` 不校验 `MIN_ENGINE`（该常量只在 `setup.py` 里），但仍**必须整代配对**（0.2.9 ↔ 0.1.55）。

## 4. 服务端源码事实（v0.2.9 相对 0.2.7）

- 运行时树差异 59 个条目：`serve/{server,frontend,replicas}.py`、`strata_cli/` 12 个、`strata_control/` 18 个（新增 `amd_four_card.py`、`driver_floor.py`、`gpu_pick.py`、`intel_spec.py`、`replica_profile.py`）、`tools/` 若干（新增 `par/`、`dgx_*` 一批）。
- `serve/server.py` 新增 `PARALLEL_STACK_ENV` / `parallel_stack_env()` / `GDN_STACK_CUDA` / `gdn_stack_env()`，全部 `env.setdefault(...)`：**自己设过的环境变量永远优先**。
- 生效条件（读代码确认，与上游描述一致）：
  - 并行栈：`parallel_slots(cfg) >= 2`（`--batch`/`--slots` 参数，或配置里的 `"parallel"`），`arch != deepseek41`，`backend ∉ {sycl, metal, vulkan, hip}`，**Linux only**。
  - GDN 栈：`arch` 默认 `qwen4exp`，`backend` 非 hip/sycl/metal/vulkan，**Linux only**。
- CLI 侧：`strata connect`（见 §7）、`strata models import`、launch 的 `--gpu` 记忆等，**我们不用 CLI 启动**，因此不涉及。

## 5. 升级步骤（2026-11-01 实际执行）

```bash
# 1) 引擎：分块下载（GitHub 单流 45 KB/s、20 流约 65 MB/min）
#    /tmp/s029/dl2.py —— 4 MiB 分片 × 155，断点续传，最后按序拼接
sha256sum strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz
# ce86b23b26a4521a1871113057534b8a34c01df0ac7b916adccf9b1c8c068f92  ✓

# 2) 源码：GitHub tree API + raw 逐文件（不走 archive tarball，之前它被截断过）
#    479 文件，378 个与 0.2.7 逐 blob 相同直接复用，101 个新下载

# 3) 树与配置
mkdir -p strata/Strata-0.2.9/engine
tar -xzf /tmp/strata-engine-0.1.55-...tar.gz -C strata/Strata-0.2.9/engine
# 配置只改 4 个字段：exe / cwd / log / vision.exe（其余 32 个参数逐字未动）
python3 -c "..."   # strata-davidau-iq3_s-027.json → -029.json

# 4) 切替（生产停机 47 s）
bash stop_strata.sh                      # 等显存 <1000 MiB，实测 9 s、显存 9 MiB
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$PWD /usr/bin/bash $PWD/run_strata_029.sh
# 09:41:45 /health loaded=true

# 5) 验收
bash test_strata_029.sh                  # PASS=21 FAIL=0

# 6) 生产指针
cp run_strata_029.sh run_strata.sh       # run_strata_027.sh 保留为回滚
```

## 6. 回滚

```bash
bash stop_strata.sh
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$HOME/dsh_workspace/strata /usr/bin/bash $HOME/dsh_workspace/strata/run_strata_027.sh
```

只回滚 0.2.9 的新默认值、不动引擎版本的话，在配置 `env` 或启动器里设：

| 目标 | 开关 |
|---|---|
| 关并行栈（9 项一起） | 任意一个设为 `0`，服务端就保留你的值（例如 `STRATA_CKPT_POOL=0`） |
| 关 GDN 张量核 | `STRATA_GDN_TC=0`、`STRATA_GDN_CONVL2=0` |

## 7. `strata connect`（终端启动文档新增章节）

`docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect` 是 **Terminal（`strata` CLI）** 的能力，用来给别的 app 写配置：

- 支持：OpenCode、Codex CLI、Claude Code、oh-my-pi(omp)、pi、Hermes Agent、OpenClaw，**以及 DeepSeek Harness**。
- `strata connect <app> [--write] [--check] [--url --key --model --context]`；`--check` 会真发一次 prompt + 一次工具调用。
- **DeepSeek Harness 是 `built_in=True`**：`strata connect deepseek-harness` 的输出就两句话——"DSH 是 Strata App 内置的编码 harness，这里没有文件要写" + "它的默认 compaction 余量（64K）比 32K 窗口还大，App 的桥接会替你设好"。
- **对我们的意义：没有可写的东西。** 我们的 DSH 是独立安装（`dsh-app`），走 `llm-pi-ai` 的 `anthropic-messages` 指向 `http://127.0.0.1:8000`，配置在 `$DSH_HOME/profiles/web/cordis.patch.yml`，与 `strata connect` 的文件目标不重叠；`contextWindow 262144` / `maxTokens 32768` 也远大于它测试用的 32K/4096 基线，不存在它说的那个 compaction 隐患。
- 真正能用的场景是：以后要让 OpenCode / Codex / Claude Code 直连本机 8000 时，`strata connect <app> --write` 一条命令搞定（会备份被替换的文件）。

## 8. 实测（2026-11-01 09:41 起，生产 8000）

**验收 21/21 PASS**（`test_strata_029.sh`）：

| 项 | 结果 |
|---|---|
| `/health` | `loaded=true`、`max_context=262144`、`images=true` |
| `/props` | `build_info "Strata 0.1.55"`、`chat_template` 243,742 chars（与 0.1.52 完全一致，模板未被 setup 覆盖） |
| 普通对话 | 11 prompt tokens → 49 completion tokens，中文正常 |
| 自动模式 | prompt_tokens 2,847（系统提示注入生效） |
| 视觉 | 64×64 PNG → 32 tokens 正常 |
| 引擎环境 | 11 个新默认值**逐个**在 `/proc/<pid>/environ` 中确认（9 并行 + 2 GDN） |
| 日志 | `engine started: …/Strata-0.2.9/engine/strata … --batch 4`、`session is up (engine 0.1.55)` |

**性能**（同机同参，`timings` 取自服务端自报）：

| 用例 | 0.1.52（升级前） | 0.1.55（升级后） | 备注 |
|---|---|---|---|
| 长 prompt 预填充（1,447 tok，冷） | 1,847.5 t/s | 1,758.0 t/s | 同一组 3 个 prompt、非交错；升级后那轮是**冷启动后首次**，专家缓存还没热 |
| 同上，换一批新 prompt（引擎已热） | — | 1,789.2 t/s | 与冷启动差 1.8%，说明上面那 -4.8% 主要是缓存温度 |
| 短对话 decode | 158.8 t/s | 164.4 t/s | |
| 长 prompt decode | 153.3 t/s | 174.6 t/s（热）/ 150.8（冷） | 抖动大 |
| **6,193 tok 预填充** | — | **4,035.3 t/s**（3 次，4035/4025/4078） | 大 prompt 摊薄开销后才看得出真实速率 |
| **12,633 tok 预填充** | — | **4,732.9 t/s**（3 次，4706/4733/4777） | |
| **4 并发（`--batch 4`）** | — | 4/4 成功，聚合 **178.5 t/s**，单请求 decode 48–60 t/s（合计 ~216） | 并行栈不崩 |

结论口径要诚实：**单次、非交错的同协议对比只能分辨 ±5% 级别的差异，而 0.2.9 的两个新默认栈理论收益也是 5% 量级**——上面的数字不足以宣称"更快"，只能说明"没有可见回退"。真正可信的证据是：21 项验收全过、4 并发不崩、12.6K prompt 4.7K t/s、以及 §2.1 里那批工具调用修复。

**资源**：引擎加载后 VRAM 46,333 MiB / 49,140 MiB（`2487 MiB of VRAM free with everything loaded`），与 0.1.52 的 46,441 MiB 基本持平。

## 9. 遗留

- 日志里 `strata: PLE table mlock failed (Cannot allocate memory; raise ulimit -l)` **与 0.1.52 / 0.1.47 一致**，不是新问题（offload 走 PINNED，mlock 只用于 CPU-executor 路径）。
- 严格交错的 A/B（同一批 prompt 在两代引擎上交替跑）**没有做**：需要多轮重启，生产中断代价大于收益。若以后要定论，最小方案是夜间做 4 轮交替、每轮 6 个新 prompt。
- `strata connect` 的 8 个 app 配置**未逐一验证**（我们用不到）；只读了生成器源码。
