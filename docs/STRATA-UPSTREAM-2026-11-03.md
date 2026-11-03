# Strata 0.2.10（engine 0.1.56）：性能零收益，纯修复升级

- 调研 + 升级时间：2026-11-03 11:36–12:26（Asia/Shanghai）
- 数据来源：GitHub Releases API（v0.2.10 全文）、`SHA256SUMS.txt`、`git trees/v0.2.10?recursive=1`（5681 blob）+ raw 逐文件抓取（`serve/ strata_control/ tools/ strata_cli/` 共 **496 文件，每个 `git hash-object` 都等于 tree 的 blob sha**）、engine 0.1.56 包内 `BUILD.json`
- 升级前：Strata **0.2.9 + engine 0.1.55**（v5.5），DavidAU Qwen3.8 IQ3_S，单 RTX 4090 48GB（sm_89），`"parallel": 4`（服务端补 `--batch 4`），`--mtp` + `--batch-mtp` + `--spec 4`
- 上一篇：[STRATA-UPSTREAM-2026-11-01.md](STRATA-UPSTREAM-2026-11-01.md)

---

## 0. 结论

**0.2.10 对我们没有任何性能收益，但仍然值得升——拿到的是 5 条正中我们链路的修复。** 已于 2026-11-03 12:19 切到 0.2.10 + 0.1.56，停机 47 s，验收 23/23。

一句话判断：**只要并发保持 4，0.2.10 的头部卖点（DFlash2 默认 drafter、全卡切分引擎）全部与我们无关。**

## 1. 为什么 DFlash2 用不上（这是本次最关键的判断）

- 上游原文：`Auto picks MTP when several requests run at once (--parallel or --batch of 2 or more)`；`DFlash2 answers one request at a time`。
- 代码佐证（`strata_control/drafting.py:535-541`）：Auto 的选择函数直接判断
  `slots > 1 or any(a in items for a in ("--batch", "--slots", "--pipeline-windows", "--parallel"))` → 为真就用 MTP。
- 我们的配置 `"parallel": 4` → 服务端起引擎时补 `--batch 4` → **Auto 必然选 MTP**。实测确认：
  `/metrics.engine.drafting = "mtp"`、日志 `strata mtp: draft layer loaded, 835 MiB of VRAM`、**没有下载那 1.2 GB 的 DFlash2 草稿层**。
- DFlash2 的收益（上游 IQ3_S 单卡：RTX PRO 5000 +7.88% greedy / +6.23% 短采样，Strix Halo +8.4%）只在**单槽**下成立。对我们这种部门服务（4 槽）换过去等于砍掉并发，不划算。

## 2. 全卡切分引擎（all-cards split）也不适用

Auto 自动开条（逐条比对我们的机器）：

| 条件 | 我们 |
|---|---|
| Linux | ✅ |
| **恰好两张** cc 8.6 / 12.0 的 NVIDIA，或**四张** R9700 | ❌ 单张 4090（sm_89） |
| Qwen3.8 文件、专家全进显存 | ✅ |
| ≥43 GiB 空闲 RAM | ✅ |
| context ≤ 131072 | ❌ 我们是 262144 |

引擎包里确实带了 `strata-split-2-sm86` / `strata-split-2-sm120` 两个额外二进制（0.1.55 没有），但都是双卡专用，单卡上不会被用到。

## 3. 命中的修复（升级的真正理由）

| 修复 | 为什么命中我们 |
|---|---|
| **#4590** Qwen 的 repeat / frequency / presence 惩罚把 **prompt 尾部 token** 也算进窗口 | 新任务会复读上一轮的旧 JSON。我们采样 + 工具 JSON，这是质量问题。`STRATA_PENALTY_PROMPT=1` 可退回旧行为 |
| **#4677** 停放的会话下一轮偶发被拒，且约 6% 的 prompt 被重读 | 我们 4 slot + `--conversation-cache-mib 8192`，正走这个机制。`STRATA_SERVE_STOP_BOUND=0` 可关 |
| **#4663 / #4700** "用过的工具视为已声明"扩到 **Anthropic 与 Responses 路由** | DSH 走的就是 `anthropic-messages`。**源码证据：0.2.9 的 `serve/server.py` 里根本没有 `anthropic_tools()`，0.2.10 新增（`server.py:8518`，行尾注释即 `#4663`）**，我们上一版只吃到了 Chat Completions 的那份 |
| **#4838** `/v1/messages` 上下文溢出改用 Anthropic 自己的措辞 | 同一条路由；客户端会压缩/收短 `max_tokens`，而不是拿同样请求重试三次 |
| **#4701 / #4751** write 内容里带独立成行的闭合标签会被截断；`tools.write` / `functions.write` / `default_api.write` 归一化成已声明的 `write` | DSH 的写文件工具与工具名归一化 |
| **#4671** 系统 memlock 限制下 PLE 表被拒 → 先抬软限到硬限，仍装不下就说清楚要设多少 | 我们日志里每代都有的 `PLE table mlock failed`（见 §9） |

## 4. 未命中（明确跳过）

DFlash2 本体、全卡切分引擎、8 GB 卡 128K 上下文分层、ISTA-DAS Coder（DeepSeek 系列）、Mac/Apple Silicon 的 DeepSeek 看图、Intel Arc、四卡 Radeon、DGX Spark、RTX 50 + CUDA 12 的首次启动编译（我们 CUDA 13 + sm_89）、App 与 Monitor 的 UI 修复（#4725/#4748/#4749/#4783 等全部是桌面 App 的）。

## 5. 引擎包与源码树事实

- 资产 `strata-engine-0.1.56-linux-x86_64-cuda-cuda13.tar.gz`，**667,699,126 B**，`sha256 df6a6dfd806289caa42becd2acc14081815bedac9ec8f5efa3821a7d10582ee5`（与 `SHA256SUMS.txt` 一致）。
- `BUILD.json`：`{"version":"0.1.56","source":"f-0291-rc1 06698c674673","archs":[75,86,89,120],"ptx":true,"cuda":"13.0","portable":true}` —— sm_89 在列。
- 包内新增 `strata-split-2-sm86`（23.5 MB）与 `strata-split-2-sm120`（27.3 MB）两个切分引擎二进制；`strata` 本体从 286.5 MB 涨到 **307.7 MB**。用不到的二进制只占 51 MB 磁盘，不影响加载。
- 源码树 496 文件（0.2.9 是 479），相对 0.2.9 新增 `strata_control/split_engine.py`、`tools/dflash2/` 等；`serve/server.py` 新增 `anthropic_tools()`、`drafting` 集成与 DFlash2→MTP 的自动重试。

## 6. 升级步骤（实际执行）

```bash
# 1) 引擎：4 MiB 分片 × 160，20 线程，387 s
sha256sum strata-engine-0.1.56-linux-x86_64-cuda-cuda13.tar.gz   # df6a6dfd…82ee5 ✓

# 2) 源码树：tree API + raw 逐文件，496/496 blob sha 校验通过
#    （复用了 0 个 0.2.9 文件——脚本参数写错，全量重下；只有 7 MB，无影响）

# 3) 树与配置
tar -xzf … -C strata/Strata-0.2.10/engine
# 配置只改 4 个字段：exe / cwd / log / vision.exe，32 个启动参数逐字未动

# 4) 切替
bash stop_strata.sh     # 9 s，显存 9 MiB
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$PWD /usr/bin/bash $PWD/run_strata_0210.sh
# 12:20:04 /health loaded=true（总计 47 s）

# 5) 验收 + 指针
bash test_strata_0210.sh        # PASS=23 FAIL=0
cp run_strata_0210.sh run_strata.sh
```

## 7. 实测（0.1.55 → 0.1.56，同机同参）

| 用例 | 0.1.55 | 0.1.56 |
|---|---|---|
| 6,193 tok prompt 预填充 | 4,035 t/s | **4,031 t/s**（-0.1%） |
| 12,633 tok prompt 预填充 | 4,733 t/s | **4,629 t/s**（-2.2%） |
| 4 并发聚合吞吐 | 178.5 t/s | 171.8 t/s（-3.7%） |
| 空载显存余量 | 2,487 MiB | 2,483 MiB |
| 专家缓存 | 33,607 MiB / 17,285 槽 | 33,607 MiB / 17,284 槽 |
| 模板 | 243,742 chars | 243,742 chars（未变） |
| `/props.build_info` | Strata 0.1.55 | **Strata 0.1.56** |

差异都在 ±4% 的运行间噪声内（同协议单次采样的分辨力只有 ±5%）。**这次升级的价值不在速度**，见 §3。

## 8. 回滚

```bash
bash stop_strata.sh
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$HOME/dsh_workspace/strata /usr/bin/bash $HOME/dsh_workspace/strata/run_strata_029.sh   # 0.2.9 + 0.1.55
# 更老：run_strata_027.sh（0.2.7 + 0.1.52）、run_strata_022.sh（0.2.2 + 0.1.47）
```

## 9. 遗留与注意

- `strata: PLE table mlock failed (…raise ulimit -l)` **仍在**（0.1.47 / 0.1.52 / 0.1.55 / 0.1.56 都一样）。实测本机 `ulimit -l` = **8192 KB = 8 MiB**，systemd 单元 `LimitMEMLOCK=8388608`（默认值），所以"抬软限到硬限"没有可抬的空间。机器 247 GiB 内存 / 180 GiB 可用，把单元改成 `LimitMEMLOCK=infinity` 就能消掉这条；但 offload 走 PINNED，mlock 只影响 CPU-executor 路径，**属于锦上添花**，暂不改。
- DFlash2 的 1.2 GB 草稿层**没有被下载**，磁盘上没有多出来的模型文件。
- 该版本引擎包体积涨到 667.7 MB（0.1.55 是 647.1 MB），多出来的是两个双卡切分二进制。
- 已知问题（上游自述，与我们相关的一条）：DeepSeek 在温度 0 下不同启动之间答案可能不同——我们是 Qwen，不受影响。
