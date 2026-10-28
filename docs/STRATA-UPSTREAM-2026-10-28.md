# Strata 上游升级评估续篇：0.2.5 / 0.2.6 / 0.2.7

- 调研时间：2026-10-28 16:30–17:10（Asia/Shanghai），**只读**：未停生产、未启停引擎、未跑 setup、未改任何生产文件
- 数据来源：GitHub Releases API（v0.2.3–v0.2.7 正文逐条读完）、release 资产 `SHA256SUMS.txt` / `strata-feed.json`、`setup.py@v0.2.7`、`expert_layout.cpp@v0.2.7` 原始文件抓取、本地 0.2.2 源码树
- 生产基线：Strata **0.2.2 服务端 + engine 0.1.47**，DavidAU Qwen3.8-Flash-Next abliterated **IQ3_S**，单 RTX 4090 48GB（**sm_89**），`--max-context 262144 --kv k8v4 --kv-resident 32768 --spec 4 --batch-mtp --prefill auto:32768 --resident-budget-gib 56 --conversation-cache-mib 8192 --vram-reserve-mib 2560 --vision`
- 上一篇：`docs/STRATA-UPSTREAM-2026-10-25.md`（当时结论「暂不升级」，因为 0.2.3/0.2.4 在 sm_89 上没有增量）

---

## 0. 结论

**0.2.5 / 0.2.6 对 Qwen 单卡没有引擎增量；0.2.7 是继 0.2.2 之后第一个真正改到 sm_89 的版本。**

- **0.2.5**：全部引擎 0.1.50。Qwen 侧只有一条与单卡相关——「async tier 自测 64 个窗口，不超过 2% 慢才保留」；其余是 DeepSeek、安装器、Terminal/App。
- **0.2.6**：Qwen 引擎**逐字节不变**（仍是 0.1.50），只升 DeepSeek 0.1.51 和 Control 0.1.51；对我们是空版本。
- **0.2.7**：所有引擎 0.1.52。与单 4090 真正相关的只有三件事：
  1. **sm_89 量化矩阵乘用「实测速度表」**（`STRATA_MMVQ_IL_89=0` 可关）——**结果 bit-for-bit 不变**，但上游自称「我们没有 Ada 卡，只有报告者的测量数据」，**没有 4090 的官方 A/B 数字**；
  2. **引擎修复**：`#1503` 会话缓存不再串用别的会话的 KV（我们开了 8 GiB conversation cache + 4 slots，这条最重要）；`#3602` 读表失败不再让下一次请求永久等待；`#2409` 恢复会话时的复制流回退；`#3911` 草稿词表行越界；未声明的 tool call 以文本返回时恢复（`#3796`）；
  3. **默认行为改变基本不落在我们卡上**：12 GB NVIDIA 的 residency 路由（≤13 GiB 才启用）、Radeon、双卡 layer-major prefill / pipelined batch 都不适用；`STRATA_DEFAULTS=026` 可整体回退。

**推荐路径：换 `serve/` 到 0.2.7 源码 + 换 `engine/` 到 0.1.52 预编译包；pack / tokenizer / 合并模板 / 启动参数全部不动。** 收益是「修复 + sm_89 表（结果不变）」，不是可观的提速；如果近期没有停机窗口，保持 0.2.2 也完全安全。

**当前唯一阻塞：GitHub 下载。** 见 §5（已于 17:38 下完，见 §8）。

---

## 1. 版本与引擎对照

| tag | 发布（UTC） | Control | Qwen 引擎 | DeepSeek 引擎 | 对我们（sm_89 单卡 Qwen） |
|---|---|---|---|---|---|
| v0.2.2（现生产） | 10-21 11:10 | 0.1.47 | **0.1.47** | — | 基线 |
| v0.2.3 | 10-23 07:44 | 0.1.48 | 0.1.48 | — | 无 sm_89 增量（详见上一篇） |
| v0.2.3.1 | 10-23 13:45 | 0.1.48.1 | 0.1.48 | — | 只修 setup 读 arch 列表 |
| v0.2.4 → 0.2.4.2 | 10-24 02:18 | 0.1.49.2 | 0.1.48 | 0.1.49.2 | Qwen 零变化；DeepSeek 无 sm_89 |
| **v0.2.5** | 10-25 10:06 | 0.1.50 | **0.1.50** | 0.1.50 | async tier 自测；其余无关 |
| **v0.2.6** | 10-25 13:45 | 0.1.51 | **0.1.50（不变）** | 0.1.51 | 空版本 |
| **v0.2.7** | 10-27 14:54 | 0.1.52 | **0.1.52** | 0.1.52 | sm_89 表 + 引擎修复 |

## 2. 引擎包事实（v0.2.7，Linux CUDA 13）

| 项 | 值 |
|---|---|
| 文件名 | `strata-engine-0.1.52-linux-x86_64-cuda-cuda13.tar.gz` |
| 大小 | **645,979,691 B**（616 MiB） |
| sha256 | `4177c1a8ffc59bf66074093b22ac55ed477bcbe84f8bd12e378b0f1236da55c7`（`SHA256SUMS.txt` 与签名 feed 一致） |
| archs | `sm_75, sm_86, sm_89, sm_120, compute_120` — **含 sm_89** |
| min_driver | NVIDIA **580.65**；本机 610.57.04 ✓ |
| 安装后体积 | 1,039,550,681 B（feed `installed_bytes`） |
| 备选 cuda12 | 933,267,301 B，sha `f08895a9…de30`，archs 含 `compute_89`，min_driver 575.51 — 不需要 |
| Control 0.1.52 | 我们直接跑 `serve/server.py`，**不需要** Terminal/Control |

## 3. 0.2.7 里真正落在「单 4090 + IQ3_S + 262144」上的条目

**会生效：**
- sm_89 量化矩阵乘实测速度表（`STRATA_MMVQ_IL_89`，默认开；关掉 `=0`）。上游原话：**bit for bit the same as before**，没有 Ada 卡实测。
- 引擎修复 `#1503`（缓存串会话）、`#3602`（读表失败永久等待）、`#2409`（恢复时复制流回退）、`#3911`（草稿词表越界）、`#3796`（未声明 tool call 恢复）、`/v1/responses` 接受 -1/0、logprobs 部分修复。
- 新 opt-in `STRATA_KVA=1`：长 prompt 最后 12 层近似读（4K+ 首次读），RTX 5070 上 14K +11%、30K +18%；**官方明确「可能丢细节」，默认关**。我们 262144 ctx 的 agent 负载可考虑试，但必须单独验收。

**不会生效（条件不符）：**
- 12 GB NVIDIA residency 路由（只对 ≤13 GiB 离散 N 卡；48G 不触发）；
- 双卡 layer-major prefill / batch pipeline / `--split-serve`（单卡）；
- Radeon RX 9000 / Strix Halo / DeepSeek 全部内容；
- 默认「Faster long prompts」是关的。

**回退开关：** `STRATA_DEFAULTS=026` 一键回到 0.2.6 行为；`STRATA_MMVQ_IL_89=0` 只关 sm_89 表；显式设置的变量/flag 仍优先。

## 4. 兼容性核对

- **MIN_ENGINE 没有抬**：`setup.py@v0.2.7:251` 仍是 `MIN_ENGINE = (0, 1, 46)`（本次抓到的是 196 KB 部分文件，但同文件内已出现 0.2.7 专属注释，版本可信）。即 0.2.7 的 serve/setup 接受 0.1.46+ 引擎；**理论上 0.2.7 serve 可以配 0.1.47 引擎（混合），但 0.2.7 的引擎侧收益全在 0.1.52，不建议停在这一步**。
- **pack 格式没变**：`expert_layout.cpp@v0.2.7` 仍按 `# strata native experts v` + 任意 `vN` 读取；我们的 pack 头是 v3，不需要重打包。
- **模板/配置无关**：合并模板在 tokenizer 目录，serve/engine 升级不碰它；`--kv k8v4`、`--spec 4`、`--batch-mtp`、`--ple-gguf` 等 18 个参数在 0.1.52 是否仍全部存在，**待下载后 `strata --help` 实测核对**。
- `--prefill auto:32768`、`--resident-budget-gib 56`、`--conversation-cache-mib 8192` 为 server 参数，0.2.7 serve 默认值有调整（#3493 自动定 cache），我们显式给值不受影响。

## 5. 下载现状（今日实测）

| 通道 | 实测速度 |
|---|---|
| GitHub release 资产（直连/7890 代理） | 16–45 KB/s，波动大，常断 |
| `gh-proxy.org` 镜像 | 42–53 KB/s（**忽略 Range，不能续传**） |
| 其他镜像（ghproxy.net / ghfast.top / moeyy / kkgithub / bgithub…） | 0–26 KB/s 或直接失败 |
| 8 并发分片 | 合计 56–81 KB/s，无线性收益 |
| 清华 TUNA（对照） | 300 KB/s → 说明是本机出口/GitHub 路由问题，不是本地链路坏死 |

按 45 KB/s 估算 646 MB ≈ **4 小时**；按当前 16 KB/s ≈ 10 小时。`/tmp/s027` 已保留 0.1.52 部分下载，可 `curl -C -` 续传。

## 6. 建议的实施步骤（网络恢复后）

1. 下载并核对：`sha256sum` = `4177c1a8…a55c7`，`tar tzf` 完整。
2. 解压 v0.2.7 源码到 `strata/Strata-0.2.7/`（`--strip-components=1`），`engine/` 用 0.1.52 包解压；**绝不跑 setup.py / update.sh**。
3. 复制 `strata-davidau-iq3_s-022.json` → `-027.json`，只改 `exe` / `vision.exe` / `cwd` / `log` 四处；`args` 逐字不动（待 `--help` 核对）。
4. 写 `run_strata_027.sh`、保留 `run_strata_022.sh` 回滚；验收脚本照 `test_strata_022.sh` 六项：`/health`、`/props` 模板 249545、普通对话、auto 模式、vision、日志 grep `build_info "Strata 0.1.52"`。
5. 停机切换：`systemctl --user stop strata-prod`（及残留 `strata-restart-fit.*`）→ `stop_strata.sh` 等显存 <1000 MiB → `systemd-run --user --unit=strata-prod … run_strata_027.sh`。
6. 记录： downtime、ready 时间、VRAM、prompt/decode tok/s（与 0.1.47 的 36–55 s ready、99–141 t/s decode 对照）。

## 7. 待办 / 未验证

- [ ] 0.1.52 `--help` 与 18 个参数逐项比对
- [ ] 0.2.7 serve 与 0.1.52 引擎的握手（`MIN_ENGINE` 只在 setup，serve 无门控，但未实测）
- [ ] 0.2.7 下 `{REASON:xomni}` / 合并模板 249545 是否仍原样生效
- [ ] `STRATA_KVA=1` 的 agent 任务质量（可选，单独一轮）
- [ ] feed 签名（`.sig`）公钥未核，本次只核 `SHA256SUMS.txt` + feed 内 sha256 一致

---

## 8. 实测结果（2026-10-28 17:38–17:46，已完成切换）

**下载与装配**
- 646 MB 引擎包：8 分片并发 + 分段 Range 续传，12 轮 ~26 分钟下完；`sha256sum` = `4177c1a8…6da55c7`（与 `SHA256SUMS.txt` / 签名 feed 一致），`tar tzf` 完整。
- 源码树未下整包；用 GitHub tree API + raw 逐文件重建 `serve/`、`strata_control/`、`tools/`、`strata_cli/`：**412 个文件全部与 tree 的 git blob sha 一致**（未变的 66 个从 0.2.2 直接复用），`py_compile` 全过，`import serve.server` 成功。
- 引擎解压后 `BUILD.json`：`version 0.1.52`、`archs [75,86,89,120]`、`vision 75;86;89;120`；`strata --version` → `strata 0.1.52`。

**切换与验收**（`strata-prod` 单元）

| 指标 | 0.2.2 / 0.1.47 | 0.2.7 / 0.1.52 |
|---|---|---|
| 停机 | 48 s（10-22） | **63 s**（17:41:56 stop → 17:42:59 ready） |
| ready | 36 s | **52 s** |
| VRAM | 46,439 MiB | **46,441 MiB** |
| `/props` 模板 | 243,742 chars | **243,742 chars**（未变） |
| `build_info` | Strata 0.1.47 | **Strata 0.1.52** |
| 短对话 decode | 99–141 t/s | **126.6 t/s**（11 prompt tokens） |
| 2847-token prompt | 3,287 t/s（10-22） | **2,031.6 t/s**（首次、冷） |
| 自动模式 prompt tokens | 2,847 | **2,847**（未变） |

- 验收脚本 `test_strata_027.sh`：**9/9 PASS**（health / ctx / images / props / build_info / 普通对话 / 自动模式 / vision / 日志）。
- Anthropic 路径：`thinking:{type:adaptive}` + `output_config:{effort:"low"}` → 200，`content` 含 `thinking`+`text`（DSH 的 `forceAdaptiveThinking` 配置仍然有效）。
- 视觉：64×64 PNG → 32 tokens，vision 0.1.52 正常。
- 日志仍出现 `strata: PLE table mlock failed (ulimit -l)` —— 0.1.47 时代相同的环境告警，不影响速度。
- 启动命令与 0.2.2 完全相同（32 个 args + 服务端追加 `--batch 4`）；pack v3、tokenizer、合并模板、config 参数均未改动。

**回滚**：`run_strata_022.sh`（0.2.2 + 0.1.47）仍在；`stop_strata.sh` + 改跑该脚本即可，旧树与旧引擎未删除。
