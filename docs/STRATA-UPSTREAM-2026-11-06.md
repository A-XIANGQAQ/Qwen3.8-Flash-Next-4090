# Strata 0.2.11（engine 0.1.57）：W16 cache 在 48 GB 上没启用，价值在 4 条修复

- 调研 + 升级时间：2026-11-06 09:24–10:20（Asia/Shanghai）
- 数据来源：GitHub Releases API（v0.2.11 全文）、`SHA256SUMS.txt`、`git trees/v0.2.11?recursive=1`（**5,780 blob**）+ raw 逐文件抓取（`serve/ strata_control/ tools/ strata_cli/` 共 **502 文件，每个 `git hash-object` 都等于 tree 的 blob sha**）、engine 0.1.57 包内 `BUILD.json`
- 升级前：Strata **0.2.10 + engine 0.1.56**（v5.6），DavidAU Qwen3.8 IQ3_S，单 RTX 4090 48GB（sm_89），`"parallel": 4`（服务端补 `--batch 4`），`--mtp` + `--batch-mtp` + `--spec 4`
- 上一篇：[STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md)

---

## 0. 结论

**0.2.11 的头部卖点 FP16 weight cache 在本机没有启用（0 MiB），因此 4 并发没有拿到上游宣称的 +5.7%。** 已于 2026-11-06 10:08 切到 0.2.11 + 0.1.57，停机 47 s，验收 24/24（23 项旧断言 + 1 项新检查）。

**为什么没启用**：cache 的规则是「分配之后卡上还要留 3,072 MiB 空闲」。本机启动后空闲只有约 2,900 MiB，直接从 0 MiB 起步：

```
strata prefill: FP16 weight cache 0 MiB (the card keeps 3072 MiB free;
                STRATA_PF_W16_CACHE_MIB=0 turns it off)
```

上游只在 **RTX PRO 5000（更大显存）** 上测过，并明确写了未测 sm_89。我们的 48 GB 卡被
`--vram-reserve-mib 2560` + KV + 专家缓存挤到 ~2.9 GiB 余量，**没有 3 GiB 可以让出来**；要硬开就得砍
`conversation-cache-mib` 或专家缓存，收益（上游 4 路 +5.7%）抵不上长上下文/命中率损失，**决定不调**。

**那为什么还升**：4 条正中我们链路（4 slot + `conversation-cache 8192` + 流式工具调用 + 频繁重启引擎）的修复，见 §3。
**模型与模板一个字节没动**：仍 DavidAU abliterated IQ3_S + 合并模板 243,742 chars（换 Unsloth v3.1 会失去破限，见 §9）。

## 1. 0.2.11 与本机相关的变更

| 变更 | 本机 | 说明 |
|---|---|---|
| **FP16 weight cache**（Linux NVIDIA，`--parallel`/`--batch` ≥2 自动开） | ⚠️ 未启用 | 需要分配后仍留 3,072 MiB 空闲；我们只有 ~2,900 MiB。上游 4 路 175.1→185.1 t/s（×1.057）、8 路 ×1.066、1 路 ×1.038；输出 bit-for-bit 相同；关 `STRATA_PF_W16_CACHE_MIB=0` |
| **K-quant prompt kernels（MMQ）** 默认开 | ❌ 不适用 | 只对 Unsloth **UD-Q4/Q5/Q6_K_XL**；我们是 IQ3_S。关：`STRATA_MMQ_KQ=0` / `STRATA_NV_KQ_DEFAULT=0` |
| 裸 `--prefill auto` 上限 8192→32768 | ❌ 不适用 | 我们显式 `--prefill auto:32768`，上游明说显式值永不被改；关 `STRATA_NV_CHUNK=0` |
| **单卡 CUDA 的 CPU/upload share 改为按 CPU 实测**（ceiling 0.95，原 0.60） | ✅ 生效 | 日志 `PCIe share: start 0.15 from 32 pool workers (assumed 65 GB/s of expert bytes) and the 25.3 GB/s link (was 0.55)`；回 0.2.10：`STRATA_PCIE_AUTO_SPLIT=0`、`STRATA_PCIE_K_RULE=0`、`STRATA_PCIE_SHARE_MAX=0.60` |
| `strata components rollback` | — | 只把 engine 回 0.1.56，Control 仍 0.1.57；完整回滚用 `run_strata_0210.sh` |
| 双卡切分引擎、R9700 gfx12、Strix Halo、12–24 GB 分层裁剪、Arc、桌面 App、chats-as-files、IME、Pillow 13 | ❌ | 与单卡 4090 服务端无关 |

## 2. 命中的修复（升级的真正理由）

| 修复 | 为什么命中我们 |
|---|---|
| **#2518** 长对话里的分支追问会落到**别的会话的 slot**，占掉别人的上下文 | 我们正是 **4 slot** + `--conversation-cache-mib 8192`。上游模拟 4 会话 ×9,600 tok ×12 次交错追问：修复前保留前缀 96,000 tok（83%），修复后 115,200（100%） |
| **#5040 / #4701** 流式工具参数带闭合标签时，若文本分小块到达会**丢结尾**（chunk 1 与 7 时 216 字符只到 139） | DSH 的工具调用参数就是流式的；write 内容截断会直接坏文件 |
| **#4451**（#2892 同段）：① 引擎重启后**旧进程的请求仍挂在新进程的 slot** 上；② `vram()` / `session_file()` 可能取走**正在跑的请求**的控制行，导致请求在 212/421 token 处被切断 | 我们每次升级都重启引擎、且长期长输出。虽然 replica 我们没开，但这两个洞与服务端控制面相关 |
| **#4271（part 1）** 不再读的流式客户端不再占住引擎的 turn（60 s 无进展写入即按断连结束） | DSH 断线/取消时不再白占一个 slot；`STRATA_STREAM_WRITE_TIMEOUT_S`（0=关） |
| **#4479 / #2481** 服务端控制台输出走**有界队列**；**#3443** `ConvCacheLog.poll` 加锁；**#3961（缓解）** 无 end 事件的流按错误处理并结束引擎 | 长时间跑 + 4 slot 交错时的稳定性 |

## 3. 未命中（明确跳过）

DFlash2 相关（我们 4 并发必然选 MTP）、全卡切分引擎（要两张 sm86/sm120 或四张 R9700，且 ctx ≤131072）、
Unsloth Dynamic v3.1 的 App/Terminal 下载与打包（我们不换模型）、DGX Spark、Mac/Windows 桌面、IME、
chats-as-files、Pillow 13.0.0 安全更新、RTX 50 + CUDA 12 首启编译（我们 CUDA 13 + sm_89）。

## 4. 引擎包与源码树事实

- 资产 `strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz`，**720,473,300 B**，
  `sha256 cc58226bffc12216a5a2e6019dfdbf4896042500df132cc3fa3c72a2388a228b`（与 `SHA256SUMS.txt` 一致；比 0.1.56 的 667,699,126 B 多 52.8 MB）。
- `BUILD.json`：`{"version":"0.1.57","source":"f-0214-int 0bdcef4682c5","archs":[75,86,89,120],"vision_archs":"75;86;89;120","ptx":true,"cuda":"13.0","portable":true,"vision":"cuda13.0"}`——sm_89 在列，ptx 仍开。
- `strata` 本体 307,680,848 B（0.1.56 也是 307.7 MB），vision 124,767,240 B。
- 源码树 **502 文件**（0.2.10 是 496）；相对 0.2.10 复用 450、新下载 52，新增 `serve/nonblocking_out.py`。
  `serve/` 的 import 集合与 0.2.10 完全一致（**无新增第三方依赖**，现有 venv 直接可用）。
- `MIN_ENGINE` 仍 `(0, 1, 46)`（`setup.py@v0.2.11:271`），serve 侧无版本门。
- 11 个默认环境变量（9 并行栈 + 2 GDN）在 0.2.11 仍在 `serve/server.py:4250 PARALLEL_STACK_ENV` / `:4302 GDN_STACK_CUDA`，逐个在引擎 `/proc/<pid>/environ` 确认生效。

## 5. 升级步骤（实际执行）

```bash
# 1) 引擎：4 MiB 分片 × 172，20 线程续传，1023 s（GitHub 单流限速 ~45 KB/s）
sha256sum /tmp/s0211/strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz   # cc58226b…228b ✓

# 2) 源码树：tree API + raw 逐文件，502/502 blob sha 校验通过
#    SRC=Strata-0.2.10，DST=Strata-0.2.11，复用 450 个文件

# 3) 树与配置
mkdir -p strata/Strata-0.2.11/engine
tar -xzf /tmp/s0211/strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz -C strata/Strata-0.2.11/engine
# 配置只改 4 个字段：exe / cwd / log / vision.exe；32 个启动参数逐字未动

# 4) 切替
bash stop_strata.sh     # 9 s，显存 9 MiB
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$PWD /usr/bin/bash $PWD/run_strata_0211.sh
# 10:08:12 /health loaded=true（总计 47 s）

# 5) 验收 + 指针
bash test_strata_0211.sh        # PASS=24 FAIL=0
cp run_strata_0211.sh run_strata.sh
```

## 6. 实测（0.1.56 → 0.1.57，同机同参、同协议）

预填充用 `/tmp/s0211/prefill.py`：每个 prompt 加 24 字符随机前缀，确保**不吃 prefix cache**，取 3 次 TTFT 中位；
并发用 `strata/concurrent.py --levels 4 --repeats 3 --max-tokens 512`。

| 用例 | 0.1.56（升级前） | 0.1.57（升级后） | 差 |
|---|---|---|---|
| 6.25K tok prompt 预填充 | **5,143 t/s** | **5,060 t/s**（复测 5,046 / 5,069） | −1.6% |
| 12.7K tok prompt 预填充 | **5,530 t/s** | **5,326 t/s**（复测 5,504 / 5,513） | −1.7% ~ 持平 |
| 4 并发聚合（暖机） | 196.3 / **214.4** / 219.1 t/s | 211.6 / 212.6 / 212.3 t/s | −1% |
| 4 并发首波（引擎刚重启） | —（当时引擎已热） | 88.4 t/s（TTFT 10.0 s，专家缓存冷） | 冷启动现象，非回归 |
| 空载显存 | 46,505 / 49,140 MiB | 46,525 / 49,140 MiB（余 ~2,483 MiB） | 持平 |
| 专家缓存 | 33,607 MiB / 17,284 槽 | 33,605 MiB / 17,284 槽 | 持平 |
| `/props.build_info` | Strata 0.1.56 | **Strata 0.1.57** | — |
| 模板 | 243,742 chars | 243,742 chars | 未变 |
| `/metrics.engine` | 0.1.56 / mtp / 4 slot | **0.1.57 / mtp / 4 slot / pcie_frac 0.15** | — |

**口径**：±2% 是这套脚本的重复测量噪声（同一版本连测 3 次，6.2K 的极差 5,046–5,069）。上表差异都在这条线附近，
**不能宣称更快，也没有可见回退**；W16 cache 未启用是关键前提（§0）。

## 7. 回滚

```bash
bash stop_strata.sh
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory=$HOME/dsh_workspace/strata /usr/bin/bash $HOME/dsh_workspace/strata/run_strata_0210.sh   # 0.2.10 + 0.1.56
# 更老：run_strata_029.sh（0.2.9 + 0.1.55）、run_strata_027.sh（0.2.7 + 0.1.52）、run_strata_022.sh（0.2.2 + 0.1.47）
```

## 8. 遗留与注意

- **W16 cache 要真的用上，得有 ≥3 GiB 空闲**。本机现状 2,483–2,897 MiB。可试的路径只有砍
  `--conversation-cache-mib 8192` 或专家缓存；按上游 4 路 +5.7% 的幅度，**不值得**（长上下文与命中率更贵）。暂不调。
- **PCIe share 默认值变了**（0.15 起，原 0.55）：0.2.11 按 CPU 实测算 share。我们 12.7K prefill 从 5,530 → 5,326 t/s（−1.7%，复测又回到 5,504–5,513），
  无法与噪声区分；若将来长 prompt 明显变慢，先试 `STRATA_PCIE_AUTO_SPLIT=0` 回旧规则再判断。
- `strata: PLE table mlock failed (…raise ulimit -l)` **仍在**（0.1.47 → 0.1.57 都一样）：`ulimit -l` = 8 MiB = systemd 单元 `LimitMEMLOCK=8388608` 默认值。
  offload 走 PINNED，mlock 只影响 CPU-executor 路径，**锦上添花**，暂不改。
- 模型/模板没换：Unsloth Dynamic v3.1（2026-11-04）**没有 abliterated 版**；同尺寸 UD-IQ3_XXS 82 GB 只有 3.06 bpw（低于现 IQ3_S 的 3.44 bpw），
  更高精度的 UD-Q3_K_XL 要 90 GB（+7%），decode 大概率跌破现状。等 SC117/DavidAU 出 v3.1 abliterated 版再走「换模型 ⇒ 同时换 froggeric v22.5 模板」。
- 上游自述的已知限制与我们相关的一条：W16 cache 的收益只在「有显存余量」的卡上成立，sm_89 未测；我们没有复现它的 +5.7%，也不该复现。
