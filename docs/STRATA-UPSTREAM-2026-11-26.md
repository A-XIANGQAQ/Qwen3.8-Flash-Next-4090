# Strata 0.2.22 + engine 0.1.69 升级（2026-11-26，v6.4）

> 结论：**升级完成并保留，0.2.20 的 Q8-TC 规避已删除并验证**。0.2.21 修复了 `layer 0 never rang`（上游 `#6713`），本机不再需要 `STRATA_HC_Q8_TC=0` / `STRATA_HC_Q8_BF16TC=0`；去掉后 Q8 超连接的 Tensor Core 路径重新启用，24/24 验收通过，日志确认 `STRATA_HC_Q8=1` 且无崩溃。速度在噪声内（prefill +0.3~1.4%，4 并发 +0.7~1.9%）。
>
> 与 0.2.20 同型；模型仍是 unsloth UD-Q4_K_XL（huihui v2 破限档 1，pack v4），模板仍是 Sharp-naze v1.5（34,422 B / 34,386 字符），30 个引擎参数与 `parallel: 4` 未动。

## 1. 发布事实

| 项 | 值 |
|---|---|
| 发布 | **v0.2.22**，2026-11-25T18:52:49Z，非预发布；中间隔着 v0.2.21（2026-11-24T10:10:03Z） |
| 版本 | App/Terminal 0.2.22；`strata-engine`（Qwen）、CUDA `strata-inkling`、Control 都是 **0.1.69**；`strata-hy4` 与非 CUDA 的 Inkling 引擎停在 0.1.67（没变） |
| Linux x86_64 CUDA13 引擎 | `strata-engine-0.1.69-linux-x86_64-cuda-cuda13.tar.gz` **593,430,761 B**，sha256 `85e2cb1f1624269eeb42c21b2693400ee898625bccb5be994ff101d3474f083f` |
| Control 0.1.69 x86_64 | `strata-control-0.1.69-linux-x86_64.tar.gz` 94,774,506 B，sha256 `f93a0a1b8b7236df579d6a2725565c0af573b65e53359383de68ea40c0755495`（本机不装） |
| `BUILD.json` | `{"version":"0.1.69","source":"tools/packaging/build_cuda_linux.sh","archs":"75;86;89;120","cuda":"13.0","portable":true,"vision_archs":"75;86;89;120"}` |
| 二进制 | `strata` **188,949,592 B**、`strata-vision` **1,240,456 B**、`strata-split-2-sm86` 63,851,544 / `sm120` 69,336,960、内置 `libcublas*.so.13` |
| 源码树 | tag `v0.2.22`：9,277 项 / 8,485 blob；`terminal/{serve,strata_control,tools,strata_cli}` 过滤后 **593 文件**（相对 0.2.20 复用 542、变更 45、新增 6，逐个 blob sha 校验 593/593） |
| `MIN_ENGINE` | `terminal/setup.py:301` 仍是 `(0, 1, 46)`，没有新的版本门 |

## 2. 0.2.21 / 0.2.22 改了什么（与本机相关的部分）

- **`#6713`：Qwen “layer 0 never rang” 修复**（0.2.21）。这正是本机 0.2.20 上 `verify: layer 0 never rang (operation not supported on global/shared address space)` 的回归（我们当时记的是同型 issue `#6742`/`#6700`/`#6745`）。升级后**删掉两个规避变量**，Q8 超连接重新走 Tensor Core kernel。
- **`#6776`：Qwen prompt 读取的一处精度修复**（0.2.21）。只改了一个求和的顺序，官方称首 token 在检查过的 prompt 上全部相同、速度不变；理论上少数回答可能在几个 token 后与 0.2.20 不同。
- **`#7070`：开 MTP 时的采样恢复**（0.2.22）。之前开 MTP 后采样设置会被丢掉、变成贪心；现在不显式给采样时按模型卡 `temperature 1.0 / top_p 0.95 / top_k 20`，显式给了则照给。DSH/Cherry 都显式带采样，这条对生产请求的影响只是“回答不再被悄悄变成贪心”。
- **`#7044`：对话缓存自动定容**（0.2.22）。当启动器没有显式给 `--conversation-cache-mib` 时，Control 按空闲内存的 40% 推算；本机直跑 serve 且显式传 `--conversation-cache-mib 8192`，**我们自己传的值优先**，不受影响。
- **agent 会话每轮复用 prompt**（0.2.21，全后端）。长会话不再每轮重读整个 prompt；对 DSH 这类多轮 agent 有潜在收益。
- **`#7132`：长自定义工具调用不再被空闲代理切断**（0.2.22）。`/v1/responses` 静默超过 10 s 就发 keep-alive；DSH 走 `/v1/messages`，不直接受益，但多一种路由更稳。
- **`#6905` 长会话崩溃修复**、**引擎连续崩 3 次后停止重试并明说**（0.2.21）—— 后者会让故障更早暴露，不会再无限自重启。
- **`#7099`/`#7101`/`#7130`**：更新后不再暂停 agent、Ask again/Edit/Branch 保留工作目录、app 内置服务器只接受自己页面的请求（含 `STRATA_ALLOWED_HOSTS`）。都是 app/Terminal 层，直跑 serve 不受影响。
- **不适用本机**：Inkling 2 Mini 的两卡修正 `#7093` 与提速、HY-4 的 SQ-16/SQ-24、两卡层切分的 RAM 预留开关 `#7089`、WDDM 回归、AMD RX 7900 的 SQ-20 回退（7%）等。
- **没有**我们关心的工具调用/思考泄漏修复（`#4751`/`#4077`/`#4949` 仍 open）。

## 3. 兼容性核对（升级前做的只读检查）

- 30 个引擎参数原样被接受：生产日志里的 `engine started: … --serve … --max-context 262144 --kv k8v4 --kv-resident 32768 --conversation-cache-mib 8192 --batch-mtp --vision --vram-reserve-mib 2560` 与 0.2.20 逐项一致（30 项；`--resident-budget-gib` 早在 v5.9 就删了）。`--vram-reserve-mib` 仍不在帮助文本但被接受（0.2.16 起就这样）。
- `strata-vision` CLI 未变；新 `strata-vision` 只有 1,240,456 B（0.1.60 起图片读取器自有）。
- pack 仍是 **v4**，引擎只拒绝比 v2 更老的 pack（`native_experts.txt has no GGUF offsets`），不需要重建。
- `check_model()` 与 `aliases` 逻辑与 0.2.20 相同（0.2.22 只是把 `#6778` 的 size-name 表加了 `sq-15`/`gyro-s` 等），所以 config 里的
  `aliases: ["Qwen3.8-Flash-Next", "qwen3.8-flash-next-q4kxl-ablit"]` 继续保留，`/v1/chat/completions` 用旧名实测 200、`model` 回显 `Qwen3.8-Flash-Next`。
- `serve/` 仍是包结构（`serve/server_parts/*`），`terminal/serve/server.py` 的 `ROOT = parents[1]` 在平铺部署下仍指向 `Strata-0.2.22/`。
- 新增文件 6 个：`strata_control/conversation_cache_size.py`（上面 `#7044`）、`strata_control/inkling_probe.py`、`inkling_probe_prompts.py`、`tools/check_inkling_prefix.py`、两个 `conftest.py`。

## 4. 部署与切替

- 目录：`strata/Strata-0.2.22/{serve,tools,strata_control,strata_cli}` 平铺 + `engine/`（解包 0.1.69）。
- config `strata/strata-unsloth-q4kxl-ablit-l1-0222.json`：相对 0220 只改 4 个路径字段（`exe`/`cwd`/`log`/`vision.exe`），**并删掉 `env.STRATA_HC_Q8_TC` 与 `env.STRATA_HC_Q8_BF16TC`**；`STRATA_ROUTE_TAIL_SKIP=0`、`undeclared_tool_calls=call`、`aliases`、30 参数、`parallel:4` 都不动。
- 启动器 `strata/run_strata_q4l1_0222.sh`（已 `chmod +x`）；`strata/run_strata.sh` 改为 exec 它。
- 切替时间线（2026-11-26）：

| 时刻 | 事件 |
|---|---|
| 09:33:07 | `stop_strata.sh` 干净停止（显存 9 MiB、8000 无监听） |
| 09:33:14 | `systemd-run` 起 0222（unit active） |
| 09:33:42 | `/health` loaded，`build_info = Strata 0.1.69`（**28 s**，页缓存命中） |
| 09:45:52 | 去掉两个规避变量后重启，09:46:20 就绪（28 s） |

- 首次验收 **23/24**：`serve_tree` 失败——生成 0222 验收脚本时 sed 只替换了 `0.2.20`→`0.2.22`，没替换转义写法 `Strata-0\.2\.20`（同一个坑在 0216 也踩过）。改成 `Strata-0\.2\.22` 后 24/24。验收脚本本身没有版本逻辑错误。

## 5. 规避删除验证（本次的重点）

删除前 config 里有两个变量（0.2.20 的 workaround）。删除后：

- 引擎进程的 `/proc/<pid>/environ` 里只剩 `STRATA_ROUTE_TAIL_SKIP=0`；
- 启动日志重新出现 `strata: STRATA_HC_Q8=1: 0.63 GiB of Q8_0 hyper-connection projections for the verify read`，且**没有任何** `never rang` / `cudaErrorInvalidAddressSpace`；
- 验收 **24/24**（普通对话、effort low、视觉、4 路并发冒烟、MTP、PLE、prompt kernel、显存全过）；
- 单流、4 并发、prefill 均正常 ⇒ `#6713` 的修复在本机 sm_89 上确认生效。

## 6. 同协议实测（0.2.20 + 0.1.66 → 0.2.22 + 0.1.69）

机器、模型、模板、参数、测量脚本完全相同（`/tmp/s0214/bench.sh`，`prefill_bench.py` + `concurrent.py` 512 tok × 3 轮）。

| 项 | 0.2.20 + 0.1.66 | 0.2.22 + 0.1.69 | 变化 |
|---|---:|---:|---:|
| prefill 6,265 tok | 2,738.5 t/s | **2,776.8 t/s** | +1.4% |
| prefill 13,032 tok | 4,986.2 t/s | **5,003.1 t/s** | +0.3% |
| 4 路聚合 r1/r2/r3 | 204.87 / 200.38 / 199.72 | **207.55 / 201.85 / 203.61** | +1.3% / +0.7% / +1.9% |
| 每路 decode 中位 | 52.76 / 52.41 / 52.52 | 53.32 / 52.54 / 52.87 | 噪声内 |
| TTFT / 墙钟 | 0.202 / 0.290 / 0.397 s；10.0 / 10.22 / 10.25 s | 0.217 / 0.305 / 0.384 s；9.87 / 10.15 / 10.06 s | 持平 |
| 单流 decode（effort off） | 125.6 t/s | 130.4 / 129.8 t/s | 噪声内 |
| 单流 decode（effort low） | 134.3 t/s | 133.7 / 126.1 t/s | 噪声内 |

MTP 接受率：off 49–52%、low 43–51%（0.2.20 同量级）。结论：**这次是修复版，不是提速版；没有可见回退**。

## 7. 回滚

- 保留 `strata/Strata-0.2.20`（源码树 + engine 0.1.66）与 `run_strata_q4l1_0220.sh`；回滚 = `bash stop_strata.sh` → `systemd-run … ./run_strata_q4l1_0220.sh`，并把 config 0220 的 `env` 两个 Q8-TC 变量加回去。
- 更早的 `Strata-0.2.16`（0.1.62）也在，但它的 config 不带 Q8 规避、可直接跑。

## 8. 复现要点

- 源码树走 `cdn.jsdelivr.net/gh/Niko1221/Strata@v0.2.22/terminal/…`（经 `127.0.0.1:7890` 代理）比 raw.githubusercontent 快且稳；本次 593 个文件里 542 个可直接按 blob sha 从 0.2.20 复用，只下载 51 个；代理偶发 `TLS connect error`，用 raw 回退重试即可。
- 验收：`bash scripts/accept_q4kxl_0222.sh`（24 项）。
- 手工确认：`/proc/<pid>/cwd` 必须是 `Strata-0.2.22`；`grep STRATA_HC_Q8=1` 日志应有、`never rang` 应无；`/v1/chat/completions` 用 `Qwen3.8-Flash-Next` 旧名应 200。
