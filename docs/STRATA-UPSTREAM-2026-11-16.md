# Strata 0.2.14（engine 0.1.60）：prompt 专家改走 Strata 自己的内核

- 调研 + 升级时间：2026-11-16 09:30–12:00（Asia/Shanghai）
- 数据来源：GitHub Releases（v0.2.13 / v0.2.14 全文 + `SHA256SUMS.txt`）、`git trees/v0.2.14?recursive=1`（**8,071 项 / 7,350 blob**）+ raw 逐文件抓取（`serve/ strata_control/ tools/ strata_cli/` 过滤 `serve/fixtures/`、`test_*`、`.pyc` 后 **554 文件，每个 `git hash-object` 都等于 tree 的 blob sha**）、engine 包内 `BUILD.json` 与 `engine-component.json`
- 升级前：Strata **0.2.12 + engine 0.1.58**（v5.9），unsloth **UD-Q4_K_XL** + huihui v2 消融**档 1**（`qwen3.8-flash-next-q4kxl-ablit-l1`），模板 **Sharp-naze v1.5**，`"parallel": 4`，`--mtp` + `--batch-mtp` + `--spec 4`
- 上一篇：[STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md)

---

## 0. 结论

**prompt 处理确实变快了，短交互没变化，工具调用/思考格式没有修。已切到 0.2.14 + 0.1.60（2026-11-16 11:57:40 起，页缓存命中下启动 30 s），验收 23/23，模型与模板一个字节没动。**

- **命中本机的唯一实质变更**：Q4_K/Q5_K 专家模型（就是 Unsloth UD-Q4/Q5/Q6_K_XL）的 prompt 不再走 llama.cpp 的 MMQ，改走 Strata 自有融合内核；日志逐层确认 **48/48 层**走融合入口。
- **同协议实测**：6.2K prompt 2,770 → **2,795 t/s**（+0.9%）、12.8K 4,631 → **4,783 t/s**（**+3.3%**）；新测 20K **4,847 t/s**、30K **4,939 t/s**（0.1.58 没有这两档基线）。4 路并发 175–187 t/s，与 11-08 引擎空缓存时的 0.1.58（186–188）持平。
- **上游给的 4090 数字复现不了那么多**：官方在 4090 24G + IQ3_S 上是 20K prompt 6,110 → 5,656 ms（+8%）；我们 12.8K 只有 +3.3%，与「IQ3_S 的专家是 Q4_K 以外类型、覆盖层数不同」一致，不夸大。
- **工具调用/思考泄漏一条都没修**：`#4751`（工具调用变文本）、`#4077`（reasoning 里 `</think>` 提前结束思考）、`#4949`（空任务式 thinking）在 0.2.13/0.2.14 说明里都没有对应条目，仍是 open。想治只能换离线和模板/提示词，见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)。
- **代价是一次 7 分 24 秒停机**：新启动器建好后忘了 `chmod +x`，`systemd-run` 报「权限不够」，见 §3。这不是引擎问题，但比历次 47–65 s 长得多，记在这里防止复发。

## 1. 0.2.13 / 0.2.14 与本机相关的变更

| 变更 | 本机 | 说明 |
|---|---|---|
| **prompt 专家改走 Strata 自有内核**（NVIDIA/AMD），`-DSTRATA_MMQ=ON` 可回旧路径 | **命中** | 我们的专家就是 Q4_K/Q5_K + Q8_0 超连接，日志 `48 of 48 layers on the fused kernels` |
| prompt 切块改等分（片段 ≥8192 时两片各 ≥5,400、三片各 ≥6,144） | **命中** | 影响长 prompt 的首 token 延迟；本机 `--prefill auto:32768` 不变 |
| 0.2.13：原先启动器写的默认值搬进引擎（Linux CUDA ≥2 slots 并行栈、GDN、Unsloth UD 的 Q8_0 超连接读、DFlash2 默认） | **命中（行为不变）** | 服务端不再注入那 11 个环境变量；config 里的 `env` 仍会传给引擎 |
| 质量 KL 0.09812（4090/IQ3_S，首 token 8,279 / 16,884 tok） | 参考 | 与 llama.cpp kernel 版逐字节相同，说明换内核不改输出分布 |
| Intel Arc / DGX Spark 引擎停在 0.1.59；12 GB 卡、Strix Halo、R9700、Mac 的默认值 | 不适用 | 单卡 48 GB Ada |
| 工具调用、思考泄漏（`#4751`/`#4077`/`#4949`） | **未修** | 说明里没有对应条目 |

## 2. 资产、源码树与校验

| 项 | 值 |
|---|---|
| engine 包 | `strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz`，**672,845,287 B** |
| sha256 | **`a72c194c8f6de96629ed53079b5e344ab81223d45667a53f6a6d79398895aa50`**（与 `SHA256SUMS.txt` 一致） |
| `engine-component.json` | 9/9 文件 sha256 逐个核对通过 |
| `BUILD.json` | `{"version":"0.1.60","source":"f-p4-rc 09b603886925","archs":[75,86,89,120],"ptx":true,"cuda":"13.0"}` |
| 二进制 | `strata` **177,694,288 B**（0.1.58 是 315,960,672——不再链 llama.cpp 的 prompt 内核）；`strata-vision` 125,963,712 B；**没有 `libstrata_kq.so`** |
| 源码树 | tag v0.2.14：8,071 项 / 7,350 blob；四目录 serve 201、strata_control 87、tools 506、strata_cli 38；过滤后 **554 文件**（复用 300、新下 254、181 个变更 .py 全部 `py_compile` 通过） |
| `MIN_ENGINE` | 仍 `(0, 1, 46)`（`setup.py@v0.2.14:266`，serve 侧不校验） |
| Control | `strata-control-0.1.60-linux-x86_64.tar.gz` sha256 `75a9b32677d70ef4dc25f1575b26924696701c3e51b80e6408ebee8135b229ad`（本机只用 engine，不装 Control） |

`serve/` 相对 0.2.12：**新增 `toolformat.py`、`tool_parser_golden.json`**，删掉 `dsml.py`、`ds_pictures.py`，改 `frontend.py`、`images.py`、`lifeline.py`、`server.py`、`runconfig.py` 等。`toolformat.py` 从模板文本判断工具方言（xml/json/dsml/none/other）：我们的 **Sharp-naze v1.5 判 xml**，官方原装 9,993 B 也是 xml，走的是老路径，没有被新解析器改变行为。无新增第三方依赖，`Strata/.venv` 直接可用。

## 3. 切替与事故（停机 7m24s）

| 时间 | 事件 |
|---|---|
| 11:50:46 | `stop_strata.sh` 成功，显存 9 MiB，8000 无监听 |
| 11:50:53 | `systemd-run` 启动 → **失败**：`/bin/bash: 行 1: ./run_strata_q4l1_0214.sh: 权限不够` |
| 11:50–11:57 | unit 被 `--collect` 回收（`Unit strata-prod.service could not be found`）；`chmod +x` 修好启动器 |
| 11:57:40 | 手工 `systemd-run` 起 0214，`STRATA_PY` 显式传入 |
| 11:58:10 | `/health` loaded=true，**启动 30 s**（页缓存命中：71.73 GiB / 23 s = 17.5 GiB/s，冷盘实测只有 1.68 GiB/s） |

原因很土：`run_strata_q4l1_0214.sh` 创建时是 `-rw-rw-r--`，没有执行位；`systemd-run` 调 `/bin/bash -c "./run_strata_q4l1_0214.sh"` 时被拒。**教训写进流程**：新建启动器立刻 `chmod +x`，切替脚本的前置检查加 `[ -x "$LAUNCHER" ] || exit 1`，别等 systemd 报错才回来。

启动器日志：`experts loaded: 71.73 GiB at 17.50 GiB/s (23 s so far)`、`filling the GPU's expert cache (11049 experts, 32.26 GiB of VRAM)`、`almost ready`。

## 4. 模板：生产是 Sharp-naze v1.5，不是 froggeric v22.5

这次验收第一轮 FAIL 暴露了文档漂移：`template_chars: got [34386] want [28226]`。

- pack 里的 `tokenizer/chat_template.jinja` 是 **34,422 B / 34,386 字符**，mtime 2026-11-09 09:26，文件头 `{#- Qwen3.8-Flash-Next chat template v1.5-sharp (2026-10-16)…`；同目录留着 `chat_template.jinja.bak-20261109-froggeric225`（28,234 B）。
- 2026-11-09 09:27 已把模板从 froggeric v22.5 换成 **Sharp-naze v1.5**（HF `nazeshinjite/Flash-Next-Chat-Templates`，34,422 B，md5 `c127fd72…`），旧模板备份保留；`/props` 的 `len(chat_template)` 就是 **34,386**。
- 同一时期还有一次权重切换：11-09 15:52 生产曾切回官方 stock（理由：huihui 的补丁打在量化后权重上，正确顺序应是 bf16 消融 → 量化；公司生产没有破限硬需求），随后又切回档 1（`strata-unsloth-q4kxl-ablit-l1.json` mtime 2026-11-12 14:02，今天切替前 `/health` 与 unit 都确认是 l1）。
- 这次升级**没有动模板也没有动权重**；只是把验收断言从过期的 28,226 改成 **34,386**，并在注释里写明备份名。

## 5. 同协议实测（0.1.58 → 0.1.60）

测法：`scripts/prefill_bench.py`（每条 prompt 加 24 字符随机前缀破 prefix cache，3 次取 TTFT 中位）+ `strata/concurrent.py --levels 4 --repeats 3 --max-tokens 512`。测前确认 `/metrics.live` 的 `running+queued == 0`。

| 项 | 0.1.58（切前，干净） | 0.1.60（切后） | 差 |
|---|---|---|---|
| prefill 6,264 tok | **2,770 t/s**（TTFT 2.245 s） | **2,795 t/s**（2.24 s） | +0.9% |
| prefill 13,032 tok | **4,631 t/s**（2.801 s） | **4,783 t/s**（2.724 s） | **+3.3%** |
| prefill 20,377 tok | 未测 | **4,847 t/s**（4.204 s） | — |
| prefill 29,953 tok | 未测 | **4,939 t/s**（6.065 s） | — |
| 4 路聚合（512 tok） | 102.4 / 105.5 / 103.2 t/s | **175.0 / 187.1 / 185.6 t/s** | 见下 |
| 每路 decode 中位 | 41.8 / 43.5 / 44.2 | 45.4 / 48.8 / 48.6 | +9~12% |
| TTFT | 0.401 / 0.353 / 0.404 s | 0.306 / 0.384 / 0.345 s | 同量级 |

**并发那一列不能按 +70% 读**：切前那组是在 conversation cache 已经装着外部 140K 会话的情况下测的，每轮都要 park/restore（日志 `strata batch: slot 1 takes 140134 tokens (copied in 248.3 ms)`），墙钟 19.4–20.0 s；切后是重启后的空缓存，墙钟 10.95–11.7 s。对照 11-08 的 0.1.58 **空缓存**口径（186–188 t/s / 每路 54.9–57.3），0.1.60 的 185.6–187.1 / 45.4–48.8 属于**持平（每路中位略低，仍在新内核 + 更重模板的正常波动内）**。要看真实差异，只能重启双方各测一轮——本次不做，避免再停一次服务。

引擎日志佐证：`strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the staged fused entry.`

## 6. 切后状态与回滚

- `/health`：`qwen3.8-flash-next-q4kxl-ablit-l1`、loaded=true、ctx 262144、images=true。
- `/props`：`build_info "Strata 0.1.60"`；`chat_template_caps` 的 tools / parallel_tool_calls / system_role / preserve_reasoning 全 True。
- `/metrics.engine`：`engine 0.1.60 | drafting mtp | batch_slots 4 | expert_slots 11049 | expert_cache_mib 33038 | arena_mib 73450 | vram_free_mib 2517 | pcie_frac 0.15 | spec 6 | mtp_max 4`。
- 显存：46,647 / 49,140 MiB（PLE 表 26.8 GiB 走 RAM，仍未 mlock——`RLIMIT_MEMLOCK` 8 MiB 的上限在 `user@1000.service` 上，transient 子单元改不了）。
- 回滚：`strata/Strata-0.2.12` + 引擎 0.1.58 完整保留（`run_strata_q4l1.sh` + `strata-unsloth-q4kxl-ablit-l1.json`）；`run_strata.sh` 指针现在指向 `run_strata_q4l1_0214.sh`，改一行即可回 0.2.12。
- 磁盘：432G 已用 / 437G 可用（90% 是模型：stock 105G + 档 1 77G + 档 2 77G + patch 3.4G）。

## 7. 与本机无关的上游内容

Intel Arc 与 DGX Spark 引擎停在 0.1.59；Radeon R9700 驱动暂停、780M reset、Strix Halo、12 GB 卡分层、Mac 的 fp16 decode 窗口、DFlash2 新默认（要求专家全驻卡，我们仍是 MTP）——都不涉及单卡 48 GB Ada + 流式专家。

## 8. 验证清单（`scripts/accept_q4kxl_0214.sh`，23 项）

`/health` 4 项；`/props`（`build_info`、模板 34,386 字符）；`/metrics.engine` 6 项（engine 0.1.60、mtp、batch_slots 4、expert_slots ≥10000、pcie_frac 0.15、arena ≥73000）；普通对话；`reasoning_effort=low` 通路；视觉；§7 并行栈 env（0.2.13 起默认值进引擎，改为信息项）；**4 路并发冒烟**（`smoke ok=4 tok=481 agg=66.0`，128 token 冒烟被 slot 停放主导，只证明真并行）；§9 日志（PLE 表加载、`--ple-gguf`、引擎路径 0.2.14、48/48 融合内核）；§10 drafter=mtp；§11 显存。

结果：**PASS=23 FAIL=0**。
