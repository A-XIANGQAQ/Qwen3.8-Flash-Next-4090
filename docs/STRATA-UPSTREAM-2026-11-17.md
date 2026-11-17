# Strata 0.2.15：serve 层的两个修复，引擎没动

> 2026-11-17 升级记录。生产：0.2.14 → **0.2.15**（服务端源码树），engine 仍 **0.1.60**（同一批文件，软链复用）。
> 模型 unsloth UD-Q4_K_XL + huihui v2 消融档 1、模板 Sharp-naze v1.5、30 个启动参数全部不变。

## 0. 结论

- **0.2.15 是 app / Terminal / Control 版本**：app 与 Terminal 0.2.15、Control 0.1.61，**engines stay at 0.1.60**（发布说明明写「the same files as in 0.2.14，从 0.2.14 release 下载」）。资产清单里已经没有 x86_64 CUDA 引擎包，只有 aarch64 0.1.59。
- 我们直跑 `serve/server.py`，跟 app/Terminal 那堆修复无关；真正落到本机的是 **两处 serve 层修复**：
  1. `serve/frontend.py` **#5979**：声明的布尔型工具参数，模型写成 Python 拼法 `True` / `False` 时现在解析成布尔；以前会掉进 JSON 解析失败分支。这是我们关心的工具调用保真度。
  2. `serve/server.py` **#6055**：没传 `--shutdown-timeout` 时也调用 `svc.begin_drain()`；关闭期间新到的请求返回 **503 `server_draining`**，不再被接进来切掉、回 0 token。对每次重启的那几十秒有用。
- 其余四目录改动 27 个文件，大多是 `strata_cli` / `strata_control`（我们不用 Control），新增两个 `serve/test_*.py` 是纯测试。
- 实测：prefill 6.2K 2,795 → **2,801 t/s**、12.8K 4,783 → **4,809 t/s**、4 路 175.0/187.1/185.6 → **182.7/184.9/185.6**，全部在 ±0.5% 噪声内——符合「引擎逐字节不变」的预期。

## 1. 资产与源码树

| 项 | 值 |
|---|---|
| 发布 | 2026-11-16T09:22:29Z；feed sequence 127 |
| engine | **无新包**；沿用 0.2.14 的 `strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz`（672,845,287 B，sha256 `a72c194c8f6de96629ed53079b5e344ab81223d45667a53f6a6d79398895aa50`） |
| Control | `strata-control-0.1.61-linux-x86_64.tar.gz` 97,550,582 B，sha256 `e780b9748bcd9e8427e4cf93f6ef79535173898c83ffbff8833f0ad9385bb36d`（本机不装） |
| Intel Arc / DGX Spark | 引擎停在 0.1.59 |
| 源码树 | tag v0.2.15：8,090 项 / 7,368 blob；四目录 serve 203、strata_control 87、tools 506、strata_cli 38；过滤 `fixtures` / `test_*` / `.pyc` 后 **554 文件**（复用 535、新下 19、18 个变更 .py 全部 `py_compile` 通过） |
| 与 0.2.14 的差异 | 四目录新增 2（`serve/test_bool_args.py`、`serve/test_web_speculation.py`，均为测试被过滤）、删除 0、变更 27；实体变更只有 `serve/server.py`、`serve/frontend.py` 两处逻辑 |
| 部署方式 | `strata/Strata-0.2.15/engine` → 软链 `../Strata-0.2.14/engine`；config 只改 `exe` / `cwd` / `log` / `vision.exe` 四个字段 |

## 2. serve 的两处 diff

`frontend.py`（#5979）：

```python
boolean = declared == "boolean" or (isinstance(declared, list) and "boolean" in declared and "string" not in declared)
if boolean and value.strip().lower() in ("true", "false"):
    return value.strip().lower() == "true"   # 模型写成 Python 拼法的布尔值
```

`server.py`（#6055）：`--shutdown-timeout` 未给时，在关闭引擎前补 `svc.begin_drain()`，让排水期的请求得到 503 而不是 0 token。

## 3. 切替记录（无事故）

| 时间 | 事件 |
|---|---|
| 14:39:37 | 前置检查：启动器可执行（这次有 `-rwxrwxr-x`）、config 存在、venv python 存在；`live.running+queued=0` |
| 14:39:37 | `stop_strata.sh`：显存 9 MiB、8000 无监听 |
| 14:39:43 | `systemd-run --user --unit=strata-prod --collect …` 起 `run_strata_q4l1_0215.sh`；单元 `active` |
| 14:40:12 | `/health` loaded=true，**就绪 28 s**（页缓存命中） |
| 14:40:12 | `build_info = Strata 0.1.60`；进程 cwd = `Strata-0.2.15`（0.2.15 与 0.2.14 的 build_info 相同，只能看 cwd） |

> 11-16 的教训已经固化进切替脚本：`[ -x "$LAUNCHER" ] || exit 1`，并且 `systemd-run` 后先查 `ActiveState` 再等 `/health`。
> 另一个小坑：启动器是 `cd Strata-0.2.15 && python ./serve/server.py`，命令行里是相对路径，**不能用 `pgrep -f Strata-0.2.15/serve/server.py` 判断树**，要读 `/proc/<pid>/cwd`。

## 4. 验收（24/24）

`qwen4090-repo/scripts/accept_q4kxl_0215.sh`：23 项继承自 0214 脚本，新增第 24 项 `serve_tree`（读进程 cwd 必须是 `Strata-0.2.15`）。关键项：

- `/health`：`model=qwen3.8-flash-next-q4kxl-ablit-l1`、loaded、ctx 262144、images true。
- `/props`：`build_info "Strata 0.1.60"`、模板 `len(chat_template)=34386`（Sharp-naze v1.5）。
- `/metrics.engine`：`engine 0.1.60`、`drafting mtp`、`batch_slots 4`、`expert_slots 11049`、`pcie_frac 0.15`、`arena_mib 73450`。
- 普通对话 / `reasoning_effort=low` / 64×64 视觉都出 token；4 路冒烟 `ok=4 tok=509 agg=92.6`。
- 日志：`PLE table loaded (not locked)`、`--ple-gguf`、引擎路径含 `Strata-0.2.15/engine/strata --serve`、prompt 内核仍是 48/48 层、`strata mtp: draft layer loaded`。
- 显存 46,329 / 49,140 MiB。

## 5. 同协议实测

| 项 | 0.2.14（11-16） | 0.2.15（11-17） | 差 |
|---|---:|---:|---|
| prefill 6,264 tok | 2,795 t/s | **2,801 t/s** | +0.2% |
| prefill 13,032 tok | 4,783 t/s | **4,809 t/s** | +0.5% |
| 4 路聚合（512 tok × 3） | 175.0 / 187.1 / 185.6 | **182.7 / 184.9 / 185.6** | 噪声内 |
| 每路 decode 中位 | 45.4 / 48.8 / 48.6 | 47.3 / 48.2 / 48.3 | 噪声内 |

方法同 v6.0：prefill 用随机前缀破 prefix cache、取 3 次 TTFT 中位；并发 4 路 × 512 token × 3 轮，测前确认 `running+queued=0`。

## 6. 回滚

- 回 0.2.14：`strata/run_strata.sh` 改指 `run_strata_q4l1_0214.sh`（树与引擎都还在）。
- 回 0.2.12 + 0.1.58：改指 `run_strata_q4l1.sh`。
- 0.2.15 的 `engine` 是软链，删 `Strata-0.2.15` 不会影响 0.2.14 的引擎文件。
