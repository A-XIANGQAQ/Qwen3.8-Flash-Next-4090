# Strata 0.2.16 + engine 0.1.62（v6.2，2026-11-19）

上游 release：<https://github.com/Niko1221/Strata/releases/tag/v0.2.16>（发布 2026-11-18T21:57:34Z，feed 128）。

**结论：已升级并保留。** 这次是**引擎换代**（0.1.60 → 0.1.62，NVIDIA/AMD/Intel Arc/Mac 全部改跑 Strata 自有代码，
不再用 ggml / llama.cpp 的矩阵内核）+ **目录布局重排**（`serve/` → `terminal/serve/`）。本机实测：
prefill 在噪声内略低 1.3–1.5%，4 并发持平，验收 24/24，工具调用正常。

## 1. 资产

| 文件 | 字节 | sha256 |
|---|---:|---|
| `strata-engine-0.1.62-linux-x86_64-cuda-cuda13.tar.gz` | 580,095,243 | `6723e1b6b6042bfe1d368723ebf873afb0fa65f4f1908d948dd865098f6cfb9e` |
| `strata-control-0.1.62-linux-x86_64.tar.gz`（本机不装） | 94,383,559 | `045b6a7d7fbfbddab814794db5763d6244c887a394403b9fe0e8a4714d9bef8c` |
| `strata-engine-0.1.62-linux-x86_64-cuda-cuda12.tar.gz` | 767,838,500 | — |

包内：`strata` 178,301,344 B、`strata-vision` **1,219,696 B**（0.1.60 是 125,963,712 B）、
`strata-split-2-sm86/sm120`、`libcublas*.so.13`。
`BUILD.json`：

```json
{"version": "0.1.62", "source": "tools/packaging/build_cuda_linux.sh",
 "archs": "75;86;89;120", "cuda": "13.0", "portable": true, "vision_archs": "75;86;89;120"}
```

`./strata --version` → `strata 0.1.62`；`./strata-vision` 的 CLI 未变
（`--mmproj <mmproj.gguf> [--model <model.gguf>] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]`）。

## 2. 目录布局变了

上游把顶层拆成 `engine/`、`app/`、`terminal/`、`docs/`、`tools/`；服务端在 `terminal/serve/server.py`，
工具在 `terminal/tools/`，控制层 `terminal/strata_control/`，CLI `terminal/strata_cli/`。

本机部署选择**平铺**：只取仓库的 `terminal/{serve,tools,strata_control,strata_cli}`，直接放在
`strata/Strata-0.2.16/` 下（`terminal/` 这一层去掉）。原因：`terminal/serve/server.py` 用
`ROOT = Path(__file__).resolve().parents[1]` 把 `ROOT` 和 `ROOT/tools` 加进 `sys.path`，放在
`Strata-0.2.16/serve/` 时 `ROOT` 正好是 `Strata-0.2.16`，与旧布局的启动方式一致；启动器里
`cd Strata-0.2.16 && python ./serve/server.py` 不用改写法。

- **不要跑 `setup.py` / `update.sh` / `--update`**：会把设置与引擎搬进 `terminal/`、把模型移到克隆旁的
  `Strata-data`，还会覆盖我们手调的 config 与模板。
- 引擎仍是可执行的 `Strata-0.2.16/engine/strata`，config 只改 `exe/cwd/log/vision.exe` 四个路径字段。

## 3. 兼容性核对（切前做完）

| 项 | 结果 |
|---|---|
| 30 个引擎参数 | 全部仍被识别；`--vram-reserve-mib` 不在 `--help` 文本里，但二进制里有该参数，实测不报 unknown argument |
| `strata-vision` CLI | 未变（视觉冒烟通过） |
| 服务端 CLI | 仍是 `--engine {mock,strata} --config --port`；config 键未变 |
| `undeclared_tool_calls` | 仍在（默认 `"text"`；我们的 config 是 `"call"`） |
| pack 格式 | **v4 可直接读**，不必重建；v5 只用于 Hadamard 折叠专家的新 pack |
| GGUF 库 | `tools/iq_pack.py` 改用内置 `tools/strata_gguf.py`；`STRATA_GGUF_PY` 只为兼容旧树保留（最多两个版本） |
| Python 依赖 | `requirements.txt` 固定 `pillow 13.0.0`（py≥3.11）。本机 venv 是 **Pillow 12.3.0**，本次升级没有升级它（服务端与视觉路径不依赖该版本差异），复现全新环境时应按 `requirements.txt` 装 |

源码树：tag v0.2.16，8,428 项 / 7,693 blob；本机只取四个目录（过滤 `fixtures/`、`test_*`、`__pycache__`），
**569/569** 文件按 git blob sha 逐个核对通过（复用 0.2.15 的 414 个，新下 155 个）。

## 4. 切替时间线（2026-11-19）

| 时刻 | 事件 |
|---|---|
| 10:08:19 | 等在飞请求结束 → `stop_strata.sh`（显存 9 MiB、8000 无监听） |
| 10:08:26 | `systemd-run --user --unit=strata-prod … ./run_strata_q4l1_0216.sh` |
| 10:08:54 | `/health` `loaded=true`（**28 s**，页缓存命中） |
| 10:08:54 | `/props.build_info = "Strata 0.1.62"`；模板未动（34,386 字符 / Sharp-naze v1.5） |
| 10:09 | 验收 `scripts/accept_q4kxl_0216.sh` → 24/24 |

- 切替脚本前置检查 `[ -x "$LAUNCHER" ]`（记 2026-11-16 漏 `chmod +x` 停机 7m24s 的教训），本次无事故。
- 首次验收曾报 1 项 FAIL（`引擎路径是 0.2.16 = 0`）：原因不是服务，而是脚本默认日志路径还写着
  `…-0215.log`（旧日志里没有 0.2.16 的启动行）。修正日志路径后 24/24。
- 生产进程：`/proc/<pid>/cwd` = `…/strata/Strata-0.2.16`（不要用 `pgrep -f Strata-0.2.16/serve/server.py`，
  命令行里只有相对路径——见 TROUBLESHOOTING #51）。

## 5. 实测（同机同协议）

prefill：每条 prompt 加 24 字符随机前缀破 prefix cache，3 次取中位。并发：4 路 × 512 token × 3 轮。

| 项 | 0.2.15 + 0.1.60 | 0.2.16 + 0.1.62 |
|---|---:|---:|
| prefill 6.2K | 2,801 t/s | **2,760 t/s**（−1.5%） |
| prefill 12.8K | 4,809 t/s | **4,747 t/s**（−1.3%） |
| 4 路聚合 | 182.7 / 184.9 / 185.6 t/s | **174.3 / 188.5 / 185.9 t/s** |
| 每路 decode 中位 | 47.3 / 48.2 / 48.3 t/s | 45.5 / 49.4 / 48.2 t/s |
| TTFT | 0.287 / 0.356 / 0.334 s | 0.321 / 0.392 / 0.368 s |

解读：prefill 的 1.3–1.5% 落在本机重复测量的噪声带（同一版本三次重复也能差 ±2%），并发持平。
上游宣称老卡（V100/2080 Ti）长 prompt 变快，但**没有给 4090 + UD-Q4_K_XL 的数字**——这里的结果说明
在 4090 上这一步换代没有带来速度收益，也没有可见回退。

模型/模板/30 个引擎参数与 0.2.15 完全一致：unsloth UD-Q4_K_XL 111.33 GB + huihui v2 破限档 1
（48 张量）+ Sharp-naze v1.5（34,422 B）+ MTP + 4 slots + `--kv k8v4 --kv-resident 32768`。

## 6. 验收清单（`scripts/accept_q4kxl_0216.sh`，24 项）

/health 4 项（loaded、ctx 262144、images、model）；/props 4 项（HTTP 200、模板 34,386、`Strata 0.1.62`、
进程 cwd 在 `Strata-0.2.16`）；/metrics 6 项（engine 0.1.62、drafting mtp、slots 4、expert_slots ≥10000、
pcie_frac 0.15、arena ≥73000）；普通对话、effort low、视觉；4 路并发冒烟（4/4、聚合 ≥60）；
日志（PLE table、PLE loaded、`--ple-gguf`、引擎路径、prompt 内核行、MTP 草稿层）；显存。

## 7. 回滚

`0.2.15 + 0.1.60` 的树与启动器都保留：

```bash
cd /home/gfyd-ai/dsh_workspace/strata
bash stop_strata.sh
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory="$PWD" --setenv=STRATA_PY="$PWD/Strata/.venv/bin/python" \
  /bin/bash -c "./run_strata_q4l1_0215.sh >> strata-prod-20261119-rollback.log 2>&1"
```

`strata/run_strata.sh` 现在 exec 的是 `run_strata_q4l1_0216.sh`；改指针即可回到 0215。
引擎 0.1.62 也可单独回退（下游 `strata components rollback` 只回引擎，服务端仍是 0.2.16），
但 0.2.16 的 serve 与 0.1.62 是配套验证过的组合，不建议混搭。
