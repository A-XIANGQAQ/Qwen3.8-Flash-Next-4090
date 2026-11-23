# Strata 0.2.20 + engine 0.1.66 升级（2026-11-23，v6.3）

> 结论：**升级完成并保留**。0.1.66 在 sm_89（RTX 4090）上有一个上游回归，默认配置下首个请求就崩；用 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 规避后一切正常。规避代价可忽略（Q8 超连接走 fallback 内核），收益是 4 并发聚合 **+7~17%**、12.8K prefill **+5.0%**。

## 1. 发布事实

| 项 | 值 |
|---|---|
| 发布 | **v0.2.20**，2026-11-22T23:20:39Z（created 22:51:46Z），feed **132** |
| 版本 | App/Terminal 0.2.20、Control **0.1.66**、**三套引擎（engine / hy4 / inkling）全 0.1.66** |
| Linux x86_64 CUDA13 引擎 | `strata-engine-0.1.66-linux-x86_64-cuda-cuda13.tar.gz` **591,563,326 B**，sha256 `13c5bebeaf24574b02307857bce6abfd3f861f3682ca98dbe36c1c461fd5e059` |
| Control 0.1.66 x86_64 | 94,663,788 B（本机不装） |
| `BUILD.json` | `{"version":"0.1.66","source":"tools/packaging/build_cuda_linux.sh","archs":"75;86;89;120","cuda":"13.0","portable":true,"vision_archs":"75;86;89;120"}` |
| 二进制 | `strata` 188,947,304 B、`strata-vision` 1,241,120 B、`strata-split-2-sm86` 61,359,176 / `sm120` 66,848,720、内置 `libcublas*.so.13` |
| 源码树 | tag `v0.2.20`：8,999 项 / 8,240 blob；`terminal/{serve,strata_control,tools,strata_cli}` 过滤后 **587 文件**（复用 487、新下 100，逐个 blob sha 校验 587/587） |

## 2. 0.2.20 改了什么（与本机相关的部分）

- **统一 4 层 MoE 执行器**（GPU + RAM + CPU + SSD）覆盖 Qwen3.8-Flash-Next、HY-4 Flash、Inkling 2 Mini，最低可到 4 GiB 显卡；`--kv-offload auto`（32K+ 自动把 KV 放主机）；`--layer-split auto`（2–4 卡）。本机单卡 + 显式 `--kv k8v4`，不受影响。
- **NVIDIA CUDA（sm_60..sm_121，含 V100 sm_70）**：13 组 Unsloth 专家对走 fused `EXP_MODE=8`，MoE decode 1.04×–1.67×，down 投影最高 2.58×；10 种 INT8 Tensor Core dense 格式（分块 prefill / DFlash2 verify 1.78×–3.26×）。我们正是 UD-Q4_K_XL 专家对。
- **长上下文 NaN 修复 `#6431`**：24K–45K prompt 的 grid 维度 65,535 溢出 + 0/0 zero-amax，`#879` 的触发面。
- NVFP4 多语言乱码/0% 投机接受率（`#6575`–`#6634`）、Intel 12–14 代 AVX2-VNNI 非法指令（`#6541`/`#6570`）、Windows WDDM 回归等 —— 与本机无关。
- **没有**工具调用/思考泄漏类修复；`#4751`/`#4077`/`#4949` 仍 open。

## 3. 命中的坑：0.1.66 在 sm_89 上 `verify: layer 0 never rang`

### 症状

首次生成（任何请求）立即失败，引擎自行退出，serve 返回：

```
{"error": {"type": "invalid_request_error",
 "message": "verify: layer 0 never rang (operation not supported on global/shared address space)"}}
```

引擎日志：

```
strata: STRATA_HC_Q8=1: 0.63 GiB of Q8_0 hyper-connection projections for the verify read
strata verify: capturing the 4-token window (2593 MiB of VRAM free)
strata verify: captured the 4-token window (upload no error, sync no error) in 8.3 ms
strata serve: verify: layer 0 never rang (operation not supported on global/shared address space)
```

模型加载、`/health`、`/props`、`/metrics` 全部正常，**只有真正 decode 时崩**。

### 上游状态

这是 0.1.66 的已知回归，官方 issue 一串，其中 **`#6742` 与本机完全同型**（RTX 4090 48GB、Linux、CUDA13 引擎、同一错误串）；另有 `#6700`（4090 Laptop sm_89）、`#6745`（3090 sm_86）、`#6706`（8 卡层切分 + PIPED_MTP）。

- 根因（issue 内 `compute-sanitizer` 定位）：v0.2.20 的 commit `534802c8d` 把 `engine/src/kernels/cuda/fused_gr.cu` 的 `q8_ld16()` 改成 `__ldg()`，但四个 sm_80+ Tensor Core kernel 读的是 `__shared__` 里的 `s_w`；`__ldg()` 只接受 global 指针 ⇒ `cudaErrorInvalidAddressSpace`（Windows 报 illegal memory access，Linux 报 `operation not supported on global/shared address space`）。
- 触发条件：`STRATA_HC_Q8=1`（CUDA 上对带 Q8_0 超连接投影的 GGUF 自动开，Unsloth `UD-*` 与 Q8_0 都命中）+ `STRATA_HC_Q8_TC=1`（sm_80+ 默认）。
- 修复 PR **`#6714` 仍 open**；v0.2.20 是当前最新 release，没有修好的版本。

### 规避（本机采用）

config 的 `env` 里加：

```json
"env": {
  "STRATA_ROUTE_TAIL_SKIP": "0",
  "STRATA_HC_Q8_TC": "0",
  "STRATA_HC_Q8_BF16TC": "0"
}
```

两个 TC kernel 关掉后，Q8 read 回落到 `gr_up_q8_fast_kernel`（官方 issue 作者验证「engine starts here」）。`STRATA_HC_Q8=0` 也可，但会放弃 Q8 超连接读、多占 0.63 GiB 显存。实测两个变量确实进了引擎进程的 `/proc/<pid>/environ`，随后 24/24 验收通过。


## 3b. 客户端模型名：0.2.20 的严格匹配（`#6621`）与 `aliases`

0.2.20 起 `check_model()` 会拒绝「看起来像 Strata 目录模型 ID」的名字（前缀 `inkling-2-mini` / `hy4-flash` / `qwen3.8-flash-next` / `qwen3.6-` / `qwen3.5-`），
即使 `strict_model` 没开：`{"code":"model_not_found","message":"The model 'Qwen3.8-Flash-Next' is not loaded; currently serving 'qwen3.8-flash-next-q4kxl-ablit-l1'"}`。
DSH（provider 模型 id `Qwen3.8-Flash-Next`）和 Cherry（同名）都会 404，`/v1/messages` 与 `/v1/chat/completions` 一样。

修法是在 config 顶层加别名（列表或逗号串），客户端不用改：

```json
"aliases": ["Qwen3.8-Flash-Next", "qwen3.8-flash-next-q4kxl-ablit"]
```

`/v1/models` 会列出 `qwen3.8-flash-next-q4kxl-ablit-l1` + 两个别名；响应里的 `model` 字段回显请求用的别名（`model_for()`）。
2026-11-23 13:04 重起（30 s）后 `/v1/messages`、`/v1/chat/completions` 用旧名都返回 200，验收 24/24 不受影响。

## 4. 切替时间线

| 时间 | 事件 |
|---|---|
| 11:23:52 | 第一次切替 stop（显存 9 MiB、8000 无监听） |
| 11:23:59 | systemd-run 起 0.2.20 + 0.1.66 |
| 11:24:27 | `/health` loaded=true（28 s），`build_info "Strata 0.1.66"` |
| 11:24:37 | 首个生成崩：`layer 0 never rang`；引擎自动重启，后续请求全失败 |
| 11:25:5x | 停止并回滚到 0.2.16 + 0.1.62 |
| **11:26:02** | 回滚就绪（`build_info Strata 0.1.62`） |
| 11:28:17 | 加 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 后重新切替 |
| **11:28:45** | `/health` loaded=true（28 s），`build_info Strata 0.1.66` |

第一次失败到回滚完成共 **2 分 10 秒**；第二次切替停机约 **35 秒**。模型、模板、30 个引擎参数一字未改。

## 5. 验收（`scripts/accept_q4kxl_0220.sh`，24 项）

`PASS=24 FAIL=0`，关键项：

- `/health`：`qwen3.8-flash-next-q4kxl-ablit-l1`、loaded、ctx 262144、images true；
- `/props`：`build_info "Strata 0.1.66"`、模板 **34,386 字符**（Sharp-naze v1.5）；
- `/metrics.engine`：`engine 0.1.66`、`drafting mtp`、`batch_slots 4`、`expert_slots 10,977`、`expert_cache_mib 32,816`、`arena_mib 73,450`、`pcie_frac 0.15`、`spec 6`、`mtp_max 4`、`kv k8v4`、`kv_resident 32768`；
- 普通对话 / effort low / 视觉 / 4 路并发冒烟 `ok=4 tok=477 agg=102.7`；
- 进程 cwd `Strata-0.2.20`；引擎路径 `Strata-0.2.20/engine/strata --serve`；
- PLE 表 26.8 GiB 已加载（未锁定，RLIMIT_MEMLOCK 8 MiB）；prompt 内核 48/48 层；`strata mtp: draft layer loaded`；
- 显存约 46,1xx / 49,140 MiB。

首跑 `serve_tree` FAIL 是脚本自身的 sed 残留（转义过的 `Strata-0\.2\.16` 没替换），修掉后 24/24。

## 6. 实测（同机同协议）

工具：`/tmp/s0214/bench.sh`（prefill 6.2K/12.8K，各 3 次取中位）+ 4 路并发（512 token × 3 轮）。

| 项 | 0.2.16 + 0.1.62 | 0.2.20 + 0.1.66（规避后） | 差 |
|---|---:|---:|---:|
| prefill 6.2K | 2,759.8 t/s | 2,738.5 t/s | **−0.8%** |
| prefill 12.8K | 4,747.3 t/s | **4,986.2 t/s** | **+5.0%** |
| 4 路聚合 | 174.34 / 188.52 / 185.94 | **204.87 / 200.38 / 199.72** | **+17.5% / +6.3% / +7.4%**（三轮中位 +7~10%） |
| 每路 decode 中位 | 45.5 / 49.4 / 48.2 | 52.76 / 52.41 / 52.52 | **+16% / +6% / +9%** |
| TTFT（4 路） | 0.321 / 0.392 / 0.368 | 0.202 / 0.290 / 0.397 | 持平 |
| 墙钟 | 11.75 / 10.86 / 11.01 s | 10.0 / 10.22 / 10.25 s | **−7~15%** |

单流 decode（512 token 量级、MTP）：

| 设置 | decode | MTP 接受率 |
|---|---:|---:|
| `reasoning_effort=off` | 125.6 t/s | 186/354（52.5%） |
| `reasoning_effort=low` | **134.3 t/s** | 230/412（55.8%） |

对照 0.1.62 的单流 133–135 t/s（low）⇒ **单流持平、并发变快**。

副作用：Q8 超连接投影占 0.63 GiB，专家槽从 11,049 降到 10,977（−72 槽，缓存 33,038 → 32,816 MiB）；空闲显存 2,517 → 2,651 MiB。对 4 路 512-token 测试无可见影响。

## 7. 回滚

```bash
cd ~/dsh_workspace/strata
bash stop_strata.sh
systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes \
  --working-directory="$PWD" --setenv=STRATA_PY="$PWD/Strata/.venv/bin/python" \
  /bin/bash -c "./run_strata_q4l1_0216.sh >> strata-prod-rollback.log 2>&1"
```

`Strata-0.2.16` + 0.1.62 保留，config `strata-unsloth-q4kxl-ablit-l1-0216.json` 与启动器都在。指针 `run_strata.sh` 现在指向 `run_strata_q4l1_0220.sh`。

## 8. 复现要点

1. 下载引擎 591,563,326 B，sha256 必须是 `13c5bebe…5e059`；解包后 `BUILD.json` 版本 0.1.66，`./strata --version` 输出 `strata 0.1.66`。
2. 源码树按 tag v0.2.20 取 `terminal/{serve,strata_control,tools,strata_cli}`，本机平铺为 `Strata-0.2.20/{serve,tools,strata_control,strata_cli}`（`server.py` 的 `ROOT = parents[1]` 正好命中树根）。
3. config 复制 0216，只改 `exe` / `cwd` / `log` / `vision.exe` 四个字段，**并在 `env` 里加上面三个变量**（sm_89 必须）。
4. 启动器 `run_strata_q4l1_0220.sh`（`cd Strata-0.2.20 && python ./serve/server.py --engine strata --config …-0220.json --port 8000`），`chmod +x`。
5. 验收 `scripts/accept_q4kxl_0220.sh`；基准 `/tmp/s0214/bench.sh after-0220`。
6. 0.1.67（含 `#6714`）发布后可去掉两个 `STRATA_HC_Q8*` 环境变量，再复测一次。

## 9. 待办 / 观察点

- 跟踪 `#6714` 合并与 0.1.67 发布；届时应去掉规避变量并重测（预期略快）。
- 0.2.20 的 `#6431` 长 prompt NaN 修复建议在 24K–45K prompt 上做一次专项复测（本机日常有 20K–57K prompt）。
- 若出现新的 `never rang`/`illegal memory access`，先看 `STRATA_HC_Q8*` 是否被 setup 或新配置重新打开。
