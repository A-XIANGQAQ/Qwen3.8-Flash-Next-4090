# v5 路线：Strata（2026-10-13 起）

> 本页记录 **v5**：把生产引擎从 v4（FreeToken 0.1.3 + 三个本地补丁）换成
> [Niko1221/Strata](https://github.com/Niko1221/Strata) 0.1.43.1。
> 版本全景见 [UPGRADE.md](UPGRADE.md)，实测数据见 [BASELINE.md](BASELINE.md)。

> [!IMPORTANT]
> **v5 不需要 v4 那三个补丁。** 它们是 FreeToken 特有问题的修补，Strata 路线上这些问题不存在
> （并发是原生 `--batch`，端口无 rendezvous 概念）。
> ⇒ **v5 是"少维护"而非"多维护"的升级**：`patches/04–06` 仅为 v4 回滚保留。
>
> ⚠️ 但 **v5 自带一处主线补丁**：[`patches/07_strata_esp_family_whitelist.patch`](../patches/07_strata_esp_family_whitelist.patch)
> ——上游把实验性投影锁在 `family in ("qwen","coder")`，unsloth 家族拿不到破限（见 §4）。
> **这不是回滚用的补丁，v5 必须打。**

---

## 1. 为什么换

两条路线跑的是**同一个基座模型**（Qwen3.8-Flash-Next），差别在引擎与权重格式。同机、同 prompt、同一服务入口、客户端计时：

| 指标 | v4（FreeToken 0.1.3） | **v5（Strata 0.1.43.1）** | 倍率 |
|---|---:|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s** | **2.0×** |
| 单流 TTFT | 2.73 s | **0.20 s** | **14×** |
| 2 路 聚合 / 每路 | 68.8 / 56.7 | **124** / **65.2** | 双向胜 |
| 4 路 聚合 / 每路 | 117.1 / **45.8** | **137** / 36.9 | 各有胜负 |
| 4 路 TTFT | 5.9 s | **0.81 s** | **7×** |
| 4 路端到端（512 token 输出） | 17.1 s | **14.7 s** | 1.16× |
| 并发扩展比（1→4 聚合） | **2.55×** | 1.16× | v4 胜 |
| 数学（15 题） | 15/15 | 15/15 | ≈ |
| 破限（强探针 14 条） | 0/14 | 0/14 | ≈ |

**端到端 = TTFT + 512 ÷ 每路 t/s** —— 这一列才是用户体感：v4 的 6 秒首字延迟把它单路吞吐的优势吃掉了。

**取舍**：v5 单流快一倍、首字快 14 倍、4 路端到端也更快；代价是**高并发扩展性弱**
（1→4 聚合只涨 1.16×，每路 36.9 vs v4 的 45.8 t/s）。1–2 路使用时 v5 全面占优；8 路以上密集并发需实测。

---

## 2. 模型：为什么是 UD-Q4_K_XL

v4 跑的是 **NVFP4**（4-bit）。要公平对比就必须**位宽对齐**，否则质量差异会误导结论。
按 [Strata docs/MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md) 选型：

| 候选 | 下载 | vs 本站 135.2 GB | 位宽 | 破限可用 |
|---|---:|---:|---|---|
| **v4 的 NVFP4** | 135.2 GB | — | **4-bit** | ✅（abliterated checkpoint） |
| UD-IQ4_XS | 93.68 GB | −30.7% | ~3.5-bit（IQ3_S/IQ4_NL 混合） | 需补丁 |
| **UD-Q4_K_XL** ← 选它 | **111.33 GB** | **−17.6%** | **4-bit** | 需补丁 |
| UD-Q5_K_XL | 158.29 GB | +17.1% | 5-bit | 精度高于 v4，不公 |
| UD-Q6_K_XL | 169.17 GB | +25.1% | 6-bit | 同上 |

**下载走 ModelScope**（`unsloth/Qwen3.8-Flash-Next-GGUF`，4 分片，字节数与 API 逐字节核对）。
HF 在本机实测仅 234 B/s，不可用。

---

## 3. 构建（Linux 无预编译包）

Strata 的 release **只有 Windows 二进制**（`strata-windows-x64*.zip`），Linux 必须自建。
本机环境已满足：CUDA 13.2 / cmake 4.2 / gcc 15.2 / 76 核。

```bash
# 1. 取源码（gh CLI）
gh repo clone Niko1221/Strata

# 2. 预解压 pinned llama.cpp（否则 setup 会去 GitHub 再拉一遍）
#    commit 见 setup.py 的 LLAMA_CPP_COMMIT = 3cf03257f219afbe7334045ff7c6a06ac68c627d
unzip llama.cpp-3cf0325.zip -d third_party/ && mv third_party/llama.cpp-3cf0325 third_party/llama.cpp

# 3. ESP 白名单补丁（**必须在 setup 之前打**——setup.py 就是被改的那个文件）
cp setup.py setup.py.orig-esp
patch -p1 < /path/to/repo/patches/07_strata_esp_family_whitelist.patch

# 4. setup（STRATA_UV=1 强制 Python 3.12；系统 python3 是 3.14，NVIDIA wheel 不齐）
STRATA_UV=1 ./setup.sh --setup --family unsloth --model UD-Q4_K_XL \
  --gguf-dir <模型目录> --source modelscope \
  --vision no --context 262144 --kv k8v4 --parallel 4 \
  --experimental-speed-projection on \
  --host 127.0.0.1 --port 8000 --yes --no-start
```

编译在 **76 核上约 1 分钟**（`[138/138] Linking CXX executable strata`），产出 `engine/strata`（59 MB），
`BUILD.json` 记 `archs: [89]`、CUDA 13.2。

> 💡 **构建与下载可重叠**：`setup.sh` **内部**的引擎编译步骤排在模型下载步骤之前。
> 下载期间先跑一次 setup 让它把引擎编好，模型到位后再跑一次补齐后续步骤。

---

## 4. 破限：用引擎自带功能，不是换模型

v4 的破限来自 **checkpoint 本身**（abliterated 权重）。v5 用的模型（Unsloth UD-Q4_K_XL）
**未破限**，破限改由 Strata 自带的 **experimental speed projection（ESP）** 提供：

- 载体：480 KB 控制向量（`Strata/data/experimental-speed-projection/`）
- 机制：第 **4–44 层**残差流上 `h -= (h·v)v`（= llama.cpp 的 `--cvec-mode project`）
- 开关：setup `--experimental-speed-projection on`；**运行时可逐请求关**（请求体 `"experimental_speed_projection": false`）

**强探针实测**（14 条经典类请求；`max_tokens=128`；`reasoning_effort: none` 让答案立刻出现以最小化生成；
只记录拒答布尔值，正文不打印不落盘）：

| 臂 | 拒答率 |
|---|---:|
| ESP **关**（stock 原始 Qwen3.8-Flash-Next） | **14/14 = 100%** |
| ESP **开** | **0/14 = 0%** |
| v4（abliterated checkpoint） | **0/14 = 0%** |

⇒ **同一引擎、同一进程、仅逐请求切换**，单变量证明破限生效，且与 v4 的 abliterated 效果等效。

> ⚠️ **ESP 的家族门**：上游只对 `family in ("qwen","coder")` 开放，Unsloth 家族被 `left off`。
> 查证 `UNSLOTH_Q4.md` 原文是 **"not tested"**（非不兼容），且引擎 `src/program/generate.cpp:672`
> 的 cvec 实现对量化无关 ⇒ 本项目对 `setup.py` 打了一处白名单补丁
> （[`patches/07`](../patches/07_strata_esp_family_whitelist.patch)，原件备份 `setup.py.orig-esp`）。
> **不打它，`--experimental-speed-projection on` 对 unsloth 家族静默无效**；打法见 §3 step 4。

---

## 5. 关键配置项

```
--batch 4                    # 每 slot 独立 session（VRAM 与主 session 同量，≈0.86 GiB/槽）
--max-context 262144         # 每 slot 独立吃满 256K
--kv k8v4                    # INT8 K + 旋转 Q4_0 V，816 B/cell
--kv-resident 32768          # KV 流式（见 §5.1）
--batch-mtp                  # 每 slot 每窗口验证一个 MTP 提议（见 §5.2）
--conversation-cache-mib 8192
--prefill auto:32768
--control-vector-scaled … --cvec-mode project --cvec-dir per-layer    # 破限
--resident-budget-gib 71     # 全部专家驻宿主内存
```

### 5.1 KV 格式与显存：`k8v4` + 必须流式

`kv_tok = 13 × cell`（12 个 QSA 层 + draft 层）：

| 格式 | B/cell | 256K 单 slot | 4 slot | 质量 |
|---|---:|---:|---:|---|
| `int8` | 1056 | 3.60 GB | 14.4 GB | 与 fp16 无法区分 |
| **`k8v4`** | **816** | **2.78 GB** | **11.1 GB** | needle 同 int8；上游实测 RTX 3090@198K **99 vs 85 t/s** |
| `q4_0` | 576 | 1.96 GB | 7.85 GB | ⛔ 长文档困惑度 +8%(1K)/+12%(8K)，256K 下不可用 |

> ⛔ **`--kv-resident 0`（KV 全放显存）直接 OOM**，实测报
> `native dense upload blk.27.attn_output.weight: out of memory`。
> ⇒ **KV 流式到宿主内存不是优化项，是 48GB 卡上 4×256K 的唯一可行解**（本站 247GB RAM 正好吃这一口）。
> 实测 `PCIe probe: 25.3 GB/s host→device`。

> ⚠️ Strata 的 KV 只有 `fp16|int8|k8v4|q4_0`，**没有 fp8**（fp8 仅用于 PLE 表）。

### 5.2 `--batch-mtp` 是并发兑现的前提

不加它，批处理退化成"没有 MTP 的那套行为"，实测 4 路聚合仅 131 t/s（vs 加后 137）、2 路 103–112（vs 123–125）。
上游 `docs/BATCHING.md` 原文如此。

---

## 6. 两个引擎不能共存

**两引擎各占本卡 90%+，任何时刻只能活一个。**
- v4：44,440–47,940 MiB
- v5：专家缓存 35.54 GiB + 4 slot × 0.86 GiB + 权重/buffers ≈ 44+ GB

⇒ 没有"影子部署 / 并存对照"这条路；切换 = 停机切换（实测约 1–2 分钟）。**要真并存只能加第二块卡或第二台机器。**

### ⛔ 切换必须等显存真正释放

实测踩到：`ft_stop.sh` 报"干净停止"时 `nvidia-smi` 仍显示 **47,712 MiB 被占**，
新引擎 `cudaMalloc` 权重竞技场直接 OOM。**判据是显存降到 <1000 MiB，不是进程消失**。
（本次因为漏这步，业务被 nginx 502 了 5 分钟。）

---

## 7. chat 模板必须由脚本保证

本站 v4 一直用 froggeric 修复版模板（见 [REPRODUCTION §5](REPRODUCTION.md)）。切换时发现 **Strata 用的是 GGUF 原装模板**，
两者渲染同一 prompt 差了 42 个 token（2378 vs 2336）：

| 模板 | 大小 | 行数 | `reasoning_effort` |
|---|---:|---:|---:|
| froggeric v22.5（v4 用） | 28,234 B | 459 | 2 |
| Strata 原装 | 9,993 B | 183 | 7 |

已把 froggeric 模板装到 Strata 的 `Strata-data/packs/<模型>/tokenizer/chat_template.jinja`，
**渲染 token 数对齐为 2336**（与 v4 一致）。

> ⚠️ 两个坑：
> 1. **Strata 每次 `setup` 都从 GGUF 重新解出该模板**，手工替换必被覆盖 ⇒ 应用动作挂在启动器上（§8）。
> 2. **默认思考档由模板决定**：froggeric 第 17 行 `_default_reasoning_effort = 'medium'`。
>    实测"不指定"与显式 `medium` 的思考长度逐字吻合（273 字符），`high` 为 451
>    ⇒ 切到 Strata 后**默认档没有变化**（Strata 文档说的 high 是**原装模板**的行为）。

---

## 8. 运维

> 下表的脚本名是本站的（`run_strata.sh` / `stop_strata.sh`，**本仓库不含这两个文件**——它们是按 §5
> 的参数写的薄封装）。自建启动器只需两件事：① 启动前跑一次 chat 模板守卫（§7）；
> ② `exec ./.venv/bin/python ./serve/server.py --engine strata --config <配置>.json --port 8000`。

| 动作 | 命令 |
|---|---|
| 启动（约 50 秒就绪） | `setsid bash run_strata.sh > strata-prod.log 2>&1 &` |
| 停止（约 10–20 秒 + 等显存） | `bash stop_strata.sh`——轮询 `nvidia-smi` 到 used <1000 MiB 才算停干净 |
| 引擎互切 | 本站用自建的一键脚本（**不在本仓库**；流程见 [OPERATIONS §1b](OPERATIONS.md)） |
| 模板守卫 | 挂在启动器上（`ensure_chat_template.sh` 式做法，幂等、fail-open）；v4 的启动器同样挂了 |

⚠️ **必须用启动器启动，不要用 `./setup.sh`** —— 后者会自己拉起模型并绕过模板守卫。

健康判据（三层，缺一不可）：进程 → **破限判据行** → 端到端推理。
`control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`

---

## 9. 未验证 / 待测

- **长上下文（128K / 256K）未实测**：配置与显存账支持 4×256K，但没在真实长文下跑过
- **多日稳定性未测**
- **引擎互切脚本的真实切换+回滚路径未测**（2026-10-13 的切换是手工做的）
- **8 路以上密集并发的扩展性未测**（本页只有 1/2/4 路）
- **质量样本仅 15 题数学**（双方 15/15，不构成质量结论）；要做质量结论需上 GSM8K-1319 之类
- ESP 对普通文本质量的影响未自测（上游报 KL 0.063、困惑度 +15% 代码 / +2.3% 文档 / +0.4% 聊天）
