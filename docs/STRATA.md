# v5 路线：Strata（2026-10-13 起；引擎 2026-10-18 升到 0.1.45）

> 本页记录 **v5**：把生产引擎从 v4（FreeToken 0.1.3 + 三个本地补丁）换成
> [Niko1221/Strata](https://github.com/Niko1221/Strata)，**当前引擎版本 0.1.45**（2026-10-18 由 0.1.43.1 升级，
> 两臂实测见 [UPGRADE.md](UPGRADE.md) §0b）。
> 版本全景见 [UPGRADE.md](UPGRADE.md)，实测数据见 [BASELINE.md](BASELINE.md)。

> [!IMPORTANT]
> **v5 不需要 v4 那三个补丁。** 它们是 FreeToken 特有问题的修补，Strata 路线上这些问题不存在
> （并发是原生 `--batch`，端口无 rendezvous 概念）。
> ⇒ **v5 是"少维护"而非"多维护"的升级**：`patches/04–06` 仅为 v4 回滚保留。
>
> ⚠️ 但 **v5 自带一处主线补丁**：[`patches/07_strata_esp_family_whitelist.patch`](../patches/07_strata_esp_family_whitelist.patch)
> ——上游把实验性投影锁在 `family in ("qwen","coder")`，unsloth 家族拿不到破限（见 §4）。
> **这不是回滚用的补丁，v5 必须打。**

> ⚠️ **v5.2 起还有一处功能补丁**：[`patches/08-auto-mode.patch`](../patches/08-auto-mode.patch)（2026-10-18）
> ——模型自带模板有 41 个 `{REASON:xxx}` 模式，但只有用户手写标签才会加载；补丁在 `serve/server.py` 的
> `Service.prepare()` 前面加一层判断，按问题内容自动挑模式（规则表在新增的 `serve/auto_mode.py`）。
> 配置里 `"auto_mode": true` 才生效，默认关；**升级引擎后要重打**（`serve/server.py` 会被覆盖）。

---

## 1. 为什么换

两条路线跑的是**同一个基座模型**（Qwen3.8-Flash-Next），差别在引擎与权重格式。同机、同 prompt、同一服务入口、客户端计时：
（下表是 **2026-10-13 用 0.1.43.1** 做的 v4↔v5 对比，作为历史基线保留；**0.1.43.1 → 0.1.45 的升级实测在
[UPGRADE.md §0b](UPGRADE.md)**，两批数字不可混引。）

| 指标 | v4（FreeToken 0.1.3） | **v5.0（Strata 0.1.43.1）** | 倍率 |
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

## 2. 模型：UD-Q4_K_XL（历史）→ 现在是 DavidAU abliterated IQ3_S

> ⚠️ **2026-10-18 起生产模型是 `DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF`（IQ3_S）**，
> 不是本节讲的 UD-Q4_K_XL。换它的原因、A/B 数字与部署要点见 [UPGRADE.md §0c](UPGRADE.md)：
> 同机同参数实测 decode +10%、1K TTFT −49%、100K decode +31%、4 路墙钟 −29%、温和拒答 4.55%→0%，
> 且**自带 abliteration ⇒ 不再需要 ESP 控制向量**。
> 本节保留为"当初为什么选 UD-Q4_K_XL"的记录，不要删——它是 §0c 的对照基线。

v4 跑的是 **NVFP4**（4-bit）。要公平对比就必须**位宽对齐**，否则质量差异会误导结论。
按 [Strata docs/MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md) 选型：

| 候选 | 下载 | vs 本站 135.2 GB | 位宽 | 破限可用 |
|---|---:|---:|---|---|
| **v4 的 NVFP4** | 135.2 GB | — | **4-bit** | ✅（abliterated checkpoint） |
| **DavidAU abliterated IQ3_S**（2026-10-18 起生产） | **83.86 GB** | **−38.0%** | **3.5 bpw** | ✅（模型自带 abliteration） |
| UD-IQ4_XS | 93.68 GB | −30.7% | ~3.5-bit（IQ3_S/IQ4_NL 混合） | 需补丁 |
| **UD-Q4_K_XL**（2026-10-13~18 生产，已删） | **111.33 GB** | **−17.6%** | **4-bit** | 需补丁 |
| UD-Q5_K_XL | 158.29 GB | +17.1% | 5-bit | 精度高于 v4，不公 |
| UD-Q6_K_XL | 169.17 GB | +25.1% | 6-bit | 同上 |

**下载**：UD-Q4_K_XL 走 ModelScope（`unsloth/Qwen3.8-Flash-Next-GGUF`，4 分片，字节数与 API 逐字节核对）；
DavidAU 那份只在 HuggingFace，走 1× 代理 **12 路并行 range**（实测 16 MB/s，6 路只有 9），两片 sha256 与 HF 的 `lfs.oid` 对齐。

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
  --vision yes --context 262144 --kv k8v4 --parallel 4 \
  --experimental-speed-projection on \
  --host 127.0.0.1 --port 8000 --yes --no-start
```

编译在 **76 核上约 1 分钟**（`[138/138] Linking CXX executable strata`），产出 `engine/strata`（59 MB），
`BUILD.json` 记 `archs: [89]`、CUDA 13.2。

> 💡 **构建与下载可重叠**：`setup.sh` **内部**的引擎编译步骤排在模型下载步骤之前。
> 下载期间先跑一次 setup 让它把引擎编好，模型到位后再跑一次补齐后续步骤。

### 3.1 升级到新 tag：只调 `build_engine()`，不要重跑 setup

上面那套 `setup.sh` 是**首次安装**用的。机器上已经跑着生产、config 里有 12 个手调参数时，
**再跑一次 setup 会出事**：`carry_over()` 明确不会合并手工加进 `args` 的选项；而且 setup 对大分片的
SHA-256 校验有已知 bug（#2902），**会删掉正确的模型分片**。详见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) #40。

升级走这条路（全程不停机，直到最后换二进制）：

```bash
cd strata/Strata
git fetch origin --tags
git worktree add ../Strata-0.1.45 v0.1.45
ln -s ../../Strata/third_party/llama.cpp ../Strata-0.1.45/third_party/llama.cpp   # worktree 里没有这个未跟踪目录
# patch 07 打在 worktree 的 setup.py 上（0.1.43.1 在 5419 行，0.1.45 漂到 6643 行，上下文一致）

# 只调编译入口，__name__ != "__main__" 保证不会跑 main()
<installed tree>/.venv/bin/python scripts/build_engine_upgrade.py --src strata/Strata-<tag>
```

产物（0.1.44 起比以前多一个库）：

| 文件 | 0.1.43.1 | 0.1.45 | 说明 |
|---|---:|---:|---|
| `engine/strata` | 59.4 MB | 77.9 MB | 引擎（含 K-quant 专家内核） |
| `engine/strata-vision` | 73.8 MB | 73.7 MB | GPU 图像编码器 |
| `engine/libstrata_kq.so` | — | **12.6 MB** | K-quant prompt 内核库，**换引擎时必须一起拷** |
| `engine/BUILD.json` | `src: 7089bf00…` | `src: b9513716…` | 版本与源码指纹 |

满编耗时（76 核，两棵树都在）：**187.8 秒**（引擎 153 个 target + vision 275 个 target）。
比 §3 里「约 1 分钟」慢，是因为 0.1.44 起多编了 K-quant 内核与 `strata_kq` 库。

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
--prefill auto:32768         # 见 §5.4：引擎的 #669 警告建议 auto，但实测 32768 快 ~20%
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

### 5.3 图像：vision 编码器（GPU 模式）

**默认是关的**——setup 交互时会问，非交互（`--yes`）下按默认「不要」走；只有显式 `--vision yes` 才装。
2026-10-13 本站才第一次装上：此前**这台机器从来没读过图**（v4 时代也没有 mmproj）。

装它要三样东西：

| 件 | 说明 |
|---|---|
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | **907,543,008 B**（0.91 GB）。来源 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`——**ModelScope 上没有这个仓库**，只能走 HF 或镜像 |
| `engine/strata-vision` | `cmake tools/vision` + `-DSTRATA_VISION_CUDA=ON`；本站自建产物 **73.75 MB**（引擎 `strata` 不受影响，**不必重编**） |
| 配置里的 `vision` 段 | `{exe, mmproj, model, gpu, max_tokens}`；`max_tokens` 默认 GPU=1024 |

> 💡 **配置里有 `vision` 段、但 `args` 里没有 `--vision` 时，serve 层会自己补上**
> `--vision --vram-reserve-mib 700` 并打一行提示 ⇒ 手写配置段即可生效，不必再跑 setup。

**代价（本站实测 2026-10-13，三轮）**：

| 项 | 装前（无 vision） | 默认 reserve 700 | **reserve 2560（定稿）** |
|---|---:|---:|---:|
| 显存 | 48,454 / 49,140 MiB | 48,335 MiB | **46,483 MiB** |
| 引擎自述空闲 | — | **329 MiB** ⛔ 不够 | **2,181 MiB** ✅ |
| GPU 专家缓存 | 35.54 GiB | 34.09 GiB | **32.28 GiB**（11054 专家） |
| 单流 TTFT | 0.20 s | 0.19 s | **0.19 s** |
| 单流纯解码 | 118 t/s | 121 t/s | **124 t/s**（6 次，117–128） |

⇒ **文本速度在噪声内没变**，尽管 GPU 专家缓存少了 3.26 GiB（resident RAM 模式仍持 37.64 GiB，
少掉的那点由文件层兜住）。上游预期的「text a few % slower」本站没测出来。

> ⛔ **`--vram-reserve-mib` 不能留默认 700——这是必踩的坑。**
> serve 层在「配置有 `vision` 段、但 `args` 里没有 `--vision`」时会**自动补 `--vram-reserve-mib 700`**，
> 那是给小卡设计的默认值。48GB 卡本身已占满时，引擎会把显存吃到**只剩 329 MiB**；而编码器一旦重启
> （图片请求失败一次它就重启一次）就需要 ~1.5 GB，于是
> `cuMemCreate(...) CUDA error: out of memory`
> → `400 the image could not be read: the vision encoder stopped` → 再重启 → 死循环。
>
> **两个判据行，都在引擎日志里**：
> - `strata serve: N MiB of VRAM free with everything loaded` —— N 必须 ≥ ~1600
> - `strata verify: no VRAM for a new window graph ... (a larger --vram-reserve-mib keeps more)` —— 出现即说明卡已被压死
>
> **编码器的显存需求与图大小无关**：实测 360×240 的小图与 1398×1167 的真截图**峰值完全相同 = 1,481 MiB**
> ——它在预热时就按 `max_tokens` 顶格分配。所以留够 ~1.5 GB 是硬要求，不是"大图才需要"。
>
> **修法**：把 `--vision --vram-reserve-mib 2560` **显式写进配置的 `args`**——写了就不会再被自动补 700。

⚠️ **不要为了开图像重跑 `setup.sh`**：`carry_over()` 明确写着「手加到 `args` 的引擎选项不合并（setup chooses
those）」——重跑会抹掉 `--kv-resident 32768` / `--batch-mtp` / `--conversation-cache-mib 8192` 这些手调项。
正确做法：手工下 mmproj + 编 `tools/vision` + 往配置里加 `vision` 段。

⚠️ **上游对 UD-Q4_K_XL 的图像标注是 untested**（#967：「用户报告能用，但维护者没在这个文件上跑过图片」）。
本站自测（2026-10-13，三色横条测试图）4 问全对、含空间顺序：从上到下红/绿/蓝 ✅、最上=红 ✅、最下=蓝 ✅、条数=3 ✅。
**但样本极小，不构成质量结论。**

> 📌 **下载提示**：HF 走代理单连接实测仅 ~220–400 KB/s（0.91 GB 要 40+ 分钟）；
> 换 **12 路并行 range 分块**后聚合 **4.1 MB/s**（约 4 分钟）。单连接被限速时值得分块。

### 5.4 prefill 块大小：引擎的 #669 警告在这里帮了倒忙

引擎启动时会警告：

```
WARNING: a prompt chunk of 32768 tokens lends 4326 cache slots to the prompt path,
but only 0 of their experts fit in RAM: the others are read from the SSD on every chunk,
and prompts can read ~3x slower than with --prefill auto (#669).
A smaller --prefill, or auto, keeps them all in RAM
```

看着像必须改。**但单变量实测推翻了它**（同机、同一批固定 prompt、每次用全新唯一前缀避开前缀缓存、各 2 次）：

| `--prefill` | 引擎选的块 | 借用的显存槽 | 22K prompt | 72K prompt |
|---|---|---:|---:|---:|
| `auto` | **8192** | 1248（3.67 GiB） | 2,416 t/s | 2,780 t/s |
| **`auto:32768`** ← 用这个 | **32768** | 4326（12.64 GiB） | **2,905 t/s** | **3,403 t/s** |

⇒ **`auto:32768` 快约 20%**，尽管它每次都从 SSD 现读专家。大块摊销得更好，警告说的"~3x"在这台机器上没有兑现。
（`auto` 的实测值两次几乎完全一致：34K→2406/2427、110K→2790/2770，不是噪声。）

> ⚠️ 这与 `--moe-strategy auto`（#24）是**同一类教训**：引擎的启发式是按一般机器调的，
> **不等于你机器上的最优**——只有单变量上机实测才定案。
>
> ✅ **0.1.45 起这行 #669 警告消失了**（两臂启动日志对比：0.1.43.1 有、0.1.45 没有）。
> 但我们仍显式写 `--prefill auto:32768`——上面的实测结论不因警告消失而改变。
>
> ⚠️ 顺带：`run_strata.sh` 用 `> strata-prod.log` **会截断日志**。重启前想留证据的话改成 `>>` 或先备份。

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
| 启动（约 50 秒就绪） | `nohup bash run_strata.sh > strata-prod.log 2>&1 &` —— **必须在自己终端里跑**，见 [OPERATIONS §1](OPERATIONS.md)（从 agent 的工具调用里起会被 cgroup 连坐） |
| 停止（约 10–20 秒 + 等显存） | `bash stop_strata.sh`——轮询 `nvidia-smi` 到 used <1000 MiB 才算停干净 |
| 引擎互切 | 本站用自建的一键脚本（**不在本仓库**；流程见 [OPERATIONS §1b](OPERATIONS.md)） |
| 模板守卫 | 挂在启动器上（`ensure_chat_template.sh` 式做法，幂等、fail-open）；v4 的启动器同样挂了 |

⚠️ **必须用启动器启动，不要用 `./setup.sh`** —— 后者会自己拉起模型并绕过模板守卫。

健康判据（三层，缺一不可）：进程 → **破限判据行** → 端到端推理。
`control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`

---

## 9. 已验证 / 未验证

**已在 0.1.45 上验证（2026-10-18，两臂实测见 [UPGRADE.md](UPGRADE.md) §0b）**：

- **100K 上下文已实测**：TTFT 21.8 s、decode 109.0 t/s（512 token × 3 次中位）
- **sm_89（4090）上没有上游 #2936 报的 decode 回退**：短上下文 +3.4%、32K 判等、100K 判等
- 视觉三色条 3/3、专家缓存 32.27 GiB / 11050 槽、启动后空闲显存 2161 MiB
- 0.1.43.1 曾在生产连续跑 **2 天 23 小时**（10-15 09:40 被 cgroup 连带杀掉，不是自己崩的）

**仍未验证**：

- **128K / 256K 上下文未实测**：配置与显存账支持 4×256K，但没在真实长文下跑过（100K 已测）
- **多日稳定性**：0.1.45 上还没跑过长时间
- **引擎互切脚本的真实切换+回滚路径未测**（2026-10-13 的切换是手工做的；10-18 的版本升级也是手工换二进制）
- **8 路以上密集并发的扩展性未测**（本页只有 1/2/4 路）
- **质量样本仅 15 题数学**（双方 15/15，不构成质量结论）；要做质量结论需上 GSM8K-1319 之类
- ESP 对普通文本质量的影响未自测（上游报 KL 0.063、困惑度 +15% 代码 / +2.3% 文档 / +0.4% 聊天）
