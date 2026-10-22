# 升级记录 (Upgrade Log)

本页记录本项目的引擎演进。当前为 **v5.3（Strata 0.2.2 服务端 + 0.1.47 预编译引擎 + DavidAU abliterated IQ3_S + 合并模板 v3）**，v1–v4（FreeToken / 官方 sglang / Lsglang）全部折叠保留作参考。

| 版本 | 时间 | 路线 | 状态 |
|---|---|---|---|
| **v5.3（当前）** | **2026-10-22** | **Strata 0.2.2 服务端 + 0.1.47 预编译引擎** + DavidAU IQ3_S + 合并模板 v3（41 键） | ✅ 已上线（见 §0d） |
| v5.2 | 2026-10-18 | **Strata 0.1.45 + DavidAU abliterated IQ3_S**（模型自带破限，**不再用 ESP**；4-bit→3.5bpw，快 10~52%，拒答 4.55%→0%） | 被 v5.3 取代（引擎升级，模型/参数未变），见 §0c |
| v5.1 | 2026-10-18 | **Strata 0.1.45** + UD-Q4_K_XL + ESP 破限（[patch 07](../patches/07_strata_esp_family_whitelist.patch) 已对位到 0.1.45 的 `setup.py:6643`） | 被 v5.2 取代；UD-Q4_K_XL 权重已删（回滚需重下 104 GB），见 §0c.4 |
| v5.0 | 2026-10-13 | **Strata 0.1.43.1** + 自带 ESP 破限 | 已被 v5.1 取代；引擎二进制留档，可回滚（见 §0b.6、§0.2） |
| v4 | 2026-09-24 | **FreeToken 0.1.3** + 三个本地补丁 + abliterated NVFP4 | ⛔ 运行时与权重已于 2026-10-18 删除，**不可回滚**（见 §1 历史记录） |
| v3 | 2026-09-23 | 官方 sglang tag **v0.5.20** + lk_moe 补丁重放 + PLE 补丁 | ⛔ 运行时已删（2026-10-18）；历史（见 §5） |
| v2 | 2026-09-14 | Lsglang `0.5.19-lkmoe` 分支合并 PR #37500 + PLE 补丁 | ⛔ 运行时已删；历史（见 §6） |
| v1 | 原始 | Lsglang 1.4.13 + lk_moe 2.4.0 | ⛔ 运行时已删；历史（见 §6） |

> 本文件名原为 `UPGRADE_UPSTREAM_FLASHNEXT.md`（标题写死 sglang 路线）。切到 v4 后改名为 `UPGRADE.md`，
> 旧链接会 404，请更新。

---

## 0. v5.0：切到 Strata（2026-10-13）

> ⚠️ 本节描述的引擎版本是 **0.1.43.1**，已被 §0b（0.1.45）取代；**路线、模型、破限方式都没变**，
> 变的只有引擎版本号与几个内部的显存/预填充行为。本节的数据作为**基线**保留，不要删——
> §0b 的所有「提升幅度」都是与这里的数字对比得出的。

> 完整路线说明见 **[STRATA.md](STRATA.md)**；本节只记「为什么切、怎么切、踩了什么坑」。

### 0.1 为什么切

同一个模型、同一 prompt、同一个服务入口，**单流 decode 2.0×、TTFT 14×**：

| 指标 | v4 | **v5** |
|---|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s** |
| 单流 TTFT | 2.73 s | **0.20 s** |
| 4 路端到端（512 token） | 17.1 s | **14.7 s** |
| 并发扩展比（1→4） | **2.55×** | 1.16× |

**取舍**：v5 单流全面占优，但**高并发扩展性弱**（每路 36.9 vs 45.8 t/s）。1–2 路场景 v5 明显更好。

### 0.2 三个硬约束（换引擎前必须知道）

1. **两引擎不能共存**：各占本卡 90%+（v4 44,440–47,940 MiB；v5 专家缓存 35.54 GiB + 4×0.86 GiB slot + 权重）。
   没有"影子部署"这条路，切换 = 停机切换（约 1–2 分钟）。
2. **停旧后必须等显存真正释放**：`ft_stop.sh` 报"干净停止"时 `nvidia-smi` 仍显示 47,712 MiB 被占，
   新引擎 `cudaMalloc` 直接 OOM。**判据是 used <1000 MiB，不是进程消失。**
   （本次漏这步 → nginx 502 五分钟，`/var/log/nginx/error.log` 留下
   `connect() failed (111: Connection refused) to upstream http://127.0.0.1:8000`。）
3. **KV 必须流式**：`--kv-resident 0`（KV 全放显存）实测直接 OOM
   （`native dense upload blk.27.attn_output.weight: out of memory`）。

### 0.3 迁移方法

```bash
# 1. 取源码 + 模型（Strata release 只有 Windows 二进制，Linux 必须自建）
gh repo clone Niko1221/Strata
modelscope download --model unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL/* --local_dir <模型目录>

# 2. setup（STRATA_UV=1 强制 py3.12；系统 python3 是 3.14，NVIDIA wheel 不齐）
STRATA_UV=1 ./setup.sh --setup --family unsloth --model UD-Q4_K_XL \
  --gguf-dir <模型目录> --source modelscope --context 262144 \
  --kv k8v4 --parallel 4 --experimental-speed-projection on \
  --host 127.0.0.1 --port 8000 --yes --no-start

# 3. 装修复版 chat 模板（setup 从 GGUF 解出的原装模板会差 42 个 token）
cp froggeric-v22.5/chat_template.jinja \
   Strata-data/packs/<模型>/tokenizer/chat_template.jinja

# 4. 启动 + 核验
#    ⛔ 必须在**你自己的终端**里跑：从 agent/CI 的工具调用里起，进程会落在 systemd
#       临时单元 dsh-subprocess-*.scope 里（setsid 也脱不出 cgroup），重启 dsh-web
#       时被整组 SIGTERM。判据与细节见 docs/OPERATIONS.md §1
nohup  bash run_strata.sh > strata-prod.log 2>&1 &
```

### 0.4 本次踩到的坑

| 坑 | 现象 | 修法 |
|---|---|---|
| **显存竞态** | 停 v4 后立刻起 v5 → `cudaMalloc` OOM，业务挂 5 分钟 | 轮询 `nvidia-smi` 到 used <1000 MiB 再起（已写进 `stop_strata.sh`/`switch_engine.sh`） |
| **模板被覆盖** | v5 用 GGUF 原装模板，同一 prompt 渲染 2378 vs v4 的 2336 | 把 froggeric v22.5 装进 pack，并把守卫**挂在启动器**上（`setup` 每次都会重新解出） |
| **`./setup.sh` 绕过守卫** | 直接跑 setup 会自己拉起模型，模板变回原装 | 一律用 `run_strata.sh` 起服务 |
| **ESP 家族门** | Unsloth 家族被上游 `left off`，`--experimental-speed-projection on` **静默不生效** | 原文是 **"not tested"** 非不兼容；引擎 cvec 对量化无关 ⇒ [patch 07](../patches/07_strata_esp_family_whitelist.patch) 把 `unsloth` 加进白名单（**必须在 `setup.sh` 之前打**；原件备份 `setup.py.orig-esp`） |
| **默认思考档误判** | 我据 Strata 文档以为默认从 medium 变 high | 实测**没变**：froggeric 模板第 17 行 `_default_reasoning_effort='medium'` 自己定默认，且引擎不覆盖它 |

### 0.5 验证结果（2026-10-13）

- `:8000` 与反向代理入口均 HTTP 200；端到端推理返回正常，**0.385 秒**
- **破限生效判据行**：`control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`
- 并发生效：`--batch: 4 slot sessions on CUDA0 (0.86 GiB each)`、`batch windows of up to 4 sequences`
- 上下文：`/v1/models` 报 `n_ctx: 262144`
- 模板对齐：同一 prompt 渲染 token 数 2336（= v4）
- 默认思考档：不指定 == 显式 medium（思考字符数逐字吻合）
- 破限强探针（14 条，ESP 逐请求切换）：**关 = 14/14 拒答，开 = 0/14**；v4 abliterated 亦 0/14

### 0.6 未验证

长上下文（128K/256K）实测、多日稳定性、8 路以上并发扩展性、`switch_engine.sh` 的真实切换+回滚路径、
质量标准样本（仅 15 题数学 15/15，不构成质量结论）。

---

## 0b. v5.1：0.1.43.1 → 0.1.45（2026-10-18）

> 这一节**不推翻 §0**：路线没变（Strata + unsloth UD-Q4_K_XL + 自带 ESP 破限 + `k8v4` 流式 KV），
> 只是引擎从 `0.1.43.1` 升到 `0.1.45`。下面的每个数字都是**同一台机器、同一份 config、同一批 prompt、
> 两个 arm 各自全新启动**的结果，不是跨时间的快照。

### 0b.1 升之前先看上游说了什么

`v0.1.44`（10-13）、`v0.1.44.1`（10-14）、`v0.1.45`（10-17）三个 tag，`v0.1.43.1...v0.1.45` 共 **546 个 commit**。
与我们这条路线真正相关的只有四条：

| 上游变化 | 对我们的意义 |
|---|---|
| **K-quant 专家的 prompt 内核成为默认**（`-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON`，另出 `libstrata_kq.so`） | 我们的 pack **就是 K-quant**（UD-Q4_K_XL）。0.1.43.1 的引擎是**缺这两个编译开关**编出来的。注意：运行期的 `STRATA_MMQ_KQ=1` 仍是 opt-in，上游自己测出 UD-Q4 上它 KL 更差（0.095 vs 0.081），**所以我们没开**。 |
| **#2228 修好**：长对话里的图片按字节哈希匹配，不再丢缓存状态 | 正中我们「vision + 长上下文 + conversation cache」的组合 |
| 0.1.45 的新解码内核（bit 与旧内核相同）+ 更大的默认 prompt chunk | 收益要靠自测；我们仍然显式写 `--prefill auto:32768` |
| `MIN_ENGINE = (0, 1, 45)` 与 `stale_binary()` 检查 | **server.py / 引擎 / vision 三个组件必须同版本**，不能只换一个 |

同时记下三条**没有采用**的东西，以免以后重复讨论：

- **`STRATA_HC_Q8=1`**（把 hyper-connection 从 Q8_0 字节直读，上游称 KL 0.0562→0.0526、每窗口 −2 ms）：
  上游是用 setup 把它写进 config 的 `env`，不是引擎默认；而且**开放 PR #2997 正是在修这条路径的
  `q8_inject` guard**。⇒ 先不碰，等它合入再做单变量实验。
- **`STRATA_ROUTE_TAIL_SKIP=0`**（专家全在 RAM 时上游建议关掉）：我们确实是全在 RAM，
  但代价是 decode −4~10%，而我们更在意 decode。⇒ 记为可选实验，没开。
- **`--vision-tokens` / `vram_elastic`**：现有 `vision.max_tokens = 1024` + 常驻 GPU 编码器已够用；
  elastic 编码器还有个「每轮都白叫一次 encoder」的未修 bug（#2454）。

### 0b.2 怎么做（不停机构建 → 冷启动 A/B → 换二进制）

1. **并行构建，不碰生产树**：`git worktree add ../Strata-0.1.45 v0.1.45`，把 `third_party/llama.cpp`
   软链过去（worktree 里没有这个未跟踪的大目录），再把 patch 07 打上（0.1.45 里行号漂到 `setup.py:6643`，
   上下文一字不差）。生产树此刻仍在跑 0.1.43.1，**零停机**。
2. **只调 `build_engine()`，绝不跑 `update.sh` / `setup.py` 主流程**。两个理由都很硬：
   `carry_over()` 明说「手工加进 `args` 的引擎选项不会被合并」（我们会丢 12 个手调参数）；
   而 #2902（open）指出 setup 对大分片会算错 SHA-256 **并删掉正确的分片文件**（最长 50 GB 一个）。
   构建脚本见 `scripts/build_engine_upgrade.py`。
3. **冷启动 A/B**：两臂各用「全新启动 + 一次 warmup」，配置逐字相同，只有 `exe` / `vision.exe` 指向不同目录。
   测量用客户端 SSE 计时（不引用引擎自己那行自相矛盾的 `done: N tokens in T s`）。
4. **换二进制**而不是重新 `setup`：`cp -a engine engine-0.1.43.1` 留档 → 把新 `engine/` 整体拷进去
   （**新版本多了 `libstrata_kq.so`，只拷两个可执行文件是不够的**）→ `git checkout v0.1.45` → 重打 patch 07。

### 0b.3 实测（两个 arm 各自全新启动，同 config 同 prompt）

| 指标 | 0.1.43.1 | **0.1.45** | Δ |
|---|---:|---:|---:|
| 短上下文 decode（6 次中位） | 139.7 t/s | **144.5 t/s** | **+3.4%** |
| 短 prompt TTFT | 0.453 s | 0.455 s | ±0 |
| 1K 上下文 TTFT | 1.88 s | **1.58 s** | **−16%** |
| 32K 上下文 TTFT | 9.50 s | **6.99 s** | **−26%** |
| 100K 上下文 TTFT（3 次中位） | 24.78 s | **21.82 s** | **−12%** |
| 32K 上下文 decode | 109.8 t/s | **111.4 t/s** | +1.5% |
| 100K 上下文 decode（512 token × 3 次中位） | 109.7 t/s | 109.0 t/s | −0.6%（噪声内，判等） |
| 4 路并发 512 token 墙钟 | 7.91 s | **7.04 s** | **−11%** |
| 4 路合计吞吐 | 64.6 t/s | **76.9 t/s** | **+19%** |
| 视觉三色条（红/绿/蓝） | 3/3 | 3/3 | 相同 |
| 专家缓存 | 32.28 GiB / 11054 | 32.27 GiB / 11050 | 相同 |
| 启动后空闲显存 | 2181 MiB | 2161 MiB | 相同 |
| **常驻 RAM 的专家** | 39.45 GiB | **52.10 GiB** | **+12.65 GiB** |
| 启动 WARNING 数 | 1（#669 prefill 块警告） | **0** | 更干净 |

两条**方法论**备注，避免以后误读：

- **100K 那次看似「decode −5.7%」是假象。** 第一轮探测在 100K 上只生成了 46 个 token（模型答得短），
  逐 token 的 draft 接受率抖动让 decode 在 80~143 t/s 之间乱跳。改成「要求 ~400 词、生成 512 token」后，
  两臂各 3 次复测才是 109.7 vs 109.0——**长上下文 decode 判等，TTFT 才是真提升**。
- 短上下文 TTFT 都是 ~0.45 s，而不是旧记录里的 0.19 s：0.19 s 那次命中了 conversation cache，
  这里两臂用的都是**全新 nonce prompt**，不可比。

### 0b.4 上游 0.1.45 在 sm_86 上有 decode 回退报告（#2936），我们的 sm_89 没复现

#2936 报道 RTX 3090（sm_86）上 0.1.45 比 0.1.43 **decode −20~27%**，而同一个人 3×2080 Ti（sm_75）的
交叉验证只有 −2.3%~−6.2%。⇒ 不是所有架构都中招，而 **sm_89（4090）在所有公开报告里都是空白**。
我们的结论只能来自自己的两臂实测：**短上下文 +3.4%、长上下文判等**，没有出现该回退。
如果以后换卡或换架构，这条要重新测，不能沿用。

### 0b.5 还有哪些「升级后要盯」的

- **`--expert-cache auto` 的落点变了**：两臂都是 32.27~32.28 GiB / ~11050 槽（因为 `--vram-reserve-mib 2560`
  是显式写的），但上游有 #2897 / #2856 两例「升级后 OOM / 缓存从 41 掉到 14」。**每次升级都要核对这两个数**：
  缓存 < 31 GiB 或空闲显存 < 1800 MiB 就判定不合格。
- **常驻 RAM 专家 +12.65 GiB**：这是 0.1.45 自己算出来的（内存 247 GB 够），不是我们设的。
  如果哪天内存吃紧，它会退回去；不需要改配置。
- **#2934 / #2923 / #2785 仍未修**（长会话里 parking 把已缓存的对话全赶走 / 智能体任务结束后缓存被清空）：
  这些是**旧版本就有的行为**，不是本次升级引入的，别把锅算到 0.1.45 头上。

### 0b.6 回滚（约 3 分钟）

```bash
bash strata/stop_strata.sh
cp -a strata/Strata/engine-0.1.43.1/. strata/Strata/engine/     # 整目录，含 BUILD.json
git -C strata/Strata checkout v0.1.43.1
git -C strata/Strata apply qwen4090-repo/patches/07_strata_esp_family_whitelist.patch   # 若行号不对，按 0.1.43.1 的 @@ -5419 手工改
bash "/path/to/start_strata_kill_gui.sh"    # 必须在自己的终端里跑，见 OPERATIONS.md §1
```

配置**不用改**：本次没有往生产 config 里加任何 `env` 项。

---

## 0c. v5.2：换模型 —— UD-Q4_K_XL → DavidAU abliterated IQ3_S（2026-10-18）

> 这一节**不推翻 §0b**：引擎还是 0.1.45，参数几乎全同，**换的是权重**。
> 目标模型：[DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF](https://huggingface.co/DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF)
> （IQ3_S，84 GB 两片；**本身已 abliterate ⇒ 不再需要 ESP 控制向量**）。

### 0c.1 为什么值得换

1. **它自带的 abliteration 替代了我们的 ESP 补丁**：不需要 `--control-vector-*`，也就不再依赖
   [patch 07](../patches/07_strata_esp_family_whitelist.patch) 的那道家族门（补丁仍保留，仅用于回滚/重跑 setup）。
2. **3.50 bpw 但质量几乎不掉**：作者/ISTA 的实测（对 BF16 基线）IQ3_S `AIME25 100.00 / GPQA-D 92.93 /
   LCB v6 86.86 / task avg 93.26`，BF16 是 93.12。
3. **作者本人就是指着 Strata 推荐跑这个模型的**（模型卡 SUPER CHARGER 段直接链到 Niko1221/Strata）。
4. 模型小 27 GB（84 vs 111）⇒ 更多专家进显存、更少读盘、常驻内存更省。

### 0c.2 A/B 实测（同机、同参数、各自全新启动；停服约 35 分钟）

| 指标 | UD-Q4_K_XL + ESP | **DavidAU IQ3_S** | Δ |
|---|---:|---:|---|
| decode（短上下文 6 次中位） | 144.5 t/s | **159.5** | +10.4% |
| TTFT 短 / 1K / 32K / 100K | 0.455 / 1.58 / 6.99 / 21.76 s | **0.407 / 0.81 / 6.14 / 19.01 s** | −11% / **−49%** / −12% / −13% |
| decode 1K / 32K / 100K | 103.2 / 111.4 / 111.4 t/s | **156.6 / 148.9 / 145.5** | +52% / +34% / +31% |
| 4 路并发墙钟 / 合计 | 7.04 s / 76.9 t/s | **4.98 s / 103.3 t/s** | −29% / +34% |
| 温和拒答 22 条 | 1/22 = 4.55% | **0/22 = 0%** | 更好 |
| 强探针 14 条 | 0/14 | 0/14 | 平 |
| 数学（难 8 / 易 15） | 8/8 · 15/15 | 8/8 · 15/15* | 平 |
| 指令 6 / 代码 9 / 32K 找针 2 | 6/6 · 9/9 · 2/2 | 6/6 · 9/9 · 2/2 | 平 |
| 视觉三色条 | 3/3 | 3/3 | 平 |
| 专家缓存 / 常驻 RAM | 11054 槽 / 52.10 GiB | **17422 槽 / 26.36 GiB** | 更好 / 更省 |
| 启动空闲显存 / WARNING | 2161 MiB / 0 | 2231 MiB / 0 | 平 |

\* 易数学那条唯一的"错"是 `6.00` 与 `6` 的字符串比较，答案本身正确 ⇒ 实际 15/15。
MTP 复用基座草稿头仍然有效：`drafts accepted 79 of 94`。

### 0c.3 部署要点（四个都会让服务起不来）

1. **`--ple-gguf` 必须指向 shard2**。这个模型的 28.80 GB PLE 表（`per_layer_token_embd.weight`，
   shape `[160, 320001536]`）**独占第二片**；shard1 里 1223 个张量中没有它。写错就是启动失败。
2. **建 pack 只调工具，不跑 setup**：
   `.venv/bin/python tools/iq_pack.py --gguf <shard1> --out <pack> --compat-bf16`
   实测 `compat-bf16: 0 tensors, 0.00 GiB` —— 这个 quant 的小型投影本来就存成 BF16/F16，
   **一个张量都不用转换**（比 Unsloth 的 UD-Q4_K_XL 干净）。
3. **GGUF 里没有 MTP 张量**（作者另有 `…-MTP-GGUF` 仓库）。按 [Strata docs/ORCA.md](https://github.com/Niko1221/Strata/blob/v0.1.45/docs/ORCA.md) 的做法复用
   基座 `mtp/rt` 草稿头，由目标模型自己校验草稿；`drafts accepted` 正常。
4. **绝不能跑 chat 模板守卫**。模型自带 221,835 字符的模板（27 推理 + 27 指令模式 + AI Constructs）。
   > 📌 2026-10-19 起已改为**合并模板**（29 段正文并进 froggeric）；2026-10-22 的 v3 为 **249,545 B**（再加 8 个 xhigh 组合键），见 [TEMPLATE.md](TEMPLATE.md)；
   > 本条记录的是 10-18 换模型当时的做法。
   `iq_pack.py` 已从 GGUF 原样导出到 `pack/tokenizer/chat_template.jinja`；
   旧的 `run_strata.sh` 会用 froggeric 模板覆盖它，等于把这个模型的核心能力删掉。
   ⇒ 生产启动器已去掉守卫，备份 `run_strata.sh.bak-undq4-20261018`。

另：服务端**不校验 `model` 字段**，客户端继续用老的 id 也能 200（实测 `PROD_SWITCHED`），**不需要改客户端**。

### 0c.4 决定与清理

按"哪个好留哪个"：**留 DavidAU，删 UD-Q4_K_XL**（用户确认）。

已删除：`models/UD-Q4_K_XL/` 104 GB、`Strata-data/packs/unsloth-ud-q4_k_xl/` 1.4 GB、
10-13 的两个残片 `models/IQ3_S/` 3.3 GB 与 `models/UD-IQ4_XS/` 828 MB。
**磁盘 340 GB → 449 GB 可用**。旧生产配置移到 `backups/strata-unsloth-ud-q4_k_xl.json.disabled`（防止误启动）。

⚠️ 代价：回滚到 UD-Q4_K_XL 需要重新下载 104 GB，没有本地副本。

---

## 0d. v5.3：0.1.45 → 0.2.2 服务端 + 0.1.47 引擎（2026-10-22）

> 上游在 10-19 到 10-21 连发 0.2.0-beta / 0.2.1（+4 个补丁版）/ 0.2.2。这一节只记「哪些与我们有关、
> 怎么升、升完实测到什么」。**模型、pack、启动参数都没换。**

### 0d.1 上游改了什么（只列影响我们的）

0.2.x 把 Strata 拆成 app / Terminal / engine 三部分；engine 从 0.1.45 升到 0.1.46（0.2.1）再到 0.1.47（0.2.2）。

| 变更 | 对我们的影响 |
|---|---|
| **Linux 预编译引擎**（签名 feed，0.1.47 包 arch 含 sm_89） | ✅ 不用再自建，省掉一次编译和 patch 07 |
| 图片默认开启；<24 GB 卡用 CPU 读图 | 我们显式 `--vision`，不变 |
| 文件层 prompt chunk 按 RAM/SSD 重新计价（0.2.0） | 长 prompt 读取速度可能变；实测 4.8K prompt 3,287 t/s |
| 解码默认调优表（sm_86 / sm_60、Gyro、P100） | **sm_89 不在表里**，4090 无预期变化（实测持平） |
| `#3465` 长 prompt 出现 NaN 后重试整段 | ✅ 长上下文更稳 |
| `#3488` 复用的 PID 不再永久卡住加载 | ✅ |
| 工具调用：模型写错开头（`<NAME>` 等）能纠正 | ✅ |
| **Chat Completions 未声明 tools 时不再返回 `tool_calls`** | ⚠️ 行为变化；我们客户端都声明 tools，实测正常 |
| `thinking_budget_tokens` 被读取、Anthropic usage 修正 | ✅ |
| `setup.py` 的 `MIN_ENGINE = (0, 1, 46)` | 服务端/引擎必须同代 |

### 0d.2 为什么用「预编译引擎 + 源码服务端」，不跑 setup

1. `setup.py` 会从 GGUF 重导**原装模板**，覆盖合并模板（TROUBLESHOOTING #49）；
2. `carry_over()` 会重写 run config，**丢掉手加的参数**；
3. 0.1.44+ 的分片 SHA 校验有已知 bug（#2902），**可能删掉正确的模型分片**；
4. 引擎包是签名 feed 的产物，sha256 与 feed 逐字节一致（`45a6df98…b85895e`），比自己编更接近上游测试过的组合。

只换 `serve/`（官方 tag 原样）与 `engine/`（官方包），模型 / pack / config / template 都不动。

### 0d.3 实测

| 指标 | 0.1.45 | **0.1.47** |
|---|---:|---:|
| 停机时间 | — | **48 秒**（09:36:05 停 → 09:36:53 可服务） |
| 启动到就绪 | 39 s | **36 s** |
| 显存（含 vision） | 46,435 MiB | **46,303 MiB** |
| `/props` 模板字符 | 243,052 | **243,742**（v3 模板 +690） |
| 4,816 token prompt 读取 | 5,951 @ 2,538 t/s（当时口径） | **3,287 t/s** |
| decode（短回答 8–160 token） | 112–157 t/s | **99–141 t/s**（不同请求长度，噪声内持平） |
| 专家缓存命中 | 98.6–99.0% | **98.6%**（105,571 / 107,098） |
| tool_calls / Anthropic / 视觉 | ✅ | ✅（0.2.2 服务端 + 0.1.47 引擎） |

**结论：速度没退，也没有可归因于 0.2.2 的提升（sm_89 不在上游调优表里）；升级收益是稳定性修复与新服务端行为。**
严格 A/B（同 prompt、同 config、两臂各跑一遍）没做——需要再停两次服，收益不足以抵消业务中断；
当前判据是「同 workload 的 decode 不下滑 + 全部验收项通过」。

### 0d.4 回滚

```bash
bash strata/stop_strata.sh
cp strata/run_strata_0145.sh strata/run_strata.sh      # 启动器指回旧树 + 旧引擎
bash strata/run_strata.sh
```

旧树 `strata/Strata/`（tag v0.1.45 + 引擎 0.1.45）与旧 config 都没动过，回滚不需要重新下载或编译。
模板 v3 与引擎无关；要连模板一起回 v2：`cp chat_template.jinja.v2 chat_template.jinja`（然后重启）。

---

## 1. 为什么切到 FreeToken（v4，2026-09-24，**已于 2026-10-18 退役**）

v3（sglang + lk_moe）的瓶颈是**架构性**的：experts 在 CPU 上算，decode 速度受 CPU 算力与单步同步延迟限制，
长上下文 decode 卡在 ~42 t/s，加载要 ~5.5 分钟。FreeToken 恰好是为「权重驻内存 + CPU 算 experts」
这一形态设计的 MoE offload 运行时，同机实测：

- **长上下文 decode +49~53%**（73K：42.0 → 62.6~63.6 t/s；145K：39.4 → 60.2 t/s）
- **254K prefill +22%**（1710 → 2081 t/s）
- **加载时间 ~8×**（~5.5 分钟 → **~39 秒**）
- 原生提供 **Anthropic 兼容 API**（`/v1/messages`），sglang 路线没有

**代价也是明确的**（详见 §4 取舍表）：8K prefill 明显变差且方差极大、短请求有 ~3.4s 固定开销、
≥4 并发不再扩展、多流冻结问题更严重。**这是一次按场景的取舍，不是全面升级。**

## 2. 迁移方法

```bash
# 1. 独立 venv（uv 管，不碰既有的 conda 环境）
uv venv ~/ft-venv --python 3.12
uv pip install --python ~/ft-venv "freetoken[accel]"

# 2. 打三个本地补丁（未上游化；install 脚本幂等，升级后重跑即可）
bash scripts/archive/install_freetoken_patches.sh ~/ft-venv/lib/python3.12/site-packages

# 3. 启动（脚本自带补丁自检 + 就绪等待；sudo/memlock 是保守默认，实测非必需——见 TROUBLESHOOTING #14）
bash scripts/archive/start_freetoken.sh
```

`accel` extra 会拉入 flashinfer-python[cu13] 0.6.18 与 sglang-kernel 0.4.5（**注意比 v3 要求的 0.4.7 低**——
两者服务于不同的引擎路径，不冲突）。版本硬约束：`torch>=2.11,<2.12`、`transformers>=5.16,<5.17`、`triton==3.6.0`。

**三个补丁各解决一个问题**：

| 补丁 | 解决什么 | 不打的后果 |
|---|---|---|
| [04 rendezvous 自动避让](../patches/04_ft_auto_dist_port.patch) | 内部 rendezvous 默认监听「API 端口+1」，该端口被占用（反代在相邻端口 / 第二个实例 / 上个进程没退干净）就启动失败 | 相邻端口被占即无法启动 |
| [05 qwen3.8 workload](../patches/05_ft_workload_qwen38.patch) | 注册本模型的专家几何（2.64MB，`ft bench bw` 默认按 7.61MB 测） | `ft bench bw --model qwen3.8-flash-next` 跑不了 |
| [06 decode-interleave](../patches/06_ft_decode_interleave.patch) | prefill 无条件优先导致长 prompt 整块占住 GPU | 正在解码的流最长冻结 17.7s「不吐字」 |

## 3. 本次踩到的坑

| # | 现象 | 根因与修法 |
|---|---|---|
| 1 | 就绪判断失效：`/health` 在模型加载完成前就返回 200 | 上游 **#537**。就绪判据改用日志行 `ready to serve`（加载期还会看到 `POST /v1/chat/completions` 被拒） |
| 2 | 首个 >8K 的 prompt 直接 `HTTP 400 Bad Request` | 默认 `auto` 只规划 8256 token 的 KV。必须显式 `--num-tokens 265216`（≈256K+3K） |
| 3 | 加完 `--num-tokens` 后 prefill OOM、worker 崩 | 专家缓存先占满，没给 KV 留空间。必须**同时**给 `--moe-cache-rate 0.40`（上游 **#150 / #401**） |
| 4 | 跑过 `ft bench bw` 后，`--moe-strategy auto` 静默变成 hybrid，实测慢 **27%** | `auto` 会采纳 benchbw profile，而该 profile 是在**错误几何**上测的（见 §2 补丁 05）。**显式固定 `--moe-strategy offload`**（上游 **#151 / #436**） |
| 5 | 并发上不去，mrr 调到 8 也没用 | 并发上限由 **mamba 槽**决定：`slots = 4*mrr + max(4, 2*mrr) + 1`。mrr=2 → 12 槽（3 路长上下文），mrr=4 → 24 槽（6 路）。「mrr>3 没用」的说法是错的 |
| 6 | `ft ctl cache rebuild` 热调后服务卡死 | 上游 **#526**：重建 OOM 会 wedge 服务器且无法回滚。**改参数一律重启**（只需 40 秒，热调不值得冒险） |
| 7 | 升级 FreeToken 后补丁静默失效 | 三个补丁都打在 site-packages 里，`uv pip install -U` 会覆盖；且 `RECORD` 不会因 patch 更新，`pip check` 看不出异常。启动脚本已内置自检，缺补丁直接拒绝启动 |

## 4. 验证结果（2026-09-24，客户端计时）

> ⚠️ **测性能前先确认窗口独占**：服务若对外开放，其他客户端会不定时打进来，实测能把 145K decode 从 39.4 压到 8.8 t/s。
> 查日志中 `#running-req` 与 HTTP 行确认（见 TROUBLESHOOTING #18）。**8K prefill 的方差尤其大，不要引用单点值。**

| 类别 | 项目 | v4（FreeToken） | v3（sglang v0.5.20） | 结果 |
|---|---|---|---|---|
| 性能 | decode @73K | **62.6–63.6 t/s** | 42.0 | ✅ +49~51% |
| 性能 | decode @145K | **60.2 t/s** | 39.4 | ✅ +53% |
| 性能 | prefill 254K | **2081 t/s**（122s） | 1710（149.2s） | ✅ +22% |
| 性能 | prefill 64K | **2140 t/s** | 1887 | ✅ +13% |
| 性能 | decode 短（512） | **64.4 t/s** | 42.5 | ✅ +52% |
| 性能 | prefill 8K | 371–1566 t/s（方差极大） | 1968–1994 | ❌ ~0.2–0.8× |
| 性能 | 并发 1/2/4/8（聚合，含 TTFT） | 35.6 / 54.0 / 64.2 / 63.7 | — | ⚠️ 见下 |
| 性能 | 并发 1/2/4/8（每流纯解码） | 58.1 / 48.0 / 49.5 / 49.0 | 单流 42.5（短上下文） | ✅ 单流远优；高并发扩展性 v4 弱 |
| 性能 | TTFT（短请求） | 3.4s（全命中 ~4.0s） | 0.42s | ❌ 8× |
| 性能 | 多流最长冻结 | 17.7s → **7.5s**（打 #484 后） | 切成 4s 块插队 | ❌ 仍劣 |
| 资源 | 模型加载 | **~39 秒** | ~5.5 分钟 | ✅ ~8× |
| 资源 | GPU 显存 | 44–46GB / 48GB | 41.7GB | 略高（moe-cache 让位） |
| 资源 | host RAM 峰值 | ~200GB（126GiB + 63.3G 专家 banks） | ~200GB | ≈ |
| 功能 | Anthropic `/v1/messages` | ✅ 原生 | ❌ | ✅ |
| 功能 | 256K 上下文 | ✅ 265216 token 池 | ✅ | ≈ |
| 部署 | SM89 专用 flash_attn wheel | 不需要（走 flashinfer） | 必需（PR #2751） | ✅ 更简单 |

> ⚠️ **并发数两个口径不可混用**：v4 的「聚合」= 总 token ÷ 墙钟（含 TTFT，故 N=1 反而低于单流速率），
> 「每流」= token ÷ 各 token 间隔之和（纯解码）。v3 的历史值（42.5/47.7/72.3/108）**不含 TTFT**——
> 因此本表**不把两边的并发聚合并排比**；可比的是单流纯解码：v4 58.1 t/s vs v3 42.5 t/s。
> 结论：v4 **单流更快，但并发扩展性更差**。

调优过程中确认的几个单变量结论（详见 [BASELINE.md](BASELINE.md)）：
`moe-cache-rate` 0.35→0.40 使 73K decode **+4%**；`FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` 使 mamba 池
1.3G→675M 且 decode 不降（65.1）；mrr 2→4 把并发上限从 3 路提到 6 路、decode 不变。

## 5. 历史：v3（2026-09-23，官方 sglang v0.5.20 基线）

<details>
<summary>展开 v3 路线记录（保留作参考；脚本与补丁仍在本仓库）</summary>

### 5.1 为什么升级到 v0.5.20

- **v0.5.20（2026-09-18）是首个正式收录 Qwen3.8-Flash-Next 的 release**（PR #37500 于 9/8 合入 main，v0.5.19 不含）
- 合入后 main 上又落了 4 个本项目直接受益的修复，v2 栈（PR 分支快照）里没有：
  - `#38851` QSA 分页 sparse-decode gather 内存安全（zero-fill scratch / int64 偏移 / gather 时反量化 FP8）
  - `#38855` sparse prefill 对 FP8 缓存前缀反量化
  - `#39446` compress gather 行数钳制（reland）
  - `#39474` 复用 alt_stream（修 decode overlap 路径反复建流的泄漏）
- **Lsglang v1.5.6 不可用**：其 `dsv4.1-lkmoe-sm80plus` 分支不含 `qwen4_exp.py`（主线转向 DeepSeek-V4.1），Flash-Next 无现成 wheel → 仍需自建
- 上游 PLE 修复 `#39928`（meta device 方案）于 **9/20** 合入 main，晚于 v0.5.20 → v3 **仍需**本地 PLE 补丁
- 官方硬件矩阵依旧**不含 4090/SM89**——v3 靠 lk_moe 混合推理 + PLE 补丁，属社区路线

### 5.2 升级方法（v0.5.20 基线重放）

构建仓是**浅克隆**（与上游无共同祖先）→ 不能 `git merge`。改用「以官方快照为基线、重放补丁」：

```
git fetch --depth=1 https://github.com/sgl-project/sglang tag v0.5.20   # 只取快照
git checkout -b v0520-lkmoe v0.5.20
git apply patches/01_lk_moe_v0520.patch          # lk_moe 集成（15 文件，直接可打）
git apply patches/02_ple_cpu_alloc.patch         # PLE CPU 分配（48GB 卡必需）
git apply patches/03_resolved_files_v0520.patch  # 2 处人工裁决差异
python -m build --wheel --no-isolation           # SGLANG_BUILD_RUST_EXTS=none
```

全流程见 [`scripts/archive/build_upstream_flashnext.sh`](../scripts/archive/build_upstream_flashnext.sh)。
**三个补丁在干净 v0.5.20 上均已验证直接可打**（无 3-way、无冲突）。

`03_resolved_files_v0520.patch` 覆盖两处需要判断的差异：
- `pyproject.toml`：取上游依赖版本（`sglang-kernel==0.4.7` 等）+ 追加 `lk_moe==2.4.1` + 包名 `lsglang` / 版本 `1.6.0+flashnext.v0520`
- `quantization/modelopt_quant.py`：**保留 lk_moe 的 CPU 常驻层跳过**（`is_gpu_resident_layer` 判定）**并接入上游新增的 megamoe 分支**

> 勘误（2026-09-24）：本文早期版本在此处声称 03 号补丁还含 `configs/qwen3_asr.py`，且称 01 号补丁为 18 个文件——
> 实际分别是 **2 个文件**和 **15 个文件**。

### 5.3 v3 踩到并修掉的 4 个坑

| # | 现象 | 根因与修法 |
|---|---|---|
| 1 | 启动即 `Exception: sglang-kernel is installed with version 0.4.6.post1, which is less than the minimum required version 0.4.7` | v0.5.20 启动强校验。**只升这一个包**（`pip install sglang-kernel==0.4.7`，dry-run 确认无连带升级） |
| 2 | `RuntimeError: get_global_server_args() is retired` | v0.5.20 废弃该 API。lk_moe 补丁里 1 处（`FusedMoE.get_max_num_group_batch_size`）→ 改 `get_schedule().chunked_prefill_size` |
| 3 | 从 root 外壳跑启动脚本时，子进程 `PermissionError: [Errno 13] Permission denied: '/root'` | Python multiprocessing spawn 的 worker 会 `os.chdir(父进程 cwd)`；root 外壳 cwd=`/root`。**启动脚本加 `cd` 到公共目录**（已内置） |
| 4 | 构建产物异常 | ① 需 `setuptools-rust`/`setuptools-scm`；② 必须 `SGLANG_BUILD_RUST_EXTS=none` 保持纯 Python 包；③ 打包前清 `python/build`（残留会让 wheel 从 20MB 涨到 76MB） |

### 5.4 v3 验证结果（2026-09-23，独占窗口对照 v2）

| 类别 | 项目 | v3 实测 | v2 基线 | 结果 |
|---|---|---|---|---|
| 功能 | /v1/models、对话、工具调用、thinking、effort 别名 | — | — | ✅ |
| 性能 | prefill 8K / 64K | 1968–1994 / 1887 t/s | 1874 / 1923 | ✅ |
| 性能 | 254K 冒烟（255k tokens） | 149.2s | 142.5s | ✅ 96% |
| 性能 | decode 512（短） | 42.5 t/s | 41.9 | ✅ |
| 性能 | decode @73K / @145K | 42.0 / 39.4 t/s（ITL p50 24ms） | 40.1 / 36.5 | ✅ |
| 性能 | 并发聚合 1/2/4/8 | 42.5 / 47.7 / 72.3 / 108 t/s | 39 / 44 / 65 / 97 | ✅ +8~11% |
| 正确性 | 98K 上下文针式检索（3 处不同深度） | 3/3 命中 | — | ✅ |
| 稳定性 | 20 轮 decode 显存曲线 + 20 分钟混合 soak | 零增长 | — | ✅ |
| 资源 | 加载 ~5.5 分钟、显存 41.7GB | 41.6GB | ✅ |

</details>

## 6. 历史：v2 / v1

<details>
<summary>展开 v2（2026-09-14，Lsglang 分支 + PR #37500）与 v1（Lsglang 1.4.13）</summary>

### 6.1 v2 路线

`Lsglang 0.5.19-lkmoe` 分支 merge 上游 PR #37500 → PLE 补丁 → 自建 wheel（`lsglang-1.5.0+flashnext.merge2/3`）。

**当时的三个拦路问题**（第 1、3 条对 v3 仍适用）：

- **PLE n-gram 表 GPU 瞬态分配 OOM（不改必炸）**：上游 `Qwen4ExpPLELayer.__init__` 先在 GPU 构造整张表
  （47.7GiB）再搬到 pinned 主机内存——48GB 卡没有路径能过，且该 GPU 占位随后被 `del`，纯浪费。
  显式 `--ple-offload-embedding` 无效（构造阶段就炸）。修复见 `patches/02_ple_cpu_alloc.patch`。
  （上游 9/20 的 #39928 用 meta device 解决同一问题，晚于 v0.5.20。）
- **pinned 内存的 memlock 限制**：pinned PLE 表需 47.7GiB 锁页内存，系统默认 `ulimit -l` 常仅 8MB。
  修复：启动脚本内置 `ulimit -l unlimited`。
- **tilelang 0.1.12 编译不兼容**：CUDA graph 捕获期报 `CUDA compiler and CUDA toolkit headers are incompatible`。
  修复：锁 **tilelang==0.1.11**。

v2 验证（2026-09-14）：12/12 通过；decode 41.5–41.9 t/s、128K 38.7、prefill 8K/64K 1999/1937；
加载 254s、显存 41GB。测速曾因 GPU 59°C 误判为退步，冷却后复原（见 TROUBLESHOOTING #11）。

### 6.2 v1 路线

Lsglang 1.4.13（guqiong96 release `lsglang-v1.4.12` 内）+ lk_moe 2.4.0，直接装 release wheel
（`scripts/archive/install.sh`），最省事但功能最少。启动脚本为 `scripts/archive/start_lsglang.sh`。

</details>

## 7. 运维方式

| | **v5（当前，Strata）** | v4（回滚，FreeToken） | v3（sglang，历史） |
|---|---|---|---|
| 启动 | `run_strata.sh`——**必须用启动器**（它带 chat 模板守卫）；约 **50 秒** | `bash scripts/archive/start_freetoken.sh`（补丁自检 + 就绪等待 + 可选关 GUI）；约 **40 秒** | `bash scripts/archive/start_lsglang_upstream.sh`；约 5 分钟 |
| 就绪判据 | 日志 `ready:` **＋破限判据行** `control vector mode = project … layers 4..44 (41 steered)` | 日志 `ready to serve`（**不能用 `/health`**） | `/health` 200 |
| 停止 | `stop_strata.sh`（含显存释放判据） | `pkill -f "ft-venv/bin/ft serve"` | `pkill -f 'sglang serve'` |
| 日志 | `strata-prod.log` | `/tmp/ft.log` | `/tmp/lsglang.log` |
| 开机自启 | 无 | 无 | 无（本项目一贯手动启动） |
| 升级/重装后必做 | **确认 [patch 07](../patches/07_strata_esp_family_whitelist.patch) 仍在**——重跑 `setup.sh` 会用上游原件覆盖 `setup.py`，ESP 静默失效 | **重打三个补丁**（`scripts/archive/install_freetoken_patches.sh`） | 确认 tilelang 未被升到 0.1.12+、sglang-kernel ≥ 0.4.7 |

> ⚠️ **两代引擎互斥**：各占本卡 90%+ 显存，任何时刻只能活一个；**停旧后必须等 `nvidia-smi` used <1000 MiB 再起新**（见 §0.2）。
> 若日后 `uv pip install -U freetoken`，务必重跑补丁脚本——否则服务会在下次启动时被自检拦下。

## 8. 回滚

```bash
# ── 主力回滚：v5 → v4 ────────────────────────────────
stop_strata.sh                                            # 停 v5（含显存释放判据）
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # 轮询到 <1000 MiB 再继续
bash scripts/archive/start_freetoken.sh                           # 约 40 秒起来

# ── 反向：v4 → v5 ───────────────────────────────────
pkill -f "ft-venv/bin/ft serve"
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # 同样要等
bash run_strata.sh                                        # 约 50 秒起来

# ── 历史路径：v4 → v3（sglang 环境、wheel、脚本全部原样保留）──
bash scripts/archive/start_lsglang_upstream.sh                    # 约 4-5 分钟起来
```

回滚都很快（引擎侧无需重建环境），但**每次切换都必须等显存真正释放**——`ft_stop.sh` 打印"干净停止"时
`nvidia-smi` 仍可能显示 47,712 MiB 被占，直接起新引擎会 `cudaMalloc` OOM（见 §0.2）。

本仓库已收录 v4 的完整版本快照 [`freeze-freetoken-0.1.3.txt`](archive/freeze-freetoken-0.1.3.txt)
（102 个包，可直接 `uv pip install -r` 复现）。建议另外保留：旧启动脚本副本、以及各版本 wheel（sglang 侧）。
