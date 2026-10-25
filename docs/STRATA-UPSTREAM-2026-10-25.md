# Strata 上游升级评估：0.2.2 → 0.2.3 / 0.2.3.1 / 0.2.4 / 0.2.4.x

- 调研时间：2026-10-25 13:2x（Asia/Shanghai），纯只读（未接触生产、未启停引擎、未跑 setup）
- 数据来源：GitHub Releases API、`setup.py@v0.2.4` 源码、Issues/搜索 API、本地 0.2.2 源码树静态阅读
- 生产基线（不在本次复核范围，按已核实事实使用）：Strata **0.2.2 服务端 + engine 0.1.47**、DavidAU Qwen3.8-Flash-Next abliterated **IQ3_S**（3.5bpw / 83.86GB）、单 RTX 4090 48GB（sm_89）、`--batch 4 --max-context 262144 --kv k8v4 --kv-resident 32768 --spec 4 --batch-mtp --prefill auto:32768 --vision`
- 上游仓库：<https://github.com/Niko1221/Strata>

---

## 0. 结论（先给答案）

**暂不升级。保持 0.2.2 服务端 + engine 0.1.47。**

三条决定性理由：

1. **0.2.4 的全部宣传内容在 sm_89 上不可达。** 0.2.4 的 DeepSeek V4.1-Flash 走**独立引擎包 `strata-dsv41`**，其 CUDA 预编译包的架构列表是 `sm_86, sm_120, compute_120`，**不含 sm_89**；`setup.py --family deepseek41` 在 4090 上直接拒绝（"DeepSeek's engine is not available for this computer yet"）。见 [#3935](https://github.com/Niko1221/Strata/issues/3935)（open，2026-10-24）、[#3947](https://github.com/Niko1221/Strata/issues/3947)、[#3994](https://github.com/Niko1221/Strata/issues/3994)。→ 即使用户改用 DeepSeek 模型也跑不起来，而用户明确要求继续用现有模型。
2. **Qwen 侧的实际增量只有"通用引擎 0.1.47 → 0.1.48"，而这个增量没有任何 sm_89 + 4090 + IQ3_S 的 A/B 数据。** 0.2.3 正文明确："The engine packages are version 0.1.48 (0.2.2 had 0.1.47)"；0.2.4 正文明确："Your Qwen engine does not change: every engine package is still the 0.1.48 file, byte for byte."。即 **0.2.4 对 Qwen 用户零引擎增量**。唯一同架构邻证是 [#3966](https://github.com/Niko1221/Strata/issues/3966) 的 RTX 4060 Ti 16GB（sm_89）单卡 IQ3_S abliterated +15% decode（0.1.43→0.1.48），但基线是 0.1.43 不是 0.1.47，不能直接套。
3. **0.1.46+ 存在长上下文 VRAM 预算类回归的开放报告，且正好打在 sm_89 + 262144 ctx + `--spec 4` 的组合上。** [#3952](https://github.com/Niko1221/Strata/issues/3952)（open）：engine 0.1.46 在 sm_89、`--max-context 262144 --spec 4` 下 VRAM 被吃到 0 MiB、verify 窗口逐个崩塌、引擎自杀重启；同机同参 0.1.43.1 跑 238K 连续 7 轮无告警。我们现处 0.1.47 未复现，但这是"往 0.1.48 走"时最该先验证的一条。

风险收益比：升级收益 ≈ 一个未量化的通用内核小版本 + 一批我们用不到的 DeepSeek/AMD/Windows 修复；升级成本 = 262144 ctx + 流式 KV + 4 并发 + MTP 这套已验证配置的重新验证，以及一次业务停机窗口。**在出现 sm_89 单卡正向 A/B 之前，不值得动。**

---

## 1. 版本时间线与发布面

发布时间为 release API 的 `published_at`（UTC），括号内北京时间为 +8。

| tag | 发布（UTC / 北京） | 标题 | Control（Terminal） | 通用引擎 | dsv41 引擎 | 强推重跑 setup？ | 改 pack/GGUF/tokenizer？ | 对 Qwen3.8 / 非 DeepSeek |
|---|---|---|---|---|---|---|---|---|
| v0.2.2（现生产） | 2026-10-21 11:10（19:10） | Strata 0.2.2 | 0.1.47 | **0.1.47** | — | 否 | 否 | 无 DeepSeek |
| v0.2.3 | 2026-10-23 07:44（15:44） | Strata 0.2.3 | 0.1.48 | **0.1.48** | — | 仅 0.2.0-beta/0.2.1–0.2.1.2 需重跑安装命令；**从 0.2.2 只需 `strata update`** | 否（正文无格式变更条目） | 通用改进 + Strix Halo/Arc/Windows；**无 DeepSeek** |
| v0.2.3.1 | 2026-10-23 13:45（21:45） | Strata 0.2.3.1 | **0.1.48.1** | 0.1.48（byte-identical） | — | 否 | 否 | 只修 setup 在 Windows/Linux NVIDIA 包上的崩溃（[#3810](https://github.com/Niko1221/Strata/issues/3810)）；引擎无变化 |
| v0.2.4 → 0.2.4.2 | 2026-10-24 02:18（10:18），正文更新 09:20（17:20） | Strata 0.2.4: DeepSeek V4.1-Flash in the Terminal (Terminal 0.2.4.2) | **0.1.49 → 0.1.49.2** | **0.1.48（逐字节不变）** | **0.1.49 → 0.1.49.2** | 否（0.2.4→0.2.4.1 只需 `strata update`） | 仅 dsv41 新增 `pack_gguf --engram-meta`；Qwen 侧无 | Qwen 引擎零变化；桌面 app 停在 0.2.3 |

要点与易踩点：

- **`v0.2.4.1` / `v0.2.4.2` 没有独立 release 页**，它们是 tags，发布说明以"Update 0.2.4.1 / 0.2.4.2"的形式**追加在 v0.2.4 同一页**里。所以"看 0.2.4 一个页面"= 看 0.2.4/0.2.4.1/0.2.4.2 三个版本。页面：<https://github.com/Niko1221/Strata/releases/tag/v0.2.4>
- 0.2.4.1 修的是"Terminal 包漏了 DeepSeek 准备脚本"，0.2.4.2 修的是 DeepSeek launch/安装与 `install.sh` 不关终端。**两者都不涉及 Qwen 引擎**（"every Qwen engine package stays as it was"）。
- 0.2.4 的桌面 app 仍是 0.2.3（DeepSeek 只在 Terminal/`strata launch`/`strata setup`）。
- 0.2.3 的 Mac 备注里出现过唯一一句"最低引擎"式表述："Gyro on a Mac needs engine 0.1.47 or newer"——**仅 Mac/Gyro，与我们无关**。
- **`MIN_ENGINE` 没有抬门槛**：本地 `Strata-0.2.2/setup.py:244` 与上游 `setup.py@v0.2.4:249` 都是 `MIN_ENGINE = (0, 1, 46)`。即 0.2.4 也接受 0.1.46 及以上的引擎，我们的 0.1.47 天然满足。

---

## 2. 引擎版本与获取面

| 项 | 值 |
|---|---|
| 0.1.48 linux CUDA 13 | `strata-engine-0.1.48-linux-x86_64-cuda-cuda13.tar.gz`，sha256 `3e11a82e325cf4968a1734446e8945d7918a5657e964c6c5d2c262aa59258f57` |
| 0.1.48 linux CUDA 12 | `strata-engine-0.1.48-linux-x86_64-cuda-cuda12.tar.gz`，sha256 `c4caf94be52a4bf1fe7a55fccb3ae064312481df1b9216f6e518cacab3dc2400` |
| 0.1.48 架构 | `BUILD.json` 的 `archs: [75, 86, 89, 120]`（据 [#3935](https://github.com/Niko1221/Strata/issues/3935) 引用；现生产 0.1.47 的 `BUILD.json` 本地实测同为 `[75,86,89,120]`, `cuda: 13.0`, `vision: cuda13.0`）。**sm_89 在内。** |
| dsv41 0.1.49.2 linux CUDA 13 | `strata-dsv41-0.1.49.2-linux-x86_64-cuda-cuda13.tar.gz`，sha256 `918a2155ad6bad0ef6d54e57280dfc7cf6fe713838ca9303300bccaebf275283`；架构 `sm_86, sm_120, compute_120` |
| 控制包 | `strata-control-0.1.49.2-linux-x86_64.tar.gz`（sha256 `2f905fe9…3cf9c6`）— 我们不用 Terminal，无关 |
| 签名/校验位置 | release 资产里的 `strata-feed.json` + `strata-feed.json.sig`（签名 feed）、`SHA256SUMS.txt`、`install-manifest.txt`、`latest.json`。本次只核对了 `SHA256SUMS.txt` 内容，**签名算法与公钥未核**（见 §7） |
| 预编译包完整性 | 0.2.4 页面的 assets 同时挂着 0.2.3 的 AppImage/exe 与 0.2.4 的 dsv41 包，属于"整页重挂"，属正常行为，不代表 app 升到 0.2.4 |

**旧 server 能否配新引擎（0.2.2 server + 0.1.48 engine）**：静态阅读 `Strata-0.2.2/serve/`（`server.py` 等全部 `.py`）**没有发现任何 engine 版本门控/上限校验**——serve 只是读 `engine.info["version"]` 用于显示与目录命名，`MIN_ENGINE` 只存在于 `setup.py`（下载/校验用）。因此"只换 engine 目录、server 不动"在代码层面是成立的，也与我们 10-22 的做法同构。**但这是代码阅读结论，未实测**（§7）。

---

## 3. DeepSeek 特化 vs 通用：哪些改动对我们生效

判断基准：单 4090（sm_89）、Qwen3.8-Flash-Next IQ3_S、262144 ctx、`--kv k8v4 --kv-resident 32768`、`--spec 4 --batch-mtp`、`--vision`。

| 改动 | 归属 | 对我们 |
|---|---|---|
| DeepSeek V4.1-Flash 模型条目 / ISTA-DASLab IQ1_M/IQ2_XXS/IQ2_XS / Engram IQ4_NL 表 / DSpark drafter | dsv41 专用 | **不生效**（且引擎拒绝 sm_89） |
| RAM expert store、hot-expert 预置 profile、SSD 直读 | dsv41（12GB 卡的 MoE 流式） | 不生效 |
| "long prompts layer by layer in big chunks instead of 16 tokens" | dsv41 段内 | 不生效（我们的 prefill 走 0.1.47 的 `auto:32768` 路径，未见此改动回移通用引擎） |
| "prompt over ~3700 tokens skips work the last 20 layers cannot use（16K −34% 时间）" | dsv41（近似计算，799 位置里 99.5% 首选词一致） | 不生效。**注意这是会改变输出的近似优化**，即使将来回移也要谨慎评估 |
| thinking budget 默认 32768 | dsv41 请求侧 | 不生效（我们非 DeepSeek 家族，且客户端自管） |
| DSpark（Unsloth 小 draft） | dsv41 | 不生效 |
| AVX-512/Zen4-5、AVX2 的 CPU expert 精确行（+25% decode，Zen4 单机实测） | dsv41（CPU 端 expert） | 不生效（我们 expert 全在显存，38 核只做辅助） |
| RDNA4 matrix-core prefill（8K +36%、32K +49%） | AMD dsv41 | 不生效 |
| **0.1.48 通用内核（单卡 decode/prefill 改进，0.2.3 声明的 "faster…" 通用部分）** | 通用引擎 | **唯一相关项**，但官方未给 sm_89/4090 数字；邻证见 §4 #3966 |
| DFlash2 作为 MTP 的替代 drafter（`strata launch --drafting dflash2`，opt-in，仅 Terminal） | 通用引擎 + Control | 部分相关（我们吃 `--spec 4 --batch-mtp`）。但需新 Control 且官方说 Auto 仍选 MTP；[#3965](https://github.com/Niko1221/Strata/issues/3965) 实测 4060 Ti 上 DFlash2 比 MTP **慢 3.3%** → 不建议 |
| 引擎自带 tuning 表查找（不经 launcher 启动也生效） | 通用 | 相关且**略有利**：我们现在是手工 exec serve/engine，正好是"不经 launcher"的场景 |
| 会话缓存按空闲内存自动定尺寸（#3493，40% 剩余、上限 8GiB） | 通用/app 配置 | 我们显式指定缓存，不生效 |
| CPU worker / VRAM 保留默认值调整（5070 12GB 上 31→45 tok/s） | app "Work profile" 默认值 | 不生效（我们手工传参，且参数冻结） |
| 系统证书信任（#3722）、agent chat 满上下文处理、`/compact`、Terminal 稳定性一批修复 | 通用/前端 | 不生效（我们只吃 HTTP API） |
| Arc SYCL 发布、Strix Halo 8-bit hyper-connection（`STRATA_HC_Q8`）、Windows 8192 chunk（`STRATA_GFX1151_CHUNK`）、Mac Metal 修复 | 他平台 | 不生效 |

**关键词命中情况**（明确记录，避免"好像看到过"）：

- `Qwen3.8`：0.2.3/0.2.4 正文**零命中**。全部 Qwen3.8 信息来自 issue 基准帖（#4058、#4054、#3197、#1849、#3568 等）。
- `GGUF` / `K-quant`：0.2.3/0.2.4 正文**无针对 K-quant 的变更**。相关的仅是 0.2.4 dsv41 的 `--engram-meta` 打包，以及 open 的量化支持请求 [#4060](https://github.com/Niko1221/Strata/issues/4060)（bartowski quants）、[#4055](https://github.com/Niko1221/Strata/issues/4055)（ggml-org IQ4_NL）——都是 feature request，不是已交付能力。
- `sm_89`：正文零命中；issue 命中 8 条，全部是 dsv41 架构缺失（#3935/#3947/#3994）、sm_89 调优表（#3659/#3453/#3999）、sm_89 基准（#3452/#3952）。
- `4090`：正文零命中；issue 命中多条，均为**社区基准**或 **dsv41 拒绝**（#3935、#3964）。

---

## 4. 性能证据（区分官方 / 社区 / 无）

| # | 来源 | 硬件 | 模型/上下文 | 版本对比 | 数字 | 可信度 |
|---|---|---|---|---|---|---|
| 1 | **官方** 0.2.3 正文 | Strix Halo（gfx1151，Windows） | "IQ3_S model" | 0.2.2 → 0.2.3，3 对 | decode **+6%~+19%**（tool +19%、code +18%、64K +13%、8K +12%、4K +8%、JSON +6%）；prefill **+5%~+14%**（prefill 绝对值 685/783/749 tok/s @4K/8K/64K） | 官方自测，但 **AMD iGPU，不可外推到 4090** |
| 2 | **官方** 0.2.4 正文 | RTX 5070 12GB / RTX 3060 12GB / 1× 与 4× R9700 | DeepSeek V4.1-Flash 2.00bpw | 内部对比（mmap 1.8-2.2 → RAM store 4.3-4.9） | 5 / 8-10 / 11 / 31（55-69 带 DSpark）tok/s | 与我们的模型和卡**都不同** |
| 3 | **社区** [#3966](https://github.com/Niko1221/Strata/issues/3966)（open, 10-24） | 单 RTX 4060 Ti 16GB（**sm_89**） | Qwen3.8-Flash-Next-**GSQ-RCO-abliterated-IQ3_S**，32K ctx | **0.1.43 → 0.1.48**，各 8 轮 | 单卡 decode 20.6 → **23.7 tok/s（+15%）**；同帖 layer-split（4060Ti+3060, pw2）**47.7 → 38.0 tok/s（−19%）** | 社区受控 A/B，模型与我们同族、同为 sm_89 单卡 → **最有参考价值的一条**。但基线是 0.1.43，无法拆分"0.1.43→0.1.47"与"0.1.47→0.1.48"各占多少 |
| 4 | **社区** [#4058](https://github.com/Niko1221/Strata/issues/4058)（open, 10-25） | RTX 5090 32GB（sm_120） | qwen3.8-flash-next-iq3_s，262144 ctx，`--spec 4 --kv int8 --kv-resident 32768` | 仅 0.1.48 绝对值，无对照 | short chat **176.3 tok/s**（143-221）、TTFT **0.25s**；4K prompt 4722 tok/s；16K prompt 6442 tok/s；agent 缓存命中 69.5% | 社区，配置与我们最接近但其卡是 sm_120 |
| 5 | **社区** [#4054](https://github.com/Niko1221/Strata/issues/4054)（open, 10-25） | 单 RTX 3090 24GB（sm_86） | 原版 Flash-Next IQ3_S，**524288 ctx**，KV k8v4 + `--kv-grow`，spec 4、spec-min-p 0.70 | 仅 0.1.48 绝对值 | 86.9 tok/s decode（TTFT 0.73s）、4K 1102 tok/s、16K 1675 tok/s、follow-up decode 94.4 tok/s | 证明 0.1.48 在 24GB 单卡 + k8v4 + 512K 上稳定可用 |
| 6 | 社区 [#3984](https://github.com/Niko1221/Strata/issues/3984)（open, 10-24） | RX 6700 XT 12GB（gfx1031，Ubuntu 26.04） | Qwen IQ2_XS，128K，engine 0.1.48 | 仅绝对值 | 58 tok/s decode、700 tok/s prefill | 旁证：0.1.48 在老 AMD 卡也可用 |
| 7 | 社区 [#3978](https://github.com/Niko1221/Strata/issues/3978)（open, 10-24） | 2× RTX 5060 Ti 16GB，IQ3-S（dsv41） | 0.2.4.2 / engine 0.1.49.2 | 手调参数 vs 内置 auto-tune | 手调 **+43.5% decode**；内置 auto-tune **−19.7%**；`--prefill auto` 慢 5-10% | 说明**别信 auto-tune**，我们手调参数是对的 |

> **明确写"无"的部分**：截至 2026-10-25 13:30，上游 **不存在** `0.1.47 → 0.1.48` 在 **RTX 4090（sm_89，48GB）** + **Qwen3.8-Flash-Next IQ3_S** + `--kv k8v4/--kv-resident 32768/--spec 4/262144 ctx` 这套配置下的 A/B 实测。官方 0.2.2/0.2.3/0.2.4 三份 release note 与我所检索的 issue 全集都没有。我们没有外部数据可以直接判定升级后是涨还是跌。

---

## 5. 回归与未修 issue（近两周，open+closed）

状态以 2026-10-25 13:30 的 API 返回为准。"影响我们？"是**判断**，依据列在理由里。

| 编号 | 标题（摘要） | 状态 | 日期 | 影响我们？ | 理由 |
|---|---|---|---|---|---|
| [#3935](https://github.com/Niko1221/Strata/issues/3935) | dsv41 预编译包只带 sm_86/sm_120，4090 被拒；主引擎 BUILD.json 含 sm_89 | **open** | 10-24 | **是（阻断 0.2.4 主功能）** | 直接决定"DeepSeek 与我们无关" |
| [#3947](https://github.com/Niko1221/Strata/issues/3947) | 请把 sm_89 加进 dsv41 CUDA 构建架构 | open | 10-24 | 是（作为升级触发信号） | 若关闭 = DeepSeek 可用性变化 |
| [#3994](https://github.com/Niko1221/Strata/issues/3994) | 为 sm_89（RTX 40）构建 DeepSeek 引擎 | open | 10-24 | 同上 | 同上 |
| [#3964](https://github.com/Niko1221/Strata/issues/3964) | "deepseek-v4.1-flash not supported ada of 4090" | open | 10-24 | 是（旁证） | 与 #3935 同一事实的用户视角 |
| [#3966](https://github.com/Niko1221/Strata/issues/3966) | 0.1.48 layer-split decode 相对 0.1.43 **−19%**；**同帖单卡 +15%** | open | 10-24 | **部分是**：单卡数值正向，layer-split 与我们无关 | 我们单卡；但基线是 0.1.43 |
| [#3952](https://github.com/Niko1221/Strata/issues/3952) | **0.1.46 长上下文 VRAM 0 MiB 死循环**（sm_89，`--max-context 262144 --spec 4`）；0.1.43.1 同参 238K 连跑 7 轮无告警；需把 `--vram-reserve-mib` 抬到 1094/1728 才停 | **open** | 10-24 | **是（长上下文预算风险）** | 配置形态与我们高度重合（sm_89 + 262144 + spec 4）；我们 0.1.47 未复现，但升级前必须先验证 |
| [#3961](https://github.com/Niko1221/Strata/issues/3961) | serve：新 agent 请求 0.2s 返回空（HTTP 200、1 chunk、无 finish_reason），引擎无日志、`/health` 正常；Linux Docker、**0.2.3 + engine 0.1.48** | open | 10-24 | 需关注 | 报告者自述**未复现**（单次）；版本与我们不同，但属 serve 请求路径 |
| [#3965](https://github.com/Niko1221/Strata/issues/3965) | DFlash2 vs MTP on RTX 4060 Ti 16GB：整体**慢 3.3%** | open | 10-24 | 是（决策项） | 明确不建议启用 DFlash2 替代 MTP |
| [#3659](https://github.com/Niko1221/Strata/issues/3659) | sm_89（Ada）的 `nv_defaults`：用 sm_86 的开关组在 RTX 4000 SFF Ada 上更划算，"先要 scope agreement 再改默认值" | open | 10-21 | **是（关键待办）** | 说明 0.1.48 **还没有 sm_89 专属默认值**；这是"下次评估"的头号信号 |
| [#3453](https://github.com/Niko1221/Strata/issues/3453) | sm_89（Ada）MMVQ-IL 表（RTX 4070 实测） | open | 10-20 | 是 | sm_89 内核调优仍在进行 |
| [#3999](https://github.com/Niko1221/Strata/issues/3999) | 增加 sm_89 MMVQ_IL matvec 表（RTX 4080 laptop 实测） | **closed** | 10-24 | 是（已收口但未入版） | 关了但属表/调优线，未见进入 0.1.48 |
| [#3958](https://github.com/Niko1221/Strata/issues/3958) | Windows 0.2.2 → 0.2.3 升级后的问题 | open | 10-24 | 否 | Windows |
| [#3981](https://github.com/Niko1221/Strata/issues/3981) | Metal：`--batch 2` 仍以 86 (`pdl_supported`) 退出、并发请求 200 空答 | open | 10-24 | 否 | macOS Metal |
| [#4033](https://github.com/Niko1221/Strata/issues/4033) | gfx1151 + 96GiB carve-out：`--expert-cache auto` 崩 "free(): chunks in smallbin corrupted"（0.2.4.1/main） | open | 10-25 | 否 | Strix Halo |
| [#4024](https://github.com/Niko1221/Strata/issues/4024) | Strix Halo 性能退化（7 评论） | open | 10-24 | 否 | gfx1151 |
| [#3813](https://github.com/Niko1221/Strata/issues/3813) | verify: layer N never rang / mtp draft illegal instruction，**0.1.48 可复现**；引擎重启后第 2 个请求必失败 | open | 10-23 | **需关注** | 涉及 MTP 与 verify 路径，我们吃 `--spec 4 --batch-mtp` |
| [#3979](https://github.com/Niko1221/Strata/issues/3979) | 引擎在 prefill 期间每 60s 自杀：10 天 24 次 stall（0.1.44.1/0.1.45/**0.1.48**），附 23 个 minidump | open | 10-24 | **需关注** | 0.1.48 在列表内 |
| [#3303](https://github.com/Niko1221/Strata/issues/3303) | q36（qwen35moe）KV 恒 fp32，`--kv`/`--kv-resident` 不解析，256K 在 16GB 上 OOM | open | 10-20 | 否（我们 KV 流式正常工作） | 反证我们这族已正确解析 `--kv`；q36 是另一族 |
| [#2936](https://github.com/Niko1221/Strata/issues/2936) | （历史）sm_86/3090 上 0.1.43→0.1.44 decode −20~27%，根因是 `STRATA_ADAPT_OVERLAP` 默认翻转 | closed | 10-17 | 否 | 已知历史，sm_89 未复现 |
| [#2441](https://github.com/Niko1221/Strata/issues/2441) / [#2453](https://github.com/Niko1221/Strata/issues/2453) | 0.1.44.1 在 A3000 / 5060 Ti 上的 decode 回归 | closed | 10-14 | 否 | 已在后续版本处理，早于我们基线 |

近期最值得盯的三条（若决定暂缓，这就是"下次重新评估"的输入）：**#3935/#3947/#3994（dsv41 是否支持 sm_89）、#3659（sm_89 默认值是否落地）、#3952（长上下文 VRAM 预算是否修复）**。

---

## 6. 结论、阈值与动作

### 6.1 结论：暂缓（保持 0.2.2 + 0.1.47）

**升级触发条件（任一满足才重新评估；建议写成硬阈值，不靠"感觉新"）**

- **T1（性能）**：出现 `0.1.47 → 0.1.48` 在 **sm_89 单卡 + Qwen3.8 IQ3_S + ≥131K ctx + `--kv k8v4 --spec 4`** 下的受控 A/B，且 **decode ≥ +8%** 或 **prefill ≥ +15%**，同时无 VRAM/verify 回归。
  → 目前**不存在**（§4 末尾）。最接近的 #4058（5090）与 #4054（3090）都不是 4090，不能替代。
- **T2（缺陷修复）**：#3952 类长上下文 VRAM 预算回归被确认在 0.1.48→后续版本中修复（有 changelog 条目或复测帖），且 #3813/#3979 这类 MTP/verify 稳定性问题在我们配置下被确认消失。
- **T3（能力可用性）**：dsv41 支持 sm_89（#3935/#3947/#3994 关闭并有新包）→ 此时 0.2.4 的内容才对我们有意义（但仍需换模型，与用户"继续用现有模型"冲突，只作为信息）。
- **T4（需求）**：确实需要某个 0.1.48-only 能力（如 DFlash2 实测胜出、或 sm_89 专属内核默认值落地 #3659）。

**反向触发（直接否决）**：任何一条 sm_89 + 4090 + Qwen 的 0.1.48 回归报告（decode 下降 / 长上下文 VRAM 崩塌 / verify 崩）出现，则继续保持不升。

### 6.2 若将来升级：最小风险步骤

原则：**只换引擎二进制，不动 server 代码、不动模型、不动 pack、不动模板、不跑 setup/update**（与我们 10-22 的路径一致）。

1. **前提校验**
   - 上游 release 的 `SHA256SUMS.txt` 里 `strata-engine-<新版本>-linux-x86_64-cuda-cuda13.tar.gz` 的 sha256 与下载件一致；核对 `BUILD.json` 的 `archs` 含 `89`。
   - 记录现网 `nvidia-smi` 基线、`/metrics` 的 `engine=` 值、以及一次固定 prompt 的 decode/prefill 基线（同 prompt、temperature 0、`max_tokens` 固定）。
2. **停机窗口**：停 serve → **等 `nvidia-smi` 的 used 降到 <1000 MiB** 再起（历史事故根因，必须等）→ 备份现有 `Strata-0.2.2/engine/` 整目录（或直接留旧版本目录，改 config 指向）。
3. **替换**：只解压新引擎到新目录，修改 `strata-davidau-iq3_s-022.json` 里的 `exe`/`vision`/`cwd`/`log` 指向新树；**`--batch 4 --max-context 262144 --kv k8v4 --kv-resident 32768 --spec 4 --batch-mtp --prefill auto:32768 --vision` 逐字不变**。
4. **灰度**：以 `--port 8001`（或临时端口）旁挂试跑同一 config，先跑 3 个固定 prompt + 1 个 ≥131K 长上下文请求，再决定是否切 8000。
5. **判据（必须全部通过才算成功）**
   - `/metrics` 的 `engine=` 显示新版本；启动日志无 `strata verify` 窗口崩塌、无 `only 0 MiB of VRAM free` 告警。
   - 长上下文请求中 free VRAM 单调不塌陷（#3952 的形态是 61→49→28→13→0 MiB）。
   - 3 个固定 prompt 的 decode/prefill 不低于基线的 −3%；`cached_tokens`/prompt 复用率未退化。
   - 并发 4 路（`--batch 4`）各 1 次 16K prompt，无空答、无 `finish_reason` 缺失（对照 #3961）。
   - 一次 ≥131K 长上下文的 `--spec 4` 轮次无 MTP verify 报错（对照 #3813）。
6. **回滚路径**：停 serve → 等 VRAM 归零 → config 指回旧 `engine/` 目录（或 `cp -a` 回备份）→ 起 serve。回滚目标已是可用快照：旧树 + 旧引擎目录全程保留，不需要重新下载。

### 6.3 若不升级：盯哪些信号

| 信号 | 具体对象 | 为什么 |
|---|---|---|
| 引擎版本跳版 | 下一个 release 的通用引擎是否仍为 0.1.48（0.2.4 已确认不变） | 只有通用引擎变更才可能影响 Qwen |
| sm_89 调优落地 | [#3659](https://github.com/Niko1221/Strata/issues/3659)、[#3453](https://github.com/Niko1221/Strata/issues/3453) 关闭 + changelog 出现 sm_89 默认值 | 这是 4090 唯一可能拿到实质收益的方向 |
| 长上下文 VRAM | [#3952](https://github.com/Niko1221/Strata/issues/3952) 的结论/修复 | 直接命中 262144 + spec 4 配置 |
| MTP/verify 稳定 | [#3813](https://github.com/Niko1221/Strata/issues/3813)、[#3979](https://github.com/Niko1221/Strata/issues/3979) | 我们一直开 `--spec 4 --batch-mtp` |
| DeepSeek 架构 | [#3935](https://github.com/Niko1221/Strata/issues/3935)/[#3947](https://github.com/Niko1221/Strata/issues/3947)/[#3994](https://github.com/Niko1221/Strata/issues/3994) 关闭 | 决定 0.2.4 主线是否对我们可用（信息价值 > 实用价值） |
| 新 release | <https://github.com/Niko1221/Strata/releases> 出现 0.2.5+ | 必须重新读正文，尤其"Qwen engine 是否变化"那一句 |

---

## 7. 未验证项（明确标注，勿当实测）

1. **`0.2.2 server + 0.1.48 engine` 的实际兼容性未实测**。§2 的"serve 无版本门控"只是对 `Strata-0.2.2/serve/*.py` 的静态阅读结论（`MIN_ENGINE` 只出现在 `setup.py`），不代表运行期行为。
2. **0.1.48 在 4090 48GB + 我方配置下的 decode/prefill 数字未测**。§4 的 #3966（4060 Ti，+15%）是**另一张卡、另一个基线版本**，不可当作我们的预期收益。
3. **#3966 的 +15% 无法拆分**：不知其中多少来自 0.1.43→0.1.47（我们已有的部分），多少来自 0.1.47→0.1.48。**我们的真实增量可能显著小于 15%。**
4. **dsv41 是否"故意"不支持 sm_89 未知**：上游文档 `docs/DEEPSEEK41.md` 只说"没测过 RTX 40"，运行期列表则明确剔除；是遗漏还是决策，上游未答复（#3935 仍 open，1 条评论）。
5. **0.2.4 的 dsv41 优化（layer-by-layer prefill、>3700 token 层跳）是否会回移到通用引擎未知**：目前只写在"DeepSeek 如何工作"段落，未见通用引擎 changelog。
6. **`strata-feed.json.sig` 的签名算法与公钥未核**：只确认文件存在且被 release 正文引用；本次仅校验了 `SHA256SUMS.txt` 的哈希清单内容。
7. **0.2.3 的 Strix Halo 数字（+6%~+19%）与其"答案非逐位一致"（near-tie 词差异）未在 NVIDIA 上复现过**：该 8-bit hyper-connection 读法标称是 gfx1151 专属，但"答案会变"这一点值得在将来任何引擎切换时都重跑一次一致性检查。
8. **0.2.4.1/0.2.4.2 的修复范围仅依据 release 正文**：未逐条核对 commit，不排除有未写进正文的变更。
9. 本次**未运行** `strata update`、`setup.py`、`strata launch`，也**未下载**任何引擎包（只读了 `SHA256SUMS.txt`）。所有"可安装/可运行"的判断均为文档与代码阅读结论。

---

## 附：关键 URL

- Releases 列表：<https://github.com/Niko1221/Strata/releases>
- v0.2.2：<https://github.com/Niko1221/Strata/releases/tag/v0.2.2>
- v0.2.3：<https://github.com/Niko1221/Strata/releases/tag/v0.2.3>
- v0.2.3.1：<https://github.com/Niko1221/Strata/releases/tag/v0.2.3.1>
- v0.2.4（含 0.2.4.1 / 0.2.4.2 更新段）：<https://github.com/Niko1221/Strata/releases/tag/v0.2.4>
- dsv41 架构拒绝（4090）：<https://github.com/Niko1221/Strata/issues/3935>
- 0.1.48 单卡 +15% / layer-split −19%：<https://github.com/Niko1221/Strata/issues/3966>
- 0.1.46 长上下文 VRAM 死循环：<https://github.com/Niko1221/Strata/issues/3952>
- sm_89 默认值待议：<https://github.com/Niko1221/Strata/issues/3659>
- MTP/verify 0.1.48 可复现异常：<https://github.com/Niko1221/Strata/issues/3813>
