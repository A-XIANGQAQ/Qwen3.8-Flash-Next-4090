# 版本历史（压缩）

> 只保留「哪一版用什么、为什么换、带走的一两个数字」。逐次升级的完整 A/B 与踩坑在 git 历史里，
> 上游版本见 [Niko1221/Strata releases](https://github.com/Niko1221/Strata/releases)。
> **v1–v4 的运行时与权重已删除**，下表只作记录，不能照做。

| 版本 | 时间 | 引擎 / 权重 | 一句话结果 | 状态 |
|---|---|---|---|---|
| **v6.4** | 2026-11-26 | **Strata 0.2.22 + engine 0.1.69** / **unsloth UD-Q4_K_XL + huihui-ai v2 消融档 1** | 0.2.21 `#6713` 修 sm_89 的 `layer 0 never rang`（原 `#6742`）⇒ 删掉两个 Q8-TC 规避变量并验证；0.2.22 `#6776` 精度修复、`#7070` MTP 采样恢复。prefill 6.2K 2,738.5→**2,776.8 t/s**（+1.4%）、12.8K 4,986.2→**5,003.1 t/s**（+0.3%）、4 路 200–205→**201–208 t/s**，单流持平；28 s 切替、验收 24/24，见 [STRATA-UPSTREAM-2026-11-26.md](STRATA-UPSTREAM-2026-11-26.md) | ✅ 当前 |
| v6.3 | 2026-11-23 | **Strata 0.2.20 + engine 0.1.66** / **unsloth UD-Q4_K_XL + huihui-ai v2 消融档 1** | `EXP_MODE=8` 融合专家（13 组 Unsloth 专家对）+ 10 种 INT8 dense：4 并发聚合 174–188 → **200–205 t/s**（+6~18%）、12.8K prefill 4,747 → **4,986 t/s**（+5.0%）、6.2K −0.8%、单流 134.3 t/s 持平；**0.1.66 在 sm_89 崩 `layer 0 never rang`**（`fused_gr.cu` 的 `__ldg()` 读 `__shared__`，上游 `#6742`，修复 PR `#6714` 未并入），用 `STRATA_HC_Q8_TC=0` + `STRATA_HC_Q8_BF16TC=0` 规避；另补 `aliases: [Qwen3.8-Flash-Next, qwen3.8-flash-next-q4kxl-ablit]`（0.2.20 `#6621` 严格模型名，DSH/Cherry 旧名恢复）；第一次切替 2m10s 后回滚、第二次 28 s 就绪、验收 24/24，见 [STRATA-UPSTREAM-2026-11-23.md](STRATA-UPSTREAM-2026-11-23.md) | 被 v6.4 取代 |
| v6.2 | 2026-11-19 | **Strata 0.2.16 + engine 0.1.62** / **unsloth UD-Q4_K_XL + huihui-ai v2 消融档 1** | 引擎换代：0.1.62 的 NVIDIA/AMD/Arc/Mac 全部改跑 Strata 自有矩阵内核（不再用 ggml/llama.cpp），`strata-vision` 从 125.9 MB 缩到 1.2 MB；仓库改 `engine/ app/ terminal/ docs/ tools/` 布局（本机平铺 `serve/`）；pack v4 直接可读、30 个参数不变；6.2K prompt 2,760 t/s、12.8K 4,747 t/s（−1.3~1.5%，噪声内）、4 并发 174.3/188.5/185.9；切替 28 s、验收 24/24，见 [STRATA-UPSTREAM-2026-11-19.md](STRATA-UPSTREAM-2026-11-19.md) | 被 v6.3 取代 |
| v6.1 | 2026-11-17 | **Strata 0.2.15 + engine 0.1.60** / **unsloth UD-Q4_K_XL + huihui-ai v2 消融档 1** | serve 层小版本：引擎与 0.1.60 逐字节相同，实际拿到 `frontend.py` **#5979**（布尔工具参数 `True`/`False` 解析）与 `server.py` **#6055**（关闭期新请求回 503 排水）；复测 6.2K prompt 2,801 t/s、12.8K 4,809 t/s、4 并发 182.7/184.9/185.6（±0.5% 噪声内）；切替 35 s 无事故、验收 24/24，见 [STRATA-UPSTREAM-2026-11-17.md](STRATA-UPSTREAM-2026-11-17.md) | 被 v6.2 取代 |
| **v6.0** | 2026-11-16 | **Strata 0.2.14 + engine 0.1.60** / **unsloth UD-Q4_K_XL + huihui-ai v2 消融档 1** | Q4_K/Q5_K 专家的 prompt 从 llama.cpp MMQ 换到 Strata 自有融合内核（日志 48/48 层）：6.2K prompt 2,770→**2,795 t/s**、12.8K 4,631→**4,783 t/s**（+3.3%）、20K 4,847、30K 4,939；4 并发 175–187 t/s 与 0.1.58 空缓存口径持平；模板 Sharp-naze v1.5（34,386 字符）、验收 23/23；**停机 7m24s**（启动器漏 `chmod +x`），见 [STRATA-UPSTREAM-2026-11-16.md](STRATA-UPSTREAM-2026-11-16.md) | 被 v6.1 取代 |
| v5.9 | 2026-11-08 | **Strata 0.2.12 + engine 0.1.58** / **unsloth UD-Q4_K_XL（4-bit）+ huihui-ai v2 消融** | 换模型：WorkBuddy 原失败题 0.1666→**0.8917**、0.0→**0.6923**；速度降（4 并发 215→186 t/s；单流 158–164→**112.8 t/s default**、133–135 t/s effort=low）；48/48 张量消融、pack v4；**11-09 模板换 Sharp-naze v1.5（34,386 字符）、权重同日曾短暂切回 stock 后又回档 1**，见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) | 被 v6.0 取代 |
| v5.8 | 2026-11-08 | **Strata 0.2.12 + engine 0.1.58** / DavidAU IQ3_S（3.4 bpw） | **速度持平**（6.25K prompt 5,177 t/s、12.7K 5,526 t/s、4 并发 215 t/s），换的是 7 条工具调用修复（`#4828` 强制 tool_choice 0/24→24/24）+ 引擎 `#5108` 长 CUDA prompt 行表越界；W16 cache 仍 **0 MiB**；停机 48 s，验收 24/24，见 [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md) | 被 v5.9 取代 |
| v5.7 | 2026-11-06 | Strata 0.2.11 + engine 0.1.57 / DavidAU IQ3_S | **FP16 weight cache 在 48 GB 上未启用（0 MiB，需留 3 GiB 空闲）**，换的是 4 条修复（#2518 slot 串台、#5040/#4701 流式工具尾段、#4451 重启后旧请求挂新 slot、#4271 断流不再占 slot）；停机 47 s，验收 24/24，见 [STRATA-UPSTREAM-2026-11-06.md](STRATA-UPSTREAM-2026-11-06.md) | 被 v5.8 取代 |
| v5.6 | 2026-11-03 | Strata 0.2.10 + engine 0.1.56 / DavidAU IQ3_S | 性能零收益、纯修复（DFlash2 与 4 并发互斥）；停机 47 s，验收 23/23，见 [STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md) | 被 v5.7 取代 |
| v5.5 | 2026-11-01 | Strata 0.2.9 + engine 0.1.55 / DavidAU IQ3_S | 第一个"默认开性能开关"的版本（并行栈 9 项 + GDN 张量核）；停机 47 s，验收 21/21 | 被 v5.6 取代 |
| v5.4 | 2026-10-28 | Strata 0.2.7 + engine 0.1.52 / DavidAU IQ3_S | sm_89 实测速度表（结果 bit-for-bit 不变）+ 会话缓存/读表修复；停机 63 s | 被 v5.5 取代 |
| v5.3 | 2026-10-22 | Strata 0.2.2 + engine 0.1.47 / DavidAU IQ3_S（3.4 bpw） | 2026-10-25 用 vLLM 基准复测，见 [BENCHMARK.md](BENCHMARK.md) | 被 v5.4 取代 |
| v5.2 | 2026-10-18 | Strata 0.1.45 / DavidAU IQ3_S | 换模型：decode +10%、1K TTFT −49%、32K/100K decode +34%/+31%；拒答 4.55%→0% | 被 v5.3 取代 |
| v5.1 | 2026-10-18 | Strata 0.1.45 / unsloth UD-Q4_K_XL + ESP 破限 | 引擎小版本：短上下文 decode 139.7→144.5 t/s、32K TTFT 9.50→6.99 s | 被 v5.2 取代；UD-Q4 权重已删 |
| v5.0 | 2026-10-13 | Strata 0.1.43.1 / unsloth UD-Q4_K_XL + ESP | 切引擎：单流 decode 59.1→118 t/s、TTFT 2.73→0.20 s（对 v4） | 被 v5.1 取代 |
| v4 | 2026-09-24 | FreeToken 0.1.3 + 3 个本地补丁 / RadixArk NVFP4 | 单流 decode 59–64 t/s、73K 62.6–63.6、8K prefill 方差极大 | ⛔ 运行时/权重已删，不可回滚 |
| v3 | 2026-09-23 | 官方 sglang v0.5.20 + lk_moe / PLE 补丁 | 73K decode 42.0、254K prefill 1710 t/s，短请求 TTFT 0.42 s | ⛔ 运行时已删 |
| v2 | 2026-09-14 | Lsglang 0.5.19-lkmoe + PR #37500 | 首次在单 4090 上跑通 Flash-Next | ⛔ 运行时已删 |
| v1 | — | Lsglang 1.4.13 + lk_moe 2.4.0 | 早期能起服务，长上下文不可用 | ⛔ 运行时已删 |

## 为什么从 v5.3 一路走到 v6.4

- **v5.4（0.2.7）**：0.2.3–0.2.6 对 Qwen 单卡无增量（0.2.6 的 Qwen 引擎与 0.2.5 逐字节相同）；0.2.7 是 0.2.2 之后第一个动 sm_89 的版本
  （实测速度表，结果 bit-for-bit 不变）+ `#1503` 会话缓存跨会话、`#3602` 读表挂死等修复。见 [STRATA-UPSTREAM-2026-10-28.md](STRATA-UPSTREAM-2026-10-28.md)。
- **v5.5（0.2.9）**：0.2.8 修工具调用（1462 轮 agent 测试 92.9% → 98.3% 按意图读对）；0.2.9 把两个性能栈改成默认开，且都命中本机：
  Linux CUDA + `--batch ≥2` 的**并行请求栈**（9 个 `STRATA_*`，上游 8 并发 1.45×）、Linux CUDA Qwen 的 **GDN 张量核 prompt 内核**（prompt +4~6%）。
  其余 0.2.8/0.2.9 内容（Strix Halo、Intel Arc、四卡 Radeon、DGX Spark、Mac、12 GB 卡路由回滚）与本机无关。见 [STRATA-UPSTREAM-2026-11-01.md](STRATA-UPSTREAM-2026-11-01.md)。
- **v5.6（0.2.10）**：**性能上零收益，纯换修复**。头部功能 DFlash2 与 4 并发互斥（Auto 在 `--batch ≥2` 时一律选 MTP），
  全卡切分引擎要两张 sm86/sm120 NVIDIA，其余（8 GB/128K、ISTA-DAS Coder、Mac、RTX 50 CUDA12 首启）无关。拿到的是 `#4590` 惩罚窗口、
  `#4677` 停放会话、`#4663/#4700` Anthropic 路由工具、`#4838` 溢出措辞、`#4701/#4751` write 截断。见 [STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md)。
- **v5.7（0.2.11）**：头部卖点 **FP16 weight cache** 要「分配后卡上仍留 3,072 MiB 空闲」，本机启动后只剩 ~2,900 MiB，
  日志直接 `FP16 weight cache 0 MiB`，上游 RTX PRO 5000 上的 4 路 +5.7% 没有复现（符合它「未测 sm_89」的说明）。
  换到的是：`#2518` 4 会话 slot 串台、`#5040/#4701` 流式工具参数尾段、`#4451` 引擎重启后旧请求挂新 slot / 控制行被抢、`#4271` 断流不再占 slot。
  0.2.11 还把单卡 CPU/upload share 改为按 CPU 实测（0.55→0.15），前后实测差在 ±2% 噪声内。见 [STRATA-UPSTREAM-2026-11-06.md](STRATA-UPSTREAM-2026-11-06.md)。
- **v5.8（0.2.12）**：**速度持平**，头部 tuned DFlash2（decode +2.4~5.0%）要求「专家全部驻留显卡」，我们 4 slots + 流式专家 ⇒ 仍选 MTP、不下载 drafter。
  拿到的是 7 条工具调用修复（`#4828` 多个工具 + 强制 `tool_choice` 从 0/24 修到 24/24 最关键，另有 `#5312`/`#5409`/`#5276`/`#4832`/`#4622`）
  与引擎 `#5108`（行表越界读取会 fault 长 CUDA prompt）、`#5096/#4844`（copy 顺序、丢失的 adaptive swap）。
  同协议实测 6.25K prompt 5,177 t/s、12.7K 5,526 t/s、4 并发 215 t/s，与 0.1.57 差在 ±1% 噪声内；停机 48 s。见 [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md)。
- **v5.9（换模型）**：引擎不动，权重从 DavidAU IQ3_S（83.86 GB，3.4 bpw）换成官方 unsloth **UD-Q4_K_XL**（111.33 GB；**Dynamic v3.0**，见 [STRATA-MODEL §2.1](STRATA-MODEL-2026-11-08.md)），
  再用 huihui-ai v2 的 `patch-all` 消融 48 个 `ssm_out` / `attn_output` 张量（不动 routed experts），pack 用 `iq_pack.py` 现建 v4；
  模板先换回 froggeric v22.5，**11-09 又换成 Sharp-naze v1.5**（34,422 B）；当晚发现档 2 会破坏工具链（推理当正文、工具调用变文本），重建为档 1（48 张量）并切换。质量明显上行，代价是 33% 的权重体积：4 并发 215→186 t/s；单流 158–164→**112.8 t/s（default）**、133–135 t/s（`reasoning_effort=low`）。见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)。
- **v6.0（0.2.14 + 0.1.60）**：唯一实质变化是 **Q4_K/Q5_K 专家的 prompt 从 llama.cpp MMQ 换到 Strata 自有融合内核**（`-DSTRATA_MMQ=ON` 可回旧路径），
  引擎二进制从 315.7 MB 降到 177.7 MB。日志确认 `48 of 48 layers on the fused kernels`；同协议实测 6.2K prompt +0.9%、12.8K **+3.3%**、20K 4,847 t/s、30K 4,939 t/s，
  并发与 0.1.58 空缓存口径持平。0.2.13 没有单独升（只是把启动器默认值搬进引擎）。工具调用/思考泄漏 `#4751`/`#4077`/`#4949` **仍未修**。
  代价是一次 7m24s 停机：新启动器漏了 `chmod +x`，`systemd-run` 报「权限不够」。见 [STRATA-UPSTREAM-2026-11-16.md](STRATA-UPSTREAM-2026-11-16.md)。
- **v6.4（0.2.22 + 0.1.69）**：0.2.21 修掉了 v6.3 的坑——`#6713` 修 sm_89 上 `q8_ld16()` 用 `__ldg()` 读共享内存的回归（即 `#6742`），本机删掉 `STRATA_HC_Q8_TC` / `STRATA_HC_Q8_BF16TC` 两个规避变量后 24/24，Q8 超连接的 TC 路径重新打开。0.2.22 另有 `#6776`（Qwen prompt 读取一处求和顺序精度修复，首 token 不变）与 `#7070`（开 MTP 时恢复采样：不显式给采样就按模型卡 temp 1.0 / top_p 0.95 / top_k 20，不再静默变贪心）。同协议实测 prefill 6.2K 2,738.5→**2,776.8 t/s**（+1.4%）、12.8K 4,986.2→**5,003.1 t/s**（+0.3%）、4 路 200–205→**201–208 t/s**（+0.7~1.9%），单流持平；28 s 切替。见 [STRATA-UPSTREAM-2026-11-26.md](STRATA-UPSTREAM-2026-11-26.md)。
- **v6.3（0.2.20）**：引擎 0.1.62 → 0.1.66，NVIDIA 侧拿到 `EXP_MODE=8` 融合专家与 INT8 dense，4 路聚合 +6~18%、12.8K prefill +5.0%，单流持平；代价是 `#6742` 在 sm_89 上把 Q8 超连接的 TC kernel 打挂，关掉 `STRATA_HC_Q8_TC`/`STRATA_HC_Q8_BF16TC` 后正常。见 [STRATA-UPSTREAM-2026-11-23.md](STRATA-UPSTREAM-2026-11-23.md)。
- **v6.1（0.2.15）**：app/Terminal/Control 版本，**引擎没动**（官方仍从 0.2.14 release 拿 0.1.60）。我们直跑 serve，实际只有两处相关修复：`frontend.py` `#5979`（模型把声明的布尔工具参数写成 `True`/`False` 时正确解析）、`server.py` `#6055`（关闭期新请求回 503 `server_draining`，不再切 0 token）。部署上把 `Strata-0.2.15/engine` 软链到 0.2.14 的引擎目录，config 只改 4 个字段。见 [STRATA-UPSTREAM-2026-11-17.md](STRATA-UPSTREAM-2026-11-17.md)。
- **遗留**：`#3952`（sm_89 + 262144 ctx + `--spec 4` 的 VRAM 崩塌报告）仍 open；我们靠 `--vram-reserve-mib 2560` + `--kv-resident 32768` 维持 ~2,480 MiB 余量。
  `PLE table mlock failed` 仍在（`user@1000` 的 `LimitMEMLOCK` 硬上限 8 MiB，要 root drop-in），细节见 [STRATA-MODEL-2026-11-08.md §11](STRATA-MODEL-2026-11-08.md)。
- **没做严格的交错 A/B**：每代引擎各需一次重启，生产中断代价大于 5% 量级的收益；现有数字只能说明"没有可见回退"。

## 已删除的东西

- FreeToken / Lsglang / 官方 sglang 的 venv、启动脚本、权重与补丁（2026-10-18 清理）；旧补丁在
  [`archive/patches/`](../archive/patches/)，旧脚本在 [`archive/scripts/`](../archive/scripts/)。
- v5.1 的旧 3 片版 UD-Q4_K_XL（111.33 GB，49.86/49.38/12.09 GB 切分）与 NVFP4（126 GiB）权重已删；
  现行 v5.9 用的是 2026-11-04 重切过的四片版（总量 111,334,654,784 → 111,334,654,778 B，只差 6 B：把 IQ4_NL engram 表独立成 shard4，权重张量未重量化），两者不是同一批文件。
- **2026-11-08 第二轮删除（用户要求，物理删除、无归档）**：DavidAU IQ3_S（83.86 GB，78.1 GiB）及其 pack/旧 mmproj、vLLM 源码 + conda env + `~/.cache/vllm`、
  `LLM_MODEL` 里的 NVFP4 / FP8 权重、Strata 0.2.2–0.2.11 旧树与旧 config/启动脚本全部删掉，共腾出 289 GB。现在只剩 0.2.12 + UD-Q4_K_XL 一条路线。
