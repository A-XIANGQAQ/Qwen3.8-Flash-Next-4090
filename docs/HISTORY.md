# 版本历史（压缩）

> 只保留「哪一版用什么、为什么换、带走的一两个数字」。逐次升级的完整 A/B 与踩坑在 git 历史里，
> 上游版本见 [Niko1221/Strata releases](https://github.com/Niko1221/Strata/releases)。
> **v1–v4 的运行时与权重已删除**，下表只作记录，不能照做。

| 版本 | 时间 | 引擎 / 权重 | 一句话结果 | 状态 |
|---|---|---|---|---|
| **v5.9** | 2026-11-08 | **Strata 0.2.12 + engine 0.1.58** / **unsloth UD-Q4_K_XL（4-bit）+ huihui-ai v2 消融** | 换模型：WorkBuddy 原失败题 0.1666→**0.8917**、0.0→**0.6923**；速度降（4 并发 215→186 t/s，单流 158–164→133–135 t/s）；96/96 张量消融、pack v4、froggeric v22.5（28,226 字符）、验收 29/29，见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) | ✅ 当前 |
| v5.8 | 2026-11-08 | **Strata 0.2.12 + engine 0.1.58** / DavidAU IQ3_S（3.5 bpw） | **速度持平**（6.25K prompt 5,177 t/s、12.7K 5,526 t/s、4 并发 215 t/s），换的是 7 条工具调用修复（`#4828` 强制 tool_choice 0/24→24/24）+ 引擎 `#5108` 长 CUDA prompt 行表越界；W16 cache 仍 **0 MiB**；停机 48 s，验收 24/24，见 [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md) | 被 v5.9 取代 |
| v5.7 | 2026-11-06 | Strata 0.2.11 + engine 0.1.57 / DavidAU IQ3_S | **FP16 weight cache 在 48 GB 上未启用（0 MiB，需留 3 GiB 空闲）**，换的是 4 条修复（#2518 slot 串台、#5040/#4701 流式工具尾段、#4451 重启后旧请求挂新 slot、#4271 断流不再占 slot）；停机 47 s，验收 24/24，见 [STRATA-UPSTREAM-2026-11-06.md](STRATA-UPSTREAM-2026-11-06.md) | 被 v5.8 取代 |
| v5.6 | 2026-11-03 | Strata 0.2.10 + engine 0.1.56 / DavidAU IQ3_S | 性能零收益、纯修复（DFlash2 与 4 并发互斥）；停机 47 s，验收 23/23，见 [STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md) | 被 v5.7 取代 |
| v5.5 | 2026-11-01 | Strata 0.2.9 + engine 0.1.55 / DavidAU IQ3_S | 第一个"默认开性能开关"的版本（并行栈 9 项 + GDN 张量核）；停机 47 s，验收 21/21 | 被 v5.6 取代 |
| v5.4 | 2026-10-28 | Strata 0.2.7 + engine 0.1.52 / DavidAU IQ3_S | sm_89 实测速度表（结果 bit-for-bit 不变）+ 会话缓存/读表修复；停机 63 s | 被 v5.5 取代 |
| v5.3 | 2026-10-22 | Strata 0.2.2 + engine 0.1.47 / DavidAU IQ3_S（3.5 bpw） | 2026-10-25 用 vLLM 基准复测，见 [BENCHMARK.md](BENCHMARK.md) | 被 v5.4 取代 |
| v5.2 | 2026-10-18 | Strata 0.1.45 / DavidAU IQ3_S | 换模型：decode +10%、1K TTFT −49%、32K/100K decode +34%/+31%；拒答 4.55%→0% | 被 v5.3 取代 |
| v5.1 | 2026-10-18 | Strata 0.1.45 / unsloth UD-Q4_K_XL + ESP 破限 | 引擎小版本：短上下文 decode 139.7→144.5 t/s、32K TTFT 9.50→6.99 s | 被 v5.2 取代；UD-Q4 权重已删 |
| v5.0 | 2026-10-13 | Strata 0.1.43.1 / unsloth UD-Q4_K_XL + ESP | 切引擎：单流 decode 59.1→118 t/s、TTFT 2.73→0.20 s（对 v4） | 被 v5.1 取代 |
| v4 | 2026-09-24 | FreeToken 0.1.3 + 3 个本地补丁 / RadixArk NVFP4 | 单流 decode 59–64 t/s、73K 62.6–63.6、8K prefill 方差极大 | ⛔ 运行时/权重已删，不可回滚 |
| v3 | 2026-09-23 | 官方 sglang v0.5.20 + lk_moe / PLE 补丁 | 73K decode 42.0、254K prefill 1710 t/s，短请求 TTFT 0.42 s | ⛔ 运行时已删 |
| v2 | 2026-09-14 | Lsglang 0.5.19-lkmoe + PR #37500 | 首次在单 4090 上跑通 Flash-Next | ⛔ 运行时已删 |
| v1 | — | Lsglang 1.4.13 + lk_moe 2.4.0 | 早期能起服务，长上下文不可用 | ⛔ 运行时已删 |

## 为什么从 v5.3 一路走到 v5.9

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
- **v5.9（换模型）**：引擎不动，权重从 DavidAU IQ3_S（83.86 GB，3.4 bpw）换成官方 unsloth v3.1 **UD-Q4_K_XL**（111.33 GB），
  再用 huihui-ai v2 的 `patch-all` 消融 96 个 `ssm_out` / `attn_output` / `ffn_down_shexp` 张量（不动 routed experts），pack 用 `iq_pack.py` 现建 v4，
  模板换回 froggeric v22.5。质量明显上行，代价是 33% 的权重体积：4 并发 215→186 t/s、单流 158–164→133–135 t/s。见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)。
- **遗留**：`#3952`（sm_89 + 262144 ctx + `--spec 4` 的 VRAM 崩塌报告）仍 open；我们靠 `--vram-reserve-mib 2560` + `--kv-resident 32768` 维持 ~2,480 MiB 余量。
  `PLE table mlock failed` 仍在：`LimitMEMLOCK` 只有 8 MiB，且 `user@1000.service` 自身硬上限就是 8 MiB，transient 单元抬不上去；要锁 PLE 表得 root 加 `user@.service.d/` drop-in（上游称最多 +6% decode）。
- **没做严格的交错 A/B**：每代引擎各需一次重启，生产中断代价大于 5% 量级的收益；现有数字只能说明"没有可见回退"。

## 已删除的东西

- FreeToken / Lsglang / 官方 sglang 的 venv、启动脚本、权重与补丁（2026-10-18 清理）；旧补丁在
  [`archive/patches/`](../archive/patches/)，旧脚本在 [`archive/scripts/`](../archive/scripts/)。
- v5.1 的旧 3 片版 UD-Q4_K_XL（111.33 GB，49.86/49.38/12.09 GB 切分）与 NVFP4（126 GiB）权重已删；
  现行 v5.9 用的是 2026-11-04 重切的 v3.1 四片版（总量同为 111.33 GB，边界不同），两者不是同一批文件。
- **2026-11-08 第二轮删除（用户要求，物理删除、无归档）**：DavidAU IQ3_S（79 GB）及其 pack/旧 mmproj、vLLM 源码 + conda env + `~/.cache/vllm`、
  `LLM_MODEL` 里的 NVFP4 / FP8 权重、Strata 0.2.2–0.2.11 旧树与旧 config/启动脚本全部删掉，共腾出 289 GB。现在只剩 0.2.12 + UD-Q4_K_XL 一条路线。
