# 版本历史（压缩）

> 只保留「哪一版用什么、为什么换、带走的一两个数字」。逐次升级的完整 A/B 与踩坑在 git 历史里，
> 上游版本见 [Niko1221/Strata releases](https://github.com/Niko1221/Strata/releases)。
> **v1–v4 的运行时与权重已删除**，下表只作记录，不能照做。

| 版本 | 时间 | 引擎 / 权重 | 一句话结果 | 状态 |
|---|---|---|---|---|
| **v5.6** | 2026-11-03 | **Strata 0.2.10 + engine 0.1.56** / DavidAU IQ3_S（3.5 bpw） | 当前生产；速度不变（4 并发继续 MTP，**不启用 DFlash2**），换的是工具调用/惩罚窗口/停放会话等修复；停机 47 s，验收 23/23，见 [STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md) | ✅ 当前 |
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

## 为什么从 v5.3 一路走到 v5.6

- **v5.4（0.2.7）**：0.2.3–0.2.6 对 Qwen 单卡无增量（0.2.6 的 Qwen 引擎与 0.2.5 逐字节相同）；0.2.7 是 0.2.2 之后第一个动 sm_89 的版本
  （实测速度表，结果 bit-for-bit 不变）+ `#1503` 会话缓存跨会话、`#3602` 读表挂死等修复。见 [STRATA-UPSTREAM-2026-10-28.md](STRATA-UPSTREAM-2026-10-28.md)。
- **v5.5（0.2.9）**：0.2.8 修工具调用（1462 轮 agent 测试 92.9% → 98.3% 按意图读对）；0.2.9 把两个性能栈改成默认开，且都命中本机：
  Linux CUDA + `--batch ≥2` 的**并行请求栈**（9 个 `STRATA_*`，上游 8 并发 1.45×）、Linux CUDA Qwen 的 **GDN 张量核 prompt 内核**（prompt +4~6%）。
  其余 0.2.8/0.2.9 内容（Strix Halo、Intel Arc、四卡 Radeon、DGX Spark、Mac、12 GB 卡路由回滚）与本机无关。见 [STRATA-UPSTREAM-2026-11-01.md](STRATA-UPSTREAM-2026-11-01.md)。
- **v5.6（0.2.10）**：**性能上零收益，纯换修复**。头部功能 DFlash2 与 4 并发互斥（Auto 在 `--batch ≥2` 时一律选 MTP，DFlash2 一次只答一个请求），
  全卡切分引擎要「恰好两张 sm86/sm120 NVIDIA」或「四张 R9700」，其余（8 GB/128K、ISTA-DAS Coder、Mac DeepSeek 图、RTX 50 CUDA12 首启）都无关。
  真正拿到的是：`#4590` Qwen 惩罚窗口不再数 prompt 尾部 token（新任务复读旧 JSON）、`#4677` 停放会话不再偶发被拒/重读 6% prompt、
  `#4663/#4700` 工具"用过的算已声明"扩到 **Anthropic 路由**（DSH 走的就是 `anthropic-messages`）、`#4838` `/v1/messages` 溢出改用 Anthropic 措辞、`#4701/#4751` write 截断与 `tools.write` 归一化。
  见 [STRATA-UPSTREAM-2026-11-03.md](STRATA-UPSTREAM-2026-11-03.md)。
- **遗留**：`#3952`（sm_89 + 262144 ctx + `--spec 4` 的 VRAM 崩塌报告）仍 open；我们靠 `--vram-reserve-mib 2560` + `--kv-resident 32768` 维持 ~2,480 MiB 余量。
  `PLE table mlock failed` 仍在（systemd 单元 `LimitMEMLOCK` 只有 8 MiB；属于 PINNED 路径之外的锦上添花，不影响服务）。
- **没做严格的交错 A/B**：每代引擎各需一次重启，生产中断代价大于 5% 量级的收益；现有数字只能说明"没有可见回退"。

## 已删除的东西

- FreeToken / Lsglang / 官方 sglang 的 venv、启动脚本、权重与补丁（2026-10-18 清理）；旧补丁在
  [`archive/patches/`](../archive/patches/)，旧脚本在 [`archive/scripts/`](../archive/scripts/)。
- UD-Q4_K_XL（111.33 GB）与 NVFP4（126 GiB）权重已删；回滚要重下。
