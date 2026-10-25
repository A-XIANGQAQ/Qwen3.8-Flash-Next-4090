# 版本历史（压缩）

> 只保留「哪一版用什么、为什么换、带走的一两个数字」。逐次升级的完整 A/B 与踩坑在 git 历史里，
> 上游版本见 [Niko1221/Strata releases](https://github.com/Niko1221/Strata/releases)。
> **v1–v4 的运行时与权重已删除**，下表只作记录，不能照做。

| 版本 | 时间 | 引擎 / 权重 | 一句话结果 | 状态 |
|---|---|---|---|---|
| **v5.3** | 2026-10-22 | **Strata 0.2.2 + engine 0.1.47** / DavidAU IQ3_S（3.5 bpw） | 当前生产；2026-10-25 用 vLLM 基准复测，见 [BENCHMARK.md](BENCHMARK.md) | ✅ 当前 |
| v5.2 | 2026-10-18 | Strata 0.1.45 / DavidAU IQ3_S | 换模型：decode +10%、1K TTFT −49%、32K/100K decode +34%/+31%；拒答 4.55%→0% | 被 v5.3 取代 |
| v5.1 | 2026-10-18 | Strata 0.1.45 / unsloth UD-Q4_K_XL + ESP 破限 | 引擎小版本：短上下文 decode 139.7→144.5 t/s、32K TTFT 9.50→6.99 s | 被 v5.2 取代；UD-Q4 权重已删 |
| v5.0 | 2026-10-13 | Strata 0.1.43.1 / unsloth UD-Q4_K_XL + ESP | 切引擎：单流 decode 59.1→118 t/s、TTFT 2.73→0.20 s（对 v4） | 被 v5.1 取代 |
| v4 | 2026-09-24 | FreeToken 0.1.3 + 3 个本地补丁 / RadixArk NVFP4 | 单流 decode 59–64 t/s、73K 62.6–63.6、8K prefill 方差极大 | ⛔ 运行时/权重已删，不可回滚 |
| v3 | 2026-09-23 | 官方 sglang v0.5.20 + lk_moe / PLE 补丁 | 73K decode 42.0、254K prefill 1710 t/s，短请求 TTFT 0.42 s | ⛔ 运行时已删 |
| v2 | 2026-09-14 | Lsglang 0.5.19-lkmoe + PR #37500 | 首次在单 4090 上跑通 Flash-Next | ⛔ 运行时已删 |
| v1 | — | Lsglang 1.4.13 + lk_moe 2.4.0 | 早期能起服务，长上下文不可用 | ⛔ 运行时已删 |

## 为什么停在 v5.3

- 0.2.3 / 0.2.3.1 / 0.2.4 的通用引擎只从 0.1.47 走到 0.1.48，**0.2.4 对 Qwen 侧逐字节不变**；
  DeepSeek V4.1-Flash 走独立 `strata-dsv41` 包，**预编译不含 sm_89**。
- 4090 + IQ3_S 的 `0.1.47 → 0.1.48` 受控 A/B 上游不存在；最接近的是 4060 Ti 单卡 +15%（基线 0.1.43，拆不出增量）。
- 0.1.46+ 有 sm_89 + 262144 ctx + `--spec 4` 的长上下文 VRAM 崩塌报告（#3952，open）。
- 结论与触发条件：见 [STRATA-UPSTREAM-2026-10-25.md](STRATA-UPSTREAM-2026-10-25.md)。

## 已删除的东西

- FreeToken / Lsglang / 官方 sglang 的 venv、启动脚本、权重与补丁（2026-10-18 清理）；旧补丁在
  [`archive/patches/`](../archive/patches/)，旧脚本在 [`archive/scripts/`](../archive/scripts/)。
- UD-Q4_K_XL（111.33 GB）与 NVFP4（126 GiB）权重已删；回滚要重下。
