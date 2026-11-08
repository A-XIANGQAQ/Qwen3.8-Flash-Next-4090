# 路线：Strata（v5.9）

把生产引擎从 v4（FreeToken + 3 个本地补丁）换成 [Niko1221/Strata](https://github.com/Niko1221/Strata)：
一个自带 C++/CUDA 引擎，复用 llama.cpp 的 ggml/gguf-py/mtmd，做 MoE 专家分层（VRAM/RAM/SSD）、
conversation cache、KV 流式与 MTP 投机解码。

**当前固定身份**：Strata **0.2.12** 服务端 + **engine 0.1.58**（预编译，arch 含 sm_89）+ unsloth **UD-Q4_K_XL**（huihui v2 消融）。
版本演进见 [HISTORY.md](HISTORY.md)，实测见 [BENCHMARK.md](BENCHMARK.md)，换模型见
[STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)，升级判断见 [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md)。

## 1. 为什么是它

同一张 4090、同一个模型家族，v5 相对 v4 的取舍（2026-10-13 同窗口 A/B，历史口径）：

| 指标 | v4 FreeToken | v5.0 Strata |
|---|---:|---:|
| 单流 decode | 59.1 t/s | **118 t/s**（2.0×） |
| 单流 TTFT | 2.73 s | **0.20 s**（14×） |
| 4 路聚合 / 每路 | 117.1 / 45.8 | 137 / 36.9 |
| 并发扩展比 1→4 | 2.55× | 1.16× |

结论：**1–2 路交互全面占优；高并发扩展性弱**。v5.3 历史实测（vLLM 口径，IQ3_S）见 [BENCHMARK.md](BENCHMARK.md)：
单流 decode 约 160–175 t/s、32K prefill 约 3.9K t/s、260K prefill 约 66 s；v5.9 换 UD-Q4_K_XL 后的当前数字在同页 §0。
没有"影子部署"：两块引擎都吃满 48GB，只能停机切换；本仓库现在只保留 Strata。

## 2. 模型：unsloth UD-Q4_K_XL（huihui v2 消融）

`unsloth/Qwen3.8-Flash-Next-GGUF` → `UD-Q4_K_XL/`（v3.1 Dynamic，2026-11-04），再用 huihui-ai v2 的
`patch-all` 做 96 张量消融。

| 项 | 值 |
|---|---|
| 量化 | **UD-Q4_K_XL**（4-bit，约 4.3 bpw 有效精度），**111.33 GB / 4 shards** |
| 破限 | **huihui v2 消融**：`ssm_out` / `attn_output` / `ffn_down_shexp` 共 96 张量，**不动 routed experts**；不需要 ESP 控制向量 |
| 模板 | **froggeric v22.5**（28,234 B / 28,226 字符），见 §9 |
| 视觉 | `mmproj-Qwen3.8-Flash-Next-BF16-unsloth.gguf`（907,542,944 B）+ `engine/strata-vision` |
| PLE 表 | `per_layer_token_embd.weight` 是 **shard4**（IQ4_NL，28.8 GB），由 `--ple-gguf` 指向它 |

下载、打补丁与 pack 生成见 [REPRODUCTION.md §4–5](REPRODUCTION.md)，换模型全过程见
[STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)。

## 3. 获取与版本配对

0.2.0 起上游发 **Linux 预编译引擎**（0.1.58 的 `BUILD.json`: `archs [75,86,89,120]`, CUDA 13.0），
不再需要自建；但**不跑 `setup.sh`**：

- setup 会从 GGUF 重导 unsloth 原装模板（9,993 B）→ 覆盖 froggeric
- `carry_over()` 会重写 run config，**手工加进 `args` 的 12 个参数不会合并**
- 0.1.44+ 的分片 SHA 校验有 #2902（大分片误判并删文件）

**必须整代配对**：`serve/server.py`（0.2.12）+ `engine/strata` + `engine/strata-vision`（0.1.58）同代。
只换其中一个不报错，但会静默降级（[TROUBLESHOOTING #39](TROUBLESHOOTING.md)）。

## 4. 关键参数

```
--batch 4                    # 4 个独立 slot session（≈0.86 GiB/槽），不是 n_ctx÷4
--max-context 262144         # 每 slot 独立 256K
--kv k8v4                    # INT8 K + 旋转 Q4_0 V，816 B/cell
--kv-resident 32768          # KV 流式：只驻留 32K，其余在宿主内存
--batch-mtp                  # 每 slot 每窗口验证一个 MTP 提议
--spec 4 --spec-min-p 0.5    # 投机解码
--mtp <Strata-data>/mtp/rt
--conversation-cache-mib 8192
--prefill auto:32768         # 实测比 auto（8192）快约 20%
--vision --vram-reserve-mib 2560
--expert-cache auto          # 落点：~32.6 GiB / 10,902 槽
```

共 **30 项**，不含 `--resident-budget-gib`：247 GB 内存足够把 72 GiB 专家全部驻留 RAM（arena 73,450 MiB），
预算模式反而慢（上游实测 118 vs 169 t/s）。config 的 `env` 里设 `STRATA_ROUTE_TAIL_SKIP=0`。

## 5. KV：`k8v4` + 必须流式

`kv_tok = 13 × cell`（12 个 QSA 层 + draft 层）：

| 格式 | B/cell | 256K 单 slot | 4 slot | 质量 |
|---|---:|---:|---:|---|
| `int8` | 1056 | 3.60 GB | 14.4 GB | 与 fp16 难以区分 |
| **`k8v4`** | **816** | **2.78 GB** | **11.1 GB** | needle 同 int8 |
| `q4_0` | 576 | 1.96 GB | 7.85 GB | ⛔ 困惑度 +8%(1K)/+12%(8K) |

`--kv-resident 0` 实测直接 OOM（`native dense upload blk.27.attn_output.weight: out of memory`）：
48GB 卡上 4×256K 的唯一解就是 KV 流式到 247GB 宿主内存。实测 PCIe H2D 25.3 GB/s。

## 6. `--batch-mtp` 与并发

不加 `--batch-mtp`，批处理退化为"没有 MTP"：实测 4 路聚合 131 t/s（加后 137）、2 路 103–112（加后 123–125）。
不报错、不警告。`--batch 4` 是**四个独立 session**；并发的扩展性见 [BENCHMARK.md](BENCHMARK.md) §4。

## 7. 视觉（GPU 编码器）

三件套同时在位：mmproj + `engine/strata-vision` + config 的 `vision` 段。默认不装；手工装，别重跑 setup。

**必踩的坑**：配置有 `vision` 段但 `args` 里没有 `--vision` 时，serve 会补 `--vram-reserve-mib 700`——
48GB 卡已被引擎吃满，编码器一重启要 ~1.5GB → OOM → 重启死循环（`400 the image could not be read`）。
所以必须显式写 `--vision --vram-reserve-mib 2560`。判据日志：

- `strata serve: N MiB of VRAM free with everything loaded`，N ≥ ~1600
- `strata verify: no VRAM for a new window graph` 出现即已压死

代价：GPU 专家缓存 34.09 → 32.28 GiB，文本速度在噪声内（见 [HISTORY.md](HISTORY.md) v5.1）。
编码器显存需求与图大小无关（360×240 与 1398×1167 峰值都 1,481 MiB）。

## 8. prefill 块：`auto:32768`

启动不再警告 #669。单变量实测（同机、固定 prompt、每次新 nonce）：

| `--prefill` | 块 | 22K prompt | 72K prompt |
|---|---:|---:|---:|
| `auto` | 8192 | 2,416 t/s | 2,780 t/s |
| **`auto:32768`** | 32768 | **2,905 t/s** | **3,403 t/s** |

大块多摊销约 20%，尽管它每次从 SSD 现读专家。**别只看引擎启发式，单变量上机实测才定案。**

## 9. 模板

生产装的是 **froggeric v22.5 原版**（28,234 B / 28,226 字符）：effort 别名、think 标记、工具调用与截断修复，
`reasoning_effort` 由请求头控制。`setup.sh` 每次会把 unsloth 原装模板（9,993 B）写回去；改完模板要重启。
DavidAU 的 29 段模式合并模板（249,545 B）只用于复现 v5.3–v5.8，见 [TEMPLATE.md](TEMPLATE.md)。

## 10. 运维与健康判据

启动/停止/升级：[OPERATIONS.md](OPERATIONS.md)。三层判据缺一不可：

1. 进程：`pgrep -x strata` + `:8000` 在听
2. 版本/模板：`/props.build_info == "Strata 0.1.58"`，`chat_template` 长度 **28226**
3. 端到端：一条中文请求正文可读；`/metrics` 的 `live.state` 回到 `idle`

破限判据：来自 huihui v2 消融（96 张量），**不该**出现 ESP 控制向量行；8 条探针（6 对抗 + 2 对照）8/8 不拒答。

## 11. 已验证 / 未验证

**已在 v5.9（0.2.12 + 0.1.58 + UD-Q4_K_XL）上验证**：

- **历史口径**（2026-10-25，IQ3_S + 0.1.47，vLLM 基准，不代表 v5.9）：1K / 8K / 32K / 64K / 128K / 260K prefill 与 1K / 32K / 128K / 260K decode，2–4 路并发；见 [BENCHMARK.md](BENCHMARK.md) §1–§9
- v5.9 实际复测范围：6.25K / 12.7K prompt、4 路 512-token 并发、单流 default/low（见下条）
- 4×256K 配置与 KV 流式；`k8v4`；`--batch-mtp` 并发
- 视觉三色条 3/3（0.1.45 时代，样本小）；专家缓存 **32,595 MiB / 10,902 槽**；arena **73,450 MiB**（专家全驻 RAM）；启动后空闲显存 2,517 MiB
- **0.2.9 的 11 个新默认环境变量逐个在引擎进程里确认生效**（9 并行 + 2 GDN），0.2.12 上仍是同一组；`drafting = mtp`；**验收 29/29**；换模型后 6.25K prompt 2,752 t/s、12.7K 4,479 t/s、4 并发聚合 186–188 t/s（2026-11-08，同协议对照 IQ3_S）
- 0.2.11 引入、0.2.12 仍为 **FP16 weight cache 0 MiB**（需分配后留 3,072 MiB 空闲，本机余 ~2,900 MiB），因此上游 4 路 +5.7% 未复现；`pcie_frac` 由 0.55 变为按 CPU 实测的 **0.15**
- 0.1.43.1 曾连续跑 2 天 23 小时（被 cgroup 连带杀，不是自崩）

**未验证**：

- 多日稳定性、>4 路密集并发的扩展性
- 质量结论：WorkBuddy Office 只抽查了 4 题（2 错 2 对），不足以给整体分数；要全量得跑完 50 题
- 7×24 稳定性（换模型后只跑了当天）
