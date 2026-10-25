# 基准实测（v5.3，2026-10-25）

> **一句话**：单流 decode **167–177 t/s**（1K→256K 基本平坦），单流 prefill **约 3.9–4.05K t/s**
> （8K 以上；1K 档 TTFT 0.82 s 里固定开销占大头），**260K 首次实测**：TTFT 66.2 s、decode 168 t/s。
> 4 路并发只把聚合吞吐拉到 **1.2×**，每路掉到 50 t/s——`--batch 4` 是并发会话，不是吞吐扩展。

工具是 **[vllm-project/vllm](https://github.com/vllm-project/vllm) `v0.7.3` 的 `benchmarks/benchmark_serving.py`**
（vendored 在 [`bench/vllm_0.7.3/`](../bench/vllm_0.7.3/)，5 处 Strata 适配逐条列在
[PATCHES.md](../bench/vllm_0.7.3/PATCHES.md)）。请求生成、客户端计时和指标聚合由 vLLM 代码完成；
本仓库只加驱动脚本 [`bench/run_matrix.py`](../bench/run_matrix.py)。

## 1. 测试身份

| 项 | 值 |
|---|---|
| 主机 | 单 RTX 4090 **48GB**（sm_89），38 核 / 76 线程，247GB RAM |
| 引擎 | Strata **0.2.2** 服务端 + **engine 0.1.47** |
| 模型 | DavidAU Qwen3.8-Flash-Next abliterated **IQ3_S**（83.86GB，2 shards，3.5 bpw） |
| 服务地址 | `127.0.0.1:8000`，`/v1/chat/completions` |
| 并发 / 上下文 | `--batch 4`，`--max-context 262144` |
| KV | `--kv k8v4 --kv-resident 32768` |
| 投机 | `--spec 4 --spec-min-p 0.5 --batch-mtp` |
| prefill / 显存 | `--prefill auto:32768`，`--vram-reserve-mib 2560`，`--vision` |
| 专家缓存 | **33,889 MiB / 17,417 槽**（`/metrics` 的 `expert_cache_mib` / `expert_slots`；0.1.45 时代是 32.28 GiB / 11054） |
| 启动后空闲显存 | 2,231 MiB |
| 测量窗口 | 2026-10-25 13:31–13:39（服务未重启、未停服），`live.state` 全程 idle，`engine_cache_reused_ratio = 0` |

配置原文（args 逐字）见生产 config `strata-davidau-iq3_s-022.json`；`--batch 4` 由 serve 层补齐。

## 2. 方法

### 2.1 请求与长度

- 不使用 ShareGPT：用固定 200 词词表拼出合成文本，按 tokenizer 截到目标 token 数。
  **这是补丁 4**：vLLM 原版「随机 token id → decode」在 byte-level BPE 上不可逆，会把 nominal 1K prompt
  变成服务器侧的 3,346 token。
- 每个 cell 用不同 seed，prompt 全唯一；`/metrics` 的 `reused` 每格为 0，**没有前缀缓存命中**。
- 关思考：`chat_template_kwargs.enable_thinking=false`（补丁 3）。否则 TTFT 测到的是「思考结束」，
  不是 prefill；decode 的 token 速率不受影响。
- `max_tokens` 固定；服务端不认 `ignore_eos`（实测早停），所以用固定词表 prompt + 足够长的 max_tokens
  让每次输出达到目标长度；每格记录实际 `output_tokens`。

### 2.2 计时与聚合

| 指标 | 来源 | 说明 |
|---|---|---|
| TTFT | vLLM 客户端 | 请求发出 → 第一个 **content** token（补丁 2）。含 HTTP/模板/排队 |
| TPOT | vLLM 客户端 | 第 2 个 content token 起的平均间隔；`decode_tok_s = 1000/TPOT` |
| prefill_tok_s（客户端） | 本仓库 | `server prompt_tokens / TTFT`；1K 档被固定开销压低 |
| prefill_tok_s（引擎） | `/metrics.requests[].prompt_ms` | 引擎纯 prefill 时间，不含模板/HTTP；更接近硬件上限 |
| decode_tok_s（引擎） | `/metrics.requests[].decode_tok_s` | 引擎纯 decode 时间；与 TPOT 交叉验证 |
| 聚合吞吐 | vLLM `output_throughput` | 墙钟（含 prefill）× 全部输出 token；并发格用这个 |
| 每路墙钟 | `output_throughput / completed` | 含该请求的 TTFT+decode |

- 每格取**中位数**；vLLM 的 p99 也存进 `summary.json`。
- **不要看 ITL 百分位**：Strata 的 SSE 会把多个 token 压进同一次 socket 读，ITL 被读缓冲打散
  （p50 可到 0.08ms）；TPOT 用整段时长/ token 数，不受影响。
- 每格先跑一次 warmup（**不同**的短 prompt，补丁 5）；原版会用 `input_requests[0]` 再测一次，
  Strata conversation cache 会把它变成 99% 命中。

### 2.3 与旧记录不可直接对表

本页之前的数字来自自写 `bench_once.py`/`llm_speedtest`，口径是「客户端掐表 + 服务端 usage」；
本页是 vLLM 的 TTFT/TPOT + 引擎 `prompt_ms/decode_tok_s`。**两套口径不能相减**。
例：旧记录 0.2.2 的 decode 是 131–141 t/s，本页是 167–177 t/s——差异主要来自计时/统计方式，
不能据此宣布引擎变快。要比较版本，必须用本页同一脚本做 A/B。

## 3. Prefill

`input_len` 是请求的目标 token；`实际` 是 `/metrics` 的中位 `prompt_tokens`（含模板约 +12）。
单流，输出 8 token（P256K 为 4）。

| cell | 目标 input | 实际 prompt | n | TTFT 中位 | TTFT p99 | 客户端 prefill | 引擎 prompt_ms | 引擎 prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1K | 1,024 | 1,036 | 5 | **0.820 s** | 0.823 s | 1,263 t/s | 802 ms | **1,291 t/s** |
| P8K | 8,192 | 8,204 | 3 | **2.358 s** | 2.364 s | 3,480 t/s | 2,333 ms | **3,517 t/s** |
| P32K | 32,768 | 32,780 | 3 | **8.369 s** | 8.400 s | 3,917 t/s | 8,321 ms | **3,939 t/s** |
| P64K | 65,536 | 65,548 | 2 | **16.247 s** | 16.374 s | 4,034 t/s | 16,172 ms | **4,053 t/s** |
| P128K | 131,072 | 131,084 | 2 | **32.721 s** | 33.137 s | 4,006 t/s | 32,591 ms | **4,022 t/s** |
| P256K | 260,000 | 260,012 | 1 | **66.237 s** | — | 3,926 t/s | 65,993 ms | **3,940 t/s** |

读取：

- **8K 以上是线性**：8K→256K 的边际速率 **(260012−8204)/(66.237−2.358) = 3,942 t/s**；
  与各档引擎值一致。TTFT ≈ 0.28 s + 253.7 µs/token。
- **1K 档不能看吞吐**：固定开销（HTTP/模板/调度）约 0.55–0.6 s，把 1K 的 prefill 压到 1.26K t/s；
  但 TTFT 0.82 s 对交互仍是可用的。
- 260K 输出 4 token、TTFT 66.2 s；服务端 `max_context=262144`，`260012+模板` 未越界。
- `--prefill auto:32768` 的收益在 0.1.45 时代是 +20%（22K/72K 单变量），本页没有重跑该单变量。

## 4. Decode

单流；`input_len` 为上下文，`output` 为生成上限。TPOT 是 vLLM 客户端中位数；引擎列来自 `/metrics`。

| cell | 上下文 | 输出 | 实际输出 | TPOT 中位 | 客户端 decode | 引擎 decode | 墙钟输出 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D1K | 1,024 | 512 | 512 | **5.96 ms** | **167.7 t/s** | 167.5 t/s | 138.0 t/s | 含 3×0.83s prefill |
| D32K | 32,768 | 256 | 256 | **5.97 ms** | **167.4 t/s** | 166.5 t/s | 22.4 t/s | 含 3×8.4s prefill |
| D128K | 131,072 | 256 | 256 | **5.64 ms** | **177.3 t/s** | 176.8 t/s | 6.5 t/s | 含 2×32.7s prefill |
| D256K | 260,000 | 256 | 256 | **5.94 ms** | **168.2 t/s** | 167.6 t/s | 3.8 t/s | 含 66.4s prefill |

读取：

- **decode 对上下文长度基本不敏感**：1K/32K/128K/256K 都在 167–177 t/s；130K 那次偏高是 2 次中位，
  不能当作"更长更快"。
- 引擎侧与客户端侧差 <1%，两套独立测量互证。
- 墙钟输出列不是 decode 速率：256 token 的输出里，32K 上下文的 prefill 占 8.4 s，墙钟自然只有 22 t/s。
- 旧记录称「32K decode 111 t/s、100K 109 t/s（0.1.45）」；本页 0.1.47 是 167/177 t/s。
  **不做因果归因**（见 §2.3），但可作为下次升级 A/B 的当前基线。

## 5. 并发

`--batch 4` = 4 个独立 slot session；每格用 `--max-concurrency` 限制同时在跑的路数。
`聚合` = 每路 TPOT 速率 × 路数（纯 decode 估算）；`墙钟` = vLLM 的 `output_throughput`（含 prefill）。

| cell | 路数 | prompt | 每路输出 | TTFT 中位 | TPOT 中位 | 每路 decode | 聚合 decode | 墙钟输出 | 相对单流 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2K | 2 | 4 × 1K | 512 | 0.924 s | 10.64 ms | 93.9 t/s | **188 t/s** | 148.5 t/s | 1.12× |
| C4K | 4 | 8 × 1K | 512 | 1.713 s | 19.85 ms | 50.4 t/s | **202 t/s** | 176.5 t/s | 1.28× |
| C4_8K | 4 | 4 × 8K | 256 | 6.155 s | 30.55 ms | 32.7 t/s | **131 t/s** | 65.3 t/s | 0.78× |

读取：

- **并发不提升聚合吞吐**：2 路聚合 1.12×、4 路 1.28×（墙钟 1.08× / 1.28×）；每路从 167 掉到 94 / 50 t/s。
  这与早期记录「4 路聚合 137 vs 单流 118–124」的形态一致：MoE CPU experts + 单卡是共享瓶颈。
- **8K 上下文的 4 路更差**：聚合 decode 131 t/s，低于单流 166；墙钟 65 t/s。长 prompt 的 prefill 与 decode
  互相挤占，4 路并发不是"免费的吞吐"。
- 所以 `--batch 4` 的正确用途是**同时服务 4 个会话**（交互场景），不是批量吞吐。要高聚合吞吐应另选引擎/多卡。
- TTFT p99 在并发下明显变长（C4K 3.45 s），因为 4 个 prefill 排队；这会影响交互体验。

## 6. 与历史/上游的关系

- **v4（FreeToken）**：单流 59–64 t/s、4 路聚合 117 t/s。本页不是同窗口 A/B，不直接比；
  形态上仍是「单流约 2.7×、4 路纯 decode 聚合约 1.7×（墙钟 1.5×）」。
- **0.1.45（v5.1/v5.2）**：旧记录 32K decode 111.4、100K 109.0；本页 0.1.47 是 166.5 / 176.8。
  差异含引擎版本 + 计时口径，**不可归因**。
- **0.1.48 / 0.2.3 / 0.2.4**：当前不升级（理由与触发条件见
  [STRATA-UPSTREAM-2026-10-25.md](STRATA-UPSTREAM-2026-10-25.md)）。本页数字就是未来升级 A/B 的**基线**：
  decode ≥+8% 或 prefill ≥+15%、且 260K 不出现 VRAM 崩塌，才考虑动。
- **上游社区**：4090 + IQ3_S + 0.1.47 的公开 A/B 不存在；最接近的 sm_89 邻证是 4060 Ti 单卡
  0.1.43→0.1.48 +15%（[#3966](https://github.com/Niko1221/Strata/issues/3966)），拆不出 0.1.47→0.1.48 的增量。

## 7. 限制与未测

- **8 路以上并发未测**；`--batch 4` 是当前上限。
- **多日稳定性未测**；本页是单窗口 8 分钟。
- **前缀缓存命中延迟未测**：本页特意全部新 prompt；生产里的复用路径（`/metrics` 的 `reused`）没有单独基准。
- **视觉未测**：`--vision` 在跑，但本页全是文本。图像编码器峰值显存见 [STRATA.md §7](STRATA.md)。
- **质量未测**：只有历史 15 题数学与 0/14 破限探针，不构成质量结论。
- **1K prefill 的固定开销**没有拆成 HTTP / 模板 / 调度三段；只给了总 TTFT。
- **`ignore_eos` 服务端不认**：输出长度靠 prompt 与 max_tokens 控制，不是硬保证；本次每格实际输出都达到了目标。

## 8. 复现

```bash
# 1) 依赖与 tokenizer（pack 里有 vocab/merges，没有 HF config）
python3.12 -m venv .venv-bench
.venv-bench/bin/pip install -r bench/requirements.txt
.venv-bench/bin/python bench/build_tokenizer.py \
  --src /path/to/Strata-data/packs/davidau-abliterated-iq3_s/tokenizer \
  --out bench/tokenizer

# 2) 确认服务 idle、没有外部流量
curl -s 127.0.0.1:8000/metrics | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["live"]["state"],d["totals"]["requests"])'

# 3) 全矩阵（约 8–10 分钟）或单格
.venv-bench/bin/python bench/run_matrix.py --tokenizer bench/tokenizer \
  --out bench/results/$(date +%F)
.venv-bench/bin/python bench/run_matrix.py --tokenizer bench/tokenizer \
  --out /tmp/one --only P32K,D32K
```

结果：`summary.json`（全字段）、`summary.csv`（扁平）、`raw/<cell>.json`（vLLM 原始）、`logs/<cell>.log`。
本次原始数据在 [`bench/results/2026-10-25/`](../bench/results/2026-10-25/)。

## 9. 附：本次所有 cell 的 vLLM 原始指标

| cell | duration s | TTFT ms | TPOT ms | output tok/s (wall) | total tok/s (wall) |
|---|---:|---:|---:|---:|---:|
| P1K | 4.29 | 820.5 | 4.33 | 9.3 | 1,203.8 |
| P8K | 7.21 | 2,357.7 | 6.54 | 3.3 | 3,411.4 |
| P32K | 25.18 | 8,369.3 | 6.68 | 0.95 | 3,905.4 |
| P64K | 32.59 | 16,247.1 | 6.90 | 0.49 | 4,021.9 |
| P128K | 65.54 | 32,721.1 | 7.06 | 0.24 | 3,999.8 |
| P256K | 66.25 | 66,236.5 | 5.51 | 0.06 | 3,924.3 |
| D1K | 11.13 | 827.0 | 5.96 | 138.0 | 413.9 |
| D32K | 28.92 | 8,420.9 | 5.97 | 22.4 | 3,421.8 |
| D128K | 67.77 | 32,655.1 | 5.64 | 6.5 | 3,874.4 |
| D256K | 67.92 | 66,399.7 | 5.94 | 3.8 | 3,831.9 |
| C2K | 13.79 | 924.3 | 10.64 | 148.5 | 445.4 |
| C4K | 22.19 | 1,713.1 | 19.85 | 176.5 | 545.6 |
| C4_8K | 13.97 | 6,155.1 | 30.55 | 65.3 | 2,411.5 |
