# 基准工具（vLLM 0.7.3 + Strata 适配）

用 **vLLM 官方 `benchmark_serving.py`** 对线上 Strata 服务做 prefill / decode / 并发测量。
方法学、矩阵与结果在 [../docs/BENCHMARK.md](../docs/BENCHMARK.md)。

## 一次性准备

```bash
python3.12 -m venv .venv-bench
.venv-bench/bin/pip install -r requirements.txt

# Strata pack 里有 vocab.json/merges.txt，没有 HF tokenizer_config；重建一个
.venv-bench/bin/python build_tokenizer.py \
  --src /path/to/Strata-data/packs/davidau-abliterated-iq3_s/tokenizer \
  --out tokenizer
```

`build_tokenizer.py` 的 pre-tokenizer 正则照抄 pack 里 `tokenizer.json` 的 `pre_pattern`（Qwen35）；
生成后用 `/v1/messages/count_tokens` 校验过：普通文本的本地计数与服务器计数只差模板固定开销。

## 跑矩阵

```bash
# 服务已在 127.0.0.1:8000；确认 /metrics 的 live.state == idle
.venv-bench/bin/python run_matrix.py --tokenizer tokenizer \
  --out results/$(date +%F)
```

每格：vLLM 客户端指标（TTFT/TPOT/吞吐）+ `/metrics` 的引擎侧每请求字段（prompt_ms/decode_tok_s/reused）。
`run_matrix.py` 会按时间窗与 prompt 长度过滤掉 warmup 和外部流量；`summary.json/.csv` 是汇总，
`raw/` 是每格 vLLM 原始 JSON，`logs/` 是 stdout。

## 口径

- **prefill**：客户端 TTFT（vLLM，含 HTTP/模板开销）与引擎 `prompt_ms`（纯引擎）分开报；
  `prefill_tok_s = prompt_tokens / 时间`。1K 档固定开销占比大，看 8K 以上。
- **decode**：vLLM `TPOT`（第 2 个 content token 起）与引擎 `decode_tok_s` 交叉验证；
  **不要用 ITL 百分位**——Strata 的 SSE 会把多个 token 压进同一次读，ITL 会被读缓冲打散。
- **并发**：`output_tok_s_wall` 是墙钟聚合（含 prefill），`per_stream_tok_s_wall` 是每路；
  TPOT 仍是单路纯解码。
- **缓存**：每格记录 `engine_cache_reused_ratio`，>0.05 视为污染，该格要重跑。
