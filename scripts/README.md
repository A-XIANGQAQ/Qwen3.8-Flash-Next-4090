# 脚本索引

## 基准（现行）

| 脚本 | 用途 |
|---|---|
| [`bench/run_matrix.py`](../bench/run_matrix.py) | prefill/decode/并发矩阵；驱动 vLLM `benchmark_serving.py`，同时抓 `/metrics` |
| [`bench/build_tokenizer.py`](../bench/build_tokenizer.py) | 从 Strata pack 的 `vocab.json`/`merges.txt` 重建 HF tokenizer（vLLM 需要） |
| [`bench/vllm_0.7.3/`](../bench/vllm_0.7.3/) | vendored vLLM 0.7.3 基准脚本 + 本地补丁说明（[PATCHES](../bench/vllm_0.7.3/PATCHES.md)） |

方法、矩阵与结果见 [docs/BENCHMARK.md](../docs/BENCHMARK.md)。

## 引擎 A/B 与校验（现行）

| 脚本 | 用途 |
|---|---|
| `arm_start.sh` | 起 A/B 单臂（`<serve-dir> <config.json> <log>`），等模型加载完 |
| `ab_probe.py` | 客户端口径单变量 A/B：decode / TTFT（含 reasoning 增量） |
| `long_ctx_probe.py` | 长上下文 decode 稳定性（N 条 100K 提示各生成 512 token） |
| `verify_model_index.py` | 按 safetensors index 校验分片齐全（历史 NVFP4 用；GGUF 分片用 `sha256sum`） |
| `find_ple_shard.py` | 查 PLE 张量在哪个 GGUF 分片（`--ple-gguf` 不能指错） |

## 历史（对应引擎已删除）

`archive/scripts/` 下：`bench_once.py`（自写单次测量，已被 vLLM 基准取代）、`build_engine_upgrade.py`
（0.1.45 时代自建引擎）、`start_lsglang*.sh` / `install.sh` / `build_upstream_flashnext.sh`（v1/v3）、
`start_freetoken.sh` / `install_freetoken_patches.sh`（v4）。这些脚本**现在跑不起来**，只作记录。
