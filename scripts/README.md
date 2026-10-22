# 脚本索引

## 现行

| 脚本 | 用途 |
|---|---|
| `arm_start.sh` | 起 A/B 单臂（`<serve-dir> <config.json> <log>`），等模型加载完 |
| `ab_probe.py` | 客户端口径单变量 A/B：decode / TTFT（含 reasoning 增量） |
| `long_ctx_probe.py` | 长上下文 decode 稳定性（N 条 100K 提示各生成 512 token） |
| `bench_once.py` | 单次 prefill/decode 测量（`MODEL_DIR` 默认值指向已退役的 NVFP4 目录，用前先设） |
| `verify_model_index.py` | 按 safetensors index 校验分片齐全 |

## 历史（对应引擎已删除）

`archive/` 下：`install.sh`（v1 Lsglang）、`build_upstream_flashnext.sh`（v3 sglang wheel）、
`start_lsglang.sh` / `start_lsglang_upstream.sh`（v3）、`start_freetoken.sh` / `install_freetoken_patches.sh`（v4）。

另有 `build_engine_upgrade.py`（0.1.45 时代自建引擎 + GPU vision 编码器，绕开 `setup.py` 的配置重写与分片校验）：
0.2.0 起上游提供 Linux 预编译引擎，生产升级已改走 [OPERATIONS §1c](../docs/OPERATIONS.md)，该脚本只在新卡/新 CUDA 时参考。

这些脚本保留为"当时怎么起的"记录，**现在跑不起来**（`~/cherrywork/freetoken-tools`、`~/.venv-lsglang` 都已删除）。
