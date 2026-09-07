# 排障手册 (Troubleshooting)

全部为实测踩坑记录。

## 1. 启动 OOM：47.7GB ngram 表被塞进 GPU (Boot OOM: 47.7GB n-gram table to GPU)

症状：启动时 `CUDA out of memory. Tried to allocate 47.69 GiB`（= 51.2B 参数的 ngram 查找表，uint8 存储），堆栈在 `qwen4_exp.py _init_qwen4_exp_layer_extensions → Qwen4ExpPLELayer`。

实测触发组合：
- ❌ 去掉 `--disable-shared-experts-fusion`（fusion 开启）
- ❌ 删除 `--max-total-tokens`（自动内存规划把 ngram 表放 GPU）
- ❌ `NUMA 变量注释 + PREFETCH_WINDOW=0 + POWER_SAVING=0 + max-running-requests 4` 的组合（prefetch0/power0 单独改动是安全的，见 BASELINE）

✅ 安全配置（本文档默认）：NUMA 3 变量启用 + 显式 `--max-total-tokens 265216` + `--disable-shared-experts-fusion` + 其余默认。
背景：`ple_offload_embedding` 的自动解析条件是 `is_cuda and dtype==bf16`（overrides.py），NVFP4 模型下 ngram offload 依赖上述组合。

## 2. 工具调用返回 tool_calls: null (Tool calling returns null)

模型输出了 `<tool_call>` 文本但 API 的 `tool_calls` 字段为 null——**`--tool-call-parser qwen3_coder` 必须显式传**，sglang 的"自动检测"不会真正启用解析。

## 3. sglang 日志吞吐字段不可信 (Log throughput field is unreliable)

日志 `input throughput (token/s)` 与客户端实测差 **12~57 倍**（实例：日志 16 t/s vs 客户端 918 t/s；日志 65 t/s vs 815 t/s；甚至出现过 0.05 t/s）。**测速一律用客户端计时**（发请求掐表 + `usage.prompt_tokens`），见 scripts/bench_once.py。

## 4. chunked-prefill-size 4096 使 prefill 暴跌 (chunk 4096 kills prefill)

8k prompt prefill：4096 → **-38%**（736 vs 1195 t/s）；8192 与 16384 无差异。保持 8192。

## 5. --cuda-graph-backend-prefill full 崩溃 (cgprefill full crashes)

`RuntimeError: QSA CUDA graph state is not initialized`——QSA（Qwen sparse attention）后端与 prefill CUDA graph 捕获冲突。必须 `disabled`（这也是 Lsglang release 命令的默认）。

## 6. resident 层数 vs 256K 上下文 (Resident layers vs 256K context)

- 每层 experts 权重 ~1.4GB 驻显存；显存总量不变是 **cache 让位**（mem-fraction 0.95 盘子里：Mamba 先让、见底后 KV 页表暴缩）
- 实测：6 层 KV=318464 ✅、7 层 KV=255616 ❌（<262144）、10 层 KV=70848 ❌
- **6 层（0-5）是保 256K 的最多层**；要更多层须牺牲上下文

## 7. reasoning_effort 枚举 (Effort enumeration)

- sglang API 层收：`none/minimal/low/medium/high/xhigh/max`（ultracode/extreme 在 API 层 400）
- 模型自带模板只认 `xhigh/medium/low`，`high` 会 400
- **解决（推荐）**：替换为 froggeric v22.5 模板（原生收全别名，见 REPRODUCTION §5），实测 7 枚举全通过
- 早期手写映射补丁已被 v22.5 取代

## 8. MTP 投机解码 (MTP speculation)

- 可启动：`--speculative-algorithm NEXTN --speculative-eagle-topk 1 --speculative-num-draft-tokens 1 --speculative-num-steps 1`（NEXTN 是 EAGLE 别名；需显式 topk 否则断言失败；内嵌 MTP 无需外部 draft 模型）
- 模型以 `Qwen4ExpForCausalLMMTP` 加载，draft/verify 图正常，accept rate 0.62~0.72
- **但 decode 无收益**（实测 35.7 vs 无 MTP 36~40 t/s）：CPU 核数是瓶颈，投机减少 GPU 步数但 CPU 每步计算量不减反增

## 9. ModelScope revision (ModelScope revision)

TomPython 项目记录的 HF revision（`7b7192...`）在 ModelScope **不存在**（只有 master）。校验靠 `verify_model_index.py`（206 shards）。

## 10. N-gram 表与 offload (N-gram table & offload)

- 51.2B 参数的 ngram 查找表在 **layer 1 的 PLE**（`layers.1.ple.ple_embedding.ngram_embedding`，128 shards，每 token 只 touch ~100-200KB）
- NVFP4 存储 25.6GB；`ple_offload_embedding` 开启时驻内存（pinned host）
- 更优方案（参考 vLLM PR #54070 / DGX Spark 社区）：NVMe mmap 磁盘 offload（省内存，代价 ~8%）

## 11. 温度漂移与测速 (Thermal drift in measurements)

同配置前后重测 decode 37.8 → 40.4 t/s（±7% 漂移，CPU 满载发热降频）。**A/B 对比必须在同窗口内做**，跨小时对比会误判参数效果。

## 12. pgrep 自匹配陷阱 (pgrep self-match trap)

`pgrep -f 'sglang serve'` 会匹配命令行含该字符串的 shell 自身（自杀 → exit 144）。用 `ps aux | grep '[s]glang serve'`。

## 立即停止条件 (Stop conditions)

host OOM / swap 持续增长 / NVIDIA Xid / 不可恢复 kernel fault / 同失败连续两次无新证据。
