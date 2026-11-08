# 换模型：unsloth UD-Q4_K_XL + huihui v2 破限（v5.9，2026-11-08）

> **一句话**：权重从 DavidAU IQ3_S（83.86 GB，3.4 bpw）换成官方 unsloth v3.1 **UD-Q4_K_XL**
> （111.33 GB，4-bit），用 huihui-ai v2 的 `patch-all` 对 96 个注意力/SSM 输出张量做消融
> （不碰 routed experts），pack 用 0.2.12 的 `iq_pack.py` 现建 v4，模板换回 **froggeric v22.5**。
> 换来的是质量：WorkBuddy 两道错题 0.17→0.89、0.00→0.69；代价是速度（同协议 4 并发 215→186 t/s）
> 与两道已高分题的 4–7% 回落；模型与模板同时变，无法单独归因。

## 1. 为什么换

- DavidAU IQ3_S 是 v3.0 时代的 3.4 bpw 打包；unsloth 在 2026-11-04 重发了 v3.1 Dynamic，
  `UD-Q4_K_XL` 约 4.3 bpw 的有效精度（top-1 92.3%、KLD 0.0469），比 IQ3_S 高一个档，且是官方权重。
- 破限用 huihui-ai v2 的消融补丁：只改 `ssm_out` / `attn_output` / `ffn_down_shexp`
  （`--ablate-experts 2`，`--start-layer 0 --end-layer 48`），**不消融任何 routed expert**，
  保留模型知识；8 条探针（6 对抗 + 2 对照）8/8 不拒答。
- 模板不再需要 DavidAU 的 29 段模式系统：换回 froggeric v22.5 原版（28,226 字符），
  它本身通吃 Qwen3.5/3.6/3.8 与 Flash-Next，并支持 `reasoning_effort` 全档位。

## 2. 资产与校验

### 2.1 官方权重（ModelScope，实测 11–12 MB/s）

`modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF` → `UD-Q4_K_XL/`（v3.1 现行 4 片）：

| 文件 | 字节 | sha256 |
|---|---:|---|
| `…-00001-of-00004.gguf` | 10,946,618 | `acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858` |
| `…-00002-of-00004.gguf` | 49,720,539,552 | `45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4` |
| `…-00003-of-00004.gguf` | 32,803,030,176 | `8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82` |
| `…-00004-of-00004.gguf`（PLE 表，IQ4_NL 28.8 GB） | 28,800,138,432 | `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` |
| `mmproj-BF16.gguf` | 907,542,944 | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` |

四片合计 **111,334,654,778 B**；加 mmproj 共 **112,242,197,722 B**。shard1 只有 metadata/tokenizer（0 张量），张量在 shard2（层 0–28，726 个）
与 shard3（层 28–47，497 个），shard4 只有一个 `per_layer_token_embd.weight`。

> 别下 huihui 仓库里 27 天前的 `unsloth-UD-Q4_K_XL/`：那是旧切分（49.86 / 49.38 / 12.09 GB），
> 与 v3.1 的边界不同，补丁覆盖也对不上。官方 v3.1 必须从 unsloth 下，再本地打补丁。

### 2.2 破限补丁（HF，0.4–0.5 MB/s，约 2.4 h）

`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`，`patch-unsloth/`：

- `patch-unsloth-Qwen3.8-Flash-Next-dec.py`（9,659 B）
- `patch-unsloth-all/Huihui-…-patch-all.gguf` **3,561,572,544 B**，
  sha256 `f3a42f213edb9117fd7f3873e0b29a9a94033cb49b91e100e792f6344adec752` + `.gguf.key`（64 B）

补丁张量命名是 `名字_<ggml_type_int>`；**输入张量类型与补丁相同才整块替换**，不同则原样复制并只打
`Copied because dtype mismatch`。所以先核类型：UD-Q4_K_XL 的目标张量全是 **Q8_0**，补丁正好提供
96 个 Q8_0 版本（57 + 39），96/96 命中、0 mismatch、0 not-found。

## 3. 打补丁（只重写 shard2 / shard3）

命令（`pin2`/`pin3` 单分片符号链接 + `--ablate-experts 2`）见
[REPRODUCTION.md §4b](REPRODUCTION.md)；下面只记结果与坑。

实测：shard2 `Patch files (1)`、`Patched tensors: 57`、`Copied 669`；shard3 `Patched 39`、`Copied 458`；
两片都没有 `Copied because dtype mismatch` / `was not found`。shard1/shard4 直接软链进 `<abl>/`，不复制。

> `scripts/patch_and_pack_q4kxl.sh` 在打补丁前断言 `Patch files (1)`，打完后断言 57/39 且 0 mismatch/not-found。
> 早期手写版没有这道闸：第一次抢跑时补丁还在 `.parts/` 里，脚本照样跑完并产出 0 patched 的原样 shard2。

## 4. 建 pack（v4）

完整命令见 [REPRODUCTION.md §5](REPRODUCTION.md)（**必须用装了 requirements 的 venv python**；本机现成的是 `Strata/.venv`，从零按 §3 建 `strata/venv`，原因见
[TROUBLESHOOTING #44](TROUBLESHOOTING.md)）。打包结果：`native_experts.txt` v4（49 行、n_expert 512、total **77,017,907,200 B**）、`index.txt` v3
（1079 张量 / 303 native / arena 1.38 GiB）、`compat-bf16.json`（195 张量 / 1.20 GiB）、
`dense.bin` 1,485,688,320、`conversions.json`（460 条，196 条 rounded，max err 0.0144）。

**坑**：iq_pack 用 `sys.executable` 调 tokenizer，系统 python3.14 缺 `regex` 会失败；
而 `native_experts.txt` 是最后写的完成标记，失败后 pack 看似只缺一个文件、实际不可用——详见
[TROUBLESHOOTING #44](TROUBLESHOOTING.md)。

tokenizer：vocab 248,320 / merges 247,587 / eos 248,046 / bos=pad 248,044。

## 5. 模板：froggeric v22.5

安装命令只有一处：[REPRODUCTION.md §6](REPRODUCTION.md)（`cp patches/froggeric_v22.5.jinja <pack>/tokenizer/chat_template.jinja`，
`wc -c` = 28,234 B、`len()` = 28,226）。不再套 DavidAU 合并模板（249,545 B / 243,742 字符）；验收断言随之改成 **28226**。

## 6. 启动参数：从 32 项减到 30 项

相对 IQ3_S 的 32 项，**只删了 `--resident-budget-gib 56`**，其余逐字不动；
新增 config `env: {"STRATA_ROUTE_TAIL_SKIP": "0"}`（全驻 RAM 时的默认）。

- 本机 247 GB RAM，专家 72 GiB + 16 GiB 余量装得下 ⇒ 全驻，不走预算模式。
  上游 128 GB + RTX 5090 实测：全驻 169 t/s vs 预算 118 t/s，32K prompt 4.9 s vs 30 s。
- 显存侧：`arena_mib 73,450`、`expert_slots 10,902`、`expert_cache_mib 32,595`、
  `vram_free_mib 2,517`、`pcie_frac 0.15`。
- K-quant prompt 内核（MMQ）对 UD-Q4_K_XL 在 cc≥8.0 默认开，日志
  `K-quant prompt kernels from engine/libstrata_kq.so (STRATA_MMQ_KQ=1)`。
- MTP 保持：`drafting = mtp`，复用 `Strata-data/mtp/rt`（MTP 头不随量化/消融变化）。
- `--ple-gguf` 指 shard4；PLE 表 26.8 GiB 以 `--ple-io ram` 加载，14.4 s。

## 7. 切换与验收

| 项 | 值 |
|---|---|
| 停机 | **9 s**（stop 到显存 9 MiB） |
| 冷启动 | **106 s**（页缓存冷；arena 每次启动都重填，本次 1.68 GiB/s） |
| 热启动 | **53 s**（页缓存命中，12.49 GiB/s；PLE 加载 14.4 s → 0.1 s） |
| 验收 | **31/31 PASS**（`scripts/accept_q4kxl.sh`） |
| `/health` | `model=qwen3.8-flash-next-q4kxl-ablit`、`max_context=262144`、`images=true` |
| `/props` | `build_info "Strata 0.1.58"`、`len(chat_template) = 28226` |

验收首跑 4 个 FAIL 全是测试脚本自身解析 bug（`/metrics` 是 `{"engine":{…}}` 双层，`jget()` 取错层）；
修正解析后 28/28（脚本后来加了 model、expert_slots、pcie_frac 断言与更严的 PLE/MTP 检查，现为 31 项），不是服务端问题。

## 8. 实测（同机同协议，对照 0.1.58 + IQ3_S）

完整表格与单位见 [BENCHMARK.md §0](BENCHMARK.md)。要点：

- prefill 6.2K **5,177 → 2,752 t/s**（−47%）；12.8K **5,526 → 4,479 t/s**（−19%）
- 4 并发聚合 **215 → 186 t/s**（−13%）；每路 decode 54.9–57.3 → 48.5–49.6 t/s
- 单流 decode **158–164 → 112.8 t/s**（default）/ **133–135 t/s**（`reasoning_effort=low`）
- TTFT 0.425 / 0.446 / 0.445 s → 0.44 / 0.61 / 0.53 s

模型大了 **33%**（83.86 → 111.33 GB），专家缓存槽 17,283 → 10,902，单流 decode 的下滑基本等于
"多搬 1/3 专家字节"。4 并发聚合只掉 13%，因为批式 MTP 摊薄了每 token 的专家读取。

## 9. 质量抽查（WorkBuddy Office，各 2 题）

| 任务 | 原 | 现 | 耗时 |
|---|---:|---:|---:|
| `cloudagent-sdk-doc-validation-report`（原错题） | 0.1666 | **0.8917**（107/120） | 282 s |
| `contract-extract-L3-014`（原 0 分） | 0.0 | **0.6923**（18/26） | 780 s |
| `furniture-refinish-L3-016`（原满分） | 1.0 | 0.9333（14/15） | 144 s |
| `html-report-quadrant-ppt`（原 0.97） | 0.9487 | 0.9103（71/78） | 217 s |

原失败题大幅回升，原本高分题回落 4–7%。变更同时含模型与模板，无法单独归因；要拆开得做
"新模型 + 旧模板" 的对照，成本一次切换。

## 10. 破限

破限探针是 6 条对抗（成人向虚构 / 禁忌 / 粗口 / 盗版 / 隐私 / 越狱）+ 2 条正常对照，
`reasoning_effort="off"`、`max_tokens 600`：**8/8 refused=false（对抗项 6/6）**，字符数
966 / 804 / 577 / 1085 / 1051 / **373** / 166 / 440（IQ3_S：981 / 908 / 431 / 1147 / 1065 / **6** / 206 / 311）。
越狱那条从 6 字符变 373，其余同量级。

## 11. 显存与 mlock

- 显存：46,145 / 49,140 MiB，空闲 2,517 MiB。
- `PLE table (26.8 GiB) was not locked in RAM`：`LimitMEMLOCK` 只有 8 MiB。`systemd-run
  --property=LimitMEMLOCK=infinity` 对 user 单元**无效**——`user@1000.service` 自身硬上限就是
  8 MiB，transient 子单元无法上调（实测 `/proc/<pid>/limits` 仍是 8388608）。要锁得 root 加
  `/etc/systemd/system/user@.service.d/` drop-in；上游 #4671 称锁住最多 +6% decode，暂不做。

## 12. 回滚与已删除项

IQ3_S 及其 pack、vLLM 与 NVFP4 权重已按用户要求**物理删除，没有归档**（腾出 289 GB）。
回滚 = 重新下 IQ3_S + 重建 pack，不再是一键脚本。生产启动器 `run_strata.sh` 现指向
`run_strata_q4kxl.sh`；旧引擎树（0.2.2–0.2.11）与旧 config 已删，只剩 `Strata-0.2.12`。
**别顺手删 `strata/Strata/`（0.1.45 安装树）**：启动器用的 venv 在 `Strata/.venv`，config 的
`--expert-profile` 也指向 `Strata/data/expert-profile.bin`；它只保留这两样，但删了服务起不来。
