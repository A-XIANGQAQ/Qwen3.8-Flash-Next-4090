# Strata 0.2.12（engine 0.1.58）：速度没变，价值在工具调用

- 调研 + 升级时间：2026-11-08 10:40–11:05（Asia/Shanghai）
- 数据来源：GitHub Releases API（v0.2.12 全文）、`SHA256SUMS.txt`、`git trees/v0.2.12?recursive=1`（**6,431 项 / 5,816 blob**）+ raw 逐文件抓取（`serve/ strata_control/ tools/ strata_cli/` 过滤后 **503 文件，每个 `git hash-object` 都等于 tree 的 blob sha**）、engine 0.1.58 包内 `BUILD.json`、官方 Qwen3.8-Flash-Next-FP8 仓库 commit 对比
- 升级前：Strata **0.2.11 + engine 0.1.57**（v5.7），DavidAU Qwen3.8 IQ3_S，单 RTX 4090 48GB（sm_89），`"parallel": 4`（服务端补 `--batch 4`），`--mtp` + `--batch-mtp` + `--spec 4`
- 上一篇：[STRATA-UPSTREAM-2026-11-06.md](STRATA-UPSTREAM-2026-11-06.md)

---

## 0. 结论

**速度没有变化（±2% 重复测量噪声内），这次拿到的是工具调用与引擎健壮性修复。** 已于 2026-11-08 10:49:56 切到 0.2.12 + 0.1.58，停机 **48 s**，验收 **24/24**，模型与模板（243,742 chars）一个字节没动。

- **头部性能项 tuned DFlash2 不适用于本机**：它只在「专家全部驻留显卡」的 NVIDIA / DGX Spark、Mac、Strix Halo、Intel Arc、12 GB NVIDIA 上自动启用；我们是 4 slots + 专家流式（26.65 GiB RAM + VRAM tier），Auto 仍然选 MTP，**不会下载那两个 339 MB / 522 MB 的 drafter**（上游明写：用 MTP 时不下载任何东西）。
- **FP16 weight cache 依旧 0 MiB**：规则没变，仍要求「分配后卡上留 3,072 MiB 空闲」，本机启动后只剩 ~2,900 MiB。
- **真正值钱的是 `#4828`**：多个工具 + 强制 `tool_choice` 时，修前 0/24、修后 24/24；加上 `#5312` Claude Desktop 的模型名、`#5409` `<function=tools NAME>`、`#5276` `<toolcall>`/`<tool-call>`、`#4832` Anthropic `tool_use_id`、`#4622` 新的 `undeclared_tool_calls` 设置。DSH 的工具调用走的就是这套 Anthropic/OpenAI 兼容面。
- **官方 Qwen3.8-Flash-Next-FP8 的 11-07 commit 对我们没有可操作内容**：只加了 `{% generation %}` 响应模板标记，渲染结果与上一版逐字相同（见 §7）；我们合并模板里的 `xhigh`/`low` 指令文本、`<think>` 约定与官方同款。

## 1. 0.2.12 与本机相关的变更

| 变更 | 本机 | 说明 |
|---|---|---|
| tuned DFlash2 drafter（decode +2.4~5.0% 的实测来源） | 不适用 | 自动启用条件要求「every expert on the card」；我们 4 slots + 流式专家 ⇒ MTP |
| 工具调用 7 条修复 | **命中** | `#4828` 强制 `tool_choice`、`#5312`、`#5409`、`#5276`、`#5281`、`#4832`、`#4622` |
| 引擎 `#5108` 行表越界读取 fault 长 CUDA prompt | **命中** | 我们经常发 6K–13K（以及 193K）token 的 prompt，属长 prompt 路径 |
| 引擎 `#5096`/`#4844` copy 顺序、丢失 adaptive swap | **命中** | 4 slot 的 checkpoint/专家分层都会走 copy 与 adaptive tier |
| `#5158` 专家池保留一个 slot 窗口（12/20/30 GiB RAM 切分启动） | 不适用 | 那是低内存启动的路径；我们 247 GiB 内存 |
| `#4972` 全部专家在卡上时 file tier 默认关 | 不适用 | 我们是流式专家，file tier 仍在用 |
| 12 GB 卡二次启动检查顺手 `STRATA_FILE_TIER_HONEST=0`、`.bad` 下载保留、Mac/Strix/R9700 默认值 | 不适用 | 与单卡 48 GB 无关 |
| 新增 `strata-dsv41-*` DeepSeek V4.1 引擎家族 | 不适用 | 另发的 445 MB 引擎，我们跑 Qwen |
| Opt-in（默认关）：`STRATA_PF_FUSED_Q50`、`STRATA_IQ4_W4A8`、`STRATA_SELECT_TWO_PASS`、`STRATA_ADAPT_FETCH`、`STRATA_ADAPT_OVERLAP`、`STRATA_PF_PEER_CPU`、`STRATA_VNNI_TIER`、`undeclared_tool_calls:"call"` | 不动 | 都是未在我们的精度/机器上验证过的实验开关 |

上游给的与 0.2.11 的对比（同一台 RTX PRO 5000 + IQ3_S）：prompt 8K 级 **×1.002**、output **+2.4%**（agent −1.5%、chat −1.0%），并且这一列**包含新 drafter 的选择**——我们拿不到这部分。

## 2. 资产、源码树与校验

| 项 | 值 |
|---|---|
| engine 包 | `strata-engine-0.1.58-linux-x86_64-cuda-cuda13.tar.gz`，**723,463,241 B** |
| sha256 | **`bbbfa8260125b94591cae30747cb0e4a9cee26d9cb0b2e284e33000c91509d54`**（与 `SHA256SUMS.txt` 一致） |
| `BUILD.json` | `{"version":"0.1.58","source":"f-0212-rc 8a42a9457db6","archs":[75,86,89,120],"vision_archs":"75;86;89;120","ptx":true,"cuda":"13.0","vision":"cuda13.0"}` |
| 二进制 | `strata` 315,960,672 B；`strata-vision` 124,767,240 B（vision 与 0.1.57 同尺寸） |
| 源码树 | tag v0.2.12：6,431 项 / 5,816 blob；四目录计数 serve 198、strata_control 81、strata_cli 38、tools 420；过滤 fixtures/测试/pyc 后 **503 文件** |
| `MIN_ENGINE` | 仍 `(0, 1, 46)`（`setup.py@v0.2.12:271`，服务端不校验） |

下载踩的两个坑（已修）：

1. **`nohup … &` 起的下载器会随启动它的父 shell 被杀**（第一次只下了 26 个分片 / 50 MB，`dl.log` 都没写成）。改用受管后台 job 重跑。
2. **两个下载器同时往同一组分片文件追加**，13 个分片被写成 4.2–6.1 MiB（> 4 MiB 理论值）。虽然 `dl.py` 会把超长分片截断，但内容可能是「重复前缀 + 截断尾部」，sha 必然不符。**清空 chunks、单进程重下**，424 s 完成，官方 sha 与拼接结果一致。

`serve/` 相对 0.2.11 只改了 `server.py` 与 `frontend.py`（新增 `nonblocking_out.py` 是 0.2.11 就有的），无新增第三方 import，`Strata/.venv` 直接可用。

## 3. 切替与验收

- **切替**：10:49:08 `stop_strata.sh`（9 s，显存 9 MiB）→ `systemd-run --user --unit=strata-prod --collect --property=RemainAfterExit=yes … run_strata_0212.sh` → 引擎 10:49:20 `engine started` → **10:49:56 `/health` loaded=true，总停机 48 s**。
- **`test_strata_0212.sh`：PASS=24 FAIL=0**（/health、/props、普通对话、自动模式 prompt_tokens 2847、视觉 32 tok、11 个环境变量逐个进程命中、drafter=mtp、FP16 cache 行、显存）。
- `/props`：`build_info "Strata 0.1.58"`、`template_chars 243742`（未变）。
- `/metrics.engine`：`engine 0.1.58 | drafting mtp | slots 4 | ctx 262144 | expert_slots 17283 | expert_cache_mib 33603 | vram_free_mib 2483 | pcie_frac "0.15"`。

## 4. 同协议实测（0.1.57 → 0.1.58）

| 用例 | 0.1.57（切前，10:43–10:44） | 0.1.58（切后，干净复测） | 差 |
|---|---|---|---|
| 6.25K tok prompt 预填充 | 5,206 t/s（6,252/6,253/6,253 tok） | **5,177 t/s**（6,256/6,254/6,254 tok） | −0.6% |
| 12.7K tok prompt 预填充 | 5,554 t/s（12,733 tok） | **5,526 t/s**（12,733 tok） | −0.5% |
| 4 并发聚合（512 tok × 3 轮） | 190.0 / 217.8 / 220.6 t/s | **207.9 / 214.8 / 214.8 t/s** | 噪声内 |
| 4 并发每路中位 | 50.7 / 57.1 / 57.9 | 54.9 / 56.5 / 57.3 | — |

**一次误报要记下来**：切完后的第一轮 4 并发只测到 116–126 t/s（墙钟 16.3–17.6 s，每路中位反而正常）。查日志发现那几分钟里引擎在同时服务**真实的长上下文请求**：

```
strata serve: prompt 193342 tokens = 192009 reused + 1333 read in 1695 ms
strata serve: prompt 193681 tokens = 193337 reused + 344 read in 1044 ms
strata conversation cache: parked 193690 tokens … bytes=3320640400 evictions=73
```

193K token 的会话会停放 2.7–3.3 GB 的 checkpoint、还触发 eviction，4 路基准和它抢 slot，聚合吞吐自然腰斩。**干净重启后（无长请求在飞）复测即回到 207–215 t/s**，与 0.1.57 的 218–221 同一水平。教训：**在有人用的生产上测并发，必须先确认有没有长请求在跑**，否则会把业务流量当成引擎回退。

## 5. FP16 weight cache（仍然 0 MiB）

规则与 0.2.11 相同：`cache = max(0, 启动时空闲显存 − 3072 MiB)`。

```
strata prefill: FP16 weight cache 0 MiB (the card keeps 3072 MiB free;
                STRATA_PF_W16_CACHE_MIB=0 turns it off)
```

48 GB 卡被 `--vram-reserve-mib 2560` + KV + 33.6 GB 专家缓存占满，启动后只剩 ~2,900 MiB，**依旧没有 3 GiB 可让**。上游仍未在 sm_89 上测过它。

## 6. 文件清单与回滚

> ⚠️ 本节写于当天下午；当晚按用户要求删除了 IQ3_S、旧 pack 与 0.2.2–0.2.11 旧树，
> 因此文中的 `run_strata_0211.sh` 等回滚脚本已不存在。当前回滚口径见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) §12。

新文件：

- `strata/Strata-0.2.12/`（serve/ + strata_control/ + tools/ + strata_cli/ 503 文件逐一校验；engine/ 来自官方包）
- `strata/strata-davidau-iq3_s-0212.json`（相对 0211 只改 `exe`/`cwd`/`log`/`vision.exe` 四个字段，32 个 args 逐字未动）
- `strata/run_strata_0212.sh`（生产指针 `run_strata.sh` 已指向它）
- `strata/test_strata_0212.sh`（24 项验收）
- `/tmp/s0212/{dl.py,fetch.py,switch.sh}`（分片下载 / tree 抓取校验 / 带失败回滚的切替）

回滚：`bash stop_strata.sh` → `systemd-run … run_strata_0211.sh`（0.2.11 + 0.1.57）。再往前 `0210 / 029 / 027 / 022` 都保留，旧树不删。`strata components rollback` 也能把 engine 单退回 0.1.57（Control 仍 0.1.58）。

## 7. 官方 Qwen3.8-Flash-Next-FP8 的 11-07 commit 审阅

commit [`7fb48d7`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/commit/7fb48d708ca6cd8eeed8f4db0433be18af96984f)（2026-11-07T04:42:53Z）标题 *Add a response template and generation tags (#6)*。与上一版（`236dfdf2`）的模板 diff **只有三处结构性改动**：

```diff
+        {{- '<|im_start|>' + message.role + '\n' }}
+        {%- generation %}
-            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + … }}
+            {{- '<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
-        {{- '<|im_end|>\n' }}
+        {{- '<|im_end|>' }}
+        {%- endgeneration %}
+        {{- '\n' }}
```

也就是把 `<|im_start|>assistant\n` 提到分支外、给 assistant 段包上 `{% generation %}` / `{% endgeneration %}`。**拼接结果与上一版逐字节相同**，渲染 prompt 不变；`{% generation %}` 是 HF tokenizer 用来标「哪一段是生成内容」（`return_assistant_tokens_mask`）的训练侧标记，**对推理零影响**。

对照我们的模板：

- 官方 `generation_config.json` 的采样默认仍是 `temperature 1.0 / top_k 20 / top_p 0.95`；`chat_template.jinja` 9,004 B。
- 我们的合并模板 243,742 chars 里，`xhigh`/`low` 的两段指令文本与官方**逐字一致**（`Reasoning effort is set to xhigh. Please think carefully through the task, …` / `…low. Keep your thinking brief and focused, …`），同样使用 `<think>` / `</think>`、`enable_thinking`、`reasoning_effort`。
- 差别在 DavidAU 的 27+27 模式体、8 个 xhigh 组合、自动选模式 v3 —— 这些官方模板没有，也不需要。

**结论：不需要动模板。** 若以后要做训练/蒸馏、需要 assistant mask，再把 `{% generation %}` 标记补进我们的模板即可（推理侧没有收益，反而多一个依赖 Jinja 扩展的风险）。FP8 权重仓库本身（131 个 shard）与我们无关：48 GB 单卡跑不了 FP8 的 125B，我们用的是 IQ3_S。

## 8. 不适用项（避免下次重复评估）

- tuned DFlash2（需专家全在卡上 / Mac / Strix Halo / Arc / 12 GB 特定量化）
- 12 GB / 20–24 GB 显存的分层裁剪与二次启动检查
- Strix Halo、Radeon gfx12、Intel Arc、Mac M5、DGX Spark（linux-aarch64）
- `--peer-device` 双卡路径（known problem：首请求 ~4% fault，0.2.11 也有；全卡切分引擎不受影响）
- DeepSeek V4.1 引擎家族（`strata-dsv41-*`，445 MB，另发）
- chats-as-files / IME / 菜单等桌面侧改动

## 9. 已知问题（上游，与本机无关）

- `--peer-device` 双卡首请求约 4% fresh start 会 fault（全卡切分引擎不受影响）
- Radeon gfx1201 手选 DFlash2 会卡 prompt；DGX Spark 首个长 prompt 慢
- 仍有一批未修完的工具调用 issue（不影响我们已验的 7 条修复）
