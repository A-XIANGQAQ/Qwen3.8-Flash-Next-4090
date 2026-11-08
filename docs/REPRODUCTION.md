# 完整复现 (Full Reproduction)

从一台干净的 Linux 机器复现 **v5.9：Strata 0.2.12 服务端 + engine 0.1.58 + unsloth UD-Q4_K_XL（huihui v2 破限）**。
只有这一条路线；v1–v4 的运行时已删除，历史见 [HISTORY.md](HISTORY.md)。

## 1. 硬件要求

- **单 RTX 4090 48GB（sm_89）**——已验证；原版 24GB 不可行（显存占用约 46.3GB）
- CPU：≥32 物理核（38 核实测；MoE experts 的调度/拷贝吃 CPU，但不是主要算力）
- RAM：**247GB 实测**；官方没有给下限，按「权重 + 专家驻留 + KV 流式」预留，建议 ≥192GB，且不要用 swap 顶容量
  - 权重 111.33GB（4 shards，4-bit）+ 专家驻留（72 GiB 全驻 RAM）+ `--kv-resident 32768` 的 KV 宿主内存
- 磁盘：模型 112GB + 补丁输出 82GB（只重写 shard2/3；shard1/4 用软链指回原文件）+ 引擎/服务端 ~1GB + venv ~2GB

## 2. 软件前提

- Linux x86_64；NVIDIA driver ≥580.65（实测 610.57.04 / CUDA 13.2）
- Python **3.12**（系统 3.14 与部分轮子不齐；用 venv）
- `gh`（下载 release）、`modelscope`（下模型）、`curl`/`tar`
- 本仓库的基准工具另需 `bench/requirements.txt`（vLLM 脚本 + transformers）

## 3. 服务端与引擎

0.2.0 起上游发 **Linux 预编译引擎**（0.1.58 的 `BUILD.json` arch 含 `75/86/89/120`，vision 同）。**不要跑 `setup.sh`**：
它会重导模板、重写 run config，0.1.44+ 还可能因 #2902 删模型分片。

```bash
export https_proxy=http://127.0.0.1:7890        # 本机访问 GitHub 需要

# 服务端源码（tag v0.2.12，不需要编译）
mkdir -p strata/Strata-0.2.12
curl -L https://github.com/Niko1221/Strata/archive/refs/tags/v0.2.12.tar.gz \
  | tar -xz -C strata/Strata-0.2.12 --strip-components=1

# 预编译引擎（CUDA 13；sha256 必须与签名 feed 一致）
gh release download v0.2.12 --repo Niko1221/Strata \
  -p 'strata-engine-0.1.58-linux-x86_64-cuda-cuda13.tar.gz' -D /tmp
sha256sum /tmp/strata-engine-0.1.58-linux-x86_64-cuda-cuda13.tar.gz
#   bbbfa8260125b94591cae30747cb0e4a9cee26d9cb0b2e284e33000c91509d54
mkdir -p strata/Strata-0.2.12/engine
tar -xzf /tmp/strata-engine-0.1.58-linux-x86_64-cuda-cuda13.tar.gz \
  -C strata/Strata-0.2.12/engine
strata/Strata-0.2.12/engine/strata --version       # 期望：strata 0.1.58
```

引擎包自带 cuBLAS/cuDART（rpath），**不需要** `LD_LIBRARY_PATH`。Python 依赖：

```bash
python3.12 -m venv strata/venv                  # 路径可自定；下文统一用这个
strata/venv/bin/pip install -r strata/Strata-0.2.12/requirements.txt

# 打补丁与 iq_pack 都要 import gguf：用任意 llama.cpp checkout 的 gguf-py
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
export STRATA_GGUF_PY=/tmp/llama.cpp/gguf-py    # 仓库脚本与 iq_pack 都认这个变量

# expert-profile.bin：生产 config 默认指向 strata/Strata/data/expert-profile.bin
mkdir -p strata/Strata/data
curl -L https://raw.githubusercontent.com/Niko1221/Strata/v0.2.12/data/expert-profile.bin \
  -o strata/Strata/data/expert-profile.bin      # 196,632 B；也可改 config 的 --expert-profile
```

`strata/Strata/` 不是安装目录，只是沿用本机的路径布局（装 venv、data/、旧 gguf-py 的地方）；
`strata/Strata-0.2.12/` 才是服务端源码树。

## 4. 模型（unsloth UD-Q4_K_XL v3.1，4 shards）

```bash
# 官方 v3.1 四片 + mmproj（ModelScope，实测 11–12 MB/s，约 3 小时）
modelscope download --model unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'UD-Q4_K_XL/*' --include 'mmproj-BF16.gguf' \
  --local_dir /path/to/unsloth-UD-Q4_K_XL

# 破限补丁（HF，约 3.56 GB；本机 0.4–0.5 MB/s，约 2.4 小时）
huggingface-cli download huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF \
  --include 'patch-unsloth/**' --local-dir /path/to/patch-huihui-v2
```

- `--local_dir` 下 modelscope 会再建 `UD-Q4_K_XL/` 子目录；`scripts/patch_and_pack_q4kxl.sh` 两种布局都认
  （平铺或嵌套），手写命令时注意路径。
- **111,334,654,778 B / 4 shards**：shard1 只有 metadata+tokenizer（0 张量），张量在 shard2（层 0–28）
  与 shard3（层 28–47），shard4 是 28.8 GB 的 IQ4_NL PLE 表。逐片 sha256 见
  [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) §2.1。
- 别下 huihui 仓库里 27 天前的 `unsloth-UD-Q4_K_XL/`：旧切分（49.86/49.38/12.09 GB），补丁覆盖对不上。
- 补丁 `patch-unsloth-all.gguf` = 3,561,572,544 B + `.gguf.key`（64 B）；`.parts/` 写完才会出现正式文件，跑前核尺寸。

### 4b. 打补丁（只重写 shard2 / shard3）

```bash
export PY=strata/venv/bin/python                 # 装了 requirements 的 venv
export PYTHONPATH=${STRATA_GGUF_PY:?先按 §3 设 STRATA_GGUF_PY}   # patch 工具 import gguf
redact() { sed -E 's/(Using key \([0-9]+ bytes\): ).*/\1<redacted>/'; }

mkdir -p pin2 pin3
ln -s /path/to/...-00002-of-00004.gguf pin2/
ln -s /path/to/...-00003-of-00004.gguf pin3/
"$PY" patch-unsloth-Qwen3.8-Flash-Next-dec.py \
  --input-path pin2 --patch-path patch-unsloth-all --output-path <abl>/ \
  --start-layer 0 --end-layer 48 --ablate-experts 1 2>&1 | redact
# shard3 必须再来一次：只跑 pin2 的话 shard3 从未消融，pack 照样能建、服务照样能起，破限悄悄丢失
"$PY" patch-unsloth-Qwen3.8-Flash-Next-dec.py \
  --input-path pin3 --patch-path patch-unsloth-all --output-path <abl>/ \
  --start-layer 0 --end-layer 48 --ablate-experts 1 2>&1 | redact
```

⚠️ 工具会打印 `Using key (64 bytes): <完整 key>`——**必须像上面的 `redact` 一样脱敏**（或直接重定向到文件），
别把 key 留在终端记录/日志里。仓库脚本已内置这条 sed。

期望（工具不会打印 `mismatch 0` 这种字符串，别按那个判）：
- 两片都出现 `Patch files (1)`（<1 就是补丁没加载，必须先停）
- shard2 `Patched tensors: 28`、`Copied tensors: 698`；shard3 `Patched tensors: 20`、`Copied tensors: 477`
- 末尾 `Note: 219/237 patch tensors were never applied` 是**正常**的（补丁里另有 BF16/Q6_K/Q5_K 变体，本量化用不到）
- 若出现 `Copied because dtype mismatch` 或 `was not found`，说明有目标张量被静默跳过，必须停下核对

可直接用仓库脚本：`bash scripts/patch_and_pack_q4kxl.sh`（内含上述断言）。
shard1 / shard4 用软链指进 `<abl>/`（实测 `lrwxrwxrwx`），不重写。

## 5. 生成 Strata pack

pack 是「GGUF 权重 + tokenizer + 专家 profile」的布局，由 Strata 自带的打包工具生成：

```bash
# PLE 表在 shard4（v3.1 布局）
strata/venv/bin/python scripts/find_ple_shard.py --strata strata/Strata-0.2.12   # 需 STRATA_GGUF_PY \
  /path/to/...-00001-of-00004.gguf /path/to/...-00002-of-00004.gguf \
  /path/to/...-00003-of-00004.gguf /path/to/...-00004-of-00004.gguf

# ⚠️ 必须用 Strata 的 venv python：系统 python3.14 缺 regex
strata/venv/bin/python strata/Strata-0.2.12/tools/iq_pack.py \
  --gguf /path/to/<abl>/...-00001-of-00004.gguf \   # 必须是打过补丁的那份；它会按同目录自动发现其余分片
  --out  /path/to/Strata-data/packs/unsloth-ud-q4kxl-ablit-l1 \
  --compat-bf16
```

**坑**：必须用装了 `requirements.txt` 的 venv python（原因与失败形态见 [TROUBLESHOOTING #44](TROUBLESHOOTING.md)）。
`iq_pack.py` 还会通过 `strata_control/_paths.gguf_py()` 找 gguf-py：优先 `STRATA_GGUF_PY`（§3 已设），
其次旧安装的 `third_party/llama.cpp/gguf-py`；都没有会直接 `sys.exit`。
本机现成的是 `strata/Strata/.venv`（旧安装遗留），等价；从零按 §3 建的是 `strata/venv`，下文用后者。
实测 UD-Q4_K_XL：`compat-bf16` 转 **195 张量 / 1.20 GiB**；`native_experts.txt` v4（49 行、n_expert 512、total 77,017,907,200 B）；
`index.txt` v3（1079 张量 / 303 native / arena 1.38 GiB）。打包会同时导出 tokenizer 与模型自带模板。

### 5b. MTP 草稿层（`--mtp` 指的 rt 目录）

仓库/模型包都不带 runtime 文件，用官方 BF16 checkpoint 里的 31 个 `mtp.*` 张量现建（HTTP range，不会下 360GB；
`STRATA_MTP_REVISION` 可覆盖默认 pin）。MTP 头**不随量化/消融变化**，所以与 IQ3_S 时代那份是同一来源。

```bash
T=strata/Strata-0.2.12/tools
strata/venv/bin/python $T/mtp_fetch.py inventory --out strata/mtp-bf16   # 只读各分片 header（几 KB/片）
strata/venv/bin/python $T/mtp_fetch.py fetch     --out strata/mtp-bf16   # 可续传；只取 mtp.* 的字节区间
strata/venv/bin/python $T/mtp_fetch.py verify    --out strata/mtp-bf16   # exit 3 = 有张量缺失/损坏
strata/venv/bin/python $T/mtp_pack.py --src strata/mtp-bf16 --experts q2_0 \
  --out strata/mtp-bf16/mtp-q2_0.gguf
strata/venv/bin/python $T/mtp_rt.py --gguf strata/mtp-bf16/mtp-q2_0.gguf \
  --out strata/Strata-data/mtp/rt
```

产物：`dense.bin` 116,099,072 B、`experts.bin` 707,788,800 B、`draft_vocab.bin` 425,196 B、`dense.txt`。
`--mtp` 指向这个目录；没有它服务起不来（`--batch-mtp` 会把草稿层加载行写进日志，验收第 9 节检查）。

## 6. 模板（froggeric v22.5）

```bash
PACK=/path/to/Strata-data/packs/unsloth-ud-q4kxl-ablit-l1/tokenizer
cp "$PACK/chat_template.jinja" "$PACK/chat_template.jinja.orig"      # 原装 9,993 B
cp patches/froggeric_v22.5.jinja "$PACK/chat_template.jinja"
wc -c "$PACK/chat_template.jinja"                                    # 28234 字节（28,226 字符）
```

v5.9 起不再套 DavidAU 的 29 段模式合并模板（249,545 B）；`/props.chat_template` 的 `len()` 是 **28226**。
`setup.sh` 每次都会重导模板，重跑 setup 后要重装一次。

## 7. 启动

配置是一个 JSON（`exe/args/cwd/tokenizer/model_name/log/parallel/vision`），核心 `args`：

```
--pack <pack> --native <shard1> --ple-gguf <shard4>
--expert-profile <Strata>/data/expert-profile.bin --expert-cache auto
--prefill auto:32768 --spec 4 --spec-min-p 0.5 --mtp <Strata-data>/mtp/rt
--max-context 262144 --kv k8v4 --kv-resident 32768
--conversation-cache-mib 8192
--batch-mtp --vision --vram-reserve-mib 2560
```

共 **30 项**（v5.8 的 32 项去掉 `--resident-budget-gib 56`，专家全驻 RAM）；config 的 `env` 加
`"STRATA_ROUTE_TAIL_SKIP": "0"`。完整 args 在 `strata/strata-unsloth-q4kxl-ablit-l1.json` 的 `args` 数组；
`run_strata_q4l1.sh` 只负责选解释器（`STRATA_PY` → `strata/venv` → `strata/Strata/.venv`）并 exec `serve/server.py`。
config 里是绝对路径：换根目录时要改 `--pack` / `--native` / `--ple-gguf` / `--expert-profile` / `--mtp` 与
`tokenizer` / `vision` 这几项。

```bash
# ⛔ 必须在你自己的终端里跑（或一次性 systemd 用户单元）。
#    从 agent/CI 的工具调用里起，进程会落在 dsh-subprocess-*.scope 的 cgroup，
#    dsh-web 重启时被整组 SIGTERM；症状是日志只有 stopping、没有 traceback。
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &

# 就绪判据：日志行 ready:（页缓存命中 53 秒；冷 106 秒）
#   ready: http://127.0.0.1:8000/v1  (OpenAI: /v1/chat/completions,
#                                     Anthropic: /v1/messages, context 262144 tokens)
curl -s 127.0.0.1:8000/health
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["build_info"], len(d["chat_template"]))'
#   期望：Strata 0.1.58 28226
```

⚠️ 起之前确认显存已空：`nvidia-smi --query-gpu=memory.used --format=csv,noheader` 轮询到 **<1000 MiB**，
否则 `cudaMalloc` OOM（[TROUBLESHOOTING #29](TROUBLESHOOTING.md)）。
⚠️ 不要用 `./setup.sh` 无参起服务——它会自己拉起模型并绕过模板。

## 8. 成功判据（v5.9）

验收脚本：`bash scripts/accept_q4kxl.sh`（只读探测，**31 项断言**；本机实测 PASS=31 FAIL=0，失败时 exit 1）。
覆盖：`/health`（loaded/ctx/images/model）、`/props`（build_info + 模板 28226）、`/metrics.engine`
（engine / drafting / slots / expert_slots / pcie_frac / arena）、一条对话、`reasoning_effort` 通路、视觉、
11 个并行栈 env、**本次启动**日志里的 PLE/`--ple-gguf`、drafter=MTP。

脚本不覆盖、需手工确认：

- [ ] `pgrep -x strata` 有结果；`:8000` 在听
- [ ] 启动后显存约 **46,145 / 49,140 MiB**（脚本只打印 `nvidia-smi`，不断言；跑过长上下文/视觉后到 ~46,570 属缓存增长）
- [ ] 视觉回答的颜色正确（脚本只断言 `completion_tokens ≥ 1`，不判色）
- [ ] 破限：`python3 scripts/probe_abliteration.py` → 8/8 不拒答（只记 refused 与长度，不存正文；有异常或拒答时 exit 1）
- [ ] 0.2.12 服务端 + 0.1.58 引擎是已验证配对（见 [TROUBLESHOOTING #39](TROUBLESHOOTING.md)）

## 不在默认范围

- 原版 24GB 4090、FP8/BF16 权重、1M YaRN、多卡
- 连续多日稳定性与 >4 路密集并发的扩展性（见 [BENCHMARK.md §6](BENCHMARK.md)）
- 公网直接暴露（建议 loopback + 反向代理，`proxy_read_timeout` ≥900s）
