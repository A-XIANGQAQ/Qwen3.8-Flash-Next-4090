# 完整复现 (Full Reproduction)

从一台干净的 Linux 机器复现 **v5.3：Strata 0.2.2 服务端 + engine 0.1.47 + DavidAU IQ3_S**。
只有这一条路线；v1–v4 的运行时已删除，历史见 [HISTORY.md](HISTORY.md)。

## 1. 硬件要求

- **单 RTX 4090 48GB（sm_89）**——已验证；原版 24GB 不可行（显存占用约 46.3GB）
- CPU：≥32 物理核（38 核实测；MoE experts 的调度/拷贝吃 CPU，但不是主要算力）
- RAM：**247GB 实测**；官方没有给下限，按「权重 + 专家驻留 + KV 流式」预留，建议 ≥192GB，且不要用 swap 顶容量
  - 权重 83.86GB（2 shards，3.5 bpw）+ 专家驻留 + `--kv-resident 32768` 的 KV 宿主内存
- 磁盘：模型 84GB + 引擎/服务端 ~1GB + venv ~2GB

## 2. 软件前提

- Linux x86_64；NVIDIA driver ≥580.65（实测 610.57.04 / CUDA 13.2）
- Python **3.12**（系统 3.14 与部分轮子不齐；用 venv）
- `gh`（下载 release）、`modelscope`（下模型）、`curl`/`tar`
- 本仓库的基准工具另需 `bench/requirements.txt`（vLLM 脚本 + transformers）

## 3. 服务端与引擎

0.2.0 起上游发 **Linux 预编译引擎**（0.1.47 的 `BUILD.json` arch 含 `75/86/89/120`）。**不要跑 `setup.sh`**：
它会重导模板、重写 run config，0.1.44+ 还可能因 #2902 删模型分片。

```bash
export https_proxy=http://127.0.0.1:7890        # 本机访问 GitHub 需要

# 服务端源码（tag v0.2.2，不需要编译）
mkdir -p strata/Strata-0.2.2
curl -L https://github.com/Niko1221/Strata/archive/refs/tags/v0.2.2.tar.gz \
  | tar -xz -C strata/Strata-0.2.2 --strip-components=1

# 预编译引擎（CUDA 13；sha256 必须与签名 feed 一致）
gh release download v0.2.2 --repo Niko1221/Strata \
  -p 'strata-engine-0.1.47-linux-x86_64-cuda-cuda13.tar.gz' -D /tmp
sha256sum /tmp/strata-engine-0.1.47-linux-x86_64-cuda-cuda13.tar.gz
#   45a6df98aaf4b6b7c05087a4ab428be8df478ddcc07193a13cbaab312b85895e
mkdir -p strata/Strata-0.2.2/engine
tar -xzf /tmp/strata-engine-0.1.47-linux-x86_64-cuda-cuda13.tar.gz \
  -C strata/Strata-0.2.2/engine
strata/Strata-0.2.2/engine/strata --version       # 期望：strata 0.1.47
```

引擎包自带 cuBLAS/cuDART（rpath），**不需要** `LD_LIBRARY_PATH`。Python 依赖：

```bash
python3.12 -m venv strata/venv                  # 路径可自定
strata/venv/bin/pip install -r strata/Strata-0.2.2/requirements.txt
```

## 4. 模型（IQ3_S，2 shards）

```bash
modelscope download --model \
  DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF \
  --include '*IQ3_S*' --local_dir /path/to/DavidAU-abliterated-IQ3_S
```

- 约 **83.86 GB / 2 shards**，3.5 bpw，**自带 abliteration**（不需要 ESP）
- ModelScope 优先；HF 在本机实测只有 ~234 B/s
- 两片都要下完再打包；完成判据是 `.done.json`（预分配文件缺块也显示满大小）

## 5. 生成 Strata pack

pack 是「GGUF 权重 + tokenizer + 专家 profile」的布局，由 Strata 自带的打包工具生成：

```bash
# 先确认 PLE 张量在哪一片：iq_pack 把 per_layer_token_embd.weight 留在 GGUF 里，
# 引擎靠 --ple-gguf 去该分片读；它不一定在 shard1。
python3 scripts/find_ple_shard.py --strata strata/Strata-0.2.2 \
  /path/to/...-00001-of-00002.gguf /path/to/...-00002-of-00002.gguf

strata/venv/bin/python strata/Strata-0.2.2/tools/iq_pack.py \
  --gguf /path/to/...-00001-of-00002.gguf \
  --out  /path/to/Strata-data/packs/davidau-abliterated-iq3_s \
  --compat-bf16
```

`--compat-bf16` 的转换量取决于该量化；DavidAU IQ3_S 实测 **0 tensors / 0.00 GiB**（本来就是 BF16/F16），
`0` 不是漏做。打包会同时导出 tokenizer 与**模型自带模板**到 `<pack>/tokenizer/`。

## 6. 合并模板

生产装的是 [合并模板](../patches/08_qwen38_froggeric_v225_with_dau_modes.jinja)（249,545 B）：
froggeric v22.5 + 模型自带 29 段模式正文 + 8 个 xhigh 组合键 + 自动选模式 v3。

```bash
PACK=/path/to/Strata-data/packs/davidau-abliterated-iq3_s/tokenizer
cp "$PACK/chat_template.jinja" "$PACK/chat_template.jinja.bak-orig"     # 原装约 221,835 B
cp patches/08_qwen38_froggeric_v225_with_dau_modes.jinja \
   "$PACK/chat_template.jinja"
wc -c "$PACK/chat_template.jinja"                                       # 249545
```

**不要挂「模板守卫」**：旧启动器会把模板强制写成纯 froggeric（28,234 B），反而冲掉合并模板。
`setup.sh` 每次都会重导模板，重跑 setup 后要重装一次。

## 7. 启动

配置是一个 JSON（`exe/args/cwd/tokenizer/model_name/log/parallel/vision`），核心 `args`：

```
--pack <pack> --native <shard1> --ple-gguf <含 PLE 的分片>
--expert-profile <pack>/expert-profile.bin --expert-cache auto
--prefill auto:32768 --spec 4 --spec-min-p 0.5 --mtp <Strata-data>/mtp/rt
--max-context 262144 --kv k8v4 --kv-resident 32768
--resident-budget-gib 56 --conversation-cache-mib 8192
--batch-mtp --vision --vram-reserve-mib 2560
```

```bash
# ⛔ 必须在你自己的终端里跑（或一次性 systemd 用户单元）。
#    从 agent/CI 的工具调用里起，进程会落在 dsh-subprocess-*.scope 的 cgroup，
#    dsh-web 重启时被整组 SIGTERM；症状是日志只有 stopping、没有 traceback。
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &

# 就绪判据：日志行 ready:（实测 36 秒）
#   ready: http://127.0.0.1:8000/v1  (OpenAI: /v1/chat/completions,
#                                     Anthropic: /v1/messages, context 262144 tokens)
curl -s 127.0.0.1:8000/health
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["build_info"], len(d["chat_template"]))'
#   期望：Strata 0.1.47 243742
```

⚠️ 起之前确认显存已空：`nvidia-smi --query-gpu=memory.used --format=csv,noheader` 轮询到 **<1000 MiB**，
否则 `cudaMalloc` OOM（[TROUBLESHOOTING #29](TROUBLESHOOTING.md)）。
⚠️ 不要用 `./setup.sh` 无参起服务——它会自己拉起模型并绕过模板。

## 8. 成功判据（v5.3）

- [ ] `pgrep -x strata` 有结果；`:8000` 在听；`/health.loaded == true`
- [ ] `/props.build_info == "Strata 0.1.47"`，`chat_template` 长度 **243742**
- [ ] 显存约 **46,303 / 49,140 MiB**（含 vision；`--vram-reserve-mib 2560`）
- [ ] 一条中文请求正文可读、无乱码
- [ ] 自动选模式：`python3 patches/08_auto_check.py --set dev+val` → dev 95.9% / val 90.0%
- [ ] 基准冒烟：`bench/run_matrix.py --only P8K`（见 [BENCHMARK.md](BENCHMARK.md)）
- [ ] 0.2.2 服务端 + 0.1.47 引擎是已验证配对（见 [TROUBLESHOOTING #39](TROUBLESHOOTING.md)）

## 不在默认范围

- 原版 24GB 4090、FP8/BF16 权重、1M YaRN、多卡
- 连续多日稳定性与 >4 路密集并发的扩展性（见 [BENCHMARK.md §6](BENCHMARK.md)）
- 公网直接暴露（建议 loopback + 反向代理，`proxy_read_timeout` ≥900s）
