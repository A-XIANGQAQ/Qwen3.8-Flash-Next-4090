# 完整复现 (Full Reproduction)

## 1. 硬件要求 (Hardware Requirements)

已验证主机（两代引擎共用同一台）：

- 单 RTX 4090 **48GB**，SM89（原版 24GB 未验证——两代栈显存占用都在 44GB 以上，远超 24GB）
- CPU：建议 ≥32 物理核（experts 在 CPU 上算，核越多越快）
- RAM：本站 **247GB**；**v5 的具体下限未测**，但两代栈都按"整模型驻内存"设计，无 swap 可用
  - **v5（Strata）**：模型 111.33 GB + 专家全部驻宿主内存（`--resident-budget-gib 71`）
    + **KV 流式到宿主内存**。实测常驻：`free` 报 68 GiB used / 198 GiB buff-cache（含 mmap 的 GGUF），引擎 RSS **129.5 GiB**
  - **v4（FreeToken）**：模型 126GiB resident + 专家 host banks 63.3G；进程峰值 ~200GB
- 磁盘：模型 **111GB（v5）/ 126GB（v4）** + venv 或构建产物 ~15GB
- 不要启用 swap 作为容量替代

## 2. 软件前提 (Software Prerequisites)

- Linux；NVIDIA driver 可运行 CUDA 13 runtime（实测 610.x）
- **uv**（v4 建 venv；v5 的 `setup.sh` 由 `STRATA_UV=1` 驱动，同样需要）
- **nvcc 需在 PATH**（本机 `/usr/local/cuda/bin`）——v4 的 Triton JIT 与 **v5 的自建编译**都要
- ModelScope CLI（两代模型都从这里下载）
- v4 路线：host 无需预装完整 CUDA toolkit（venv 会拉入 `cuda-toolkit` pip 包）；
  **v5 路线必须自建**，需要可用的 C++ 工具链（cmake / gcc）

## 3. 环境安装 (Environment Install)

**当前路线（v5）= Strata，见 §3C（先看这个）**。v4（FreeToken）/ v3 / v1（sglang）的安装步骤保留为历史——
对应运行时与 venv 都已删除，要复活得先重装引擎。

> ⚠️ 两代引擎各占本卡 90%+ 显存，**同一时刻只能跑一个**（装在同一台机器上互不冲突，但起服务前必须停掉另一个，
> 且**等显存降到 <1000 MiB**，见 [TROUBLESHOOTING #29](TROUBLESHOOTING.md)）。

### 3A. v4：FreeToken + 三个本地补丁（历史，运行时已删除）

```bash
uv venv ~/ft-venv --python 3.12
uv pip install --python ~/ft-venv "freetoken[accel]"

# 打三个本地补丁（幂等，可重复执行；升级 freetoken 后必须重跑）
bash scripts/archive/install_freetoken_patches.sh ~/ft-venv/lib/python3.12/site-packages

# 验证
~/ft-venv/bin/ft --help
```

`accel` extra 拉入 flashinfer-python[cu13] 0.6.18.post1 与 sglang-kernel 0.4.5。
版本硬约束（由 freetoken 0.1.3 的元数据决定）：`torch>=2.11,<2.12`、`transformers>=5.16,<5.17`、`triton==3.6.0`。
完整 102 包快照见 [`freeze-freetoken-0.1.3.txt`](archive/freeze-freetoken-0.1.3.txt)。

> 三个补丁**均未上游化**，且都打在 site-packages 里——`uv pip install -U` 会静默覆盖它们。
> 启动脚本 `scripts/archive/start_freetoken.sh` 内置自检，缺补丁会直接拒绝启动。

### 3B. 历史路线（sglang，保留可复现）

<details>
<summary>展开 v3 / v1 的 sglang 安装步骤</summary>

**v3（官方 v0.5.20 基线）= 自建 wheel：**

```bash
conda create -n lsglang-next python=3.12 && conda activate lsglang-next
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install build setuptools-rust setuptools-scm      # 构建期依赖

bash scripts/archive/build_upstream_flashnext.sh              # 取 v0.5.20 + 打 3 个补丁 + 构建

pip install --no-deps <构建产物>/lsglang-1.6.0+flashnext.v0520-py3-none-any.whl
pip install "lk_moe==2.4.1" "sglang-kernel==0.4.7" "tilelang==0.1.11" "flashinfer-python[cu13]==0.6.18"
pip install ./flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl   # SM89 必需
```

> `sglang-kernel==0.4.7` 是 v0.5.20 的启动强校验；`tilelang` 必须 0.1.11（见 TROUBLESHOOTING #15）。

**v1（Lsglang 1.4.13）= 直接装 release wheel：**

```bash
bash scripts/archive/install.sh
```

脚本下载两个 wheel（GitHub release `lsglang-v1.4.12` tag）、建 conda env、按序安装
（torch → flash_attn → lsglang）。`flash_attn-2.8.4+pr2751` 是 **SM89 必需**——
head_dim=256 的 128×64 tile 超 4090 shared memory，PR2751 改 128×32。

</details>

### 3C. v5：Strata（Linux 必须自建）

Strata 的 release **只有 Windows 二进制**（`strata-windows-x64*.zip`），Linux 没有预编译包。
实测环境：CUDA 13.2 / cmake 4.2 / gcc 15.2 / 76 核。`STRATA_UV=1` 强制 Python 3.12
（系统 python3 是 3.14，NVIDIA wheel 不齐）。

```bash
# 1. 取源码（gh CLI）
gh repo clone Niko1221/Strata && cd Strata

# 2. 预解压 pinned llama.cpp（否则 setup 会去 GitHub 再拉一遍）
#    commit 见 setup.py 的 LLAMA_CPP_COMMIT = 3cf03257f219afbe7334045ff7c6a06ac68c627d
unzip llama.cpp-3cf0325.zip -d third_party/ && mv third_party/llama.cpp-3cf0325 third_party/llama.cpp

# 3. ESP 白名单补丁（否则 unsloth 家族拿不到破限，见 TROUBLESHOOTING #35）
cp setup.py setup.py.orig-esp
patch -p1 < /path/to/repo/patches/07_strata_esp_family_whitelist.patch

# 4. setup：编译引擎 + 下模型 + 生成配置（--no-start 先不起服务）
STRATA_UV=1 ./setup.sh --setup --family unsloth --model UD-Q4_K_XL \
  --gguf-dir <模型目录> --source modelscope \
  --vision yes --context 262144 --kv k8v4 --parallel 4 \
  --experimental-speed-projection on \
  --host 127.0.0.1 --port 8000 --yes --no-start
```

编译在 76 核上约 **1 分钟**（`[138/138] Linking CXX executable strata`），产出 `engine/strata`（59 MB），
`BUILD.json` 记 `archs: [89]`、CUDA 13.2。

> 💡 **构建与下载可重叠**：setup 的引擎编译（step 4）在模型下载（step 5）之前
> ⇒ 下载期间先跑 setup 把引擎编好，模型到位后再跑一次补齐 step 6–7。

## 4. 模型下载 (Model Download)

### 4A. 当前（v5.2）：`DavidAU/…-GSQ-RCO-GGUF` → IQ3_S

```bash
modelscope download --model \
  DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF \
  --include '*IQ3_S*' --local_dir /path/to/DavidAU-abliterated-IQ3_S
```

- 约 **83.86 GB / 2 shards**，3.5 bpw
- **自带 abliteration**：不需要 ESP 控制向量 ⇒ 也不需要 [patch 07](../patches/07_strata_esp_family_whitelist.patch)
- ⚠️ **HF 在本机实测仅 234 B/s**，不可用——一律走 ModelScope

**视觉编码器（要读图，`--vision yes`）**：`mmproj-Qwen3.8-Flash-Next-BF16.gguf`（907,543,008 B），
来自 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`——**ModelScope 上没有这个仓库**，只能走 HF 或镜像，
配合 `tools/vision` 的 `strata-vision` 二进制。实测 HF 走代理单连接 ~220–400 KB/s，**12 路并行分块可到 4.1 MB/s**，
详见 [STRATA.md §5.3](STRATA.md)。

### 4B. 历史：`unsloth/Qwen3.8-Flash-Next-GGUF` → UD-Q4_K_XL

<details>
<summary>展开（2026-10-18 前的生产模型，111.33 GB / 4 shards）</summary>

```bash
modelscope download --model unsloth/Qwen3.8-Flash-Next-GGUF \
  --include '*UD-Q4_K_XL*' --local_dir /path/to/UD-Q4_K_XL
```

- 约 **111.33 GB / 4 shards**，字节数与 ModelScope API 逐条核对过
- 这个路线**需要 patch 07**：上游把 ESP 白名单锁在 `family in ("qwen","coder")`，unsloth 家族拿不到破限
- 换到 IQ3_S 的实测对比见 [UPGRADE §0c](UPGRADE.md)

</details>

### 4C. 历史：`RadixArk/Qwen3.8-Flash-Next-NVFP4`（v4，运行时已删除）

<details>
<summary>展开（126 GiB / 206 shards）</summary>

```bash
modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local_dir /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
```

- 约 **126 GiB / 206 shards**
- ⚠️ **revision**：TomPython 项目记录的 HF revision（`7b7192...`）在 **ModelScope 不存在**，直接下 master
- 校验：`python3 scripts/verify_model_index.py <dir> --expected-shards 206`

</details>

## 5. Chat Template：合并模板（froggeric v22.5 + 模型自带模式）

v5 生产用的**不是纯 froggeric**，而是 [合并模板](../patches/08_qwen38_froggeric_v225_with_dau_modes.jinja)
（248,855 字节）：froggeric v22.5 的渲染修复 + 模型自带 29 段模式正文 + 自动选模式 v2。
规则、词表、评测见 [TEMPLATE.md](TEMPLATE.md)。

为什么不直接用 froggeric：它的默认 effort / think 标记仍然需要，但模型自带的 29 个模式（coder / matlock /
house / story …）是这份 GGUF 的主要卖点，纯 froggeric 会让它们全部失效。

> **模板住的位置两代不同，这决定了"怎么保证它不被覆盖"**：
> - **v4（FreeToken，历史）**：`<模型目录>/chat_template.jinja`——重下 checkpoint 会覆盖。
> - **v5（Strata，当前）**：`<Strata-data>/packs/<模型>/tokenizer/chat_template.jinja`——**每次 `setup` 都从
>   GGUF 重新解出，必覆盖** ⇒ 装完 setup 要手工重装一次，判据是 `wc -c` = **248855**。

### 5A. v5 的装法（当前）

```bash
PACK=<Strata-data>/packs/<模型>/tokenizer

cp "$PACK/chat_template.jinja" "$PACK/chat_template.jinja.bak-orig"      # 备份原装（约 9,993 B）
cp qwen4090-repo/patches/08_qwen38_froggeric_v225_with_dau_modes.jinja \
   "$PACK/chat_template.jinja"

wc -c "$PACK/chat_template.jinja"                                        # 期望 248855
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["chat_template"]))'
#   期望 243052（重启后）
```

**不要再挂"chat 模板守卫"**：旧启动器里的守卫会把 pack 模板强制写成纯 froggeric（28,234 B），
反而把合并模板冲掉；本站启动器已移除该守卫（备份 `run_strata.sh.bak-undq4-20261018`）。

**对齐判据**：不命中任何模式时，渲染结果与纯 froggeric 一致（同一 prompt 2,336 token）；
命中模式时额外加上该模式正文（约 +2.1k～3.3k token）。默认 effort 是 `medium`
（froggeric 第 17 行 `_default_reasoning_effort`，**原装模板**默认 xhigh）。**重启生效**。

回滚：`cp chat_template.jinja.bak-orig chat_template.jinja`（回原装）或
`cp chat_template.jinja.v1 chat_template.jinja`（回 v1 合并规则，246,686 B）。

### 5B. v4 的装法（历史，FreeToken）

```bash
cd /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
cp chat_template.jinja chat_template.jinja.bak-orig
curl -L -o chat_template.jinja \
  "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja?download=true"
```

## 6. 启动 (Start)

**v5（当前，Strata）**——约 **50 秒**就绪，就绪判据是日志行 `ready:`：

```bash
#    ⛔ 必须在**你自己的终端**里跑：从 agent/CI 的工具调用里起，进程会落在 systemd
#       临时单元 dsh-subprocess-*.scope 里（setsid 也脱不出 cgroup），重启 dsh-web
#       时被整组 SIGTERM。判据与细节见 docs/OPERATIONS.md §1
nohup  bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
tail -f strata/strata-prod.log
# 期望：ready: http://127.0.0.1:8000/v1  (OpenAI: /v1/chat/completions,
#                                        Anthropic: /v1/messages, context 262144 tokens)
```

⚠️ **必须用启动器**，不要用 `./setup.sh` 无参起服务——它会自己拉起模型、并把 pack 里的合并模板重导成**原装**（判据 `wc -c` = 226745，见 §5A）。
⚠️ **起之前确认显存已空**：`nvidia-smi --query-gpu=memory.used --format=csv,noheader` 轮询到 **<1000 MiB**，
否则 `cudaMalloc` OOM（[TROUBLESHOOTING #29](TROUBLESHOOTING.md)）。

**v4（回滚，FreeToken）**——约 **40 秒**，就绪判据是日志行 `ready to serve`。
编辑 `scripts/archive/start_freetoken.sh` 顶部配置区（模型路径、`FT_BIN`、`SITE_PACKAGES`）后：

```bash
bash scripts/archive/start_freetoken.sh     # 内置补丁自检 + 就绪等待
```

⚠️ **不要用 `/health` 判断 v4 就绪**——它在加载完成前就返回 200（上游 #537）。

```bash
curl -s localhost:8000/v1/models                      # 模型名 + context 262144
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'
```

（v3，历史：`bash scripts/archive/start_lsglang_upstream.sh`，sglang 路线约 5 分钟。）

## 7. 成功定义 (Definition of Success)

**v5（Strata）——三层判据，缺一不可**：

- [ ] **进程到位**：`pgrep -x strata` 有结果，serve 层在听 `:8000`
- [ ] **模板是合并版**：`wc -c <pack>/tokenizer/chat_template.jinja` = **248855**（原装是 226745）
- [ ] **破限**：当前模型自带 abliteration，日志里**不该**出现 ESP 控制向量行（那是 UD-Q4_K_XL 路线的东西）
- [ ] **端到端真推理**：发一条中文请求，正文可读、无乱码
- [ ] 模型分片齐（**2 shards**，IQ3_S）、`/v1/models` 返回预期 id
- [ ] RAM 有余量、swap 为 0、无 Xid/OOM
- [ ] 显存 ~46.4 GB（2026-10-19 实测 **46,435 / 49,140 MiB**，含 vision）
- [ ] （装了图的话）发一张测试图，模型能正确描述——并确认 `pgrep -x strata-vision` 有进程
- [ ] **模式自动选择**：`python3 qwen4090-repo/patches/08_auto_check.py --set dev+val` 应得 dev 95.9% / val 90.0%
- [ ] ⚠️ **本次未验证**：长上下文（128K/256K）、工具调用结构化、多日稳定性

**v4（FreeToken）**：

- [ ] 206 shards 校验通过
- [ ] 日志出现 `ready to serve`，进程存活
- [ ] `/v1/models` 返回预期模型名
- [ ] 中文请求可读、无乱码
- [ ] RAM 有余量、swap 为 0、无 Xid/OOM
- [ ] GPUs 显存 44–46GB（未跑满 48GB）
- [ ] （可选）8K 无缓存 prefill 客户端计时——⚠️ 该项方差极大，见 [BASELINE](BASELINE.md)

## 不在默认范围 (Out of Scope)

- 单卡 24GB 原版 4090
- FP8/BF16 权重（fp8 在 sglang 路线上有已知问题）
- **MTP 的早期结论已作废**：sglang 路线下"可启动但 CPU 瓶颈无收益"、FreeToken 侧上游对 qwen4_exp 的 MTP 直接丢弃；
  **v5 的 Strata 把 MTP 当真**（`--spec 4` + `--batch-mtp`，见 [TROUBLESHOOTING #31](TROUBLESHOOTING.md)）
- 262144 以上（1M YaRN）
- 公网服务（建议 loopback + 反向代理）
