# 完整复现 (Full Reproduction)

## 1. 硬件要求 (Hardware Requirements)

已验证主机：

- 单 RTX 4090 **48GB**，SM89（原版 24GB 未验证——v4 栈实测显存占用 44–46GB，远超 24GB）
- CPU：建议 ≥32 物理核（experts 在 CPU 上算，核越多越快）
- RAM：**≥247GB**（模型 126GiB resident + 专家 host banks 63.3G；进程峰值 ~200GB）
- 磁盘：模型 126GB + venv ~15GB
- 不要启用 swap 作为容量替代

## 2. 软件前提 (Software Prerequisites)

- Linux；NVIDIA driver 可运行 CUDA 13 runtime（实测 610.x）
- **uv**（创建 venv 与装包；也可用 conda，但本项目的验证环境是 uv venv）
- **nvcc 需在 PATH**（本机 `/usr/local/cuda/bin`）——Triton 要 JIT 编译内核，缺了起不来
- ModelScope CLI（模型下载）
- host 无需预装 CUDA toolkit 的完整开发环境（venv 会拉入 `cuda-toolkit` pip 包）

## 3. 环境安装 (Environment Install)

**当前路线（v4）= FreeToken**；历史路线（v3/v1，sglang）见 §3B。

### 3A. v4：FreeToken + 三个本地补丁

```bash
uv venv ~/ft-venv --python 3.12
uv pip install --python ~/ft-venv "freetoken[accel]"

# 打三个本地补丁（幂等，可重复执行；升级 freetoken 后必须重跑）
bash scripts/install_freetoken_patches.sh ~/ft-venv/lib/python3.12/site-packages

# 验证
~/ft-venv/bin/ft --help
```

`accel` extra 拉入 flashinfer-python[cu13] 0.6.18.post1 与 sglang-kernel 0.4.5。
版本硬约束（由 freetoken 0.1.3 的元数据决定）：`torch>=2.11,<2.12`、`transformers>=5.16,<5.17`、`triton==3.6.0`。
完整 102 包快照见 [`freeze-freetoken-0.1.3.txt`](freeze-freetoken-0.1.3.txt)。

> 三个补丁**均未上游化**，且都打在 site-packages 里——`uv pip install -U` 会静默覆盖它们。
> 启动脚本 `scripts/start_freetoken.sh` 内置自检，缺补丁会直接拒绝启动。

### 3B. 历史路线（sglang，保留可复现）

<details>
<summary>展开 v3 / v1 的 sglang 安装步骤</summary>

**v3（官方 v0.5.20 基线）= 自建 wheel：**

```bash
conda create -n lsglang-next python=3.12 && conda activate lsglang-next
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install build setuptools-rust setuptools-scm      # 构建期依赖

bash scripts/build_upstream_flashnext.sh              # 取 v0.5.20 + 打 3 个补丁 + 构建

pip install --no-deps <构建产物>/lsglang-1.6.0+flashnext.v0520-py3-none-any.whl
pip install "lk_moe==2.4.1" "sglang-kernel==0.4.7" "tilelang==0.1.11" "flashinfer-python[cu13]==0.6.18"
pip install ./flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl   # SM89 必需
```

> `sglang-kernel==0.4.7` 是 v0.5.20 的启动强校验；`tilelang` 必须 0.1.11（见 TROUBLESHOOTING #15）。

**v1（Lsglang 1.4.13）= 直接装 release wheel：**

```bash
bash scripts/install.sh
```

脚本下载两个 wheel（GitHub release `lsglang-v1.4.12` tag）、建 conda env、按序安装
（torch → flash_attn → lsglang）。`flash_attn-2.8.4+pr2751` 是 **SM89 必需**——
head_dim=256 的 128×64 tile 超 4090 shared memory，PR2751 改 128×32。

</details>

## 4. 模型下载 (Model Download)

```bash
export HF_HUB_OFFLINE=0
modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local_dir /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
```

- 约 **126GiB / 206 shards**
- ⚠️ **revision**：TomPython 项目记录的 HF revision（`7b7192...`）在 **ModelScope 不存在**，直接下 master
- 校验：
  ```bash
  python3 scripts/verify_model_index.py /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4 --expected-shards 206
  # 期望: {"index_shards": 206, "expected_shards": 206, "missing": [], "empty": []}
  ```

## 5. Chat Template：froggeric v22.5（推荐）

模型自带的 RadixArk 模板对 `reasoning_effort` 严格枚举 `xhigh/medium/low`，客户端发 `high` 会 400。**推荐替换为 [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) 顶层 `chat_template.jinja`（v22.5）**——作者声明兼容 Qwen3.8 Flash-Next，实测（2026-09）验证：effort 别名全收（`high/max/ultracode/extreme`→xhigh、`minimal`→low、`none/off`→关思考）、工具调用/中文/thinking 开关全部正常。

```bash
cd /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
cp chat_template.jinja chat_template.jinja.bak-orig        # 备份原模板
curl -L -o chat_template.jinja \
  "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja?download=true"
```

注意：v22.5 默认 effort 为 `medium`（原模板默认 xhigh）——客户端显式传 `reasoning_effort` 则不受影响。**需重启服务生效**。回滚：`cp chat_template.jinja.bak-orig chat_template.jinja`。

（早期版本用手写映射补丁，已被 v22.5 原生功能取代。）

## 6. 启动 (Start)

编辑启动脚本顶部的配置区（模型路径、`FT_BIN`、`SITE_PACKAGES`）后：

```bash
# v4（当前，FreeToken；内置 sudo 提权 + memlock + 补丁自检 + 就绪等待）
bash scripts/start_freetoken.sh

# v3（历史，sglang）
bash scripts/start_lsglang_upstream.sh
```

首次启动约 **40 秒**（206 shards 权重 + 专家 banks；sglang 路线约 5 分钟）。
**就绪判据是日志行 `ready to serve`**——不要用 `/health`，它在加载完成前就返回 200（上游 #537）。

```bash
tail -f /tmp/ft.log                                   # 等 "ready to serve"
curl -s localhost:8000/v1/models                      # 模型名 + max_model_len=262144
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'
```

## 7. 成功定义 (Definition of Success)

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
- MTP（sglang 路线下可启动但 CPU 瓶颈无收益；FreeToken 侧上游对 qwen4_exp 的 MTP 直接丢弃）
- 262144 以上（1M YaRN）
- 公网服务（建议 loopback + 反向代理）
