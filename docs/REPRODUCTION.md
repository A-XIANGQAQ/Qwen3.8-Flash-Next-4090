# 完整复现 (Full Reproduction)

## 1. 硬件要求 (Hardware Requirements)

已验证主机：

- 单 RTX 4090 **48GB**，SM89（原版 24GB 未验证——权重 GPU 部分 ~17GB + KV 6GB + Mamba 6.5GB ≈ 30GB，超出 24GB）
- CPU：建议 ≥32 物理核（decode 是 CPU 瓶颈，核越多越快）；LK_THREADS 建议 = 物理核数 − 2
- RAM：**≥247GB**（模型 126GB resident + 运行时；`--disable-shared-experts-fusion` 下实测进程峰值 ~200GB）
- 磁盘：模型 126GB + wheel/环境 ~30GB
- 不要启用 swap 作为容量替代

## 2. 软件前提 (Software Prerequisites)

- Linux；NVIDIA driver 可运行 CUDA 13 runtime（实测 610.x）
- conda/miniconda
- ModelScope CLI（模型下载）
- host 无需 SGLang/CUDA toolkit

## 3. 环境安装 (Environment Install)

```bash
bash scripts/install.sh
```

脚本内容（或手动执行）：
1. 下载两个 wheel（GitHub release `lsglang-v1.4.12` tag）：
   - `lsglang-1.4.13-py3-none-any.whl`（SHA256 `2ccc92f9...`）
   - `flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl`（440MB，**SM89 必需**——head_dim=256 的 128×64 tile 超 4090 shared memory，PR2751 改 128×32）
2. `conda create -n lsglang python=3.12`
3. 安装顺序（关键）：
   ```bash
   pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
   pip install ./flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl   # 先装，让依赖解析看到
   pip install ./lsglang-1.4.13-py3-none-any.whl
   ```

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

## 5. Chat Template 补丁（可选但推荐）

模型模板的 `reasoning_effort` 严格枚举 `xhigh/medium/low`，客户端发 `high` 会 400。打映射补丁（参考 froggeric v22.4 思路，仅入口归一化不改行为）：

```bash
cd /path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4
cp chat_template.jinja chat_template.jinja.bak-effort
patch -p0 < /path/to/repo/patches/chat_template.effort.patch
```

效果：`high/max/ultracode/extreme`→xhigh、`minimal`→low、`none`→关思考；其余行为不变。**需重启服务生效**。

## 6. 启动 (Start)

编辑 `scripts/start_lsglang.sh`（模型路径、`SGLANG_BIN`、对外名、端口）后：

```bash
bash scripts/start_lsglang.sh
tail -f /tmp/lsglang.log   # 等 "The server is fired up and ready to roll!"
```

首次启动约 5 分钟（126GB 加载 + KV/Mamba cache + CUDA graph 捕获）。验证：

```bash
curl -s localhost:8000/health                        # 200
curl -s localhost:8000/v1/models                      # 模型名 + max_model_len=262144
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 7. 成功定义 (Definition of Success)

- [ ] 206 shards 校验通过
- [ ] `/health` 200，engine ready
- [ ] `/v1/models` 返回预期模型名
- [ ] 中文请求可读、无乱码
- [ ] RAM 有余量、swap 为 0、无 Xid/OOM
- [ ] （可选）8k 无缓存 prefill 客户端计时 ~7s（见 BASELINE）

## 不在默认范围 (Out of Scope)

- 单卡 24GB 原版 4090；FP8/BF16 权重（fp8 在 Lsglang 有已知问题）
- MTP（可启动但 CPU 瓶颈无收益，见 TROUBLESHOOTING）
- 262144 以上（1M YaRN）
- 公网服务（建议 loopback + 反向代理）
