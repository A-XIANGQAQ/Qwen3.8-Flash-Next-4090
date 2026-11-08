# 运维手册 (Operations)

> 当前只有 **v5.9：Strata 0.2.12 服务端 + engine 0.1.58 + unsloth UD-Q4_K_XL（huihui v2 破限）+ froggeric v22.5**。
> 引擎细节见 [STRATA.md](STRATA.md)，换模型见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md)，
> 模板见 [TEMPLATE.md](TEMPLATE.md)，实测见 [BENCHMARK.md](BENCHMARK.md)。

## 1. 启动 (Start)

```bash
# ⛔ 必须在你自己的终端里跑。绝不要从 agent / CI 的工具调用里 nohup/setsid 起：
#    2026-10-15 实测，进程仍落在 systemd 临时单元 dsh-subprocess-*.scope 的 cgroup 里，
#    setsid 只脱离终端/会话、脱不出 cgroup；dsh-web 重启时 server/strata/strata-vision 整组被 SIGTERM。
#    症状：日志只有 stopping (Ctrl+C again...)，没有 traceback / CUDA / OOM。
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
# 就绪：日志行 ready:（0.1.58 + UD-Q4_K_XL 实测：热启 53 秒、首次冷启 106 秒）
#   ready: http://127.0.0.1:8000/v1  (OpenAI: /v1/chat/completions,
#                                     Anthropic: /v1/messages, context 262144 tokens)
```

推荐一次性 systemd 用户单元（cgroup 与 dsh-web 是兄弟，但**不是开机自启**）：

```bash
systemd-run --user --unit=strata-prod --collect \
  --property=RemainAfterExit=yes \
  --working-directory=/path/to/strata/Strata \
  bash /path/to/strata/run_strata.sh
# systemctl --user status strata-prod   |  journalctl --user -u strata-prod -f
# systemctl --user stop strata-prod
```

⚠️ **不要用 `./setup.sh` 起服务**：它会重导 unsloth 原装模板（9,004 字节）、重写 config；
当前用 froggeric v22.5（28,234 字节），靠启动器保证不被覆盖（见 [TROUBLESHOOTING #49](TROUBLESHOOTING.md)）。
⚠️ **起之前显存必须已空**（`nvidia-smi` used <1000 MiB），否则 `cudaMalloc` OOM（[#29](TROUBLESHOOTING.md)）。

## 1a. 模板校验（froggeric v22.5）

历史模式表在 [TEMPLATE.md](TEMPLATE.md)（v5.9 起不再使用）。常用：

```bash
wc -c <pack>/tokenizer/chat_template.jinja            # 28234（froggeric v22.5）
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["chat_template"]))'   # 28226
```

v5.9 起模板是 froggeric v22.5 原版，`reasoning_effort`（off/low/medium/high/xhigh）由请求头控制；
旧合并模板的自动选择脚本 `patches/08_auto_check.py` 只在复现 v5.8 时用。

## 1b. 引擎升级（同代最小路径）

原则：**只换服务端源码 + 引擎目录，不动模型 / pack / 模板 / config 参数，不跑 setup/update**。
升级前先读 release 与 issue；当前判断见 [STRATA-UPSTREAM-2026-11-08.md](STRATA-UPSTREAM-2026-11-08.md)。

```bash
# ① 下载并校验（sha256 必须等于签名 feed；BUILD.json 的 archs 必须含 89）
gh release download <tag> --repo Niko1221/Strata \
  -p 'strata-engine-<ver>-linux-x86_64-cuda-cuda13.tar.gz' -D /tmp
sha256sum /tmp/strata-engine-<ver>-linux-x86_64-cuda-cuda13.tar.gz

# ② 新树解压到独立目录（生产树继续跑，零停机）
curl -L https://github.com/Niko1221/Strata/archive/refs/tags/<tag>.tar.gz \
  | tar -xz -C strata/Strata-<tag> --strip-components=1
mkdir -p strata/Strata-<tag>/engine
tar -xzf /tmp/strata-engine-<ver>-...tar.gz -C strata/Strata-<tag>/engine

# ③ 复制生产 config，只改 exe / vision.exe / cwd / log；args 逐字不动
# ④ 停机切换（实测中断 48 秒）
bash strata/stop_strata.sh          # 轮询到显存 <1000 MiB 才算停完
bash strata/run_strata.sh           # 指向新树/新 config
```

**准入线（任一条不过就回滚）**：短上下文 decode ≥0.97×基线；1K/32K/100K 的 TTFT 不劣于基线；
长上下文（≥131K）free VRAM 不出现 61→49→28→13→0 MiB 的塌陷（#3952）；专家缓存 ≥31 GiB；
启动后空闲显存 ≥1800 MiB；视觉三色条 3/3；4 路并发无空答；启动 60 秒内无新 WARNING。

**三条铁律**：

1. **server / engine / vision 必须同代**。0.2.12 的 `MIN_ENGINE` 仍是 `(0,1,46)`（只在 setup.py 校验；serve 侧无版本门控），但新引擎特性只在 0.1.58；
   只换一个会静默少功能（[#39](TROUBLESHOOTING.md)）。
2. **不要跑 setup 主流程**：`carry_over()` 不合并手工 args（30 个手调参数会丢）；#2902 还会删大分片。
3. **`engine/` 整目录替换**，不能只拷 `strata` + `strata-vision`：0.1.44 起还有 `libstrata_kq.so`。

**回滚**：引擎层面停服务 → 等显存 <1000 MiB → 换回旧 `engine/` 目录；但**旧模型与旧 tree 已在 2026-11-08 物理删除**，
模型回滚要重下权重并重建 pack（见 [STRATA-MODEL-2026-11-08.md](STRATA-MODEL-2026-11-08.md) §12）。

### 1.1 0.2.9 起（0.2.10 / 0.2.12 沿用）服务端会自动塞 11 个环境变量（要知道，不然会误判"参数没生效"）

`serve/server.py` 在起引擎前用 `env.setdefault()` 补默认值，**自己设过的值永远优先**：

| 触发条件 | 被设的变量 | 关掉的方法 |
|---|---|---|
| Linux + `--batch`/`--parallel` ≥2 + CUDA Qwen 引擎 | `STRATA_CKPT_POOL=1`、`STRATA_CKPT_POOL_READY=4`、`STRATA_SLOT_CKPT_MOVE=1`、`STRATA_SLOT_COPY_DIRECT=1`、`STRATA_SLOT_COPY_BATCHSYNC=1`、`STRATA_BATCH_GDN_GROUPS=1`、`STRATA_BATCH_ATTN_ROWS=1`、`STRATA_BATCH_SELF_COMMIT=1`、`STRATA_QSA_SLOTS=1` | 配置 `env` 或启动器里把任意一个设 `0` |
| Linux + `arch=qwen4exp` + 非 HIP/SYCL/Metal | `STRATA_GDN_TC=1`、`STRATA_GDN_CONVL2=2` | `STRATA_GDN_TC=0` / `STRATA_GDN_CONVL2=0` |

验证（只看引擎进程自己的环境，不看 shell）：

```bash
EPID=$(pgrep -x strata); tr '\0' '\n' < /proc/$EPID/environ | grep -E 'STRATA_(CKPT_POOL|QSA_SLOTS|GDN_)'
```

### 1.2 0.2.11 起的两条新默认（0.2.12 沿用，在 48 GB 卡上要知道）

1. **FP16 weight cache**：`--parallel`/`--batch` ≥2 时自动开，规则是「分配后卡上仍留 3,072 MiB 空闲」。
   本机启动后只剩 ~2,900 MiB，于是从 0 MiB 起步，日志固定这一行——**看到它不是故障**：

   ```
   strata prefill: FP16 weight cache 0 MiB (the card keeps 3072 MiB free; STRATA_PF_W16_CACHE_MIB=0 turns it off)
   ```

   想硬开只能砍 `--conversation-cache-mib`（8192）或专家缓存换 3 GiB 空间，本机判断不划算。
2. **单卡 CPU/upload share 按 CPU 实测算**：日志 `PCIe share: start 0.15 from 32 pool workers … (was 0.55)`。
   回 0.2.10 的固定规则：`STRATA_PCIE_AUTO_SPLIT=0`、`STRATA_PCIE_K_RULE=0`、`STRATA_PCIE_SHARE_MAX=0.60`
   （写在 config 的 `env` 里即可，服务端只 `setdefault`，自己设的优先）。

## 2. 观察

```bash
tail -f strata/strata-prod.log      # Traceback / OOM / Xid / verify / ready:
pgrep -x strata                     # 引擎（精确名，不会自匹配）
pgrep -x strata-vision              # 装了图才有
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
free -h                             # 权重 mmap + 专家驻留：used 高、swap 应为 0
curl -s 127.0.0.1:8000/metrics      # 引擎/并发/每请求/缓存/硬件，见 §6
```

## 3. 停止 / 重启

```bash
bash strata/stop_strata.sh          # 停 serve + engine，并等显存降到 <1000 MiB
```

> ⛔ **停完必须等显存真正释放再起下一个**：进程消失 ≠ 显存释放（实测残留 47,712 MiB，
> 立刻起会 `cudaMalloc ... out of memory`）。判据只有 `nvidia-smi` 的 used <1000 MiB。
> ⚠️ 不要 `pkill -f "serve/server.py"`：会匹配到自己的 shell。用 `pgrep -x strata` 或
> `ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && /[s]erve[^ ]*\/server\.py/ {print $1}'`。

## 4. 日志

- 就绪：`ready: http://127.0.0.1:8000/v1 ... context 262144 tokens`
- 会话：`session is up (engine 0.1.58)`；并发：`--batch: 4 slot sessions on CUDA0`
- 视觉：`strata serve: N MiB of VRAM free with everything loaded`，**N 必须 ≥1600**
- 每请求：`prompt ... tokens = ... reused + ... read in ... ms (... tok/s), ... generated in ... ms (... tok/s)`
- ⚠️ 启动器用 `> log` 会**截断**旧日志；重启前要留证据就先备份或改 `>>`
- ⚠️ 日志里的 `done: N tokens in T s` 自相矛盾（[#3](TROUBLESHOOTING.md)），**不要用**；以客户端或 `/metrics` 为准

## 5. 常用 curl

```bash
curl -s 127.0.0.1:8000/v1/models
curl -s 127.0.0.1:8000/health
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["build_info"],len(d["chat_template"]))'

# 文本（OpenAI chat；服务端不支持 /v1/completions）
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-q4kxl-ablit","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'

# 关思考 + 固定输出（基准里用同一套）
#   "chat_template_kwargs":{"enable_thinking":false}

# 工具调用
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"北京天气？"}],"max_tokens":128,"tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'

# Anthropic 兼容
curl -s 127.0.0.1:8000/v1/messages -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"...","max_tokens":64,"messages":[{"role":"user","content":"你好"}]}'

# 只数 token（不生成；基准校准用）
curl -s 127.0.0.1:8000/v1/messages/count_tokens -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"Hello world"}]}'
```

## 6. `/metrics`（排障与基准的引擎侧证据）

| 字段 | 含义 |
|---|---|
| `engine.engine` / `engine.model` | 引擎版本（0.1.58）/ served name |
| `engine.kv` / `kv_resident` / `batch_slots` | `k8v4` / 32768 / 4 |
| `engine.expert_cache_mib` / `expert_slots` | 专家缓存；升级后低于 ~31 GiB 判定不合格 |
| `engine.vram_free_mib` | 启动后空闲显存；应 ≥1800 |
| `live.state` / `prompt_tokens` / `prefill_tok_s_mean` | 当下在做什么 |
| `requests[]` | 最近 500 条：`prompt_tokens`、`reused`、`prompt_ms`、`decode_ms`、`decode_tok_s`、`hit_rate` |
| `conversation_cache` | 命中/复用/eviction；判断缓存是否被外部流量污染 |
| `hardware` / `history` | GPU/CPU/显存/功率，1 秒粒度 |

## 7. GUI / 无头与反向代理

- 桌面 GUI 与引擎抢显存：建议服务期间关 GUI（本项目启动脚本可选 `STRATA_NO_SUDO=1` 的无提权分支）。
- 远程：SSH 转发，或 loopback + nginx；**`proxy_read_timeout` 默认 60s 会掐断长请求 → 502，建议 ≥900s**。
- 端口：只用 `:8000`（serve 层）；引擎没有额外对外端口。
- **不要开机自启**：用一次性 systemd 单元或手动脚本（见 §1）。
