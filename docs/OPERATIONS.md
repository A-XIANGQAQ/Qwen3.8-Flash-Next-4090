# 运维手册 (Operations)

> 当前只有 **v5.3：Strata 0.2.2 服务端 + engine 0.1.47 + DavidAU IQ3_S**。
> 引擎细节见 [STRATA.md](STRATA.md)，升级判断见 [STRATA-UPSTREAM-2026-10-25.md](STRATA-UPSTREAM-2026-10-25.md)，
> 模板见 [TEMPLATE.md](TEMPLATE.md)，实测见 [BENCHMARK.md](BENCHMARK.md)。

## 1. 启动 (Start)

```bash
# ⛔ 必须在你自己的终端里跑。绝不要从 agent / CI 的工具调用里 nohup/setsid 起：
#    2026-10-15 实测，进程仍落在 systemd 临时单元 dsh-subprocess-*.scope 的 cgroup 里，
#    setsid 只脱离终端/会话、脱不出 cgroup；dsh-web 重启时 server/strata/strata-vision 整组被 SIGTERM。
#    症状：日志只有 stopping (Ctrl+C again...)，没有 traceback / CUDA / OOM。
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
# 就绪：日志行 ready:（0.1.47 实测 36 秒）
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

⚠️ **不要用 `./setup.sh` 起服务**：它会重导原装模板（大小 226745）、重写 config；
当前模型自带 22 万字符模板，靠启动器保证不被覆盖（见 [TROUBLESHOOTING #45/#49](TROUBLESHOOTING.md)）。
⚠️ **起之前显存必须已空**（`nvidia-smi` used <1000 MiB），否则 `cudaMalloc` OOM（[#29](TROUBLESHOOTING.md)）。

## 1a. 模式自动选择（模板内）

规则/词表/评测全在 [TEMPLATE.md](TEMPLATE.md)。常用：

```bash
python3 patches/08_auto_check.py --set dev+val        # 离线回归（渲染真实模板）
python3 patches/08_auto_check.py --case "讲个笑话"     # 单条，带分数
wc -c <pack>/tokenizer/chat_template.jinja            # 249545 = v3
```

判定：**分数 ≥2 且严格高于第二名才选，否则不选**。线上判据：`讲个笑话` 的 `prompt_tokens` ≈ 2847（选了 comedy）。

## 1b. 引擎升级（同代最小路径）

原则：**只换服务端源码 + 引擎目录，不动模型 / pack / 模板 / config 参数，不跑 setup/update**。
升级前先读 release 与 issue；当前判断见 [STRATA-UPSTREAM-2026-10-25.md](STRATA-UPSTREAM-2026-10-25.md)。

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

1. **server / engine / vision 必须同代**。0.2.2 的 `MIN_ENGINE = (0,1,46)`，`stale_binary()` 会拒绝不匹配的二进制；
   只换一个会静默少功能（[#39](TROUBLESHOOTING.md)）。
2. **不要跑 setup 主流程**：`carry_over()` 不合并手工 args（12 个手调参数会丢）；#2902 还会删大分片。
3. **`engine/` 整目录替换**，不能只拷 `strata` + `strata-vision`：0.1.44 起还有 `libstrata_kq.so`。

**回滚**：停服务 → 等显存 <1000 MiB → config 指回旧 `engine/`（或整目录拷回）→ 起服务。
旧树与旧引擎目录全程保留，不需要重下。

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
- 会话：`session is up (engine 0.1.47)`；并发：`--batch: 4 slot sessions on CUDA0`
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
  -d '{"model":"qwen3.8-flash-next-davidau-iq3s-ablit","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'

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
| `engine.engine` / `engine.model` | 引擎版本（0.1.47）/ served name |
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
