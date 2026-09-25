# 运维手册 (Operations)

## 1. 启动 (Start)

```bash
# v4（当前，FreeToken）：自动 sudo 提权 + ulimit -l unlimited + 补丁自检 + 就绪等待 + 可选关 GUI
bash scripts/start_freetoken.sh

# v3（历史，sglang）：bash scripts/start_lsglang_upstream.sh
tail -f /tmp/ft.log
```

- 就绪标志：日志行 `ready to serve`（约 **40 秒**）
- ⚠️ **不要用 `/health` 判断就绪**——FreeToken 的 `/health` 在模型加载完成前就返回 200（上游 #537）。
  加载期间打 `/v1/chat/completions` 会被拒。
- ⚠️ 脚本必须在**普通用户**的 shell 里运行（会自己 sudo 提权）。root 外壳的 cwd 会让 spawn 出的 worker 崩溃（见 TROUBLESHOOTING）
- 启动前脚本会自检三个本地补丁是否在位——缺任何一个直接退出并打印重打命令

## 2. 观察 (Observe)

```bash
python3 ft_stats.py                 # /v1/stats + cache status + nvidia-smi 一览
curl -s localhost:8000/v1/stats      # kv/mamba 池、累计 token、ttft 均值/p95
curl -s localhost:8000/v1/models
watch -n 1 nvidia-smi
free -h                             # 模型 126GiB resident + 专家 banks ≈ 200GB
tail -f /tmp/ft.log                 # 关注: Traceback / OOM / Xid / swap 增长
```

`ft_stats.py` 与 `/v1/stats` 提供的 TTFT 统计是**服务端口径**，与客户端计时不同——结论以客户端计时为准（见 BASELINE）。

## 3. 停止 / 重启 (Stop / Restart)

```bash
# 停
pkill -f "ft-venv/bin/ft serve"
# 重启（约 40 秒）
bash scripts/start_freetoken.sh
```

⚠️ 不要用 `pgrep -f` 取 PID 后再 kill——该模式可能匹配到命令行含此字符串的 shell 自身。
改用 `ps aux | grep '[f]t-venv/bin/ft serve'` 确认。

⚠️ **改参数一律走「停 + 重启」**：40 秒就能起回来，而 `ft ctl cache rebuild` 热调在上游有把服务器
wedge 住的记录（#526，重建 OOM 后无法回滚）。

## 4. 日志 (Logs)

- `/tmp/ft.log`（启动脚本指定；每次启动截断重写）
- 关键日志行：
  - `Parsed arguments:`——**全部生效参数**，排查配置问题先看这行
  - `PLE disk backend: io_uring, O_DIRECT`——PLE 表走磁盘后端（不占锁页内存）
  - `Allocating 265216 tokens for KV cache, K + V = 6.26 GiB`——KV 池大小
  - `Free memory after initialization:`——初始化后剩余显存
  - `Prefill batch / Decode batch` + `#running-req` / `#queue-req` / `#mamba-slot: x/N`——实时负载
  - `ready to serve`——就绪
  - ⚠️ 日志里的 `input throughput` / `gen throughput` 是**服务内部口径**，与客户端实测差距可达数量级
    （见 TROUBLESHOOTING #3），**不要用来做结论**

## 5. 常用 curl 验证 (Sanity Checks)

```bash
# 模型信息（工具客户端读这个限制长度）
curl -s localhost:8000/v1/models

# 中文请求
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'

# 工具调用（FreeToken 默认 tool_call_parser=qwen3_coder，无需显式传参）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"北京天气？"}],"max_tokens":128,"tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
# 期望 message.tool_calls 为结构化数组（非 null）

# Anthropic 兼容接口（v4 新增能力；v3 的 sglang 路线没有）
curl -s localhost:8000/v1/messages -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"<served_name>","max_tokens":64,"messages":[{"role":"user","content":"你好"}]}'
```

前缀缓存命中数可从 `usage.prompt_tokens_details.cached_tokens` 读出（启动脚本开了 `--enable-cache-report`）；
Anthropic 格式对应 `usage.cache_read_input_tokens`。

## 6. 边界行为 (Boundary Behavior)

- 上下文上限 262144；KV 是**单一共享池** 265216 tokens（启动时预分配，不随上下文伸缩）
- 长上下文的并发上限由 **mamba 槽**决定：`slots = 4*mrr + max(4, 2*mrr) + 1`
  | `--max-running-requests` | mamba 槽 | 可并行的长上下文路数 | 实测 |
  |---|---|---|---|
  | 2 | 12 | 3 路 | 显存省（配合 cache-rate 0.35 约 42.9G） |
  | **4（本项目生产）** | **24** | **6 路** | 显存 44.4G，decode 65.0/65.3 @73K 不掉 |

  > 「mrr>3 没用」是误解——mrr 直接决定并发上限。
- 生成长度到 262144 总长 → 截断（`finish_reason=length`）
- 多路并行时的**吐字停顿**：FreeToken 的调度里 prefill 默认无条件优先，长 prompt 的 prefill 会整块占住 GPU，
  正在解码的那一路就「突然不吐字」——本机实测最长冻结 **17.7s**。`--decode-interleave-every 2`（本地补丁，
  上游 PR #484）把停顿切成 4 段，最长降到 **7.5s**（总等待时间不变）。
  ⚠️ N 必须按自己的 prompt 长度定：43K prompt 只有 6 个 prefill 块，作者在 PR 里推荐的 N=8 永远到不了阈值、完全无效。

## 7. GUI / 无头主机共存 (Headless Notes)

- 126GB 权重 + 专家 banks 需 ~200GB RAM——若主机有桌面环境，建议服务运行期间关闭 GUI（LightDM 等）。
  `scripts/start_freetoken.sh` 会在检测到 lightdm 时自动停掉（`STOP_GUI=off` 可关掉该行为）
- 远程访问：SSH local forwarding 或 loopback + 反向代理（nginx 需配 `proxy_read_timeout`，默认 60s 会掐断长请求 → 502；建议 900s）
- 端口分工：`8000` API / `8001` nginx / `8002` FreeToken 内部 rendezvous（`FT_DIST_PORT` 补丁解耦，
  否则默认取 API 端口+1 会撞 8001）
