# 运维手册 (Operations)

> **两代引擎互斥**：v5.1（Strata 0.1.45）与 v4（FreeToken）各占本卡 90%+ 显存，**任何时刻只能活一个**。
> 切换是停机切换（约 1–2 分钟）。详细路线见 [STRATA.md](STRATA.md)，同代版本升级见 §1c。

## 1. 启动 (Start)

```bash
# ── v5（当前，Strata）──────────────────────────────
# ⛔ **必须在你自己的终端里跑**。绝不要从 agent / CI 的工具调用里用 `setsid ... &` 起。
#    2026-10-15 09:40 实测踩到：从 agent 的 bash 工具里起的进程，**仍落在 systemd 临时单元
#    `dsh-subprocess-*.scope` 的 cgroup 里**——`setsid` 只脱离终端/会话，**脱不出 cgroup**。
#    重启 dsh-web 时该 scope 被回收，server + strata + strata-vision **整组**被 SIGTERM/SIGKILL。
#    症状极具误导性：日志只留 `stopping (Ctrl+C again to end the engine at once)`、
#    **没有任何 traceback / CUDA / OOM** —— 那不是崩溃，是被信号停的。
#    判据（systemd 侧才有）：
#      journalctl -b 0 | grep -E 'dsh-subprocess.*(SIGTERM|SIGKILL)'
#
#    从自己终端起则落在登录会话（KillUserProcesses 默认不杀 + Linger=yes），**关掉终端也不会死**。
nohup bash strata/run_strata.sh > strata/strata-prod.log 2>&1 &
# 桌面版（含「等显存 <1000 MiB」+ 关 GUI）：bash ~/桌面/一些启动脚本/start_strata_kill_gui.sh
# 就绪标志：日志行 "ready:"（约 40–55 秒）
#   ready: http://127.0.0.1:8000/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context 262144 tokens)

# ── 推荐：一次性 systemd 用户单元（2026-10-18 起本站生产就用这个）─────────
# 比 nohup 强在 cgroup：单元的 cgroup 是 dsh-web.service 的**兄弟**，dsh-web 重启不会连坐；
# 又不是开机自启（transient、无 [Install]、--collect 退出即消失，重启后照样要手动拉）。
systemd-run --user --unit=strata-prod --collect \
  --working-directory=~/dsh_workspace/strata/Strata \
  bash ~/dsh_workspace/strata/run_strata.sh
#   systemctl --user status strata-prod      # 状态
#   journalctl --user -u strata-prod -f      # 日志
#   systemctl --user stop strata-prod        # 停（要换桌面脚本启动前必须先停，否则 8000 被占）

# ⚠️ 不要用 ./setup.sh 起服务 —— 它无参数时会自己拉起模型并【绕过 chat 模板守卫】，模板会退回原装
# ⚠️ **当前生产模型是 DavidAU abliterated IQ3_S**，它自带 22 万字符模板，
#    启动器里那条 froggeric 模板守卫已被移除（旧版备份 run_strata.sh.bak-undq4-20261018）。见 TROUBLESHOOTING #45
# ⚠️ 必须先确认显存已空（见 §3），否则 cudaMalloc OOM

# ── v4（FreeToken）⛔ 2026-10-18 已删除运行时与权重，下面命令仅存档，不再可用 ──
bash scripts/start_freetoken.sh        # 自动 sudo 提权 + ulimit -l unlimited + 补丁自检 + 就绪等待
tail -f /tmp/ft.log
```

- v4 就绪标志：日志行 `ready to serve`（约 **40 秒**）
- ⚠️ **不要用 `/health` 判断就绪**——FreeToken 的 `/health` 在模型加载完成前就返回 200（上游 #537）。
  加载期间打 `/v1/chat/completions` 会被拒。
- ⚠️ v4 脚本必须在**普通用户**的 shell 里运行（会自己 sudo 提权）。root 外壳的 cwd 会让 spawn 出的 worker 崩溃（见 TROUBLESHOOTING）
- 启动前 v4 脚本会自检三个本地补丁是否在位——缺任何一个直接退出并打印重打命令

## 1b. 引擎互切 (Switch / Rollback)

```bash
# switch_engine.sh 是本站自建的一键互切脚本，**不在本仓库**（流程见下方，可自行实现）
S=switch_engine.sh

bash $S B --status          # 只看当前是谁（A=v4 FreeToken, B=v5 Strata）
bash $S B                   # 切到 v5
bash $S A                   # 切回 v4（回滚）

# 或直接手工
bash scripts/start_freetoken.sh   # 回滚 v4
bash stop_strata.sh               # 停 v5（含显存释放判据）
```

`switch_engine.sh` 的流程：停当前 → **轮询等 `nvidia-smi` used <1000 MiB** → 起目标 →
按引擎区分的就绪判据 → **走服务入口跑真实推理探针** → 失败则自动回滚。
⚠️ 其**真实切换 + 回滚路径尚未实测**（2026-10-13 的切换是手工做的）。

## 1c. 引擎版本升级 (Upgrading to a new upstream tag)

> 2026-10-18 用这套流程从 `0.1.43.1` 升到 `0.1.45`，实测数据见 [UPGRADE.md](UPGRADE.md) §0b。

**顺序不能变**：并行构建 → 冷启动 A/B → 达标才换二进制。**任何一步都不跑 `update.sh` / `setup.py` 主流程**。

```bash
# ① 并行构建一棵新树（生产树继续跑，零停机）
git -C strata/Strata fetch origin --tags
git -C strata/Strata worktree add strata/Strata-<tag> <tag>
ln -s ../../Strata/third_party/llama.cpp strata/Strata-<tag>/third_party/llama.cpp   # 未跟踪的大目录
#    把补丁 07 打上（行号随 tag 漂移，上下文不变）
#    然后只调 build_engine()：<installed tree>/.venv/bin/python scripts/build_engine_upgrade.py --src strata/Strata-<tag>
#    产物：engine/strata、engine/strata-vision、engine/libstrata_kq.so（0.1.44 起新增）、engine/BUILD.json

# ② 冷启动 A/B：两臂同一份 config，只有 exe / vision.exe 指向不同目录
bash scripts/arm_start.sh <serve-dir> <config.json> <server.log>
python scripts/ab_probe.py --arm A            # 6×decode + 1K/32K/100K + 4 路 + 视觉
python scripts/long_ctx_probe.py --arm A --reps 3  # 长上下文 decode 单独复测（短答案会淹没信号）

# ③ 达标才换（停机约 3 分钟）
bash strata/stop_strata.sh
cp -a strata/Strata/engine strata/Strata/engine-<old>        # 留档，回滚用
cp -a strata/Strata-<tag>/engine/. strata/Strata/engine/     # 整目录：kq 库也要
git -C strata/Strata stash && git -C strata/Strata checkout <tag>
git -C strata/Strata apply qwen4090-repo/patches/07_strata_esp_family_whitelist.patch
bash "/path/to/start_strata_kill_gui.sh"                     # 自己的终端
```

**准入线**（任一条不过就不切）：短上下文 decode ≥ 0.97× 基线；32K/100K 的 decode 判等；
1K/32K/100K 的 TTFT 不劣于基线；专家缓存 ≥ 31 GiB；启动后空闲显存 ≥ 1800 MiB；
视觉三色条 3/3；4 路并发墙钟 ≤ 基线；启动 60 秒内无新 WARNING。

**三条铁律**：

1. **server.py / 引擎 / vision 必须同版本**。0.1.45 的 `setup.py` 写死 `MIN_ENGINE = (0, 1, 45)`，
   且 `stale_binary()` 会拒绝安装「版本与源码不符」的二进制——只换其中一个会得到诡异行为。
2. **只调 `build_engine()`，不要跑 setup 主流程**：`carry_over()` 不会合并手工加进 `args` 的选项
   （12 个手调参数会丢）；且 setup 对大分片算 SHA-256 有已知 bug（#2902），**会删掉正确的模型分片**。
3. **`engine/` 要整目录替换**，不能只拷 `strata` + `strata-vision`：0.1.44 起多了一个
   `libstrata_kq.so`（K-quant prompt 内核库），漏了它等于退回旧内核。

回滚见 [UPGRADE.md](UPGRADE.md) §0b.6：`engine-<old>/` 整目录拷回 + `git checkout <old tag>` + 重打补丁。

## 2. 观察 (Observe)

```bash
# ── v5（Strata）─────────────────────────────
tail -f strata/strata-prod.log       # 关注: Traceback / OOM / Xid / 就绪行 "ready:"
pgrep -x strata                      # 引擎存活（精确匹配，不会自匹配）
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 起停前后的显存判据
free -h                              # v5 实测: 68Gi used / 198Gi buff-cache（含 mmap 的 GGUF）

# ── v4（FreeToken）──────────────────────────
python3 ft_stats.py                  # /v1/stats + cache status + nvidia-smi 一览
curl -s localhost:8000/v1/stats      # kv/mamba 池、累计 token、ttft 均值/p95
tail -f /tmp/ft.log                  # 关注: Traceback / OOM / Xid / swap 增长

# ── 通用 ───────────────────────────────────
curl -s localhost:8000/v1/models
watch -n 1 nvidia-smi
```

`ft_stats.py` 与 `/v1/stats`（v4）提供的 TTFT 统计是**服务端口径**，与客户端计时不同——结论以客户端计时为准（见 BASELINE）。

## 3. 停止 / 重启 (Stop / Restart)

```bash
# ── v5（Strata）──────────────────────────────
bash stop_strata.sh             # 停服务端 + 引擎，并【等显存降到 <1000 MiB】才算成功

# ── v4（FreeToken）───────────────────────────
ps -eo pid,args | grep '[f]t-venv/bin/ft serve'   # 先确认真实 PID（见下方自匹配警告）
kill <pid>
bash scripts/start_freetoken.sh        # 重启（约 40 秒）
```

> ⛔ **停完必须等显存真正释放再起下一个引擎。** 实测：`ft_stop.sh` 打印"干净停止"时
> `nvidia-smi` 仍显示 **47,712 MiB 被占**，立刻起 Strata 会 `cudaMalloc ... out of memory`。
> **判据是 `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 降到 <1000 MiB，不是进程消失。**
> （2026-10-13 漏这步 → nginx 502 五分钟。）
> 一条命令自查：`nvidia-smi --query-gpu=memory.used --format=csv,noheader`

⚠️ **`pkill -f` / `pgrep -f` 都会把「命令行里含该字符串的自己的 shell」一并匹配**——实测踩过
（`pkill -f "ft-venv/bin/ft serve"` 会连发起它的 shell 一起杀掉）。两种稳妥写法：

- `ps -eo pid,args | grep '[f]t-venv/bin/ft serve'` 拿到真实 PID 再 `kill`
- 用引擎二进制的**精确名**匹配：Strata 就是 `pgrep -x strata`（不会自匹配）

⚠️ **改参数一律走「停 + 重启」**：40 秒就能起回来，而 `ft ctl cache rebuild` 热调在上游有把服务器
wedge 住的记录（#526，重建 OOM 后无法回滚）。

## 4. 日志 (Logs)

- **v5（Strata）**：`strata/strata-prod.log`（启动器重定向）。
  两个必看行：**就绪** `ready: http://127.0.0.1:8000/v1 ... context 262144 tokens`；
  **破限判据** `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`。
- **v4（FreeToken）**：`/tmp/ft.log`（启动脚本指定；**每次启动截断重写——重启前先备份**）
- v4 关键日志行：
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

## 6. 边界行为 · v4 (Boundary Behavior · FreeToken)

> ⚠️ **本节整段是 v4 的调度与池化行为，不适用于 v5。** v5 的对应语义差别很大：
> 并发是**每 slot 独立 session**（不是共享池），KV **每 slot 独立**且可流式到宿主内存
> （`--kv k8v4 --kv-resident 32768`），**没有 mamba 槽**这套东西。见 [STRATA.md §5](STRATA.md)。

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
- 端口：只需指定 API 端口。FreeToken 内部 rendezvous 端口**自动挑空闲的**（补丁 04，
  从 API 端口+1 起向上找，被占用就顺延并打日志）；需要固定时显式设 `FT_DIST_PORT`
