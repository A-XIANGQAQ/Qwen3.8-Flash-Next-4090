# 排障手册 (Troubleshooting)

全部为实测踩坑记录。

> **适用性标记**（2026-09-24 起引擎为 v4 = FreeToken）：
> 🟢 **通用**（两代引擎都适用）｜🟡 **v4 适用**｜⚪ **已归档**（v3 sglang 栈专属，脚本与补丁仍保留可复现）
>
> 编号保持不变，以免其它文档里的交叉引用失效。

## 🟢 1. 启动 OOM：47.7GB ngram 表被塞进 GPU (Boot OOM: 47.7GB n-gram table to GPU)

症状：启动时 `CUDA out of memory. Tried to allocate 47.69 GiB`（= 51.2B 参数的 ngram 查找表，uint8 存储），堆栈在 `qwen4_exp.py _init_qwen4_exp_layer_extensions → Qwen4ExpPLELayer`。

实测触发组合：
- ❌ 去掉 `--disable-shared-experts-fusion`（fusion 开启）
- ❌ 删除 `--max-total-tokens`（自动内存规划把 ngram 表放 GPU）
- ❌ `NUMA 变量注释 + PREFETCH_WINDOW=0 + POWER_SAVING=0 + max-running-requests 4` 的组合（prefetch0/power0 单独改动是安全的，见 BASELINE）

✅ 安全配置：NUMA 3 变量启用 + 显式 `--max-total-tokens 265216` + `--disable-shared-experts-fusion` + 其余默认。
背景：`ple_offload_embedding` 的自动解析条件是 `is_cuda and dtype==bf16`（overrides.py），NVFP4 模型下 ngram offload 依赖上述组合。

> v4 说明：FreeToken 默认 `--ple-backend disk`（io_uring + O_DIRECT），PLE 表不常驻 GPU，本坑在 v4 上不复现；
> 但 v4 有自己的一对参数陷阱，见 #22。

## 🟢 2. 工具调用返回 tool_calls: null (Tool calling returns null)

模型输出了 `<tool_call>` 文本但 API 的 `tool_calls` 字段为 null——**必须显式启用解析器**。

- ⚪ v3（sglang）：`--tool-call-parser qwen3_coder` 必须显式传，sglang 的"自动检测"不会真正启用解析。
- 🟡 v4（FreeToken）：`tool_call_parser='qwen3_coder'` **默认已启用**（日志 `Parsed arguments` 可见），无需传参。

## 🟢 3. 服务日志吞吐字段不可信 (Log throughput field is unreliable)

- v3：sglang 日志 `input throughput (token/s)` 与客户端实测差 **12~57 倍**（实例：日志 16 t/s vs 客户端 918 t/s；甚至出现过 0.05 t/s）。
- v4：FreeToken 日志的 `input throughput` / `gen throughput` 同样是服务内部口径，与客户端差异可达数量级。

**测速一律用客户端计时**（发请求掐表 + `usage.*_tokens`），见 `scripts/bench_once.py`。

## ⚪ 4. chunked-prefill-size 4096 使 prefill 暴跌 (chunk 4096 kills prefill)

*（v3 sglang 专属，已归档）*

8k prompt prefill：4096 → **-38%**（736 vs 1195 t/s）；8192 与 16384 无差异。保持 8192。

## ⚪ 5. --cuda-graph-backend-prefill full 崩溃 (cgprefill full crashes)

*（v3 sglang 专属，已归档）*

`RuntimeError: QSA CUDA graph state is not initialized`——QSA（Qwen sparse attention）后端与 prefill CUDA graph 捕获冲突。必须 `disabled`。

## ⚪ 6. resident 层数 vs 256K 上下文 (Resident layers vs 256K context)

*（v3 sglang 专属，已归档）*

- 每层 experts 权重 ~1.4GB 驻显存；显存总量不变是 **cache 让位**（mem-fraction 0.95 盘子里：Mamba 先让、见底后 KV 页表暴缩）
- 实测：6 层 KV=318464 ✅、7 层 KV=255616 ❌（<262144）、10 层 KV=70848 ❌
- **6 层（0-5）是保 256K 的最多层**

> v4 的对应旋钮是 `--moe-cache-rate`（决定多少专家常驻显存），结论见 #22 与 BASELINE。

## 🟢 7. reasoning_effort 枚举 (Effort enumeration)

- v3 sglang API 层收：`none/minimal/low/medium/high/xhigh/max`（ultracode/extreme 在 API 层 400）
- 模型自带模板只认 `xhigh/medium/low`，`high` 会 400
- **解决（推荐）**：替换为 froggeric v22.5 模板（原生收全别名，见 REPRODUCTION §5），实测 7 枚举全通过

## ⚪ 8. MTP 投机解码 (MTP speculation)

*（v3 sglang 专属，已归档）*

- 可启动：`--speculative-algorithm NEXTN` 等参数；模型以 `Qwen4ExpForCausalLMMTP` 加载，accept rate 0.62~0.72
- **但 decode 无收益**（35.7 vs 无 MTP 36~40 t/s）：CPU 核数是瓶颈
- v4 上：上游对 qwen4_exp 的 MTP 直接丢弃（下游分支有 +12.2% decode 的报告，未合并）

## 🟢 9. ModelScope revision (ModelScope revision)

TomPython 项目记录的 HF revision（`7b7192...`）在 ModelScope **不存在**（只有 master）。校验靠 `verify_model_index.py`（206 shards）。

## 🟢 10. N-gram 表与 offload (N-gram table & offload)

- 51.2B 参数的 ngram 查找表在 **layer 1 的 PLE**（`layers.1.ple.ple_embedding.ngram_embedding`，128 shards，每 token 只 touch ~100-200KB）
- NVFP4 存储 25.6GB；v3 下 `ple_offload_embedding` 驻 pinned host；**v4 下默认走磁盘后端**（io_uring + O_DIRECT）
- 更优方案（参考 vLLM PR #54070 / DGX Spark 社区）：NVMe mmap 磁盘 offload（省内存，代价 ~8%）

## 🟢 11. 温度漂移与测速 (Thermal drift in measurements)

同配置前后重测 decode 37.8 → 40.4 t/s（±7% 漂移，CPU 满载发热降频）。**A/B 对比必须在同窗口内做**，跨小时对比会误判参数效果。

## 🟢 12. pgrep 自匹配陷阱 (pgrep self-match trap)

`pgrep -f '<引擎进程串>'` 会匹配命令行含该字符串的 shell 自身（自杀 → exit 144）。用 `ps aux | grep '[f]t-venv/bin/ft serve'`。

## ⚪ 13. PLE 表 GPU 瞬态分配 OOM (Upstream PLE transient GPU alloc)

*（v3 sglang 专属，已归档）*

上游 Flash-Next 实现里 `Qwen4ExpPLELayer` 会**先在 GPU 构造整张 PLE 表再搬到 pinned 内存**——48GB 卡必炸；
且该 GPU 占位随后即被 `del`，**纯浪费**。修复：`patches/02_ple_cpu_alloc.patch`。
（上游已用 meta device 方案在 PR #39928 修复，但晚于 v0.5.20。）

## 🟢 14. 锁页内存与 memlock 限制 (memlock)

锁页内存分配需要放开 `ulimit -l`（系统默认常仅 8MB），启动脚本已内置 `ulimit -l unlimited`（因此需要 root——脚本自动 sudo 提权）。

> **v4 的原因与 v3 不同**：v3 是 PLE 表需 47.7GiB 锁页内存；
> **v4 下 PLE 走磁盘后端**，`ulimit -l` 真正服务的是**专家 host banks 的 mlock 常驻**。

## ⚪ 15. tilelang 0.1.12 编译不兼容 (tilelang version)

*（v3 sglang 专属，已归档）*

CUDA graph 捕获期报 `CUDA compiler and CUDA toolkit headers are incompatible` = tilelang **0.1.12** 与本环境 nvcc/CCCL 组合冲突。**锁 tilelang==0.1.11**。

## ⚪ 16. sglang-kernel 版本强校验 (v0.5.20 requires >= 0.4.7)

*（v3 sglang 专属，已归档）*

启动即 `Exception: sglang-kernel is installed with version 0.4.6.post1, ...`。
> v4 注意：`freetoken[accel]` 装的是 **sglang-kernel 0.4.5**，比 v3 要求的 0.4.7 低。
> 两者服务于不同引擎路径，不冲突，但**不要为了"满足 v3"把它升上去**而破坏 v4 环境。

## ⚪ 17. get_global_server_args() 在 v0.5.20 已废弃 (retired API)

*（v3 sglang 专属，已归档）*

v0.5.20 把 `get_global_server_args()` 改为**硬报错**。lk_moe 集成补丁里 1 处踩到（已修入 `patches/01`）。

## 🟢 18. 测量窗口被外部流量污染 (measurement pollution)

服务对局域网开放时，**外部客户端会不定时打 8000 口**。实测同一 145K 上下文 decode：独占时 **39.4 t/s**，
撞上 2-3 个并发请求时掉到 **8.8 / 18.5 t/s**。
**任何性能结论前先查日志确认窗口独占**：测量区间内 `#running-req` ≤ 1 且只有自己那 1 条 HTTP 行。
批测工具（bench_serving、llm_speedtest 等）同样会被污染。

## 🟢 19. 启动脚本的 cwd 陷阱 (spawn worker chdir)

启动脚本若在 **root 外壳**中执行（如 `su -` 后运行），cwd=`/root`；引擎的 multiprocessing spawn 会让 worker
`os.chdir(父进程 cwd)`，降权到普通用户的 worker 无法进入 `/root` → 初始化即崩：
`PermissionError: [Errno 13] Permission denied: '/root'`。
**修复**：启动脚本开头 `cd` 到公共目录（v3/v4 脚本均已内置）。

## ⚪ 20. 构建产物形制（v0.5.20 自建 wheel）

*（v3 sglang 专属，已归档）*

`SGLANG_BUILD_RUST_EXTS=none` 保持纯 Python 包；打包前清 `python/build`（否则 wheel 从 20MB 涨到 76MB）。

---

# v4（FreeToken）专属坑

## 🟡 21. `/health` 在就绪前就返回 200

FreeToken 的 `/health` 在模型加载完成前就返回 200（上游 **#537**），因此 **不能用它判断就绪**——
此时打 `/v1/chat/completions` 会被拒。

**判据改用日志行** `ready to serve`（启动脚本已如此实现，且进程退出会立刻报错）。
加载期间日志里能看到被拒的请求。

## 🟡 22. KV 默认只有 8256 token：>8K prompt 直接 400 / prefill OOM

默认 `auto` 只规划 **8256 token** 的 KV，首个 >8K 的 prompt 直接 `HTTP 400 Bad Request`（上游 #150）。
但**只加 `--num-tokens` 会换来 prefill OOM**——专家缓存先占满、没给 KV 留空间（上游 #401）。

**两个必须同时给**：

```bash
--num-tokens 265216     # ≈ 256K + 3K
--moe-cache-rate 0.40   # 给 KV 留空间
```

## 🟡 23. 内部 rendezvous 端口被占用导致启动失败（已自动绕开）

FreeToken 的内部 torch.distributed rendezvous 默认监听 **API 端口 + 1**，一旦该端口被占用就直接启动失败。
实际部署里这很常见，不只是"反向代理正好在下一个端口"：

- 反代/Nginx 在相邻端口（本项目 8001 就是这种情况）
- 同机起了第二个实例
- 上一个进程没退干净，端口还在 TIME_WAIT / 仍被 LISTEN

**修复**：`patches/04_ft_auto_dist_port.patch` —— `launch_server()` 在**父进程**里解析一次，
从 API 端口+1 起向上找第一个能 bind 的端口，整段占用则退回内核分配的临时端口，并打日志说明。

> ⚠️ 为什么必须在父进程解析：rendezvous 地址要求**各 rank 完全一致**，而 worker 侧的
> `distributed_addr` 是个 property，若各进程自行扫描就会各挑各的、握手失败。
> worker 由 `multiprocessing` spawn 产生、继承父进程环境，因此读到同一个值。

仍需固定端口时，显式设 `FT_DIST_PORT` 即可（它作为起始偏好；被占用时打 WARNING 并顺延）。

## 🟡 24. `--moe-strategy auto` 会静默变成 hybrid（实测慢 27%）

在跑过 `ft bench bw` 的机器上会留下 profile 文件，`auto` 会采纳它并解析为 **hybrid**。
而实跑对照：**offload 62.6–63.6 t/s vs hybrid 46.6 t/s（-27%）**，cpu 策略 35.6 t/s。

根因（上游 **#151 / #436**）：bench 看的是带宽，而实跑瓶颈是**单步同步延迟 + 小批量 CPU GEMV 效率**——
hybrid 下 GPU 73–100%、CPU 仅 1–18%，两边都没饱和。几何守卫（上游 PR #278）也救不了本模型。

**必须显式固定 `--moe-strategy offload`。**

## 🟡 25. `ft ctl cache rebuild` 热调有 wedge 风险

上游 **#526**：缓存重建 OOM 会把服务器卡死且无法回滚。
**改参数一律「停止 + 重启」**——本栈重启只要 ~40 秒，不值得冒热调的险。

## 🟡 26. 升级 FreeToken 会静默丢掉三个补丁

三个补丁都打在 `site-packages` 里，`uv pip install -U` 会覆盖；且 `RECORD` 不会因 patch 更新，
所以 `pip check` 之类看不出异常，直到启动失败或行为异常才发现。

**修复**：`bash scripts/install_freetoken_patches.sh <site-packages>`（幂等，可重复执行）。
`scripts/start_freetoken.sh` 内置自检，缺补丁会**直接拒绝启动**并打印重打命令。

## 🟡 27. 并发上限由 mamba 槽决定，不是 mrr 数值本身

`slots = 4*mrr + max(4, 2*mrr) + 1`：mrr=2 → 12 槽（3 路长上下文）；mrr=4 → 24 槽（6 路）。

「mrr 调到 3 以上没用」是误解——它直接决定并发上限。实测 mrr=4 时显存 44.4G，decode @73K 65.0/65.3 不掉。

## 🟡 28. 多流「突然不吐字」：prefill 整块占住 GPU

FreeToken 调度里 prefill 默认无条件优先，长 prompt 的 prefill 会整块占住 GPU，正在解码的流最长冻结 **17.7s**。

**修复**：`patches/06`（上游 PR #484 重放）+ `--decode-interleave-every 2` → 最长冻结 **7.5s**（-58%）。

⚠️ **N 必须按自己的 prompt 长度定**：43K prompt 只有 6 个 prefill 块，作者在 PR 里推荐的 N=8 永远到不了阈值、完全无效。

---

## 立即停止条件 (Stop conditions)

host OOM / swap 持续增长 / NVIDIA Xid / 不可恢复 kernel fault / 同失败连续两次无新证据。
