# 排障手册 (Troubleshooting)

全部为实测踩坑记录。

> **适用性标记**（2026-10-13 起引擎为 **v5 = Strata**；v4 = FreeToken 已转为回滚路径）：
> 🟢 **通用**（多代引擎都适用）｜🟡 **v4 适用**（FreeToken）｜🔵 **v5 适用**（Strata）｜⚪ **已归档**（v3 sglang 栈专属，脚本与补丁仍保留可复现）
>
> 编号保持不变，以免其它文档里的交叉引用失效。**🔵 #29 起为 v5 专属坑**（页面末尾）。

## ⚪ 1. 启动 OOM：47.7GB ngram 表被塞进 GPU (Boot OOM: 47.7GB n-gram table to GPU)

症状：启动时 `CUDA out of memory. Tried to allocate 47.69 GiB`（= 51.2B 参数的 ngram 查找表，uint8 存储），堆栈在 `qwen4_exp.py _init_qwen4_exp_layer_extensions → Qwen4ExpPLELayer`。

实测触发组合：
- ❌ 去掉 `--disable-shared-experts-fusion`（fusion 开启）
- ❌ 删除 `--max-total-tokens`（自动内存规划把 ngram 表放 GPU）
- ❌ `NUMA 变量注释 + PREFETCH_WINDOW=0 + POWER_SAVING=0 + max-running-requests 4` 的组合（prefetch0/power0 单独改动是安全的，见 BASELINE）

✅ 安全配置：NUMA 3 变量启用 + 显式 `--max-total-tokens 265216` + `--disable-shared-experts-fusion` + 其余默认。
背景：`ple_offload_embedding` 的自动解析条件是 `is_cuda and dtype==bf16`（overrides.py），NVFP4 模型下 ngram offload 依赖上述组合。

> v4 说明：FreeToken 默认 `--ple-backend disk`（io_uring + O_DIRECT），PLE 表不常驻 GPU，本坑在 v4 上不复现；
> 但 v4 有自己的一对参数陷阱，见 #22。
>
> 🔵 v5 说明：Strata 走 GGUF，PLE 表按 pack 内的形态加载，上述 `--disable-shared-experts-fusion` /
> `--max-total-tokens` 那套旋钮**不存在**，本坑不适用。

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
- 🔵 **v5**：同样靠模板解决，但注意**默认档由模板而非引擎决定**，且 `/v1/models` 会**少报**枚举——见 #33

## ⚪ 8. MTP 投机解码 (MTP speculation)

*（v3/v4 时代的结论，**已被 v5 推翻**）*

- v3：可启动（`--speculative-algorithm NEXTN` 等），模型以 `Qwen4ExpForCausalLMMTP` 加载，accept rate 0.62~0.72，
  **但 decode 无收益**（35.7 vs 无 MTP 36~40 t/s）：CPU 核数是瓶颈
- v4：上游对 qwen4_exp 的 MTP 直接丢弃（下游分支有 +12.2% decode 的报告，未合并）
- **v5：MTP 是当真用的**——`--spec 4`，且**必须显式加 `--batch-mtp`**，不加则批处理静默退化（见 #31）

## 🟢 9. ModelScope revision (ModelScope revision)

TomPython 项目记录的 HF revision（`7b7192...`）在 ModelScope **不存在**（只有 master）。校验靠 `verify_model_index.py`（206 shards）。

## 🟢 10. N-gram 表与 offload (N-gram table & offload)

- 51.2B 参数的 ngram 查找表在 **layer 1 的 PLE**（`layers.1.ple.ple_embedding.ngram_embedding`，128 shards，每 token 只 touch ~100-200KB）
- NVFP4 存储 25.6GB；v3 下 `ple_offload_embedding` 驻 pinned host；**v4 下默认走磁盘后端**（io_uring + O_DIRECT）
- 更优方案（参考 vLLM PR #54070 / DGX Spark 社区）：NVMe mmap 磁盘 offload（省内存，代价 ~8%）

## 🟢 11. 温度漂移与测速 (Thermal drift in measurements)

同配置前后重测 decode 37.8 → 40.4 t/s（±7% 漂移，CPU 满载发热降频）。**A/B 对比必须在同窗口内做**，跨小时对比会误判参数效果。

## 🟢 12. pgrep 自匹配陷阱 (pgrep self-match trap)

`pgrep -f '<引擎进程串>'` 会匹配命令行含该字符串的 shell 自身（自杀 → exit 144）。`pkill -f` 同理。
稳妥写法：`ps -eo pid,args | grep '[f]t-venv/bin/ft serve'` 拿真实 PID 再 `kill`。
🔵 **v5 没这个问题**：引擎二进制名就是 `strata`，`pgrep -x strata` 精确匹配不会自匹配（见 #36）。

## ⚪ 13. PLE 表 GPU 瞬态分配 OOM (Upstream PLE transient GPU alloc)

*（v3 sglang 专属，已归档）*

上游 Flash-Next 实现里 `Qwen4ExpPLELayer` 会**先在 GPU 构造整张 PLE 表再搬到 pinned 内存**——48GB 卡必炸；
且该 GPU 占位随后即被 `del`，**纯浪费**。修复：`patches/02_ple_cpu_alloc.patch`。
（上游已用 meta device 方案在 PR #39928 修复，但晚于 v0.5.20。）

## 🟡 14. 锁页内存与 memlock 限制 (memlock)

锁页内存分配需要放开 `ulimit -l`（系统默认常仅 8MB）。v4 启动脚本内置了 `ulimit -l unlimited` + 自动 sudo 提权。

> ⚠️ **但 2026-10-02 实测：这两项在本机都不是必需的。** 非 root 启动成功 3 次，
> 运行中 `VmLck: 0 kB`——offload 走 PINNED，mlock 只服务 CPU-executor 路径，本栈用不到。
> ⇒ 脚本里那两行属**保守默认**，不要据此宣称"启动必须 root"。
>
> v3 与 v4 的原因不同：v3 是 PLE 表真需 47.7GiB 锁页内存；
> v4 下 PLE 走磁盘后端，`ulimit -l` 名义上服务专家 host banks 的 mlock 常驻，但实测未生效也不影响。
> **v5（Strata）完全不涉及这套。**

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
🔵 **v5**：Strata 的 `serve/server.py` 同理——启动器应先 `cd` 到源码目录再起。

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

**这不该成为部署约束**——rendezvous 是本实现的内部细节，占用它的可能是任何东西：相邻端口上跑着
反向代理或别的服务、上一个进程没退干净（端口还在 TIME_WAIT 状态）。启动失败完全没必要，
用户也不该被迫去查"到底谁占了 API 端口+1"。

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

# v5（Strata）专属坑

## 🔵 29. 切换引擎时 `cudaMalloc` OOM：停旧 ≠ 显存已释放

`ft_stop.sh` 打印"干净停止"时，`nvidia-smi` 仍显示 **47,712 MiB 被占**；立刻起 Strata 直接炸：

```
cudaMalloc(1485606400) for the weight arena failed (out of memory): 489 MiB of 48533 MiB VRAM free
```

**判据是显存降到 <1000 MiB，不是进程消失**：

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 轮询到 <1000 再起下一个
```

（2026-10-13 因漏这步，业务被 nginx 502 了 5 分钟。）

## 🔵 30. `--kv-resident 0` 必 OOM：KV 流式不是"优化项"

KV 全放显存会报 `native dense upload blk.27.attn_output.weight: out of memory`。
48GB 卡上 4×256K 的**唯一可行解**就是 KV 流式到宿主内存（`--kv-resident 32768`）。
实测 `PCIe probe: 25.3 GB/s host→device`。

⚠️ Strata 的 KV 只有 `fp16 | int8 | k8v4 | q4_0`，**没有 fp8**（fp8 仅用于 PLE 表）。
`kv_tok = 13 × cell`：int8 = 1056 B/cell、**k8v4 = 816**、q4_0 = 576
（q4_0 长文困惑度 +8%(1K) / +12%(8K)，256K 下不可用）。

## 🔵 31. 不加 `--batch-mtp`：MTP 收益静默消失

批处理会退化成"没有 MTP 的那套行为"：实测 4 路聚合仅 **131 t/s**（加后 137）、2 路 103–112（加后 123–125）。
**不报错、不警告，只是变慢**——上游 `docs/BATCHING.md` 原文如此。

## 🔵 32. `setup.sh` 每次都会覆盖 chat 模板

Strata 每次 `setup` 都从 GGUF **重新解出** `<pack>/tokenizer/chat_template.jinja`，手工替换必被覆盖。
后果不是报错而是**静默换模板**：同一 prompt 渲染 **2378（原装） vs 2336（froggeric）** token，
且**默认思考档随之改变**（见 #33）。

**修复**：把"应用模板"挂在**启动器**上（幂等、fail-open），并且**不要用 `./setup.sh` 无参起服务**——
它会自己拉起模型并绕开守卫。

## 🔵 33. 默认思考档由 chat 模板决定，不是引擎

上游文档说"默认 high"，那说的是**原装模板**的行为。本站用 froggeric v22.5，第 17 行写死
`_default_reasoning_effort = 'medium'` ⇒ 实测"不指定"与显式 `medium` 的思考长度**逐字吻合**（273 字符），
`high` 是 451。**⇒ 换到 Strata 后默认档没有变。**

另：`/v1/models` advertise 的枚举会**少于**实际支持——启动日志里是
`supported=['high','low','max','medium','minimal','none','xhigh'] default=medium`。**别只信 `/v1/models`。**

## 🔵 34. `--batch N` 不是 `n_ctx ÷ N`

Strata 的 `--batch N` 是**每 slot 独立 session**（VRAM 与主 session 同量，≈0.86 GiB/槽），
**不是** llama.cpp 那套切分。按 llama.cpp 直觉去算 KV 会低估显存好几倍。

## 🔵 35. ESP 破限默认对 unsloth 家族关闭

上游只对 `family in ("qwen","coder")` 开放实验性投影，Unsloth 家族被 left off。
但 `UNSLOTH_Q4.md` 原文是 **"not tested"**（不是"不兼容"），且 cvec 实现
（`src/program/generate.cpp:672`）与量化无关 ⇒ 对 `setup.py` 打一处白名单补丁（备份 `setup.py.orig-esp`）后
ESP 在 UD-Q4_K_XL 上实测生效（强探针：ESP 开 0/14 拒答、关 14/14）。

## 🔵 36. 引擎进程名就叫 `strata`，可以放心用 `pgrep -x`

与 FreeToken 不同，v5 的引擎二进制名就是 `strata`，**精确匹配不会自匹配**（对照 #12 的自匹配陷阱）：

```bash
pgrep -x strata                                # 引擎
ps -eo pid,args | grep '[s]erve/server.py'     # serve 层（方括号写法避免自匹配）
```

## 🔵 37. `400 this server was started without the vision encoder`

发图直接被 serve 层挡回这个 400（`serve/server.py` 在 `self.vision is None` 时抛）。

**不是坏了，是没装**：Strata 的图像能力要三件东西同时在位——mmproj 文件 + `engine/strata-vision` 二进制 +
配置里的 `vision` 段。而 setup **默认不装**：交互式会问，非交互（`--yes`）下按默认「不要」走。
本站 2026-10-13 之前一直如此（**v4 时代也没有 mmproj**，不是换引擎弄丢的）。

⚠️ 别误判成"模型不支持图"：`MODELS["UD-Q4_K_XL"]` 的 `vision` 是 **`True`**（允许装，只标了 untested）。

装法与代价见 [STRATA.md §5.3](STRATA.md)。三条要点：

- **手工装，别重跑 `setup.sh`**——`carry_over()` 会丢掉手加到 `args` 的引擎选项
- 配置里补上 `vision` 段后，serve 层会自己加 `--vision --vram-reserve-mib 700`，不必改 `args`
- **必须重启**才生效（编码器在引擎之前预热，引擎据此规划显存）

## 🔵 38. `400 the image could not be read: the vision encoder stopped`（装了图却崩）

**与 #37 不同**：#37 是根本没装，本条的编码器**装好了、能启动、还预热成功**，一遇上真请求就崩，然后被反复重启。

引擎日志里的真凶：

```
strata-vision: model files loaded in 0.8 s
strata-vision: warmed up at 1024 image tokens in 0.3 s
CUDA error: out of memory
  in function alloc at .../ggml-cuda.cu:588    cuMemCreate(&handle, reserve_size, &prop, 0)
```

**根因：显存预算不足，不是编码器坏了。** serve 层在「配置有 `vision` 段但 `args` 里没有 `--vision`」时
会自动补 `--vram-reserve-mib 700`——那是给小卡设计的默认值。48GB 卡本身已占满时，引擎会把显存吃到
**只剩 329 MiB**；编码器一旦重启就需要 ~1.5 GB ⇒ OOM ⇒ serve 层再重启它 ⇒ 死循环。

**两个判据行（引擎日志）**：

| 行 | 含义 |
|---|---|
| `strata serve: N MiB of VRAM free with everything loaded` | **N 必须 ≥ ~1600**；本站默认时只有 329 |
| `strata verify: no VRAM for a new window graph ... (a larger --vram-reserve-mib keeps more)` | 出现即说明整卡已被压死，引擎自己的 CUDA graph 都保不住 |

**编码器的需求与图大小无关**：实测 360×240 小图与 1398×1167 真截图**峰值都是 1,481 MiB**
（预热即按 `max_tokens` 顶格分配）⇒ 留够 ~1.5 GB 是硬要求。

**修法**：把 `--vision --vram-reserve-mib 2560` **显式写进配置的 `args`**（写了就不会被自动补 700），重启。
本站代价：GPU 专家缓存 34.09 → **32.28 GiB**，但文本速度在噪声内没变（TTFT 0.19 s、纯解码 124 t/s）。

---

## 🔴 39. 升级引擎后行为诡异：三个组件必须是同一个版本

0.1.45 的 `setup.py` 顶部写死了 `MIN_ENGINE = (0, 1, 45)`，`build_engine()` 里的 `stale_binary()` 还会
对比「二进制里的版本」与「源码版本」，不一致就拒绝安装。所以**不能只换 `serve/server.py`，也不能只换引擎**。

- 症状形态：server 起来了、`/v1/models` 也 200，但某些能力（`--parallel`、新的 `RUNNING` 行解析）
  静默降级——因为 0.1.45 的 server 会按「引擎太老」处理，不报错，只少给功能。
- 判据：`engine/BUILD.json` 的 `version`、`engine/strata --version`、`serve/server.py` 里
  `PARALLEL_MAX`、`setup.py` 里 `MIN_ENGINE` 四者应当指向同一个 tag。
- 修法：整套换。见 [OPERATIONS.md](OPERATIONS.md) §1c。

## 🔴 40. 别让 `update.sh` / `setup.py` 碰你的模型分片

两条都是上游已知问题，落在我们这种「自己建、自己调参数」的用法上代价很大：

1. **`carry_over()` 不合并手工加进 `args` 的引擎选项**（`setup.py` 自己的注释就写着
   "Engine options added by hand to `args` are not merged (setup chooses those)"）。我们 config 里
   有 12 个手调参数（`--prefill auto:32768`、`--kv-resident 32768`、`--vram-reserve-mib 2560`、
   cvec 那一组……），跑一次 setup 就被它自己的选择覆盖。
2. **#2902（open）**：setup 的 SHA-256 校验在大分片（43–50 GB）上会算错，**并且把「校验失败」的文件删掉**。
   实际字节是对的（`sha256sum` / `certutil` 与 Hub 的 `lfs.oid` 一致）。重下 50 GB 只是代价之一。

⇒ 升级只调 `build_engine()`（`scripts/build_engine_upgrade.py` 就是这个用途），绕开下载、校验、写 config 的全过程。

## 🟡 41. 想开 `STRATA_HC_Q8=1`？先看 #2997

上游 0.1.44 起会**由 setup** 给 Unsloth 文件写 `STRATA_HC_Q8=1`（hyper-connection 从 Q8_0 字节直读，
KL 0.0562→0.0526、每窗口约 −2 ms）。它是 setup 写的，**不是引擎默认**——我们没跑 setup，所以现在没开。

而且开放 PR **#2997 正在修这条路径的 `q8_inject` guard**。在它合入之前开这个开关，等于用一条
上游自己都还在修的代码路径。⇒ 本站结论：**暂不启用**；要试就单变量试，并且比对输出而不是只看速度。

## 🔵 42. 停止脚本把自己的 shell 杀了（`[s]erve` 括号技巧的第二次翻车）

2026-10-18 实测：把匹配式从 `serve/server.py` 放宽成 `serve[^ ]*/server\.py`（为了兼容 A/B 期间的
`serve-0.1.43.1/server.py`）之后，**调用方自己的命令行**里只要还出现一串 `serve/server.py`（比如同一条
bash 命令里另有一个 `grep serve/server.py`），`ps | awk` 就会匹配到那个 bash 并把它 kill 掉——
`[s]erve` 的括号技巧只防「模式串本身出现在自己的 cmdline 里」，防不住**别处出现的同一段文本**。

**修法**：再用进程名限定一次，只认真正的 python 进程：

```bash
ps -eo pid=,comm=,args= | awk -v me="$$" '$1 != me && $2 ~ /^python/ && /[s]erve[^ ]*\/server\.py/ {print $1}'
```

判据：`pgrep -x strata`（引擎二进制名精确匹配，安全）+ 上面这条（只杀 python）。**别再按纯命令行文本匹配。**

---

## 🔴 43. 第三方 GGUF 起不来：`--ple-gguf` 指错了分片

`iq_pack.py` 把 28.8 GB 的 PLE 表（`per_layer_token_embd.weight`）**留在 GGUF 里**
（`NOT_IN_PACK = {"per_layer_token_embd.weight"}`），引擎靠 `--ple-gguf` 去那个文件里读。

**它不一定在 shard1。** DavidAU 那份 IQ3_S 里：

| 分片 | 大小 | 张量数 | PLE |
|---|---:|---:|---|
| `-00001-of-00002` | 55.06 GB | 1223 | ❌ 没有 |
| `-00002-of-00002` | 28.80 GB | **1** | ✅ `shape [160, 320001536]` |

判据（只读索引，不读权重，几秒）：

```bash
.venv/bin/python dsh_workspace/strata/find_ple_shard.py <shard1> <shard2>
```

写错分片的症状是启动就失败（不是慢，也不是 OOM）。⇒ `--ple-gguf` 指到含 PLE 的那一片。

## 🔴 44. 给第三方/微调 GGUF 建 pack：只调 `iq_pack.py`，别跑 setup

```bash
.venv/bin/python tools/iq_pack.py --gguf <shard1> --out <pack> --compat-bf16
```

- **必须等两片都下完**（缺片直接报错，不会悄悄跳过）。
- `--compat-bf16` 的转换量取决于这个 quant 自己怎么存小投影：DavidAU 的 IQ3_S 实测
  **`compat-bf16: 0 tensors, 0.00 GiB`**（本来就是 BF16/F16）；Unsloth 的 UD-Q4_K_XL 则需要转。
  **0 个张量意味着这一层没有额外量化损失**，是好消息不是漏做。
- 打包会从 GGUF 导出**它自己的 tokenizer 与 chat 模板**到 `<pack>/tokenizer/`。
- 别用 setup 主流程：`carry_over()` 会丢手调参数，且 setup 的 SHA-256 校验在大分片上有 bug（#2902）。

## 🔴 45. 带自带模板的模型被套上通用模板（能力被删掉）

我们的旧 `run_strata.sh` 每次启动都会把 froggeric 修复版模板拷进"第一个 pack 的 tokenizer"。
对 UD-Q4_K_XL 是必要的（原装模板有 bug）；但对 **DavidAU 这类自带 22 万字符模板、27 推理 + 27 指令模式
+ AI Constructs 的模型，这等于把它的核心能力删掉**，而且**没有任何报错**——只是"变笨了"。

判据：

```bash
wc -c <pack>/tokenizer/chat_template.jinja    # 该模型的应是 ~221835，不是 froggeric 的那份
```

修法：**给这类模型用不带守卫的启动器**（本站 `run_strata.sh` 已去掉守卫，
备份在 `run_strata.sh.bak-undq4-20261018`，回滚 UD-Q4_K_XL 时要一起恢复）。

## 🔵 46. 并行下载：完成判据是 `.done.json`，不是文件大小

预分配（`truncate`）过的文件**缺块也显示满大小**，用 `stat`/`ls` 判断"下完了"会直接拿坏文件去打包。

- 判据：`<file>.done.json` 存在（它是 sha256 校验通过后才写出来的）
- 未完成时是 `<file>.progress.json`（记录已完成块，可续传）
- 实测吞吐（HF + 1× 代理）：**6 路 ≈ 9 MB/s，12 路 ≈ 16 MB/s**；不是线路封顶就别急着降并发
- SSL 偶发 `UNEXPECTED_EOF_WHILE_READING` 会让个别块失败；有 `.progress.json` 就只是续传，不是重下

---

## 🔴 47. 用 `systemd-run` 跑"起完就退出"的启动脚本：模型被单元回收连带杀掉

`start_strata_kill_gui.sh` 的结构是 **起服务 → 等 ready → 打印完成 → 退出**。
把它直接当 `systemd-run` 的主进程时，脚本一退出，systemd 认为单元完成，**回收整个 cgroup**，
刚起来的 server/engine 被一起 SIGTERM：

```
14:15:48 ready: http://127.0.0.1:8000/v1 ...
14:15:54 strata-prod.service: Consumed 40.9s CPU time        <- 单元退出
[strata] stopping (Ctrl+C again to end the engine at once) ... <- 模型被停
```

**修法**：给单元加 `RemainAfterExit=yes`，让单元在"脚本已退出"之后仍保持 active，cgroup 不被回收：

```bash
systemd-run --user --unit=strata-prod --collect \
  --property=RemainAfterExit=yes \
  --setenv=STRATA_NO_SUDO=1 \
  --working-directory=~/dsh_workspace/strata/Strata \
  bash "/path/to/start_strata_kill_gui.sh"
# 之后：systemctl --user is-active strata-prod  -> active (exited)
#       停止照样用 systemctl --user stop strata-prod（会把里面的 server/engine 一起收掉）
```

判据：`systemctl --user show strata-prod -p SubState --value` 应为 `exited` 而 `is-active` 仍为 `active`；
脚本退出 30 秒后 `curl :8000/v1/models` 仍 200。

`STRATA_NO_SUDO=1` 是脚本 2026-10-18 新增的无提权分支（当前用户已是目标用户、lightdm 已关时用）；
你自己终端里直接跑仍然走 sudo 路径，行为不变。

---

## 🔴 48. 只发 `{REASON:xxx}` 一个标签：模型回你一句「你好」

模板会把标签**从 user 文本里整段删掉**。如果一条消息**只有**标签，删完 user 回合就是空的：

```
<|im_start|>user
<|im_end|>            ← 一个字都没有
```

模型手里握着上万字的关系/人格提示，面前却是空提问，于是去揣测「用户发了条空消息」再打招呼。
2026-10-18 实测：`{REASON:help}` 单独发 → 2,288 prompt tokens，回复「你好」；
`{REASON:help} 显示帮助菜单` → 2,291 tokens，直接吐出帮助菜单。

⇒ **标签后面必须跟真正的问题。**（自动选择不改变这一点：它只挑模式，不替你提问。）

## 🔴 49. 自定义/合并的 chat 模板被 `setup.sh` 冲掉

`setup.sh` 每次都会从 GGUF **重新导出** `tokenizer.chat_template.jinja`；手工替换的模板（froggeric、
或本仓库的[合并模板](../patches/08_qwen38_froggeric_v225_with_dau_modes.jinja)）**必被覆盖**，而且不报错，
只表现为"模式突然不生效了"。

判据：

```bash
wc -c <pack>/tokenizer/chat_template.jinja
#   v2 合并模板 248855 | v1 合并模板 246686 | froggeric 28234 | 原装 226745
curl -s 127.0.0.1:8000/props | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["chat_template"]))'
#   v2 243052 | v1 241214
```

恢复：

```bash
cp qwen4090-repo/patches/08_qwen38_froggeric_v225_with_dau_modes.jinja \
   strata-data/packs/<模型>/tokenizer/chat_template.jinja
# 然后重启（约 40 秒就绪）
```

---

## 🔴 50. 怎么判断"自动选模式"到底有没有生效

症状通常不是报错，而是"感觉怪"：该用 coder 的时候答得很泛，或者随便聊两句突然被套上豪斯医生的人格。

判据按顺序：

```bash
# 1. 模板对不对（不是原装、不是 froggeric）
wc -c <pack>/tokenizer/chat_template.jinja          # 248855 = v2
curl -s 127.0.0.1:8000/props | grep -c 'dau-modes-v2'   # 1

# 2. 规则对不对（离线，渲染真实模板，不碰 GPU）
python3 qwen4090-repo/patches/08_auto_check.py --set dev+val
#   期望 dev 95.9% / val 90.0% / 错选 0

# 3. 线上单条
python3 qwen4090-repo/patches/08_auto_check.py --case "讲个笑话"
#   输出 mode : comedy，scores 里 comedy 2、其余 0

# 4. 线上真实请求（prompt_tokens 是硬证据）
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"讲个笑话"}],"max_tokens":1}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["usage"]["prompt_tokens"])'
#   约 2847 = comedy 正文进去了；11 = 完全没选模式
```

常见误判：

| 现象 | 真相 |
|---|---|
| 问技术问题没进 coder | 可能是**故意的弃权**（`top1≥2` 且领先第二名才算命中）。看 `--case` 的 scores，别猜 |
| 一句话被套上 persona | 词表里某个强信号词误伤（如 `代码` 出现在闲聊里）。给 `--explain` 看是哪个词，再决定删/降权 |
| 换了词表却没变化 | 忘了重新生成模板或没重启；`/props` 里的版本串能分辨 |
| 同一会话里风格突然变 | 自动判定看**整段对话**，中途继续追加内容会累计分数；换模式请新开会话或敲 `{REASON:xxx}` |

---

## 立即停止条件 (Stop conditions)

host OOM / swap 持续增长 / NVIDIA Xid / 不可恢复 kernel fault / 同失败连续两次无新证据。
