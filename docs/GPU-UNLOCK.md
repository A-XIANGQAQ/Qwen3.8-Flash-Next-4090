# GPU 张量速率解锁（gpu-unlock-ada）在本机实测

模块：[`JustAResearcher/gpu-unlock-ada`](https://github.com/JustAResearcher/gpu-unlock-ada) v1.1.1
（commit `4e7fefcc`）。它在 NVIDIA 驱动绑定前运行签名固件 + AD102 payload，改写 SS0/SS1 与
FEAT/WPR 权限位，把「FP16/BF16 输入 + FP32 累加」的张量核发射率从半速提到全速。

## 一句话结论

**解锁成功，但本仓库这套 IQ3_S 量化推理吃不到它**：8K prefill 0.983×、32K 0.993×（判据内无差异）。
指令级 2× 是真实的，收益只出现在非量化的 FP16/BF16 矩阵乘上。

## 卡与环境

| 项 | 值 |
|---|---|
| 卡 | 改装 48 GB RTX 4090（AD102，`10de:2684`），VBIOS `95.02.3C.00.02` |
| 系统 | Ubuntu 26.04.1 / 内核 `7.0.0-38-generic` |
| 驱动 | `610.57.04`（open kernel modules, DKMS） |
| 其他 | CUDA 13.2 编译基准；450 W 功耗上限；未改频率 |

## 解锁证据

`journalctl -k -b`：`gpu_unlock: unlocked`；AD102 权限回读 `valid=1 full=1`。
RM getter `0x20801230`：`fmla32/imla2/imla4` 由 `1/1/1`（半速）→ `0/0/0`（全速）。
Xid 0 条；约 30 分钟满载后 46 ℃；6 组数值用例 `failures=0 max_abs=0`。

## 指令级性能（同一张卡，解锁前后）

| 模式 | stock | 解锁 | 倍数 |
|---|---|---|---|
| `f16/f32`（FP32 累加） | 164.9 TF/s | **328.9** | **1.995×** |
| `bf16/f32`（FP32 累加） | 164.8 | **328.7** | **1.995×** |
| `f16/f16` | 330.8 | 330.8 | 1.000× |
| `s8/s32` / `dp4a` / `ffma` | 659.4 / 164.0 / 78.5 | 657.7 / 163.5 / 78.6 | ≈1.00× |

cuBLAS：FP16 4096³ 154.3 → **216.3（1.40×）**、8192³ 173.6 → 189.6；BF16 4096³ 155.0 → 155.3（两次复现未改善）。

## FP8：不豁免，但应用层看不全

FP8 的累加器必然是 FP32，所以它和解锁覆盖的那条路径同类。三组数据（都标了探针类型）：

| 来源 | 卡 | 探针 | 结果 |
|---|---|---|---|
| 本机 | 改装 48G 4090 | 寄存器 `mma.m16n8k32` e4m3→f32 | 解锁态 **603.1 TF/s** = 标称 660.6 的 **91%**（同批 `f16/f32` 330.6，锁速态 164.9） |
| 社区 | RTX 4080 SUPER | 8192³ E4M3 GEMM | **199.4 → 310.4 TF/s（+55.7%）**，第二次 +49.8%（作者注明 FP8 波动大、不用于判定） |
| 社区 | RTX 4090（驱动 610.57.04） | 功耗侧信道 | 解锁峰值 **346 W vs 313 W**，MMA 与 GEMM 都触发 SW Power Cap，最低频率 2505 vs 2655 MHz |

指令级能到 2×，应用级 GEMM 只到 ~1.5×，差在功耗墙（同样预算分给更多通路 → 降频），不是代码问题。
上游记录：`docs/evidence/rtx4090-610.57.04/`；社区实测：[goldlionren/gpu-unlock-4080s](https://github.com/Goldlionren/gpu-unlock-4080s)。

## 业务负载（本仓库这套栈）

Strata 0.2.2 + engine 0.1.47、IQ3_S、262144 上下文，同一 `run_matrix.py`、同 seed、同 5 个 cell；
解锁侧 4 遍（第 1 遍为冷态，取后 3 遍中位）。原始数据见
[`bench/results/2026-10-25/gpu-unlock/`](../bench/results/2026-10-25/gpu-unlock/summary.csv)。

| cell | prefill stock → 解锁（中位） | 倍数 |
|---|---|---|
| **P8K** | 3518.8 → **3457.8** | **0.983×** |
| P32K | 3916.6 → 3890.3 | 0.993× |
| P1K | 1351.4 → 1193.1 | 0.883×（见"未验证"） |
| D1K decode | 184.8 → 206.6 | 1.118×（抖动 ±12%，不作结论） |
| D32K decode | 164.0 → 154.3 | 0.941×（同上） |

没有收益的原因很直接：这个文件是**混合量化**（解 GGUF 头实测，176.9 G 元素 / 78.09 GiB）——
IQ4_NL 47.4%、IQ2_S 19.0%、IQ3_XXS 16.1%、IQ3_S 9.5%、自定义类型 4.3%（元素占比）。
F32/F16/BF16 张量确实存在（777 个），但只占 **0.43% 的元素**（1.8% 字节，最大 50 MiB），
且集中在 320 维瓶颈支路、MoE 路由门、逐层 embedding —— 不是每 token 的 GEMM。
即便这 0.43% 快 2×，端到端也动不了半个百分点，低于 1–4% 的重复波动。
同一批测试里 `dp4a`、`s8/s32` 纹丝不动即为佐证。
投机解码的 draft 模型也不是纯 BF16：它的清单是 **76.7% Q8_0 + 23.3% BF16**（元素占比），
BF16 部分是 rank-320 的 hyper-connection 支路和 512×2560 的 gate —— 窄矩阵、带宽受限；
占多数的 Q8_0 正是解锁不动的那条 int8 路径，且每个被接受的 draft 仍要跑一次主模型的量化验证前向。
真正可能吃到这 2× 的是 BF16 负载（例如同目录那个 BF16 视觉投影器），本次未测。

## 未验证

- 只有 1 次冷启动；未测热重启、未做多次冷启动重复。
- 未做 matched stock 复测（stock 侧是久跑引擎的单次采样），小 prompt 差异不可归因。
- 未测非量化（FP16/BF16 权重）负载、其它引擎、训练、长期稳定性。

## 复现

```bash
# 一次性：装 dkms + 固件包并持久写入内核命令行
sudo bash gpu-unlock-4090/persist-unlock.sh
# 冷启动后
bash gpu-unlock-4090/post-boot-verify.sh     # 证据 + 速率回读
bash gpu-unlock-4090/phase-e-bench-unlocked.sh
```

## 风险与回滚

模块在 VBIOS/FWSEC 不匹配、卡已被驱动绑定、固件状态不新鲜时**主动拒绝执行**，保持 stock。
热重启可能不解锁（设计如此）。若日志报"固件未停"，**断电，不要 `rmmod`**。

```bash
sudo bash gpu-unlock-4090/persist-unlock.sh --revert        # 关闭（下次启动生效）
sudo apt purge -y gpu-unlock-dkms gpu-unlock-firmware       # 卸载
sudo update-initramfs -u -k all
```

上游已提交脱敏证据：`docs/evidence/rtx4090-48g-modded/`（对应 issue #17）。
