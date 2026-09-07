#!/bin/bash
# ==========================================
# Qwen3.8-Flash-Next-NVFP4 @ single RTX 4090 48GB (SM89) via Lsglang
# 单卡 4090 48GB 部署启动脚本（实测定稿参数，2026-09）
# Usage: 修改下方配置变量后直接运行
# ==========================================
set -e

# ---------- 配置区（按需修改） ----------
MODEL_DIR=/path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4   # 模型目录（ModelScope 下载）
SERVED_NAME=Qwen3.8-Flash-Next-NVFP4                    # 对外模型名
PORT=8000                                               # 服务端口
LK_CORES=36                                             # CPU 物理核数 - 2（本机 38 核 → 36）
LOG=/tmp/lsglang.log                                    # 日志路径
SGLANG_BIN=/path/to/conda/envs/lsglang/bin/sglang       # sglang 可执行文件（install.sh 安装的 env）
# -------------------------------------------------

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# 多路/NUMA 机器启用以下 3 项（实测：启用与否改变 lk_moe embedding 加载路径，
# 单路机器上移除需与 prefetch/power 组合一起验证，见 docs/TROUBLESHOOTING.md）
export LVLLM_MOE_NUMA_ENABLED=1
export LVLLM_EMBEDDING_NUMA_ENABLED=1
export LVLLM_ENABLE_NUMA_INTERLEAVE=1
export LK_THREAD_BINDING=CPU_CORE
export LK_THREADS=${LK_CORES}          # 物理核预留 2 核给系统
export LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5   # 6 层常驻 GPU（7 层保不住 256K，见 BASELINE）
export OMP_NUM_THREADS=1
export LVLLM_GPU_PREFETCH_WINDOW=0      # 关预取（实测单独关闭安全）
export LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024
export LK_POWER_SAVING=0                # 关省电
export SGLANG_WARMUP_TIMEOUT=3600
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nohup "$SGLANG_BIN" serve \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-running-requests 2 \
    --chunked-prefill-size 8192 \
    --max-total-tokens 265216 \
    --mem-fraction-static 0.95 \
    --context-length 262144 \
    --cuda-graph-backend-prefill disabled \
    --disable-shared-experts-fusion \
    --reasoning-parser auto \
    --tool-call-parser qwen3_coder \
    > "$LOG" 2>&1 &

echo "Lsglang started, PID $!, log: $LOG"
