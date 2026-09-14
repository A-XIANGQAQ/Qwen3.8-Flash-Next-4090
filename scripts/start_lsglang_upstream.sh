#!/bin/bash
# ==========================================
# Qwen3.8-Flash-Next-NVFP4 @ single RTX 4090 48GB (SM89)
# v2 启动脚本：上游 Flash-Next + lk_moe + PLE CPU 补丁（见 docs/UPGRADE_UPSTREAM_FLASHNEXT.md）
# 与 v1 脚本的差异：
#   - --ple-offload-embedding 显式开启（新版自动解析可能因 dtype 判定不启用）
#   - 需要 root 放开 memlock（PLE pinned 表 47.7GiB）→ 脚本自动 sudo 提权
#   - 模型进程仍以你的普通用户身份运行（不是 root）
# Usage: 修改下方配置变量后直接运行（会提示一次 sudo 密码）
# ==========================================
set -e

if [ "$(id -u)" -ne 0 ]; then exec sudo -E bash "$0" "$@"; fi
ulimit -l unlimited
RUN_USER="${SUDO_USER:-$(id -un)}"   # 以调用者身份跑模型

# ---------- 配置区（按需修改） ----------
MODEL_DIR=/path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4   # 模型目录（ModelScope 下载）
SERVED_NAME=Qwen3.8-Flash-Next                          # 对外模型名
PORT=8000                                               # 服务端口
LK_CORES=36                                             # CPU 物理核数 - 2（38 核 → 36）
LOG=/tmp/lsglang.log                                    # 日志路径
SGLANG_BIN=/path/to/conda/envs/lsglang-next/bin/sglang  # v2 环境（build_upstream_flashnext.sh 产出）
# -------------------------------------------------

if lsof -i :"$PORT" -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用，跳过启动"; exit 0
fi

: > "$LOG"; chown "$RUN_USER" "$LOG"
runuser -u "$RUN_USER" -- env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    LVLLM_MOE_NUMA_ENABLED=1 \
    LVLLM_EMBEDDING_NUMA_ENABLED=1 \
    LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
    LK_THREAD_BINDING=CPU_CORE \
    LK_THREADS="$LK_CORES" \
    LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 \
    OMP_NUM_THREADS=1 \
    LVLLM_GPU_PREFETCH_WINDOW=0 \
    LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
    LK_POWER_SAVING=0 \
    SGLANG_WARMUP_TIMEOUT=3600 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
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
        --ple-offload-embedding \
        --cuda-graph-backend-prefill disabled \
        --disable-shared-experts-fusion \
        --reasoning-parser auto \
        --tool-call-parser qwen3_coder \
        > "$LOG" 2>&1 &

echo "Lsglang (upstream v2) started as $RUN_USER, log: $LOG"
echo "就绪标志: The server is fired up and ready to roll!（首次 ~4-5 分钟）"
