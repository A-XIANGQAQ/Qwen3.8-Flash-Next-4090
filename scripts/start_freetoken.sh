#!/bin/bash
# ==========================================
# Qwen3.8-Flash-Next-NVFP4 @ single RTX 4090 48GB (SM89)
# v4 启动脚本：FreeToken 0.1.3 + 三个本地补丁（见 docs/UPGRADE.md）
#
# 关键参数（都有实测理由，改前先读 docs/BASELINE.md）：
#   --moe-strategy offload  必须显式固定！跑过 `ft bench bw` 的机器上已落了 profile，
#                           auto 会静默升级为 hybrid，而实测 hybrid 比 offload 慢 27%
#                           （decode 46.6 vs 62.6 t/s）——上游 #151/#436 描述的正是这个现象
#   --num-tokens 265216     默认 auto 只给 8256 token KV，>8K 的 prompt 直接 400（上游 #150）
#   --moe-cache-rate 0.40   给 KV 留空间。这两个必须同时给，否则专家缓存先占满 → prefill OOM（上游 #401）
#   --decode-interleave-every 2
#                           每 2 个 prefill 步插一个 decode 步（本地补丁 = 上游 PR #484）。
#                           N 要按自己的 prompt 长度定：43K prompt 只有 6 个 prefill 块，
#                           作者推荐的 N=8 永远到不了阈值、完全无效。实测最长冻结 17.7s → 7.5s
#
# 前置：
#   · 需要 root 放开 memlock（专家 host banks 走 mlock 常驻）→ 脚本自动 sudo 提权，模型仍以普通用户跑
#   · cd 到公共目录：spawn 出的 worker 会 chdir 到父进程 cwd，脚本若在 root 外壳里跑
#     （cwd=/root）子进程会 PermissionError 崩溃
#   · 就绪判据是日志行 "ready to serve"，不是 /health —— /health 在加载完成前就返回 200（上游 #537）
#
# Usage: 修改下方配置变量后直接运行（会提示一次 sudo 密码）
# ==========================================
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then exec sudo -E bash "$0" "$@"; fi
ulimit -l unlimited                  # 专家 host banks 的 mlock 需要（默认 ulimit -l 常仅 8MB）
cd /tmp                              # spawn worker 的 chdir 目标，必须在所有用户下可访问
RUN_USER="${SUDO_USER:-$(id -un)}"   # 以调用者身份跑模型

# ---------- 配置区（按需修改） ----------
MODEL_DIR=/path/to/RadixArk--Qwen3.8-Flash-Next-NVFP4   # 模型目录（ModelScope 下载）
SERVED_NAME=Qwen3.8-Flash-Next                          # 对外模型名
PORT=8000                                               # API 端口（客户端零改动）
DIST_PORT=8002                                          # 内部 rendezvous；默认 = API 端口+1，会撞 nginx 的 8001
LOG=/tmp/ft.log                                         # 日志路径
FT_BIN=/path/to/ft-venv/bin/ft                          # FreeToken 可执行（uv venv 产出）
SITE_PACKAGES=/path/to/ft-venv/lib/python3.12/site-packages   # 补丁自检与重打用
STOP_GUI=auto                                           # auto = 检测到 lightdm 就停；off = 不停
# -------------------------------------------------

# ---------- 补丁自检 ----------
# FreeToken 升级（uv pip install -U）会覆盖 site-packages，三个补丁会静默丢失。
# 缺任何一个都直接退出，而不是带着半套补丁启动。
missing=0
grep -q "FT_DIST_PORT" "$SITE_PACKAGES/freetoken/server/args.py" 2>/dev/null \
    || { echo "⚠️  补丁 04（FT_DIST_PORT）未应用"; missing=1; }
grep -q "qwen3.8-flash-next" "$SITE_PACKAGES/freetoken/moe/benchbw.py" 2>/dev/null \
    || { echo "⚠️  补丁 05（qwen3.8 workload）未应用"; missing=1; }
[ -f "$SITE_PACKAGES/freetoken/scheduler/interleave.py" ] \
    || { echo "⚠️  补丁 06（decode-interleave）未应用"; missing=1; }
if [ "$missing" -ne 0 ]; then
    cat <<EOF
    FreeToken 可能被升级过，补丁已丢失。重打：
      cd "$SITE_PACKAGES"
      patch -p1 < patches/04_ft_dist_port.patch
      patch -p1 < patches/05_ft_workload_qwen38.patch
      patch -p2 < patches/06_ft_decode_interleave.patch
EOF
    exit 1
fi

# ---------- 端口检查 ----------
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "⚠️  端口 $PORT 已被占用（若在跑 sglang: pkill -f 'sglang serve'）"; exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":$DIST_PORT "; then
    echo "⚠️  rendezvous 端口 $DIST_PORT 已被占用"; exit 1
fi

# 126GiB 权重 + 专家 host banks 峰值 ~200GB 主机内存：先丢页缓存再起
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

: > "$LOG"; chown "$RUN_USER" "$LOG"
echo "🚀 启动 FreeToken（API $PORT / rendezvous $DIST_PORT / nginx 8001 不受影响）..."
runuser -u "$RUN_USER" -- env \
    PATH="/usr/local/cuda/bin:/usr/bin:/bin" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FT_DIST_PORT="$DIST_PORT" \
    FREETOKEN_MAMBA_SSM_DTYPE=bfloat16 \
    nohup "$FT_BIN" serve \
        --model "$MODEL_DIR" \
        --served-model-name "$SERVED_NAME" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-running-requests 4 \
        --num-tokens 265216 \
        --moe-cache-rate 0.40 \
        --moe-strategy offload \
        --decode-interleave-every 2 \
        --enable-cache-report \
        > "$LOG" 2>&1 &

echo "FreeToken started as $RUN_USER, log: $LOG"

echo "⏳ 等待模型就绪（约 40 秒；进程退出即报错）..."
for i in $(seq 1 240); do
    if grep -q "ready to serve" "$LOG" 2>/dev/null; then echo "✅ 模型已就绪！"; break; fi
    if ! pgrep -f "$FT_BIN serve" >/dev/null 2>&1; then
        echo "❌ 启动失败：服务进程已退出。日志尾部："; tail -30 "$LOG"; exit 1
    fi
    sleep 5
done
grep -q "ready to serve" "$LOG" 2>/dev/null || echo "⚠️ 尚未就绪，继续观察日志: tail -f $LOG"

# ---------- 可选：关闭 GUI ----------
# 128GB 权重 + 运行时需 ~200GB RAM，桌面环境会挤内存（见 docs/OPERATIONS.md §7）
if [ "$STOP_GUI" != "off" ] && systemctl is-active lightdm >/dev/null 2>&1; then
    echo "🖥️  正在关闭 GUI (LightDM)..."; systemctl stop lightdm
fi
