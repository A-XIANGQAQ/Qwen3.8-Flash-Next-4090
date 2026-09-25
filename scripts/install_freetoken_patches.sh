#!/bin/bash
# ==========================================
# 给 FreeToken 打三个本地补丁（v4 路线）
#
#   · 装完 freetoken 后跑一次
#   · `uv pip install -U freetoken` 会覆盖 site-packages 导致补丁静默丢失 → 之后需重跑
#     （scripts/start_freetoken.sh 内置自检，缺补丁会直接拒绝启动）
#
# Usage:
#   bash scripts/install_freetoken_patches.sh <site-packages 路径>
#   # 例如 ~/ft-venv/lib/python3.12/site-packages
#
# 幂等：已打过的补丁会跳过，可安全重复执行。
# ==========================================
set -euo pipefail

SITE_PACKAGES="${1:-}"
if [ -z "$SITE_PACKAGES" ]; then
    echo "用法: bash $0 <site-packages 路径>"
    echo "  例: bash $0 ~/ft-venv/lib/python3.12/site-packages"
    exit 1
fi
if [ ! -d "$SITE_PACKAGES/freetoken" ]; then
    echo "❌ $SITE_PACKAGES 下找不到 freetoken/ —— 路径对吗？"
    exit 1
fi

PATCHES_DIR="$(cd "$(dirname "$0")/../patches" && pwd)"
cd "$SITE_PACKAGES"

applied=0
run_patch() {   # $1=补丁文件  $2=标记文件  $3=标记串  $4=patch 层级
    local file="$1" marker_file="$2" marker="$3" level="$4"
    if grep -q "$marker" "$marker_file" 2>/dev/null; then
        echo "⏭️  已应用，跳过: $(basename "$file")"
        return
    fi
    echo "🔧 应用: $(basename "$file")  (-p$level)"
    patch "-p$level" < "$file"
    applied=$((applied + 1))
}

run_patch "$PATCHES_DIR/04_ft_dist_port.patch" \
          "freetoken/server/args.py"    "FT_DIST_PORT"       1
run_patch "$PATCHES_DIR/05_ft_workload_qwen38.patch" \
          "freetoken/moe/benchbw.py"    "qwen3.8-flash-next" 1

# 06 新增了 freetoken/scheduler/interleave.py，用它是否存在作标记
if [ -f "freetoken/scheduler/interleave.py" ]; then
    echo "⏭️  已应用，跳过: 06_ft_decode_interleave.patch"
else
    echo "🔧 应用: 06_ft_decode_interleave.patch  (-p2)"
    patch -p2 < "$PATCHES_DIR/06_ft_decode_interleave.patch"
    applied=$((applied + 1))
fi

echo
if [ "$applied" -eq 0 ]; then
    echo "✅ 三个补丁此前均已应用，无需改动。"
else
    echo "✅ 本次应用了 $applied 个补丁。重启服务生效: bash scripts/start_freetoken.sh"
fi
