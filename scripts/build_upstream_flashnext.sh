#!/bin/bash
# ==========================================
# 构建「官方 sglang v0.5.20 + lk_moe + PLE 补丁」wheel（4090 48GB 用）
# 路线: 取官方 tag v0.5.20 快照 → 依次重放 3 个补丁（不再依赖 Lsglang 分支）
#   01_lk_moe_v0520.patch       lk_moe 集成（CPU experts / NUMA / GPU 常驻层）
#   02_ple_cpu_alloc.patch      PLE n-gram 表直接分配到 CPU（48GB 卡必须）
#   03_resolved_files_v0520.patch  两处需人工裁决的差异（pyproject/modelopt）
# 详见: docs/UPGRADE.md（v3 历史章节）
# 注意: 本脚本属于 **v3（sglang）历史路线**，当前生产为 v4（FreeToken），见 scripts/start_freetoken.sh
# 用法: conda activate <目标环境> && bash scripts/build_upstream_flashnext.sh [构建目录]
# 产物: <构建目录>/wheels/lsglang-1.6.0+flashnext.v0520-py3-none-any.whl
# 构建期依赖: pip install build setuptools-rust setuptools-scm
# ==========================================
set -euo pipefail

BUILD_DIR=${1:-$HOME/build/lsglang-v0520}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="$REPO_ROOT/patches"

# 网络受限环境：GitHub 的 HTTP/2 易断流，强制 HTTP/1.1（并绕开本地代理如有）
GIT="git -c http.version=HTTP/1.1"

echo "==> [1/6] 取官方 sglang v0.5.20 快照（浅克隆，只要这一个 tag）"
mkdir -p "$(dirname "$BUILD_DIR")"
if [ ! -d "$BUILD_DIR/.git" ]; then
    $GIT init -q "$BUILD_DIR"
    cd "$BUILD_DIR"
    git remote add origin https://github.com/sgl-project/sglang 2>/dev/null || true
    $GIT fetch --depth=1 origin tag v0.5.20
else
    cd "$BUILD_DIR"
fi
git checkout -q -B v0520-lkmoe v0.5.20
echo "    base: $(git log -1 --format='%h %s' v0.5.20 | cut -c1-80)"

echo "==> [2/6] 应用 lk_moe 集成补丁（15 文件）"
git apply --check "$P/01_lk_moe_v0520.patch"
git apply "$P/01_lk_moe_v0520.patch"

echo "==> [3/6] 应用 PLE CPU 分配补丁"
git apply "$P/02_ple_cpu_alloc.patch"

echo "==> [4/6] 应用「依赖与模型文件」决议补丁"
git apply "$P/03_resolved_files_v0520.patch"

echo "==> [5/6] 构建 wheel（版本号由 03 号补丁写死为 1.6.0+flashnext.v0520）"
cd python
rm -rf build ./*.egg-info        # 必须清理：残留会让 wheel 混入 build/ 垃圾（体积翻数倍）
SGLANG_BUILD_RUST_EXTS=none \
    python -m build --wheel --no-isolation --outdir "$BUILD_DIR/wheels"

echo "==> [6/6] 完成"
ls -la "$BUILD_DIR"/wheels/*.whl

cat <<'EOF'

──────────────────────────────────────────────
安装（在目标 conda 环境中执行）:
  pip install --no-deps <上面的 wheel>            # 纯 Python 包，无需编译
  pip install "lk_moe==2.4.1"
  pip install "sglang-kernel==0.4.7"             # v0.5.20 启动强校验 >=0.4.7（必装）
  pip install "tilelang==0.1.11"                 # 必须锁 0.1.11，0.1.12 编译不兼容（TROUBLESHOOTING #15）
  pip install "flashinfer-python[cu13]==0.6.18"
  pip install "flash_attn-2.8.4+pr2751-...whl"   # SM89 必需（见 install.sh / 复现文档）

启动: 用本仓库的 scripts/start_lsglang_upstream.sh
      （内置 sudo 提权 + ulimit -l unlimited，PLE pinned 表需要）
──────────────────────────────────────────────
EOF
