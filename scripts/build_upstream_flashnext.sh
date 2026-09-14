#!/bin/bash
# ==========================================
# 构建「上游 Flash-Next + lk_moe」Lsglang wheel（4090 48GB 用）
# 路线: Lsglang 0.5.19-lkmoe 分支 + 合并上游 PR #37500 + PLE CPU 分配补丁
# 详见: docs/UPGRADE_UPSTREAM_FLASHNEXT.md
# 用法: conda activate <目标环境> && bash scripts/build_upstream_flashnext.sh [构建目录]
# 产物: <构建目录>/wheels/lsglang-1.5.0+flashnext.merge2-py3-none-any.whl
# ==========================================
set -euo pipefail

BUILD_DIR=${1:-$HOME/build/lsglang-next}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$REPO_ROOT/patches/01_ple_cpu_alloc.patch"
VERSION="1.5.0+flashnext.merge2"

# 部分网络环境下 GitHub 的 HTTP/2 会断流，强制 HTTP/1.1
GIT="git -c http.version=HTTP/1.1"

echo "==> [1/6] 克隆 Lsglang 0.5.19-lkmoe（如已存在则跳过）"
mkdir -p "$(dirname "$BUILD_DIR")"
if [ ! -d "$BUILD_DIR/.git" ]; then
    $GIT clone --depth 100 -b 0.5.19-lkmoe https://github.com/guqiong96/Lsglang "$BUILD_DIR"
fi
cd "$BUILD_DIR"

echo "==> [2/6] 取上游 PR #37500（Flash-Next 支持）"
$GIT fetch --depth 300 https://github.com/sgl-project/sglang pull/37500/head:pr37500

echo "==> [3/6] 合并（冲突全部取上游侧；上游版本已含 fork 同类修复）"
git -c user.name=build -c user.email=build@localhost merge --no-commit --no-ff pr37500 || true
CONFLICTS=$(git diff --name-only --diff-filter=U)
if [ -n "$CONFLICTS" ]; then
    echo "$CONFLICTS" | while read -r f; do
        echo "    冲突解决(取上游版): $f"
        git checkout --theirs -- "$f"
        git add "$f"
    done
fi

echo "==> [4/6] 应用 PLE CPU 分配补丁"
git apply "$PATCH"
echo "    PLE 补丁已应用 ✓"

echo "==> [5/6] 构建 wheel（v$VERSION）"
cd python
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
pip wheel --no-deps --no-build-isolation -w "$BUILD_DIR/wheels" .

echo "==> [6/6] 完成"
ls -la "$BUILD_DIR"/wheels/*.whl

cat <<'EOF'

──────────────────────────────────────────────
安装（在目标 conda 环境中执行）:
  pip install --no-deps <上面的 wheel>
  pip install "lk_moe==2.4.1"
  pip install "tilelang==0.1.11"                 # 必须锁 0.1.11，0.1.12 编译不兼容（文档 §3.3）
  pip install "flashinfer-python[cu13]==0.6.18"  # 与 0.5.19 对齐

启动: 用本仓库的 scripts/start_lsglang_upstream.sh
      （内置 sudo 提权 + ulimit -l unlimited，PLE pinned 表需要）
──────────────────────────────────────────────
EOF
