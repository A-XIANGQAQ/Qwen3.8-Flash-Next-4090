#!/bin/bash
# ==========================================
# Qwen3.8-Flash-Next-NVFP4 @ 4090: 环境安装
# conda env (py3.12) + torch cu130 + Lsglang wheel + flash_attn PR2751 wheel
# 素材来源: guqiong96/Lsglang release lsglang-v1.4.12 (内含 1.4.13 wheel)
# ==========================================
set -euo pipefail

# ---------- 配置区 ----------
CONDA_EXE=${CONDA_EXE:-conda}           # conda 可执行文件
ENV_NAME=lsglang
WHEEL_DIR=${WHEEL_DIR:-/tmp/lsglang-wheels}   # wheel 下载目录
RELEASE_URL="https://github.com/guqiong96/Lsglang/releases/download/lsglang-v1.4.12"
# SHA256 校验（发布页提供）
LSGLANG_SHA="2ccc92f9d242bd13636f60a60ec822fb8ede8f255d3d4d51faf27da53de54e52"
FA_SHA=""   # flash_attn wheel 的 SHA 从 release API 的 digest 字段获取后填入
# --------------------------------

echo "==> [1/4] 下载 wheel（GitHub release）"
mkdir -p "$WHEEL_DIR"
cd "$WHEEL_DIR"
if [ ! -f lsglang-1.4.13-py3-none-any.whl ]; then
    curl -L -o lsglang-1.4.13-py3-none-any.whl \
        "$RELEASE_URL/lsglang-1.4.13-py3-none-any.whl"
fi
if [ ! -f flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl ]; then
    curl -L -o flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl \
        "$RELEASE_URL/flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl"
fi
echo "==> [2/4] SHA256 校验"
echo "$LSGLANG_SHA  lsglang-1.4.13-py3-none-any.whl" | sha256sum -c -
[ -n "$FA_SHA" ] && echo "$FA_SHA  flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl" | sha256sum -c -

echo "==> [3/4] 创建 conda env (python 3.12)"
"$CONDA_EXE" create -n "$ENV_NAME" python=3.12 -y
ENV_BIN="$("$CONDA_EXE" env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')/bin"
[ -x "$ENV_BIN/python" ] || ENV_BIN="$("$CONDA_EXE" info --base)/envs/$ENV_NAME/bin"

echo "==> [4/4] 安装依赖（顺序关键：torch cu130 → flash_attn → lsglang）"
"$ENV_BIN/pip" install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
"$ENV_BIN/pip" install "$WHEEL_DIR"/flash_attn-2.8.4+pr2751-cp312-cp312-linux_x86_64.whl
"$ENV_BIN/pip" install "$WHEEL_DIR"/lsglang-1.4.13-py3-none-any.whl

echo "==> 完成。验证:"
"$ENV_BIN/python" -c "import sglang, flash_attn; print('sglang:', sglang.__version__); print('flash_attn:', flash_attn.__version__)"
echo "启动时把 scripts/start_lsglang.sh 的 SGLANG_BIN 指向: $ENV_BIN/sglang"
