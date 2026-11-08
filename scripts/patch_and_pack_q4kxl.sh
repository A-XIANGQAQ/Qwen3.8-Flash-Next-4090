#!/usr/bin/env bash
# 在官方 unsloth UD-Q4_K_XL(v3.1) 上打 huihui v2 破限补丁，然后建 Strata pack。
# 只重写 shard2/shard3（目标张量全在这两片，且都是 Q8_0）；shard1/4 无目标张量，软链指回官方文件。
#
# 依赖：$STRATA_PY（或 strata/venv、strata/Strata/.venv）里装了 numpy/requirements，
#       且能 import gguf（STRATA_GGUF_PY 指向任意 llama.cpp checkout 的 gguf-py）。
# 可覆盖：STRATA_ROOT（工作区根）、STRATA_SRC（UD-Q4_K_XL 四片所在目录）、STRATA_WORK（临时目录）。
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${STRATA_ROOT:-$(cd "$REPO/.." && pwd)}
S=$ROOT/strata

PY=${STRATA_PY:-}
if [ -z "$PY" ]; then
  for c in "$S/venv/bin/python" "$S/Strata/.venv/bin/python"; do [ -x "$c" ] && PY=$c && break; done
fi
[ -n "$PY" ] || { echo "FAIL: 需要装了 requirements 的 venv python（设 STRATA_PY）"; exit 1; }

# gguf-py 不能绑死在某个安装布局上：先用现成 import，再试 STRATA_GGUF_PY / 常见目录。
if ! "$PY" -c 'import gguf' >/dev/null 2>&1; then
  for g in "${STRATA_GGUF_PY:-}" \
           "$S/Strata/third_party/llama.cpp/gguf-py" \
           "$S/Strata-0.2.12/third_party/llama.cpp/gguf-py" \
           "$ROOT/.ref/llama.cpp/gguf-py"; do
    if [ -n "$g" ] && [ -d "$g/gguf" ]; then
      export STRATA_GGUF_PY=$g
      export PYTHONPATH=$g${PYTHONPATH:+:$PYTHONPATH}
      break
    fi
  done
fi
"$PY" -c 'import gguf' >/dev/null 2>&1 || {
  echo "FAIL: $PY 不能 import gguf。装一个 llama.cpp 的 gguf-py 并设 STRATA_GGUF_PY=<llama.cpp>/gguf-py"
  exit 1
}

# 下载布局有两种：--local_dir 直接平铺，或 modelscope 保留 UD-Q4_K_XL/ 子目录。
SRC=${STRATA_SRC:-}
if [ -z "$SRC" ]; then
  for d in "$ROOT/models/unsloth-UD-Q4_K_XL" "$ROOT/models/unsloth-UD-Q4_K_XL/UD-Q4_K_XL"; do
    if [ -s "$d/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" ]; then SRC=$d; break; fi
  done
fi
[ -n "$SRC" ] || { echo "FAIL: 找不到 UD-Q4_K_XL 四片（设 STRATA_SRC）"; exit 1; }

# 消融档位与输出位置可覆盖，默认是 huihui v2 发布版所用的轻档 1（ssm_out + attn_output）。
# 档位说明：0 = ssm_out + ffn_down_shexp；1 = ssm_out + attn_output；2 = 三者全消（最重）。
ABLATE=${STRATA_ABLATE:-1}
case "$ABLATE" in 0|1|2) ;; *) echo "FAIL: STRATA_ABLATE 只能是 0/1/2"; exit 1;; esac
ABL=${STRATA_ABL_DST:-$ROOT/models/unsloth-UD-Q4_K_XL-abliterated-l$ABLATE}
PACK_OUT=${STRATA_PACK_OUT:-$S/Strata-data/packs/unsloth-ud-q4kxl-ablit-l$ABLATE}
PDIR=$ROOT/models/patch-huihui-v2
PS=$PDIR/patch-unsloth-Qwen3.8-Flash-Next-dec.py
PATCH=$PDIR/patch-unsloth-all
WORK=${STRATA_WORK:-${TMPDIR:-/tmp}/strata-q4kxl-patch}
mkdir -p "$ABL" "$WORK/pin2" "$WORK/pin3"

n2=Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
n3=Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
n1=Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
n4=Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf

for f in "$n1" "$n2" "$n3" "$n4"; do [ -s "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }; done
[ -s "$PS" ] || { echo "missing $PS"; exit 1; }
[ -s "$PATCH/Huihui-Qwen3.8-Flash-Next-abliterated-patch-all.gguf" ] || { echo "missing patch gguf"; exit 1; }
[ -s "$PATCH/Huihui-Qwen3.8-Flash-Next-abliterated-patch-all.gguf.key" ] || { echo "missing patch key"; exit 1; }

ln -sfn "$SRC/$n1" "$ABL/$n1"
ln -sfn "$SRC/$n4" "$ABL/$n4"
ln -sfn "$SRC/$n2" "$WORK"/pin2/$n2
ln -sfn "$SRC/$n3" "$WORK"/pin3/$n3

redact() { sed -E 's/(Using key \([0-9]+ bytes\): ).*/\1<redacted>/'; }

echo "== patch shard2 (ablate=$ABLATE -> $ABL) $(date +%T)"
"$PY" "$PS" --input-path "$WORK"/pin2 --patch-path "$PATCH" --output-path "$ABL" \
  --start-layer 0 --end-layer 48 --ablate-experts "$ABLATE" 2>&1 | redact | tee "$WORK"/patch2.log | tail -6

echo "== patch shard3 $(date +%T)"
"$PY" "$PS" --input-path "$WORK"/pin3 --patch-path "$PATCH" --output-path "$ABL" \
  --start-layer 0 --end-layer 48 --ablate-experts "$ABLATE" 2>&1 | redact | tee "$WORK"/patch3.log | tail -6

[ -s "$ABL/$n2" ] && [ -s "$ABL/$n3" ] || { echo "FAIL 补丁输出缺失"; exit 1; }
echo "== checks（0 张量/类型不匹配都必须失败，不能只看输出文件存在）"
grep -q "Patch files (1)" "$WORK"/patch2.log || { echo "FAIL shard2 补丁未加载"; exit 1; }
grep -q "Patch files (1)" "$WORK"/patch3.log || { echo "FAIL shard3 补丁未加载"; exit 1; }
# 期望张量数随档位变化（huihui patch 里 Q8_0 版的命中数）：0=84、1=48、2=96。
# 不按分片写死：shard2/3 的层边界在 layer 28 上不是整数切分（实测 1 档为 28/20）。
case "$ABLATE" in 0) want=84;; 1) want=48;; 2) want=96;; esac
p2=$(grep -oE "Patched tensors: [0-9]+" "$WORK"/patch2.log | tail -1 | awk '{print $3}')
p3=$(grep -oE "Patched tensors: [0-9]+" "$WORK"/patch3.log | tail -1 | awk '{print $3}')
mm=$(cat "$WORK"/patch2.log "$WORK"/patch3.log | grep -c "Copied because dtype mismatch" || true)
nf=$(cat "$WORK"/patch2.log "$WORK"/patch3.log | grep -c "was not found" || true)
total=$((${p2:-0}+${p3:-0}))
echo "patched shard2=$p2 shard3=$p3 total=$total (want $want) | dtype mismatch=$mm | not found=$nf"
[ "$total" = "$want" ] || { echo "FAIL patched 张量数 $total，档位 $ABLATE 期望 $want"; exit 1; }
[ "$mm" = 0 ] && [ "$nf" = 0 ] || { echo "FAIL 有张量被静默跳过"; exit 1; }
ls -l "$ABL"

echo "== build pack $(date +%T)"
"$PY" "$S/Strata-0.2.12/tools/iq_pack.py" --gguf "$ABL/$n1" \
  --out "$PACK_OUT" --compat-bf16 \
  2>&1 | tee "$WORK"/iqpack.log | tail -8

[ -s "$PACK_OUT/native_experts.txt" ] || { echo "FAIL pack 缺 native_experts.txt（iq_pack 未完成）"; exit 1; }
echo "== template（unsloth 基线用回 froggeric v22.5，不套 DavidAU 的模式体）"
cp -f "$REPO/patches/froggeric_v22.5.jinja" \
      "$PACK_OUT/tokenizer/chat_template.jinja"
TS=$(wc -c < "$PACK_OUT/tokenizer/chat_template.jinja")
[ "$TS" = 28234 ] || { echo "FAIL 模板 $TS B（期望 28234）"; exit 1; }
echo "template = $TS B"
head -3 "$PACK_OUT/native_experts.txt"
ls -l "$PACK_OUT"
echo "== DONE $(date +%T)"
df -h "$ROOT" | tail -1
