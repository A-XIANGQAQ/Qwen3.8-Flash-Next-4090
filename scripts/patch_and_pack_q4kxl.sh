#!/usr/bin/env bash
# 在官方 unsloth UD-Q4_K_XL(v3.1) 上打 huihui v2 破限补丁，然后建 Strata pack。
# 只重写 shard2/shard3（目标张量全在这两片，且都是 Q8_0）；shard1/4 无目标张量，硬链接。
set -euo pipefail

ROOT=${STRATA_ROOT:-/home/gfyd-ai/dsh_workspace}
REPO=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/strata
PY=${STRATA_PY:-}
if [ -z "$PY" ]; then
  for c in "$S/venv/bin/python" "$S/Strata/.venv/bin/python"; do [ -x "$c" ] && PY=$c && break; done
fi
[ -n "$PY" ] || { echo "FAIL: 需要装了 requirements 的 venv python（设 STRATA_PY）"; exit 1; }
SRC=$ROOT/models/unsloth-UD-Q4_K_XL
ABL=$ROOT/models/unsloth-UD-Q4_K_XL-abliterated
PDIR=$ROOT/models/patch-huihui-v2
PS=$PDIR/patch-unsloth-Qwen3.8-Flash-Next-dec.py
PATCH=$PDIR/patch-unsloth-all
export PYTHONPATH=$S/Strata/third_party/llama.cpp/gguf-py
WORK=${STRATA_WORK:-${TMPDIR:-/tmp}/strata-q4kxl-patch}
mkdir -p "$ABL" "$WORK/pin2" "$WORK/pin3"

n2=Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
n3=Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
n1=Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
n4=Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf

for f in "$n1" "$n2" "$n3" "$n4"; do [ -s "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }; done
[ -s "$PATCH/Huihui-Qwen3.8-Flash-Next-abliterated-patch-all.gguf" ] || { echo "missing patch gguf"; exit 1; }
[ -s "$PATCH/Huihui-Qwen3.8-Flash-Next-abliterated-patch-all.gguf.key" ] || { echo "missing patch key"; exit 1; }

ln -f "$SRC/$n1" "$ABL/$n1"
ln -f "$SRC/$n4" "$ABL/$n4"
ln -sf "$SRC/$n2" "$WORK"/pin2/$n2
ln -sf "$SRC/$n3" "$WORK"/pin3/$n3

redact() { sed -E 's/(Using key \([0-9]+ bytes\): ).*/\1<redacted>/'; }

echo "== patch shard2 $(date +%T)"
python3 "$PS" --input-path "$WORK"/pin2 --patch-path "$PATCH" --output-path "$ABL" \
  --start-layer 0 --end-layer 48 --ablate-experts 2 2>&1 | redact | tee "$WORK"/patch2.log | tail -6

echo "== patch shard3 $(date +%T)"
python3 "$PS" --input-path "$WORK"/pin3 --patch-path "$PATCH" --output-path "$ABL" \
  --start-layer 0 --end-layer 48 --ablate-experts 2 2>&1 | redact | tee "$WORK"/patch3.log | tail -6

echo "== checks（0 张量/类型不匹配都必须失败，不能只看输出文件存在）"
grep -q "Patch files (1)" "$WORK"/patch2.log || { echo "FAIL shard2 补丁未加载"; exit 1; }
grep -q "Patch files (1)" "$WORK"/patch3.log || { echo "FAIL shard3 补丁未加载"; exit 1; }
p2=$(grep -c "Patched tensors: 57" "$WORK"/patch2.log || true)
p3=$(grep -c "Patched tensors: 39" "$WORK"/patch3.log || true)
mm=$(cat "$WORK"/patch2.log "$WORK"/patch3.log | grep -c "Copied because dtype mismatch" || true)
nf=$(cat "$WORK"/patch2.log "$WORK"/patch3.log | grep -c "was not found" || true)
echo "patched shard2=$p2 shard3=$p3 | dtype mismatch=$mm | not found=$nf"
[ "$p2" = 1 ] && [ "$p3" = 1 ] || { echo "FAIL patched 张量数不对（期望 57/39）"; exit 1; }
[ "$mm" = 0 ] && [ "$nf" = 0 ] || { echo "FAIL 有张量被静默跳过"; exit 1; }
ls -l "$ABL"

echo "== build pack $(date +%T)"
"$PY" "$S/Strata-0.2.12/tools/iq_pack.py" --gguf "$ABL/$n1" \
  --out "$S/Strata-data/packs/unsloth-ud-q4kxl-ablit" --compat-bf16 \
  2>&1 | tee "$WORK"/iqpack.log | tail -8

echo "== template（unsloth 基线用回 froggeric v22.5，不套 DavidAU 的模式体）"
cp -f "$REPO/patches/froggeric_v22.5.jinja" \
      "$S/Strata-data/packs/unsloth-ud-q4kxl-ablit/tokenizer/chat_template.jinja"
wc -c "$S/Strata-data/packs/unsloth-ud-q4kxl-ablit/tokenizer/chat_template.jinja"
head -3 "$S/Strata-data/packs/unsloth-ud-q4kxl-ablit/native_experts.txt"
ls -l "$S/Strata-data/packs/unsloth-ud-q4kxl-ablit"
echo "== DONE $(date +%T)"
df -h /home/gfyd-ai | tail -1
