#!/bin/bash
# 验收：Strata 0.2.12 + engine 0.1.58 + unsloth UD-Q4_K_XL（huihui v2 破限，pack v4）。
# 只读探测，不改任何状态。用法: bash test_strata_q4kxl.sh
set -u
B=${STRATA_BASE:-http://127.0.0.1:8000}
M=${STRATA_MODEL:-qwen3.8-flash-next-q4kxl-ablit}
L=${STRATA_LOG:-/home/gfyd-ai/dsh_workspace/strata/strata-unsloth-q4kxl.log}
PY=${STRATA_PY:-/home/gfyd-ai/dsh_workspace/strata/Strata/.venv/bin/python}
[ -x "$PY" ] || PY=/home/gfyd-ai/dsh_workspace/strata/venv/bin/python
pass=0; fail=0
ck() { if [ "$2" = "$3" ]; then echo "  PASS $1 = $2"; pass=$((pass+1)); else echo "  FAIL $1: got [$2] want [$3]"; fail=$((fail+1)); fi; }
ckn() { if [ "$2" -ge "$3" ] 2>/dev/null; then echo "  PASS $1 = $2 (>= $3)"; pass=$((pass+1)); else echo "  FAIL $1: got [$2] want >= $3"; fail=$((fail+1)); fi; }

echo "=== 1. /health ==="
H=$(curl -s --max-time 5 $B/health); echo "$H"
ck loaded "$(echo "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("loaded"))')" "True"
ck ctx "$(echo "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("max_context"))')" "262144"
ck images "$(echo "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("images"))')" "True"
ck model "$(echo "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("model"))')" "qwen3.8-flash-next-q4kxl-ablit"

echo "=== 2. /props 模板与引擎版本 ==="
P=$(curl -s --max-time 10 $B/props -o /tmp/props-q4kxl.json -w '%{http_code}')
ck http "$P" "200"
ck template_chars "$(python3 -c 'import json;print(len(json.load(open("/tmp/props-q4kxl.json")).get("chat_template","")))')" "28226"
BI=$(python3 -c 'import json;print(json.load(open("/tmp/props-q4kxl.json")).get("build_info"))')
echo "  build_info = $BI"
ck build_info "$BI" "Strata 0.1.58"

echo "=== 3. /metrics.engine ==="
ME=$(curl -s --max-time 5 $B/metrics)
echo "  $ME" | head -c 400; echo
jget() { echo "$ME" | python3 -c "import json,sys;d=json.load(sys.stdin).get('engine') or {};print(d.get('$1'))" 2>/dev/null; }
ck engine "$(jget engine)" "0.1.58"
ck drafting "$(jget drafting)" "mtp"
ck slots "$(jget batch_slots)" "4"
ckn arena_mib "$(jget arena_mib)" 73000

echo "=== 4. 普通对话 ==="
R=$(curl -s --max-time 120 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64,\"stream\":false}" $B/v1/chat/completions)
echo "$R" | head -c 400; echo
ckn finish_tokens "$(echo "$R" | python3 -c 'import json,sys;u=json.load(sys.stdin).get("usage",{});print(u.get("completion_tokens") or 0)')" 1

echo "=== 5. froggeric v22.5 的 reasoning_effort 通路 ==="
R=$(curl -s --max-time 180 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? 只答数字\"}],\"max_tokens\":32,\"stream\":false,\"reasoning_effort\":\"low\"}" $B/v1/chat/completions)
echo "$R" | head -c 300; echo
ckn effort_finish "$(echo "$R" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("usage",{}).get("completion_tokens") or 0)')" 1

echo "=== 6. 视觉（新第一片 + unsloth mmproj） ==="
"$PY" - <<PY > /tmp/imgtest-q4kxl.json
import base64, io, json
from PIL import Image
b=io.BytesIO(); Image.new('RGB',(64,64),(200,30,30)).save(b,'PNG')
u='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
print(json.dumps({"model":"$M","messages":[{"role":"user","content":[{"type":"text","text":"What color is this image? One word."},{"type":"image_url","image_url":{"url":u}}]}],"max_tokens":32,"stream":False}))
PY
V=$(curl -s --max-time 240 -H 'Content-Type: application/json' -d @/tmp/imgtest-q4kxl.json $B/v1/chat/completions)
echo "$V" | head -c 300; echo
ckn vision_tokens "$(echo "$V" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens") or 0)' 2>/dev/null || echo 0)" 1

echo "=== 7. 并行栈默认值（同 0.2.11/0.2.12） ==="
EPID=$(pgrep -f "engine/strata --serve" | head -1)
echo "  engine pid = ${EPID:-<none>}"
if [ -n "${EPID:-}" ] && [ -r "/proc/$EPID/environ" ]; then
  E=$(tr '\0' '\n' < "/proc/$EPID/environ")
  for kv in STRATA_CKPT_POOL=1 STRATA_CKPT_POOL_READY=4 STRATA_SLOT_CKPT_MOVE=1 STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1 STRATA_BATCH_GDN_GROUPS=1 STRATA_BATCH_ATTN_ROWS=1 STRATA_BATCH_SELF_COMMIT=1 STRATA_QSA_SLOTS=1 STRATA_GDN_TC=1 STRATA_GDN_CONVL2=2; do
    ck "env $kv" "$(echo "$E" | grep -x "$kv" | head -1)" "$kv"
  done
else
  echo "  FAIL 读不到引擎 environ，无法证明并行栈默认值生效"; fail=$((fail+1))
fi

echo "=== 8. 新模型/native/PLE 日志 ==="
grep -E "PLE table|native_experts|pack v|engine started|fatal|error|Error|failed" "$L" 2>/dev/null | tail -8
ckn "PLE table 行" "$(grep -c "PLE table" "$L" 2>/dev/null)" "1"
ckn "PLE 表已加载" "$(grep -c "PLE table loaded" "$L" 2>/dev/null)" "1"
ckn "ple-gguf 在参数里" "$(grep -c -- "--ple-gguf " "$L" 2>/dev/null)" "1"

echo "=== 9. drafter 必须是 MTP ==="
DR=$(grep -c "strata mtp: draft layer loaded" "$L" 2>/dev/null || true)
if [ "${DR:-0}" -ge 1 ]; then ck drafter "$(jget drafting)" "mtp"; else echo "  FAIL drafter: 日志里没有 MTP 草稿层加载行"; fail=$((fail+1)); fi
if grep -qi "dflash2" "$L" 2>/dev/null; then echo "  WARN 日志里出现 dflash2（应是 MTP）"; fi

echo "=== 10. 显存 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo "=== 结果：PASS=$pass FAIL=$fail ==="
if [ "$fail" = 0 ]; then echo OK; else echo "CHECK LOG ABOVE"; exit 1; fi
