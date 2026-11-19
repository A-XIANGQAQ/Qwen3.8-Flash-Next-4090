#!/bin/bash
# 验收：Strata 0.2.16 + engine 0.1.62 + unsloth UD-Q4_K_XL（huihui v2 破限档 1，pack v4）。
# 只读探测；第 8 节是 4 路并发冒烟（会占满 4 个 slot 约 10 秒）。
# 用法: bash scripts/accept_q4kxl_0215.sh
# 可覆盖：STRATA_BASE / STRATA_MODEL / STRATA_LOG / STRATA_PY / STRATA_ROOT
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${STRATA_ROOT:-$(cd "$HERE/../.." && pwd)}
B=${STRATA_BASE:-http://127.0.0.1:8000}
M=${STRATA_MODEL:-qwen3.8-flash-next-q4kxl-ablit-l1}
L=${STRATA_LOG:-$ROOT/strata/strata-unsloth-q4kxl-ablit-l1-0216.log}

PY=${STRATA_PY:-}
if [ -z "$PY" ]; then
  for c in "$ROOT/strata/venv/bin/python" "$ROOT/strata/Strata/.venv/bin/python" python3; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "FAIL: 找不到 python（设 STRATA_PY）"; exit 1; }

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/accept-q4kxl-0216.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
PJ=$TMPD/props.json
pass=0; fail=0
ck() { if [ "$2" = "$3" ]; then echo "  PASS $1 = $2"; pass=$((pass+1)); else echo "  FAIL $1: got [$2] want [$3]"; fail=$((fail+1)); fi; }
ckn() { if [ "$2" -ge "$3" ] 2>/dev/null; then echo "  PASS $1 = $2 (>= $3)"; pass=$((pass+1)); else echo "  FAIL $1: got [$2] want >= $3"; fail=$((fail+1)); fi; }

echo "=== 1. /health ==="
H=$(curl -s --max-time 5 $B/health); echo "$H"
ck loaded "$(echo "$H" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("loaded"))')" "True"
ck ctx "$(echo "$H" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("max_context"))')" "262144"
ck images "$(echo "$H" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("images"))')" "True"
ck model "$(echo "$H" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("model"))')" "$M"

echo "=== 2. /props 模板与引擎版本 ==="
P=$(curl -s --max-time 10 $B/props -o "$PJ" -w '%{http_code}')
ck http "$P" "200"
# 2026-11-09 起生产模板是 Sharp-naze v1.5（34,422 B / 34,386 字符）；froggeric v22.5 备份在
# pack tokenizer/chat_template.jinja.bak-20261109-froggeric225。旧口径 28,226 只适用于 11-09 之前。
ck template_chars "$("$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1])).get("chat_template","")))' "$PJ")" "34386"
BI=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("build_info"))' "$PJ")
echo "  build_info = $BI"
ck build_info "$BI" "Strata 0.1.62"
# 0.2.16 换了引擎（build_info 已是 0.1.62）和目录布局；serve_tree 仍用于确认跑的是新源码树。
# 启动器是 `cd Strata-0.2.16 && python ./serve/server.py`，命令行里只有相对路径，所以看进程 cwd。
ckn serve_tree "$(for p in $(pgrep -f '[s]erve/server\.py'); do readlink /proc/$p/cwd; done | grep -c 'Strata-0\.2\.16' || true)" 1

echo "=== 3. /metrics.engine ==="
ME=$(curl -s --max-time 5 $B/metrics)
echo "  $ME" | head -c 400; echo
jget() { echo "$ME" | "$PY" -c "import json,sys;d=json.load(sys.stdin).get('engine') or {};print(d.get('$1'))" 2>/dev/null; }
ck engine "$(jget engine)" "0.1.62"
ck drafting "$(jget drafting)" "mtp"
ck slots "$(jget batch_slots)" "4"
ckn expert_slots "$(jget expert_slots)" 10000
ck pcie_frac "$(jget pcie_frac)" "0.15"
ckn arena_mib "$(jget arena_mib)" 73000

echo "=== 4. 普通对话 ==="
R=$(curl -s --max-time 180 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64,\"stream\":false}" $B/v1/chat/completions)
echo "$R" | head -c 400; echo
ckn finish_tokens "$(echo "$R" | "$PY" -c 'import json,sys;u=json.load(sys.stdin).get("usage",{});print(u.get("completion_tokens") or 0)')" 1

echo "=== 5. reasoning_effort 通路（模板 Sharp-naze v1.5）==="
R=$(curl -s --max-time 180 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? 只答数字\"}],\"max_tokens\":32,\"stream\":false,\"reasoning_effort\":\"low\"}" $B/v1/chat/completions)
echo "$R" | head -c 300; echo
ckn effort_finish "$(echo "$R" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("usage",{}).get("completion_tokens") or 0)')" 1

echo "=== 6. 视觉（新第一片 + unsloth mmproj） ==="
if "$PY" -c 'import PIL' >/dev/null 2>&1; then
  "$PY" - "$M" > "$TMPD/img.json" <<'PY'
import base64, io, json, sys
from PIL import Image
b = io.BytesIO(); Image.new('RGB', (64, 64), (200, 30, 30)).save(b, 'PNG')
u = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What color is this image? One word."},
    {"type": "image_url", "image_url": {"url": u}}]}], "max_tokens": 32, "stream": False}))
PY
  V=$(curl -s --max-time 300 -H 'Content-Type: application/json' -d @"$TMPD/img.json" $B/v1/chat/completions)
  echo "$V" | head -c 300; echo
  ckn vision_tokens "$(echo "$V" | "$PY" -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens") or 0)' 2>/dev/null || echo 0)" 1
else
  echo "  FAIL 视觉：$PY 没有 PIL（装 pillow 或设 STRATA_PY 指向带 PIL 的 venv）"; fail=$((fail+1))
fi

echo "=== 7. 并行栈：0.2.13 起默认值搬进引擎（服务端不再注入环境变量） ==="
EPID=$( { pgrep -x strata; pgrep -x strata-host; } 2>/dev/null | head -1 )
echo "  engine pid = ${EPID:-<none>}"
if [ -n "${EPID:-}" ] && [ -r "/proc/$EPID/environ" ]; then
  tr '\0' '\n' < "/proc/$EPID/environ" | grep -E '^STRATA_(CKPT|SLOT|BATCH|QSA|GDN)' || echo "  （environ 里没有 STRATA_BATCH_* 等变量：符合 0.2.13+ 把默认值内建到引擎的行为）"
else
  echo "  WARN 读不到引擎 environ（仅信息，不影响判定）"
fi

echo "=== 8. 4 路并发冒烟（证明 batch_slots=4 真的在并行） ==="
COUT=$(STRATA_BASE=$B STRATA_MODEL=$M "$PY" - <<'PY' 2>/dev/null
import json, os, threading, time, urllib.request
base = os.environ["STRATA_BASE"]; model = os.environ["STRATA_MODEL"]
res = []
def one(i):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"用一句话解释并发槽位 {i} 号。"}],
                       "max_tokens": 128, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        res.append((d.get("usage") or {}).get("completion_tokens") or 0)
    except Exception:
        res.append(0)
t0 = time.perf_counter()
ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
wall = max(time.perf_counter() - t0, 1e-6)
ok = sum(1 for r in res if r > 0)
print(f"smoke ok={ok} tok={sum(res)} agg={sum(res)/wall:.1f}")
PY
)
echo "  $COUT"
ckn concurrent_ok "$(echo "$COUT" | sed -n 's/.*smoke ok=\([0-9]*\).*/\1/p')" 4
# 128 token 的短冒烟被 slot 停放/恢复开销主导，只证明四路真的并行；正式并发数看 bench.sh 的 512 token 口径
ckn concurrent_agg_tps "$(echo "$COUT" | sed -n 's/.*agg=\([0-9]*\).*/\1/p')" 60

echo "=== 9. 新模型/native/PLE 日志（只看本次启动） ==="
SEG=$TMPD/start.log
if [ -r "$L" ]; then
  n=$(grep -n "engine started" "$L" 2>/dev/null | tail -1 | cut -d: -f1)
  if [ -n "${n:-}" ]; then tail -n +"$n" "$L" > "$SEG"; else cp "$L" "$SEG"; fi
  grep -E "PLE table|native_experts|pack v|fatal|error|Error|failed" "$SEG" | tail -8
  ckn "PLE table 行" "$(grep -c "PLE table" "$SEG")" 1
  ckn "PLE 表已加载" "$(grep -c "PLE table loaded" "$SEG")" 1
  ck "ple-gguf 在参数里" "$(grep -c -- "--ple-gguf " "$SEG")" "1"
  ck "引擎路径是 0.2.16" "$(grep -c "Strata-0.2.16/engine/strata --serve" "$SEG")" "1"
  KQ=$(grep -ciE "own kernels|prompt kernels" "$SEG" || true)
  if [ "${KQ:-0}" -ge 1 ]; then
    echo "  PASS prompt kernel: $(grep -iE "own kernels|prompt kernels" "$SEG" | head -1 | cut -c1-160)"
  else
    echo "  WARN 日志里没看到 own kernels/prompt kernels 行（0.1.62 新内核的提示行，措辞可能不同）"
  fi
else
  echo "  FAIL 读不到日志 $L（设 STRATA_LOG）"; fail=$((fail+1))
fi

echo "=== 10. drafter 必须是 MTP ==="
DR=0; [ -r "$L" ] && DR=$(grep -c "strata mtp: draft layer loaded" "$L" 2>/dev/null || true)
if [ "${DR:-0}" -ge 1 ]; then ck drafter "$(jget drafting)" "mtp"; else echo "  FAIL drafter: 日志里没有 MTP 草稿层加载行"; fail=$((fail+1)); fi
if [ -r "$L" ] && grep -qi "dflash2" "$L" 2>/dev/null; then echo "  WARN 日志里出现 dflash2（应是 MTP）"; fi

echo "=== 11. 显存 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo "=== 结果：PASS=$pass FAIL=$fail ==="
if [ "$fail" = 0 ]; then echo OK; else echo "CHECK LOG ABOVE"; exit 1; fi
