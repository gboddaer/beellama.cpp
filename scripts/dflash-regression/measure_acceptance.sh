#!/usr/bin/env bash
# Measure DFlash acceptance stats from llama-server.
# Usage: measure_acceptance.sh <llama-server> <label> <port> <out-dir>
set -u -o pipefail
SRV="${1:?server}"; LABEL="${2:?label}"; PORT="${3:?port}"; OUT="${4:?out-dir}"
mkdir -p "$OUT"
TARGET=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
DEV=Vulkan0
ERR="$OUT/${LABEL}.server.err"
OUTJSON="$OUT/${LABEL}.resp.json"

# longer/complex prompt for stable acceptance stats
read -r -d '' PROMPT <<'PROMPT_EOF' || true
Solve this step by step, showing all work, then give the final answer on its own line as "Answer: ...".

A farmer has 100 meters of fencing to enclose a rectangular field that borders a straight river. The farmer does not need to fence the side along the river. What are the dimensions (length and width) that maximize the enclosed area, and what is that maximum area?
PROMPT_EOF

# start server
"$SRV" -m "$TARGET" --device "$DEV" -ngl 999 -c 8192 --no-mmap --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type dflash --spec-draft-model "$DRAFT" --spec-draft-device "$DEV" --spec-draft-ngl 999 --spec-draft-n-max 3 \
  --host 127.0.0.1 --port "$PORT" > "$ERR" 2>&1 &
SRV_PID=$!
sleep 2

# wait for ready (up to 120s)
ready=0
for i in $(seq 1 120); do
  if grep -qE 'listening on http|server is listening|all slots are ready' "$ERR" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "$LABEL: server died early"; break; fi
  sleep 1
done
echo "$LABEL: ready=$ready pid=$SRV_PID"

if [ "$ready" = "1" ]; then
  # send generation request
  curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"prompt":%s,"n_predict":256,"temperature":0,"seed":7,"stream":false}' "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
    > "$OUTJSON" 2>>"$ERR"
  echo "$LABEL: curl rc=$? resp_bytes=$(wc -c < "$OUTJSON" 2>/dev/null)"
  # give the server a moment to flush the per-slot acceptance stats
  sleep 2
fi

# shut down our server only
kill "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null

echo "=== $LABEL acceptance stats ==="
grep -E 'draft acceptance|statistics .*#gen drafts|n_draft_total|mean len' "$ERR" | tail -10
echo "=== $LABEL server stderr tail ==="
tail -5 "$ERR"