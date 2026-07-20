#!/usr/bin/env bash
# Verify the slot-routing fix: one server, N sequential requests, log slot+acceptance each.
# Usage: verify_slotfix.sh <llama-server> <label> <port> <out-dir> [n_requests]
set -u -o pipefail
SRV="${1:?server}"; LABEL="${2:?label}"; PORT="${3:?port}"; OUT="${4:?out-dir}"; N="${5:-4}"
mkdir -p "$OUT"
TARGET=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
DEV=Vulkan0
ERR="$OUT/${LABEL}.server.err"

"$SRV" -m "$TARGET" --device "$DEV" -ngl 999 -c 8192 --no-mmap --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type dflash --spec-draft-model "$DRAFT" --spec-draft-device "$DEV" --spec-draft-ngl 999 --spec-draft-n-max 3 \
  --host 127.0.0.1 --port "$PORT" > "$ERR" 2>&1 &
PID=$!
for i in $(seq 1 60); do grep -q 'listening on http' "$ERR" 2>/dev/null && break; sleep 1; done

PROMPTS=(
  "Solve step by step: A farmer has 100m of fence along a river to enclose a rectangular field. What dimensions maximize the area?"
  "Explain step by step how to compute 17 times 23 using the distributive property, then give the answer."
  "List the steps to write a haiku about autumn, then write one. Think step by step first."
  "Reason step by step: if x+3=11, what is x? Show the work then give the final answer."
)
for r in $(seq 0 $((N-1))); do
  P="${PROMPTS[$((r % ${#PROMPTS[@]}))]}"
  resp="$OUT/${LABEL}.req${r}.json"
  curl -s --max-time 300 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"prompt":%s,"n_predict":160,"temperature":0,"seed":7,"stream":false}' "$(printf '%s' "$P" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
    > "$resp" 2>>"$ERR"
done
sleep 2
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null

echo "=== $LABEL: per-request slot + acceptance ==="
grep -E 'selected slot|draft acceptance' "$ERR" | sed 's/.*selected slot by LRU, /slot: /; s/.*draft acceptance =/  acc:/' | paste - - | head -"$N"
echo "=== $LABEL: prefill flush mismatch count ===" 
grep -c 'prefill flush mismatch' "$ERR"
echo "=== $LABEL: dflash stats line (aggregate) ==="
grep -E 'statistics .*dflash' "$ERR" | tail -1