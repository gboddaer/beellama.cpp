#!/usr/bin/env bash
# Multi-slot DFlash correctness + perf: one server, N sequential long-prompt requests.
# Captures per-slot output (garble check) + acceptance + t/s.
# Usage: verify_corr.sh <llama-server> <label> <port> <out-dir> [n_requests]
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
  "Solve step by step, showing all work, then give the final answer on its own line as \"Answer: ...\". A farmer has 100 meters of fencing to enclose a rectangular field that borders a straight river (no fence on the river side). What dimensions maximize the enclosed area, and what is that maximum area?"
  "Explain step by step how to compute 17 times 23 using the distributive property, then state the final result. Be thorough and show each intermediate step."
  "Reason step by step through this problem: if 3x + 7 = 22, what is x? Show every algebraic step, then give the final answer as \"Answer: x = ...\"."
  "Write a short Python function that returns the n-th Fibonacci number iteratively. Explain the logic step by step first, then show the code, then trace it for n=6."
)
for r in $(seq 0 $((N-1))); do
  P="${PROMPTS[$((r % ${#PROMPTS[@]}))]}"
  resp="$OUT/${LABEL}.req${r}.json"
  curl -s --max-time 400 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"prompt":%s,"n_predict":300,"temperature":0,"seed":7,"stream":false}' "$(printf '%s' "$P" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
    > "$resp" 2>>"$ERR"
  # extract generated text
  python3 -c "import json,sys; d=json.load(open('$resp')); open('$OUT/${LABEL}.gen${r}.txt','w').write(d.get('content',''))" 2>/dev/null || true
done
sleep 2
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null

echo "=== $LABEL: per-slot acceptance + t/s ==="
grep -E 'print_timing: id.*draft acceptance|print_timing: id.*n_decoded' "$ERR" | tail -8
echo "=== $LABEL: errors (mismatch/incomplete/D2D/corrupt) ==="
grep -cE 'prefill flush mismatch|incomplete target hidden|D2D ring write failed|incomplete ring span|incomplete GPU capture' "$ERR" || echo 0
echo "=== $LABEL: garble check (non-ASCII char count per gen) ==="
for r in $(seq 0 $((N-1))); do
  f="$OUT/${LABEL}.gen${r}.txt"
  if [ -s "$f" ]; then
    printf "gen%s: bytes=%s non-ascii=%s first80=%s\n" "$r" "$(wc -c < "$f")" "$(LC_ALL=C grep -oP '[^\x00-\x7F]' "$f" 2>/dev/null | wc -l)" "$(head -c 80 "$f" | tr '\n' ' ')"
  fi
done