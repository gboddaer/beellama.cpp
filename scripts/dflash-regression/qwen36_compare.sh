#!/usr/bin/env bash
set -u -o pipefail

OUT_DIR="${1:-/tmp/qwen36-dflash-regression}"
mkdir -p "$OUT_DIR"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

CLI="$ROOT/build_vulkan/bin/llama-cli"
TARGET="/fast/models/Qwen3.6-27B-UD-Q6_K_XL.gguf"
DRAFT="/fast/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf"
PROMPT="$OUT_DIR/prompt.txt"
SUMMARY="$OUT_DIR/summary.txt"

cat > "$PROMPT" <<'PROMPT_EOF'
Answer in final form only. Do not reveal private reasoning. Summarize the Vulkan DFlash risk profile for Qwen3.6-27B on W7800 in five concise bullet points. Mention non-DFlash baseline, DFlash verifier correctness, RADV device loss interpretation, prompt throughput, and generation throughput.
PROMPT_EOF

: > "$SUMMARY"
echo "branch=$(git branch --show-current || true)" >> "$SUMMARY"
echo "commit=$(git rev-parse HEAD)" >> "$SUMMARY"

run_case() {
  local name="$1"
  shift
  local out="$OUT_DIR/$name.out"
  local err="$OUT_DIR/$name.err"
  rm -f "$out" "$err"
  /usr/bin/timeout 900 "$CLI" \
    -m "$TARGET" \
    --device Vulkan1 \
    -ngl 999 \
    --ctx-size 32768 \
    --no-mmap \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --seed 7 \
    --temp 0 \
    -n 384 \
    --single-turn \
    --simple-io \
    "$@" \
    -f "$PROMPT" \
    > "$out" 2> "$err"
  local rc=$?
  echo "$name.rc=$rc" >> "$SUMMARY"
  grep -Eo '\[ Prompt: [^]]+ \| Generation: [^]]+ \]' "$out" | tail -1 | sed "s/^/$name.perf=/" >> "$SUMMARY" || true
  if [ -s "$err" ]; then
    echo "$name.stderr=$(tr '\n' ' ' < "$err" | cut -c1-500)" >> "$SUMMARY"
  fi
  return "$rc"
}

if [ ! -x "$CLI" ]; then
  echo "missing llama-cli: $CLI" | tee -a "$SUMMARY"
  exit 2
fi

run_case nodflash --spec-type none
nodflash_rc=$?
run_case dflash \
  --spec-type dflash \
  --spec-draft-model "$DRAFT" \
  --spec-draft-device Vulkan1 \
  --spec-draft-ngl 999 \
  --spec-draft-n-max 3
dflash_rc=$?

python3 scripts/dflash-regression/extract_cli_output.py "$OUT_DIR/nodflash.out" "$OUT_DIR/nodflash.generated.txt"
python3 scripts/dflash-regression/extract_cli_output.py "$OUT_DIR/dflash.out" "$OUT_DIR/dflash.generated.txt"

if [ "$nodflash_rc" -ne 0 ] || [ "$dflash_rc" -ne 0 ]; then
  echo "result=runtime-fail" >> "$SUMMARY"
  cat "$SUMMARY"
  exit 1
fi

if cmp -s "$OUT_DIR/nodflash.generated.txt" "$OUT_DIR/dflash.generated.txt"; then
  echo "result=pass-identical-output" >> "$SUMMARY"
  cat "$SUMMARY"
  exit 0
fi

# Deterministic greedy speculative decoding should match the target output.
echo "result=fail-output-mismatch" >> "$SUMMARY"
echo "--- nodflash generated ---" >> "$SUMMARY"
head -40 "$OUT_DIR/nodflash.generated.txt" >> "$SUMMARY"
echo "--- dflash generated ---" >> "$SUMMARY"
head -40 "$OUT_DIR/dflash.generated.txt" >> "$SUMMARY"
cat "$SUMMARY"
exit 1
