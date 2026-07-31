#!/usr/bin/env bash
# Bench DFlash vs non-DFlash on Vulkan0 (iGPU) for a given llama-cli binary.
# Usage: bench_dflash_v0.sh <llama-cli-path> <label> <out-dir>
set -u -o pipefail

CLI="${1:?llama-cli path}"
LABEL="${2:?label}"
OUT="${3:?out-dir}"
mkdir -p "$OUT"

TARGET="/crypt/models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf"
DEV="Vulkan0"
CTX=8192
NMAX=256

PROMPT="$OUT/prompt.txt"
cat > "$PROMPT" <<'PROMPT_EOF'
What is 2+2? First think step by step, then give the final answer on its own line as "Answer: X".
PROMPT_EOF

run_case() {
  local name="$1"; shift
  local out="$OUT/${LABEL}_${name}.out"
  local err="$OUT/${LABEL}_${name}.err"
  rm -f "$out" "$err"
  /usr/bin/timeout 1200 "$CLI" \
    -m "$TARGET" \
    --device "$DEV" \
    -ngl 999 \
    --ctx-size $CTX \
    --no-mmap \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --seed 7 \
    --temp 0 \
    -n $NMAX \
    --single-turn \
    --simple-io \
    "$@" \
    -f "$PROMPT" \
    > "$out" 2> "$err"
  local rc=$?
  echo "${LABEL}.${name}.rc=$rc"
  grep -Eo '\[ Prompt: [^]]+\]' "$out" | tail -1 | sed "s/^/${LABEL}.${name}.prompt_perf=/"
  grep -Eo 'Generation: [0-9.]+ t/s' "$out" | tail -1 | sed "s/^/${LABEL}.${name}.gen_perf=/"
  # Extract generated text after the prompt echo
  awk '/^\[ End of generation|^Answer:/' "$out" >/dev/null 2>&1
  sed -n '/^What is 2+2?/,/^\[ /p' "$out" | tail -n +2 > "$OUT/${LABEL}_${name}.gen.txt" 2>/dev/null || true
  if [ -s "$err" ]; then
    echo "${LABEL}.${name}.stderr=$(tr '\n' ' ' < "$err" | cut -c1-300)"
  fi
}

run_case nodflash --spec-type none
run_case dflash \
  --spec-type dflash \
  --spec-draft-model "$DRAFT" \
  --spec-draft-device "$DEV" \
  --spec-draft-ngl 999 \
  --spec-draft-n-max 3