#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/qwen36-three-way}"
mkdir -p "$OUT_DIR"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FORK_BASE="$(git merge-base origin/main ggml/master)"
UPSTREAM_SNAPSHOT="$(git merge-base ggml/master origin/merge_llama_into_beellama_2)"
FORK_HEAD="origin/main"
MERGE_HEAD="origin/merge_llama_into_beellama_2"

FILES=(
  common/speculative.cpp
  common/speculative.h
  src/llama-context.cpp
  src/models/qwen35.cpp
  src/models/qwen35moe.cpp
  src/models/dflash_draft.cpp
  ggml/src/ggml-vulkan/ggml-vulkan.cpp
)

{
  echo "FORK_BASE=$FORK_BASE"
  echo "UPSTREAM_SNAPSHOT=$UPSTREAM_SNAPSHOT"
  echo "FORK_HEAD=$(git rev-parse "$FORK_HEAD")"
  echo "MERGE_HEAD=$(git rev-parse "$MERGE_HEAD")"
} > "$OUT_DIR/refs.txt"

git diff --stat "$FORK_BASE".."$FORK_HEAD" -- "${FILES[@]}" > "$OUT_DIR/fork-delta.stat" || true
git diff --stat "$FORK_BASE".."$UPSTREAM_SNAPSHOT" -- "${FILES[@]}" > "$OUT_DIR/upstream-delta.stat" || true
git diff --stat "$FORK_BASE".."$MERGE_HEAD" -- "${FILES[@]}" > "$OUT_DIR/merge-result.stat" || true

git diff --unified=60 "$FORK_BASE".."$FORK_HEAD" -- "${FILES[@]}" > "$OUT_DIR/fork-delta.diff" || true
git diff --unified=60 "$FORK_BASE".."$UPSTREAM_SNAPSHOT" -- "${FILES[@]}" > "$OUT_DIR/upstream-delta.diff" || true
git diff --unified=60 "$FORK_BASE".."$MERGE_HEAD" -- "${FILES[@]}" > "$OUT_DIR/merge-result.diff" || true

# Focused snippets for DFlash/Qwen hidden capture/output/verifier semantics.
PATTERN='dflash|DFlash|hidden_gpu|prefill_gpu|capture|n_outputs|n_outputs_max|inp_out_ids|embeddings_nextn|nextn|verify|accept|rollback|ring_state|set_state|get_state|seq_id|cross|draft|llm_build_dflash|build_recurrent_attn'
for f in fork-delta upstream-delta merge-result; do
  grep -n -E "$PATTERN" "$OUT_DIR/$f.diff" > "$OUT_DIR/$f.focus.txt" || true
done

for file in "${FILES[@]}"; do
  safe="${file//\//__}"
  git blame -L 1,260 "$MERGE_HEAD" -- "$file" > "$OUT_DIR/blame-${safe}.txt" || true
done

echo "Wrote $OUT_DIR"
