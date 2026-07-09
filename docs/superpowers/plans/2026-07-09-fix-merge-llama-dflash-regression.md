# Fix Merge Llama DFlash Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and fix the Qwen3.6-27B DFlash correctness and performance regression on `merge_llama_into_beellama_2` while preserving the upstream llama.cpp merge and BeeLlama fork behavior.

**Architecture:** Treat this as a three-way integration bug: gboddaer `origin/main` is the BeeLlama baseline, `ggml/master` is upstream llama.cpp, and `merge_llama_into_beellama_2` is the attempted integration. The fix must be driven by reproducible branch comparisons, three-way semantic diffs/blame, and minimal changes to the DFlash/Qwen35 integration path.

**Tech Stack:** C++, CMake, Vulkan backend, BeeLlama DFlash, Qwen35/Qwen3.6 model graph builders, local GGUF models in `/fast/models`, three-way git diff/blame, shell/Python regression scripts.

## Global Constraints

- Current working branch for fixes: `merge_llama_into_beellama_2`.
- Do not push, create PRs, or commit unless the user explicitly asks.
- Use `/home/gbo/.local/share/pi-node/node-v22.23.1-linux-x64/bin/cmake` if the system CMake is too old.
- Tests must target W7800 as `Vulkan1` / RADV NAVI31, not `Vulkan0` / Renoir.
- DFlash comparison target model: `/fast/models/Qwen3.6-27B-UD-Q6_K_XL.gguf`.
- DFlash comparison draft model: `/fast/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`.
- Do not label RADV device loss as VRAM OOM unless allocation evidence supports it.
- Follow systematic debugging: no code fix until the failing behavior boundary is identified by evidence.
- Prefer three-way semantic diff/blame over bisect for this integration branch. Bisect is optional only if the three-way analysis is inconclusive.
- Avoid unrelated refactors. Preserve upstream llama.cpp merge behavior unless evidence shows the merge resolution is wrong.
- Update `TASK_PROGRESS.md` after every meaningful investigation or code step. Do not leave progress only in chat.
- Every report and every `TASK_PROGRESS.md` update must separate **FACTS**, **HYPOTHESES**, **TEST RESULTS**, and **NEXT STEPS**.

---

## Background for the next agent

This repository is gboddaer's BeeLlama fork of llama.cpp. The branch `merge_llama_into_beellama_2` is intended to integrate upstream `ggml-org/llama.cpp` into the BeeLlama fork while preserving BeeLlama-specific features, especially DFlash speculative decoding for Qwen3.6/Qwen35-style models.

Known branch/history facts from this session:

- `origin/main` is the current gboddaer BeeLlama baseline.
- `ggml/master` was fetched from `https://github.com/ggml-org/llama.cpp.git master`.
- `origin/merge_llama_into_beellama_2` is the attempted integration branch.
- `origin/main` is an ancestor of `origin/merge_llama_into_beellama_2`.
- Current `ggml/master` is not an ancestor of `origin/merge_llama_into_beellama_2`; the merge branch appears integrated to an earlier upstream snapshot.
- The upstream snapshot merge-base found in this session was `f708a5b2c vulkan: roll bk loop in matmul for asahi linux (#24663)`.
- The merge-base of `origin/main` and current `ggml/master` was `6ddc9430b readme : add status badges (#24104)`.

Known runtime facts from this session:

- Hardware target is W7800/RADV NAVI31 as `Vulkan1`; `Vulkan0` is Renoir iGPU and must not be selected.
- Non-DFlash Qwen3.6-27B Q6_K_XL was stable on both `origin/main` and `merge_llama_into_beellama_2`, around `11.0 tok/s` generation.
- DFlash Qwen3.6-27B Q6_K_XL was coherent and faster on `origin/main`, around `14.3 tok/s` generation.
- DFlash Qwen3.6-27B Q6_K_XL on `merge_llama_into_beellama_2` completed without crashing but produced noisy/incoherent output and slower generation, around `4.7 tok/s` in the larger-prompt repeat.
- No RADV device loss or GPU reset was observed during those tests.

Existing progress file:

- `TASK_PROGRESS.md` already exists and contains prior merge/DFlash investigation notes.
- It includes prior sections such as HF-035 through HF-040, with explicit proven facts and hypotheses about DFlash quality regressions, hidden GPU capture, `n_outputs_max`, graph invalidation, and single-slot versus multi-slot behavior.
- Any agent continuing this plan must read `TASK_PROGRESS.md` before changing code and must append progress there after each task.

Important distinction:

- A **fact** is something directly observed in git history, code, logs, tests, or command output.
- A **hypothesis** is an explanation that might account for facts but is not yet proven.
- Do not write hypotheses as facts. Use labels exactly: `FACT`, `HYPOTHESIS`, `TEST RESULT`, `NEXT STEP`.

---

## Required progress logging protocol

After each task or significant sub-step, append a dated entry to `TASK_PROGRESS.md` using this template:

```markdown
## YYYY-MM-DD HH:MM UTC - Merge DFlash regression investigation

### FACTS
- <observed command, commit, diff, log, or test result>

### HYPOTHESES
- <possible explanation, clearly marked as unproven>

### TEST RESULTS
- Command: `<exact command>`
- Result: `<pass/fail/measurement>`
- Output files: `<paths>`

### NEXT STEPS
- <next concrete action>
```

Rules:

- If no hypothesis is formed, write `- None yet.` under `HYPOTHESES`.
- If a hypothesis is disproven, move it to a new fact line such as `FACT: Hypothesis X was disproven by command Y`.
- Include exact branch names, commit hashes, and output paths.
- Do not delete existing `TASK_PROGRESS.md` content.
- Do not rely on chat history as the source of truth.

---

## File Structure

The plan will create or modify these files:

- Create: `scripts/dflash-regression/qwen36_compare.sh`
  - Runs deterministic non-DFlash and DFlash CLI comparisons on the local W7800 and extracts correctness/performance metrics.
- Create: `scripts/dflash-regression/three_way_dflash_diff.sh`
  - Captures focused three-way diffs and blame information for DFlash/Qwen35 files.
- Create: `scripts/dflash-regression/extract_cli_output.py`
  - Normalizes `llama-cli` output by removing banners, prompt echo, spinners, and performance lines before comparing generated text.
- Modify after root cause is known: one or more of:
  - `src/models/qwen35.cpp`
  - `src/models/qwen35moe.cpp`
  - `src/models/dflash_draft.cpp`
  - `src/llama-context.cpp`
  - `common/speculative.cpp`
  - `common/speculative.h`
- Test: `tests/test-dflash-ring.cpp` only if a ring or state-save issue is proven.
- Test: local script `scripts/dflash-regression/qwen36_compare.sh` for model-backed correctness and performance.
- Progress log: `TASK_PROGRESS.md` must be updated after every task with facts, hypotheses, test results, and next steps.

---

### Task 1: Add a deterministic local comparison harness

**Files:**
- Create: `scripts/dflash-regression/extract_cli_output.py`
- Create: `scripts/dflash-regression/qwen36_compare.sh`

**Interfaces:**
- Produces: `scripts/dflash-regression/qwen36_compare.sh <output-dir>` returning exit code `0` for pass and nonzero for fail.
- Produces: `<output-dir>/summary.txt` containing branch, commit, correctness result, prompt t/s, generation t/s, and stderr snippets.
- Consumes: local models in `/fast/models` and `build_vulkan/bin/llama-cli`.

- [ ] **Step 1: Create the output extractor**

Create `scripts/dflash-regression/extract_cli_output.py` with this content:

```python
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: extract_cli_output.py <input> <output>", file=sys.stderr)
    sys.exit(2)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(errors="replace").replace("\b", "")

# Remove llama banner and command menu.
marker = "[Start thinking]"
idx = text.find(marker)
if idx >= 0:
    generated = text[idx:]
else:
    # Fallback: keep text after the prompt truncation marker if present.
    prompt_marker = "... (truncated)"
    idx = text.find(prompt_marker)
    generated = text[idx + len(prompt_marker):] if idx >= 0 else text

# Remove perf and process lifecycle lines.
lines = []
for line in generated.splitlines():
    if re.search(r"\[ Prompt: .* Generation: .*\]", line):
        continue
    if line.strip() in {"Exiting...", "Loading model..."}:
        continue
    if line.startswith("build      :") or line.startswith("model      :"):
        continue
    lines.append(line.rstrip())

normalized = "\n".join(lines).strip()
dst.write_text(normalized + "\n")
```

- [ ] **Step 2: Create the comparison script**

Create `scripts/dflash-regression/qwen36_compare.sh` with this content:

```bash
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
```

- [ ] **Step 3: Make scripts executable**

Run:

```bash
chmod +x scripts/dflash-regression/extract_cli_output.py scripts/dflash-regression/qwen36_compare.sh
```

Expected: command exits `0`.

- [ ] **Step 4: Run on `origin/main` as the known-good BeeLlama baseline**

Run:

```bash
git switch --detach origin/main
cmake --build build_vulkan --target llama-cli -j "$(nproc)"
scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-origin-main
```

Expected: exit `0`, `result=pass-identical-output`, DFlash generation throughput at or above non-DFlash.

- [ ] **Step 5: Run on `merge_llama_into_beellama_2` as the known-bad integration branch**

Run:

```bash
git switch merge_llama_into_beellama_2
cmake --build build_vulkan --target llama-cli -j "$(nproc)"
scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-merge-branch
```

Expected before the fix: nonzero exit with `result=fail-output-mismatch` or DFlash generation throughput far below non-DFlash.

- [ ] **Step 6: Human checkpoint**

Do not modify C++ yet. Confirm the script reproduces the same pattern already observed manually:

```text
origin/main: DFlash coherent and faster
merge_llama_into_beellama_2: DFlash corrupt or slower while non-DFlash is stable
```

---

### Task 2: Establish the three-way merge comparison points

**Files:**
- Create: `scripts/dflash-regression/three_way_dflash_diff.sh`
- Modify: `TASK_PROGRESS.md`

**Interfaces:**
- Consumes: git refs `origin/main`, `ggml/master`, and `origin/merge_llama_into_beellama_2`.
- Produces: `/tmp/qwen36-three-way/refs.txt` and focused diff files under `/tmp/qwen36-three-way/`.
- Produces: a `TASK_PROGRESS.md` entry that lists facts separately from hypotheses.

- [ ] **Step 1: Fetch/update upstream master and record refs**

Run:

```bash
git fetch origin --prune
git fetch https://github.com/ggml-org/llama.cpp.git master:refs/remotes/ggml/master --no-tags
mkdir -p /tmp/qwen36-three-way
{
  echo "origin_main=$(git rev-parse origin/main)"
  echo "merge_branch=$(git rev-parse origin/merge_llama_into_beellama_2)"
  echo "ggml_master=$(git rev-parse ggml/master)"
  echo "base_fork_upstream=$(git merge-base origin/main ggml/master)"
  echo "upstream_snapshot_in_merge=$(git merge-base ggml/master origin/merge_llama_into_beellama_2)"
  echo "base_fork_merge=$(git merge-base origin/main origin/merge_llama_into_beellama_2)"
} | tee /tmp/qwen36-three-way/refs.txt
```

Expected: command exits `0`. `base_fork_merge` should equal `origin/main`, proving the merge branch contains the BeeLlama baseline. `upstream_snapshot_in_merge` identifies which upstream snapshot the branch appears to have integrated.

- [ ] **Step 2: Record these as facts in `TASK_PROGRESS.md`**

Append an entry to `TASK_PROGRESS.md` using the required template. Include these facts exactly:

```markdown
### FACTS
- `origin/main` commit: `<hash from refs.txt>`
- `origin/merge_llama_into_beellama_2` commit: `<hash from refs.txt>`
- `ggml/master` commit: `<hash from refs.txt>`
- `merge-base(origin/main, ggml/master)`: `<hash from refs.txt>`
- `merge-base(ggml/master, origin/merge_llama_into_beellama_2)`: `<hash from refs.txt>`
- `merge-base(origin/main, origin/merge_llama_into_beellama_2)`: `<hash from refs.txt>`
```

Under `HYPOTHESES`, write `- None yet.` unless there is already evidence for a specific failing boundary.

- [ ] **Step 3: Create the three-way diff helper**

Create `scripts/dflash-regression/three_way_dflash_diff.sh` with this content:

```bash
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
```

- [ ] **Step 4: Make the helper executable and run it**

Run:

```bash
chmod +x scripts/dflash-regression/three_way_dflash_diff.sh
scripts/dflash-regression/three_way_dflash_diff.sh /tmp/qwen36-three-way
ls -lh /tmp/qwen36-three-way
```

Expected: diff/stat/focus/blame files are created.

- [ ] **Step 5: Summarize the three-way diff as facts only**

Run:

```bash
printf '%s\n' '--- refs ---'
cat /tmp/qwen36-three-way/refs.txt
printf '%s\n' '--- fork delta stat ---'
cat /tmp/qwen36-three-way/fork-delta.stat
printf '%s\n' '--- upstream delta stat ---'
cat /tmp/qwen36-three-way/upstream-delta.stat
printf '%s\n' '--- merge result stat ---'
cat /tmp/qwen36-three-way/merge-result.stat
printf '%s\n' '--- focused merge result ---'
head -160 /tmp/qwen36-three-way/merge-result.focus.txt
```

Expected: this shows which DFlash/Qwen files are changed by the fork, by upstream, and by the merge result. Do not infer root cause yet. Append observed file-level facts to `TASK_PROGRESS.md`.

---

### Task 3: Perform three-way semantic diff/blame of the DFlash data path

**Files:**
- Read-only initially:
  - `TASK_PROGRESS.md`
  - `/tmp/qwen36-three-way/*.diff`
  - `/tmp/qwen36-three-way/*.focus.txt`
  - `/tmp/qwen36-three-way/blame-*.txt`
  - `common/speculative.cpp`
  - `common/speculative.h`
  - `src/llama-context.cpp`
  - `src/models/qwen35.cpp`
  - `src/models/qwen35moe.cpp`
  - `src/models/dflash_draft.cpp`

**Interfaces:**
- Consumes: reproduction results from Task 1 and three-way diff artifacts from Task 2.
- Produces: a written fact/hypothesis analysis in `TASK_PROGRESS.md` and in this plan under `Root Cause Notes`.

- [ ] **Step 1: Read existing progress before analysis**

Run:

```bash
grep -n -E 'HF-03[5-9]|HF-040|PROVEN FACTS|HYPOTH|qwen35|hidden_gpu|n_outputs|graph invalidation|quality regression|single-slot|multi-slot' TASK_PROGRESS.md | head -220
```

Expected: prior investigation notes are visible. Treat prior `PROVEN FACTS` as facts only if they include a concrete test/log/result. Treat prior theories as hypotheses even if the wording sounds confident.

- [ ] **Step 2: Analyze target hidden-state capture semantics**

Compare the fork baseline, upstream delta, and merge result for `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp`:

```bash
for f in fork-delta upstream-delta merge-result; do
  echo "=== $f qwen hidden/output focus ==="
  grep -n -E 'diff --git a/src/models/qwen35|diff --git a/src/models/qwen35moe|hidden_gpu|prefill_gpu|dflash_capture|capture_n|inp_out_ids|embeddings_nextn|nextn|n_layer|build_forward_expand|t_logits|t_embd|result_output|h_nextn' /tmp/qwen36-three-way/$f.diff | head -220
done
```

Record as facts:
- whether BeeLlama's DFlash hidden capture logic exists in `origin/main`;
- whether upstream changed the surrounding graph/output semantics;
- whether the merge result preserved, removed, duplicated, or changed that logic.

Only after those facts are written may the agent add hypotheses such as: `HYPOTHESIS: DFlash hidden capture uses the wrong token span after upstream output-selection changes.`

- [ ] **Step 3: Analyze drafter graph/output semantics**

Run:

```bash
for f in fork-delta upstream-delta merge-result; do
  echo "=== $f dflash_draft focus ==="
  grep -n -E 'diff --git a/src/models/dflash_draft.cpp|llm_build_dflash_draft|llm_build_dflash_kv_update|n_outputs|n_outputs_max|build_arch_graph|logits|output|cross|ctx|ubatch|kv_update' /tmp/qwen36-three-way/$f.diff | head -220
done
```

Record facts about whether drafter output sizing, graph mode selection, or KV update behavior differs between working `origin/main` and the merge result.

- [ ] **Step 4: Analyze verifier/speculative state semantics**

Run:

```bash
for f in fork-delta upstream-delta merge-result; do
  echo "=== $f speculative/verifier focus ==="
  grep -n -E 'diff --git a/common/speculative|diff --git a/src/llama-context.cpp|common_speculative_get_state|common_speculative_set_state|ring_state|draft|verify|accept|rollback|seq_id|id_last|target|logits|cross|hidden|allocate_slots|active_slot' /tmp/qwen36-three-way/$f.diff | head -260
done
```

Record facts about whether single-slot DFlash state, verifier acceptance, rollback, or hidden allocation semantics differ between working `origin/main` and the merge result.

- [ ] **Step 5: Use blame to tie suspicious merge-result lines to commits**

For each suspicious line range identified in Steps 2 to 4, run a blame command. Example for `src/models/qwen35.cpp` lines 150 to 290:

```bash
git blame -L 150,290 origin/merge_llama_into_beellama_2 -- src/models/qwen35.cpp
```

Expected: output identifies whether the suspicious merge-result lines came from upstream, BeeLlama baseline, or merge-port commits such as `c602e87a0`. Record exact commit hashes as facts.

- [ ] **Step 6: Write the fact/hypothesis report**

Append this section to `TASK_PROGRESS.md` and then copy the same content under `Root Cause Notes` at the bottom of this plan:

```markdown
## YYYY-MM-DD HH:MM UTC - Three-way DFlash semantic diff report

### FACTS
- Reproduction fact: `<origin/main DFlash result and perf>`
- Reproduction fact: `<merge branch DFlash result and perf>`
- Three-way ref fact: `<base and upstream snapshot commits>`
- File fact: `<specific file and line/range observation>`
- Blame fact: `<specific line/range comes from commit hash and subject>`

### HYPOTHESES
- HYPOTHESIS 1: `<specific possible root cause tied to facts above>`
- HYPOTHESIS 2: `<optional second possible root cause tied to facts above>`

### TEST RESULTS
- Command: `<exact command>`
- Result: `<pass/fail/measurement>`
- Output files: `/tmp/qwen36-three-way/...`

### NEXT STEPS
- `<one concrete diagnostic or minimal code experiment>`
```

Do not proceed to code changes until at least one hypothesis is tied to exact file/line facts and has a proposed minimal diagnostic.

---

### Task 4: Add targeted diagnostics only at the proven failing boundary

**Files:**
- Modify exactly one boundary area based on Task 3:
  - `src/models/qwen35.cpp` or `src/models/qwen35moe.cpp` for hidden capture graph issues.
  - `src/models/dflash_draft.cpp` for drafter output or graph mode issues.
  - `common/speculative.cpp` and `common/speculative.h` for acceptance/state issues.
  - `src/llama-context.cpp` for hidden allocation, graph invalidation, or rollback issues.

**Interfaces:**
- Produces: logs gated by `GGML_DFLASH_QA_TRACE=1`.
- Must not change behavior when `GGML_DFLASH_QA_TRACE` is unset.

- [ ] **Step 1: Add a local trace gate near the failing boundary**

If the failing boundary is in `common/speculative.cpp`, add this near existing DFlash diagnostic helpers:

```cpp
static bool common_dflash_qa_trace_enabled() {
    static int enabled = []() {
        const char * env = std::getenv("GGML_DFLASH_QA_TRACE");
        return env && std::atoi(env) != 0;
    }();
    return enabled != 0;
}
```

If the failing boundary is in another file, use the same function name with file-local `static` linkage in that file.

- [ ] **Step 2: Log only the minimum values needed to prove the hypothesis**

For a Qwen35 hidden capture boundary, log this shape data before building hidden GPU copies:

```cpp
if (common_dflash_qa_trace_enabled()) {
    fprintf(stderr,
        "[dflash-qa] qwen35 layer=%d hidden_n_seqs=%d cur_ne0=%lld cur_ne1=%lld capture_tokens=%lld capture_seqs=%lld\n",
        il,
        cparams.hidden_gpu_n_seqs,
        (long long) cur->ne[0],
        (long long) cur->ne[1],
        (long long) dflash_capture_n_tokens_dfl,
        (long long) dflash_capture_n_seqs_dfl);
}
```

For a speculative verifier boundary, log this per verification step:

```cpp
if (common_dflash_qa_trace_enabled()) {
    fprintf(stderr,
        "[dflash-qa] verify seq=%d drafted=%d accepted=%d id_last=%d\n",
        (int) seq_id,
        (int) output_len,
        (int) committed_len,
        (int) id_last);
}
```

For a drafter graph boundary, log graph mode and output sizing:

```cpp
if (common_dflash_qa_trace_enabled()) {
    fprintf(stderr,
        "[dflash-qa] draft graph n_outputs=%lld n_outputs_max=%lld n_tokens=%lld\n",
        (long long) params.n_outputs,
        (long long) params.n_outputs_max,
        (long long) params.n_tokens);
}
```

- [ ] **Step 3: Build and run with diagnostics**

Run:

```bash
cmake --build build_vulkan --target llama-cli -j "$(nproc)"
GGML_DFLASH_QA_TRACE=1 scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-qa-trace
```

Expected: trace output appears only for the selected boundary and explains where DFlash diverges.

- [ ] **Step 4: Remove or keep diagnostics behind the env gate**

If the diagnostics are not useful after the fix, remove them. If they catch a class of future regressions and are low noise when disabled, keep them behind `GGML_DFLASH_QA_TRACE=1`.

---

### Task 5: Implement the minimal root-cause fix

**Files:**
- Modify only the file or files proven by Tasks 2 to 4.

**Interfaces:**
- Produces: DFlash greedy output identical to non-DFlash output for the local Qwen3.6 comparison script.
- Preserves: non-DFlash behavior and current Vulkan ring bounds fixes.

- [ ] **Step 1: Choose exactly one fix hypothesis**

Use the root-cause evidence to select one of these fix types:

```text
A. Restore or correct Qwen35 hidden capture graph semantics.
B. Restore or correct drafter graph output sizing.
C. Restore or correct speculative state save/load for the single-slot path.
D. Restore or correct verifier accept/reject logic.
E. Restore or correct Vulkan DFlash copy semantics only if trace proves copied data is corrupted.
```

Expected: exactly one letter is chosen. Do not combine fixes.

- [ ] **Step 2: Apply the smallest code change**

Use one of these minimal strategies based on the chosen hypothesis:

```text
A. Compare `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` to `origin/main`; restore the DFlash hidden capture, prefill capture, output selection, or graph invalidation semantics that the trace proves are missing or wrong.
B. Compare `src/models/dflash_draft.cpp` to `origin/main`; remove the output limiting or graph-mode change that causes draft logits or KV update to use the wrong output span.
C. In `common/speculative.cpp` or `src/llama-context.cpp`, restore state save/load behavior for the single-slot path and only add multi-slot handling where `seq_id` is actually used.
D. In `common/speculative.cpp`, restore the verifier comparison to target logits and reject draft tokens when target top token differs.
E. In `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, adjust only the proven DFlash copy boundary; do not disable Vulkan GPU DFlash ring by default.
```

Expected: one small diff, no broad refactor, no unrelated formatting.

- [ ] **Step 3: Build**

Run:

```bash
cmake --build build_vulkan --target llama-cli -j "$(nproc)"
```

Expected: build succeeds.

- [ ] **Step 4: Run the regression comparison**

Run:

```bash
scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-after-fix
```

Expected: `result=pass-identical-output` and DFlash generation throughput is not below non-DFlash throughput.

- [ ] **Step 5: If the fix fails, revert it and return to Task 3**

Run:

```bash
git diff > /tmp/qwen36-failed-fix.diff
git restore common/speculative.cpp common/speculative.h src/llama-context.cpp src/models/qwen35.cpp src/models/qwen35moe.cpp src/models/dflash_draft.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp 2>/dev/null || true
```

Expected: failed attempt is saved for review, worktree returns to the pre-fix state for the relevant files.

---

### Task 6: Verify non-regression across the existing DFlash tests and the manual model tests

**Files:**
- No new files unless Task 5 required diagnostics that should stay.

**Interfaces:**
- Consumes: fixed code from Task 5.
- Produces: verification evidence in terminal output and `/tmp/qwen36-after-fix/summary.txt`.

- [ ] **Step 1: Run existing unit tests**

Run:

```bash
cmake --build build_vulkan --target test-dflash-ring test-dflash-plumbing -j "$(nproc)"
./build_vulkan/bin/test-dflash-ring
./build_vulkan/bin/test-dflash-plumbing
```

Expected: both tests pass.

- [ ] **Step 2: Run Qwen3.6 local regression comparison**

Run:

```bash
scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-after-fix
cat /tmp/qwen36-after-fix/summary.txt
```

Expected: `result=pass-identical-output`.

- [ ] **Step 3: Run the larger difficult prompt used during investigation**

Run non-DFlash:

```bash
/usr/bin/timeout 1200 ./build_vulkan/bin/llama-cli \
  -m /fast/models/Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --ctx-size 32768 \
  --no-mmap \
  --spec-type none \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --seed 7 \
  --temp 0 \
  -n 384 \
  --single-turn \
  --simple-io \
  -f /tmp/qwen36_27b_large_difficult_prompt.txt \
  >/tmp/qwen36-final-nodflash.out 2>/tmp/qwen36-final-nodflash.err
```

Run DFlash:

```bash
/usr/bin/timeout 1200 ./build_vulkan/bin/llama-cli \
  -m /fast/models/Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --device Vulkan1 \
  -ngl 999 \
  --ctx-size 32768 \
  --no-mmap \
  --spec-type dflash \
  --spec-draft-model /fast/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-device Vulkan1 \
  --spec-draft-ngl 999 \
  --spec-draft-n-max 3 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --seed 7 \
  --temp 0 \
  -n 384 \
  --single-turn \
  --simple-io \
  -f /tmp/qwen36_27b_large_difficult_prompt.txt \
  >/tmp/qwen36-final-dflash.out 2>/tmp/qwen36-final-dflash.err
```

Expected: both exit `0`. DFlash output is coherent and should match non-DFlash under greedy settings or differ only if a documented sampler setting explains the difference. DFlash generation throughput should be at least close to `origin/main` DFlash baseline from this session, around `14 t/s` for the 384-token local prompt.

- [ ] **Step 4: Check for GPU reset evidence**

Run:

```bash
dmesg 2>/dev/null | grep -Ei 'amdgpu|radv|gpu reset|ring timeout|device lost|navi|vulkan' | tail -80 || true
```

Expected: no new device-loss or GPU-reset lines attributable to the test.

---

### Task 7: Prepare the fix for human review

**Files:**
- Modify: this plan file, appending final evidence.
- Do not create commits unless the user explicitly asks.

**Interfaces:**
- Produces: review summary with root cause, fix, tests, and remaining risks.

- [ ] **Step 1: Capture final diff summary**

Run:

```bash
git diff --stat
git diff -- common/speculative.cpp common/speculative.h src/llama-context.cpp src/models/qwen35.cpp src/models/qwen35moe.cpp src/models/dflash_draft.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp scripts/dflash-regression 2>/dev/null | head -240
```

Expected: diff is small and focused.

- [ ] **Step 2: Append final evidence to this plan**

Append:

```markdown
## Final Evidence

**Root cause:** `<specific root cause>`

**Fix summary:** `<specific files and behavior changed>`

**Verification:**
- `test-dflash-ring`: `<pass/fail>`
- `test-dflash-plumbing`: `<pass/fail>`
- Qwen3.6 deterministic DFlash comparison: `<pass/fail and perf>`
- Larger prompt non-DFlash: `<perf>`
- Larger prompt DFlash: `<perf>`
- GPU reset/device loss check: `<result>`

**Remaining risks:** `<specific unresolved risks, or none>`
```

- [ ] **Step 3: Human commit gate**

Stop and ask the user whether they want a commit. If yes, use a concise commit message with `Assisted-by:` and do not push.

---

## Self-Review

**Spec coverage:** This plan covers the requested three-way framing, compares `origin/main`, upstream `ggml/master`, and `merge_llama_into_beellama_2`, reproduces the observed DFlash correctness/performance issue, uses three-way semantic diff/blame before any code change, requires root-cause evidence before fixes, and verifies the fix with both unit tests and local model-backed tests.

**Placeholder scan:** The plan avoids `TBD`, `TODO`, and vague test instructions. The only bracketed fields are in evidence sections that must be filled with measured data during execution.

**Type consistency:** Script interfaces are consistent: `qwen36_compare.sh <output-dir>` writes runtime comparison output and `summary.txt`; `three_way_dflash_diff.sh <output-dir>` writes refs, stats, focused diffs, and blame artifacts.

## Root Cause Notes

### 2026-07-09 18:49 UTC - Code-verified Qwen35 tree-aware verifier graph comparison (gboddaer/main reference)

**FACTS**
- Comparison refs: `gboddaer/main` = `130ea2480` (reference), merge branch HEAD = `c5c17182e`. Code trusted over the prior 19:30 comments.
- Tree-aware op usage in graph builders: `gboddaer/main` has `ggml_ssm_conv_tree`/`ggml_gated_delta_net_tree`/`tree_parent_ids` in qwen35.cpp, qwen35moe.cpp, delta-net-base.cpp (3 matches each); merge branch has 0 in all three.
- qwen35.cpp: hidden_gpu capture PRESENT in merge (matches main). prefill_gpu capture ABSENT in merge (main lines 259-296). tape_gpu capture block ABSENT in merge (main lines 583+). tree_mode conv dispatch ABSENT in merge (main lines 513-528).
- qwen35moe.cpp: ALL THREE capture blocks (hidden_gpu main:237, prefill_gpu main:268, tape_gpu main:593) ABSENT in merge; tree_mode conv dispatch ABSENT (main lines 523-536).
- delta-net-base.cpp build_delta_net tree branch ABSENT in merge (main lines 440-457, `ggml_gated_delta_net_tree` with `tree_ssm_intermediates`/`persist_inter`).
- build_conv_state signature diff (merge always transposes internally; main has `qkv_mixed_transposed` param, called with `true` after an explicit external transpose) VERIFIED EQUIVALENT: the transposed `qkv_mixed` in main is consumed ONLY by build_conv_state; downstream uses the saved `qkv_mixed_pretranspose`. Merge transposes a local copy. Net: one transpose before concat in both. NOT a root cause.
- build_delta_net_fused K argument (merge passes K=1 explicitly; main omits) VERIFIED EQUIVALENT: `K_actual` is derived as 1 from the 4D recurrent state in both. NOT a root cause.
- tape_gpu INFRASTRUCTURE is PRESENT and near-identical on merge: `dflash_tape_gpu_layer::qkv` allocated at llama-context.cpp:2268, `tape_gpu_n_seqs` set from `dflash_capture->tapes` (2106, 2672), graph-invalidation tracking present (6653-7041).
- tape_gpu CONSUME sites PRESENT on merge and read `gpu_layer->qkv` directly: llama-context.cpp:3450, 3800 (`get_tensor_data(gpu_layer->qkv, ...)`), 5671. Gating `use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv` (3756/5648). When true, line 3800 reads `gpu_layer->qkv` which ONLY the missing graph-builder block would write.
- ALTERNATIVE CPU readback path exists on merge: `dflash_read_tensor` (llama-context.cpp:1773) via `tape_name_map` (2060-2071, includes `"linear_attn_qkv_mixed-"+il` matching qwen35.cpp:277) populates `tape.qkv_mixed`; CPU->GPU upload at 3637/3652. Whether the missing GPU block corrupts output depends on `use_gpu_qkv` being true at runtime.
- Existing diagnostic on merge: `[dflash-tr-conv] layer=%d OK use_gpu_qkv=%d ...` (llama-context.cpp:3770), gated by `GGML_DFLASH_DEBUG=1` (env var `GGML_DFLASH_DEBUG`, dflash_diagnostic_debug_enabled at 1346-1352).
- Test-model relevance: Qwen3.6-27B is DENSE -> uses qwen35.cpp (not qwen35moe.cpp). So for the observed regression, the Vulkan-relevant qwen35.cpp differences are: prefill_gpu (missing) and tape_gpu (missing, NEW). qwen35moe.cpp absences only affect MoE models. Tree pieces inert on Vulkan (ops unimplemented in Vulkan backend on both branches).

**HYPOTHESES**
- HYPOTHESIS A (newly elevated, Vulkan-relevant): missing tape_gpu graph-builder capture block in qwen35.cpp causes the corruption. `gpu_layer->qkv` is allocated and, when `use_gpu_qkv==true` on Vulkan, the replay path (llama-context.cpp:3800) reads it unwritten -> corrupt recurrent-state conv replay -> corrupt verifier logits on multi-token batches. Consistent with prior "first bad token from verifier row 0 of a multi-token batch" and "rollback necessary-but-not-sufficient" findings. NOT YET PROVEN: needs `use_gpu_qkv` runtime confirmation.
- HYPOTHESIS B (from 19:30, still open): prefill_gpu capture missing in qwen35.cpp -> stale drafter cross-attention context. May bite longer prompts (for the 17-token test prompt the capture span = whole prompt).
- DISPROVEN for Vulkan (re-affirmed): tree-aware conv/GDN dispatch port fixes the Vulkan regression. Vulkan implements neither tree op on either branch, so `tree_parent_ids` is null on Vulkan and both branches take the identical non-tree path. Tree port is for CUDA/tree-verify completeness only.

**TEST RESULTS**
- Static code comparison only (no build/run). Output files: `/tmp/cmp/{qwen35,qwen35moe,deltanet}.{main,merge}.cpp`.
- New finding vs the 19:30 entry: the tape_gpu graph-builder capture block is missing in BOTH qwen35.cpp and qwen35moe.cpp on merge, and qwen35moe.cpp is missing all three capture blocks. The 19:30 NEXT STEPS list (a)-(e) omitted tape_gpu entirely.

**NEXT STEPS**
- Confirm HYPOTHESIS A at runtime before any code change (systematic debugging): build `llama-cli` (Vulkan) in the worktree; run Qwen3.6-27B DFlash comparison with `GGML_DFLASH_DEBUG=1`; inspect `[dflash-tr-conv] ... use_gpu_qkv=%d ...` lines. `use_gpu_qkv=1` during multi-token verification/replay confirms the missing tape_gpu block is a live bug (gpu_layer->qkv read unwritten). `use_gpu_qkv=0` with a working CPU-readback fallback disproves HYPOTHESIS A for the graph-builder path -> pivot to HYPOTHESIS B / server-side verifier+rollback.
- If HYPOTHESIS A confirmed, apply the minimal port (Task 5 Strategy A, extended): restore tape_gpu capture in qwen35.cpp AND qwen35moe.cpp from `gboddaer/main` (qkv_mixed_pretranspose + k_conv/v_conv/gate/beta_presigmoid copies into tgpu->layers[li]), PLUS prefill_gpu (HYPOTHESIS B) and hidden_gpu for qwen35moe. Tree pieces ported for CUDA completeness but not expected to affect Vulkan.
- If HYPOTHESIS A disproven, save failed diff per Task 5 Step 5 and pivot to the server-side verifier/rollback boundary.
