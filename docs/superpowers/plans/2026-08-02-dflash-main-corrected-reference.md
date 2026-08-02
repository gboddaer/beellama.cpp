# DFlash Main Corrected Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backport the minimum persistent and concurrent speculative correctness behavior onto `130ea2480`, fully qualify it, freeze it as `main-corrected-reference`, and push that branch to `gboddaer` and `boditec` without creating a PR.

**Architecture:** Keep `130ea2480` as the immutable historical control. Add only request reset, prompt-cache isolation, speculative state reset, and target launch serialization to the corrected branch. Use the work branch harness externally, prequalify the dirty source, commit the source, rebuild from the clean commit, run the full final matrix, then commit the qualification record and push only when every gate passes.

**Tech Stack:** C++17, Python 3 standard library, CMake, CTest, Vulkan/RADV GFX1151, llama-server completion API, Git worktrees.

## Global Constraints

- Corrected branch: `main-corrected-reference`.
- Corrected worktree: `/crypt/beellama.cpp/.worktrees/main-corrected-reference`.
- Historical reference: `130ea2480bee8268907a102bd814e276f4e88bdf`.
- Work control: `986ce7aed9b30a00145422c1cbb75da9a6557d05`.
- External harness: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/bench/vulkan-gap`.
- Evidence root: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z`.
- Preserve the historical reference worktree and binary unchanged.
- Do not import graph, verifier, replay, copy, Vulkan queue, scheduler, draft-horizon, or cache-performance changes.
- Use ASCII only in source comments and committed documentation.
- Do not create a PR or push to `origin`.
- Commit and push only after all applicable gates pass.

## Fixed Protocol

```text
VK_GAP_MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
VK_GAP_DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
VK_GAP_BATCH=512
VK_GAP_UBATCH=128
VK_GAP_CTX_SIZE=2048
VK_GAP_NP=1
VK_GAP_TEMP=0
VK_GAP_TOP_K=20
VK_GAP_CACHE_TYPE_K=q4_0
VK_GAP_CACHE_TYPE_V=q4_0
VK_GAP_SPEC_DRAFT_N_MAX=8
VK_GAP_DFLASH_CROSS_CTX=512
VK_GAP_SPEC_BRANCH_BUDGET=0
VK_GAP_NO_CACHE_PROMPT=0
GGML_DFLASH_GPU_RING=0
```

---

### Task 1: Freeze Anchors and Reproduce the Persistent Failure

**Files:**
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/preflight.txt`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/red-{mtp,dflash}.jsonl`
- Do not modify production source.

**Interfaces:**
- Consumes: historical binary, corrected branch at its unmodified base, and work-control harness.
- Produces: pinned provenance and a failing integration test for the stale-state defect.

- [ ] **Step 1: Verify branch and revisions**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
HIST=/crypt/beellama.cpp/.worktrees/dflash-main-130ea2480
WORK=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
mkdir -p "$Q"
test "$(git -C "$CORR" branch --show-current)" = main-corrected-reference
test "$(git -C "$CORR" rev-parse HEAD)" = 130ea2480bee8268907a102bd814e276f4e88bdf
test "$(git -C "$HIST" rev-parse HEAD)" = 130ea2480bee8268907a102bd814e276f4e88bdf
test "$(git -C "$WORK" rev-parse HEAD)" = 986ce7aed9b30a00145422c1cbb75da9a6557d05
test -z "$(git -C "$CORR" status --short --untracked-files=no)"
```

- [ ] **Step 2: Record binary, model, and device provenance**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
HIST=/crypt/beellama.cpp/.worktrees/dflash-main-130ea2480
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
{
  echo "historical_head=$(git -C "$HIST" rev-parse HEAD)"
  echo "corrected_base=$(git -C "$CORR" rev-parse HEAD)"
  echo "historical_server_sha256=$(sha256sum "$HIST/build-vulkan-parity/bin/llama-server" | awk '{print $1}')"
  echo "model_sha256=$(sha256sum /crypt/models/Qwen3.6-27B-Q4_K_M.gguf | awk '{print $1}')"
  echo "draft_sha256=$(sha256sum /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf | awk '{print $1}')"
  "$HIST/build-vulkan-parity/bin/llama-server" --version
  "$HIST/build-vulkan-parity/bin/llama-server" --list-devices
} | tee "$Q/preflight.txt"
grep -q 'Vulkan0: AMD Radeon Graphics (RADV GFX1151)' "$Q/preflight.txt"
```

- [ ] **Step 3: Run model-free harness tests**

```bash
WORK=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
cd "$WORK"
python3 -m unittest bench.vulkan-gap.test_harness_strict bench.vulkan-gap.test_repeated_requests -v
```

Expected: all tests pass.

- [ ] **Step 4: Run the failing persistent MTP and DFlash tests**

```bash
HIST=/crypt/beellama.cpp/.worktrees/dflash-main-130ea2480
WORK=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
export VK_GAP_SERVER="$HIST/build-vulkan-parity/bin/llama-server"
export VK_GAP_MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
export VK_GAP_DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
export VK_GAP_BATCH=512 VK_GAP_UBATCH=128 VK_GAP_CTX_SIZE=2048
export VK_GAP_NP=1 VK_GAP_TEMP=0 VK_GAP_TOP_K=20
export VK_GAP_CACHE_TYPE_K=q4_0 VK_GAP_CACHE_TYPE_V=q4_0
export VK_GAP_SPEC_DRAFT_N_MAX=8 VK_GAP_DFLASH_CROSS_CTX=512
export VK_GAP_SPEC_BRANCH_BUDGET=0 VK_GAP_NO_CACHE_PROMPT=0
export GGML_DFLASH_GPU_RING=0
unset GGML_DFLASH_FORCE_REDECODE GGML_DFLASH_PROFILE GGML_VK_PERF_LOGGER
set +e
VK_GAP_PORT=18300 python3 "$WORK/bench/vulkan-gap/run.py" mtp coding \
  --repetitions 1 --gen-tokens 256 \
  --reference-label historical-130ea2480 \
  --external-source-head 130ea2480bee8268907a102bd814e276f4e88bdf \
  --output "$Q/red-mtp.jsonl"
mtp_status=$?
VK_GAP_PORT=18301 python3 "$WORK/bench/vulkan-gap/run.py" dflash coding \
  --repetitions 1 --gen-tokens 256 \
  --reference-label historical-130ea2480 \
  --external-source-head 130ea2480bee8268907a102bd814e276f4e88bdf \
  --output "$Q/red-dflash.jsonl"
dflash_status=$?
set -e
test "$mtp_status" -eq 1
test "$dflash_status" -eq 1
python3 - <<'PY'
import json, pathlib
q = pathlib.Path('/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z')
for mode in ('mtp', 'dflash'):
    row = json.loads((q / f'red-{mode}.jsonl').read_text().splitlines()[0])
    assert row['measurement']['valid'] is False
    assert 'prompt_echo' in row['measurement']['invalid_reasons']
print('RED reproduced: persistent speculative prompt echo')
PY
```

Expected: both runs fail exactly because the post-warmup measured row contains `prompt_echo`.

---

### Task 2: Implement the Minimal Correctness Backport

**Files:**
- Modify: `common/speculative.h`
- Modify: `common/speculative.cpp`
- Modify: `tools/server/server-context.cpp`
- Test through: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/prequal-*.jsonl`

**Interfaces:**
- Produces: `common_speculative_reset(common_speculative *, llama_seq_id)` and serialized clean speculative launches.
- Preserves: BASE scheduling and all DFlash compute/performance paths.

- [ ] **Step 1: Add the speculative per-request reset declaration**

Add immediately before `common_speculative_begin()` in `common/speculative.h`:

```cpp
// reset per-request state before reusing a speculative slot
void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id);
```

- [ ] **Step 2: Add the minimal reset implementation**

Add after `common_speculative_get_draft_params()` in `common/speculative.cpp`:

```cpp
void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        switch (impl->type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(impl.get());
                mtp->pending_h[seq_id].assign(mtp->n_embd, 0.0f);
                mtp->i_batch_beg[seq_id] = -1;
                mtp->i_batch_end[seq_id] = -1;
                mtp->verify_h[seq_id].clear();
                mtp->verify_h_rows[seq_id] = 0;
                mtp->last_n_drafted[seq_id] = 0;
            } break;
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                auto * dflash = static_cast<common_speculative_impl_dflash *>(impl.get());
                dflash->discard_cross_ring("new request");
            } break;
            default:
                break;
        }
    }
}
```

- [ ] **Step 3: Clear slot state during reset**

Inside `server_slot::reset()`, extend the existing `if (can_speculate())` block:

```cpp
        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_pad_i_batch.clear();
            spec_ckpt.clear();
            prompt_clear(false);
        }
```

- [ ] **Step 4: Bypass prompt-cache restore for speculative slots**

In `get_available_slot()`, replace the prompt save/load block with:

```cpp
                if (ret->can_speculate()) {
                    ret->prompt_clear(true);
                } else {
                    if (tokens.size() > 0) {
                        ret->prompt_save(*prompt_cache);
                    }

                    if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                        ret->prompt_clear(false);
                    }
                }
```

Keep the existing prompt-cache update and recurrent shrink/expand calls around this block.

- [ ] **Step 5: Reset draft KV and implementation state at launch**

Add at the beginning of `launch_slot_with_task()`:

```cpp
        if (slot.can_speculate()) {
            if (slot.ctx_dft) {
                common_context_seq_rm(slot.ctx_dft, slot.id, -1, -1);
            }
            common_speculative_reset(slot.get_spec(), slot.id);
        }
```

- [ ] **Step 6: Serialize speculative launches and clear target memory**

Add next to the other server scheduling helpers:

```cpp
    bool any_slot_processing() const {
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                return true;
            }
        }
        return false;
    }
```

Before `task.is_parent()` in `process_single_task()`, add:

```cpp
                    if (slot->can_speculate() && any_slot_processing()) {
                        queue_tasks.defer(std::move(task));
                        break;
                    }
```

Immediately before each speculative parent or single launch, add:

```cpp
                        if (slot->can_speculate()) {
                            llama_memory_clear(llama_get_memory(ctx_tgt), true);
                        }
```

Do not clear target memory while another slot is processing.

- [ ] **Step 7: Check the source scope**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
git -C "$CORR" diff --check
git -C "$CORR" diff --stat
changed=$(git -C "$CORR" diff --name-only | grep -v '^docs/superpowers/' || true)
test "$changed" = $'common/speculative.cpp\ncommon/speculative.h\ntools/server/server-context.cpp'
```

---

### Task 3: Build and Prequalify the Dirty Source

**Files:**
- Create: `build-vulkan-corrected/`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/prequal-*.jsonl`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/source.patch`

**Interfaces:**
- Consumes: Task 2 source.
- Produces: a correctness-qualified source patch suitable for the source commit.

- [ ] **Step 1: Configure and build**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
cmake -S "$CORR" -B "$CORR/build-vulkan-corrected" \
  -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$CORR/build-vulkan-corrected" --target \
  llama-server test-dflash-ring test-dflash-plumbing -j"$(nproc)"
```

- [ ] **Step 2: Run every DFlash CTest present at the reference**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
ctest --test-dir "$CORR/build-vulkan-corrected" --output-on-failure -R \
'^(test-dflash-ring|test-dflash-plumbing)$'
```

Expected: all tests pass.

- [ ] **Step 3: Re-run the persistent MTP and DFlash tests**

Use the Task 1 environment with:

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
VK_GAP_PORT=18310 python3 "$WORK/bench/vulkan-gap/run.py" mtp coding \
  --repetitions 2 --gen-tokens 256 --output "$Q/prequal-mtp.jsonl"
VK_GAP_PORT=18311 python3 "$WORK/bench/vulkan-gap/run.py" dflash coding \
  --repetitions 2 --gen-tokens 256 --output "$Q/prequal-dflash.jsonl"
```

Expected: four measured rows are valid and speculative draft counts are nonzero.

- [ ] **Step 4: Run short concurrent smoke gates**

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
VK_GAP_PORT=18312 python3 "$WORK/bench/vulkan-gap/repeated_requests.py" mtp \
  --np 2 --rounds 1 --concurrent --gen-tokens 512 --output "$Q/prequal-mtp-concurrent.json"
VK_GAP_PORT=18313 python3 "$WORK/bench/vulkan-gap/repeated_requests.py" dflash \
  --np 2 --rounds 1 --concurrent --gen-tokens 512 --output "$Q/prequal-dflash-concurrent.json"
```

Expected: four responses per mode are structurally valid and contamination-free.

- [ ] **Step 5: Freeze and verify the prequalified source patch**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
git -C "$CORR" diff -- common/speculative.cpp common/speculative.h tools/server/server-context.cpp > "$Q/source.patch"
sha256sum "$Q/source.patch" | tee "$Q/source-patch.sha256"
git -C "$CORR" diff --check
```

- [ ] **Step 6: Commit the source only**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
cd "$CORR"
git add common/speculative.cpp common/speculative.h tools/server/server-context.cpp
git commit -m $'server: isolate speculative state between requests\n\nAssisted-by: OpenAI'
```

Do not commit the design, plan, build tree, or evidence yet.

---

### Task 4: Rebuild the Clean Commit and Run the Full Qualification Matrix

**Files:**
- Recreate: `build-vulkan-corrected/`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/{historical,corrected}-*.jsonl`
- Create: concurrent JSON results and logs.

**Interfaces:**
- Consumes: clean source commit from Task 3.
- Produces: final performance, correctness, and deterministic-output evidence.

- [ ] **Step 1: Clean-rebuild from the committed source**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
rm -rf "$CORR/build-vulkan-corrected"
cmake -S "$CORR" -B "$CORR/build-vulkan-corrected" \
  -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$CORR/build-vulkan-corrected" --target \
  llama-server test-dflash-ring test-dflash-plumbing -j"$(nproc)"
ctest --test-dir "$CORR/build-vulkan-corrected" --output-on-failure -R \
'^(test-dflash-ring|test-dflash-plumbing)$'
```

- [ ] **Step 2: Record clean binary provenance**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
{
  echo "source_commit=$(git -C "$CORR" rev-parse HEAD)"
  echo "source_patch_sha256=$(sha256sum "$Q/source.patch" | awk '{print $1}')"
  sha256sum "$CORR/build-vulkan-corrected/bin/llama-server"
  sha256sum "$CORR/build-vulkan-corrected/bin/libllama.so.0"
  "$CORR/build-vulkan-corrected/bin/llama-server" --version
  "$CORR/build-vulkan-corrected/bin/llama-server" --list-devices
} | tee "$Q/corrected-provenance.txt"
grep -q 'Vulkan0: AMD Radeon Graphics (RADV GFX1151)' "$Q/corrected-provenance.txt"
```

- [ ] **Step 3: Capture five independent historical rows per mode**

Use the fixed protocol and historical server, then run:

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/dflash-main-130ea2480/build-vulkan-parity/bin/llama-server
for item in 'base 18320' 'mtp 18321' 'dflash 18322'; do
  set -- $item; mode=$1; export VK_GAP_PORT=$2
  python3 "$WORK/bench/vulkan-gap/run.py" "$mode" coding \
    --repetitions 5 --gen-tokens 256 --skip-warmup --restart-between-reps \
    --reference-label historical-130ea2480 \
    --external-source-head 130ea2480bee8268907a102bd814e276f4e88bdf \
    --output "$Q/historical-$mode-coding.jsonl"
done
```

Expected: all 15 first-request rows are valid.

- [ ] **Step 4: Capture five independent corrected first-request rows per mode**

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
for item in 'base 18330' 'mtp 18331' 'dflash 18332'; do
  set -- $item; mode=$1; export VK_GAP_PORT=$2
  python3 "$WORK/bench/vulkan-gap/run.py" "$mode" coding \
    --repetitions 5 --gen-tokens 256 --skip-warmup --restart-between-reps \
    --reference-label main-corrected-reference-first \
    --external-source-head "$(git -C "$CORR" rev-parse HEAD)" \
    --output "$Q/corrected-first-$mode-coding.jsonl"
done
```

Expected: all 15 first-request rows are valid and token/content hashes are deterministic within each mode. Report historical/corrected hash differences without filtering.

- [ ] **Step 5: Capture five corrected persistent rows per mode**

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
for item in 'base 18335' 'mtp 18336' 'dflash 18337'; do
  set -- $item; mode=$1; export VK_GAP_PORT=$2
  python3 "$WORK/bench/vulkan-gap/run.py" "$mode" coding \
    --repetitions 5 --gen-tokens 256 \
    --reference-label main-corrected-reference-persistent \
    --external-source-head "$(git -C "$CORR" rev-parse HEAD)" \
    --output "$Q/corrected-persistent-$mode-coding.jsonl"
done
```

Expected: all 15 persistent rows are valid. Report row 1 separately and require rows 2-5 to be token/content-hash-identical within each mode.

- [ ] **Step 6: Run final concurrent MTP and DFlash gates**

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
VK_GAP_PORT=18340 python3 "$WORK/bench/vulkan-gap/repeated_requests.py" mtp \
  --np 2 --rounds 3 --concurrent --gen-tokens 512 --output "$Q/corrected-mtp-concurrent.json"
VK_GAP_PORT=18341 python3 "$WORK/bench/vulkan-gap/repeated_requests.py" dflash \
  --np 2 --rounds 3 --concurrent --gen-tokens 512 --output "$Q/corrected-dflash-concurrent.json"
```

Expected: six responses per mode, all structurally valid, no cross-contamination, and nonzero draft counts.

- [ ] **Step 7: Capture a trace-visible corrected DFlash profile**

```bash
export VK_GAP_SERVER=/crypt/beellama.cpp/.worktrees/main-corrected-reference/build-vulkan-corrected/bin/llama-server
export VK_GAP_PORT=18342
export GGML_DFLASH_PROFILE=summary,verify GGML_DFLASH_PROFILE_SYNC_SPLIT=1
export LLAMA_ARG_LOG_VERBOSITY=4
python3 "$WORK/bench/vulkan-gap/run.py" dflash coding \
  --repetitions 1 --gen-tokens 256 --skip-warmup \
  --reference-label main-corrected-reference \
  --external-source-head "$(git -C "$CORR" rev-parse HEAD)" \
  --output "$Q/corrected-dflash-profile.jsonl"
unset GGML_DFLASH_PROFILE GGML_DFLASH_PROFILE_SYNC_SPLIT LLAMA_ARG_LOG_VERBOSITY
```

Assert that the referenced log contains `dflash profile: decode=` and `verify_sync_split=`.

---

### Task 5: Analyze the Qualification and Apply the Freeze Gate

**Files:**
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/analyze.py`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/summary.json`
- Create: `/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/analysis.txt`

**Interfaces:**
- Consumes: Task 4 JSONL and concurrent JSON.
- Produces: one exact `qualified` or `rejected` decision.

- [ ] **Step 1: Write the analyzer**

The analyzer must:

1. Load five historical, five corrected first-request, and five corrected persistent rows for each mode.
2. Assert every coding row is valid.
3. Assert MTP and DFlash draft counts are nonzero.
4. Calculate historical, corrected first-request, and corrected persistent median `predicted_per_second`.
5. Calculate corrected-first/historical ratios for BASE, MTP, and DFlash.
6. Require corrected-first token and content hashes to be deterministic within each mode. Report every historical/corrected-first hash difference without treating a valid changed output as hidden evidence.
7. Report every corrected-persistent token/content hash without duration or length filtering. Require rows 2-5 to be hash-identical within each mode and report row 1 as the post-warmup transition.
8. Validate both concurrent files for six responses, nonzero tokens/drafts, and no cross-prompt contamination.
9. Apply these exact performance gates:

```python
0.95 <= corrected_first['base'] / historical['base'] <= 1.05
0.95 <= corrected_first['mtp'] / historical['mtp'] <= 1.05
0.95 <= corrected_first['dflash'] / historical['dflash'] <= 1.05
corrected_persistent['dflash'] >= 28.457
corrected_persistent['dflash'] / corrected_persistent['base'] >= 1.5
```

10. Write `classification: qualified` only if every correctness and performance gate passes; otherwise write `classification: rejected` and exit 1.

- [ ] **Step 2: Run the analyzer**

```bash
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
cd "$Q"
python3 analyze.py 2>&1 | tee analysis.txt
```

Expected: exit 0 and `classification: qualified`.

- [ ] **Step 3: Verify source and branch cleanliness before recording**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
test "$(git -C "$CORR" branch --show-current)" = main-corrected-reference
test -z "$(git -C "$CORR" status --short --untracked-files=no)"
git -C "$CORR" diff --check
python3 -c 'import json; assert json.load(open("/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/summary.json"))["classification"] == "qualified"'
```

---

### Task 6: Record, Commit, Verify, and Push the Frozen Reference

**Files:**
- Create: `docs/superpowers/milestones/M1-main-corrected-reference-qualified.md`
- Existing uncommitted design and plan are included in the record commit.

**Interfaces:**
- Consumes: qualified summary and clean source commit.
- Produces: frozen branch on both approved remotes.

- [ ] **Step 1: Write the qualification milestone**

Record:

```text
historical base SHA and binary hash
corrected source commit and source patch hash
corrected server and libllama hashes
model and draft hashes
Vulkan device
historical and corrected BASE/MTP/DFlash medians
all corrected/historical ratios
corrected DFlash/BASE ratio
persistent valid counts
concurrent valid counts and contamination result
profile marker presence
source file list
final objectives verbatim
classification: qualified
```

Point to the exact evidence root and every summary file.

- [ ] **Step 2: Self-review committed documentation**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
python3 - <<'PY'
from pathlib import Path
root = Path('/crypt/beellama.cpp/.worktrees/main-corrected-reference/docs/superpowers')
for path in root.rglob('*.md'):
    text = path.read_text()
    assert not any(ord(c) > 127 for c in text), path
    for bad in ('T' + 'BD', 'T' + 'ODO', 'production-' + 'verified'):
        assert bad not in text, (path, bad)
print('documentation self-review passed')
PY
git -C "$CORR" diff --check
```

- [ ] **Step 3: Commit the qualification record**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
cd "$CORR"
git add docs/superpowers/specs/2026-08-02-dflash-main-corrected-reference-design.md \
        docs/superpowers/plans/2026-08-02-dflash-main-corrected-reference.md \
        docs/superpowers/milestones/M1-main-corrected-reference-qualified.md
git commit -m $'docs: freeze corrected DFlash reference qualification\n\nAssisted-by: OpenAI'
```

- [ ] **Step 4: Run final verification after the record commit**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
Q=/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
test -z "$(git -C "$CORR" status --short)"
git -C "$CORR" diff --check
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
python3 -m unittest bench.vulkan-gap.test_harness_strict bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir "$CORR/build-vulkan-corrected" --output-on-failure -R \
'^(test-dflash-ring|test-dflash-plumbing)$'
python3 -c 'import json; assert json.load(open("/crypt/tmp/dflash-main-corrected-reference-20260802T085318Z/summary.json"))["classification"] == "qualified"'
```

- [ ] **Step 5: Push the corrected branch to both approved remotes**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
git -C "$CORR" push gboddaer main-corrected-reference
git -C "$CORR" push boditec main-corrected-reference
```

Do not push to `origin`. Do not create a PR.

- [ ] **Step 6: Verify both remote heads**

```bash
CORR=/crypt/beellama.cpp/.worktrees/main-corrected-reference
head=$(git -C "$CORR" rev-parse HEAD)
test "$(git -C "$CORR" ls-remote gboddaer refs/heads/main-corrected-reference | awk '{print $1}')" = "$head"
test "$(git -C "$CORR" ls-remote boditec refs/heads/main-corrected-reference | awk '{print $1}')" = "$head"
echo "main-corrected-reference frozen at $head"
```

## Final Objectives

```text
work DFlash median >= 28.457 t/s
work DFlash median >= 1.5x work BASE
work/corrected-reference BASE ratio between 0.95 and 1.05
BASE and MTP correctness and performance preserved
all measured BASE, MTP, and DFlash outputs valid
final BASE, MTP, and DFlash persistent gates pass on one server without restart
final MTP and DFlash concurrent gates pass without cross-request text or token contamination
```
