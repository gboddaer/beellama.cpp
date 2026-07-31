# Handoff: Vulkan DFlash Task 4 — Root Cause Found and Fixed

## Scope

This handoff closes Task 4. The `-np 2` DFlash assertion crash has been resolved.

## Root Cause

**File:** `src/llama-context.cpp`, function `llama_context::decode()`

**Symptom:** With `-np 2` DFlash, the second slot's `post_decode` crashed with:
```
common/sampling.cpp:154: GGML_ASSERT(logits != nullptr) failed
get_logits_ith: invalid logits id 9, reason: batch.logits[9] != true
```

**Trace:**
1. Main decode processes 18 tokens (9 per slot) with all `logits=true` → `output_ids[0..17]` valid, `n_outputs=18`
2. Slot 0's `post_decode` calls `common_sampler_sample_and_accept_n` → succeeds
3. Slot 0 has a rollback (not all drafts accepted) → `llama_dflash_rollback()` + re-eval decode with `logits=false`
4. Re-eval decode calls `output_reserve(0)` → fills `output_ids` with -1, sets `n_outputs=0`
5. Slot 1's `post_decode` calls `common_sampler_sample_and_accept_n` with `spec_i_batch=[9..17]`
6. `llama_get_logits_ith(ctx_tgt, 9)` → `output_ids[9]` is -1 (or `n_outputs=0`) → crash

**Why `-np 1` worked:** Only one slot, so `post_decode` runs before any rollback re-eval clobbers state. Or the rollback happens after `post_decode` completes for the only slot.

**Why `-np 2` failed:** Two slots → slot 0's rollback re-eval clobbers `output_ids`/`n_outputs` → slot 1's `post_decode` sees corrupted state.

## Fix

**File:** `src/llama-context.cpp` — two complementary changes.

### Change 1: `llama_context::decode()` — save/restore `n_outputs`

At function entry, detect if the batch has any logits requests. If not (re-eval case), save `this->n_outputs` before the decode. At function exit, restore `this->n_outputs` if it was saved.

### Change 2: `llama_context::output_reserve()` — skip `output_ids` fill when `n_outputs == 0`

For re-eval batches (all logits=false, n_outputs=0), skip the `std::fill(output_ids, -1)` so output_ids entries from the prior successful decode survive. The decode() caller saves/restores n_outputs around such calls, so the output buffer remains correctly sized.

**Why both changes are needed:**
- Fix 1 alone: `output_ids` is still filled with -1 by `output_reserve`, so `llama_get_logits_ith` still fails.
- Fix 2 alone: `output_ids` survives but `n_outputs=0`, so `output_resolve_row` throws "corrupt output buffer".
- Both together: `output_ids` preserves valid entries AND `n_outputs` is restored, so `output_resolve_row` returns valid rows within the correctly-sized buffer.

## Verification

### `-np 2` concurrent DFlash smoke (the failing case)

```bash
bash /tmp/task4-np2-smoke.sh
```

Result:
```
coding|OK|length|8|108
math|OK|length|67|128
```

Both requests return OK with nonzero `draft_n`. No crash.

### `-np 1` repeated requests (regression check)

```bash
bash /tmp/task4-np1-smoke.sh
```

Result: 3 requests, all with `draft_n=105`, identical valid code output. No regression.

### Harness tests

```
Ran 10 tests in 0.005s — OK
```

### CTest

```
100% tests passed, 0 tests failed out of 6
test-sampling, test-sampling-grammar, test-server-prompt-checkpoint,
test-dflash-ring, test-dflash-plumbing, test-dflash-decode — all passed
```

## Uncommitted code change

Only `src/llama-context.cpp` is intentionally modified (the `decode()` save/restore of `n_outputs`).

The previous uncommitted change to `tools/server/server-context.cpp` (restoring the `n_tokens() > 0` guard around the speculative reset) was reverted via `git checkout`. The committed code at `2b85a8780` has the unconditional reset with virtual hook, which is the intended state.

Do not treat the diagnostic logging as part of the fix — all `DIAG_*` fprintf calls and the `server.cpp` DIAG_BUILD_CHECK were removed.

## Next stages

The plan's Tasks 1-5 (correctness) are now complete:
- Task 1: MTP/DFlash correct across repeated requests ✓
- Task 2: DFlash isolation with `-np 2` ✓
- Task 3: Harness validation ✓
- Task 4: Root cause and fix ✓

Remaining: Task 5 (performance recovery against old-fork reference). This requires building a clean reference worktree at `adb92b36af` and running controlled benchmarks.

## Commands for verification

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2

# Verify the fix is in place
grep -n 'Save n_outputs for re-eval' src/llama-context.cpp

# Build
cmake --build build-vulkan --target llama-server -j"$(nproc)"

# Run -np 2 smoke
bash /tmp/task4-np2-smoke.sh

# Run harness
python3 -m unittest bench.vulkan-gap.test_harness_strict -v

# Run CTest
ctest --test-dir build-vulkan -R 'test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode'
```

## Git state

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- HEAD: `1ce1505d3`
- Uncommitted: `src/llama-context.cpp` only
- Do not commit or push without explicit user approval.
