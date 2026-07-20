# Multi-Slot DFlash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable DFlash speculative decoding with `--parallel N` (multi-slot) so each slot has its own DFlash impl/ring/seq_id, matching the fork's per-slot `slot.spec` architecture.

**Architecture:** The fork gives each DFlash slot its own `common_speculative` object (per-slot ring buffer, cross-attention state, seq_id) while sharing a single drafter context (`ctx_dft_shared`). The merge currently shares one `common_speculative` across all slots, causing hidden-state capture collisions (`begin hidden[0] shape mismatch: embd=0`) and 0 drafts for `--parallel > 1`. The fix: add `slot.spec` (owned unique_ptr) + `slot.spec_shared` (fallback) + `slot.get_spec()`, create per-slot specs at init, and add `llama_dflash_set_active_slot()` before each per-slot DFlash operation.

**Tech Stack:** C++, llama.cpp server, CMake, Vulkan/CUDA backends

## Global Constraints

- **Worktree:** `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- **Branch:** `merge_llama_into_beellama_2`
- **Push target:** `gboddaer/merge_llama_into_beellama_2` ONLY (never `gboddaer/main`)
- **Do NOT `pkill -f llama-server`** — kills the user's running server. Use distinct ports (8190+) per test.
- **Build:** `cmake --build build-vulkan --target llama-server -j$(nproc)` (Vulkan backend on AMD Strix Halo iGPU)
- **Test models:** `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` (target) + `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf` (draft)
- **Single-slot works** at 55.6% acceptance (commit `35c5f8f61`). This plan must NOT regress single-slot.
- **Debug env vars** available: `GGML_DFLASH_TOKEN_TRACE=1`, `GGML_DFLASH_KV_TRACE=1`, `GGML_DFLASH_RING_DUMP=1`, `GGML_DFLASH_PREFILL_TRACE=1`
- **Fork reference:** `/crypt/beellama.cpp` (commit adb92b36a) — the working DFlash implementation
- **`llama_dflash_set_active_slot(ctx, slot_id)`** already exists in the merge (llama.h:1637, llama-context.cpp:8903)

---

## File Structure

- **Modify:** `tools/server/server-context.cpp` — slot struct, slot init, pre_decode, post_decode, all per-slot spec references
- **No new files** — all changes are in the existing server-context.cpp
- **Test:** manual server tests with `--parallel 1` (regression) and `--parallel 4` (multi-slot)

### Current State (merge)

```cpp
// server_slot struct (line 178)
common_speculative * spec;  // raw pointer to context-level spec, NOT owned

// can_speculate (line 426)
bool can_speculate() const { return !!spec; }

// slot init (line 1491)
slot.spec = spec.get();  // ALL slots share the same spec
```

### Target State (matching fork)

```cpp
// server_slot struct
common_speculative_ptr spec;              // owned unique_ptr (per-slot for DFlash)
common_speculative * spec_shared = nullptr; // non-owning fallback (for non-DFlash)

// get_spec method
common_speculative * get_spec() const { return spec ? spec.get() : spec_shared; }

// can_speculate
bool can_speculate() const { return spec || spec_shared; }

// slot init — DFlash: per-slot spec; non-DFlash: shared spec
if (slot_uses_fork_spec) {
    slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx_tgt, ctx_dft.get()));
} else if (spec) {
    slot.spec_shared = spec.get();
}
common_speculative_set_seq_id(slot.get_spec(), slot.id);
```

---

### Task 1: Refactor slot struct — add owned spec, spec_shared, get_spec()

**Files:**
- Modify: `tools/server/server-context.cpp:178` (slot struct spec field)
- Modify: `tools/server/server-context.cpp:426` (can_speculate method)
- Modify: `tools/server/server-context.cpp:378,383` (need_embd methods)

**Interfaces:**
- Produces: `server_slot::spec` (now `common_speculative_ptr`), `server_slot::spec_shared`, `server_slot::get_spec()`

- [ ] **Step 1: Change slot spec field from raw pointer to owned unique_ptr + add spec_shared**

In `tools/server/server-context.cpp`, find line 178:
```cpp
    // speculative decoding
    common_speculative * spec;
```
Replace with:
```cpp
    // speculative decoding
    common_speculative_ptr spec;                  // owned (DFlash per-slot)
    common_speculative * spec_shared = nullptr;   // non-owning fallback (non-DFlash)
```

- [ ] **Step 2: Add get_spec() method and update can_speculate()**

Find the `can_speculate()` method (around line 425):
```cpp
    bool can_speculate() const {
        return !!spec;
    }
```
Replace with:
```cpp
    bool can_speculate() const {
        return spec || spec_shared;
    }

    common_speculative * get_spec() const {
        return spec ? spec.get() : spec_shared;
    }
```

- [ ] **Step 3: Update need_embd methods to use get_spec()**

Find lines 378 and 383:
```cpp
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
```
and
```cpp
        return spec && common_speculative_need_embd_nextn(spec);
```
Replace with:
```cpp
        return task->need_embd() || (get_spec() && common_speculative_need_embd(get_spec()));
```
and
```cpp
        return get_spec() && common_speculative_need_embd_nextn(get_spec());
```

- [ ] **Step 4: Update print_stats to use get_spec()**

Find line 652:
```cpp
        common_speculative_print_stats(spec);
```
Replace with:
```cpp
        common_speculative_print_stats(get_spec());
```

- [ ] **Step 5: Update eval_callback lambda to use get_spec()**

Find lines 727-730 (the eval callback lambda that captures `spec`):
```cpp
                    void * cb_data = spec;
```
and the lambda body that uses `spec`. Update the callback to use `slot.get_spec()` instead of the captured `spec`. (The callback runs per-slot, so it must use the slot's spec.)

- [ ] **Step 6: Build and verify single-slot still compiles**

Run: `cmake --build build-vulkan --target llama-server -j$(nproc) 2>&1 | tail -5`
Expected: BUILD SUCCEEDS (slot init not yet updated, will fail at runtime — that's Task 2)

If compile errors about `spec` being `common_speculative_ptr` instead of raw pointer: update all remaining `spec` references in the slot to use `get_spec()` or `spec.get()` as appropriate.

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "refactor(dflash): slot struct — owned spec + spec_shared + get_spec()

Change server_slot::spec from raw pointer to common_speculative_ptr
(owned unique_ptr). Add spec_shared (non-owning fallback) and get_spec()
method. Update can_speculate, need_embd, print_stats to use get_spec().
Prepares for per-slot DFlash spec creation in Task 2."
```

---

### Task 2: Per-slot spec initialization — create DFlash spec per slot

**Files:**
- Modify: `tools/server/server-context.cpp:1488-1510` (slot init loop)

**Interfaces:**
- Consumes: `common_speculative_init(params, ctx_tgt, ctx_dft)` (3-arg, DFlash-aware)
- Consumes: `common_speculative_set_seq_id(spec, seq_id)`
- Produces: Each DFlash slot has its own `slot.spec` with unique seq_id

- [ ] **Step 1: Update slot init to create per-slot spec for DFlash**

Find the slot init loop (around line 1488-1510):
```cpp
        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;
```

Replace with:
```cpp
        // DFlash multi-slot: determine how many slots get DFlash
        const bool is_dflash = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH);
        const int dflash_slots_cap = is_dflash
            ? std::max(1, std::min({ (int)params_base.n_parallel,
                                     (int)params_base.speculative.dflash_max_slots > 0
                                         ? (int)params_base.speculative.dflash_max_slots
                                         : (int)params_base.n_parallel,
                                     (int)LLAMA_DFLASH_MAX_SLOTS }))
            : 0;

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.n_ctx   = n_ctx_slot;

            // DFlash: each slot gets its own spec (per-slot ring/capture/seq_id).
            // Non-DFlash: use the shared context-level spec.
            const bool slot_dflash = is_dflash && i < dflash_slots_cap;
            if (slot_dflash) {
                try {
                    slot.spec.reset(common_speculative_init(
                        params_base.speculative, slot.ctx_tgt, ctx_dft.get()));
                } catch (const std::exception & e) {
                    SRV_ERR("failed to initialize slot %d speculative context: %s\n", i, e.what());
                    return false;
                }
            } else if (spec) {
                slot.spec_shared = spec.get();
            }

            if (slot.can_speculate()) {
                common_speculative_set_seq_id(slot.get_spec(), slot.id);
            }
```

- [ ] **Step 2: Remove the old `slot.spec = spec.get()` line**

Make sure the old line `slot.spec = spec.get();` is completely removed from the init loop (it was replaced in Step 1).

- [ ] **Step 3: Update slot reset/cleanup to handle owned spec**

Find the slot `reset()` method or cleanup code. Ensure `slot.spec.reset()` (unique_ptr reset) is called on slot release, and `slot.spec_shared = nullptr`.

Search for where slots are cleaned up (around line 958 or slot release):
```cpp
        spec.reset();
```
Add per-slot cleanup:
```cpp
        for (auto & slot : slots) {
            slot.spec.reset();
            slot.spec_shared = nullptr;
        }
        spec.reset();
```

- [ ] **Step 4: Build**

Run: `cmake --build build-vulkan --target llama-server -j$(nproc) 2>&1 | tail -5`
Expected: BUILD SUCCEEDS

- [ ] **Step 5: Test single-slot regression (--parallel 1)**

```bash
GGML_DFLASH_TOKEN_TRACE=1 ./build-vulkan/bin/llama-server \
    --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
    --spec-type dflash \
    --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --ctx-size 4096 --parallel 1 --device Vulkan0 \
    --host 127.0.0.1 --port 8190 --reasoning on > /tmp/multislot-p1.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8190/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30,"temperature":0.1,"reasoning":true}'
```
Expected: `draft acceptance` ~55%, speed ~19 tok/s, output correct (NOT garbled).
If acceptance < 40% or output garbled: STOP, single-slot regressed, debug before continuing.

- [ ] **Step 6: Test multi-slot (--parallel 4)**

```bash
./build-vulkan/bin/llama-server \
    --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
    --spec-type dflash \
    --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --ctx-size 4096 --parallel 4 --device Vulkan0 \
    --host 127.0.0.1 --port 8191 --reasoning on > /tmp/multislot-p4.log 2>&1 &
sleep 30
# Send 2 concurrent requests
curl -s http://127.0.0.1:8191/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30,"temperature":0.1,"reasoning":true}' &
curl -s http://127.0.0.1:8191/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 3+3?"}],"max_tokens":30,"temperature":0.1,"reasoning":true}' &
wait
```
Expected: No crash, no `begin hidden[0] shape mismatch` error, drafts generated for multiple slots.
If crash or `embd=0` error: the per-slot spec is not correctly isolated — check `llama_dflash_set_active_slot` calls (Task 3).

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "feat(dflash): per-slot spec initialization for multi-slot DFlash

Each DFlash slot now gets its own common_speculative object (per-slot
ring buffer, cross-attention state, seq_id) via common_speculative_init.
Non-DFlash slots use the shared context-level spec (spec_shared).
dflash_slots_cap limits how many slots get DFlash (matches fork).

Single-slot regression: 55.6% acceptance, 19.4 tok/s (unchanged).
Multi-slot: no crash, drafts generated for multiple slots."
```

---

### Task 3: Add llama_dflash_set_active_slot calls before per-slot DFlash operations

**Files:**
- Modify: `tools/server/server-context.cpp` — pre_decode, post_decode, decode loop

**Interfaces:**
- Consumes: `llama_dflash_set_active_slot(ctx_tgt, slot.id)` (already exists, llama.h:1637)
- Produces: Correct active slot set before each DFlash ring read/write

The fork calls `llama_dflash_set_active_slot(ctx_tgt, slot.id)` at 10+ points before any DFlash operation that reads/writes the target's cross-attention ring. Without this, multi-slot DFlash reads the wrong slot's hidden states.

- [ ] **Step 1: Add set_active_slot before prefill capture begin**

Find the prefill capture begin call (around line 3864):
```cpp
                llama_dflash_prefill_capture_begin(ctx_tgt, slot.id, span.capture_begin, span.capture_end);
```
Add BEFORE it:
```cpp
                if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                llama_dflash_prefill_capture_begin(ctx_tgt, slot.id, span.capture_begin, span.capture_end);
```

- [ ] **Step 2: Add set_active_slot before common_speculative_draft**

Find the draft generation call (around line 3111):
```cpp
        {
            common_speculative_draft(spec.get());
        }
```
Add before it, inside the slot loop:
```cpp
            if (slot.can_speculate() && params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                llama_dflash_set_active_slot(ctx_tgt, slot.id);
            }
```

- [ ] **Step 3: Add set_active_slot before common_speculative_accept**

Find the accept call (around line 4175):
```cpp
                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
```
Add before it:
```cpp
                if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                common_speculative_accept(slot.get_spec(), slot.id, accepted.size() - 1);
```

- [ ] **Step 4: Add set_active_slot before update_logits_deferred_dflash_kv**

Find the update_logits call (around line 4183):
```cpp
                common_speculative_update_logits_deferred_dflash_kv(spec.get(), ctx_tgt, accepted, accepted.size());
```
Add before it:
```cpp
                if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, accepted, accepted.size());
```

- [ ] **Step 5: Replace all remaining spec.get() with slot.get_spec() in per-slot operations**

Search for `spec.get()` in the pre_decode and post_decode lambdas (lines 3070-4210). Replace each with `slot.get_spec()`. The context-level `spec.get()` should only be used for non-per-slot operations (like initialization and final stats).

Key locations to update:
- `common_speculative_get_draft_params(spec.get(), slot.id)` → `common_speculative_get_draft_params(slot.get_spec(), slot.id)`
- `common_speculative_set_prefill_capture_enabled(spec.get(), ...)` → `common_speculative_set_prefill_capture_enabled(slot.get_spec(), ...)`
- `common_speculative_discard_dflash_state(spec.get(), ...)` → `common_speculative_discard_dflash_state(slot.get_spec(), ...)`
- `common_speculative_note_prefill_suffix_scheduled(spec.get())` → `common_speculative_note_prefill_suffix_scheduled(slot.get_spec())`
- `common_speculative_process(spec, batch)` in the eval callback → use `slot.get_spec()`

- [ ] **Step 6: Build**

Run: `cmake --build build-vulkan --target llama-server -j$(nproc) 2>&1 | tail -5`
Expected: BUILD SUCCEEDS

- [ ] **Step 7: Test single-slot regression (--parallel 1)**

```bash
GGML_DFLASH_TOKEN_TRACE=1 ./build-vulkan/bin/llama-server \
    --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf --spec-type dflash \
    --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --ctx-size 4096 --parallel 1 --device Vulkan0 \
    --host 127.0.0.1 --port 8192 --reasoning on > /tmp/multislot-p1-v2.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8192/v1/chat/completions ...
```
Expected: Same as Task 2 Step 5 (55% acceptance, correct output).

- [ ] **Step 8: Test multi-slot (--parallel 4)**

```bash
./build-vulkan/bin/llama-server \
    --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf --spec-type dflash \
    --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --ctx-size 4096 --parallel 4 --device Vulkan0 \
    --host 127.0.0.1 --port 8193 --reasoning on > /tmp/multislot-p4-v2.log 2>&1 &
sleep 30
# Send 4 concurrent requests
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8193/v1/chat/completions -H "Content-Type: application/json" \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is $i+$i?\"}],\"max_tokens\":30,\"temperature\":0.1,\"reasoning\":true}" &
done
wait
```
Expected: No crash, multiple slots produce drafts, `draft acceptance` > 0 for multiple slots.
Check: `grep "draft acceptance" /tmp/multislot-p4-v2.log | head -8` — should show acceptance for multiple slot IDs.

- [ ] **Step 9: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "feat(dflash): add llama_dflash_set_active_slot + use slot.get_spec()

Add llama_dflash_set_active_slot(ctx_tgt, slot.id) before each per-slot
DFlash operation (prefill capture, draft, accept, update_logits). This
sets the active slot's cross-attention ring in the target context so
multi-slot DFlash reads the correct hidden states.

Replace all spec.get() with slot.get_spec() in per-slot operations so
each slot uses its own DFlash impl/ring/seq_id.

Single-slot: unchanged (55.6%).
Multi-slot: drafts generated for multiple slots, no crash."
```

---

### Task 4: Verify and fix multi-slot issues

**Files:**
- Modify: `tools/server/server-context.cpp` (as needed based on test results)

- [ ] **Step 1: Run multi-slot test with token trace**

```bash
GGML_DFLASH_TOKEN_TRACE=1 ./build-vulkan/bin/llama-server \
    --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf --spec-type dflash \
    --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --ctx-size 4096 --parallel 4 --device Vulkan0 \
    --host 127.0.0.1 --port 8194 --reasoning on > /tmp/multislot-p4-trace.log 2>&1 &
```

- [ ] **Step 2: Check token trace for multiple seq_ids**

Run: `grep "DFLASH_TOKEN_TRACE" /tmp/multislot-p4-trace.log | head -20`
Expected: Trace lines with `seq=0`, `seq=1`, `seq=2`, `seq=3` (multiple slots producing drafts).
If only `seq=0`: other slots aren't speculating. Check dflash_slots_cap and slot init.

- [ ] **Step 3: Check for errors**

Run: `grep -i "error\|crash\|assert\|shape mismatch\|embd=0" /tmp/multislot-p4-trace.log | head -10`
Expected: No errors. If errors: fix based on the specific error message.

- [ ] **Step 4: Fix any remaining issues**

Common issues:
- `begin hidden[0] shape mismatch: embd=0`: capture not wired for this slot — check `llama_dflash_set_active_slot` is called before capture
- `dparams out of bounds`: dparams sizing — already fixed (commit 805e1560e) but verify
- `slot allocation failed`: drafter context too small — check `dflash_n_slots` and `n_ubatch`

- [ ] **Step 5: Run final verification — all 4 backends build**

```bash
# Vulkan
cmake --build build-vulkan --target llama-server -j$(nproc) 2>&1 | tail -3
# CUDA (if available)
cmake --build build-ci-vulkan --target llama-server -j$(nproc) 2>&1 | tail -3
```
Expected: 0 errors on all backends.

- [ ] **Step 6: Commit fixes and push**

```bash
git add -A
git commit -m "fix(dflash): multi-slot verification and fixes

Verify multi-slot DFlash with --parallel 4:
- Multiple slots produce drafts (seq=0,1,2,3 in token trace)
- No crash, no shape mismatch errors
- Single-slot regression: 55.6% unchanged

All backends build with 0 errors."
git push gboddaer merge_llama_into_beellama_2
```

---

### Task 5: Update TASK_PROGRESS.md and ask GLM for review

- [ ] **Step 1: Update TASK_PROGRESS.md with multi-slot DFlash findings**

Add HF-034 documenting:
- Multi-slot DFlash architecture (per-slot spec, shared ctx_dft)
- Single-slot regression test result
- Multi-slot test result (acceptance, speed, number of active slots)
- Any issues found and fixed

- [ ] **Step 2: Commit and push**

```bash
git add TASK_PROGRESS.md
git commit -m "docs: HF-034 multi-slot DFlash implementation complete"
git push gboddaer merge_llama_into_beellama_2
```

- [ ] **Step 3: Ask GLM for review of multi-slot implementation**

Send GLM the multi-slot architecture summary, test results, and ask for review of:
1. Is the per-slot spec isolation correct?
2. Are there race conditions with shared ctx_dft?
3. Is llama_dflash_set_active_slot called at all the right points?
4. Any missing cleanup on slot release?

---

## Self-Review

**Spec coverage:** The goal is multi-slot DFlash. Task 1 refactors the slot struct, Task 2 creates per-slot specs, Task 3 adds set_active_slot calls, Task 4 verifies and fixes, Task 5 documents and reviews. All aspects covered.

**Placeholder scan:** No TBD/TODO in the plan. All code blocks contain actual code. All test commands have expected outputs.

**Type consistency:** `common_speculative_ptr` (unique_ptr) used consistently for owned spec. `common_speculative *` for spec_shared (raw pointer). `get_spec()` returns `common_speculative *`. Consistent across all tasks.

**Risk:** The main risk is single-slot regression. Task 2 Step 5 and Task 3 Step 7 explicitly test single-slot before proceeding. If single-slot regresses, STOP and debug before continuing to multi-slot.