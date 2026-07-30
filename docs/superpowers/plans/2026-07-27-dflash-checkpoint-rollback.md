# DFlash Checkpoint Rollback + Batch Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 3.4× DFlash performance gap between the merge (6.18 t/s) and the fork (21+ t/s) by switching from reeval-based rollback to checkpoint-based rollback with GPU-to-GPU copies, and removing batch padding overhead.

**Architecture:** The merge's DFlash rollback uses `llama_dflash_rollback` + `llama_decode` reeval (216ms, 78-88% of verify cycles) — 31% of eval time. The fork uses checkpoint-based rollback (zero reeval) with `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for GPU-to-GPU copies. The merge already has `GGML_DFLASH_FORCE_CKPT_ROLLBACK` which eliminates reeval (tested: 0 calls, correct output), but the checkpoint restore takes 19080ms (73% of eval) because it's missing the `ON_DEVICE` flag — copying the entire KV cache through CPU memory. Additionally, the merge pads every verify batch to `1 + n_draft_max = 9` tokens, while the fork uses exact batch sizes (avg 5.9) — the padding adds ~30% overhead to each target decode. This plan: (1) enables checkpoint rollback for DFlash by default, (2) adds the `ON_DEVICE` flag for GPU-to-GPU copies, (3) removes verify batch padding, (4) optimizes the reeval fallback path as a safety net.

**Tech Stack:** C++, Vulkan, llama.cpp speculative decoding (DFlash), ggml state serialization

## What This Plan Is (and Is NOT)

**This plan ports the fork's checkpoint-based rollback approach to the merge.** It does NOT port the fork's entire DFlash architecture (`common_speculative_process`, reduced verify mode, etc.) — those are larger efforts documented as future work.

### What it IS
- **Enabling checkpoint rollback for DFlash by default** — the merge already has the `GGML_DFLASH_FORCE_CKPT_ROLLBACK` env var and checkpoint infrastructure; this makes it the default
- **Adding `ON_DEVICE` flag** — the fork uses `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for GPU-to-GPU checkpoint copies; the merge uses only `PARTIAL_ONLY` (CPU round-trip)
- **Removing batch padding** — the merge pads verify batches to max size; the fork uses exact sizes
- **A regression fix** — the merge's reeval path was a workaround; checkpoint rollback is the fork's intended approach

### What it is NOT
- NOT porting `common_speculative_process` — the fork's unified batch processing function is a separate, larger effort
- NOT porting reduced verify mode — the fork's argmax-only verification for greedy sampling is future work
- NOT fixing tape replay — the GPU tape is allocated on Vulkan, causing empty CPU tape; tape_replay would fail 100% of the time; the fork doesn't use tape_replay either (0 calls); checkpoint rollback is the right approach
- NOT porting the fork's per-cycle profiling — the fork's DFlash flow is architecturally different

### Why checkpoint rollback is safe
Tested with `GGML_DFLASH_FORCE_CKPT_ROLLBACK=1`: output is coherent, acceptance rate increases (0.33→0.67), zero reeval calls. The checkpoint save/restore uses `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext` which handle the `ON_DEVICE` flag internally for GPU-to-GPU copies. The fork uses the same mechanism at 41 t/s with zero reeval.

## Global Constraints

- **Worktree:** `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- **Build:** `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc)`
- **Target model:** `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` (16GB, Qwen3.6-27B, hybrid DeltaNet+attention)
- **Draft model:** `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf` (986MB, DFlash drafter)
- **Benchmark command:** `build/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf --spec-type dflash --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1`
- **Do NOT commit unless the user explicitly asks.**
- **Fork (for comparison):** `/crypt/beellama.cpp` with build at `build-vulkan/bin/llama-server`
- **Current merge DFlash: 6.18 t/s.** Fork DFlash: 21+ t/s. Baseline (no DFlash): 12.8 t/s.
- **Escape hatch:** `GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1` env var to fall back to reeval if checkpoint rollback has issues.

## Measured Performance Data (instrumentation, 100-token benchmark)

### Merge (current, with reeval)
| Component | Calls | Total ms | Avg ms | % eval |
|-----------|-------|----------|--------|--------|
| target_decode | 27 | 5567 | 206 | 34% |
| rollback_reeval | 21 | 4669 | 222 | 31% |
| batch_draft | 25 | 1269 | 51 | 8% |
| other/overhead | — | 2675 | — | 16% |

### Merge (with GGML_DFLASH_FORCE_CKPT_ROLLBACK=1, no ON_DEVICE)
| Component | Calls | Total ms | Avg ms | % eval |
|-----------|-------|----------|--------|--------|
| target_decode | 29 | 5997 | 207 | 23% |
| checkpoint_restore | 16 | ~19080 | ~1193 | 73% |
| batch_draft | 16 | 964 | 60 | 4% |
| reeval | 0 | 0 | — | 0% |

### Fork (reference)
| Component | Calls | Total ms | Avg ms | % eval |
|-----------|-------|----------|--------|--------|
| target_decode | 29 | 3779 | 130 | ~92% |
| checkpoint_restore | — | included in decode | ~0 | ~0% |
| batch_draft | — | — | — | — |
| reeval | 0 | 0 | — | 0% |

**Key insight:** The fork's checkpoint restore is effectively zero-cost (ON_DEVICE GPU-to-GPU copy). The merge's checkpoint restore is 1193ms/call because it copies through CPU memory. Adding `ON_DEVICE` should make the merge's checkpoint restore similarly fast.

---

### Task 1: Enable checkpoint rollback for DFlash by default

**Why:** The merge has `GGML_DFLASH_FORCE_CKPT_ROLLBACK` env var that forces checkpoint-based rollback (no reeval). This eliminates 78-88% of 216ms reeval calls. We make it the default for DFlash, with `GGML_DFLASH_DISABLE_CKPT_ROLLBACK` as an escape hatch.

**Files:**
- Modify: `tools/server/server-context.cpp` (3 locations: lines 3283, 3410, 4468)

**Interfaces:**
- Consumes: `params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)`, `std::getenv`
- Produces: `force_dflash_ckpt` (bool, now always true for DFlash unless disabled)

- [ ] **Step 1: Read the current code at all 3 locations**

Run: `grep -n "force_dflash_ckpt" tools/server/server-context.cpp`

You should see 3 definitions of `force_dflash_ckpt`, all identical:
```cpp
const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && std::getenv("GGML_DFLASH_FORCE_CKPT_ROLLBACK");
```

- [ ] **Step 2: Replace all 3 `force_dflash_ckpt` definitions**

There are 3 identical lines. Use the edit tool to replace each one. Each occurrence is on a single line. The old text is the same for all 3, so you need to include surrounding context to make each unique.

**Location 1 (line ~3283, in prepare-drafting path):**

oldText:
```
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && std::getenv("GGML_DFLASH_FORCE_CKPT_ROLLBACK");
                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || force_dflash_ckpt;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
```

newText:
```
                const bool dflash_ckpt_disabled = std::getenv("GGML_DFLASH_DISABLE_CKPT_ROLLBACK");
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && !dflash_ckpt_disabled;
                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || force_dflash_ckpt;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
```

**Location 2 (line ~3410, in save-checkpoint-before-drafting path):**

oldText:
```
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && std::getenv("GGML_DFLASH_FORCE_CKPT_ROLLBACK");
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt)) ||
                   force_dflash_ckpt;
```

newText:
```
                const bool dflash_ckpt_disabled = std::getenv("GGML_DFLASH_DISABLE_CKPT_ROLLBACK");
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && !dflash_ckpt_disabled;
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt)) ||
                   force_dflash_ckpt;
```

**Location 3 (line ~4468, in restore-checkpoint-after-rejection path):**

oldText:
```
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && std::getenv("GGML_DFLASH_FORCE_CKPT_ROLLBACK");
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt)) ||
                    force_dflash_ckpt;
```

newText:
```
                const bool dflash_ckpt_disabled = std::getenv("GGML_DFLASH_DISABLE_CKPT_ROLLBACK");
                const bool force_dflash_ckpt = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) && !dflash_ckpt_disabled;
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt)) ||
                    force_dflash_ckpt;
```

- [ ] **Step 3: Verify the edits applied correctly**

Run: `grep -n "force_dflash_ckpt\|dflash_ckpt_disabled\|GGML_DFLASH_FORCE_CKPT\|GGML_DFLASH_DISABLE_CKPT" tools/server/server-context.cpp`

**Pass criterion:** 3 occurrences of `dflash_ckpt_disabled`, 3 of `force_dflash_ckpt` with `!dflash_ckpt_disabled`, 0 occurrences of `GGML_DFLASH_FORCE_CKPT_ROLLBACK`.

- [ ] **Step 4: Build**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors.

- [ ] **Step 5: Quick correctness test (checkpoint rollback without ON_DEVICE — will be slow but correct)**

```bash
timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/test_task1.log &
sleep 25
curl -s -o /tmp/test_task1.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
python3 -c "import json; d=json.load(open('/tmp/test_task1.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message'].get('reasoning_content',''); print('OK len='+str(len(c)) if len(c)>50 else 'FAIL'); print(c[:200])"
```

**Pass criterion:** `OK` with coherent text about fibonacci. This confirms checkpoint rollback works by default (no env var needed). It will be SLOW (~3.8 t/s) because ON_DEVICE flag is not added yet — that's Task 2.

**Fail criterion:** HTTP error, empty response, or garbage → checkpoint rollback has a correctness issue. Test with `GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1` to confirm it falls back to reeval correctly.

---

### Task 2: Add ON_DEVICE flag to checkpoint save/restore

**Why:** Without `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, the checkpoint save/restore copies the entire KV cache through CPU memory (1193ms/call). With ON_DEVICE, it's a GPU-to-GPU copy (~0ms, like the fork). This is the single highest-impact change.

**Files:**
- Modify: `tools/server/server-context.cpp` (4 locations: lines 3422, 3434, 4496, 4502)

**Interfaces:**
- Consumes: `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (defined in `include/llama.h:914`, value `2`)
- Produces: GPU-to-GPU checkpoint copies instead of CPU round-trips

- [ ] **Step 1: Read the current code at all 4 locations**

Run: `grep -n "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY" tools/server/server-context.cpp`

Confirm lines 3422, 3434, 4496, 4502 all use `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` without `ON_DEVICE`.

- [ ] **Step 2: Add ON_DEVICE to checkpoint save (target, line ~3422)**

oldText:
```
                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

newText:
```
                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
```

- [ ] **Step 3: Add ON_DEVICE to checkpoint save (draft, line ~3434)**

oldText:
```
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

newText:
```
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
```

- [ ] **Step 4: Add ON_DEVICE to checkpoint restore (target, line ~4496)**

oldText:
```
                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

newText:
```
                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
```

- [ ] **Step 5: Add ON_DEVICE to checkpoint restore (draft, line ~4502)**

oldText:
```
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

newText:
```
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
```

- [ ] **Step 6: Verify the edits applied correctly**

Run: `grep -n "ON_DEVICE" tools/server/server-context.cpp`

**Pass criterion:** 4 occurrences of `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, all combined with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` via `|`.

- [ ] **Step 7: Build**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors.

- [ ] **Step 8: Benchmark — checkpoint rollback WITH ON_DEVICE (should be fast now)**

```bash
timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_task2.log &
sleep 25
curl -s -o /tmp/bench_task2.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance" /tmp/bench_task2.log
python3 -c "import json; d=json.load(open('/tmp/bench_task2.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message'].get('reasoning_content',''); print('OK len='+str(len(c)) if len(c)>50 else 'FAIL'); print(c[:200])"
```

**Pass criterion:**
- `eval time` shows `≥ 8.0 tokens per second` (was 3.84 without ON_DEVICE, should jump significantly)
- Output is coherent (OK, not FAIL)
- `python3 -c "import re; l=open('/tmp/bench_task2.log').read(); m=re.findall(r'(\S+) tokens per second', l); tps=float(m[-1]) if m else 0; print(f'PASS tps={tps:.1f}' if tps>=8.0 else f'FAIL tps={tps:.1f} (expected >=8.0)')"`

**Fail criterion:** tps < 8.0 → ON_DEVICE flag is not being handled correctly. Check if `llama_state_seq_get_data_ext` supports ON_DEVICE on Vulkan. If not, the fallback is to keep `PARTIAL_ONLY` (slow but correct) and investigate the Vulkan backend's ON_DEVICE support.

---

### Task 3: Remove verify batch padding

**Why:** The merge pads every verify batch to `1 + n_draft_max = 9` tokens for graph reuse. But the fork uses exact batch sizes (avg 5.9) and is 3.4× faster. The padding overhead (decoding dummy tokens + KV cleanup) is worse than graph rebuilds. The fork has 65% graph reuse vs the merge's 182%, but is much faster — graph rebuilds are not the bottleneck.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines 543-559, in `generate_draft`)

**Interfaces:**
- Consumes: `task->params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)`, `get_n_draft_max()`, `spec_pad_i_batch`
- Produces: Variable-size verify batches (exact `1 + spec_draft.size()`)

- [ ] **Step 1: Read the current padding code**

Run: `sed -n '538,565p' tools/server/server-context.cpp`

You should see the DFlash verify padding block:
```cpp
            // DFlash verify padding: pad the batch to a fixed size (n_draft_max + 1)
            // so every verify ubatch has the same n_tokens. This allows the graph
            // to be reused across decode cycles (can_reuse checks n_tokens equality).
            // Without padding, varying draft acceptance rates change n_tokens every
            // cycle, causing 0 graph reuses and full graph rebuilds each decode.
            // Ported from fork adb92b36a:4964-4990 (GGML_DFLASH_VERIFY_PAD).
            if (task && task->params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                const int n_draft_max = get_n_draft_max();
                const int target_batch_size = 1 + n_draft_max; // sampled + max drafts
                const int current_batch_size = 1 + (int) spec_draft.size();
                const int pad_count = std::max(0, target_batch_size - current_batch_size);
                for (int i = 0; i < pad_count; ++i) {
                    spec_pad_i_batch.push_back(batch.size());
                    add_ok &= batch.add(this->id, sampled, pos0 + i, true);
                }
                if (pad_count > 0) {
                    SLT_DBG(*this, "dflash verify pad: pad_count=%d target=%d current=%d\n", pad_count, target_batch_size, current_batch_size);
                }
            }
```

- [ ] **Step 2: Remove the padding block**

oldText (the entire padding block from comment to closing brace):
```
            // DFlash verify padding: pad the batch to a fixed size (n_draft_max + 1)
            // so every verify ubatch has the same n_tokens. This allows the graph
            // to be reused across decode cycles (can_reuse checks n_tokens equality).
            // Without padding, varying draft acceptance rates change n_tokens every
            // cycle, causing 0 graph reuses and full graph rebuilds each decode.
            // Ported from fork adb92b36a:4964-4990 (GGML_DFLASH_VERIFY_PAD).
            if (task && task->params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
                const int n_draft_max = get_n_draft_max();
                const int target_batch_size = 1 + n_draft_max; // sampled + max drafts
                const int current_batch_size = 1 + (int) spec_draft.size();
                const int pad_count = std::max(0, target_batch_size - current_batch_size);
                for (int i = 0; i < pad_count; ++i) {
                    spec_pad_i_batch.push_back(batch.size());
                    add_ok &= batch.add(this->id, sampled, pos0 + i, true);
                }
                if (pad_count > 0) {
                    SLT_DBG(*this, "dflash verify pad: pad_count=%d target=%d current=%d\n", pad_count, target_batch_size, current_batch_size);
                }
            }
```

newText:
```
            // No DFlash verify padding — use exact batch size (1 + spec_draft.size()).
            // The fork uses variable batch sizes and achieves 3.4× higher throughput.
            // Graph rebuilds for varying batch sizes are cheaper than decoding
            // padding tokens + KV cleanup. The fork has 65% graph reuse vs our
            // 182%, but is much faster — graph reuse is not the bottleneck.
```

- [ ] **Step 3: Check if spec_pad_i_batch is used elsewhere**

Run: `grep -n "spec_pad_i_batch" tools/server/server-context.cpp`

If `spec_pad_i_batch` is used in other places (e.g., to skip padding tokens during acceptance), those uses should still work — the vector will just be empty (no padding tokens to skip).

- [ ] **Step 4: Build**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors.

- [ ] **Step 5: Quick correctness test**

```bash
timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/test_task3.log &
sleep 25
curl -s -o /tmp/test_task3.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
python3 -c "import json; d=json.load(open('/tmp/test_task3.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message'].get('reasoning_content',''); print('OK len='+str(len(c)) if len(c)>50 else 'FAIL'); print(c[:200])"
grep "tokens per second" /tmp/test_task3.log | tail -1
```

**Pass criterion:** `OK` with coherent text. TPS should be similar or slightly better than Task 2 (padding removal mainly helps small batches).

---

### Task 4: Optimize reeval fallback path (logits=false, no padding)

**Why:** When checkpoint rollback is disabled (`GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1`) or for non-DFlash speculative types, the reeval path is the fallback. The merge's reeval uses `logits=true` (computes output unnecessarily) and pads to max batch size. The fork's reeval uses `logits=false` (state advance only) and exact batch size. This task optimizes the fallback to match the fork.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines 4554-4593, in the DFlash rollback reeval block)

**Interfaces:**
- Consumes: `n_reeval`, `slot.dflash_n_pos_before_draft`, `slot.sampled`, `accepted`, `ctx_tgt`
- Produces: Cheaper reeval decode (no logits, no padding, no KV cleanup)

- [ ] **Step 1: Read the current reeval code**

Run: `sed -n '4554,4595p' tools/server/server-context.cpp`

You should see the reeval block with padding and `logits=true` (the 5th arg to `common_batch_add`).

- [ ] **Step 2: Replace the reeval block with the optimized version**

oldText (the entire reeval block from `if (n_reeval > 0)` to the closing `}` before the QA_TRACE):
```
                        if (n_reeval > 0) {
                            // Pad the reeval batch to the same size as the verify batch
                            // (n_draft_max + 1) so the graph can be reused across decode
                            // cycles. Without padding, reeval batches have varying sizes
                            // (1..n_draft_max) which prevents graph reuse.
                            const int n_draft_max = slot.get_n_draft_max();
                            const int target_batch_size = 1 + n_draft_max;
                            const int pad_count = std::max(0, target_batch_size - n_reeval);
                            llama_batch batch_reeval = llama_batch_init(target_batch_size, 0, 1);
                            for (int j = 0; j < n_reeval; ++j) {
                                const llama_pos pos = slot.dflash_n_pos_before_draft + j;
                                // Use prompt tokens (already pushed during batch build)
                                // rather than reconstructing from accepted[] — this
                                // matches the fork's approach and is more robust.
                                const llama_token tok = (j == 0) ? slot.sampled : accepted[j - 1];
                                common_batch_add(batch_reeval, tok, pos, { slot.id }, true);
                            }
                            // Add padding tokens at positions beyond the actual tokens.
                            // These are dummy tokens with output=true that only serve to
                            // make the batch size match the verify batch for graph reuse.
                            // Their KV entries are removed after decode.
                            const llama_pos pad_pos_start = slot.dflash_n_pos_before_draft + n_reeval;
                            for (int i = 0; i < pad_count; ++i) {
                                common_batch_add(batch_reeval, slot.sampled, pad_pos_start + i, { slot.id }, true);
                            }
                            const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
                            llama_batch_free(batch_reeval);
                            // Remove KV entries created by padding tokens
                            if (pad_count > 0) {
                                auto * mem = llama_get_memory(ctx_tgt);
                                llama_memory_seq_rm(mem, slot.id, pad_pos_start, -1);
                            }
                            if (std::getenv("GGML_DFLASH_QA_TRACE")) {
                                fprintf(stderr, "[DFLASH_QA] rollback_reeval slot=%d n_reeval=%d ret=%d\n",
                                    slot.id, n_reeval, ret_reeval);
                            }
                        }
```

newText:
```
                        if (n_reeval > 0) {
                            // Reeval with exact batch size (no padding) and logits=false
                            // (only state advance, no output). Matches the fork's approach.
                            // logits=false avoids unnecessary output computation; no padding
                            // avoids decoding dummy tokens + KV cleanup.
                            llama_batch batch_reeval = llama_batch_init(n_reeval, 0, 1);
                            for (int j = 0; j < n_reeval; ++j) {
                                const llama_pos pos = slot.dflash_n_pos_before_draft + j;
                                const llama_token tok = (j == 0) ? slot.sampled : accepted[j - 1];
                                common_batch_add(batch_reeval, tok, pos, { slot.id }, false);
                            }
                            const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
                            llama_batch_free(batch_reeval);
                            if (std::getenv("GGML_DFLASH_QA_TRACE")) {
                                fprintf(stderr, "[DFLASH_QA] rollback_reeval slot=%d n_reeval=%d ret=%d\n",
                                    slot.id, n_reeval, ret_reeval);
                            }
                        }
```

- [ ] **Step 3: Verify the edit applied correctly**

Run: `grep -n "pad_count\|target_batch_size\|pad_pos_start" tools/server/server-context.cpp | grep -v "spec_pad\|543\|559"`

**Pass criterion:** No occurrences of `pad_count`, `target_batch_size`, or `pad_pos_start` in the reeval block (they should only remain in the verify padding block at line 543, which was removed in Task 3 — so there should be 0 occurrences total if Task 3 was done).

Run: `grep -n "logits.*false\|, false);" tools/server/server-context.cpp | grep "batch_reeval\|reeval"`

**Pass criterion:** The reeval `common_batch_add` calls use `false` (not `true`) for the logits parameter.

- [ ] **Step 4: Build**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors.

- [ ] **Step 5: Test reeval fallback with checkpoint disabled**

```bash
GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1 timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/test_task4.log &
sleep 25
curl -s -o /tmp/test_task4.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
python3 -c "import json; d=json.load(open('/tmp/test_task4.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message'].get('reasoning_content',''); print('OK len='+str(len(c)) if len(c)>50 else 'FAIL'); print(c[:200])"
grep "tokens per second" /tmp/test_task4.log | tail -1
```

**Pass criterion:** `OK` with coherent text. This confirms the reeval fallback (with `logits=false`, no padding) produces correct output when checkpoint is disabled. TPS should be ≥ 6.0 (similar to or slightly better than the original 6.18 t/s, since logits=false and no padding reduce reeval cost).

---

### Task 5: Final benchmark and verification

**Why:** Confirm that all changes together (checkpoint rollback + ON_DEVICE + no padding + optimized reeval fallback) produce correct output and improved performance. Compare against the fork and the previous merge baseline.

**Files:**
- No code changes — benchmark only

- [ ] **Step 1: Run 300-token DFlash benchmark on the merge (with all fixes)**

```bash
timeout 180 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_merge_final.log &
sleep 25
curl -s -o /tmp/bench_merge_final.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Implement a complete binary search tree in C++ with insertion, deletion, in-order traversal, and a print function. Include proper memory management and a main function that demonstrates all operations."}],"max_tokens":300,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance\|graphs reused\|statistics.*dflash" /tmp/bench_merge_final.log
python3 -c "import json; d=json.load(open('/tmp/bench_merge_final.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message'].get('reasoning_content',''); print('OK len='+str(len(c)) if len(c)>100 else 'FAIL'); print(c[:300])"
```

**Pass criterion:**
- `eval time` shows `≥ 10.0 tokens per second` (was 6.18, target ≥10)
- `draft acceptance` shows `≥ 0.40`
- Output is coherent (OK, not FAIL)
- `python3 -c "import re; l=open('/tmp/bench_merge_final.log').read(); m=re.findall(r'(\S+) tokens per second', l); tps=float(m[-1]) if m else 0; print(f'PASS tps={tps:.1f}' if tps>=10.0 else f'FAIL tps={tps:.1f} (expected >=10.0)')"`

- [ ] **Step 2: Run 300-token DFlash benchmark on the fork for comparison**

```bash
timeout 180 /crypt/beellama.cpp/build-vulkan/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_fork_final.log &
sleep 25
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Implement a complete binary search tree in C++ with insertion, deletion, in-order traversal, and a print function. Include proper memory management and a main function that demonstrates all operations."}],"max_tokens":300,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance" /tmp/bench_fork_final.log
```

- [ ] **Step 3: Run baseline (no DFlash) on the merge to confirm no regression**

```bash
timeout 60 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 999 -c 8192 --port 18080 \
  2>/tmp/bench_merge_baseline.log &
sleep 20
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "tokens per second" /tmp/bench_merge_baseline.log | tail -1
```

**Pass criterion:** `eval time` shows `≥ 11.0 tokens per second` (no regression in non-DFlash path).

- [ ] **Step 4: Produce a comparison table**

```bash
echo "=== COMPARISON TABLE ==="
echo "| Metric | Merge (before) | Merge (after) | Fork | Merge baseline |"
echo "|--------|----------------|---------------|------|-----------------|"
echo "| Decode TPS | 6.18 | $(grep 'tokens per second' /tmp/bench_merge_final.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)') | $(grep 'tokens per second' /tmp/bench_fork_final.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)') | $(grep 'tokens per second' /tmp/bench_merge_baseline.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)') |"
echo ""
echo "Improvement: 6.18 -> $(grep 'tokens per second' /tmp/bench_merge_final.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)') t/s"
```

**Pass criterion:** The table prints without errors. Merge (after) TPS ≥ 10.0. The remaining gap to the fork is documented for future work.

- [ ] **Step 5: Test the escape hatch (GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1)**

```bash
GGML_DFLASH_DISABLE_CKPT_ROLLBACK=1 timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_disable_ckpt.log &
sleep 25
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "tokens per second" /tmp/bench_disable_ckpt.log | tail -1
```

**Pass criterion:** TPS ≥ 6.0 (reeval fallback works, with optimized logits=false + no padding). Output is coherent. This confirms the escape hatch works.

---

## Future Work (NOT in this plan — documented for reference)

This plan targets checkpoint rollback + ON_DEVICE + padding removal. The remaining gap to the fork (if any) has these causes:

### 1. Port `common_speculative_process` (architectural)
The fork uses `common_speculative_process(spec, batch_view)` for unified batch processing — handling verification, acceptance, and rollback in one function. The merge uses separate `common_sampler_sample_and_accept_n` + `dflash_rollback` + reeval. Porting the fork's unified flow would eliminate the architectural difference.

### 2. Port reduced verify mode
The fork uses `dflash_select_reduced_verify_plan` + `dflash_sample_reduced_verify` for greedy sampling — only requesting argmax (top-k) instead of full logits. This reduces GPU work per decode. The merge always requests full logits.

### 3. Vulkan timeline semaphore for cross-queue ordering
If checkpoint rollback + ON_DEVICE still has cross-queue visibility issues, a Vulkan timeline semaphore between compute and transfer queues would provide GPU-side ordering without CPU blocking.

### 4. Fix GPU tape on Vulkan
The GPU tape is allocated on Vulkan but doesn't capture data (GDN ops not supported). This causes the CPU tape to be empty, making tape_replay impossible. Disabling GPU tape allocation on Vulkan (or fixing the GDN ops) would enable tape_replay as an alternative to checkpoint rollback.