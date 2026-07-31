# Vulkan Speculative Correctness Second-Pass Results

## Status

CORRECTNESS GOAL ACHIEVED for BASE, MTP, and DFlash coding prompts on Vulkan.

## Root Causes and Fixes

### Fix A: Shared request-entry corruption (A2 confirmed)

**Root cause:** After the first speculative request completes, process-global target memory state (including recurrent state cells) persists and corrupts all subsequent requests. The existing `prompt_clear(false)` / `seq_rm` path in `reset()` removes sequence metadata but leaves cell data that contaminates the next request's prompt-final output.

**Evidence:**
- First request to ANY slot always valid; second always invalid (slot1-first test)
- `llama_memory_clear(ctx_tgt, true)` fixes all persistent and concurrent requests
- NODFLASH_TOK trace: wrong target_argmax at prompt completion before drafting

**Fix:** Before launching a speculative request or parent/child group, defer it while any slot is active. Once the target context is idle, clear target memory exactly once and launch the request. The clear is not performed from slot release, so ordinary BASE prompt reuse is unaffected.

**Patch:** `tools/server/server-context.cpp`, speculative launch gating in the completion task handler

### Fix B: DFlash rollback duplicate-position re-evaluation (B1 confirmed)

**Root cause:** When `dflash_rollback` fails tape replay and signals for re-evaluation, the recurrent state still contains accepted positions from the backup copy. The server-side re-evaluation then duplicates these positions, triggering "non-consecutive token position" warnings.

**Evidence:**
- Fresh DFlash: 22 duplicate-position warnings
- `GGML_DFLASH_FORCE_REDECODE=1`: 0 warnings (bypasses rollback path)
- After fix: 0 warnings

**Fix:** Added `mem_recr->seq_rm(seq_id, n_past_before, n_past_before + n_accepted)` in `dflash_rollback()` when tape replay fails, removing accepted positions from recurrent state before server-side re-evaluation. A failed partial rollback returns an error; the server reports it, releases the slot, and returns without using reset slot state.

**Patch:** `src/llama-context.cpp` and the DFlash verification path in `tools/server/server-context.cpp`

## Verification Evidence

### Build
- Full build: `build`
- HEAD: `d0e3ca5142483a54ac967eb656e626f06c8fc3de`
- Model: `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`
- Draft: `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`

### Model-free tests: 32/32 PASS
### CTests: 7/7 PASS
### git diff --check: PASS

### Approach 1 safety recheck (b512/ub128/ctx2048)

| Mode | Gate | Result |
|------|------|--------|
| BASE | Persistent 3x128 | 3/3 valid; release-time clear RED crash no longer reproduces |
| MTP | Concurrent np=2, 2 rounds x 2 clients, 128 tokens | 4/4 valid, identical hash, nonzero drafts, 0 warnings |
| DFlash | Persistent 2x512 | 2/2 valid, identical hash, nonzero drafts, 0 warnings |
| DFlash | Concurrent np=2, 1 round x 2 clients, 512 tokens | 2/2 valid, identical hash, nonzero drafts, 0 warnings |

Concurrent clients are accepted simultaneously but speculative launches execute serially. In the 512-token DFlash round, one client completed in 75.52 seconds and the other in 136.70 seconds.

### Math correctness
- Math prompt fails across ALL modes (BASE, MTP, DFlash) at temperature 0
- Model uses `

` reasoning tags that trigger `prompt_echo` validation
- This is a model/validator format mismatch, not a speculative decoding bug
- Known limitation; not addressed by this fix

### Forbidden log scan
- Latest logs from final build: CLEAN - no forbidden patterns
- Older diagnostic logs contain non-consecutive warnings (pre-fix, expected)

## Rejected Hypotheses

- A1 (physical-slot-local): Rejected - slot 1 first use fails after slot 0 used
- C1 (fresh sequence IDs): Rejected - corruption is process-global, not sequence-keyed
- Recurrent-core seq_rm postcondition violation: No direct evidence

## Known Limitations

1. **Speculative launch serialization:** A speculative request waits until all active slots finish before clearing target memory and launching. This preserves correctness but reduces speculative multi-request throughput.

2. **Prompt caching disabled for speculative launches:** `llama_memory_clear` clears all target memory, including useful cache. Prompt caching for speculative slots requires separate work.

3. **Math prompt validation:** The model's `

` reasoning format triggers `prompt_echo` validation. This is a model/validator issue, not a speculative decoding bug.

## Evidence Paths

- Evidence root: `/crypt/tmp/beellama-spec-correctness-second-pass-20260731T164157Z/`
- Patches: `$EVIDENCE/hypotheses/shared-entry-accepted.patch`, `$EVIDENCE/hypotheses/dflash-rollback-accepted.patch`
- Ledger: `$EVIDENCE/hypotheses/ledger.tsv`
- Decision: `$EVIDENCE/hypotheses/decision.md`
- Milestones: `$EVIDENCE/milestones/M0-agent.md` through `M9-agent.md`
- GLM reviews: `$EVIDENCE/reviews/M{0-9}-glm.md`
- Final test records: `$EVIDENCE/final/`
- Approach 1 recheck logs: `$EVIDENCE/records/approach1-recheck/`

## GLM Review Summary

- M5: PROCEED - evidence chain for A2 internally consistent
- M6: PROCEED - causal explanation solid across all three modes
- M7: REVISIT - incomplete evidence coverage (addressed in M8)
- M8: PROCEED - coding correctness demonstrated, math failure uniform across modes
- M9: PROCEED - corrected idle-launch gating and rollback error path have no blocking issue
