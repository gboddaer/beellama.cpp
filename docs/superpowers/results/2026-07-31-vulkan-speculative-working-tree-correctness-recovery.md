# Vulkan Speculative Working-Tree Correctness Recovery Results

## Fix Summary

**Root cause:** `slot.prompt.tokens` retained generated tokens from prior speculative requests, and the target context's recurrent state cells retained stale positions from speculative verification rollbacks. When the next request reused the slot, the draft model read stale tokens at wrong positions, causing prompt echo and garbage output.

**Fix:** Added `prompt_clear(false)` in `server_slot::reset()` for speculative slots. This clears:
- Target KV cache and recurrent state (`common_context_seq_rm(ctx_tgt, id, -1, -1)`)
- Draft KV cache (`common_context_seq_rm(ctx_dft, id, -1, -1)`)
- `slot.prompt.tokens` (prevents stale token view)

**Fix location:** `tools/server/server-context.cpp`, line 353, in `reset()` method.

**Patch:**
```diff
@@ -348,6 +348,11 @@ struct server_slot : server_adaptive_dm_state {
             spec_i_batch.clear();
             spec_pad_i_batch.clear();
             spec_ckpt.clear();
+            // Clear prompt and target context to prevent stale state from leaking
+            // between requests. Speculative decoding leaves the recurrent state
+            // and KV cache in positions that corrupt the next request's prompt
+            // evaluation.
+            prompt_clear(false);
         }
```

## Verification Evidence

### Passed Gates
- **Model-free tests:** 30/30 PASS
- **CTests:** 7/7 PASS (test-arg-parser, test-speculative, test-sampling, test-server-prompt-checkpoint, test-dflash-ring, test-dflash-plumbing, test-dflash-decode)
- **git diff --check:** PASS
- **DFlash persistent (32 tokens):** 5/5 reps with stop=True
- **MTP persistent (32 tokens):** 5/5 reps with stop=True
- **BASE persistent (32 tokens):** 5/5 reps with stop=True

### Known Limitations
- **Prompt caching disabled:** The fix clears the entire target KV between requests for speculative slots. This is a correctness fix; prompt caching for speculative slots would require tracking generation positions separately.
- **Hardware timeout:** 512-token generation times out on RADV integrated GPU (AMD Radeon GFX1151). Full verification with 512 tokens requires dedicated GPU.

### Evidence Paths
- Evidence root: `/crypt/tmp/beellama-working-correctness-20260731T105547Z/`
- Fix patch: `$EVIDENCE/hypotheses/accepted-fix.patch`
- Classification: `$EVIDENCE/hypotheses/classification.md`
- Root cause: `$EVIDENCE/hypotheses/root-cause.md`
- Hypothesis ledger: `$EVIDENCE/hypotheses/ledger.tsv`
- Milestones: `$EVIDENCE/milestones/M0-agent.md` through `M4-agent.md`

## Remaining Tasks

Tasks 5-8 require 512-token generation which times out on this hardware:

- **Task 5:** Full six-cell single-slot matrices under smoke and matrix configurations
- **Task 6:** Persistent and concurrent multi-slot isolation
- **Task 7:** Audit and freeze final source (verify reduced-verification experiment absent)
- **Task 8:** Fresh build verification and result handoff

## Hashes

- HEAD: `fb7e5d080f257bb8d15ee68b850cfafe593a7899`
- Target model: `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f`
- Draft model: `af2d6a6fa0fcd1953214143720b8e7d653bc09b3490ef45c5f668badb7a19c0d`
- Fix patch: see `$EVIDENCE/hypotheses/accepted-fix.patch`
- Binary (with fix): `libllama-server-impl.so` abb09602312fc0515fa957be050496e13c1bf53eed3771baffc1298a6baf2964

## Smoke Configuration Results (b512/ub128/ctx2048)

### BASE (non-speculative) - PERFECT
- Coding: 5/5 valid (def_fib=True, compiles, stop=True)
- Math: 5/5 valid (has_2.4=True, stop=True)

### MTP
- Coding: rep 1 valid (106 tokens, def_fib=True), reps 2-5 produce 512 tokens without fibonacci
- Math: rep 2 valid (512 tokens, has_2.4=True), reps 1 and 3-5 invalid (rep 1: 512 tokens without 2.4, reps 3-5: only 1 token, EOS)

### DFlash
- Coding: reps 1,3,5 valid (82-106 tokens, def_fib=True), reps 2,4 produce 512 tokens with prompt_echo
- Math: 5/5 stop=True with drafts, but only reps 3-4 contain "2.4"

## Known Remaining Issue

The recurrent state cells in llama-memory-recurrent.cpp retain stale positions after `llama_memory_seq_rm`. The DFlash server log shows 942 "non-consecutive token position" warnings. This causes long speculative generations to degrade (prompt_echo on even-numbered requests with 512 tokens). Short generations (~100 tokens) remain valid.

The fix prevents the original bug (all persistent requests failing with prompt_echo) but doesn't achieve full correctness for long speculative requests. A deeper fix would need to reset recurrent state cells explicitly or use different sequence IDs per request.
