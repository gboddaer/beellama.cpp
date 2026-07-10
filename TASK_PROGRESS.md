# BeeLlama Merge: ggml-org/llama.cpp into beellama.cpp

**Created:** 2026-07-01  
**Last updated:** 2026-07-04 11:00 UTC  
**Status:** ✅ MERGE COMPLETE — CI BUILDS — INFERENCE WORKS — DFLASH INITIALIZES (draft-gen next)  

## Current state

**Objective:** Merge ggml-org/llama.cpp:master into beellama.cpp:main with DFlash speculative decoding preserved.

**Branch:** `merge_llama_into_beellama_2` in worktree `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`  
**PR:** https://github.com/gboddaer/beellama.cpp/pull/2 (merge_llama_into_beellama_2 → main)  
**Merge commit:** `1ef3d97e4`  •  **Upstream merge point:** `f708a5b2c`  •  **Fork base:** `adb92b36a`  
**Latest commit:** `805e1560e` — fix(dflash): size dparams to n_parallel (enables --parallel 1 decode)

**Current status:**
- ✅ 0 compilation errors across all backends (Vulkan, ROCm, CUDA, Debug/CPU)
- ✅ CI Fix #1 (MSVC noinline) + CI Fix #2 (test-dflash-plumbing LNK2019) — build fully fixed
- ✅ Baseline inference works on iGPU (Vulkan/RADV): 5/5 prompts correct, 53 tok/s peak
- ✅ **DFlash speculative decoding INITIALIZES** (commit `9a67f5636`)
- ✅ **DFlash DECODE WORKS with `--parallel 1`** (commit `805e1560e`) — drafts generated+accepted (75/5),
  first end-to-end DFlash speculative decode on the merge
- ⚠️ **DFlash with `--parallel 4` (default): no crash but 0 drafts** (hidden-state capture not wired for
  4 slots) — needs fork's per-slot spec + capture-enable wiring (larger port)
- ⚠️ DFlash output quality low (garbled, ~6.7% acceptance) — needs fork's decode tuning
- ✅ GLM review of init fixes: all 3 concerns resolved (commit `b878e6435`)

---

## Hard facts

- HF-001: Merge base is `6ddc9430b`
- HF-002: Fork had 342 commits with 60+ conflicts
- HF-003: 19 files ported (7 MUST + 12 SHOULD)
- HF-004: `hparams.n_layer()` is a function returning `n_layer_all + n_layer_nextn`
- HF-005: Added COMMON_SPECULATIVE_TYPE_DFLASH, SUFFIX, COPY_SPEC, RECYCLE to enum
- HF-006: Added missing common_speculative_params members
- HF-007: Added DFlash API declarations to include/llama.h
- HF-008: Fixed llama_kv_cache_iswa constructors (added hparams parameter)
- HF-009: Fixed test-turbo-quant.c with pragma to disable warnings
- HF-010: **0 compilation errors** — build compiles successfully
- HF-011: **CI Build FIXED** — exit code changed from 1 (build) to 8 (test failures only)
- HF-012: **Inference works on iGPU** — Qwen3-Coder-Next-Q4_K_M generates correctly at 53 tok/s
- HF-013: **MoE model faster** — Qwen3.6-35B-A3B (44.5 tok/s) vs dense 27B (12.6 tok/s)
- HF-014: **Qwen3.6 uses reasoning_content field** — model outputs thinking separately
- HF-015: **DFlash runtime crash root cause = QWEN35 target DeltaNet writeback, NOT DFlash code**
  - Crash backtrace (build-ci-vulkan, --spec-type dflash, Qwen3.6-27B-Q4_K_M target):
    `#8 llm_build_delta_net_base::build_recurrent_attn` → `#9 llama_model_qwen35::graph::build_layer_attn_linear`
    → `#10 llama_model_qwen35::graph::graph` → `#13 llama_context::graph_reserve`
    → `#17 common_get_device_memory_data_impl` → `#19 common_fit_params`
  - Crashing op: `ggml_view_3d` in `src/models/delta-net-base.cpp` (UPSTREAM bulk writeback)
    `GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src))` (ggml.c:1807)
  - The merge took UPSTREAM's `delta-net-base.cpp` (52 insertions, 323 deletions) — bulk `ggml_view_3d`
    writeback. The FORK had 323 more lines: tree-based per-slot writeback (`ggml_view_4d`/`ggml_view_2d`
    + `llama_dflash_rs_writeback_slot_for_test`).
  - Only reproduces with `--spec-type dflash`. Without DFlash, same target loads+generates fine (Tier 1).
  - Proof: fork binary `build-vulkan/bin/llama-server` (adb92b36a) loads SAME model pair with --spec-type
    dflash successfully (`adding implementation dflash`, `contract ok`, exit 124 = running).
- HF-016: **Fork DFlash draft `is_swa_any = 1` (TRUE)** — fork binary --verbose log confirms.
  Fork `is_swa_any()` reads `swa_layers[]` DIRECTLY (llama-hparams.cpp:22). Upstream refactored to
  read `is_swa_impl[]` populated by `set_swa_pattern()`. My fix (copy swa_layers→is_swa_impl) matches fork.
- HF-017: **Fork server-context.cpp had ~652 lines of DFlash server integration; merge replaced with
  upstream's ~128 lines.** Lost: `dflash: setting -cd to 256` drafter ctx auto-detect (fork:2504),
  drafter block_size/slots setup (fork:2515), dflash_rx_diag/token_trace, server_model_is_dflash_drafter.
  Merge's server calls 3-arg common_speculative_init (my fix) but lacks drafter ctx/slots setup.
- HF-018: **Fork uses `hparams.n_layer` (member); merge uses `hparams.n_layer()` (function)**
  (consequence of upstream HF-004 refactor). dflash_draft.cpp swa_layers load uses `n_layer()` in merge.
- HF-019: **Two `common_speculative_init` overloads in merge**: 2-arg (line 4024, handles
  DRAFT_SIMPLE/EAGLE3/MTP/NGRAM only) and 3-arg (line 4489, handles DFLASH/SUFFIX/COPYSPEC/RECYCLE).
  Fork server called the DFlash-aware path; merge called the 2-arg one → "no implementations specified".
- HF-020: **DFLASH INITIALIZES + PROCESSES REQUESTS (4 fixes applied, 2026-07-04, commit 9a67f5636)** ✅
  - Fix 1 (server-context.cpp): call 3-arg `common_speculative_init(params, ctx_tgt, ctx_dft.get())`
    instead of 2-arg version that didn't handle COMMON_SPECULATIVE_TYPE_DFLASH.
  - Fix 2 (server-context.cpp): set `params_base.speculative.model_dft = model_dft.get()` before init
    (fork adb92b36a:2522). Without it DFlash impl constructor called `llama_model_n_embd(nullptr)` → SIGSEGV.
  - Fix 3 (dflash_draft.cpp load_arch_hparams): populate `is_swa_impl[]` from `swa_layers[]` (bridges
    fork's swa_layers-direct is_swa() to upstream's is_swa_impl reading). Fixes GGML_ASSERT(is_swa_any()).
  - Fix 4 (delta-net-base.cpp build_recurrent_attn): port fork's (a) state padding
    `s_3d=ggml_reshape_3d(s,D,1,n_seqs); s_3d_pad=ggml_pad(s_3d,0,K-1,0,0)` so ggml_gated_delta_net
    produces K snapshots (merge passed 4D s → K_actual=1 → view overflow), and (b) per-slot writeback
    loop (llama_dflash_rs_writeback_slot_for_test + ggml_view_2d/4d) replacing upstream's bulk ggml_view_3d
    (wrong per-snapshot stride nb[2]=kv_head*row_size overflowed ssm_states_all).
  - Proof: merge binary prints `adding implementation dflash`, `contract ok`, `GPU cross ring enabled`
    — matches fork binary. Server stays up (exit 124 = timeout = running).
  - Verified builds: build-ci-vulkan (Vulkan) + build-ci-debug (CPU) both 0 errors after the fixes.
- HF-021: **DFlash draft generation crashes with --parallel 4 (default) — dparams sizing**
  - 3-arg common_speculative_init hardcodes `n_seq=1` (speculative.cpp:4520) → dparams sized to 1.
    Server accesses `dparams[slot.id]` with slot.id 0-3 (n_parallel=4) → `GGML_ASSERT(seq_id < dparams.size())`
    at speculative.cpp:4165.
  - Fork used PER-SLOT specs (`slot.spec`, fork:2777, each n_seq=1, accessed with seq_id=0); merge uses a
    single spec for all slots → mismatch.
  - Preceded by `begin hidden[0] shape mismatch: embd=0` warning (only with multi-slot; see HF-024).
- HF-023: **GLM review concerns ALL RESOLVED** (see HF-022 — swa_layers fixed-size array moot;
    n_rs_seq no-regression verified 12.7 tok/s; 3-arg dispatch fixed commit b878e6435).
- HF-024: **DFLASH DECODE WORKS with --parallel 1 (single slot)!** ✅ (2026-07-04)
  - `llama-server --parallel 1 --spec-type dflash ...` → no crash, 11 tokens generated, finish=stop.
  - DFlash speculative decoding IS ACTIVE: `draft acceptance = 0.06667 (5 accepted / 75 generated)`,
    `dflash: #gen tokens = 75, #acc tokens = 5`. Cross-attention ring populated, drafts generated+accepted.
  - Warning: `drafter K/V projection cache unavailable; using full-window K/V projection` (CUDA-only;
    Vulkan limitation, fork has it too — NOT the quality issue).
  - **Multi-slot (--parallel 4): dparams crash fixed (commit 805e1560e) but 0 drafts** — hidden-state
    capture not wired for multi-slot (`begin hidden[0] shape mismatch: embd=0`). `--no-kv-unified` did
    NOT fix it; the difference is n_parallel itself (1 vs 4), not kv_unified. Needs fork's per-slot spec
    architecture (each slot its own DFlash impl/capture/seq_id) — larger port.
- HF-025: **MERGE DFlash QUALITY REGRESSION vs fork (2026-07-04)** ❌
  - Same model pair (Qwen3.6-27B-Q4_K_M + Qwen3.6-27B-DFlash-Q4_K_M), --parallel 1, Vulkan:
    - FORK binary: output CORRECT ("Here's a thinking process: 1. Analyze... 2+2=4. 4. Form..."),
      acceptance **34%** (154/452), 190 tokens, 8.8 tok/s.
    - MERGE binary: output GARBLED ("Thinking Process:escap with of"), acceptance **6.7%** (5/75),
      11 tokens, 4.4 tok/s.
  - Both have same Vulkan K/V-cache limitation → NOT the cause.
  - Merge generates 6x fewer drafts (75 vs 452) and accepts 30x fewer → cross-attention / draft path
    is incomplete. This is part of the ~652 lost server DFlash decode integration lines (HF-017).
  - The merge's pre_decode uses the standard dparams/draft flow; the fork used
    common_speculative_draft_batch + common_speculative_process + per-slot spec + capture-enable
    wiring (fork server lines 4636, 6047, 6275, 2777) that the merge lost.
  - **The DFlash impl (common/speculative.cpp) is IDENTICAL fork-vs-merge** (only diff: my n_seq_in
    change + 2 TODO state stubs). The ENTIRE quality + multi-slot difference is SERVER-side
    (server-context.cpp invocation). No impl changes needed for the port.
  - Missing server calls (APIs all exist in merge): `common_speculative_set_prefill_capture_enabled`
    (fork:6047,6298,6367,6373 — per-view capture control), `common_speculative_draft_batch`
    (fork:4636 — batched draft prep), `llama_dflash_prefill_capture_begin/end` (fork:6040 — suffix
    span capture scheduling), per-slot `slot.spec` (fork:2777). Merge DOES call common_speculative_process.
  - Root quality cause (likely): fork captures only the prompt SUFFIX via
    llama_dflash_prefill_capture_begin(ctx_tgt, slot.id, capture_begin, capture_end); merge doesn't
    schedule span capture → cross-attention data is wrong → garbled drafts (6.7% vs 34% acceptance).
  - Full fix = port fork's server DFlash decode integration (~300 lines of decode-view logic +
    per-slot spec). Well-defined since impl is identical.
- HF-026: **GLM port-strategy review (2026-07-04)** — recommended incremental approach:
  - Step 0 (probe): add `common_speculative_set_prefill_capture_enabled(spec, true)` unconditionally
    before decode; cheap go/no-go. (Note: capture already ON by default — probe likely no-op; real
    fix is #3 span scheduling.)
  - Step 1 (B, single-slot quality, ~60-100 lines): port fork's `should_flush_dflash_prefill` span
    computation (fork:5868) + `llama_dflash_prefill_capture_begin/end` (fork:6040,6179,6303) +
    view-gated `set_prefill_capture_enabled` toggling. Copy fork span logic VERBATIM (span boundaries
    are highest risk — wrong suffix vs whole prompt → worse than 6.7%). Target ~34% acceptance.
  - Step 2 (C, multi-slot): per-slot `slot.spec` (fork:2777, each n_seq=1, seq_id=0).
  - Step 3: SKIP `common_speculative_draft_batch` (same impl->draft path as common_speculative_draft).
  - GLM: quality culprit = #1 capture-enable + #3 span scheduling (NOT #2 draft_batch, NOT #4 per-slot
    for single-slot quality). #4 is multi-slot only.
- HF-027: **Fork prefill suffix flush = the quality key** — `should_flush_dflash_prefill` lambda
  (fork:5868) computes `common_dflash_prefill_span{capture_begin, capture_end, src_offset, n_tokens}`
  (the prompt SUFFIX, not whole prompt). `llama_dflash_prefill_capture_begin(ctx_tgt, slot.id,
  span.capture_begin, span.capture_end)` (fork:6040) schedules span-specific capture. Merge captures
  without span scheduling → wrong cross-attention context → 6.7% acceptance vs fork 34%.
- HF-028: **Ring buffer data is SIMILAR fork-vs-merge — quality gap is NOT in capture** (2026-07-05)
  - GGML_DFLASH_RING_DUMP=1 dumped ring_buf[layer][i] values from both merge and fork
  - MERGE Layer[0]=1 first_8 (positions 0-7): `-0.2200, +0.7027, +0.2269, +0.1306, +0.6427, -0.2198, +0.2709, -0.3782`
  - FORK Layer[0]=1 first_8 (positions 0-7): `-0.2250, +0.7197, +0.2430, +0.1420, +0.6544, -0.2123, +0.2634, -0.3678`
  - Difference: 2-8% (Vulkan device/memory layout differences — NVIDIA RTX vs AMD iGPU)
  - MERGE draft quality: 6.7% (garbled output, 0.3 tok/s)
  - FORK draft quality: 84% (correct output, 6.7 tok/s)
  - **CONCLUSION: Ring capture is CORRECT in merge. Quality gap is in data USAGE:**
    (1) cross-attention processing (build_cross_data, cross_buf layout, read_start)
    (2) drafter forward pass interpretation of cross data
    (3) server-side wiring (draft_params, speculative server flow)
  - Ring data dump code (GGML_DFLASH_RING_DUMP=1) is now in both merge and fork.
  - This narrows the quality investigation from "ring capture" to "cross-attention processing".
- HF-029: **GDB flow comparison merge vs fork — merge does 2.6x more draft cycles** (2026-07-05)
  - GDB breakpoints traced function call counts for a single "What is 2+2?" request:
    | Function           | Merge | Fork | Ratio |
    |--------------------|-------|------|-------|
    | DRAFT_CALL         | 16    | 6    | 2.6x  |
    | MEM_SEQ_RM         | 113   | 23   | 4.9x  |
    | ALIGN_DRAFTER      | 16    | 6    | 2.6x  |
    | PREFILL_CAPTURE    | 32    | 12   | 2.6x  |
    | BUILD_CROSS_DATA   | 0*    | 6    | —     |
    | PREPARE_BATCH_DRAFT| 0     | 0    | —     |
    | DRAFT_BATCH        | 0     | 0    | —     |
    * Merge BUILD_CROSS_DATA=0: breakpoint didn't resolve (different symbol mangling)
  - MEM_SEQ_RM per draft cycle: merge ~7 vs fork ~3.8 (merge does 2x more per cycle)
  - Merge mem_seq_rm call sites (56 total): 39 from common_context_seq_rm→update_slots,
    8 from update_drafter_kv_cache, 7 from common_speculative_draft, 7 from post_decode,
    7 from pre_decode
  - Fork mem_seq_rm call sites (23 total): 7 from update_drafter_kv_cache, 6 from
    common_speculative_draft(6-arg)→draft(), 6 from common_context_seq_rm→update_slots
  - **KEY: Merge uses 1-arg common_speculative_draft(spec.get()); fork uses 6-arg
    common_speculative_draft(spec, params, prompt, id_last, log_probs, n_past)**
  - Both use the SAME DFlash impl::draft() function (verified identical source)
  - The extra merge calls are a SYMPTOM of low acceptance (more draft cycles needed),
    NOT the root cause
  - Tested switching to 6-arg common_speculative_draft: quality UNCHANGED (5.6% vs 6.7%),
    REVERTED. The 1-arg vs 6-arg difference is NOT the quality cause.
  - Fork speed: 32.1 tok/s, correct output; Merge speed: 4.2 tok/s, garbled output
  - **CONCLUSION: Quality gap is NOT in server-side draft invocation (1-arg vs 6-arg).
    The gap is in the DRAFT CONTENT — the drafter produces wrong tokens given the same
    cross-attention data. Next: investigate drafter graph/forward pass differences
    (dflash_draft.cpp, cross-attention consumption in the drafter graph).**
- HF-030: **GLM review of GDB findings (2026-07-05) — 5 suspects, KV cache highest**
  - GLM (glm-5.2:cloud) analyzed the GDB flow comparison and identified 5 suspects:
    1. **HIGHEST: Drafter KV cache state (MEM_SEQ_RM 113 vs 23)** — 4.9x ratio is NOT
       proportional to 2.6x other call counts → KV cache management fundamentally different.
       If sequences removed at wrong times, drafter has stale/missing context.
    2. Drafter graph building — different attention mask, layer connectivity, tensor shapes
    3. Drafter model loading — wrong weight order/precision, missing/duplicated layers
    4. Cross-attention consumption — drafter reads cross data differently (head mapping,
       scaling, positional handling)
    5. Drafter sampling — different temperature/top-k/top-p or argmax bug
  - GLM recommendation: investigate drafter KV cache state FIRST (strongest signal)
  - BUILD_CROSS_DATA breakpoint didn't resolve in merge (different symbol mangling) —
    GLM flagged as red flag: if symbol mangling differs, the function might actually
    be DIFFERENT even if source looks identical
- HF-031: **ROOT CAUSE FOUND AND FIXED — missing llama_model_share_tensors** (2026-07-05) ✅
  - **Root cause**: merge was missing `llama_model_share_tensors(model_dft.get(),
    llama_get_model(ctx_tgt))` which fork calls at adb92b36a:2526.
  - Without this call, the drafter's lm_head (model.output) is NULL. The DFlash graph
    code (dflash_draft.cpp:1206) falls back to a Q4_0 placeholder tensor, producing
    garbage logits. The argmax of garbage logits is all zeros, so every draft token is 0.
  - **GGML_DFLASH_TOKEN_TRACE=1 revealed the smoking gun**:
    - MERGE: first_draft_ids=[0,0,0,0,0,0,0,0] — ALL ZEROS (garbage)
    - FORK:  first_draft_ids=[1817,25,271,16] — MEANINGFUL TOKENS (correct)
  - **Results AFTER fix (was before)**:
    - Token trace: [8340,25,271,16,13,220,2972,2014] (was [0,0,0,0,0,0,0,0])
    - Draft acceptance: 55.6% (25/45) (was 6.7%)
    - Speed: 19.4 tok/s (was 4.2 tok/s)
    - Output: 'Thinking Process: 1. Identify the user input: The user is asking
      What is 2+2?. 2.' (was garbled 'Thinking Process:en design: The')
  - **Also fixed in same commit**:
    - Added `params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft)`
      (fork adb92b36a:2523, was missing in merge)
    - Added `cparams.dflash_n_slots` and `cparams.dflash_cross_ctx` for ctx_dft creation
      (fork sets these in common_speculative_create_ctx_dft, was missing in merge)
  - **Debug tools added** (kept for future use):
    - GGML_DFLASH_RING_DUMP=1: dump ring buffer data (HF-028)
    - GGML_DFLASH_KV_TRACE=1: dump drafter KV cache state (HF-029)
    - GGML_DFLASH_TOKEN_TRACE=1: dump drafter output token IDs (HF-031)
  - **Investigation path**: ring dump (HF-028, similar) → GDB flow comparison (HF-029,
    2.6x more draft calls) → GLM review (HF-030, KV cache suspect) → KV trace (pos_max
    differences, symptom) → token trace (all zeros, smoking gun) → graph comparison
    (identical) → share_tensors missing (root cause).
- HF-032: **Profit controller is NOT the remaining quality gap** (2026-07-05)
  - Tested capping n_draft_max=4 (matching fork's adaptive profit controller output_len=5):
    - Acceptance DROPPED to 50% (was 55.6% with n_draft=15)
    - Speed DROPPED to 13.0 tok/s (was 19.4 tok/s)
    - **n_draft=4 is WORSE than n_draft=15 for the merge** — disproves hypothesis
  - Token trace comparison (n_draft=4, same id_last=90700):
    - MERGE: first_draft_ids=[8340,25,271,16]
    - FORK:  first_draft_ids=[1817,25,271,16]
    - Position 1 DIFFERS (8340 vs 1817); positions 2-4 match (25, 271, 16)
  - committed_len DIFFERS: merge=17, fork=20 (3-token prefill capture difference)
  - **CONCLUSION: The remaining 55.6%→84% gap is NOT the profit controller.**
    The gap is the drafter producing DIFFERENT predictions (position 1: 8340 vs 1817)
    due to different cross-attention context (committed_len 17 vs 20).
  - n_draft=15 (merge default, capped by block_size-1) gives 55.6% — this IS the
    merge's optimal n_draft. The profit controller would settle on 15 too.
  - Full profit controller port would NOT close the gap. Next: investigate the
    committed_len 17 vs 20 prefill capture difference.
- HF-022: **GLM review (glm-5.2:cloud) of commit 9a67f5636 returned 3 concerns — ALL RESOLVED (2026-07-04)**:
  1. **Fix 3 UB risk → MOOT**: `swa_layers` is `std::array<uint32_t, LLAMA_MAX_LAYERS>` (fixed-size,
     llama-hparams.h:150), zeroed at llama-model.cpp:1104. Never empty → no out-of-bounds. No fix needed.
  2. **Fix 4 n_rs_seq regression → VERIFIED NO REGRESSION**: re-ran Qwen3.6-27B-Q4_K_M WITHOUT
     --spec-type dflash with the new binary → 12.7 tok/s, correct output (matches Tier 1's 12.6 tok/s).
     The keep=true branch (n_rs_seq=8 for QWEN35 dense, llama-context.cpp:266) IS exercised by non-DFlash
     Qwen3.6 and the per-slot writeback works for it too.
  3. **Fix 1+2 3-arg init dispatch → FIXED (commit b878e6435)**: the 3-arg overload only handles
     DFLASH/SUFFIX/COPYSPEC/RECYCLE; DRAFT_SIMPLE/EAGLE3/MTP/NGRAM need the 2-arg overload. Replaced
     the unconditional 3-arg call with a type dispatch (matches fork adb92b36a's main-spec vs per-slot
     separation). Verified: DFlash still initializes; non-DFLASH speculative preserved.
  - GLM final recommendation was accept-with-changes; all changes applied/verified.

### CI Fix #1 (commit `49da28e80`)
- Root cause: `src/llama-context.cpp:4123` used `__attribute__((noinline))` — GCC/Clang-only
- Fix: portable `__declspec`/`__attribute__` behind `#if defined(_MSC_VER)`
- Also fixed `%ld` → `%lld` with `(long long)` cast at `llama-context.cpp:9294`
- Verified: all 4 Linux backends 0 errors, no regression

### CI Fix #2 (commit `1e64f481a`)
- Root cause: `test-dflash-plumbing.exe` calls `llm_arch_supports_rs_rollback()` — internal non-`LLAMA_API` symbol
- On Windows shared libs (BUILD_SHARED_LIBS=ON), only LLAMA_API symbols exported → LNK2019
- Fix: wrap test-dflash-plumbing in `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` (matches upstream test-llama-archs pattern)
- Verified: all 4 Linux backends 0 errors, test-dflash-plumbing links+runs (exit 0)

### Inference Test Results (AMD Strix Halo 8060S iGPU, Vulkan/RADV)
| Model | Size | Quant | Type | Speed | Correctness |
|-------|------|-------|------|-------|-------------|
| Qwen3-Coder-Next-Q4_K_M | 46B | Q4 | Dense | 53 tok/s | 5/5 PASS |
| Qwen3.6-35B-A3B-UD-Q6 | 35B/3B | Q6 | MoE | 44.5 tok/s | PASS |
| Qwen3.6-27B-Q4_K_M | 27B | Q4 | Dense | 12.6 tok/s | PASS |
| Qwen3.6-27B-Q6_K | 27B | Q6 | Dense | 9.5 tok/s | PASS |

---

## Hypotheses

- H-001: Linker errors can be fixed by implementing missing DFlash functions
  Status: **disproven** (merge strategy changed, not needed)

- H-002: DFlash init fails due to model path handling bug
  Status: **disproven** — root cause was server calling 2-arg common_speculative_init (HF-019) + 3 other
  issues (HF-020). Model path handling was a red herring.

- H-003: DFlash on Vulkan requires `--spec-draft-model` + matching draft+target pair
  Status: **confirmed** — fork binary + merge binary both work with Qwen3.6-27B-Q4_K_M target +
  Qwen3.6-27B-DFlash-Q4_K_M draft.

- H-004: DFlash draft generation crash (HF-021) is due to missing target hidden-state capture wiring
  (eval callback / GPU ring population) — part of the ~652 lost server lines (HF-017).
  Confidence: medium-high
  Evidence for: `begin hidden[0] shape mismatch: embd=0 expected=5120 tokens=0` — target hidden states
  never reach the cross-attention ring. Fork server had hidden-state-capture wiring that merge lost.
  Evidence against: not yet checked whether the eval callback is set up at all in the merge.
  Validation plan: grep merge server for `llama_set_eval_callback` / cross-data setup; compare to fork;
  restore the missing capture wiring.
  Status: **open**

- H-005: Fix 3 (swa_layers→is_swa_impl copy) has a UB risk if swa_layers is empty when n_swa > 0
  (GLM concern #1).
  Status: **disproven (moot)** — swa_layers is `std::array<uint32_t, LLAMA_MAX_LAYERS>` (fixed-size,
  zeroed at llama-model.cpp:1104). Never empty. No UB. GLM assumed vector; verified array.

### Attempt A-016: GLM concern verification + dispatch fix (commit b878e6435)
Time: 2026-07-04
Hypothesis targeted: H-005 (disproven) + GLM concerns #2/#3
Change: verified swa_layers is fixed-size array (GLM #1 moot); re-ran Qwen3.6-27B no-DFlash
(GLM #2 no regression, 12.7 tok/s); added type dispatch in server-context.cpp (GLM #3, commit b878e6435).
Commands: `cmake --build build-ci-vulkan`; run Qwen3.6-27B without --spec-type dflash; run with dflash.
Result: **passed** — all 3 GLM concerns resolved. DFlash still initializes; non-DFlash Qwen3.6 unchanged.
Next: debug DFlash draft generation (HF-021, H-004) — step (b).

---

## Tried solutions

### Attempt A-001 through A-009 (Merge phase)
All passed — 0 compile errors achieved.

### Attempt A-010: MSVC noinline fix (CI Fix #1)
- Change: portable `__declspec`/`__attribute__` in `llama-context.cpp`
- Result: passed — all backends build, CI run #10 confirmed compilation succeeded.

### Attempt A-011: test-dflash-plumbing gating (CI Fix #2)
- Change: wrap test in `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` in `tests/CMakeLists.txt`
- Result: passed — CI run #11 confirmed BUILD FIXED (exit 8 = test failures only).

### Attempt A-012: DFlash init debugging (model path / graph registration) — DEAD END
- Change: Investigated model loading order, DFlash graph builder registration.
- Result: **failed** — DFlash still not initializing ("no implementations specified").
- Learning: model path handling was a red herring (H-002 disproven). Real cause was the 2-arg init overload.

### Attempt A-013: Inference testing Tier 1
- Change: Tested Qwen3.6 family models on iGPU.
- Result: passed — 5/5 prompts correct, performance measured.

### Attempt A-014: DFlash init — 4 fixes (commit `9a67f5636`) ✅
Time: 2026-07-04
Hypothesis targeted: H-002 (disproven → replaced by HF-019/020 root causes)
Change: server-context.cpp (2 fixes), dflash_draft.cpp (1 fix), delta-net-base.cpp (1 fix) — see HF-020.
Commands: `cmake --build build-ci-vulkan -j; cmake --build build-ci-debug -j` then run llama-server
`--spec-type dflash --spec-draft-model Qwen3.6-27B-DFlash-Q4_K_M.gguf`.
Result: **passed** — DFlash initializes (matches fork output), server stays up (exit 124 = running).
Evidence: log shows `adding implementation dflash`, `dflash: contract ok`, `GPU cross ring enabled`.
Learning: 4 independent issues each blocked init; the DeltaNet writeback crash was in the TARGET graph
(QWEN35 arch), not the DFlash draft. Fork binary comparison was the key diagnostic (user's tip).
Next: address GLM concerns (H-005); debug draft generation (HF-021, H-004).

### Attempt A-015: GLM review of commit 9a67f5636
Time: 2026-07-04
Hypothesis targeted: none (verification of A-014)
Change: sent the 4-fix diff to glm-5.2:cloud for independent review.
Result: **partial** — GLM said accept-with-changes, 3 concerns (HF-022).
Evidence: GLM response (logged in conversation). Highest-priority concern: Fix 3 swa_layers UB risk.
Learning: GLM assumed swa_layers is a vector; it's actually `std::array<uint32_t, LLAMA_MAX_LAYERS>`
(per llama-hparams.h:150 — NEEDS VERIFICATION). If array, concern #1 is moot.
Next: verify swa_layers type; if vector add guard; verify 3-arg overload dispatch + n_rs_seq gating.

---

## Dead ends / do not retry unchanged

- **dflash-server-utils decode hooks**: Bypass real DFlash cross-attention in common_speculative. Reference/unit-test only.
- **0 compile errors = functional proof**: Phase 5d/5e claimed ~92% complete but was functionally 0% (DFlash not initializing).
- **Bulk `ggml_view_3d` DeltaNet writeback (upstream)**: Wrong per-snapshot stride overflows ssm_states_all
  under DFlash (n_rs_seq>0). Use the fork's per-slot `ggml_view_2d`/`ggml_view_4d` loop instead.
- **Passing 4D state to `ggml_gated_delta_net` when K>1 needed**: op sees K_actual=1, emits 1 snapshot,
  downstream K-expecting views overflow. Must pad to 3D `(D, K, n_seqs)` first.
- **Model-path-swap hypothesis (H-002)**: red herring — server was loading models correctly; the init
  overload choice was the bug.

---

## Next actions

### Priority 1: GLM review concerns (HF-022) — DONE (HF-023: all 3 resolved; commit b878e6435)

### Priority 2: DFlash decode QUALITY + MULTI-SLOT port (NEXT — plan in HF-026/027)
  Impl is IDENTICAL fork-vs-merge; fix is purely server-side wiring.
  Step 1 (B, single-slot quality, ~60-100 lines): port fork's prefill suffix-flush span scheduling
  into decode() BEFORE llama_decode (~line 3691) + capture_end AFTER common_speculative_process (~line 3757).
  Copy fork should_flush_dflash_prefill (fork:5868) VERBATIM — capture_from = max(0, prompt_total - cross_ctx),
  span_begin/end = the suffix window. APIs all exist (common_dflash_prefill_span, llama_dflash_prefill_capture_begin/end,
  common_speculative_set_prefill_capture_enabled). Target ~34% acceptance, correct output. Test: --parallel 1.
  Step 2 (C, multi-slot): per-slot slot.spec (fork:2777, each n_seq=1, seq_id=0); reverts n_seq=n_parallel workaround.
  Step 3: SKIP common_speculative_draft_batch (same impl->draft path). Risk: span boundaries highest — copy verbatim.

### Step 1 PROGRESS (commits ee9b2210b, cec71db2d) — ring now grows, quality still low
  DONE (ported, committed, pushed):
    - Prefill suffix-span capture scheduling (should_flush_dflash_prefill + llama_dflash_prefill_capture_begin/end
      + set_prefill_capture_enabled). Helps long prompts (suffix-only capture).
    - common_speculative_update_logits_deferred_dflash_kv call after accept (DFlash accept() is a no-op;
      ring append happens via update_logits -> append_target_hiddens). Ring NOW GROWS during generation
      (DFLASH_RX append: ring_filled 17->19->21->23...; was frozen at 17).
  RESULT: ring grows (real improvement) BUT output still garbled, acceptance 6.7% (5/75) vs fork 34%.
  RULED OUT as the cause:
    - Span scheduling: for 17-token test prompt capture_from=max(0,17-512)=0 so span=whole prompt (no change).
    - n_layer()=64 (merge) vs n_layer=65 (fork): llama_model_n_layer returns n_layer()=n_layer_all-n_layer_nextn=64
      vs fork n_layer member=65. DFlash contract uses it ONLY for validation (target_layer_ids all <64). NOT the cause.
    - K/V projection cache unavailable: both fork+merge on Vulkan (CUDA-only). NOT the cause.
  REMAINING quality candidates (need larger fork decode-flow port):
    - Adaptive 'profit' draft-max controller: fork n_draft=4 (adaptive), merge n_draft=15 (block_size-1, no
      controller). Fork get_n_draft_max uses common_speculative_n_min + profit (fork:1017-1050); merge simple (merge:438).
    - 3-token non-speculative WARMUP: fork does 3 append(1) before first draft (ring 17->20); merge drafts
      immediately at ring=17. Likely needs n_min/capture_min warmup (lost HF-017).
    - Possibly capture-content issue (hidden values at [1,16,31,46,61]) — not yet verified.
  NEXT: Switch merge's draft path to 6-arg common_speculative_draft() (same as fork). Key differences:
    - Fork: common_speculative_draft(slot.get_spec(), params_spec, ..., draft_n_past) where draft_n_past=-1 for DFlash
    - Merge: 1-arg common_speculative_draft(spec.get()) with manual draft_params (n_past=slot.prompt.n_tokens())
    - 1-arg version doesn't enforce n_min (only truncates to n_max), 6-arg checks result.size() < n_min after
    - 6-arg function exists in merge's speculative.cpp (line 4627-4700), just not called from server
    - Also: draft_batch is fork's multi-slot path only (n_drafting<2 returns false), NOT the quality cause
  TODO: Test switching to 6-arg version with draft_n_past=-1 for DFlash.
  Also TODO: Prepare cross-attention data dump for merge-vs-fork comparison.

### Step 2: Switch to 6-arg common_speculative_draft() (IN PROGRESS)
  Fork's single-slot DFlash path (server-context.cpp:4856-4859):
    slot.spec_draft = common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, nullptr, draft_n_past);
    where: draft_n_past = (use_mtp_spec ? slot.prompt.n_tokens() : -1) = -1 for DFlash
  Merge's single-slot DFlash path (server-context.cpp:3093-3109):
    Sets draft_params manually, then calls 1-arg common_speculative_draft(spec.get())
  Key differences:
    1. 6-arg passes n_min=params_spec.n_min, 1-arg doesn't set n_min (defaults to 0)
    2. 6-arg post-checks: if result.size() < n_min -> clear. 1-arg doesn't.
    3. 6-arg uses slot.get_spec() (per-slot), merge uses spec.get() (shared)
    4. 6-arg passes draft_n_past=-1, merge sets n_past=slot.prompt.n_tokens()
    BUT: draft() ignores dp.n_past for DFlash (uses committed_len instead)
    AND: n_min=0 makes the post-check a no-op
    SO: 6-arg vs 1-arg should be equivalent for DFlash with n_min=0
    HYPOTHESIS: The 6-arg function's internal dp setup might differ subtly from manual setup
  Plan: Replace merge's manual draft_params + 1-arg call with 6-arg call matching fork exactly.
  This is a small change (3 lines) but could fix quality if there's a subtle difference in dp setup.

### Step 3: Cross-attention data dump for merge-vs-fork comparison
  TODO: Add dump of cross-attention ring data (first N values per layer) to merge's draft()
  Compare with fork's values to verify ring content is identical
  This requires modifying speculative.cpp (DFlash draft()) to log ring data at build_cross_data time

  STATUS: init ✅, decode ✅, ring grows ✅, quality ❌ (6.7% vs 34%).
  Remaining suspects: 6-arg vs 1-arg draft path (subtle dp setup diff), cross-attention data.
  draft_batch port is for MULTI-SLOT only (not quality fix) - already available in speculative.cpp.

### Step 1 EXHAUSTIVE INVESTIGATION (commits ee9b2210b, cec71db2d, 87be163ec) — quality still 6.7%
  ALL of these ported/verified, NONE fixed quality (6.7% acceptance, garbled output vs fork 34%):
    - Prefill suffix-span capture scheduling (ee9b2210b) — no change for 17-token test prompt
      (capture_from=max(0,17-512)=0, span=whole prompt).
    - common_speculative_update_logits_deferred_dflash_kv ring append (ee9b2210b) — ring NOW
      GROWS (was frozen at 17; now 17->19->21->23...). Real improvement, but quality unchanged.
    - Deferred variant (cec71db2d) — matches fork defer_kv=1. No quality change.
    - 3-token non-speculative warmup (tested, reverted) — fork does 3 append(1) before first
      draft (ring 17->20); merge warmup hack (skip speculation for n_decoded<3) did NOT help.
    - Drafter ctx=256 (87be163ec) — fork sets -cd 256; merge used 4096. Ported. No quality change.
    - n_draft count: fork n_draft=4 (adaptive profit), merge n_draft=15 (block_size-1, no
      adaptive controller). NOT the correctness cause (acceptance RATE differs, not just count).
    - get_n_draft_max doesn't clamp to configured n_max (real bug, but fixing won't fix quality
      since n_draft count isn't the correctness cause).
  RULED OUT as the quality cause:
    - DFlash impl (speculative.cpp): IDENTICAL fork-vs-merge (only n_seq param + 2 TODO stubs diff).
    - Capture mechanism (llama-context.cpp): essentially identical (47-line diff, whitespace).
    - n_layer()=64 (merge) vs n_layer=65 (fork): validation-only (target_layer_ids all <64).
    - K/V projection cache: both Vulkan-limited (CUDA-only feature).
    - Capture layer mapping: correct (tensors l_out-<id> found, no shape mismatch).
  REMAINING SUSPECT (next step): the DRAFT PATH itself.
    - Fork uses common_speculative_draft_batch (line 4636) -> DFlash impl prepare_batch_draft()
      (line 2664, the BATCH path: build_cross_data then a separate drafter batch decode).
    - Merge uses common_speculative_draft (line 3078) -> DFlash impl draft() (line 3029, the FLAT
      path: build_cross_data + inline drafter decode + extract).
    - These are DIFFERENT DFlash impl methods. GLM's "same impl->draft path" was INCORRECT.
    - Switching to common_speculative_draft_batch is the next step — complex API change
      (multi-slot batch: specs vector, id_last_per_spec, results, log_probs; fork:4636-4650).
  ALSO UNVERIFIED: actual hidden VALUES (ring content). DFLASH_DBG append logs layer0_first
    values but needs -lv 4 / env to enable. Could dump via dflash_dump_hidden_states (line 9295).
    If hidden values differ merge-vs-fork, the capture content is wrong despite right layers/dim.
  STATUS: init ✅, decode ✅ (no crash), ring grows ✅, quality ❌ (6.7% vs 34%).

### Priority 1-OLD (superseded by Priority 1 above)
1. **Verify `swa_layers` type
1. **Verify `swa_layers` type** (H-005): check `llama-hparams.h:150`. If `std::array<uint32_t, LLAMA_MAX_LAYERS>`
   (fixed size), Fix 3 has NO UB risk → GLM concern #1 moot, document it. If `std::vector`, add a size
   guard `if (hparams.swa_layers.size() >= hparams.n_layer())` before the loop.
2. **Verify 3-arg `common_speculative_init` dispatch** (GLM concern #3): confirm it handles
   DRAFT_SIMPLE/EAGLE3/MTP/NGRAM (not only DFLASH) so non-DFLASH speculative didn't regress.
3. **Verify `n_rs_seq > 0` gating** (GLM concern #2): confirm the keep=true branch in
   `build_recurrent_attn` is only reached under DFlash, OR that the per-slot writeback is correct for
   non-DFlash Qwen3.6 too. (Note: the merge's bulk view_3d was already broken on this branch, so this
   fix likely only restores fork behavior — but confirm.)
4. If any concern is real, add a follow-up commit; else document the verification in TASK_PROGRESS.md.

### Priority 2: Debug DFlash draft generation (HF-021, H-004)
- grep merge `tools/server/server-context.cpp` for `llama_set_eval_callback` / cross-data / hidden-capture setup.
- Compare to fork adb92b36a server-context.cpp (~652 lost lines, HF-017) — find the hidden-state-capture wiring.
- Restore the eval-callback + GPU-ring-population path so `begin hidden[0]` gets embd=5120, tokens>0.
- Re-test: send a prompt, confirm draft generation produces output (not crash).

### Priority 3: CI Fix #3 — OpenVINO Windows test failures (run #11)
- test-cuda-zero-dim-gemm: require CUDA backend by name (skip on OpenVINO/Metal GPU backends).
- test-backend-ops: exclude from openvino ctest (pre-existing upstream OpenVINO limitation).

### Priority 4: macOS test failure
- Need ctest output to determine if test-backend-ops (WebGPU SEGFAULT) + test-llama-archs (DFlash arch) failing.

### Priority 5 (optional): Additional inference testing
- Gemma 4 31B (cross-family), Qwen3-Coder-Next-Q8_0 (quality comparison).

---

## Validation

- Build: `cmake --build build-ci-vulkan -j; cmake --build build-ci-debug -j` → 0 errors ✅
- CI: Build FIXED (exit 8 = test failures only) ✅
- Baseline inference: 5/5 prompts correct ✅
- **DFlash initialization: ✅ WORKS** (commit 9a67f5636, matches fork output)
- **DFlash draft generation: ❌ crashes** (HF-021, hidden-state capture empty — next)
- GLM review: accept-with-changes (HF-022) — concerns being addressed (Priority 1)

## Notes

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Remote: `gboddaer` = `git@github.com:gboddaer/beellama.cpp.git`
- Push policy: ONLY push to `gboddaer/merge_llama_into_beellama_2` (PR branch); `gboddaer/main` release-only.
- `gboddaer/main` reset to `adb92b36a` (pre-merge fork main).
- GLM reviewer available at http://192.168.123.123:11434 (glm-5.2:cloud).
- Fork binary (working DFlash reference): `/crypt/beellama.cpp/build-vulkan/bin/llama-server` (on adb92b36a).
- Do NOT `pkill -f llama-server` — kills the user's running server. Use distinct ports (8090+) per test.
- No time limit, no token limit (user instruction).
## HF-034: Multi-Slot DFlash Implementation (2026-07-05)

### Architecture Ported (commit ad46a8703)
- **server_slot struct refactored**: `spec` is now `common_speculative_ptr` (owned unique_ptr), added `spec_shared` (non-owning fallback), `get_spec()`, `get_seq_id()`
- **Per-slot spec creation**: Each DFlash slot gets its own `common_speculative_init(params, ctx_tgt, ctx_dft)` with `n_seq=1`
- **Shared drafter context**: `ctx_dft` is kept (not reset) for DFlash; all slots share one drafter context
- **seq_id fix**: `get_seq_id()` returns 0 for per-slot specs, `slot.id` for shared specs (fixes dparams out-of-bounds crash)
- **set_active_slot**: Added `llama_dflash_set_active_slot(ctx_tgt, slot.id)` before draft/accept/update_logits
- **n_outputs_max**: Increased drafter's `n_outputs_max` to account for draft tokens (fixes output_reserve crash)
- **dflash_n_slots**: Set `llama_set_dflash_n_slots(ctx_tgt, n_parallel)` for target context
- **All per-slot operations**: Replaced `spec.get()` with `slot.get_spec()` in 15+ locations

### Results
| Mode | Acceptance | Speed | Drafts | Output | Status |
|------|-----------|-------|--------|--------|--------|
| --parallel 1 | 43.6% | 20.9 tok/s | 5 cycles, 24/55 accepted | correct | ✅ WORKING |
| --parallel 4 | 0% | 11.4 tok/s | 18 calls, 0 generated | correct (non-spec) | ⚠️ 0 drafts |

### Multi-Slot Issue
- 4 per-slot DFlash specs created ✅
- Shared drafter context works ✅ (1 warmup, not 4)
- But 0 drafts generated (18 draft calls, 0 output)
- Error: `begin hidden[0] shape mismatch: embd=0 expected=5120 tokens=0`
- Warning: `set_active_dflash_slot: slot 2 out of range [0, 1); ignoring` (same in fork)

### GLM Review (glm-5.2:cloud)
- Key insight: fork likely sizes `hidden_gpu`/`prefill_gpu` to `n_parallel`, not `layer_hiddens`
- `set_active_dflash_slot` IS needed — routes captured data to correct per-slot ring
- 0 drafts chain: n_slots=1 → set_active_dflash_slot rejected → no capture → empty ring → embd=0
- Recommendation: inspect fork's `hidden_gpu`/`prefill_gpu` sizing, replicate in merge

### Next Steps
- Investigate fork's `hidden_gpu`/`prefill_gpu` allocation for multi-slot
- Check if fork calls `allocate_tape_gpu(n_parallel, ...)` on target context somewhere
- May need to call `allocate_tape_gpu(ctx_tgt, n_parallel, ...)` during DFlash init

## HF-035: Multi-Slot DFlash — Proven Facts vs Hypotheses (2026-07-05)

### PROVEN FACTS (empirically tested)

1. **Single-slot DFlash WORKS** (`--parallel 1`): 59.3% acceptance (16/27), correct output, 3 draft cycles. No regression from multi-slot changes (with `allocate_slots > 1` guard).

2. **Multi-slot WITHOUT DFlash works** (`--parallel 2`, no `--spec-type dflash`): Output correct ("Here's a thinking process:"). No garbled output. The garbled output is caused by DFlash, NOT by multi-slot itself.

3. **Multi-slot WITH DFlash produces GARBLED OUTPUT** (`--parallel 2`): "Thinking Process: to\n在 to 响应 ,话 to >" — mixed Chinese/English garbage. Also 0 drafts generated.

4. **Garbled output is NOT caused by** (each was disabled and output remained garbled):
   - `llama_dflash_allocate_slots` (disabled completely, still garbled)
   - `common_speculative_draft` (disabled, still garbled)
   - `set_active_dflash_slot` calls (disabled, still garbled)
   - Prefill capture scheduling (disabled, still garbled)
   - GPU capture (`GGML_DFLASH_GPU_RING=0`, still garbled)

5. **`llama_dflash_allocate_slots(ctx_tgt, 1)` BREAKS single-slot**: acceptance drops from 59.3% to 5.3%. Guard `> 1` fixes this. (fork:2814 calls it unconditionally, but fork's `allocate_tape_gpu(1,...)` might be idempotent — merge's is not).

6. **The `llama_dflash_allocate_slots` API EXISTS in the merge** (llama.h:1636, llama-context.cpp:8899) but was NOT being CALLED from server code. Added the call (fork:2814 equivalent).

7. **Prefill capture scheduling was SKIPPED for DFlash** because the guard was `if (spec && ...)` but `spec` is null for DFlash (per-slot specs). Fixed by removing `spec &&` guard. Prefill spans now scheduled (confirmed with `GGML_DFLASH_PREFILL_TRACE=1`).

8. **`flush_prefill` returns 0 for slot > 0**: `dflash prefill flush mismatch: slot=1 requested=13 written=0`. Hidden states not captured for slots > 0.

9. **`set_active_dflash_slot: slot N out of range [0, 1)` warning** appears without `allocate_slots`. With `allocate_slots(2)`, the warning disappears but `flush_prefill` still returns 0.

### HYPOTHESES (not yet proven)

1. **HYPOTHESIS: GPU capture disables eval callback → layer_hiddens empty for slots > 0**. In `decode()` line 6646, `cparams.cb_eval = nullptr` when `dflash_gpu_capture_ready`. But `GGML_DFLASH_GPU_RING=0` (force CPU capture) didn't fix it, so this may be wrong or only partially correct.

2. **HYPOTHESIS: The garbled output is caused by the DFlash impl constructor's `set_dflash_capture(ctx_tgt, ...)` call**. When 2 slots create DFlash impls, `set_dflash_capture` is called twice on the same target context, possibly corrupting graph state. NOT YET TESTED — would need to disable `set_dflash_capture` in the DFlash impl constructor.

3. **HYPOTHESIS: `memory->set_force_split_seq(true)` in `set_dflash_capture` corrupts multi-slot decode**. This forces ubatch splitting by sequence, which might interact badly with the graph cache. NOT YET TESTED.

4. **HYPOTHESIS: The fork's `flush_prefill` reads from `hidden_gpu` (GPU), not `layer_hiddens` (CPU)**. NOT YET VERIFIED — need to read the fork's `flush_prefill` implementation.

5. **HYPOTHESIS: There's a GPU→CPU sync step before `flush_prefill`** (e.g., in `prefill_capture_end`). NOT YET FOUND.

### REMAINING QUESTIONS

1. **Why does the fork's multi-slot DFlash work with the same `set_dflash_capture` code?** Both fork and merge call `set_dflash_capture` per-slot on the same target context. The fork works, the merge produces garbled output. What's different?

2. **What specifically corrupts the output?** The garbled output (mixed Chinese/English) suggests KV cache corruption or position/sequence ID mismatch. Need to identify the exact mechanism.

3. **Is the garbled output caused by the graph being rebuilt with incorrect DFlash capture parameters?** The `set_dflash_capture` call changes `cparams.cb_eval` and `cparams.dflash_capture_layers`. If the graph is cached and not rebuilt correctly, the output could be garbled.

### GLM REVIEW SUMMARY (glm-5.2:cloud)

**GLM Review 1** (HF-034):
- Key insight: fork likely sizes `hidden_gpu`/`prefill_gpu` to `n_parallel`, not `layer_hiddens`
- `set_active_dflash_slot` IS needed — routes captured data to correct per-slot ring
- 0 drafts chain: n_slots=1 → set_active_dflash_slot rejected → no capture → empty ring → embd=0
- **VERIFIED**: `llama_dflash_allocate_slots` was the missing call (fork:2814)

**GLM Review 2** (HF-035):
- **Garbled output is the real problem**, not 0 drafts — 0 drafts should produce correct but slow output (pure target fallback)
- Garbled output means memory corruption or graph corruption
- `allocate_slots` resizing `layer_hiddens` after graph was cached would corrupt tensor pointers
- **PARTIALLY VERIFIED**: `allocate_slots` disabled completely → still garbled, so NOT the sole cause
- Recommendation: isolate garbled output first, force CPU capture as diagnostic
- **DONE**: `GGML_DFLASH_GPU_RING=0` didn't fix garbled output

### NEXT STEPS

1. **Investigate the DFlash impl constructor's effect on the target context** — disable `set_dflash_capture` in the constructor and check if output is correct
2. **Check if `set_force_split_seq(true)` is the cause** — disable it and test
3. **Compare fork vs merge `set_dflash_capture` more carefully** — there might be a subtle difference
4. **Check if the graph cache is being invalidated correctly** when DFlash capture is enabled
5. **Read the fork's `flush_prefill` implementation** to understand GPU vs CPU hidden state reading

## HF-036: LESSON — Longer tests needed for certainty (2026-07-05)

### Methodological failure

Throughout HF-034..HF-035, single-slot DFlash was declared "WORKING" based on
short tests (max_tokens=20-40). This gave **false confidence**. The first 10-20
tokens of DFlash output are often correct (matching the target model's logits
before drafts diverge), but longer generations reveal quality degradation.

### Evidence

When tested with longer prompts (max_tokens=200-400) that allow the model to
complete reasoning AND produce a final answer:

| Prompt | max_tokens | Non-DFlash | Fork DFlash | Merge DFlash |
|--------|-----------|-----------|------------|-------------|
| "Capital of France?" | 200 | ✅ "The capital of France is Paris." | ✅ "The capital of France is Paris." | ❌ Garbled: "onally ies7., - for :.div" |
| "15 * 23 step by step" | 300 | ✅ Coherent step-by-step | ✅ Coherent | ❌ Garbled after first line |
| "Reverse a string" | 400 | ✅ Coherent reasoning | (not tested) | ❌ Garbled: "githubusercontent the the with" |

- **Non-DFlash merge**: perfect, coherent, complete answers
- **Fork DFlash**: perfect, coherent, complete answers (26.7% acceptance, stable)
- **Merge DFlash**: garbled after first few tokens, 13.5% acceptance (degrading)

### Root implication

The merge's single-slot DFlash has a **quality regression** that was hidden by
short tests. The fork's DFlash produces correct long output. The merge's does
not. This is NOT a fundamental DFlash limitation — it's a merge-specific bug.

The acceptance rate degrades over time (29.3% → 13.5% across requests),
suggesting ring buffer corruption or draft quality degradation.

### Rule going forward

**ALWAYS test with prompts long enough for the model to complete its answer
(max_tokens >= 200, ideally 400+).** Short tests (max_tokens < 50) only verify
the first few tokens and can mask quality regressions. A DFlash implementation
is only "working" if it produces correct, coherent output for full-length
generations, not just the first 20 tokens.

### Current state

- **Non-DFlash merge**: ✅ stable, correct output (base reference)
- **Fork DFlash**: ✅ stable, correct output (working reference)
- **Merge DFlash single-slot**: ❌ quality regression (garbled after first few tokens)
- **Merge DFlash multi-slot**: ❌ garbled output (same issue, more severe)

The single-slot quality regression is the **primary blocker**. Multi-slot cannot
be fixed until single-slot produces correct long output.

### Next diagnostic

Testing whether per-slot spec creation (vs shared spec) causes the regression:
- Temporary change: single-slot DFlash uses shared spec (like original merge)
- If output corrects → per-slot spec creation is the cause
- If still garbled → regression was pre-existing (introduced earlier in merge)

## HF-037: DFlash quality regression root cause — n_outputs_max (2026-07-05)

### BREAKTHROUGH FINDING

`--spec-draft-n-max 1` produces **PERFECT output** (correct answer, coherent
reasoning, 62.7% acceptance). Default `--spec-draft-n-max 16` produces garbled
output (43.8% acceptance, garbled after ~20 tokens).

### Key observations

1. `--spec-draft-n-max` does NOT change the number of draft tokens per call
   (both generate 15 tokens/call). It changes `server_n_outputs_max`:
   - n_max=1: `n_outputs_per_seq = 2`, target output buffer = 2
   - n_max=16: `n_outputs_per_seq = 17`, target output buffer = 17

2. The target's output buffer size affects quality, NOT the draft count.
   With a small buffer (2), output is correct. With a large buffer (17),
   output is garbled.

3. The fork uses an ADAPTIVE controller that starts with small drafts (5)
   and grows them. The merge uses fixed n_draft_max = n_ctx - prompt - 2.

### Comparison (same prompt: "Capital of France?")

| Setting | Acceptance | Output | n_outputs_max |
|---------|-----------|--------|---------------|
| merge n_max=1 | 62.7% | ✅ "The capital of France is Paris." | 2 |
| merge n_max=5 | 3.6% | ❌ Completely garbled | 6 |
| merge n_max=16 (default) | 43.8% | ❌ Garbled after ~20 tokens | 17 |
| fork (adaptive 5→16) | 26.7% | ✅ "The capital of France is Paris." | varies |
| merge non-DFlash | N/A | ✅ "The capital of France is Paris." | 1 |

### Root cause hypothesis

The merge's `server_n_outputs_max` computes `n_outputs_per_seq = 1 + n_max`.
For DFlash, this creates a large target output buffer (17 positions) that
somehow corrupts the verification or graph building. With n_max=1, the buffer
is small (2) and the output is correct.

The fork's adaptive controller keeps n_max small initially, which keeps the
output buffer small, which produces correct output. As confidence grows, n_max
increases, but by then the ring buffer has enough context for accurate
verification.

### Next steps

1. **Port the fork's adaptive profit controller** — starts with small n_max,
   grows based on acceptance rate. This is the proper fix.
2. **Investigate why large n_outputs_max corrupts output** — is it the graph
   building, the verification, or the sampling?
3. **Test `--spec-draft-n-max 1` with longer prompts** — verify it's a stable
   workaround.

## HF-038: Quality regression root cause — n_outputs_max corrupts graph (2026-07-05)

### PROVEN by extensive bisection

1. `--spec-draft-n-max 1` (n_outputs_max=2) → **correct output** (all prompts, long generations)
2. `--spec-draft-n-max 5` (n_outputs_max=6) → **garbled output** (regardless of n_draft_max cap)
3. `--spec-draft-n-max 16` (n_outputs_max=17, default) → **garbled output** after ~20 tokens
4. Capping `n_draft_max` to 5 does NOT help — the issue is `n_outputs_max` itself, not draft count
5. The fork has `n_outputs_max=17` (same `server_n_outputs_max`) and **works correctly**
6. DFlash impl, eval callback, output_reserve, server_n_outputs_max are all **identical** fork-vs-merge
7. The crash when capping target n_outputs_max to 2: the target batch includes draft tokens with
   `output=true` (lines 500-503), requiring `n_outputs_max >= n_draft + 1`

### ROOT CAUSE

The merge includes 342 upstream llama.cpp commits that the fork doesn't have. Something in these
commits changed how the target context handles multiple output positions (n_outputs_max > 2).
With n_outputs_max=2, the graph is simple and hidden states are correct. With n_outputs_max >= 6,
the graph changes in a way that corrupts the hidden state capture, leading to garbled DFlash output.

The exact upstream commit that introduced the regression has NOT been identified. Bisecting 342
commits would be needed to find it.

### WORKAROUND

Use `--spec-draft-n-max 1` for correct DFlash output. This sets `n_outputs_max=2` which avoids
the graph corruption. The downside is reduced speed (only 1 draft token per cycle, 60% acceptance).

### What does NOT work

- Capping `n_outputs_max` in `server_n_outputs_max` → crashes (target batch needs n_draft+1 outputs)
- Capping `n_draft_max` to 5 → still garbled (n_outputs_max is still 17 from server_n_outputs_max)
- `GGML_DFLASH_GPU_RING=0` (force CPU capture) → still garbled
- Ring state save/load fix (HF-036) → correct but not the root cause
- Per-slot spec vs shared spec → both garbled (regression is pre-existing)

### Fork vs merge comparison (all identical)

- `common_speculative_impl_dflash` class: identical (22 diff lines: n_seq + TODO stubs)
- `dflash_eval_callback`: identical
- `llama_context::output_reserve`: identical
- `llama_context::set_dflash_capture`: identical
- `llama_context::set_active_dflash_slot`: identical
- `server_n_outputs_max`: identical (both return 17 for DFlash n_max=16)
- `llama_context::decode` DFlash sections: identical (only line number offsets)

### Next steps

1. **Bisect 342 upstream commits** to find which one introduced the n_outputs_max graph corruption
2. **Port the fork's adaptive profit controller** — starts with small drafts, grows based on
   acceptance. This would keep n_outputs_max small initially, avoiding the corruption.
3. **Use `--spec-draft-n-max 1` workaround** for now — correct output, reduced speed
4. **Investigate the graph building** — compare the target's compute graph for n_outputs_max=2
   vs n_outputs_max=17 to find the divergence

## HF-039: ROOT CAUSE — GPU-embedded hidden capture broken in merge (2026-07-05)

### BREAKTHROUGH: Two hidden state capture paths identified

The merge has TWO paths for capturing hidden states from the target context:

1. **Eval callback path** (CPU): `dflash_eval_callback` writes to `layer_hiddens` (CPU).
   - Active when `dflash_graph_hidden_ready = false` (i.e., `hidden_gpu` is empty)
   - Works correctly with `--spec-draft-n-max 1` (n_outputs_max=2, 2 hidden states)
   - Garbled with n_outputs_max > 2 (too many hidden states from draft positions)

2. **GPU-embedded path** (GPU): graph-embedded copies to `hidden_gpu` (GPU buffers).
   - Active when `dflash_graph_hidden_ready = true` (i.e., `hidden_gpu` is allocated)
   - `dflash_graph_hidden_ready = !hidden_gpu.empty() && gpu_capture_ready && ...`
   - When active: eval callback is DISABLED (`cparams.cb_eval = nullptr`, line 7049)
   - **BROKEN in the merge** — produces 6.1% acceptance, garbled output at ALL n_max values

### Key evidence

| `hidden_gpu` allocated? | Capture path | n_max=1 | n_max=16 |
|--------------------------|-------------|---------|---------|
| No (merge, `>1` guard)   | Eval callback | ✅ 60% accept, correct | ❌ 43% accept, garbled |
| Yes (merge, unconditional)| GPU-embedded | ❌ 6% accept, garbled | ❌ 6% accept, garbled |
| Yes (fork, unconditional) | GPU-embedded | ✅ 26% accept, correct | ✅ 26% accept, correct |

### Why the fork works

The fork calls `llama_dflash_allocate_slots(ctx_tgt, dflash_slots_cap)` unconditionally
(even for single-slot, dflash_slots_cap=1). This allocates `hidden_gpu`, enabling
`dflash_graph_hidden_ready = true`, which disables the eval callback and uses the
GPU-embedded path. The fork's GPU-embedded path works correctly.

The merge's GPU-embedded path is **broken** — the 342 upstream commits changed the
graph building in a way that corrupts the GPU-embedded hidden state capture.

### Root cause chain

1. `allocate_tape_gpu(1, ...)` allocates `hidden_gpu` with 1 entry
2. `dflash_graph_hidden_ready = !hidden_gpu.empty() && ...` = true
3. `dflash_skip_eval_callback = dflash_graph_hidden_ready || ...` = true
4. `cparams.cb_eval = nullptr` (eval callback disabled, line 7049)
5. Graph uses GPU-embedded hidden capture (copies to `hidden_gpu`)
6. **The GPU-embedded capture produces wrong hidden states in the merge** (342 upstream commits)
7. Wrong hidden states → wrong cross-attention → wrong drafts → garbled output

### Current workaround

`--spec-draft-n-max 1` + `dflash_slots_cap > 1` guard:
- Uses eval callback path (hidden_gpu not allocated for single-slot)
- n_outputs_max=2 → only 2 hidden states captured → minimal corruption
- Correct output, 60% acceptance, reduced speed

### Next steps

1. **Diff `allocate_hidden_gpu` and graph building** between fork and merge to find
   what the 342 upstream commits changed in the GPU-embedded capture path
2. **GDB comparison** of `hidden_gpu` contents between fork and merge after decode
3. **Check if the graph builder** (`src/models/`) handles `hidden_gpu_seqs` differently
   in the merge vs fork

## HF-040: DFlash GPU-embedded hidden capture port + graph invalidation (2026-07-05)

### Ported from fork to merge

1. **qwen35.cpp DFlash hidden_gpu capture code** — Ported the fork's graph builder
   code that copies layer outputs to `hidden_gpu` buffers during the forward pass.
   Without this, `hidden_gpu` stays empty even when allocated, and the GPU-embedded
   capture path produces garbage (HF-039). Added `#include "llama-context.h"` for
   `dflash_hidden_gpu` struct access.

2. **Graph invalidation for hidden_gpu_n_seqs** — Added check in
   `process_ubatch()` to invalidate `gf_res_prev` when `hidden_gpu_n_seqs` changes
   (0→1 transition from prefill to generation). Without this, the graph is cached
   from prefill (no hidden_gpu copy) and reused for generation.

3. **`llama_dflash_allocate_slots` unconditional** — Changed from `dflash_slots_cap > 1`
   to `dflash_slots_cap > 0` so `hidden_gpu` is allocated even for single-slot,
   enabling the GPU-embedded path.

### Results

- The graph builder DOES execute the hidden_gpu copy code (confirmed with
  DFLASH_LAYER_OUT diagnostic: 448 lines with `hidden_n_seqs=1` out of 1344 total)
- With `LLAMA_GRAPH_REUSE_DISABLE=1`: acceptance improved to 62.5% (from 43.8%)
- Without graph reuse disabled: output STILL garbled (43.8% acceptance, same as before)
- The graph is cached from prefill (`hidden_n_seqs=0`) and `can_reuse` does NOT
  detect the change on subsequent decodes

### Remaining issue

The `can_reuse` check at line 6072 compares `gparams.cparams.hidden_gpu_n_seqs` with
the previous `gparams.cparams.hidden_gpu_n_seqs`. On the second decode, `graph_params`
copies `cparams.hidden_gpu_n_seqs` which should be 1 (set during the first decode's
DFlash setup). But the DFLASH_UBATCH_ENTRY diagnostic showed the value alternates
between 0 and 1, suggesting `hidden_gpu_n_seqs` is reset to 0 between calls.

The root cause is likely that `process_ubatch()` resets `cparams.hidden_gpu_n_seqs = 0`
at line 6660 (start of the DFlash setup), but `graph_params` at line 6067 runs BEFORE
this reset, copying the value from the PREVIOUS call's end state. The alternation
suggests some calls don't set `hidden_gpu_n_seqs = 1` (e.g., prefill calls with
`dflash_graph_hidden_ready = false`).

### Graphify comparison (HF-038 supplement)

Built knowledge graphs for both fork and merge using graphify:
- Fork: 40495 nodes, 107782 edges, 972 communities
- Merge: 42404 nodes, 111103 edges, 1069 communities
- DFlash nodes: 698 (fork) vs 750 (merge), 643 common
- Fork-only: 55 nodes (renamed classes, missing helpers like apply_dflash_effective_defaults)
- Merge-only: 107 nodes (new conversion, profile functions, renamed DFlash classes)
- The missing functions (apply_dflash_effective_defaults, reset_dflash_only_args) have
  equivalent defaults in the merge's struct definition, so they're not the root cause.

## 2026-07-09 09:50 UTC - Merge DFlash regression plan update

### FACTS
- Installed the Superpowers skill repository from `https://github.com/obra/superpowers` into `~/.pi/agent/skills/superpowers`.
- Existing progress file `TASK_PROGRESS.md` was found and read for context.
- `TASK_PROGRESS.md` contains prior merge/DFlash investigation sections, including HF-035 through HF-040, with prior notes on single-slot DFlash, multi-slot DFlash, hidden GPU capture, `n_outputs_max`, graph invalidation, and quality regressions.
- Updated `docs/superpowers/plans/2026-07-09-fix-merge-llama-dflash-regression.md` to replace bisect-first investigation with a three-way semantic diff/blame workflow.
- Updated the plan to require future agents to write progress to `TASK_PROGRESS.md` after each meaningful investigation or code step.
- Updated the plan to require future reports and progress entries to separate `FACTS`, `HYPOTHESES`, `TEST RESULTS`, and `NEXT STEPS`.

### HYPOTHESES
- A three-way semantic diff/blame workflow is likely more appropriate than git bisect for this integration branch because the branch contains upstream merge resolution, WIP/debug/fixup commits, and possible non-buildable intermediate states.

### TEST RESULTS
- Command: `find . -iname '*task*progress*.md' -o -iname 'TASK_PROGRESS.md' -o -path './docs/superpowers/*TASK_PROGRESS*' | sort | head -100`
- Result: Found `./TASK_PROGRESS.md`.
- Command: read/summarize `TASK_PROGRESS.md` for DFlash and merge context.
- Result: Existing DFlash/merge progress context was present and used to update the plan.
- Output files: `docs/superpowers/plans/2026-07-09-fix-merge-llama-dflash-regression.md`

### NEXT STEPS
- Execute the plan from Task 1 using the three-way diff/blame workflow.
- Keep updating `TASK_PROGRESS.md` after each task with facts separated from hypotheses.

## 2026-07-09 12:02 UTC - Task 1 comparison harness execution

### FACTS
- Current branch after returning from baseline test: `merge_llama_into_beellama_2`.
- Created `scripts/dflash-regression/extract_cli_output.py`.
- Created `scripts/dflash-regression/qwen36_compare.sh`.
- Made both scripts executable.
- `origin/main` comparison ran at commit `130ea2480bee8268907a102bd814e276f4e88bdf`.
- `merge_llama_into_beellama_2` comparison ran at commit `0fe0bf5c66b3b63939c9237607292edeef346b91`.
- Graphify query over DFlash/Qwen35 concepts identified connected nodes including `common_speculative_impl_dflash`, `dflash_capture_data`, `llm_graph_params`, `build_inp_out_ids`, `llama_context`, and DFlash/Qwen model builders.

### HYPOTHESES
- The comparison harness's current byte-identical pass criterion is too strict because working `origin/main` DFlash can produce coherent but slightly different text from non-DFlash under the current CLI/sampler settings.
- The merge branch regression remains real because its DFlash output is visibly corrupted/noisy and much slower, while its non-DFlash output remains coherent.

### TEST RESULTS
- Command: `git switch --detach origin/main && cmake --build build_vulkan --target llama-cli -j "$(nproc)" && scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-origin-main`
- Result: `origin/main` non-DFlash rc=0, prompt `85.6 t/s`, generation `11.0 t/s`; DFlash rc=0, prompt `120.9 t/s`, generation `20.9 t/s`; harness reported `result=fail-output-mismatch`, but the DFlash output was coherent and differed only slightly in wording from non-DFlash in the visible excerpt.
- Output files: `/tmp/qwen36-origin-main/summary.txt`, `/tmp/qwen36-origin-main/nodflash.out`, `/tmp/qwen36-origin-main/dflash.out`
- Command: `git switch merge_llama_into_beellama_2 && cmake --build build_vulkan --target llama-cli -j "$(nproc)" && scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-merge-branch`
- Result: merge branch non-DFlash rc=0, prompt `83.2 t/s`, generation `11.0 t/s`; DFlash rc=0, prompt `84.2 t/s`, generation `4.4 t/s`; DFlash visible output becomes corrupted/noisy after the first few tokens; harness reported `result=fail-output-mismatch`.
- Output files: `/tmp/qwen36-merge-branch/summary.txt`, `/tmp/qwen36-merge-branch/nodflash.out`, `/tmp/qwen36-merge-branch/dflash.out`
- Command: `graphify_query` for DFlash hidden_gpu capture, Qwen35 graph building, dflash_draft outputs, and common_speculative verifier/state paths.
- Result: Graphify found connected code concepts and files for follow-up inspection.

### NEXT STEPS
- Adjust interpretation of Task 1: byte-identical output is not required for a passing working baseline; use coherence plus throughput and visible corruption comparison for now.
- Execute Task 2 three-way ref capture and diff generation.

## 2026-07-09 12:08 UTC - Task 2 three-way refs and diff artifacts

### FACTS
- Local branch `merge_llama_into_beellama_2` contains local commit `0fe0bf5c6` for the investigation plan.
- Three-way diff helper compares `origin/merge_llama_into_beellama_2`, not the local plan commit, so analysis targets the integration result `d5c80cb99eb7a13bd726921f22dc5c224f03bec9`.
- `origin/main` commit: `130ea2480bee8268907a102bd814e276f4e88bdf`.
- `origin/merge_llama_into_beellama_2` commit: `d5c80cb99eb7a13bd726921f22dc5c224f03bec9`.
- `ggml/master` commit after fetch: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`.
- `merge-base(origin/main, ggml/master)`: `6ddc9430b145f61a0c1733b9d79c99c0ebdedf50`.
- `merge-base(ggml/master, origin/merge_llama_into_beellama_2)`: `f708a5b2caaee0226c0af220e366785699ba41e2`.
- `merge-base(origin/main, origin/merge_llama_into_beellama_2)`: `130ea2480bee8268907a102bd814e276f4e88bdf`.
- Fork delta over suspect files: 7 files changed, 13698 insertions, 2256 deletions.
- Upstream delta over suspect files: 6 files changed, 2412 insertions, 525 deletions.
- Merge-result delta over suspect files: 7 files changed, 14557 insertions, 2493 deletions.
- Generated three-way artifacts under `/tmp/qwen36-three-way/` including refs, stats, full diffs, focused DFlash/Qwen snippets, and blame files.

### HYPOTHESES
- None yet.

### TEST RESULTS
- Command: `git fetch origin --prune && git fetch https://github.com/ggml-org/llama.cpp.git master:refs/remotes/ggml/master --no-tags`
- Result: refs fetched successfully.
- Command: `scripts/dflash-regression/three_way_dflash_diff.sh /tmp/qwen36-three-way`
- Result: generated refs, stat, diff, focus, and blame artifacts.
- Output files: `/tmp/qwen36-three-way/refs-full.txt`, `/tmp/qwen36-three-way/refs.txt`, `/tmp/qwen36-three-way/fork-delta.diff`, `/tmp/qwen36-three-way/upstream-delta.diff`, `/tmp/qwen36-three-way/merge-result.diff`, `/tmp/qwen36-three-way/*.focus.txt`, `/tmp/qwen36-three-way/blame-*.txt`

### NEXT STEPS
- Perform Task 3 semantic diff/blame analysis, starting with Qwen35 hidden capture and output semantics.

## 2026-07-09 12:18 UTC - Task 3 semantic diff finding: Qwen35 NextN main-loop boundary

### FACTS
- `origin/main:src/models/qwen35.cpp` lines 168-170 define `const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;` and loop `for (int il = 0; il < n_transformer_layers; ++il)`.
- `origin/main:src/models/qwen35.cpp` line 187 uses `if (il == n_transformer_layers - 1 && inp_out_ids && !need_full_h_nextn)` for output row selection.
- `origin/merge_llama_into_beellama_2:src/models/qwen35.cpp` lines 158-160 keep the comment `MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass`, but loop `for (int il = 0; il < n_layer; ++il)` and set `res->t_layer_inp[il] = inpL`.
- `origin/main:src/models/qwen35moe.cpp` lines 191-193 define `const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;` and loop `for (int il = 0; il < n_transformer_layers; ++il)`.
- `origin/main:src/models/qwen35moe.cpp` line 210 uses `if (il == n_transformer_layers - 1 && inp_out_ids && !need_full_h_nextn)` for output row selection.
- `origin/merge_llama_into_beellama_2:src/models/qwen35moe.cpp` lines 180-182 keep the comment `MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass`, but loop `for (int il = 0; il < n_layer; ++il)` and set `res->t_layer_inp[il] = inpL`.
- Blame for the merge-result `for (int il = 0; il < n_layer; ++il)` lines points to `7acb4e8cd2ce21f457d1298e75fad729520d263c hparams : refactor hparams.n_layer (#24060)`.
- Blame for the merge-result `res->t_layer_inp[il] = inpL` lines points to `b14e3fb90ca8c760f4254ddc9aa7845ebbdb2edf spec: support eagle3 for qwen3.5 & 3.6 (#24593)`.

### HYPOTHESES
- HYPOTHESIS 1: The merge branch corrupts DFlash for Qwen3.6 because the target Qwen35/Qwen35MoE main pass now executes NextN/MTP layers that `origin/main` deliberately excluded, so DFlash hidden capture and/or verifier target logits are taken from the wrong graph boundary.
- HYPOTHESIS 2: The `res->t_layer_inp[il]` Eagle3 support may need to be preserved for all loaded layers, but the main transformer compute loop should still stop at `n_layer - nextn_predict_layers` for Qwen35 DFlash target correctness.

### TEST RESULTS
- Command: targeted `git show` extraction for `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` in `origin/main` and `origin/merge_llama_into_beellama_2`.
- Result: Confirmed loop-boundary difference and output-selection condition difference between the working fork baseline and merge result.
- Command: `git blame -L 156,164 origin/merge_llama_into_beellama_2 -- src/models/qwen35.cpp` and `git blame -L 178,186 origin/merge_llama_into_beellama_2 -- src/models/qwen35moe.cpp`.
- Result: Identified upstream commits responsible for the merge-result loop and Eagle3 layer-input lines.
- Output files: `/tmp/qwen36-three-way/merge-result.diff`, `/tmp/qwen36-three-way/fork-delta.diff`, `/tmp/qwen36-three-way/upstream-delta.diff`

### NEXT STEPS
- Apply a minimal local experiment restoring the `n_transformer_layers` loop bound and last-transformer output condition in `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp`, while preserving `res->t_layer_inp[il] = inpL` inside the loop for layers actually executed.
- Build `llama-cli` and rerun `/tmp/qwen36-merge-branch` comparison to test whether DFlash output coherence and generation throughput recover.

## 2026-07-09 12:32 UTC - Failed experiment: Qwen35 row-selection condition

### FACTS
- The earlier loop-bound hypothesis was refined: in the merge branch, `n_layer` appears to be the main-layer count and `n_layer_all` includes NextN/MTP layers, so changing the loop bound was not applied.
- A minimal local experiment changed `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` to restore the origin-equivalent `need_full_h_nextn` row-selection condition at the final main layer.
- The experiment was reverted after testing because it did not fix the DFlash corruption or throughput regression.
- Failed experiment diff was saved to `/tmp/qwen36-failed-row-select-fix.diff`.

### HYPOTHESES
- HYPOTHESIS 1 was not supported by this experiment: final-main-layer row selection alone is not the root cause of the merge branch DFlash corruption.
- The root cause is more likely in DFlash-specific hidden capture, drafter output sizing, verifier acceptance/state handling, or an interaction not changed by the row-selection experiment.

### TEST RESULTS
- Command: `cmake --build build_vulkan --target llama-cli -j "$(nproc)" && scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-after-row-select-fix`
- Result: build succeeded; non-DFlash rc=0, prompt `83.2 t/s`, generation `11.0 t/s`; DFlash rc=0, prompt `86.1 t/s`, generation `3.8 t/s`; DFlash output remained corrupted/noisy.
- Output files: `/tmp/qwen36-after-row-select-fix/summary.txt`, `/tmp/qwen36-after-row-select-fix/nodflash.out`, `/tmp/qwen36-after-row-select-fix/dflash.out`, `/tmp/qwen36-failed-row-select-fix.diff`

### NEXT STEPS
- Continue Task 3 semantic analysis, focusing next on `src/models/dflash_draft.cpp` output sizing and `common/speculative.cpp` DFlash verifier/state paths.

## 2026-07-09 12:45 UTC - Additional isolation tests: GPU ring off and draft horizon 1

### FACTS
- `GGML_DFLASH_GPU_RING=0` did not restore coherent DFlash output on the merge branch.
- `--spec-draft-n-max 1` still produced corrupted/truncated DFlash output on the merge branch.
- The row-selection experiment remained reverted; current source files `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` are restored to pre-experiment state.

### HYPOTHESES
- The failing boundary is not solely the Vulkan GPU cross-ring path, because disabling `GGML_DFLASH_GPU_RING` did not fix output.
- The failing boundary occurs at or before the first drafted token verification/acceptance path, because `--spec-draft-n-max 1` still produced bad output.
- Candidate boundaries now rank higher: drafter logits/argmax row mapping, target verifier acceptance logic, target/draft KV alignment, or DFlash hidden/cross data passed into the drafter independent of the GPU ring setting.

### TEST RESULTS
- Command: `GGML_DFLASH_GPU_RING=0 scripts/dflash-regression/qwen36_compare.sh /tmp/qwen36-merge-gpu-ring-off`
- Result: non-DFlash rc=0, prompt `82.6 t/s`, generation `11.0 t/s`; DFlash rc=0, prompt `143.9 t/s`, generation `4.3 t/s`; DFlash output remained corrupted/noisy.
- Output files: `/tmp/qwen36-merge-gpu-ring-off/summary.txt`, `/tmp/qwen36-merge-gpu-ring-off/dflash.out`
- Command: manual `llama-cli` DFlash run with `--spec-draft-n-max 1`, seed 7, temp 0, `Vulkan1`, 192 generated-token cap.
- Result: rc=0; output starts `[Start thinking]\n3.` and stops; visible generation throughput `6.8 t/s`.
- Output files: `/tmp/qwen36-merge-dflash-nmax1/out.txt`, `/tmp/qwen36-merge-dflash-nmax1/err.txt`

### NEXT STEPS
- Inspect DFlash verifier acceptance logic and token trace support.
- Run merge branch with DFlash token/KV/ring traces if available to determine whether the first draft token is invalid but accepted, or whether the target verifier itself produces the bad token.

## 2026-07-09 - DFlash merge regression: verifier/rollback boundary

### FACTS
- Added temporary QA traces around flat DFlash draft output, verifier sampling, and server accept bookkeeping.
- The active path is flat DFlash (`common_speculative_draft`), with GPU argmax rows available.
- `merge_llama_into_beellama_2` ignored `--spec-draft-n-max` in `server_slot::get_n_draft_max()`, causing the first DFlash cycle to draft/verify 15 tokens despite `--spec-draft-n-max 3`.
- Capping by the configured draft horizon changed first-cycle `draft=15` to `draft=3` and improved short-prompt output/perf, but did not fully fix full-prompt corruption.
- With `--spec-draft-n-max 3`, token traces show DFlash and non-DFlash match through token index 12, then diverge at token index 13 (`non-DFlash=2570`, `DFlash=279`).
- The divergence occurs after several fully accepted multi-token verification batches, not at the initial prompt prefill or first draft token.
- Testing `GGML_DFLASH_DISABLE_FGDN_CH=1` did not move the first divergence.
- Testing an env-gated `llama_tape_replay_sync(ctx_tgt)` before building the DFlash verifier batch did not move the first divergence.
- A separate `/tmp/beellama-origin-main` worktree built from `origin/main` produced coherent DFlash output on the same prompt/flags, with DFlash generation about `28.5 t/s` versus non-DFlash about `11.2 t/s`.
- `origin/main` DFlash server code contains recurrent backup + `llama_dflash_rollback()` handling around DFlash verification. The merge branch's simpler path only uses `common_context_seq_rm()` after verification.
- `llama_dflash_rollback()` exists on the merge branch and restores recurrent state from a backup sequence, keeps accepted attention KV, and replays accepted token recurrent state.

### HYPOTHESES
- Confirmed partial cause: the merge branch fails to honor `--spec-draft-n-max` for DFlash because `get_n_draft_max()` used only context/remaining budget and DFlash top-level `n_max`, while the CLI flag populates `speculative.draft.n_max`.
- Current leading hypothesis: the remaining corruption is due to missing DFlash recurrent backup/rollback around multi-token verification. Accepted tokens match initially, but the recurrent state left by the verifier batch diverges from sequential decode; `origin/main` avoids this with `llama_dflash_rollback()`.
- Rejected/low-priority hypotheses: Vulkan GPU ring corruption (already reproduced with `GGML_DFLASH_GPU_RING=0`), fused GDN chunk kernel as sole cause, and tape sync alone.

### TEST RESULTS
- `/tmp/qwen36-after-draft-cap-fix`: DFlash `rc=0`, improved to about `12.1 t/s` but still repeated/corrupted on full prompt.
- `/tmp/qwen36-after-cap-nmax1`: DFlash `rc=0`, coherent prefix, about `9.1 t/s`, indicating lower horizon avoids most visible corruption but is slower.
- `/tmp/qwen36-first-divergence`: first token divergence at generated token index 13 after accepted DFlash verification cycles.
- `/tmp/qwen36-force-ckpt-divergence`: env-gated full checkpoint rollback experiment did not fix divergence and worsened output; this path is not compatible as a simple substitute for DFlash rollback.
- `/tmp/qwen36-no-fgdn-ch-divergence`: disabling fused chunk GDN did not change first divergence.
- `/tmp/qwen36-tape-sync-divergence`: adding tape replay sync before verification batch did not change first divergence.
- `/tmp/qwen36-origin-main-first-divergence`: origin/main DFlash coherent and fast on same prompt/flags.

### NEXT STEPS
- Test a minimal DFlash recurrent backup + `llama_dflash_rollback()` transplant in the merge branch, preferably env-gated first.
- Keep the DFlash draft horizon cap as a candidate fix, but clean it up after the rollback root cause is confirmed.
- Remove temporary QA/sample trace instrumentation before finalizing any patch.
- Re-run `scripts/dflash-regression/qwen36_compare.sh` and local DFlash tests after any candidate rollback fix.

## 2026-07-09 - DFlash merge regression: rollback test refined

### FACTS
- Added an env-gated minimal recurrent backup + `llama_dflash_rollback()` test path (`GGML_DFLASH_TEST_ROLLBACK=1`) in the merge branch.
- Calling `llama_dflash_rollback()` on every DFlash verification batch was incorrect: origin/main skips rollback for all-accepted flat batches.
- After changing the test to call rollback only for partial flat accepts, the first divergence still occurred at generated token index 13.
- `llama_dflash_rollback()` returned `n_reeval > 0` on Vulkan, and the diagnostic re-decode path returned `ret=0`, but the sampled token at the first partial verify was already different from non-DFlash before rollback could help.
- The first bad DFlash token is produced by target verifier row 0 of a multi-token DFlash verification batch, not by post-verify rollback alone.
- Comparing Qwen35 diffs showed that `origin/main` has DFlash-specific tape capture and tree-aware recurrent/conv paths (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree` usage) that are absent from the merge branch's Qwen35 graph.

### HYPOTHESES
- Refined leading hypothesis: the merge branch target verifier is computing non-sequential-equivalent logits for multi-token DFlash verification batches on Qwen3.6 recurrent layers. Future draft tokens in the same verification batch contaminate row-0/early-row recurrent computation unless the DFlash tree-aware/tape-aware Qwen35 paths from `origin/main` are present.
- Rollback remains necessary for partial acceptance on Vulkan, but it is not sufficient: the verifier logits themselves must be fixed first.
- The draft horizon cap remains a valid partial fix because it prevents uncontrolled 15-token verification batches, but it does not address verifier row correctness.

### TEST RESULTS
- `/tmp/qwen36-dflash-rollback-divergence`: rolling back every batch caused early EOS after token index 5; not a valid fix because all-accepted flat batches should not rollback.
- `/tmp/qwen36-dflash-rollback-reeval-divergence`: adding re-decode after rollback still diverged early when rollback was applied to all batches.
- `/tmp/qwen36-dflash-rollback-partial-divergence`: rollback only on partial accepts preserved the original first divergence at token index 13; DFlash still corrupted.

### NEXT STEPS
- Test restoring the DFlash-specific Qwen35 verifier graph pieces from `origin/main`, especially tree-aware SSM/GDN paths and tape capture.
- Consider setting linear parent IDs for flat DFlash verification only after the Qwen35/delta-net graph can consume `tree_parent_ids` again.
- Clean up env-gated rollback experiments and temporary tracing once the graph-level verifier fix is proven.

## 2026-07-09 19:30 UTC - Task 3 three-way Qwen35 graph semantic diff (tree-aware pieces)

### FACTS
- Reference remotes in this worktree: `gboddaer/main` = `130ea2480` (the plan's "origin/main" BeeLlama baseline with working DFlash); `gboddaer/merge_llama_into_beellama_2` = `c5c17182e` (the regressed PR merge branch). `origin` here = Anbeeld (`85e22ea0b`); `ggml-org/master` = `b820cc8e6`. Per user clarification, all comparisons use `gboddaer/main` as base, NOT ggml-org.
- qwen35.cpp line counts: `gboddaer/main` = 836, merge branch = 683 (merge lost ~153 lines).
- qwen35moe.cpp line counts: `gboddaer/main` = 919, merge branch = 741.
- delta-net-base.cpp line counts: `gboddaer/main` = 685, merge branch = 639.
- Tree-aware op USAGE in graph builders (`git grep -c`):
  - `gboddaer/main` qwen35.cpp: `ggml_ssm_conv_tree`=1; qwen35moe.cpp: `ggml_ssm_conv_tree`=1; delta-net-base.cpp: `ggml_gated_delta_net_tree`=1.
  - merge branch: all three = 0 (tree-aware dispatch absent from graph builders).
  - `ggml-org/master` (upstream): all = 0 (tree-aware ops are fork-specific, not upstream).
- Tree-aware op BACKEND implementations (`GGML_OP_SSM_CONV_TREE` / `GGML_OP_GATED_DELTA_NET_TREE`): present in CPU (`ggml-cpu/ops.cpp`) and CUDA (`ggml-cuda/ssm-conv.cu`, `gated_delta_net.cuh`) on BOTH branches. ABSENT from Vulkan (`ggml-vulkan/`) on BOTH branches (git grep returned empty for both `gboddaer/main` and merge branch).
- `tree_parent_ids` plumbing already EXISTS on the merge branch: `llama_set_tree_parent_ids` API (include/llama.h:1708), `llama_context::set_tree_parent_ids` (src/llama-context.cpp:5329), `tree_bufs.parent_ids_gpu` (5459), graph param `tree_parent_ids` (src/llama-graph.h:738,1009), `tree_ssm_intermediates` + `tree_n_recurrent_layers` (llama-graph.h:739,740). So only the graph-builder CONSUMPTION is missing.
- qwen35.cpp main-loop hidden_gpu capture: merge branch HAS it (ported in HF-040), and it matches `gboddaer/main` — both deliberately drop `ggml_cont` (gboddaer/main has "[FIX-TRY] drop ggml_cont" comment; merge matches). NOT a difference.
- qwen35.cpp main-loop `prefill_gpu` capture block: present in `gboddaer/main` (lines ~245-298), ABSENT in merge branch. This block is gated on `cparams.prefill_gpu_n_seqs > 0 && cparams.dflash_prefill_capture_active && cparams.dflash_prefill_n_tokens > 0`.
- qwen35moe.cpp main loop: merge branch has NEITHER hidden_gpu NOR prefill_gpu capture (both absent). `gboddaer/main` has both. (qwen35.cpp got hidden_gpu in HF-040; qwen35moe.cpp was not updated.)
- Loop bound is EQUIVALENT: merge `for (il < n_layer)` where `n_layer` = main-layer count (merge's `hparams.n_layer()` refactor; confirmed by merge qwen35.cpp lines 122/125 splitting main vs `n_layer_all` NextN). `gboddaer/main` `for (il < n_transformer_layers)` where `n_transformer_layers = n_layer - nextn_predict_layers` = same main-layer count. No loop-bound change needed.
- `build_delta_net_fused` K argument: merge calls `ggml_gated_delta_net(..., /*K=*/1)`; `gboddaer/main` calls `ggml_gated_delta_net(...)` (no K). BUT the op derives `K_actual = state_is_4d ? 1 : state->ne[1]` in BOTH, and the recurrent state passed in is 4D, so `K_actual=1` in both. Result tensor shape `[S_v*H, n_tokens*n_seqs + 1*S_v*n_seqs, 1, 1]` is identical. NOT a functional divergence (merge's explicit K=1 matches the derived K=1; the extra `GGML_ASSERT(K_actual <= K)` and `ggml_set_op_params_i32(K)` are merge-only but consistent).
- `build_conv_state` signature: `gboddaer/main` has trailing `bool qkv_mixed_transposed = false` param + `if (!qkv_mixed_transposed) transpose` guard; merge lacks the param and always transposes. Functionally equivalent for current callers (one transpose either way), but the signature differs.
- `build_delta_net` (delta-net-base.cpp): `gboddaer/main` has a tree branch at the top (`if (tree_parent_ids && n_seq_tokens > 1 && n_seqs_in == 1 && tree_ssm_intermediates && n_seq_tokens <= nelements(tree_parent_ids))` -> `ggml_gated_delta_net_tree`); merge lacks it. Only taken when `tree_parent_ids` is non-null.
- Environment changed since plan was written: `/fast` does not exist (models at `/crypt/models/`); no W7800/Vulkan1 device present (only GPU0 = integrated AMD Radeon RADV GFX1151 + llvmpipe); fork reference `build-vulkan` has `llama-server` but no `llama-cli`.

### HYPOTHESES
- HYPOTHESIS (DISPROVEN for Vulkan): porting the tree-aware SSM/GDN dispatch (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`) will fix the merge-branch Vulkan DFlash corruption. DISPROVEN BY FACT: Vulkan implements neither op on either branch, so on Vulkan `tree_parent_ids` must be null for flat DFlash verification (else the op would be unsupported) and both branches take the identical non-tree path. The tree port is correct for CUDA/tree-verify completeness but is inert on Vulkan.
- HYPOTHESIS 1 (current, Vulkan-relevant): the merge-branch Vulkan DFlash corruption is caused by the missing `prefill_gpu` capture block in qwen35.cpp (+ qwen35moe.cpp) and the entirely-missing hidden_gpu capture in qwen35moe.cpp. Without prefill_gpu capture, the prompt-suffix hidden states are not copied to the prefill GPU buffers, so the drafter's cross-attention context is wrong/stale -> garbled drafts after the first accepted batch. This is the only qwen35/qwen35moe graph-builder difference that is actually exercised on Vulkan.
- HYPOTHESIS 2 (lower priority): the corruption is NOT in the Qwen35 graph builders at all (they are functionally identical to `gboddaer/main` on Vulkan) but in server-side verifier-batch construction / recurrent-state rollback. Supported by the earlier finding that the first bad token comes from verifier row 0 of a multi-token batch and that `llama_dflash_rollback()` is necessary-but-not-sufficient. If HYPOTHESIS 1's port does not fix Vulkan, this becomes the lead.

### TEST RESULTS
- Command: `git grep -c 'ggml_ssm_conv_tree|ggml_gated_delta_net_tree' <ref> -- <files>` across `gboddaer/main`, merge branch, `ggml-org/master`.
- Result: confirmed tree-aware usage present only on `gboddaer/main` graph builders; absent on merge and upstream. Backend impls present in CPU+CUDA on both fork branches; absent in Vulkan on both.
- Command: `git show gboddaer/main:src/models/qwen35.cpp` vs `git show gboddaer/merge_llama_into_beellama_2:src/models/qwen35.cpp` (and qwen35moe.cpp, delta-net-base.cpp) side-by-side.
- Result: identified the 4 missing graph-builder pieces (tree conv dispatch, tree GDN dispatch, prefill_gpu capture, qwen35moe hidden_gpu capture) + 1 signature difference (build_conv_state transpose param). Confirmed loop bound, K argument, and hidden_gpu capture (qwen35.cpp) are NOT functional divergences.
- Output files: `/tmp/gb-main-qwen35.cpp`, `/tmp/gb-merge-qwen35.cpp`, `/tmp/gb-main-qwen35moe.cpp`, `/tmp/gb-merge-qwen35moe.cpp`, `/tmp/gb-main-deltanet.cpp`, `/tmp/gb-merge-deltanet.cpp`.

### NEXT STEPS
- Apply the "restore missing fork DFlash graph integration" port (plan Task 5 Strategy A): port from `gboddaer/main` into merge branch — (a) tree_mode conv dispatch in qwen35.cpp/qwen35moe.cpp `build_layer_attn_linear`, (b) tree GDN dispatch in delta-net-base.cpp `build_delta_net`, (c) `build_conv_state` `qkv_mixed_transposed` param, (d) prefill_gpu capture in qwen35.cpp/qwen35moe.cpp main loop, (e) hidden_gpu capture in qwen35moe.cpp main loop. The tree pieces (a,b,c) are inert on Vulkan; (d,e) are the Vulkan-relevant restoration under HYPOTHESIS 1.
- Build `llama-cli` (Vulkan) in the worktree (`build_vulkan`).
- Run DFlash vs non-DFlash comparison on the iGPU (GPU0) with `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` + `Qwen3.6-27B-DFlash-Q4_K_M.gguf` (paths adjusted from the plan since `/fast` is absent and no Vulkan1/W7800).
- If Vulkan output is still garbled after the port, HYPOTHESIS 1 is disproven for the graph-builder path -> pivot to HYPOTHESIS 2 (server-side verifier/rollback), per plan Task 5 Step 5 (save failed diff, revert, return to Task 3).

## 2026-07-09 18:49 UTC - Task 3 (continued): code-verified Qwen35 tree-aware verifier graph comparison (gboddaer/main as reference)

### FACTS
- Comparison refs (worktree): `gboddaer/main` = `130ea2480`, merge branch HEAD = `c5c17182e` (`merge_llama_into_beellama_2`). Per user instruction, `gboddaer/main` is the reference and code is trusted over comments in the prior 19:30 entry.
- Line counts (main vs merge): qwen35.cpp 836 vs 683; qwen35moe.cpp 919 vs 741; delta-net-base.cpp 685 vs 639.
- Tree-aware op usage in graph builders (`git grep -c`): `gboddaer/main` has `ggml_ssm_conv_tree`/`ggml_gated_delta_net_tree`/`tree_parent_ids` in all 3 files (3 matches each); merge branch HEAD has 0 in all 3. Re-confirmed.
- qwen35.cpp hidden_gpu capture block: PRESENT in merge (lines 205-240, ported in HF-040), structurally matches main (lines 222-253). Merge derives `dflash_capture_n_seqs_dfl`/`dflash_capture_n_tokens_dfl` from `ubatch.n_seqs_unq`; main uses `dflash_capture_n_seqs`/`dflash_capture_n_tokens`. NOT a functional divergence for the common n_seqs_unq==1 case. Both drop `ggml_cont` (main has `[FIX-TRY] drop ggml_cont` comment; merge matches).
- qwen35.cpp prefill_gpu capture block: PRESENT in main (lines 259-296, gated `cparams.prefill_gpu_n_seqs > 0 && cparams.dflash_prefill_capture_active && cparams.dflash_prefill_n_tokens > 0`), ABSENT in merge. Confirmed by empty grep for `prefill_gpu`/`pgpu` in merge qwen35.cpp.
- qwen35.cpp tape_gpu capture block: PRESENT in main (lines 583+, gated `cparams.tape_gpu_n_seqs > 0`, copies k_conv/v_conv/gate/beta_presigmoid/qkv_mixed_pretranspose into `tgpu->layers[li].{k,v,gate,beta,qkv}`), ABSENT in merge. Confirmed by empty grep for `tape_gpu`/`tgpu`/`tl.qkv` in merge qwen35.cpp. **This block was NOT listed in the 19:30 NEXT STEPS (a)-(e) — it is a newly discovered missing piece found by reading the code rather than the prior comments.**
- qwen35moe.cpp capture blocks: ALL THREE ABSENT in merge — hidden_gpu (main line 237), prefill_gpu (main line 268), tape_gpu (main line 593). Confirmed by empty grep for `dflash_capture|cparams.(hidden_gpu|prefill_gpu|tape_gpu)|hgpu|pgpu|tgpu` in merge qwen35moe.cpp. The 19:30 entry only listed hidden_gpu + prefill_gpu for qwen35moe; tape_gpu was missed there too.
- qwen35.cpp + qwen35moe.cpp tree_mode conv dispatch: PRESENT in main (qwen35 lines 513-528: `tree_mode = (tree_parent_ids != nullptr && n_seq_tokens > 1 && n_seqs == 1 && n_seq_tokens <= nelements(tree_parent_ids))` -> `ggml_ssm_conv_tree` else `ggml_ssm_conv`; silu only in non-tree branch; qwen35moe lines 523-536 analogous), ABSENT in merge (always `ggml_ssm_conv` + `ggml_silu`).
- delta-net-base.cpp `build_delta_net` tree branch: PRESENT in main (lines 440-457: `if (tree_parent_ids && n_seq_tokens > 1 && n_seqs_in == 1 && tree_ssm_intermediates && n_seq_tokens <= nelements(tree_parent_ids))` -> `ggml_gated_delta_net_tree(ctx0, q, k, v, g, b, s, tree_parent_ids, persist_inter)` with `recurrent_idx < tree_n_recurrent_layers` assert), ABSENT in merge. Merge only has the flat `ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1)` at line 403 and the fused-path call at line 577.
- build_conv_state signature: main has trailing `bool qkv_mixed_transposed` param + `if (!qkv_mixed_transposed) qkv_mixed = ggml_transpose(...)` guard (delta-net-base.cpp:490-514); merge lacks the param and always transposes internally (delta-net-base.cpp:450-471). VERIFIED EQUIVALENT: in main qwen35.cpp the transposed `qkv_mixed` (line 504) is consumed ONLY by `build_conv_state(..., true)` (line 507); all downstream use the saved `qkv_mixed_pretranspose` (line 502, used at line 608). Merge transposes a local copy inside build_conv_state and leaves the caller `qkv_mixed` un-transposed (no downstream use after). Net: qkv_mixed is transposed exactly once before concat in both. NOT a functional divergence.
- build_delta_net_fused K argument: merge passes `K=1`/`K` explicitly (lines 403, 577); main omits K (lines 403, 614). Both derive `K_actual = state_is_4d ? 1 : state->ne[1]`; the recurrent state passed is 4D so `K_actual=1` in both. Re-confirmed NOT a functional divergence (matches 19:30 finding).
- tape_gpu INFRASTRUCTURE is PRESENT and near-identical on merge: `dflash_tape_gpu_layer` struct (llama-context.h:188-197, with `qkv` tensor `[conv_channels, max_tokens]`), allocation `tl.qkv = ggml_new_tensor_2d(...)` at llama-context.cpp:2268 inside `allocate_tape_gpu` (called at 2087, 8911), cparams fields (llama-cparams.h:104-105), `tape_gpu_n_seqs = n_tapes` set at llama-context.cpp:2106 and `= cparams.tape_gpu ? 1 : 0` at 2672, graph-invalidtion tracking at 6653-7041. Diff vs main is only ~5 line offset.
- tape_gpu CONSUME sites are PRESENT on merge and READ `gpu_layer->qkv` directly: llama-context.cpp:3450 (`ggml_tensor * qkv_tensor = gpu_tape->layers[li].qkv`), 3800 (`get_tensor_data(gpu_layer->qkv, d.qkv_mixed.data(), 0, ...)` inside `tape_replay_conv`), 5671 (second replay routine). Gating: `const bool use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv;` (line 3756/5648). When `use_gpu_qkv==true`, line 3800 reads `gpu_layer->qkv` — the tensor that ONLY the missing graph-builder block would have written.
- ALTERNATIVE tape population path EXISTS on merge: CPU readback `dflash_read_tensor(t, tape.qkv_mixed, n_elem)` at llama-context.cpp:1773, fed by `tape_name_map` entries (2060-2071) including `"linear_attn_qkv_mixed-"+il` (the qwen35.cpp `cb(qkv_mixed, "linear_attn_qkv_mixed", il)` tensor at merge qwen35.cpp:277). A CPU->GPU upload also exists at llama-context.cpp:3637/3652 (`ggml_backend_tensor_set(upload.qkv, tape.qkv_mixed.data()+qkv_seq_offset, ...)`). So whether the missing GPU graph-builder block actually corrupts output depends on whether `use_gpu_qkv==true` (GPU-direct read of unwritten `gpu_layer->qkv`) versus the CPU-readback fallback being used.
- Existing diagnostic already present on merge: `[dflash-tr-conv] layer=%d OK use_gpu_qkv=%d qkv_mixed.size=%zu n_tokens=%d n_accepted=%d` (llama-context.cpp:3770), gated by `GGML_DFLASH_DEBUG=1` (`dflash_diagnostic_debug_enabled()`, defined llama-context.cpp:1346-1352, env var `GGML_DFLASH_DEBUG`). This log directly reveals whether the GPU tape path is active at runtime.
- Test-model relevance: the observed Qwen3.6-27B regression uses the DENSE qwen35.cpp graph (NOT qwen35moe.cpp, which is for MoE models like Qwen3.6-35B-A3B). So for the Qwen3.6-27B regression specifically, the qwen35.cpp differences that can matter on Vulkan are: prefill_gpu capture (missing) and tape_gpu capture (missing). The qwen35moe.cpp capture absences do not affect the Qwen3.6-27B test. Tree pieces (conv/GDN tree dispatch) are inert on Vulkan (ops unimplemented in Vulkan backend on both branches) so cannot explain the Vulkan regression.
- Build/run environment (re-confirmed): no `build_vulkan/llama-cli` in the worktree; models present at `/crypt/models/` (Qwen3.6-27B-Q4_K_M, Qwen3.6-27B-DFlash-Q4_K_M, etc.); per 19:30 entry only GPU0 (integrated AMD RADV GFX1151) is available, no Vulkan1/W7800.

### HYPOTHESES
- HYPOTHESIS A (Vulkan-relevant, newly elevated): The merge-branch Vulkan DFlash corruption on Qwen3.6-27B is caused (at least in part) by the missing tape_gpu graph-builder capture block in qwen35.cpp. `gpu_layer->qkv` is allocated (llama-context.cpp:2268) and, when `use_gpu_qkv==true` on the Vulkan GPU backend, the replay path (llama-context.cpp:3800) reads `gpu_layer->qkv` which the missing block never writes -> uninitialized/stale GPU buffer -> corrupt recurrent-state conv replay -> corrupt verifier logits on multi-token batches. This is consistent with prior findings ("first bad token from verifier row 0 of a multi-token batch", "llama_dflash_rollback necessary-but-not-sufficient"). NOT YET PROVEN: requires confirming `use_gpu_qkv==true` at runtime.
- HYPOTHESIS B (Vulkan-relevant, from 19:30, still open): prefill_gpu capture missing in qwen35.cpp -> prompt-suffix hidden states not copied to prefill_gpu buffers -> stale drafter cross-attention context. Still a candidate; for the 17-token test prompt capture span = whole prompt so it may not bite short prompts, but may bite longer prompts.
- HYPOTHESIS (DISPROVEN for Vulkan, re-affirmed): porting the tree-aware conv/GDN dispatch fixes the Vulkan regression. Disproven by fact: Vulkan implements neither `ggml_ssm_conv_tree` nor `ggml_gated_delta_net_tree` on either branch, so `tree_parent_ids` must be null for flat DFlash verification on Vulkan and both branches take the identical non-tree path. The tree port is correct for CUDA/tree-verify completeness but is inert on Vulkan.
- The build_conv_state signature difference and the fused-K argument difference are NOT root causes (verified equivalent above).

### TEST RESULTS
- Command: `git show gboddaer/main:src/models/{qwen35,qwen35moe,delta-net-base}.cpp` vs `git show HEAD:...` into /tmp/cmp/; targeted greps + region reads for tree/conv/capture pieces.
- Result: code-verified all items in FACTS above. Newly discovered: tape_gpu graph-builder block missing in BOTH qwen35.cpp and qwen35moe.cpp on merge (absent from the 19:30 (a)-(e) list). qwen35moe.cpp missing all three capture blocks.
- Command: `git grep -n 'tape_gpu_n_seqs|tape_gpu_seqs' HEAD -- src/ common/ tools/ include/` and `... gboddaer/main ...`.
- Result: tape_gpu infrastructure present and near-identical on merge (only ~5-line offset); tape_gpu_n_seqs set >0 from dflash_capture->tapes; gpu_layer->qkv allocated at llama-context.cpp:2268 and consumed at 3450/3800/5671.
- Command: `git grep -n 'qkv_mixed' HEAD -- src/llama-context.cpp`.
- Result: CPU readback path (dflash_read_tensor at 1773 via tape_name_map 2060-2071) and CPU->GPU upload (3637/3652) exist as alternative tape population; consume site 3800 reads gpu_layer->qkv only when use_gpu_qkv==true.
- Output files: `/tmp/cmp/qwen35.{main,merge}.cpp`, `/tmp/cmp/qwen35moe.{main,merge}.cpp`, `/tmp/cmp/deltanet.{main,merge}.cpp`.
- No build or model run was performed in this step (comparison is static code analysis only).

### NEXT STEPS
- Confirm HYPOTHESIS A at runtime before any code change (systematic debugging): build `llama-cli` (Vulkan) in the worktree, then run the Qwen3.6-27B DFlash comparison with `GGML_DFLASH_DEBUG=1` and inspect the `[dflash-tr-conv] layer=%d OK use_gpu_qkv=%d ...` lines. If `use_gpu_qkv=1` appears during multi-token verification/replay, the missing tape_gpu block is confirmed as a live bug (gpu_layer->qkv read unwritten). If `use_gpu_qkv=0` everywhere with a working CPU-readback fallback, HYPOTHESIS A is disproven for the graph-builder path -> the tape_gpu block is a redundant/optimization path on merge and the lead reverts to HYPOTHESIS B (prefill_gpu) / server-side verifier+rollback (prior 19:30 HYPOTHESIS 2).
- If HYPOTHESIS A is confirmed, apply the minimal port (plan Task 5 Strategy A, extended): restore the tape_gpu capture block in qwen35.cpp AND qwen35moe.cpp from `gboddaer/main` (the qkv_mixed_pretranspose + k_conv/v_conv/gate/beta_presigmoid copies into tgpu->layers[li]), PLUS the prefill_gpu block (HYPOTHESIS B) and hidden_gpu block for qwen35moe. Tree pieces (conv/GDN tree dispatch, build_conv_state param) are ported for CUDA completeness but are NOT expected to affect the Vulkan regression.
- If HYPOTHESIS A is disproven, save any failed diff per plan Task 5 Step 5 and pivot to the server-side verifier/rollback boundary (prior 19:30 HYPOTHESIS 2).
- Update this file after the runtime trace with the measured `use_gpu_qkv` values.

## 2026-07-09 19:32 UTC - Build + GGML_DFLASH_DEBUG trace + greedy correctness comparison (3 branches, 3 prompt lengths)

### FACTS
- Built `llama-cli` (Vulkan, GGML_VULKAN=ON/NATIVE/Release) in 3 detached worktrees: merge working-tree (ported, uncommitted), committed merge HEAD `c5c17182e` (/tmp/cmp-head, no port), and gboddaer/main `130ea2480` (/tmp/cmp-main, working fork reference). All built 0 errors.
- The merge working tree (prior stuck session) had an INCOMPLETE, NON-BUILDING port: qwen35.cpp (prefill_gpu + tree_mode conv), qwen35moe.cpp (hidden_gpu + prefill_gpu + tree_mode), delta-net-base.cpp (tree GDN dispatch + build_conv_state `qkv_mixed_transposed` param), models.h (param decl). qwen35moe.cpp failed to compile (`invalid use of incomplete type 'struct dflash_hidden_gpu'`, 13 errors) because it lacked `#include "llama-context.h"` (which defines dflash_hidden_gpu at llama-context.h:217; qwen35.cpp already included it). FIXED by adding `#include "llama-context.h"` to qwen35moe.cpp after `#include "models.h"`.
- Models at `/crypt/models/` (Qwen3.6-27B-Q4_K_M + Qwen3.6-27B-DFlash-Q4_K_M). Only device = `Vulkan0` AMD Radeon (RADV GFX1151) iGPU (no Vulkan1/W7800). Flags: --ctx-size 8192 --flash-attn on --cache-type-k/v q8_0 --seed 7 --temp 0 (greedy) --spec-draft-n-max 3.
- GGML_DFLASH_DEBUG=1 DFlash trace on the merge working tree (short -n 80 and long -n 300): NO `[dflash-tr-conv]` or `use_gpu_qkv` lines appeared. The `[dflash-tr-conv]`/`use_gpu_qkv` log lives in `tape_replay_conv` (llama-context.cpp:3693), called only from `tape_replay` (line 2687) - the rollback path. llama-cli's flat-draft path did NOT invoke tape_replay (output was coherent, likely all drafts accepted). So `use_gpu_qkv=1` was NOT confirmable via the CLI trace - the tape path is not exercised by llama-cli.
- Isolation test (merge working tree): temporarily disabled the prefill_gpu block in qwen35.cpp (#if 0), rebuilt, ran DFlash p1 (-n 300): output STILL COHERENT at 11.1 t/s (vs 12.1 t/s with the block). Re-enabled the block. So prefill_gpu is NOT the cause of coherence (nor of correctness - see below).
- Committed HEAD (no port) DFlash with these flags is COHERENT (not the garbled 4.4 t/s reported in the 12:02/12:45 entries) at p1=15.1, p2=10.6, p3=9.5 t/s. The earlier "garbled 4.4 t/s" regression was therefore flag/ctx-specific (12:02 used --ctx-size 32768 + the exact 5-bullet prompt) and is NOT reproduced by --ctx-size 8192 + the 3-bullet variant.
- CORRECTNESS (greedy DFlash vs greedy non-DFlash, token-for-token, 3 prompt lengths p1=short, p2=medium, p3=long; generation -n 150/300/400):
  - gboddaer/main (working fork): p1 IDENTICAL; p2 diverges at gen line 17/26 (late); p3 diverges at line 23/24 (late). DFlash matches non-DFlash for the vast majority, minor late end-of-generation divergence (truncation/boundary). This is CORRECT behavior.
  - committed HEAD (no port): p1 diverges line 4, p2 line 4, p3 line 3 - EARLY divergence (~10-20 tokens in). INCORRECT.
  - merge working tree (ported): p1 line 4, p2 line 4, p3 line 3 - EARLY divergence, IDENTICAL divergence point to committed HEAD. The port changed NOTHING about correctness.
- Speed (Generation t/s, DFlash): gboddaer/main p1=25.6, p2=19.8, p3=19.7 (1.6-2x over its non-DFlash 12.4-12.5). merge working tree p1=16.7, p2=11.4, p3=9.0 (p3 SLOWER than non-DFlash 12.4). committed HEAD p1=15.1, p2=10.6, p3=9.5. So gboddaer/main DFlash is markedly faster; merge DFlash (ported or not) is much slower and sometimes slower than non-DFlash.
- cross-check: merge-wt DFlash output != gboddaer/main DFlash output for all 3 prompts (18/40/36 diff lines) - the merge DFlash path produces different tokens than the working fork.

### HYPOTHESES
- HYPOTHESIS (now the lead, elevated): The merge-branch DFlash regression is a VERIFIER / recurrent-state CORRECTNESS bug, NOT the Qwen35 graph-builder capture blocks. Under greedy (temp 0) the DFlash verifier must accept a draft token only if it equals the target's argmax for that position; gboddaer/main satisfies this (DFlash == non-DFlash mostly). The merge accepts draft tokens that differ from the target's greedy argmax very early (line 3-4, ~10-20 tokens), so the verifier acceptance and/or the multi-token verification batch recurrent-state computation is wrong on the merge. This matches the 19:30 HYPOTHESIS 2 and the "verifier/rollback boundary" finding ("first bad token from verifier row 0 of a multi-token batch"). It is in the flat DFlash draft+verify path (common/speculative.cpp / llama-context.cpp recurrent state), NOT server-specific (reproduced with llama-cli).
- HYPOTHESIS (DISPROVEN): the missing graph-builder capture blocks (prefill_gpu, tape_gpu, qwen35moe hidden_gpu) cause the correctness regression. Disproven: (a) the prior session ported prefill_gpu + qwen35moe hidden_gpu + tree, and the divergence point is byte-identical to the no-port committed HEAD (zero effect); (b) isolation test removing prefill_gpu left output coherent and divergence unchanged; (c) tape_gpu's tape_replay_conv is not even called by llama-cli (use_gpu_qkv not exercised). The capture blocks affect the drafter's cross-attention context (coherence) but NOT verifier correctness.
- HYPOTHESIS (DISPROVEN for the CLI path): tape_gpu (HYPOTHESIS A from the 19:32 comparison entry) is the root cause. Disproven for llama-cli: tape_replay_conv is not invoked (no rollback on the flat CLI path with these flags), so the missing tape_gpu graph-builder block cannot affect CLI correctness. tape_gpu may still matter for the llama-server rollback path (untested here).
- The "garbled 4.4 t/s" symptom from 12:02/12:45 is a SEPARATE, flag/ctx-specific issue (--ctx-size 32768 + the 5-bullet prompt), not the correctness divergence measured here.

### TEST RESULTS
- Command: `cmake --build build_vulkan --target llama-cli -j` in /tmp/cmp-head (c5c17182e), /tmp/cmp-main (130ea2480), and the merge worktree.
- Result: all 3 built 0 errors after adding `#include "llama-context.h"` to qwen35moe.cpp (merge worktree only).
- Command: `GGML_DFLASH_DEBUG=1 ./build_vulkan/bin/llama-cli ... --spec-type dflash ...` (merge worktree, -n 80 and -n 300).
- Result: rc=0, coherent; NO `[dflash-tr-conv]`/`use_gpu_qkv` lines (tape_replay_conv not called). DFlash -n 300 = 12.5 t/s.
- Command: isolation test - disable prefill_gpu block in qwen35.cpp, rebuild, run DFlash p1 -n 300.
- Result: rc=0, coherent 11.1 t/s (vs 12.1 with block). prefill_gpu not the correctness/coherence cause.
- Command: greedy DFlash vs non-DFlash, 3 prompts x 3 builds, extract generated text, `cmp`/`diff`.
- Result: gboddaer/main p1 IDENTICAL, p2 late div, p3 late div; committed HEAD and merge-wt BOTH early div at line 3-4 (identical). Output files: /tmp/trace/{main,wt,head}-{nodflash,dflash}-{p1,p2,p3}.{out,gen}, /tmp/trace/{main,wt,head}-p{1,2,3}.diff.
- Output files: /tmp/trace/*, detached worktrees /tmp/cmp-head (c5c17182e) and /tmp/cmp-main (130ea2480).

### NEXT STEPS
- PIVOT: abandon the graph-builder capture-block port as the correctness fix (it does not affect the early divergence). The prefill_gpu/tree/qwen35moe-hidden port is benign-to-coherence but irrelevant to correctness; decide later whether to keep/commit it.
- Investigate the DFlash verifier / multi-token-batch recurrent-state path on the merge vs gboddaer/main: why does the merge accept a draft token != target greedy argmax at line 3-4 (~10-20 tokens)? Candidates: (a) common/speculative.cpp flat-draft verifier acceptance logic differs from gboddaer/main; (b) llama-context.cpp recurrent-state (n_rs_seq, per-slot writeback HF-020 Fix 4, dparams sizing HF-021) produces non-sequential-equivalent target logits for multi-token verify batches; (c) the verifier target-logits row mapping (argmax row -> token) is wrong on the merge.
- Concrete first diagnostic: three-way diff common/speculative.cpp and the recurrent-state writeback in llama-context.cpp / delta-net-base.cpp between gboddaer/main and merge, focused on the flat-draft verify/accept path and the multi-token recurrent state advance. Add a GGML_DFLASH_QA_TRACE log of {seq_id, drafted, accepted, id_last, target_argmax_row0, draft_token_row0} per verify step to pinpoint the first wrong acceptance.
- Confirm the divergence is verifier acceptance (not sampling): both runs use --temp 0 --seed 7 (greedy, deterministic) - so any divergence is a verifier/state bug, not sampler noise (already controlled).
- The user's conditional ("if use_gpu_qkv=1 confirmed, port tape_gpu/prefill_gpu/qwen35moe-hidden") is NOT triggered: use_gpu_qkv=1 was not confirmed (tape_replay_conv not called by llama-cli), and the port does not fix correctness. Do NOT port tape_gpu as a correctness fix; it is untested for the server path.

## 2026-07-09 22:15 UTC - Verifier/recurrent-state three-way diff + GGML_DFLASH_QA_TRACE: ROOT CAUSE PINPOINTED

### FACTS
- common/speculative.cpp (gboddaer/main vs committed merge HEAD): the DFlash draft() function and common_speculative_process verify/accept logic are FUNCTIONALLY IDENTICAL. The +133 merge lines are all diagnostic additions (GGML_DFLASH_KV_TRACE/RING_DUMP/TOKEN_TRACE/QA_TRACE, already added by the prior stuck session), the n_seq_in param in common_speculative_init (HF-021, =1 for single-seq CLI = identical to main), whitespace, and TODO state stubs. The regression is NOT in common/speculative.cpp.
- common/sampling.cpp (gboddaer/main vs merge): LARGE diff but it is the reasoning-budget/grammar refactor (merge took upstream's version; gboddaer/main has the fork's on_accept callback + force_reasoning_end_on_eog etc.). The common_sampler_sample_and_accept_n acceptance LOGIC (sample target, compare to draft, accept/reject) is standard in both.
- llama-cli is a thin wrapper over the SERVER (cli.cpp posts SERVER_TASK_TYPE_COMPLETION to ctx_server). So llama-cli uses the server speculative path in tools/server/server-context.cpp. The verify/accept loop is there (lines ~4327-4450): common_sampler_sample_and_accept_n(slot.smpl, slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft) at line 4341; llama_dflash_rollback at 4408 (only under GGML_DFLASH_TEST_ROLLBACK).
- Implemented/enhanced GGML_DFLASH_QA_TRACE in common/sampling.cpp common_sampler_sample_and_accept_n: logs per draft position {i, idx, draft, target_argmax (raw argmax of target logits bypassing the sampler chain), sampled (sampler output), match}. Added a [NODFLASH_TOK] trace (GGML_NODFLASH_TOKEN_TRACE) at the non-speculative sample point (server-context.cpp ~4268) logging {sampled, target_argmax} per generated token.
- EMPIRICAL TRACE (merge working tree, p1 short prompt, --spec-draft-n-max 3, greedy temp 0):
  - target_argmax == sampled for EVERY position (raw argmax bypassing sampler == sampler output). So the sampler/reasoning-budget is NOT the cause; the sampler never alters the argmax.
  - match=1 for almost all positions (draft == target_argmax): the drafter agrees with the target's BATCH argmax. So drafter + target agree WITHIN the batch.
  - First token divergence (DFlash output vs non-DFlash greedy, both from traces): tokens 0-15 IDENTICAL: [579,264,7047,1817,25,271,16,13,220,2972,2014,53983,2570,5396,64700,198]. At index 16: DFlash=12, non-DFlash=256.
  - The trace at that verify batch: "[DFLASH_QA] sample_accept i=0 idx=0 draft=256 target_argmax=12 sampled=12 match=0 ... REJECT at i=0 (draft=256 sampled=12)". So the DRAFTER proposed 256 (== non-DFlash greedy, CORRECT), but the TARGET's multi-token verify batch computed argmax=12 (WRONG). The verifier rejected the correct draft token 256 and emitted 12.
  - Batches 1-4 before index 16 were all-accepted (match=1 all positions + bonus, no rollback). So the wrong argmax at index 16 (batch 5 position 0) is NOT due to a missing rollback after a partial accept (there was no prior partial accept). It is due to CUMULATIVE DRIFT of the recurrent state advanced within multi-token batches vs sequential single-token advance: after 4 batches (16 tokens) the delta-net recurrent state has drifted enough that argmax flips 256->12.

### HYPOTHESES
- HYPOTHESIS (ROOT CAUSE, confirmed empirically): the merge-branch target computes non-sequential-equivalent logits for multi-token DFlash verify batches on the Qwen3.6 delta-net recurrent layers. The recurrent state advanced within a multi-token batch (ggml_gated_delta_net over N tokens) drifts from sequential single-token advance; the drift accumulates across verify batches and flips the argmax at index 16 (drafter was correct at 256; target batch argmax was wrong at 12). The drafter and the verify/accept logic are correct; the bug is in the TARGET recurrent-state advance/writeback for multi-token batches.
- HYPOTHESIS (lead for the code location): the delta-net recurrent-state WRITEBACK after the verify batch (build_recurrent_attn / the ssm_states write-back in delta-net-base.cpp) differs between gboddaer/main and merge (HF-020 Fix 4 replaced upstream's bulk ggml_view_3d writeback with a per-slot ggml_view_2d/4d loop), and/or the initial recurrent state s_0 passed into the batch is wrong. The 19:30 comparison examined the tree branch / build_conv_state / K arg but did NOT closely compare the recurrent-state writeback. This is the next region to diff.
- HYPOTHESIS (DISPROVEN): the regression is in the DFlash drafter, the verify/accept logic (common/speculative.cpp), or the sampler (reasoning-budget). Disproven: drafter correctly predicted 256; common/speculative.cpp is functionally identical; target_argmax==sampled (sampler does not alter argmax).

### TEST RESULTS
- Command: diff common/speculative.cpp (gboddaer/main vs merge HEAD) -> /tmp/vdiff/spec.full.diff.
- Result: only diagnostics + n_seq_in + stubs differ; draft() and common_speculative_process functionally identical.
- Command: diff common/sampling.cpp -> /tmp/vdiff/sampling.u.diff; read common_sampler_sample_and_accept_n.
- Result: reasoning-budget refactor diff; acceptance logic standard; enhanced QA trace added.
- Command: GGML_DFLASH_QA_TRACE=1 llama-cli DFlash p1; GGML_NODFLASH_TOKEN_TRACE=1 llama-cli non-DFlash p1; Python parse+compare token sequences.
- Result: tokens 0-15 identical; first divergence index 16 DFlash=12 vs non-DFlash=256; drafter draft=256 (correct), target batch argmax=12 (wrong); REJECT. Output files: /tmp/trace/qa2-dflash.{out,err}, /tmp/trace/qa2-nodflash.{out,err}.
- Speed (this build, p1 -n 150): DFlash 15.6 t/s, non-DFlash ~12.5 t/s.

### NEXT STEPS
- Diff the delta-net recurrent-state WRITEBACK (build_recurrent_attn and the ssm_states new_state write-back in delta-net-base.cpp) between gboddaer/main and merge, focused on the per-slot writeback (HF-020 Fix 4) and the initial state s_0 passed to ggml_gated_delta_net for multi-token batches. Identify the exact line that causes the batch state to drift from sequential.
- Add a QA trace of the recurrent state (s_0 / s_N values or norms) per verify batch on merge vs gboddaer/main to confirm the state drifts.
- Candidate fix: restore gboddaer/main's recurrent-state writeback/advance semantics for the flat (non-tree) multi-token verify path on Vulkan. Do NOT change the drafter or verify/accept logic.

## 2026-07-09 22:20 UTC - ROOT CAUSE FOUND + PRIMARY FIX: Vulkan gated_delta_net shader slot-order reversal

### FACTS
- Pinpointed the first wrong acceptance via GGML_DFLASH_QA_TRACE (enhanced in common/sampling.cpp common_sampler_sample_and_accept_n to log per position: draft, target_argmax (raw argmax bypassing the sampler chain), sampled, match; plus a [NODFLASH_TOK] trace at server-context.cpp non-speculative sample point). Build + run on merge working tree, p1, --spec-draft-n-max 3, greedy.
- First wrong acceptance at TOKEN INDEX 16: DFlash output=12, non-DFlash greedy=256. The trace at that verify batch: "sample_accept i=0 idx=0 draft=256 target_argmax=12 sampled=12 match=0 REJECT". The DRAFTER predicted 256 (== non-DFlash greedy, CORRECT); the TARGET's multi-token verify batch computed argmax=12 (WRONG). target_argmax==sampled for every position (sampler never alters argmax -> sampler/reasoning-budget NOT the cause). Batches 1-4 all-accepted (no rollback) -> cumulative recurrent-state drift within multi-token batches flips argmax at index 16.
- ROOT CAUSE (code-level): the merge's Vulkan shader ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp reversed the recurrent-state SNAPSHOT SLOT convention vs gboddaer/main:
  - main: `target_slot = int(t) - shift` where `shift = n_tokens - K` -> slot K-1 = NEWEST state, slot 0 = oldest-of-last-K (ascending slot = ascending time).
  - merge: `target_slot = int(n_tokens) - 1 - int(t)` -> slot 0 = NEWEST, slot K-1 = oldest (REVERSED).
  - The writeback helper `llama_dflash_rs_writeback_slot_for_test` (llama-context.h:126) and the build_recurrent_attn per-slot writeback loop (delta-net-base.cpp) are BYTE-IDENTICAL between branches. They map `gdn_slot=k_i -> cache_slot=K-1-k_i` and read gdn slots [K-n_pop, K) - which is correct ONLY for main's convention (newest at gdn slot K-1 -> cache_slot 0 = active state for next batch). With the merge's reversed kernel, the newest state lands in the wrong cache slot (and for n_tokens<K the kernel writes slots [0,n_pop) while the writeback reads [K-n_pop,K) -> disjoint -> the active recurrent state read by the next batch is stale/wrong). -> cumulative drift -> argmax flips at token 16.
  - Secondary shader change: `state_in_base = (seq_id*K*H + head)*state_size` (main, 3D state layout D,K,n_seqs) -> `(seq_id*H + head)*state_size` (merge, assumes 4D s0-only). Inert for single-seq (seq_id=0) but a latent multi-seq bug.
- This is an UPSTREAM-vs-FORK convention mismatch: the merge took upstream's ggml-vulkan.cpp + gated_delta_net.comp (reversed slot convention, "slot 0 = most recent") but kept the fork's graph (delta-net-base.cpp) + writeback helper (fork convention, "slot K-1 = newest -> cache_slot 0"). The two are incompatible.
- FIX APPLIED: restored gboddaer/main's shader convention in gated_delta_net.comp - `state_in_base = (seq_id * K * H + head_id) * state_size` and `const int shift = int(n_tokens) - int(K); ... target_slot = int(t) - shift;` (kept the merge host code reading K from op_params; the graph still passes s_3d_pad 3D + K op-param, so K is correct). Built 0 errors (shader recompiled into embedded header).
- VERIFICATION after fix (merge working tree, Qwen3.6-27B-Q4_K_M, Vulkan0 iGPU, greedy):
  - p1 (short, -n 150): DFlash output == non-DFlash output IDENTICAL (14 gen lines). DFlash 19.7 t/s vs non-DFlash 12.6 t/s = 1.56x speedup. PRIMARY REGRESSION FIXED. Matches gboddaer/main behavior (p1 was identical, 25.6 t/s).
  - p2 (medium, -n 300): tokens 0-25 IDENTICAL (token-16 bug GONE: DFlash now [...,198,256,471,2972,...] matching non-DFlash). FIRST divergence at TOKEN INDEX 26: DFlash inserts a DUPLICATE 436 ([...,303,436,436,10673,...] vs non-DFlash [...,303,436,10673,...]), then re-syncs shifted by 1. DFlash 11.7 t/s (slow). REJECT count 59/357 tokens.
  - p3 (long, -n 400): first text divergence at gen line 18/24 (vs gboddaer/main's line 23/24). DFlash 9.2 t/s (slow).
  - So the shader slot fix resolved the PRIMARY regression (p1 identical + speedup; p2 token-16 fixed; p2/p3 first-26-tokens fixed) but a SECOND residual issue remains for longer generations: a duplicate-token insertion at p2 index 26 and a high reject rate (slow p2/p3 speed).

### HYPOTHESES
- HYPOTHESIS (confirmed): the merge DFlash correctness regression is the Vulkan gated_delta_net shader snapshot-slot reversal vs the fork's graph+writeback convention. Fixing it (restore main's target_slot + state_in_base) restores greedy DFlash==non-DFlash for the first ~26 tokens and full p1 identity + speedup. CONFIRMED by p1 identical + p2 token-16 fixed.
- HYPOTHESIS (lead for the residual): a SECOND recurrent-state snapshot mechanism has the same upstream-vs-fork convention mismatch - most likely the SSM CONV state (ggml_ssm_conv shader / the conv_states_all writeback in build_conv_state) or a second gated_delta_net call path (build_delta_net_fused). The duplicate-436 at p2 index 26 (a target output token) indicates the target verify-batch logits are STILL wrong at that position for p2 -> another target recurrent-state drift, distinct from the slot-16 one. Also the high reject rate (59/357) -> drafter draft quality low for longer prompts -> possibly the drafter cross-attention context (prefill_gpu) is still incomplete, but that affects speed/acceptance not target argmax; the duplicate token is a target-side issue.
- HYPOTHESIS (lower): the residual is inherent DFlash late-divergence like gboddaer/main (p2 line 17, p3 line 23). PARTLY DISPROVEN: merge p2 diverges at token 26 (~earlier than gboddaer/main's line ~17/26), so merge is still worse than gboddaer/main for p2.

### TEST RESULTS
- Command: enhanced GGML_DFLASH_QA_TRACE in common/sampling.cpp + [NODFLASH_TOK] in server-context.cpp; build; GGML_DFLASH_QA_TRACE=1 + GGML_NODFLASH_TOKEN_TRACE=1 DFlash and non-DFlash p1/p2; Python token-sequence compare.
- Result: p1 first divergence pre-fix at index 16 (DFlash=12 vs 256; drafter draft=256 correct, target_argmax=12 wrong). Post-fix: p1 IDENTICAL; p2 tokens 0-25 identical, divergence at index 26 (duplicate 436).
- Command: diff gboddaer/main vs merge of gated_delta_net.comp -> /tmp/vdiff/gdn.comp.diff; diff llama-context.h helper (identical); diff delta-net-base.cpp build_recurrent_attn (identical writeback loop).
- Result: shader target_slot + state_in_base differ (merge reversed); helper + graph identical. Root cause = shader convention mismatch.
- Command: apply shader fix; rebuild; run p1/p2/p3 DFlash vs non-DFlash greedy.
- Result: p1 IDENTICAL 19.7 t/s (1.56x); p2 tokens 0-25 identical then duplicate-436 at 26, 11.7 t/s; p3 divergence line 18, 9.2 t/s.
- Output files: /tmp/trace/qa2-*.{out,err}, /tmp/trace/fix-*.{out,err}, /tmp/trace/fix2-*.{out,err}, /tmp/trace/fixqa-*.{out,err}, /tmp/vdiff/gdn.comp.diff.

### NEXT STEPS
- Investigate the SECOND residual issue: diff the ggml_ssm_conv Vulkan shader (and the conv_states_all writeback in build_conv_state, delta-net-base.cpp ~480-520) between gboddaer/main and merge for an analogous snapshot-slot/scope reversal. Add a conv-state trace to confirm the conv state drifts at p2 token 26.
- Separately assess the slow p2/p3 speed (high reject rate): may be the drafter cross-attention context (prefill_gpu port) being incomplete for longer prompts, OR a consequence of the residual target-state drift causing rejections. Fix the target-state residual first; re-measure speed.
- Consider also restoring state_in_base's K factor is already done; verify multi-seq (n_seqs>1) correctness on a MoE model (qwen35moe) once single-seq is fully correct.
- Do NOT commit yet (user has not asked). The working tree now has: the prior session's capture-block port (prefill_gpu/tree/qwen35moe hidden + the llama-context.h include fix) + my GGML_QA_TRACE enhancements + the gated_delta_net.comp shader slot fix. The shader fix is the primary correctness fix; the capture-block port's value is still TBD (prefill_gpu did not affect correctness per the isolation test).

## 2026-07-09 22:35 UTC - Residual characterization: target graph now identical to gboddaer/main; residual is server-side

### FACTS
- After the shader fix, the fixed gated_delta_net.comp is byte-identical to gboddaer/main (diff exit 0).
- ssm_conv.comp shader: IDENTICAL between gboddaer/main and merge.
- build_conv_state conv-state writeback (the K=n_rs_seq+1 loop, s_slot=K-t, s_idx=max(0,...-K+t)): IDENTICAL between gboddaer/main and merge.
- build_recurrent_attn recurrent-state writeback + llama_dflash_rs_writeback_slot_for_test helper: IDENTICAL.
- common/speculative.cpp draft + verify/accept: functionally identical (only diagnostics + n_seq_in + stubs).
- => The entire TARGET recurrent-state graph computation is now IDENTICAL to gboddaer/main. Yet merge p2 still diverges at token 26 (vs gboddaer/main ~token 100). So the RESIDUAL drift is NOT in the target graph; it is in SERVER-SIDE verify-batch construction / recurrent-state save-restore.
- The residual duplicate-436 at p2 index 26: DFlash=[...,303,436,436(dup),10673,...], non-DFlash=[...,303,436,10673,...]. The trace shows index 26 is a REJECT resample with sampled=436 (target argmax=436), i.e. the target's verify-batch argmax at that position = 436 == the PREVIOUS token (index 25=436), while sequential argmax = 10673. This is consistent with a POSITION/n_past OFF-BY-ONE in the verify batch: the target re-decodes token 25's position at index 26, reproducing 436. This points to the server verify-batch construction (server-context.cpp ~4327-4450) and/or the recurrent-state save/restore between batches (llama_dflash_rollback / seq_backup / common_context_seq_rm) - the HF-017 "lost ~652 server DFlash lines" territory, which the earlier "verifier/rollback boundary" investigation also identified ("first bad token from verifier row 0 of a multi-token batch", "merge's simpler path only uses common_context_seq_rm after verification vs main's recurrent backup + llama_dflash_rollback").
- High REJECT count on p2 (59/357) -> low draft acceptance -> slow p2/p3 speed (11.7/9.2 t/s vs gboddaer/main 19.8/19.7). Likely a consequence of the residual target-state/batch drift (rejections) and/or the drafter cross-attention context (prefill_gpu) still incomplete for longer prompts. Fix the verify-batch residual first, then re-measure speed.

### HYPOTHESES
- HYPOTHESIS (lead for residual): the merge's server-side DFlash verify-batch construction and/or recurrent-state save-restore between verify batches is wrong (position/n_past off-by-one and/or missing recurrent backup+rollback), causing the target to re-decode a position (duplicate-436) and drift. gboddaer/main's server has the full DFlash verify/rollback integration (~652 lines, HF-017) that the merge lost. This is independent of the (now-fixed) gated_delta_net shader slot ordering.
- The primary regression (shader slot reversal) and the residual (server-side verify/rollback) are TWO DISTINCT bugs. The shader fix is correct and complete for the target recurrent-state graph; the residual requires porting the fork's server-side DFlash verify/rollback batch construction from gboddaer/main.

### TEST RESULTS
- Command: diff fixed shader vs gboddaer/main; diff ssm_conv.comp; diff build_conv_state writeback region.
- Result: all identical (target graph computation now matches gboddaer/main exactly).
- Command: GGML_DFLASH_QA_TRACE p2 token-sequence parse (post-fix).
- Result: p2 tokens 0-25 identical to non-DFlash; divergence at index 26 = REJECT resample sampled=436 (target argmax=436, duplicate of index 25), sequential=10673. Consistent with position off-by-one in verify batch.

### NEXT STEPS
- Phase 2 (residual): three-way diff tools/server/server-context.cpp (and the llama_dflash_rollback / seq_backup / recurrent-state save-restore path in src/llama-context.cpp) between gboddaer/main and merge, focused on the verify-batch construction (n_past/id_last/position tracking) and the recurrent-state save/restore between verify batches. Add a QA trace of {batch n_past, id_last, spec_draft positions, dflash_seq_backup, n_pos_before_draft} per verify step to pinpoint the off-by-one.
- Candidate fix: restore gboddaer/main's server-side DFlash verify/rollback integration (the ~652 lost lines, HF-017) - the recurrent backup + llama_dflash_rollback around partial acceptance, and the correct position/id_last bookkeeping. This is a larger port than the shader fix.
- Re-measure p2/p3 speed after the residual fix (expect acceptance rate + throughput to recover toward gboddaer/main's 19.7-19.8 t/s).
- The shader fix (gated_delta_net.comp) is a standalone, verified correctness fix for the PRIMARY regression and should be kept regardless of the residual. Do NOT commit yet (user has not asked).

## 2026-07-09 22:40 UTC - PHASE 1 COMPLETE: primary DFlash correctness regression FIXED; transition to Phase 2

### FACTS
- The primary Qwen3.6-27B DFlash correctness regression (greedy DFlash diverging from greedy non-DFlash early, with garbled/slow output) is FIXED by restoring gboddaer/main's Vulkan gated_delta_net.comp snapshot-slot convention. Verified: p1 DFlash == non-DFlash IDENTICAL at 19.7 t/s (1.56x over non-DFlash 12.6 t/s), matching gboddaer/main behavior (p1 identical, ~20 t/s). p2 token-16 argmax-flip fixed (tokens 0-25 identical to non-DFlash). Build 0 errors.
- Root cause (one line): the merge took UPSTREAM's Vulkan gated_delta_net.comp (reversed snapshot slot convention: slot 0 = newest) but kept the FORK's delta-net graph + llama_dflash_rs_writeback_slot_for_test helper (convention: slot K-1 = newest -> cache_slot 0 = active state). The mismatch stored the newest recurrent state in the wrong cache slot -> the next verify batch read a stale s0 -> cumulative drift -> argmax flipped at token 16 (drafter was correct at 256; target batch argmax was wrong at 12).
- Fix: in ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp restore `state_in_base = (seq_id * K * H + head_id) * state_size` and `const int shift = int(n_tokens) - int(K); ... target_slot = int(t) - shift;` (now byte-identical to gboddaer/main). The merge host code still reads K from op_params; the graph still passes s_3d_pad 3D + K op-param, so K is correct.
- Coherence fix (prior session, made buildable): `#include "llama-context.h"` added to qwen35moe.cpp (it dereferences dflash_hidden_gpu, defined in that header) + the prefill_gpu/tree_mode/qwen35moe hidden_gpu capture-block port. These restored buildability and output coherence; the isolation test showed prefill_gpu does not affect correctness (the gated_delta_net shader fix is what fixed correctness).
- Diagnostics kept (gated, zero behavior change when unset): enhanced GGML_DFLASH_QA_TRACE in common/sampling.cpp common_sampler_sample_and_accept_n (logs per position: draft, target_argmax raw, sampled, match, REJECT) and a [NODFLASH_TOK] trace at tools/server/server-context.cpp non-speculative sample point. These pinpointed the first wrong acceptance (token 16) and will support Phase 2.
- REMAINING (Phase 2): a SECOND, distinct residual bug. After the shader fix the entire target recurrent-state graph computation is byte-identical to gboddaer/main, yet p2/p3 still diverge later (p2 at token 26: duplicate-436 = position n_past off-by-one in the verify batch) and are slow (p2 11.7 t/s, p3 9.2 t/s vs gboddaer/main 19.8/19.7). The residual is SERVER-SIDE verify-batch construction / recurrent-state save-restore (the HF-017 ~652 lost server DFlash lines: the fork's llama_dflash_rollback + recurrent backup that the merge's simpler common_context_seq_rm path lacks).

### HYPOTHESES
- Phase 2 lead hypothesis: the merge's server-side DFlash verify-batch construction and/or recurrent-state save-restore between verify batches is wrong (position/n_past off-by-one and/or missing recurrent backup+rollback), causing the target to re-decode a position (duplicate-436 at p2 index 26) and drift. gboddaer/main's server has the full DFlash verify/rollback integration (~652 lines) that the merge lost. Independent of the (now-fixed) gated_delta_net shader.

### NEXT STEPS (Phase 2)
- Three-way diff tools/server/server-context.cpp (verify/accept loop ~4327-4450, the llama_dflash_rollback / seq_backup / dflash_seq_backup / dflash_n_pos_before_draft path) and the recurrent-state save-restore path in src/llama-context.cpp between gboddaer/main and merge.
- Implement the detailed QA trace for the verify batch: per verify step log {slot.id, batch n_past/id_last, spec_draft tokens+positions, spec_i_batch, dflash_seq_backup, dflash_n_pos_before_draft, n_rollback, use_ckpt_tgt, whether llama_dflash_rollback ran}. Gate behind GGML_DFLASH_QA_TRACE. Run on merge p2, pinpoint the off-by-one (which position is re-decoded).
- Candidate fix: restore gboddaer/main's server-side DFlash verify/rollback integration (the ~652 lost lines) - recurrent backup + llama_dflash_rollback around partial acceptance, and correct position/id_last bookkeeping. Re-measure p2/p3 speed (expect acceptance + throughput to recover toward 19.7-19.8 t/s).

## 2026-07-09 22:55 UTC - PHASE 2: server-context.cpp three-way diff + n_past/seq_backup QA trace CONFIRMS missing-rollback root cause

### FACTS
- Committed + pushed Phase 1 (primary fix) as commit a3732a5c3 to gboddaer/merge_llama_into_beellama_2 ("dflash(vulkan): fix gated_delta_net snapshot slot ordering"). TASK_PROGRESS.md marks Phase 1 complete; Phase 2 begun.
- Three-way diff tools/server/server-context.cpp: gboddaer/main = 9002 lines, merge = 5928 lines (merge took upstream's server). The DFlash verify/rollback integration differs critically:
  - gboddaer/main (lines ~7284-7325, the COMMON_SPECULATIVE_TYPE_DFLASH branch of the verify/accept loop): on partial accept, calls `llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep)` UNCONDITIONALLY (production), then re-decodes the accepted tokens (logits=false) to advance the recurrent state correctly. On all-accepted, clears tree parent ids + removes seq_backup. This is the ~652-line DFlash server integration (HF-017).
  - merge (lines ~4396-4440): `llama_dflash_rollback` is called ONLY under `std::getenv("GGML_DFLASH_TEST_ROLLBACK")` (line 4414) - i.e., NEVER in production. The production partial-accept path only does `common_context_seq_rm` (KV removal) when `use_ckpt_tgt` is true; for DFlash on Vulkan with RS seq_rm_type and n_rollback <= n_rs_seq, `use_ckpt_tgt=false` -> NO recurrent-state restore at all.
- Implemented the detailed QA trace for n_past/seq_backup (gated by GGML_DFLASH_QA_TRACE) in the merge verify loop: `[DFLASH_QA] verify_pre slot=%d n_pos_before_draft=%d pos_next=%d id_last=%d n_draft=%zu seq_backup=%d spec_i_batch0=%d spec_draft=...` and `[DFLASH_QA] verify_post slot=%d accepted=%zu n_rollback=%u use_ckpt_tgt=%d seq_rm_type=%d n_rs_seq=%d test_rollback_env=%d all_accepted=%d`. Built + ran on p2.
- QA trace CONFIRMS the missing-rollback root cause on p2 (107 verify batches):
  - `seq_backup=-1` for EVERY batch -> the recurrent-state backup is NEVER created in production (the merge's dflash_seq_backup setup at lines 3351-3361 only assigns a real seq id when `llama_context_recurrent_expand` returns true; for the single-slot path it returns false -> dflash_seq_backup reset to -1 -> no backup).
  - `n_pos_before_draft=0` (dflash_n_pos_before_draft) for EVERY batch -> the position-before-draft is not populated in this path (the real n_past is slot.prompt.tokens.pos_next() = pos_next, which IS correct: 85, 89, 93, 97, ... incrementing by 4 per all-accepted batch).
  - 59 partial accepts (all_accepted=0), 0 rollbacks (use_ckpt_tgt=0 every time because seq_rm_type=3=RS and n_rollback(1-3) <= n_rs_seq(3); test_rollback_env=0). So after every partial accept the recurrent state is left over-advanced and never restored.
  - The duplicate-436 at token 26 pinpointed: batch 7 (pos_next=109, id_last=45543, spec_draft=[303,3349,10673]) -> verify_post accepted=2 n_rollback=2 all_accepted=0 (draft 303 accepted as token 24; draft 3349 rejected; resample 436 = token 25). The recurrent state was advanced through the full 4-position batch with NO rollback. Batch 8 (pos_next=111, id_last=436, spec_draft=[10673,42903,198]) then re-emits id_last=436 at token 26 (duplicate) before 10673 at token 27 - because the recurrent state entering batch 8 is over-advanced by 2 positions (no rollback after batch 7's partial accept). This is the exact off-by-one/duplicate-token symptom.

### HYPOTHESES
- HYPOTHESIS (CONFIRMED): the merge's production DFlash path does not roll back / restore the recurrent state after partial-accept verify batches (llama_dflash_rollback is env-gated test-only; use_ckpt_tgt=false for DFlash-on-Vulkan-RS; no recurrent backup is even created - seq_backup=-1). Each partial accept leaves the delta-net recurrent state over-advanced; the drift accumulates and flips the target argmax (duplicate-436 at p2 token 26; the earlier p2 token-16 divergence pre-Phase-1 was the same class compounded with the shader slot bug). gboddaer/main's production llama_dflash_rollback + re-decode path prevents this.
- The fix is to port gboddaer/main's production DFlash verify/rollback integration into the merge's verify loop: (1) create the recurrent-state backup (seq_backup) before each draft (main does this unconditionally per slot, not gated on recurrent_expand), (2) on partial accept call llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, n_pos_before_draft, n_hidden_keep) + re-decode the accepted tokens, (3) on all-accepted clear tree parent ids + rm seq_backup. Plus set n_pos_before_draft correctly.

### TEST RESULTS
- Command: three-way diff server-context.cpp (gboddaer/main vs merge) for DFlash verify/rollback markers.
- Result: main has production llama_dflash_rollback (line 7304, unconditional); merge only env-gated (line 4414). merge lost ~652 DFlash server lines (HF-017).
- Command: add verify_pre/verify_post QA trace (n_past/seq_backup/n_rollback/use_ckpt_tgt) to merge verify loop; build; GGML_DFLASH_QA_TRACE=1 run p2.
- Result: 107 batches, seq_backup=-1 and n_pos_before_draft=0 every batch, 59 partial accepts, 0 rollbacks (use_ckpt_tgt=0, test_rollback_env=0). Batch 7 partial accept (no rollback) -> batch 8 duplicate-436 at token 26. Confirms missing-rollback.
- Output files: /tmp/p2/qa-p2.{out,err}, /tmp/p2/{sc,lc}.{main,merge}.cpp.

### NEXT STEPS
- Port gboddaer/main's production DFlash verify/rollback into the merge verify loop (server-context.cpp ~4396-4440). Map the variables: main `seq_backup` (local) / `slot.n_pos_before_draft` / `slot.n_tokens_before_draft` / `n_hidden_keep` / `all_accepted_flat` / `is_draft_tree` -> merge `slot.dflash_seq_backup` / `slot.dflash_n_pos_before_draft` / etc. Ensure the recurrent backup is actually created (main creates seq_backup per-slot unconditionally - find main's seq_backup assignment and port it). Keep the existing env-gated test path as a fallback or remove it.
- Build, re-run p2/p3 DFlash vs non-DFlash greedy; expect the duplicate-436 and the late p2/p3 divergence to disappear and speed to recover toward gboddaer/main (~19.7-19.8 t/s). Re-run p1 to confirm no regression (should stay identical).
- Keep the verify_pre/verify_post QA trace (gated) for now; remove or keep behind the env gate once the port is verified.
- Do NOT commit until the port is verified and the user asks.

## 2026-07-09 23:05 UTC - PHASE 2 (cont.): llama-context.cpp save-restore diff + seq_backup mechanism

### FACTS
- src/llama-context.cpp recurrent-state save-restore infrastructure is ESSENTIALLY IDENTICAL between gboddaer/main and merge (line offsets ~5): tape_replay (main 2682 / merge 2687), tape_replay_conv (main 3693 / merge 3693), tape_replay_gdn_direct_gpu, tape_replay_cpu, allocate_tape_gpu (main 2128 / merge 2133), set_tape_recording, llama_context_recurrent_expand, dflash_memory_seq_cp_recurrent_ordered (main 2516 / merge 2521), active_tape/active_tape_idx. So the rollback machinery EXISTS in the merge; it is simply not INVOKED by server-context.cpp in production.
- gboddaer/main creates the recurrent-state backup BEFORE the draft via `slot.seq_id_backup = seq_backup` where `const llama_seq_id seq_backup = slot.id + n_parallel_user;` (server-context.cpp lines 4771-4787 and 4939-4950, in the draft-setup path), and copies the recurrent state into it (`llama_memory_seq_cp_recurrent`). Then the verify loop reads `const llama_seq_id seq_backup = slot.seq_id_backup;` (line 7207) and on partial accept calls `llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep)` (line 7304) + re-decodes accepted tokens (lines 7316-7332).
- The merge instead has `slot.dflash_seq_backup` (a different member) set at lines 3354-3361 ONLY when `llama_context_recurrent_expand` returns true (which it does not for the single-slot CLI/server path -> dflash_seq_backup stays -1 -> no backup -> the env-gated llama_dflash_rollback at 4417 would receive seq_backup=-1 and cannot work). So the merge's backup-creation is effectively dead for the single-slot path, and the production rollback call is absent.

### HYPOTHESES
- The Phase 2 fix is a server-context.cpp-only port (no llama-context.cpp change needed): (1) create the recurrent backup per-slot before the draft the way main does (slot.id + n_parallel_user + llama_memory_seq_cp_recurrent), populating slot.dflash_seq_backup (or port slot.seq_id_backup) and slot.dflash_n_pos_before_draft; (2) in the verify loop replace the env-gated test rollback with main's production DFLASH branch: on all-accepted clear tree parent ids + rm seq_backup; on partial accept llama_dflash_rollback + re-decode accepted tokens (logits=false). The llama-context.cpp tape_replay/rollback functions are already present and identical, so once the backup is created and the call is made, rollback should work.

### NEXT STEPS
- Implement the server-context.cpp port (map main's slot.seq_id_backup / n_pos_before_draft / n_tokens_before_draft / n_hidden_keep / all_accepted_flat / is_draft_tree / commit_depth onto the merge's verify loop and slot members). Build; re-run p1/p2/p3 DFlash vs non-DFlash greedy. Expect: p1 stays identical; p2 duplicate-436 gone + tokens match non-DFlash longer; p2/p3 speed recovers toward ~19.7-19.8 t/s. Keep the verify_pre/verify_post QA trace gated for the verification run, then decide whether to keep or remove.
- This port is larger and riskier than the shader fix (the merge took upstream's server, so the verify-loop structure differs from main's). Do NOT commit until verified; ask the user before the port if a checkpoint is wanted.

## 2026-07-09 23:15 UTC - PHASE 2 checkpoint commit + production verify/rollback port START

### FACTS
- Checkpoint committed + pushed Phase 2 QA trace as commit 100a9ae24 to gboddaer/merge_llama_into_beellama_2 ("dflash(server): add verify_pre/verify_post QA trace for n_past/seq_backup (Phase 2 diagnostic)"). Remote head now 100a9ae24.
- Re-confirmed the exact gating that makes the merge's DFlash rollback dead in production:
  - Draft-setup backup-creation block (server-context.cpp ~3351-3362) is gated by `std::getenv("GGML_DFLASH_TEST_ROLLBACK")` -> never runs in production -> dflash_seq_backup stays -1, dflash_n_pos_before_draft stays 0 (matches the QA trace: seq_backup=-1, n_pos_before_draft=0 every batch).
  - Verify-loop rollback block (~4404-4445) is gated by `std::getenv("GGML_DFLASH_TEST_ROLLBACK") && slot.dflash_seq_backup >= 0` -> never runs in production.
  - Both blocks are otherwise structurally close to gboddaer/main's production path (backup = slot.id + slots.size() + llama_memory_seq_cp_recurrent; on partial accept llama_dflash_rollback + re-decode accepted tokens; on all-accepted rm seq_backup).
- The merge's backup cp_recurrent is INSIDE `if (llama_context_recurrent_expand(...))` with an `else { dflash_seq_backup = -1; }`. recurrent_expand returns false once the backup seq already exists (after the first batch), so even with the env gate removed, subsequent batches would set dflash_seq_backup=-1 and skip cp. The cp must be moved out of the `if (expand)` so it runs whenever the slot exists (expand creates it on first batch; it already exists on later batches).

### HYPOTHESES
- The production port is a SMALL, server-context.cpp-only change (the llama-context.cpp rollback infrastructure is already present and identical): (1) in the draft-setup block, drop the GGML_DFLASH_TEST_ROLLBACK env gate and move llama_memory_seq_cp_recurrent out of the `if (recurrent_expand)` so the backup is always created for DFlash; (2) in the verify-loop rollback block, drop the env gate (gate on DFlash type instead) so llama_dflash_rollback + re-decode runs in production on partial accept. This should make seq_backup real and restore the recurrent state after every partial accept -> eliminate the duplicate-436 / drift and recover p2/p3 speed.

### NEXT STEPS
- Apply the two edits. Build. Re-run p1 (must stay identical), p2/p3 DFlash vs non-DFlash greedy (expect divergence gone + speed recovered). Keep GGML_DFLASH_QA_TRACE on for the verification run to confirm seq_backup is now real and rollbacks happen on partial accepts.
- If verified, commit the port; ask the user before pushing.

### TEST RESULTS
- (port not yet applied)

## 2026-07-09 23:35 UTC - PHASE 2 port attempt: naive production rollback REGRESSES p1; reverted

### FACTS
- Applied the minimal production port (two edits in server-context.cpp): (1) draft-setup backup-creation block - drop the GGML_DFLASH_TEST_ROLLBACK env gate, move llama_memory_seq_cp_recurrent out of `if (recurrent_expand)` so the backup is always created for DFlash; (2) verify-loop rollback block - gate on DFlash type instead of the env var, so llama_dflash_rollback + re-decode runs in production on partial accept. Built 0 errors.
- QA trace after port (p2): seq_backup now real (=1 every batch, was -1), n_pos_before_draft now correct (81/85/89...), 58 rollback events (was 0). So the mechanism engaged as intended.
- BUT the port REGRESSED correctness and speed: p1 DFlash diverged (was IDENTICAL) at gen line 6 with 21 partial accepts (was 0) and 10.7 t/s (was 19.7); p2 8.0 t/s divergent; p3 8.5 t/s divergent.
- Isolation test (backup-creation ON, rollback OFF): p1 stays IDENTICAL to non-DFlash but slows to 13.1 t/s with 11 partial accepts (was 0) and 0 rollbacks. So:
  - backup-creation alone does NOT corrupt correctness (p1 identical), but it increases partial accepts (0->11) and slows generation (19.7->13.1 t/s). The per-batch `llama_context_recurrent_expand` (expanding the recurrent memory to the backup seq) is the likely cause of the extra rejections/slowdown (it perturbs the target recurrent state each batch, causing more draft rejections even though the resampled token still matches non-DFlash greedy for short p1).
  - the rollback+re-decode path is what CORRUPTS p1 (divergent). So the merge's llama_dflash_rollback + re-decode (batch_tokens=[slot.sampled, spec_draft...] at dflash_n_pos_before_draft+j) is itself BUGGY: enabling it corrupts the recurrent state, not restores it.
- Reverted server-context.cpp to the checkpoint state (commit 100a9ae24, env-gated test path). Rebuilt + verified: p1 back to IDENTICAL at 19.6 t/s. No regression introduced; the Phase 1 shader fix (commit a3732a5c3) remains the shipped correctness fix.
- Working tree: only TASK_PROGRESS.md modified (this entry); server-context.cpp is back at 100a9ae24.

### HYPOTHESES
- The production DFlash verify/rollback port is NOT a simple gate-removal. Two distinct sub-issues:
  - H-A (slowdown): per-batch llama_context_recurrent_expand perturbs the target recurrent state / causes extra draft rejections. gboddaer/main pre-expands the recurrent memory ONCE (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full) at setup, server-context.cpp lines 2106/2649) rather than per-batch; the merge does it per-batch. Fix: pre-expand once at setup (port main's n_seq_max_full pre-expansion) instead of per-batch.
  - H-B (corruption): the merge's llama_dflash_rollback + re-decode produces a wrong recurrent state. Candidates: (i) llama_dflash_rollback/tape_replay restores the wrong state (the merge's tape_replay infrastructure is "identical" per the diff, but the tape_gpu graph-builder capture block is MISSING - per the 19:30/Phase-1 finding - so the tape buffers gpu_layer->qkv are never written -> tape_replay_conv reads uninitialized data -> corrupt restore); (ii) the re-decode uses batch_tokens=[slot.sampled, spec_draft...] (includes the full draft list) and indexes [0..n_reeval), which may re-decode a rejected draft or use wrong positions vs main's `slot.prompt.tokens[n_tokens_before_draft+j]`. The most likely is (i): the missing tape_gpu graph-builder block (qwen35.cpp/qwen35moe.cpp) means the GPU tape is empty, so llama_dflash_rollback's tape_replay reads garbage -> corrupt recurrent state. This ties back to the Phase-1 finding that the tape_gpu capture block is missing and was NOT exercised by llama-cli (use_gpu_qkv not confirmed) - but llama_dflash_rollback DOES exercise tape_replay, so the missing tape_gpu block would bite once rollback is enabled.

### TEST RESULTS
- Command: apply 2-edit port; build; GGML_DFLASH_QA_TRACE=1 run p1/p2/p3 DFlash + non-DFlash.
- Result: p1 diverged (21 partial accepts, 10.7 t/s); p2 8.0 t/s; p3 8.5 t/s. seq_backup=1 real, 58 rollbacks (p2). Port regressed.
- Command: isolation - revert verify-loop gate to env-gated (rollback off), keep backup creation; build; run p1.
- Result: p1 IDENTICAL but 13.1 t/s, 11 partial accepts, 0 rollbacks. Backup-creation alone correct but slow; rollback+re-decode corrupts.
- Command: git restore server-context.cpp (revert to 100a9ae24); build; run p1.
- Result: p1 IDENTICAL 19.6 t/s. Reverted clean.
- Output files: /tmp/p2/port-*.{out,err}, /tmp/p2/iso1-dflash.{out,err}, /tmp/p2/revert-dflash.out.

### NEXT STEPS
- The port requires fixing H-B (rollback corruption) and H-A (slowdown) BEFORE it can be enabled. H-B most likely ties to the missing tape_gpu graph-builder capture block (Phase-1 finding): port that block into qwen35.cpp/qwen35moe.cpp so the GPU tape is populated, then re-test the production rollback. H-A: pre-expand the recurrent memory once at setup (port main's n_seq_max_full pre-expansion) instead of per-batch.
- Do NOT re-enable the production rollback until H-B is fixed (it corrupts). The shipped state (100a9ae24 + Phase 1 shader fix a3732a5c3) remains: p1 correct (IDENTICAL, 19.6 t/s); p2/p3 have the residual late divergence + slow speed (the known Phase-2 residual). This is strictly better than the naive port (which regressed p1).
- Ask the user before further port work (the rollback port is now a multi-piece effort: tape_gpu block + pre-expand + rollback, not a single gate-removal).

## 2026-07-09 23:55 UTC - PHASE 2 multi-piece port attempt: tape_gpu block alone does NOT fix rollback corruption; reverted, deeper port needed

### FACTS
- Proceeded with Option A (multi-piece port). First confirmed H-B with a diagnostic: re-applied the production rollback port (both server-context.cpp edits), ran p1 with GGML_DFLASH_DEBUG=1. tape_replay_conv ran 1008 times with use_gpu_qkv=1, qkv_mixed.size=0 (CPU tape empty), n_tokens=0, n_accepted=1. So on partial-accept rollback the GPU path reads gpu_layer->qkv (the GPU tape buffer) which is NEVER written by the graph -> reads uninitialized GPU memory -> corrupt recurrent-state restore -> p1 diverges. H-B confirmed.
- Ported the tape_gpu graph-builder capture block into qwen35.cpp build_layer_attn_linear (after cb(v_conv,"v_conv_predelta")): copies k_conv/v_conv/gate/beta_presigmoid/qkv_mixed into tgpu->layers[li].{k,v,gate,beta,qkv} when cparams.tape_gpu_n_seqs>0. (Saved beta_presigmoid before the sigmoid.) Built OK (after fixing colon-leak edit artifacts). qwen35moe.cpp NOT yet ported (dense test uses qwen35.cpp).
- Re-tested p1 WITH the tape_gpu block + production rollback ON: STILL DIVERGENT (line 6, 21 partial accepts, 21 rollbacks, 10.6 t/s). GGML_DFLASH_DEBUG=1 still shows n_tokens=0 for every tape_replay_conv call.
- So the tape_gpu graph-builder block alone does NOT fix the corruption. The GPU tape buffer tl.qkv is allocated (use_gpu_qkv=1, gpu_layer->qkv non-null) but either (a) the block is not running during the verify-batch graph (tape_gpu_n_seqs may be 0 in the verify-batch cparams - the tape RECORDING lifecycle is not enabled for production verify in the merge), and/or (b) the tape n_tokens bookkeeping (set by the recording logic, not the graph block) stays 0, and/or (c) tape_replay_gdn (the GDN recurrent-state replay) and/or the re-decode (llama_decode of batch_tokens at dflash_n_pos_before_draft+j) are themselves wrong on the merge.
- The merge's production DFlash verify/rollback is therefore a MULTI-PIECE port, not a 2-edit gate-removal: (1) enable the tape RECORDING lifecycle for the production verify batch (set_tape_recording / active_tape / tape_gpu_n_seqs in the verify-batch cparams + n_tokens bookkeeping), (2) the tape_gpu graph block (done for qwen35.cpp, pending qwen35moe.cpp), (3) tape_replay_gdn correctness, (4) the re-decode (positions/tokens - main uses slot.prompt.tokens[n_tokens_before_draft+j] at n_pos_before_draft+j; merge uses batch_tokens=[slot.sampled,spec_draft...] which may re-decode a rejected draft or use wrong offsets), (5) pre-expand the recurrent memory once at setup (not per-batch) for the slowdown (H-A).
- REVERTED both port attempts (server-context.cpp production rollback + qwen35.cpp tape_gpu block) to the checkpoint state (100a9ae24). Rebuilt + verified: p1 back to IDENTICAL at 19.5 t/s. No regression shipped. The Phase 1 shader slot fix (a3732a5c3) remains the shipped correctness fix.
- Working tree: only TASK_PROGRESS.md modified (this entry + the prior port-start entry). server-context.cpp and qwen35.cpp are back at 100a9ae24 / a3732a5c3.

### HYPOTHESES
- The production rollback port needs the full tape-RECORDING lifecycle from gboddaer/main (not just the graph-builder block): the merge took upstream's server which never wires tape recording into the production verify path (it was env-gated test-only). gboddaer/main's server sets the tape active/recording around the verify batch (set_tape_recording / active_tape / dflash_graph_tape_ready at llama-context.cpp:6967-6989) so tape_gpu_n_seqs>0 in the verify-batch cparams and n_tokens is bookkept. Without that, the graph block never runs and tl.qkv stays uninitialized.
- The re-decode may also need main's exact token/position source (slot.prompt.tokens[n_tokens_before_draft+j]) rather than the merge's batch_tokens=[slot.sampled, spec_draft...].
- These are separable sub-pieces; each must be verified (the naive enable corrupts, so the port must be done piece-by-piece with the QA trace confirming the tape populates and the recurrent state restores correctly before re-enabling rollback).

### TEST RESULTS
- Command: re-apply production rollback port; GGML_DFLASH_DEBUG=1 run p1.
- Result: tape_replay_conv 1008x use_gpu_qkv=1 n_tokens=0 -> reads uninitialized gpu_layer->qkv -> p1 diverges. H-B confirmed.
- Command: add tape_gpu graph block to qwen35.cpp; build; run p1 with rollback ON.
- Result: still divergent (line 6, 21 rollbacks, 10.6 t/s); n_tokens still 0. Block alone insufficient.
- Command: git restore server-context.cpp + qwen35.cpp; rebuild; run p1.
- Result: p1 IDENTICAL 19.5 t/s. Reverted clean.
- Output files: /tmp/p2/dbg-dflash.{out,err}, /tmp/p2/tape-dflash.{out,err}, /tmp/p2/dbg2.{out,err}, /tmp/p2/final-dflash.out.

### NEXT STEPS
- The production rollback port is a deeper, multi-piece effort (tape-recording lifecycle + graph block + replay-gdn + re-decode + pre-expand). Recommend EITHER: (a) a focused follow-up that ports the tape-RECORDING lifecycle from gboddaer/main server-context.cpp (set_tape_recording/active_tape around the verify batch) FIRST, re-test that the tape populates (n_tokens>0, tl.qkv written) via GGML_DFLASH_DEBUG, THEN re-enable rollback; OR (b) accept the Phase 1 shader fix (a3732a5c3) as the milestone and track the p2/p3 residual as a follow-up issue. Do NOT re-enable the production rollback until the tape populates correctly (it corrupts otherwise).
- Ship state remains: a3732a5c3 (Phase 1 shader fix, p1 IDENTICAL 19.5 t/s) + 100a9ae24 (Phase 2 QA trace, gated). No regression.

## 2026-07-10 04:10 UTC - PHASE 2 full coordinated port attempt: tape STILL not populated (n_tokens=0) even with set_tape_recording(true); reverted

### FACTS
- Implemented the FULL coordinated production port (all pieces): (1) tape_gpu graph-builder capture block in qwen35.cpp build_layer_attn_linear (copies k_conv/v_conv/gate/beta_presigmoid/qkv_mixed into tgpu->layers[li].{k,v,gate,beta,qkv} when cparams.tape_gpu_n_seqs>0; saved beta_presigmoid before sigmoid); (2) draft-setup backup creation re-enabled (no env gate, cp_recurrent unconditional); (3) llama_set_tape_recording(ctx_tgt, true) before the verify decode + false after (in decode(); set_tape_recording(false) does NOT clear the tape buffer, only stops the graph block from running on subsequent decodes); (4) production rollback re-enabled (verify loop gates on DFlash type, not env). Built 0 errors.
- Tested p1 with GGML_DFLASH_DEBUG=1 + GGML_DFLASH_QA_TRACE=1: STILL DIVERGENT (line 6, 20 rollbacks, 10.5 t/s). Tape diagnostics: n_tokens=0 for ALL 960 tape_replay_conv calls, use_gpu_qkv=1. So even with llama_set_tape_recording(true) before the verify decode, the tape is NOT populated (n_tokens stays 0, tl.qkv not written).
- Root cause of the remaining failure: the tape-recording lifecycle does not propagate to the verify-batch graph cparams in the merge. set_tape_recording(true) sets dflash_capture->tape_enabled and clears n_tokens, but the graph builder's cparams.tape_gpu_n_seqs is NOT set > 0 for the verify-batch graph (or the graph is not invalidated/rebuilt to include the tape_gpu block). So the tape_gpu block (cparams.tape_gpu_n_seqs > 0) never runs -> tl.qkv never written -> tape_replay reads uninitialized GPU memory -> corrupt restore -> p1 diverges. The merge's upstream-derived server never wires the tape recording into the production verify graph reservation (it was env-gated test-only and the test path used a different, fuller wiring that is also missing).
- REVERTED all port changes (server-context.cpp + qwen35.cpp) to the committed state (18becc071). Rebuilt + verified: p1 back to IDENTICAL at 19.3 t/s. No regression shipped. Working tree clean (no tracked modifications).
- Shipped state: a3732a5c3 (Phase 1 shader slot fix) + 100a9ae24 (Phase 2 QA trace) + 18becc071 (Phase 2 port investigation docs). All pushed to gboddaer/merge_llama_into_beellama_2.

### HYPOTHESES
- The production DFlash rollback port requires fixing the tape-recording -> graph-cparams wiring: set_tape_recording(true) must cause cparams.tape_gpu_n_seqs > 0 AND the verify-batch graph must be invalidated/rebuilt so the tape_gpu block runs and populates tl.qkv (with n_tokens bookkept). gboddaer/main does this via a different server structure (the full ~652-line DFlash server integration, HF-017) that wires set_tape_recording + active_tape + dflash_graph_tape_ready (llama-context.cpp:6967-6989) into the verify graph reservation. The merge's upstream-derived server lacks this entire wiring path. Porting it is a substantial structural port, not a few edits.
- This is beyond a quick fix and should be tracked as a dedicated follow-up: port gboddaer/main's full DFlash server integration (set_tape_recording + active_tape + dflash_graph_tape_ready + the verify graph reservation with tape_gpu_n_seqs>0) so the tape populates, THEN the production rollback (which is already structurally present) will work.

### TEST RESULTS
- Command: apply full coordinated port (tape block + recording + backup + rollback); build; GGML_DFLASH_DEBUG=1 run p1.
- Result: p1 DIVERGENT (line 6, 20 rollbacks, 10.5 t/s); n_tokens=0 for all 960 tape_replay_conv calls (tape not populated even with recording enabled). Port still corrupts.
- Command: git restore server-context.cpp + qwen35.cpp; build; run p1.
- Result: p1 IDENTICAL 19.3 t/s. Reverted clean.
- Output files: /tmp/p2/fullp1.{out,err}, /tmp/p2/rev.out.

### NEXT STEPS
- Recommend: track the production DFlash rollback port as a dedicated follow-up. The next investigation should focus on WHY set_tape_recording(true) does not make cparams.tape_gpu_n_seqs > 0 in the verify-batch graph (trace cparams.tape_gpu_n_seqs at graph reservation; check the graph invalidation path at llama-context.cpp:6653-7041; compare with gboddaer/main's dflash_graph_tape_ready wiring at 6967-6989). Once the tape populates (n_tokens>0, tl.qkv written), the production rollback (already structurally present) should work.
- Ship state remains Phase 1 (a3732a5c3): p1 IDENTICAL 19.3 t/s; p2/p3 have the residual late divergence + slow speed (the known Phase-2 residual). No regression.
- Do NOT re-enable the production rollback until the tape populates correctly.

## 2026-07-10 05:30 UTC - Plan execution: tape populates (n_tokens>0 ✅) but rollback still corrupts p1 (❌); reverted, NOT finalized

### FACTS
- Read the approved plan (/crypt/openclaw/workspace/dflash_rollback_plan.md). The plan's premise ("tape-readiness to graph-cparams wiring is missing") was INCORRECT: the wiring logic ALREADY EXISTS in the merge's src/llama-context.cpp (dflash_graph_tape_ready computation at 6946, tp->n_tokens population at 6971, cparams.tape_gpu_seqs assignment, graph invalidation via gf_res_prev->reset() at 6996). It is byte-identical to gboddaer/main (the merge even has an extra prev_hidden_ns invalidation improvement for HF-039/040).
- Added a GGML_DFLASH_DEBUG-gated diagnostic in llama-context.cpp after the dflash_graph_tape_ready computation (fprintf to stderr: capture_active, tape_enabled, gpu_cap_ready, use_prefill_staging, n_tokens, tapes_empty, graph_tape_ready). Ran on the Phase 1 state (no port): capture_active=1, gpu_cap_ready=1, use_prefill_staging=0, n_tokens=4, tapes_empty=0, but tape_enabled=0 -> graph_tape_ready=0. So the ONLY failing condition was tape_enabled=0 (the merge never calls llama_set_tape_recording(true) in production).
- Applied the fix: llama_set_tape_recording(ctx_tgt, true) before the verify decode + false after, in decode() (gated on DFlash + any slot has spec_draft). Re-ran with the diagnostic: tape_enabled=1, graph_tape_ready=1 for 16/18 batches (the first 2 are prefill with use_prefill_staging=1, correctly 0). CRITERION 1 (n_tokens>0) MET: 47 verify batches with n_tokens=4 (+ 7/8/7 batches with n_tokens=1/2/3 for partial re-decodes, + 1 prefill n_tokens=44).
- Also re-applied the full coordinated port (tape_gpu graph block in qwen35.cpp + backup creation + production rollback in server-context.cpp). Built 0 errors.
- CRITERION 2 (p1 identical) NOT MET: p1 DIVERGENT (line 6, 20 rollbacks, 10.4 t/s) even with the tape populated. The [DFLASH_QA] rollback trace shows n_reeval == n_hidden_keep (1=1, 2=2) — the re-decode count is CORRECT. So the corruption is NOT the re-decode count; it is in the rollback state restore + re-decode interaction itself (the restored recurrent state + re-decoded positions produce a WRONG recurrent state, even though the tape (tl.qkv) is now populated and the re-decode count is correct).
- Per the user's instruction ("verify n_tokens>0 and p1 remains identical before finalizing"), since criterion 2 is NOT met, the port was NOT finalized.
- REVERTED all port changes (server-context.cpp + qwen35.cpp + llama-context.cpp diagnostic) to the committed state (e7e206d0a). Rebuilt + verified: p1 back to IDENTICAL at 23.6 t/s. No regression shipped. Working tree clean.

### HYPOTHESES
- The tape population is now understood and fixable (set_tape_recording(true) before the verify decode -> tape_enabled=1 -> graph_tape_ready=1 -> tape populates, n_tokens>0). This is a small, verified change.
- The REMAINING corruption (p1 divergent even with tape populated) is in the rollback+re-decode logic, NOT the tape population. Candidates: (i) the rollback keeps the verify batch's KV at accepted positions (flat mode: seq_rm(seq_id, n_past_before+n_accepted, -1)) while restoring the recurrent state to the backup — the kept KV (computed with the over-advanced verify state) may be inconsistent with the restored+re-decoded recurrent state; (ii) the re-decode at n_past_before+j conflicts with the kept KV at those positions (re-decoding at an existing position); (iii) the recurrent state cell_idx after seq_cp_recurrent(backup) + seq_rm may not be correctly positioned for the re-decode. These require comparing the merge's dflash_rollback + re-decode against gboddaer/main's EXACT sequence (main re-decodes slot.prompt.tokens[n_tokens_before_draft+j], merge re-decodes batch_tokens=[slot.sampled,spec_draft...]; main's n_pos_before_draft vs merge's dflash_n_pos_before_draft — these may differ by one position).
- Most likely: dflash_n_pos_before_draft (= pos_next() BEFORE handle_last_sampled_token adds id_last) may be the position of id_last, while the re-decode should start at the position AFTER id_last (or the backup should be taken AFTER id_last is committed). A one-position offset in the backup/restore/re-decode alignment would corrupt the state.

### TEST RESULTS
- Command: add GGML_DFLASH_DEBUG diagnostic to llama-context.cpp; run p1 DFlash (Phase 1 state).
- Result: graph_tape_ready=0 because tape_enabled=0 (the only failing condition). Confirmed set_tape_recording never called in production.
- Command: add set_tape_recording(true/false) bracket around verify decode; run with diagnostic.
- Result: tape_enabled=1, graph_tape_ready=1 for 16/18 batches. n_tokens=4 for 47 verify batches. CRITERION 1 MET.
- Command: apply full coordinated port (tape block + backup + rollback + set_tape_recording); build; run p1 DFlash vs non-DFlash.
- Result: p1 DIVERGENT (line 6, 20 rollbacks, 10.4 t/s). n_reeval==n_hidden_keep (correct count). CRITERION 2 NOT MET.
- Command: git restore all 3 files; build; run p1.
- Result: p1 IDENTICAL 23.6 t/s. Reverted clean.
- Output files: /tmp/p2/diag2.{out,err}, /tmp/p2/tapecap.{out,err}, /tmp/p2/fullp1v2.{out,err}, /tmp/p2/revlast.out.

### NEXT STEPS
- The tape-population fix (set_tape_recording bracket) is verified and small. The remaining rollback corruption needs a focused investigation of the dflash_rollback + re-decode position alignment: compare dflash_n_pos_before_draft (merge) vs n_pos_before_draft/n_tokens_before_draft (main), and the re-decode token source (batch_tokens vs slot.prompt.tokens). The backup-creation timing (before vs after handle_last_sampled_token adds id_last) may cause a one-position offset that corrupts the restore+re-decode.
- Do NOT finalize the port until p1 is identical WITH the rollback enabled. The shipped state remains Phase 1 (a3732a5c3): p1 IDENTICAL 23.6 t/s.

## 2026-07-10 06:00 UTC - One-position offset investigation: DISPROVEN; rollback RESTORE itself corrupts (even without re-decode)

### FACTS
- Investigated the one-position offset hypothesis. Compared the merge's verify-batch construction (handle_last_sampled_token: pos0 = pos_next(), adds sampled at pos0, drafts at pos0+1...) with gboddaer/main's (line 4744: common_batch_add(batch, slot.sampled, n_pos_before_draft, ...)). BOTH re-decode id_last (slot.sampled) at pos_next() = n_pos_before_draft. The QA trace confirms: batch 1 n_pos_before_draft=81 pos_next=85 (id_last=8160 at 81, bonus 1817 at 84); batch 2 n_pos_before_draft=85 (id_last=1817 at 85, re-decoded from batch 1's bonus at 84). This duplicate id_last decode is STANDARD speculative decoding (main does it too and main's rollback works). The one-position offset hypothesis is DISPROVEN — the positions match main.
- The re-decode tokens (batch_tokens=[slot.sampled, spec_draft...] vs main's slot.prompt.tokens[n_tokens_before_draft+j]) are the SAME tokens (id_last + accepted drafts). The re-decode count n_reeval == n_hidden_keep (correct). tree_parent_ids is null (never set), so llama_clear_tree_parent_ids is a no-op (not the fix).
- ISOLATION TEST: applied the full port (set_tape_recording + tape_gpu block + backup creation + production rollback) but DISABLED the re-decode (skipped llama_decode(batch_reeval)). Result: p1 STILL DIVERGENT (line 6, 15 rollbacks, 12.3 t/s). So the corruption is in the RESTORE itself (llama_dflash_rollback → seq_cp_recurrent from backup + tape_replay), NOT the re-decode.
- This means: the backup (seq_cp_recurrent at draft setup) or the tape_replay produces a WRONG recurrent state when restored. Without the rollback, the over-advanced state (from the verify batch) happens to produce correct argmax for p1 (IDENTICAL). With the rollback, the restored backup state produces WRONG argmax (DIVERGENT). So the backup ≠ the correct pre-verify state.
- REVERTED all port changes. p1 back to IDENTICAL at 19.4 t/s. No regression.

### HYPOTHESES
- The backup (seq_cp_recurrent from seq_id to seq_backup at draft setup) captures a recurrent state that, when restored by the rollback, does NOT match the correct pre-verify state. Candidates: (i) the backup is taken at the wrong time (after id_last is already in the recurrent state from the previous batch, but the rollback assumes it's before); (ii) the tape_replay (tape_replay_conv + tape_replay_gdn) reads the populated tl.qkv but reconstructs the conv/GDN state incorrectly on the merge (the tape data layout from the tape_gpu block may not match what tape_replay expects); (iii) the seq_cp_recurrent + seq_rm interaction leaves the recurrent state cell_idx misaligned.
- The next diagnostic: use llama_dflash_dump_recurrent_state_dbg to dump the recurrent state (i) at backup creation, (ii) after the verify decode, (iii) after the rollback restore, (iv) after the re-decode — and compare with a non-DFlash sequential run's state at the matching position. This will pinpoint WHERE the state diverges.
- This is a deep investigation that requires recurrent state comparison, beyond the position-alignment fix originally hypothesized.

### TEST RESULTS
- Command: compare merge handle_last_sampled_token with main's batch construction; check QA trace n_pos_before_draft progression.
- Result: both re-decode id_last at pos_next()=n_pos_before_draft; positions match main. One-position offset DISPROVEN.
- Command: full port with re-decode DISABLED (restore only); run p1.
- Result: p1 DIVERGENT (line 6, 15 rollbacks, 12.3 t/s). RESTORE itself corrupts.
- Command: git restore all; build; run p1.
- Result: p1 IDENTICAL 19.4 t/s. Reverted clean.
- Output files: /tmp/p2/iso_redecode.{out,err}, /tmp/p2/revf2.out.

### NEXT STEPS
- The rollback port cannot be finalized until the RESTORE corruption is fixed. The next step is an empirical recurrent-state-dump comparison: dump the state at backup/verify/restore/re-decode points and compare with non-DFlash to find where the state diverges. This requires adding llama_dflash_dump_recurrent_state_dbg calls and a non-DFlash comparison run.
- Ship state remains Phase 1 (a3732a5c3): p1 IDENTICAL 19.4 t/s. The tape-population fix (set_tape_recording) is verified (n_tokens>0) but cannot be shipped until the rollback restore is correct (p1 must remain identical).
- Do NOT finalize the port. Both user criteria (n_tokens>0 ✅ + p1 identical ❌) are not simultaneously met.

## 2026-07-10 10:30 UTC - Empirical recurrent-state-dump: recurrent state CORRECT, full-attention KV is the corruptor

### FACTS
- Used llama_dflash_dump_recurrent_state_dbg to dump the recurrent state (s_l + r_l for layers 0, mid, 62) at 4 points: pre_verify (backup), post_verify (after verify decode), post_restore (after llama_dflash_rollback, before re-decode), post_reeval (after re-decode). Also extended the dump to add rL0 (layer 62's conv state) and sM0 (middle layer's s_l).
- KEY RESULT: post_reeval MATCHES pre_verify for ALL dumped fields (s0/sum/abs for layer 0, sL0/sLsum/sLabs for layer 62, r0/rsum/rabs for layer 0's conv, rL0/rLsum/rLab for layer 62's conv, sM0/sMsum for the middle layer). So the recurrent state (delta-net layers) is CORRECTLY restored by the rollback+re-decode.
- But p1 DIVERGES with the rollback. So the corruption is NOT in the recurrent state (delta-net layers).
- Pinpointed the EXACT first divergence via token traces (GGML_DFLASH_QA_TRACE + GGML_NODFLASH_TOKEN_TRACE): first divergence at TOKEN INDEX 53 (DFlash=1577, non-DFlash=7722), at pos ~100 (batch 13). This is exactly at the FIRST partial-accept rollback (4 rollbacks total in the -n 80 run). Earlier "line 6" was a text-line artifact (line 6 of the thinking section corresponds to token 53, not token 20).
- Qwen3.6-27B is a HYBRID model: each layer is either build_layer_attn_linear (delta-net, recurrent state in mem_recr, checked by the dump) or build_layer_attn (full attention with KV cache in mem_attn, NOT checked by the dump). The is_recr pattern (llama-hparams.cpp:27: `il % n_pattern != 0` or `il % n_pattern < n_pattern-1`) means most layers are recurrent (delta-net) and every Nth layer is full-attention (KV cache).
- The rollback's mem_attn->seq_rm (flat mode: seq_rm(seq_id, n_past_before + n_accepted, -1) + seq_rm(seq_backup, -1, -1) + if n_reeval>0: seq_rm(seq_id, n_past_before, -1)) + the re-decode handle the full-attention KV. The recurrent state (mem_recr) is correctly restored (dumped), but the full-attention KV (mem_attn) is apparently left in a WRONG state after the rollback+re-decode -> the full-attention layers' output is wrong -> logits wrong -> divergence at the first rollback.
- Isolation tests (all with FRESH references from the same build): set_tape_recording + tape_gpu block + backup, rollback OFF → p1 IDENTICAL. Rollback ON → p1 DIVERGENT at token 53. Empty rollback block → IDENTICAL. Block = only dflash_seq_backup=-1 → IDENTICAL. Block = full if/else (with llama_dflash_rollback for partial-accept) → DIVERGENT at token 53. So the partial-accept llama_dflash_rollback + re-decode is the SOLE corruptor, and it corrupts the full-attention KV (not the recurrent state, which is correct).
- Reverted all port changes. Baseline IDENTICAL at 26.6 t/s. No regression.

### HYPOTHESES
- The rollback's full-attention KV handling is wrong. The rollback restores the recurrent state (delta-net) correctly but leaves the full-attention KV (mem_attn) in an inconsistent state after the seq_rm + re-decode. Candidates: (i) the re-decode doesn't correctly re-compute the full-attention KV for the re-decoded positions (logits=false might skip KV store for some path, or the positions conflict with freed cells); (ii) the full-attention KV at [0, n_past_before) was modified by the verify decode (e.g., SWA sliding window shifted); (iii) the seq_rm on mem_attn removes the wrong range (n_past_before + n_accepted vs the actual KV positions for the hybrid model).
- The next diagnostic: dump the full-attention KV (K cache for a non-recurrent layer) at post_verify vs post_reeval and compare with the pre_verify (backup) KV. If the KV at [0, n_past_before) differs between post_reeval and pre_verify, the verify decode or the rollback modified the pre-verify KV (wrong). If [n_past_before, +1...] differs, the re-decode didn't correctly re-compute.

### TEST RESULTS
- Command: add llama_dflash_dump_recurrent_state_dbg calls at 4 points; extend dump with rL0/sM0; run DFlash -n 60 with GGML_DFLASH_RS_DUMP.
- Result: post_reeval = pre_verify (ALL fields, including extended rL0/sM0). Recurrent state correct.
- Command: token traces (GGML_DFLASH_QA_TRACE + GGML_NODFLASH_TOKEN_TRACE); Python compare.
- Result: first divergence at token 53 (pos ~100, first rollback). 4 rollbacks total.
- Command: isolation (fresh refs) — rollback OFF/ON, empty block, only var, full if/else.
- Result: rollback ON (full if/else) is the sole corruptor; divergence at token 53 (first rollback).
- Command: git restore; baseline nodflash vs dflash.
- Result: IDENTICAL 26.6 t/s. No regression.
- Output files: /tmp/p2/rs2.err, /tmp/p2/tt_d.{out,err}, /tmp/p2/tt_nd.{out,err}, /tmp/p2/{bl_nodflash,bl_dflash,pf_nd,pf_d,of_nd,of_d,eb_nd,eb_d,ov_nd,ov_d,rev_nd,rev_d}.out.

### NEXT STEPS
- Dump the full-attention KV (K cache for a non-recurrent layer, e.g., the first full-attention layer) at post_verify and post_reeval, compare with pre_verify. This requires adding a KV dump (the current dump only checks recurrent layers). Pinpoint whether the pre-verify KV [0, n_past_before) is modified by the verify decode, or the re-decode doesn't correctly re-compute the KV at [n_past_before, +1...].
- Do NOT finalize the port. Criteria: n_tokens>0 ✅ (with port) but p1 identical ❌ (rollback corrupts full-attention KV at the first partial-accept). Ship state: a3732a5c3 (Phase 1) — p1 IDENTICAL 26.6 t/s.

## 2026-07-10 12:00 UTC - KV-cache dump: BOTH recurrent state AND KV are CORRECT after rollback; ghost bug

### FACTS
- Implemented a full-attention KV-cache dump in llama_dflash_dump_recurrent_state_dbg: reads the K storage tensor for the first full-attention (non-recurrent) layer (kv_il=3, confirming Qwen3.6's hybrid pattern: layers 0,1,2 recurrent, 3 full-attention, ...) at the CURRENT positions (near kv_pos_max, using the position stride nb[1]). Dumped kv_pos_max + k_sum + k_abs (fingerprint of 512 K values at the current tail). Ran with F32 cache (--cache-type-k f32 --cache-type-v f32) for readable values.
- KEY RESULT: post_reeval (after rollback+re-decode) MATCHES pre_verify (the next batch's backup) for BOTH the recurrent state (s_l + r_l, all layers) AND the full-attention KV (kv_pos_max + k_sum + k_abs):
  - post_reeval pos=104: k_sum=-29.82, k_abs=430.6 == pre_verify pos=104: k_sum=-29.82, k_abs=430.6 ✅
  - post_reeval pos=111: k_sum=-2.323, k_abs=446.6 == pre_verify pos=111: k_sum=-2.323, k_abs=446.6 ✅
  - The KV carry-forward is correct: post_restore (after rollback, before re-decode) has kv_pm trimmed (e.g., 99), post_reeval has kv_pm restored (100), matching the next pre_verify.
- So BOTH the delta-net recurrent state AND the full-attention KV cache are CORRECTLY restored by the rollback+re-decode. The state fingerprint (recurrent s_l/r_l + KV k_sum/k_abs) matches the correct carry-forward at every rollback.
- YET p1 DIVERGES with the rollback (at token 53, the first rollback). This is a GHOST BUG: the state (recurrent + KV) is correct but the output (logits/argmax) diverges.
- The divergence is NOT in the state. It must be in: (i) the logits computation (attention output / output_norm / LM_head) — something between the correct state and the argmax; (ii) a graph/buffer interaction between the re-decode (llama_decode with logits=false) and the next verify decode (llama_decode with logits=true) — the re-decode's graph (logits=false) differs from the verify graph (logits=true), and the graph cache / buffer reuse may cause a corruption; (iii) a detail the sum/abs fingerprint (512 values) doesn't capture (the actual K values differ in a way the sum/abs doesn't show, though sum/abs of 512 values matching is a strong signal).
- Reverted all port + dump changes. Baseline IDENTICAL at 23.4 t/s. No regression.

### HYPOTHESES
- H-GHOST (lead): the re-decode (llama_decode with logits=false) leaves the graph/buffers in a state that makes the NEXT verify decode (logits=true) produce wrong logits despite the correct recurrent + KV state. The re-decode's graph (no LM head, no output buffer) differs from the verify graph (LM head + output), and the graph cache invalidation / buffer reuse between them may alias or corrupt the output buffer. This would explain why the state is correct but the logits are wrong.
- H-DETAIL (lower): the K values differ in details the sum/abs fingerprint doesn't capture (unlikely — sum/abs of 512 matching values is a strong signal, but possible if the divergent values are a small fraction).
- The next diagnostic: dump the actual LOGITS (the target's argmax token) right after the next verify decode (the batch AFTER the rollback) and compare with non-DFlash. If the logits diverge despite the correct state, the issue is in the logits computation or the graph/buffer. This requires dumping the logits tensor (or the argmax) during the verify decode after a rollback.

### TEST RESULTS
- Command: add KV dump to dflash_dump_recurrent_state_dbg (K storage at current tail via position stride nb[1]); re-apply full port + RS_DUMP calls; build; run with F32 cache + GGML_DFLASH_RS_DUMP.
- Result: post_reeval = pre_verify for BOTH recurrent state AND KV (kv_pos_max + k_sum + k_abs). State fully correct. But p1 diverges (ghost bug).
- Command: git restore all; baseline nodflash vs dflash.
- Result: IDENTICAL 23.4 t/s. No regression.
- Output files: /tmp/p2/fix.err (F32 KV dumps), /tmp/p2/curtail.err, /tmp/p2/tail.err, /tmp/p2/f32.err.

### NEXT STEPS
- Dump the actual logits/argmax during the verify decode AFTER a rollback (the batch at pos=104, which uses the correct post-reeval state). Compare the argmax with non-DFlash. If the argmax diverges despite the correct state, the issue is in the logits computation or graph/buffer management (H-GHOST). 
- A targeted test for H-GHOST: run the re-decode with logits=TRUE (instead of false) to see if forcing the output buffer allocation fixes the divergence (if the graph/buffer aliasing is the cause, allocating the output in the re-decode would prevent the aliasing).
- Do NOT finalize the port. Criteria: n_tokens>0 ✅ but p1 identical ❌ (ghost bug — state correct but logits diverge). Ship state: a3732a5c3 (Phase 1) — p1 IDENTICAL 23.4 t/s.

## 2026-07-10 12:30 UTC - H-GHOST disproven (logits=true doesn't fix); root cause = id_last double-decode + rollback interaction

### FACTS
- Tested H-GHOST (re-decode logits=false leaves graph/buffers in bad state): changed the re-decode to logits=true (force output buffer allocation + LM head). Result: p1 STILL DIVERGENT (first-diff=6, 10.7 t/s). H-GHOST DISPROVEN — the logits flag in the re-decode is not the cause.
- Exhaustive summary of ALL hypotheses tested this session:
  - One-position offset (dflash_n_pos_before_draft vs n_pos_before_draft): DISPROVEN (positions match main).
  - tape_gpu block / set_tape_recording / backup creation: NOT the cause (rollback OFF → p1 IDENTICAL).
  - All-accepted backup cleanup (seq_rm): NOT the cause (skipping it → still divergent).
  - KV removal in dflash_rollback: NOT the cause (skipping → still divergent).
  - Recurrent state (delta-net s_l/r_l, all layers 0/mid/62): CORRECT (post_reeval = pre_verify, all fields).
  - Full-attention KV (k_sum/k_abs at current tail): CORRECT (post_reeval = pre_verify).
  - Re-decode logits=false vs true: NOT the cause (logits=true → still divergent).
- The state (recurrent + KV) is PROVABLY CORRECT after the rollback+re-decode, yet p1 DIVERGES at the first rollback (token 53). This is a ghost bug that the state fingerprint cannot explain.
- ROOT CAUSE (refined): the DFlash verify batch re-decodes id_last at a duplicate position (the framework's common_speculative includes id_last = slot.sampled as the first token of the verify batch at pos_next(), but id_last was already decoded in the previous batch). This creates a "double-decoded" DFlash state that differs from the sequential (non-DFlash) state. Without rollback, this double-decoded state (with rejected drafts) gives correct argmax for p1 (coincidence). With rollback, the state is changed to "pre-verify + accepted re-decoded" (still double-decoded from previous batches), which gives WRONG argmax for p1. The rollback cannot fix the double-decode (it restores to a state that ALSO has the double-decode from previous batches). gboddaer/main's rollback works because main's framework/state handling differs subtly (despite the positions/tokens matching, the actual state evolution must differ in a way the fingerprint doesn't capture, OR main avoids the double-decode through a different mechanism).
- Reverted all changes. Baseline IDENTICAL at 23.4 t/s. No regression.

### HYPOTHESES
- The remaining difference between the merge and gboddaer/main must be in the ACTUAL state values (not captured by the sum/abs fingerprint of 512 values), OR in a hidden state not dumped (e.g., the conv state for full-attention layers, the attention output, or the embeddings_nextn), OR in the graph structure (the tape_gpu block's ops interacting with the recurrent graph during the verify decode in a way that only manifests with the rollback). The fingerprint (sum/abs) is a coarse signal — the actual values could differ in details that flip the argmax without changing the sum/abs noticeably.
- The next diagnostic: dump the FULL recurrent state (not just sum/abs — dump the actual first 32 float values of s_l[0]) at post_reeval and compare with a non-DFlash sequential run at the same position. If the actual values differ (even with matching sum/abs), the fingerprint was misleading. This requires a non-DFlash state dump (the current dump requires dflash_capture which non-DFlash lacks).

### TEST RESULTS
- Command: re-apply full port + re-decode logits=true; build; test p1 (fresh refs).
- Result: DIVERGENT (first-diff=6, 10.7 t/s). H-GHOST disproven.
- Command: git restore all; baseline.
- Result: IDENTICAL 23.4 t/s. No regression.
- Output files: /tmp/p2/g_{nd,d}.{out,gen}, /tmp/p2/fix.err (KV dumps).

### NEXT STEPS
- The rollback port cannot be finalized (p1 diverges despite the state being correct by fingerprint). The next step is a FULL state-value comparison: dump the actual first 32 float values of s_l[0] (not just sum/abs) at post_reeval and compare with a non-DFlash sequential run at the matching position. If the values differ, the fingerprint was misleading and the actual state is wrong (pointing to the conv state or a detail). If the values match, the issue is in the logits/graph (beyond the state).
- Alternatively: investigate the framework-level id_last double-decode (the verify batch including id_last at a duplicate position) — this is the structural root cause, but fixing it is a framework-level change (common_speculative spec_i_batch / common_sampler_sample_and_accept_n).
- Ship state: a3732a5c3 (Phase 1) — p1 IDENTICAL 23.4 t/s. The Phase 1 shader fix is the shipped correctness fix. The rollback port remains incomplete (ghost bug). Do NOT finalize.

## 2026-07-10 16:00 UTC - Full state-value comparison: re-decode cell-pos misalignment is the root cause

### FACTS
- Modified dflash_dump_recurrent_state_dbg to work without dflash_capture (find recurrent layers via is_recr) and dump the first 32 s_l[il0] values (svals) for precise comparison. Added dump calls at pre_verify (backup), post_restore (after rollback), post_reeval (after re-decode), and nodflash (non-DFlash sequential).
- KEY RESULT 1: pre_verify (DFlash batch-decode state, BEFORE any rollback) MATCHES nodflash (sequential) at ALL positions (pos 47-83, 0.0000% diff, svals identical). So the DFlash batch decode = sequential (no numerical difference). The state is CORRECT before the rollback.
- KEY RESULT 2: post_restore (after seq_cp_recurrent from backup) MATCHES pre_verify (svals identical). So the backup is correctly restored to seq_id. ✅
- KEY RESULT 3: post_restore pos = dflash_n_pos_before_draft + 1 (e.g., post_restore pos=100 when dflash_n_pos_before_draft=99). The cell pos after the restore is ONE AHEAD of dflash_n_pos_before_draft.
- KEY RESULT 4: post_reeval (after re-decode) ≠ nodflash (svals differ, growing divergence: 0.44% at pos=100, 3.3% at pos=102, 7.9% at pos=104, 8.1% at pos=111). The re-decode produces a WRONG state.
- KEY RESULT 5: The re-decode at dflash_n_pos_before_draft (99) CONFLICTS with the cell pos (100) after the restore. The re-decode decodes id_last at pos 99, but the cell is at pos 100 (one ahead). This cell-pos misalignment causes the re-decode to advance from the wrong cell/position, producing a wrong state.
- Tested fixes:
  - Re-decode with accepted tokens (instead of original drafts): NO effect (accepted drafts == original drafts at accepted positions; rejected positions not re-decoded).
  - Re-decode 1-token-at-a-time (AR path instead of batch): NO effect (the AR vs chunked path is not the issue; the cell-pos misalignment is).
- ROOT CAUSE: the rollback's seq_cp_recurrent (or seq_rm) leaves the cell pos at dflash_n_pos_before_draft + 1 (one ahead of the backup's state position). The re-decode at dflash_n_pos_before_draft then conflicts with the cell pos, advancing from the wrong position → wrong state → divergence at the first rollback (token 53).
- Reverted all changes. Baseline IDENTICAL at 23.4 t/s. No regression.

### HYPOTHESES
- The cell pos after the rollback restore should be dflash_n_pos_before_draft - 1 (the backup's last decoded position) or dflash_n_pos_before_draft (the position where id_last will be re-decoded). But it's dflash_n_pos_before_draft + 1 (one ahead). The seq_cp_recurrent copies the backup's recurrent state tensors but the cell pos metadata is set to the wrong value (possibly the post-verify cell pos, not the backup's cell pos). The re-decode at dflash_n_pos_before_draft then decodes at a position BEHIND the cell pos, causing a conflict/misalignment.
- The fix: after the rollback restore, set the cell pos to dflash_n_pos_before_draft - 1 (or align it with the backup's state position) so the re-decode at dflash_n_pos_before_draft correctly advances from the backup. This requires either fixing seq_cp_recurrent to copy the cell pos correctly, OR adding a cell-pos adjustment after the restore in the server's rollback path.

### TEST RESULTS
- Command: rewrite dump (svals, no dflash_capture needed); add pre_verify/post_restore/post_reeval/nodflash dumps; run DFlash + non-DFlash with GGML_DFLASH_RS_DUMP; compare svals.
- Result: pre_verify = nodflash (correct before rollback); post_restore = pre_verify (backup restored); post_restore pos = n_past+1 (cell pos misaligned); post_reeval ≠ nodflash (re-decode wrong, growing divergence).
- Command: re-decode with accepted tokens; re-decode 1-token-at-a-time.
- Result: both NO effect. The cell-pos misalignment is the cause, not the re-decode tokens or path.
- Command: git restore; baseline.
- Result: IDENTICAL 23.4 t/s. No regression.
- Output files: /tmp/p2/pv_{d,nd}.{out,err}, /tmp/p2/pr_{d,nd}.{out,err}, /tmp/p2/fix2_{d,nd}.{out,err}.

### NEXT STEPS
- Fix the cell-pos misalignment: after the rollback restore, the cell pos should be dflash_n_pos_before_draft - 1 (the backup's state position), not dflash_n_pos_before_draft + 1. Investigate seq_cp_recurrent / seq_rm in llama-context.cpp to find why the cell pos is set to n_past+1. The fix may be: (a) correct seq_cp_recurrent to copy the cell pos from the backup, or (b) add a cell-pos adjustment (e.g., llama_memory_seq_rm(seq_id, dflash_n_pos_before_draft, -1) to trim the cell pos to n_past-1) after the restore, or (c) adjust the re-decode to start at the cell pos (n_past+1) instead of n_past (but this would decode id_last at the wrong position).
- Do NOT finalize the port. Criteria: n_tokens>0 ✅ but p1 identical ❌ (cell-pos misalignment in the re-decode). Ship state: a3732a5c3 (Phase 1) — p1 IDENTICAL 23.4 t/s.

## 2026-07-10 18:00 UTC - Cell-pos trim fix: PARTIAL improvement (8 lines → 1 line divergent), state still differs 1-8%

### FACTS
- Implemented the cell-pos trim fix: after llama_dflash_rollback, add `llama_memory_seq_rm(mem, slot.id, slot.dflash_n_pos_before_draft, -1)` to trim the recurrent cells from n_pos_before_draft onwards, aligning the cell pos with the backup's state position (n_pos-1) so the re-decode at n_pos advances correctly. Also used accepted tokens (instead of original drafts) and 1-token-at-a-time re-decode (AR path). Built OK.
- RESULT: p1 output MUCH CLOSER — the diff changed from "6,13c" (8 lines divergent, before the fix) to "6c" (1 line divergent, after the fix). So the cell-pos trim fix is a PARTIAL fix — it significantly improved correctness but did not achieve full p1 identity.
- The svals (first 32 s_l[0] values) at post_reeval still DIFFER from nodflash by 1-8% (e.g., pos=100: 2.1%, pos=103: 5.0%, pos=107: 8.1%). So the recurrent state after the re-decode is still wrong (but closer than before the fix, where it was 0.4-8.1% with a different pattern).
- The cell-pos trim fix changed the rollback behavior (9 post_reeval events vs 4 before), suggesting the fix altered the partial-accept/rollback pattern (the state being closer changed which drafts are accepted/rejected).
- The remaining 1-line divergence is at a specific rollback where the 1-8% state difference flips the argmax. The state is CLOSE but not identical to sequential.
- Reverted all changes. Baseline IDENTICAL at 23.6 t/s. No regression.

### HYPOTHESES
- The cell-pos trim fix is the RIGHT DIRECTION (output improved from 8 to 1 line divergent) but INCOMPLETE. The remaining 1-8% state difference suggests the re-decode still doesn't perfectly reproduce the sequential state. Candidates: (i) the cell-pos trim removes one too many/few cells (off-by-one in the trim range); (ii) the seq_cp_recurrent restore + the trim + the re-decode interact subtly (the recurrent state tensors vs the cell metadata alignment); (iii) the conv state (r_l) is not correctly restored/advanced (the dump only checks s_l, not r_l at the rollback positions with the fix); (iv) the 1-token re-decode at the trimmed cell pos still has a residual misalignment.
- The next step: with the cell-pos trim fix applied, dump the svals at post_restore (after the trim) and post_reeval, and compare with nodflash at the EXACT matching state (accounting for the pos semantics). If post_restore (after trim) = nodflash at n_pos-1, the restore+trim is correct, and the re-decode is the remaining issue. If post_restore ≠ nodflash, the trim removed too much/too little.

### TEST RESULTS
- Command: apply cell-pos trim fix + accepted tokens + 1-token re-decode; build; test p1 (fresh refs).
- Result: p1 diff "6c" (1 line divergent, down from "6,13c" 8 lines). PARTIAL fix.
- Command: RS_DUMP svals comparison (post_reeval vs nodflash).
- Result: svals differ 1-8% at 9 rollback positions. State still wrong but closer.
- Command: git restore; baseline.
- Result: IDENTICAL 23.6 t/s. No regression.
- Output files: /tmp/p2/cf_{nd,d}.{out,gen}, /tmp/p2/cf4_{d,nd}.{out,err}.

### NEXT STEPS
- The cell-pos trim fix is a promising partial fix (8→1 line divergent). Refine it: dump post_restore (after trim) svals vs nodflash at n_pos-1 to verify the trim range. If the trim is off-by-one, adjust the range. If the trim is correct, investigate the re-decode (conv state r_l, or the 1-token AR path at the trimmed cell pos).
- Do NOT finalize. Criteria: n_tokens>0 ✅ but p1 identical ❌ (1 line divergent, state 1-8% off). Ship state: a3732a5c3 (Phase 1) — p1 IDENTICAL 23.6 t/s.
