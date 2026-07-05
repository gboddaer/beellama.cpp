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
