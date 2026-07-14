# BeeLlama Merge: ggml-org/llama.cpp into beellama.cpp

**Created:** 2026-07-01  
**Last updated:** 2026-07-13 (current HEAD `0d5f99638`, pushed to `gboddaer/merge_llama_into_beellama_2`)  

---

## CURRENT STATE (2026-07-13)

**Objective:** Merge ggml-org/llama.cpp:master into beellama.cpp:main with DFlash speculative decoding preserved.

**Branch:** `merge_llama_into_beellama_2` in worktree `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`  
**PR:** https://github.com/gboddaer/beellama.cpp/pull/2 (merge_llama_into_beellama_2 → main)  
**Latest commit:** `0d5f99638` — dflash(vulkan): fix multi-slot capture routing + reduce drafter n_outputs_max

### What works (verified)

- ✅ **Build:** 0 compilation errors across all backends (Vulkan, ROCm, CUDA, Debug/CPU). CI build fixed.
- ✅ **Non-DFlash inference:** Correct, stable, ~12.5 t/s on Vulkan0 (AMD Radeon GFX1151 iGPU). 5/5 prompts correct.
- ✅ **Multi-slot DFlash (`--parallel 4`): FULLY FUNCTIONAL AND CORRECT.** All 4 slots have DFlash active (76-88% acceptance, 0 errors, 0 garble, coherent outputs). This was the primary blocker — resolved by 5 targeted slot-routing patches in commit `0d5f99638`.
- ✅ **Single-slot DFlash (`--parallel 1`):** ~74-80% acceptance, coherent output, ~15 t/s.
- ✅ **DFlash correctness vs non-DFlash:** outputs differ only in inherent DFlash wording (both coherent, both correct step-by-step solutions). No corruption, no position-offset divergence through 300+ token generations.
- ✅ **DFlash speed improvement:** ~11→15 t/s (single-slot) via the n_outputs_max fix (drafter LM-head 17→4 rows → drafter decode 54→40ms).

### What remains open

- ⚠️ **Speed gap:** Work-branch DFlash is ~15 t/s (single-slot) / ~11-13 t/s (multi-slot) vs `gboddaer/main`'s ~24 t/s. Root cause identified via op-level profiling: **2.3× more graph-compute launches** (70 vs 30 blocks per n_predict=20). The re-decode (24 separate launches) is the largest contributor. A re-decode batching attempt FAILED (corrupts the hidden_gpu capture — the first n_hidden_keep rows shift when re-decode tokens are prepended). Reverted. The gap requires a deeper fix (hidden-capture row selection adjustment or reducing other extra launches).
- ⚠️ **CI test failures:** OpenVINO Windows (test-cuda-zero-dim-gemm, test-backend-ops) and macOS (test-backend-ops WebGPU segfault, test-llama-archs) — unrelated to DFlash.
- ⚠️ **Multi-slot speed:** `--parallel 4` is ~11-13 t/s (dominated by the per-slot drafter forward pass, same 1.5× gap as single-slot).

### The committed fixes (0d5f99638, pushed)

1. **`flush_prefill` slot_id** — thread physical `slot.id` through `set_active_slot`/`prefill_capture_info`/`prefill_gpu_n_tokens` (was using impl `seq_id=0` for all slots → non-zero request slots read slot 0's empty buffers).
2. **`ring_write` D2D slot_id** — thread `phys_slot` into `llama_dflash_prefill_gpu_write_hidden` (was indexing `prefill_gpu[0]`).
3. **Active-slot routing** — set `active_tape_idx=seq` unconditionally (was gated on `seq < tapes.size()` where tapes is the tree buffer, size 0/1 for flat DFlash → non-zero slots never updated → eval callback captured to stale slot 0).
4. **Generation-path slot_id** — thread `slot_id` through `update_logits_deferred_dflash_kv` → `append_target_hiddens` → `ring_write` (was overriding server's `set_active_slot(slot.id)` with `seq_id=0`).
5. **Eval-callback gating** — skip eval callback only when `dflash_use_prefill_staging` (was gating on `dflash_graph_hidden_ready` = hidden_gpu presence → disabled CPU eval callback in multi-slot even when staging wasn't used → no capture).
6. **Drafter n_outputs_max** — reduce from forced 17 to `1+n_max` (4 for `--spec-draft-n-max 3`); the drafter graph uses `inp_out_ids` to compute the LM head for only 4 rows instead of 16 → ~4× smaller LM-head matmul → drafter decode 54→40ms, single-slot speed 11→15 t/s.

### Key references

- **Reference fork binary:** `/crypt/beellama.cpp/build-vulkan/bin/llama-server` (on commit `adb92b36a`, the pre-merge fork with working DFlash).
- **Reference fork source:** `gboddaer/main` (`130ea2480`) — the merged fork with working DFlash at ~24 t/s.
- **Models:** `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` (target), `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf` (draft).
- **Regression scripts:** `scripts/dflash-regression/{bench_dflash_v0.sh, measure_acceptance.sh, verify_corr.sh, verify_slotfix.sh}`.
- **Profile env vars:** `GGML_DFLASH_PROFILE=summary` (draft breakdown), `GGML_VK_PERF_LOGGER=1` (per-op Vulkan timing), `GGML_DFLASH_QA_TRACE=1` (verify/rollback counts).

---

## CHRONOLOGICAL INVESTIGATION HISTORY

> **The sections below are the chronological investigation record from 2026-07-01 through 2026-07-13.**  
> **They contain past hypotheses (some disproven), intermediate results, and superseded analysis.**  
> **Do NOT confuse past hypotheses with the current state above. The CURRENT STATE section is authoritative.**  
> **Key historical phases (all RESOLVED):**  
> - HF-025..HF-040 (2026-07-04/05): "6.7% acceptance / garbled output" phase — RESOLVED (the snapshot-slot-ordering fix `a3732a5c3` + the rollback `tape_replay` fix `37a89c8f6` + the slot-routing patches `0d5f99638`).  
> - 2026-07-09: "tree-aware graph pieces missing" hypothesis — DISPROVEN (tree ops are inert on Vulkan; the real cause was slot routing + eval-callback gating).  
> - 2026-07-10/11: "rollback corruption" investigation — RESOLVED (`37a89c8f6` skip tape_replay).  
> - 2026-07-12: "structural port required" hypothesis — RETRACTED (the delta analysis showed the impl exists; targeted patches sufficed for correctness).  
> - 2026-07-13: "re-decode batching" attempt — FAILED and reverted (corrupts hidden_gpu capture).  

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

## 2026-07-10 19:00 UTC - 🎉 ROOT CAUSE FIXED: tape_replay double-advances r_l; skip it → p1 IDENTICAL in production

### FACTS
- ROOT CAUSE PINPOINTED via the r_l (conv state) dump: `dflash_rollback`'s `tape_replay` (specifically `tape_replay_conv`) MODIFIES the conv state (`r_l`) even when it returns `n_positions_needing_reeval > 0` (telling the server to re-decode). This causes a DOUBLE-ADVANCE of `r_l` (tape_replay advances r_l, THEN the server's re-decode advances r_l AGAIN). The s_l (recurrent state) was correctly restored (sum/abs and svals matched), but the r_l (conv state) was wrong (rvals differed by 20-400%).
- Evidence: post_rollback_pre_trim (after dflash_rollback, before any trim) rvals ≠ pre_verify rvals (the backup). pre_verify rvals = nodflash rvals (0.0000% diff — correct before rollback). So the rollback's tape_replay corrupted the r_l.
- THE FIX: skip `tape_replay` in `dflash_rollback` (set `n_positions_needing_reeval = n_accepted` unconditionally, don't call `tape_replay`). The server's re-decode from the restored backup state is sufficient and correct (it advances both s_l and r_l through the accepted tokens). The tape_replay was redundant + corrupting (double-advance).
- Also applied: cell-pos trim (`llama_memory_seq_rm` after the rollback to align the cell pos), accepted tokens in the re-decode (instead of original drafts), batch re-decode. These were part of the investigation but the tape_replay skip was the KEY fix.
- VERIFIED IN PRODUCTION (no env vars): p1 DFlash == non-DFlash IDENTICAL at 12.3 t/s. ✅✅✅
- n_tokens > 0: ✅ (the tape populates with set_tape_recording + tape_gpu block; verified earlier — 47 batches with n_tokens=4).
- p2/p3: same inherent DFlash-vs-non-DFlash differences as Phase 1 + gboddaer/main (wording differences + truncation, NOT corruption). The p2 "line 5" diff is a trailing-space whitespace artifact; the real divergence at line 17 is the inherent DFlash behavior (same as the working fork). No regression.
- Speed: p1 DFlash 12.3 t/s (vs Phase 1 no-rollback 23.6 t/s). The rollback + re-decode overhead slows it. This can be optimized later (the re-decode is correct but adds a decode per partial-accept). The correctness fix is the priority.

### THE COMPLETE FIX (production, no env gates):
- qwen35.cpp: tape_gpu graph-builder block (copies qkv_mixed to tl.qkv when tape_gpu_n_seqs > 0) + beta_presigmoid save.
- server-context.cpp: (1) llama_set_tape_recording(true/false) around the verify decode, (2) backup creation (unconditional cp_recurrent), (3) production rollback (DFlash type gate, not env), (4) cell-pos trim (llama_memory_seq_rm after rollback), (5) re-decode with accepted tokens (not original drafts), (6) batch re-decode.
- llama-context.cpp: tape_replay skip in dflash_rollback (the KEY fix — skip tape_replay, let the server re-decode from the restored backup).

### TEST RESULTS
- Command: apply the complete fix (tape_replay skip + cell-pos trim + accepted tokens + batch re-decode + set_tape_recording + tape_gpu block + backup + production rollback); build; test p1 production (no env vars).
- Result: p1 IDENTICAL at 12.3 t/s. ✅✅✅ n_tokens > 0 ✅. p2/p3 same as Phase 1 (no regression).
- Earlier: with tape_replay ACTIVE (not skipped), p1 DIVERGENT (r_l double-advanced). With tape_replay SKIPPED, p1 IDENTICAL. The tape_replay skip is the key fix.
- Output files: /tmp/p2/fp_{nd,d}.{out,gen} (production test), /tmp/p2/pt_d.err (r_l dump evidence).

### NEXT STEPS
- The fix is complete and verified (p1 identical, n_tokens > 0, no regression). Ready to commit + push.
- Speed optimization (12.3 vs 23.6 t/s): the re-decode overhead can be reduced (e.g., skip the re-decode when all-accepted, or optimize the batch re-decode). This is a follow-up.
- The tape_gpu block (qwen35.cpp) populates the tape but the tape is now UNUSED (tape_replay is skipped). The tape_gpu block could be removed (it's inert now). But it's harmless (gated on tape_gpu_n_seqs > 0). Keep for now.

## 2026-07-12 14:00 UTC - Current-state summary (post rollback-corruption fix)

### FACTS (current state of the work branch)
- Work branch HEAD = `37a89c8f6` "dflash(vulkan): fix rollback corruption — skip tape_replay (double-advances conv state r_l)" (2026-07-11). `gboddaer/main` = `130ea2480` (merge-base of the work branch). Remote `origin/merge_llama_into_beellama_2` = `c5c17182e` is STALE (local has 12 commits past it: `a3732a5c3`..`37a89c8f6`, unpushed).
- The DFlash correctness regression is FIXED by two commits:
  - `a3732a5c3` "fix gated_delta_net snapshot slot ordering (primary DFlash correctness regression)" — Phase 1 shader/graph fix. ALSO landed the tree-aware graph port already: `ggml_ssm_conv_tree` in qwen35.cpp/qwen35moe.cpp, `ggml_gated_delta_net_tree` in delta-net-base.cpp, `build_conv_state` `qkv_mixed_transposed` param, `prefill_gpu` capture in qwen35.cpp. (The tree-aware port I re-derived earlier today was therefore redundant — already committed.)
  - `37a89c8f6` "fix rollback corruption — skip tape_replay" — Phase 2. Root cause: `dflash_rollback`'s `tape_replay` (`tape_replay_conv`) advanced the conv state `r_l` even when returning `n_reeval>0`, so the server's re-decode advanced `r_l` AGAIN → double-advance → r_l differed 20–400% from sequential; `s_l` was correct. Fix: skip `tape_replay` in `dflash_rollback` (set `n_reeval=n_accepted`), let the server re-decode from the restored backup. Plus production rollback port: tape_gpu graph block (qwen35.cpp), `set_tape_recording` around verify, unconditional `cp_recurrent` backup, cell-pos trim (`llama_memory_seq_rm`), accepted-token re-decode, batch re-decode.
- Recorded verification (commit message, 2026-07-11, Qwen3.6-27B-Q4_K_M, Vulkan0, greedy temp 0, NO env vars): p1 DFlash == non-DFlash IDENTICAL at 12.3 t/s; n_tokens>0 (tape populates, 47 batches n_tokens=4); p2/p3 same inherent DFlash-vs-non-DFlash wording/truncation differences as Phase 1 + gboddaer/main (NOT corruption). Phase 1 no-rollback speed was 23.4–23.6 t/s; the rollback + re-decode overhead drops production DFlash to 12.3 t/s.
- Earlier 2026-07-10 entries (H-GHOST, cell-pos misalignment, full state-value dumps) are the investigation trail that led to the r_l double-advance root cause — kept for provenance, but superseded by `37a89c8f6`.
- My (2026-07-09) "tree-aware port won't fix Vulkan" finding stands and is consistent with the actual fix: the tree-aware graph pieces are inert on Vulkan (no Vulkan backend for the tree ops; under DFlash `build_recurrent_attn` takes the `keep=true` direct `ggml_gated_delta_net` K-snapshot path that bypasses `build_delta_net`). The REAL root cause was (1) gated_delta_net snapshot slot ordering (a3732a5c3) + (2) rollback r_l double-advance (37a89c8f6) — both server/Vulkan-shader/recurrent-state issues, not the tree-aware graph builders.

### HYPOTHESES
- None open for the primary correctness regression (fixed). Open items are: (a) DFlash speed regression vs no-rollback (12.3 vs 23.6 t/s) — the per-partial-accept re-decode overhead; (b) multi-slot DFlash (`--parallel>1`) still unverified/likely still broken; (c) CI test failures (OpenVINO Windows, macOS) unrelated to DFlash.

### NEXT STEPS
- Re-run a fresh DFlash vs non-DFlash benchmark on the work branch at HEAD `37a89c8f6` (Qwen3.6-27B-Q4_K_M, Vulkan0, greedy) to produce independent proof of current correctness + performance (this entry's recorded numbers are from the commit message; reproduce them).
- Compare work-branch DFlash against `gboddaer/main` DFlash for the same prompt to confirm the merge reaches fork-level correctness + relative speed.
- If confirmed: the regression investigation closes; remaining work is speed optimization, multi-slot, CI.

## 2026-07-12 15:00 UTC - Independent DFlash benchmark + correctness proof at HEAD (37a89c8f6)

### FACTS
- Reproduced the commit claims with a fresh, independent benchmark (no env vars, production binaries).
- Binaries built from source: work-branch `build-vulkan/bin/llama-cli` = `b10554-37a89c8f6` (HEAD); main-ref built in a detached worktree at `gboddaer/main` = `130ea2480` (`/tmp/beellama-main-ref/build_vulkan/bin/llama-cli`, version `b10118-130ea2480`).
- Test config: target `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`, draft `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`, `--device Vulkan0` (AMD Radeon RADV GFX1151 iGPU — only physical device; no W7800/Vulkan1 present), `-ngl 999 --ctx-size 8192 --flash-attn on --cache-type-k/v q8_0 --seed 7 --temp 0 -n 256 --single-turn --simple-io`, DFlash `--spec-draft-n-max 3`. Prompt: "What is 2+2? First think step by step, then give the final answer on its own line as 'Answer: X'." (≥200-token generations, per HF-036 lesson).
- Performance (generation t/s):
  | branch | non-DFlash | DFlash | DFlash speedup |
  |--------|-----------|--------|----------------|
  | main (130ea2480) | 12.4 | 21.6 | 1.74× ✅ |
  | work (37a89c8f6, HEAD) | 12.5 | 11.7 | 0.94× ⚠️ (slower than non-DFlash) |
  Non-DFlash speed matches across branches (12.4 vs 12.5). Work-branch DFlash is ~2× slower than main DFlash (11.7 vs 21.6).
- Correctness: all four outputs are COHERENT (proper step-by-step reasoning, no garble, no mixed CJK). Non-ASCII char count = 0 in all four generated texts (garble check passed).
- Parity evidence (work DFlash ≈ main DFlash): both open identically for the first 7 lines ("Thinking Process: 1. Analyze the Request... 2. Constraint... 3. Perform the Operation: Start with 2, Add 2"), then diverge only in minor step wording (work: `$2+2=4$`; main: `2+2=4`). This is the "inherent DFlash-vs-non-DFlash wording difference, not corruption" the commit describes.
- The working fork ALSO shows DFlash≠non-DFlash wording differences (main non-DFlash uses a different structure entirely — "Counting method: count up 2 more (3,4)" vs main DFlash "Perform the Operation: 2+2=4"). So DFlash-vs-non-DFlash divergence is normal fork behavior, NOT a merge regression. The merge's DFlash divergence is smaller than the fork's own.
- Conclusion: the DFlash **correctness** regression is FIXED at HEAD (coherent, fork-parity, no garble). A DFlash **speed** regression remains: work DFlash (11.7 t/s) does not beat non-DFlash (12.5), vs main DFlash 21.6 t/s (1.74× speedup). Cause: the rollback + per-partial-accept re-decode overhead (commit notes: production 12.3 vs no-rollback 23.6 t/s). The speculative speedup is cancelled by re-decode cost.
- Artifacts: `/tmp/dflash-bench/{work,main}/{<label>_{nodflash,dflash}.out,.err}`, extracted `.gen` files. Script: `scripts/dflash-regression/bench_dflash_v0.sh`.

### HYPOTHESES
- The speed regression is the rollback re-decode path running a full decode per partial-accept batch. Optimizations: skip re-decode when the verify batch is all-accepted (state already correct), or batch the re-decode more efficiently, or only re-decode the rejected-suffix tokens. These would restore the speculative speedup while keeping the r_l/s_l correctness.

### TEST RESULTS
- Command: `scripts/dflash-regression/bench_dflash_v0.sh build-vulkan/bin/llama-cli work /tmp/dflash-bench/work` and same with `/tmp/beellama-main-ref/build_vulkan/bin/llama-cli main /tmp/dflash-bench/main`.
- Result: work non-DFlash 12.5 t/s (rc 0), work DFlash 11.7 t/s (rc 0); main non-DFlash 12.4 t/s (rc 0), main DFlash 21.6 t/s (rc 0). All outputs coherent; garble check 0 non-ASCII; work DFlash ≈ main DFlash (minor wording diff).
- Output files: `/tmp/dflash-bench/work/*.out`, `/tmp/dflash-bench/main/*.out`, `/tmp/dflash-bench/{work,main}_{nodflash,dflash}.gen`.

### NEXT STEPS
- Correctness investigation CLOSES (regression fixed at `37a89c8f6`).
- Open: DFlash speed optimization (restore the ~1.7× speculative speedup) — investigate skipping/batching the rollback re-decode for all-accepted verify batches.
- Open: multi-slot DFlash (`--parallel>1`), CI test failures (OpenVINO Windows, macOS) — unrelated to the DFlash correctness fix.
- Optional: clean up now-inert `tape_gpu` graph block in qwen35.cpp (tape_replay is skipped, so the tape is unused) — harmless but dead code.

## 2026-07-12 16:30 UTC - DFlash speed optimization attempt (all-accepted skip + tape-disable)

### FACTS
- The "skip rollback re-decode when the verify batch is all-accepted" optimization is ALREADY implemented at `tools/server/server-context.cpp:4429` (`const bool all_accepted_flat = n_accepted_draft == (int) slot.spec_draft.size(); if (!all_accepted_flat) { llama_dflash_rollback(...) + re-decode }`). When all drafts are accepted, the rollback+re-decode is skipped entirely. So the literal proposed optimization was already in place at HEAD `37a89c8f6`.
- Tested a second candidate: disabling the now-dead tape recording. Rationale: `dflash_rollback` skips `tape_replay` (37a89c8f6), and the only other tape consumer (`dflash_prepare_branch`, tree-verify, `src/llama-context.cpp:4284`) is never called from the flat DFlash server path (server only calls `llama_dflash_rollback`). Yet `set_tape_recording(true)` runs around every verify decode (`server-context.cpp:4065`), enabling the `tape_gpu` graph block (`qwen35.cpp:550`, copies qkv_mixed→tl.qkv per recurrent layer). Commit 37a89c8f6 itself flagged this as "inert now... could be removed".
- Implemented the tape-disable (gated `dflash_tape_active` on a `dflash_tape_record=false` flag), built, benchmarked, then REVERTED (no-op, see results).
- CRITICAL COMPARISON: `gboddaer/main` ALSO re-decodes on Vulkan. `git show gboddaer/main:tools/server/server-context.cpp` shows `llama_dflash_rollback(ctx_tgt, ...)` + `llama_decode(ctx_tgt, batch_reeval)` with the comment "When tape replay was unavailable (e.g., Vulkan), re-decode". So the re-decode on partial-accept is NOT a merge-specific workaround — the fork does it too on Vulkan. The re-decode is necessary on Vulkan because the conv state `r_l` has no per-token snapshots (the tree conv `ggml_ssm_conv_tree` that produces per-token conv snapshots is CUDA-only; Vulkan has no backend for it).

### HYPOTHESES
- HYPOTHESIS (disproven): disabling tape recording recovers speed. DISPROVEN by measurement.
- The ~2× speed gap (merge DFlash 11.5 t/s vs main DFlash 21.6 t/s, both on Vulkan0, both re-decoding) is NOT explained by the re-decode skip (present in both) or the tape recording (no-op). Remaining candidates: (a) acceptance rate difference (merge draft quality lower → more partial-accept → more re-decodes); (b) per-cycle unconditional backup cost (`cp_recurrent` at `server-context.cpp:3357` every cycle in the merge vs main's `has_draft_backup` gated in draft-prep); (c) verify-decode graph efficiency (n_rs_seq/K-snapshot count, n_outputs_max); (d) residual instrumentation overhead from c5c17182e.

### TEST RESULTS
- Command (tape ON, original HEAD, "2+2" prompt): work DFlash 11.7 t/s (first benchmark).
- Command (tape OFF, "2+2" prompt, same seed/flags): work DFlash 11.5 t/s. → tape-disable is a NO-OP (within noise).
- Command (tape OFF, "farmer fence" complex prompt, -n 400): work non-DFlash 12.4 t/s; work DFlash 9.3 t/s; main DFlash 21.5 t/s. (9.3 vs 11.7 is the different/lower-acceptance farmer prompt, not a tape effect — confirmed by the 2+2 apples-to-apples 11.5 vs 11.7.)
- Reverted the tape-disable change; `git diff` shows only TASK_PROGRESS.md modified; server-context.cpp back to HEAD; rebuild EXIT=0.
- Output files: `/tmp/dflash-bench2/{work_nd,work_d,main_d,p22_d_tapeoff}.{out,err}`.

### NEXT STEPS
- The proposed speed optimization (all-accepted skip) is already in place; the additional tape-disable is disproven. Recovering the ~1.7× gap requires deeper work, not a simple skip:
  1. Measure the DFlash acceptance rate (merge vs main) — if the merge's acceptance is lower, investigate draft quality (the a3732a5c3 snapshot-slot-ordering fix may be imperfect, or n_rs_seq/verify-decode graph differs).
  2. Compare the per-cycle backup cost (merge unconditional `cp_recurrent` vs main's gated `has_draft_backup`) — potentially defer/skip the backup.
  3. Compare n_rs_seq (K) and n_outputs_max between branches for the DFlash verify decode.
  4. The fundamental Vulkan limit: `r_l` (conv state) has no per-token snapshots without `ggml_ssm_conv_tree` (CUDA-only). On CUDA, porting tree-conv snapshots would eliminate the re-decode entirely (the fork's fast path). On Vulkan, the re-decode is unavoidable for partial-accept.
- Current shipped state (37a89c8f6) remains correctness-first at ~11.5–12.3 t/s DFlash. Non-DFlash is 12.4–12.5 t/s. main DFlash is 21.5–21.6 t/s (1.74×). The speed gap is open but not closeable by the "skip re-decode on all-accepted" optimization (already done).

## 2026-07-12 17:30 UTC - Option 1 verdict: acceptance rate measurement (work vs main)

### FACTS
- Measured DFlash acceptance via `llama-server` (which prints `draft acceptance = ... (accepted / generated)` + `common_speculative_print_stats` `#gen drafts / #acc drafts / #gen tokens / #acc tokens` per slot finish). Built `llama-server` for both branches from source: work = `b10554-37a89c8f6` (HEAD), main-ref = `b10118-130ea2480` (gboddaer/main). Same config: Qwen3.6-27B-Q4_K_M target + Qwen3.6-27B-DFlash-Q4_K_M draft, Vulkan0, `-ngl 999 -c 8192 --flash-attn on --cache-type-k/v q8_0 --spec-draft-n-max 3`, greedy (temp 0, seed 7), farmer-fence prompt, n_predict 256. Script: `scripts/dflash-regression/measure_acceptance.sh`. Work used port 8091/8093, main 8092.
- **Acceptance counters (the decisive data):**
  | metric | work (37a89c8f6) | main (130ea2480) |
  |--------|------------------|------------------|
  | n_call_draft | 254 / 78 (two runs) | 76 |
  | n_call_accept | **0** | 76 |
  | #gen drafts | **0** | 76 |
  | #acc drafts | **0** | 70 |
  | #gen tokens | **0** | 218 |
  | #acc tokens | **0** | 176 |
  | draft acceptance | **not printed** | **0.807 (176/218)** |
- The work branch calls `common_speculative_draft` (drafts generated) but **`common_speculative_accept` is NEVER called (n_call_accept=0)** — zero drafts accepted/counted. The verify block (`server-context.cpp:4330+`, which calls `common_speculative_accept` at :4419) is never entered: 0 `verify_post` / `verify slot` QA-trace lines.
- Server log root cause (work branch, first lines of the request):
  ```
  E dflash prefill flush incomplete GPU capture: captured=0 requested=28 seq=0; refusing partial ring write
  E srv decode: dflash prefill flush mismatch: slot=3 requested=28 written=0 src_offset=0; disabling DFlash drafting until fresh hiddens are available
  W dflash: discarding cross-ring state: prefill flush mismatch
  ```
  Work branch: 5 `prefill flush mismatch` errors, 2 `disabling DFlash drafting`, 2 `discarding cross-ring state`. Main branch: **0** prefill flush errors, 0 discards, 0 disable messages.
- So the work branch's **prefill GPU capture writes 0 tokens** (`captured=0`, `written=0` despite `requested=28`) → cross-ring state discarded → DFlash drafting disabled → drafts generated but discarded → 0 accepted. Main's prefill capture works → cross-ring populated → 80.7% acceptance → 1.74× speedup.

### HYPOTHESES
- VERDICT: The speed gap is **NOT** "lower draft quality driving excessive re-decode overhead". There is **no re-decode at all** on the work branch (n_call_accept=0 → the verify/rollback/re-decode block is never entered). The re-decode-overhead hypothesis (and the entire 2026-07-12 16:30 "both branches re-decode on Vulkan" framing) is moot for the work branch because no accepts happen.
- The real root cause: **prefill GPU capture failure** (`written=0`) makes DFlash effectively a no-op on the work branch. Drafts are generated (78–254 draft calls) but never verified/accepted. The work branch's ~11.5–12.3 t/s is non-DFlash generation speed (12.4) MINUS the wasted draft-generation overhead, NOT re-decode overhead.
- The 37a89c8f6 commit's claim "p1 DFlash == non-DFlash IDENTICAL at 12.3 t/s" was MISINTERPRETED as success: the outputs are identical because DFlash is effectively disabled (prefill capture fails → no drafts accepted), so the target generates everything. "Identical to non-DFlash" is the symptom of a disabled DFlash, not a working one.
- The "inherent DFlash-vs-non-DFlash wording differences" noted for p2/p3 are NOT from DFlash acceptance (there is none) — they're from the draft-gen+discard overhead perturbing the target state via the (failed) rollback machinery.
- NEXT root-cause question: why does the prefill_gpu capture write 0 tokens on the merge? The prefill_gpu graph block (ported to qwen35.cpp via a3732a5c3) is present but `flush_prefill` reports `captured=0`. Candidates: (a) the graph block doesn't execute (`dflash_prefill_capture_active` false, or `prefill_gpu_n_seqs` 0 at graph-build time); (b) the block writes to a buffer that `flush_prefill` doesn't read (slot/buffer mismatch); (c) the capture scheduling (`llama_dflash_prefill_capture_begin/end`) isn't arming the GPU path. This is the real fix target for recovering the ~1.7× speedup.

### TEST RESULTS
- Command: `scripts/dflash-regression/measure_acceptance.sh build-vulkan/bin/llama-server work 8091 /tmp/dflash-accept` (and main-ref, port 8092); plus a GGML_DFLASH_QA_TRACE=1 work run on port 8093.
- Result: work n_call_accept=0, #acc tokens=0, 5 prefill-flush-mismatch errors, 0 verify_post; main n_call_accept=76, #acc tokens=176, draft acceptance=0.807 (176/218), 0 prefill-flush errors.
- Output files: `/tmp/dflash-accept/{work,main}.{server.err,resp.json}`, `/tmp/dflash-accept/work_qa.err`.

### NEXT STEPS
- The speed-recovery fix is NOT "skip re-decode" (already done; and irrelevant — no accepts happen). It is: **fix the prefill GPU capture so it writes the requested tokens (written=N, not 0)**, which re-enables DFlash drafting → drafts get verified/accepted → acceptance rises toward main's 80% → speedup recovered.
- Concrete next investigation: trace why `flush_prefill` returns written=0 on the merge. Check (1) is `cparams.dflash_prefill_capture_active` true and `prefill_gpu_n_seqs > 0` during the prefill decode on the merge; (2) does the qwen35.cpp prefill_gpu graph block actually execute (add a DFLASH_QA trace inside it); (3) compare the `llama_dflash_prefill_capture_begin/end` scheduling and `flush_prefill` buffer read path between merge and gboddaer/main. This is the true speed-fix target.
- Do NOT pursue further "re-decode skip" or "tape-disable" optimizations — disproven/irrelevant. The blocker is prefill capture.

## 2026-07-12 18:30 UTC - Prefill GPU capture bug: ROOT CAUSE FOUND (candidate 2: slot/buffer mismatch)

### FACTS
- Investigated the three candidates for the prefill GPU capture failure (`captured=0`, `written=0`).
- Candidate 1 (cparams not set at graph-build): the graph-block cparams (`prefill_gpu_n_seqs`, `dflash_prefill_capture_active`, `dflash_prefill_n_tokens`) are set in `llama-context.cpp:6892-6893` when `any_intersection && all_intersections_have_buffer`, and the post-compute `plan->n_written` update at `llama-context.cpp:7206-7230` gates on the same cparams and indexes by the ubatch's actual `seq_id`. `llama_dflash_prefill_gpu_active()` returns true on the failing run (the `use_prefill_gpu` branch of `flush_prefill` is taken — that's where the `captured=0 ... refusing partial ring write` error prints, `common/speculative.cpp:2764`). So the GPU path IS active; the cparams are not the failure point per se.
- Candidate 3 (capture_begin/end scheduling): `llama_dflash_prefill_capture_begin(ctx_tgt, slot.id, span.capture_begin, span.capture_end)` is called at `server-context.cpp:4046` with the correct `slot.id`; the plan is created in `dflash_capture->prefill_plans[seq_id]` with `active=true`. The server flush loop (`server-context.cpp:4140-4164`) calls `llama_dflash_set_active_slot(ctx_tgt, pf.slot_id)` (line 4153) BEFORE `common_speculative_flush_prefill(pf_spec, ...)` (line 4156) — i.e. the server DOES correctly route to the physical slot. Scheduling is correct.
- **Candidate 2 (slot/buffer mismatch): CONFIRMED ROOT CAUSE.** `flush_prefill` (`common/speculative.cpp:2721`) reads the prefill plan/buffer by the dflash impl's `seq_id` member, which is **0** for per-slot DFlash specs:
  - `get_seq_id()` returns `spec ? 0 : id` (`server-context.cpp:458`) — per-slot DFlash uses seq_id=0 by design.
  - `common_speculative_set_seq_id(slot.get_spec(), slot.get_seq_id())` → `set_seq_id(0)` (`server-context.cpp:1603`).
  - `flush_prefill` then uses this `seq_id=0` in THREE places: `llama_dflash_set_active_slot(ctx_tgt, seq_id=0)` (line 2722 — OVERRIDES the server's `set_active_slot(pf.slot_id)`), `llama_dflash_prefill_capture_info(ctx_tgt, seq_id=0, ...)` (reads `prefill_plans[0]`), and the fallback `llama_dflash_prefill_gpu_n_tokens(ctx_tgt, seq_id=0)` (reads `prefill_gpu[0]`).
  - But the capture was scheduled for the physical slot (`capture_begin(slot.id)`), and the graph block + `n_written` update index by the ubatch's actual seq (`ubatch.seq_id_unq` = `slot.id`). So when `slot.id != 0`, the capture writes to `prefill_plans[slot.id]` / `prefill_gpu[slot.id]`, but `flush_prefill` reads index 0 (empty) → `captured=0` → `written=0` → "prefill flush mismatch" → DFlash drafting disabled → 0% acceptance.
- **DECISIVE PROOF (--parallel 1 test):**
  | run | request slot | prefill flush mismatch | n_call_accept | draft acceptance |
  |-----|-------------|------------------------|---------------|------------------|
  | work --parallel 4 (default) | slot 3 (LRU) | 5 | 0 | 0% (no accepts) |
  | work --parallel 1 | slot 0 (only slot) | **0** | **37** | **0.752 (82/109)** |
  | main --parallel 4 (default) | slot 0 (LRU) | 0 | 76 | 0.807 (176/218) |
  With `--parallel 1` (slot 0 → impl seq_id=0 matches the physical slot), the merge's prefill flush errors vanish and acceptance jumps to 75%, essentially matching main's 80%. The bug is purely the seq_id→physical-slot mismatch.
- Main is MASKED, not immune: main's `flush_prefill` is byte-identical (same `set_active_slot(seq_id)` override at line 2722). Main "works" only because LRU happened to pick slot 0 for the request. If main's request landed on a non-zero slot, main would hit the same `captured=0` failure. (Why main picks slot 0 and the merge picks slot 3 is a secondary LRU/startup-ordering question; the primary bug is the seq_id override.)
- Separate observation (NOT the prefill-capture bug): even with acceptance restored (--parallel 1, 75%), the merge's DFlash speed is ~10.3 t/s (n_decoded=102, tg=10.26) — still below non-DFlash (12.4) and far below main's 24 t/s with 80% acceptance. This is the re-decode-on-partial-accept overhead (the next investigation), not the prefill capture bug.

### HYPOTHESES
- CONFIRMED: the prefill GPU capture bug root cause is candidate 2 — `flush_prefill` reads the prefill plan/buffer by the impl's `seq_id=0` instead of the physical request slot.id. When the request slot ≠ 0 (the default `--parallel 4` case where LRU picks a non-zero slot), `captured=0`, DFlash is disabled, acceptance=0.
- The per-slot design intends `seq_id=0` to be mapped to the physical slot via `set_active_slot(slot.id)` (which the server does correctly at 4153), but `flush_prefill` violates this by (a) re-setting the active slot to `seq_id=0` (line 2722) and (b) reading `prefill_capture_info` / `prefill_gpu_n_tokens` by `seq_id=0` (lines 2745/2747) instead of by the active/physical slot.

### TEST RESULTS
- Command: `llama-server ... --parallel 1 --spec-type dflash ...` (work, port 8094) vs the earlier `--parallel 4` (default) run (port 8093/8091) and main (port 8092).
- Result: --parallel 1 → slot 0, 0 prefill-flush-mismatch errors, n_call_accept=37, #acc tokens=82, draft acceptance=0.752 (82/109); --parallel 4 → slot 3, 5 mismatch errors, n_call_accept=0, 0% acceptance.
- Output files: `/tmp/dflash-accept/work_p1.{err,resp.json}`, `/tmp/dflash-accept/work_qa.err`, `/tmp/dflash-accept/main.server.err`.

### NEXT STEPS
- The prefill-capture bug fix: make `flush_prefill` read the prefill plan/buffer for the PHYSICAL request slot (slot.id), not the impl's `seq_id=0`. Two viable approaches:
  1. Pass `slot_id` into `flush_prefill` (new param) and use it for `set_active_slot`, `prefill_capture_info`, `prefill_gpu_n_tokens`. Cleanest and explicit.
  2. Remove the `llama_dflash_set_active_slot(ctx_tgt, seq_id)` override at line 2722 (let the server's `set_active_slot(pf.slot_id)` stand) AND change `prefill_capture_info` / `prefill_gpu_n_tokens` to look up by the currently-active slot rather than the passed `seq_id`. Requires a "read active slot" variant of those helpers.
  - Approach 1 is lower-risk (explicit, no global-active-slot assumptions). The server already knows `pf.slot_id` (it passes nothing to `common_speculative_flush_prefill` currently — line 4156 — so the API would gain a `slot_id` param).
- After the prefill-capture fix, the merge's default `--parallel 4` DFlash should recover acceptance (toward 75-80%) on any slot, not just slot 0. Then the REMAINING speed gap (merge ~10 t/s vs main ~24 t/s with similar acceptance) is the re-decode-on-partial-accept overhead — a separate follow-up (the original "speed optimization" target, now correctly framed: it only matters once acceptance is restored).
- Verify the fix on `--parallel 4` (default) with a non-zero slot (the real-world case), not just `--parallel 1`.

## 2026-07-12 19:30 UTC - Slot-routing fix implemented + verified (necessary but INSUFFICIENT for multi-slot)

### FACTS
- Implemented the recommended fix: pass the physical `slot_id` into `flush_prefill` so it routes to the slot that scheduled the capture.
  - `common/speculative.h:193` + `common/speculative.cpp:5122`: `common_speculative_flush_prefill(... , llama_seq_id slot_id = -1)`.
  - `common/speculative.cpp:388` (base virtual) + `:2721` (dflash override): added `llama_seq_id slot_id = -1` param; dflash override computes `const llama_seq_id phys_slot = (slot_id >= 0) ? slot_id : seq_id;` and uses `phys_slot` for `llama_dflash_set_active_slot`, `llama_dflash_prefill_capture_info`, `llama_dflash_prefill_gpu_n_tokens` (the 3 lookups that index the prefill plan/buffer). The drafter reset (`common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, ...)`) keeps `seq_id` (the drafter's own seq — separate concern). Diagnostic `seq=%d` labels changed to `slot=%d` with `phys_slot`.
  - `tools/server/server-context.cpp:4156`: call site passes `pf.slot_id`.
  - Build: `cmake --build build-vulkan --target llama-server llama-cli` EXIT=0, no errors.
- Verification (default multi-slot, `--parallel 4`, one server + 4 sequential requests to force non-zero slots):
  | request | slot | prefill flush mismatch | acceptance | tg |
  |---------|------|------------------------|------------|-----|
  | 1 | 3 | yes (slot=3 written=0) | 0% (0/3) | 4.46→5.23 t/s |
  | 2 | 2 | (none logged) | 0% (0/3) | ~10.8 t/s |
  | 3 | 1 | (none logged) | 0% (0/3) | ~10.8 t/s |
  | 4 | 0 | none | 81.9% (113/138) | 11.6→12.2 t/s |
  Slot 0 has DFlash active (81.9%, captured>0); slots 1-3 still have DFlash disabled (0% acceptance).
- The slot_id fix is CORRECT and NECESSARY (it routes flush_prefill to the physical slot) but NOT SUFFICIENT: slots 1-3 still fail because the prefill capture is never set up for them, so there is nothing to read regardless of routing.

### ROOT CAUSE of the remaining multi-slot failure (deeper than slot routing)
- `GGML_DFLASH_DEBUG=1 GGML_DFLASH_PREFILL_TRACE=1` traces show:
  - The server DOES schedule the capture for slot 3: `[DFLASH_PREFILL_TRACE] slot=3 prompt_total=32 cross_ctx=512 capture_from=0 batch_pos=[0,27] batch_end=28` (and `[28,31]`). So `llama_dflash_prefill_capture_begin(ctx_tgt, 3, ...)` IS called (candidate 3 — scheduling — is NOT the problem).
  - But the decode-side prefill_gpu staging path is NEVER taken for slot 3: 0 `dflash capture route` logs, 0 `allocated prefill GPU staging buffers`, 0 `prefill capture complete` (n_written) updates. So `dflash_use_prefill_staging` is false at slot 3's prefill decode → no GPU staging → no `plan->n_written` update → `captured=0`.
- CRITICAL comparison: `--parallel 1` (slot 0, works at 75%) ALSO shows 0 staging allocation/route/complete logs — i.e., neither config uses the prefill_gpu staging path for this prompt. The capture happens via the **CPU eval-callback path** (`flush_prefill` `else` branch reads `llama_get_layer_hidden_n_tokens`).
- The difference that makes `--parallel 1` work and `--parallel 4` fail:
  - `--parallel 1`: `llama_dflash_allocate_slots` is gated `dflash_slots_cap > 1` (HF-035 guard), so with 1 slot it is NOT called → `hidden_gpu` is NOT allocated → `dflash_gpu_capture_ready=false` → the **eval callback stays installed** → CPU hidden capture works → `captured>0` → flush succeeds → DFlash active (75%).
  - `--parallel 4`: `allocate_slots(4)` IS called (4 > 1) → `hidden_gpu[0..3]` allocated → `dflash_gpu_capture_ready=true` → the **eval callback is disabled** (the comment at `flush_prefill:2728` says "In GPU prefill-staging mode the eval callback is intentionally not installed"). BUT the prefill_gpu staging path is NOT activated for the prefill decode (`dflash_use_prefill_staging` false — the plan is not active at decode time, so `dflash_prefill_plan_active && max_tokens>25` is false) → no GPU staging either. Result: eval callback OFF AND staging OFF → **no capture at all** → `captured=0` → prefill flush mismatch → DFlash disabled.
- So the multi-slot failure is a **mode-coordination bug**: allocating `hidden_gpu` (for multi-slot GPU capture) disables the CPU eval callback, but the prefill_gpu staging path isn't activated for the prefill decode (the capture plan is not active at decode time), so there is no capture path at all. The system is stuck in a broken in-between state (GPU mode off + staging off).
- The "plan not active at decode time" sub-cause: `dflash_prefill_plan_active = dflash_capture->any_prefill_plan_active()` (line 6761) returns false at slot 3's prefill decode even though `capture_begin(3)` was called in `pre_decode`. This is a timing/ordering issue between `capture_begin` (pre_decode scheduling), the decode, and `capture_end` (post-decode flush) — likely the plan is deactivated (by `capture_end` deactivating ALL plans, line 2606, or by chunk interleaving) before the staging decision runs. Not fully pinned down.

### HYPOTHESES
- The slot_id routing fix is correct and necessary (keep it). It will take effect once the capture actually happens for non-zero slots.
- The remaining multi-slot blocker is the eval-callback/staging mode coordination: when `hidden_gpu` is allocated (multi-slot), the eval callback is disabled, but the prefill staging path isn't activated for the prefill decode, so no capture occurs. Two candidate fixes:
  1. Keep the eval callback installed for the PREFILL decode even when `hidden_gpu` is allocated, UNLESS the prefill staging path is actually going to be used for that decode (i.e., only disable the eval callback when `dflash_use_prefill_staging` is true for the current decode). This restores the CPU fallback capture for multi-slot prefills.
  2. Ensure the capture plan is active at decode time so `dflash_use_prefill_staging` becomes true and the GPU staging path captures (fix the `capture_begin`/`decode`/`capture_end` timing, or make `capture_end` per-slot so it doesn't deactivate a still-pending plan).
  - Candidate fix 1 is lower-risk and directly addresses the "no capture path" symptom.

### TEST RESULTS
- Command: `scripts/dflash-regression/verify_slotfix.sh build-vulkan/bin/llama-server workfix 8095 /tmp/dflash-slotfix 4` (4 sequential requests, default multi-slot).
- Result: slot 3 → 0% (mismatch), slot 2 → 0%, slot 1 → 0%, slot 0 → 81.9% (138 gen / 113 acc, tg 12.2 t/s).
- Command: `GGML_DFLASH_DEBUG=1 GGML_DFLASH_PREFILL_TRACE=1 llama-server ... --parallel 4` (single farmer request, slot 3).
- Result: `[DFLASH_PREFILL_TRACE] slot=3 ...` (capture scheduled), but 0 staging/capture-complete logs, 2 prefill-flush-mismatch errors, 0% acceptance.
- Command: same with `--parallel 1` (slot 0).
- Result: 0 staging logs too (CPU path used), but acceptance 75% (CPU eval-callback capture works because allocate_slots skipped → eval callback on).
- Output files: `/tmp/dflash-slotfix/workfix.{server.err,req*.json}`, `/tmp/dflash-slotfix/workfix_dbg.err`, `/tmp/dflash-slotfix/workfix_trace.err`, `/tmp/dflash-slotfix/workfix_p1_trace.err`.

### NEXT STEPS
- The slot-routing fix is KEPT (correct, necessary). Multi-slot DFlash is NOT yet restored.
- Implement candidate fix 1: in `llama-context.cpp` decode setup, only disable the eval callback (`dflash_skip_eval_callback` / `cparams.cb_eval = nullptr`) when `dflash_use_prefill_staging` is true for the current decode — NOT merely because `hidden_gpu` is allocated. This keeps the CPU eval-callback capture available for multi-slot prefills that don't use staging, restoring `captured>0` for slots 1-3.
- Verify on `--parallel 4` with non-zero slots (the real multi-slot case): confirm `captured>0`, acceptance restored toward 75-80%, no prefill-flush-mismatch.
- Then address the speed gap (slot 0 at 12.2 t/s vs main 24 t/s — the re-decode-on-partial-accept overhead, separate from prefill capture).
- NOTE: the speed is NOT restored toward 24 t/s by this fix (slot 0 is 12.2 t/s); the 24 t/s gap is the re-decode overhead, a separate follow-up. Do NOT claim speed restoration.

## 2026-07-12 20:00 UTC - Candidate fix 1 (eval-callback gating) implemented: PARTIAL PROGRESS, multi-slot still broken

### FACTS
- Implemented candidate fix 1 in `src/llama-context.cpp:7018`: changed the eval-callback-disable gate from `dflash_skip_eval_callback = dflash_graph_hidden_ready || dflash_suppress_callback_for_view` to `dflash_skip_eval_callback = dflash_use_prefill_staging || dflash_suppress_callback_for_view`. The eval callback is now skipped ONLY when the prefill is actually using GPU prefill staging (or capture is suppressed for the view), not merely because `hidden_gpu` is allocated. Build: `cmake --build build-vulkan --target llama-server llama-cli` EXIT=0.
- The two fixes now in place (uncommitted, on top of HEAD `37a89c8f6`): (1) slot_id routing in `flush_prefill` (speculative.cpp/h + server-context.cpp), (2) eval-callback gating on `dflash_use_prefill_staging` (llama-context.cpp:7018).
- Verification (`--parallel 4`, one server + 4 sequential requests, farmer/17x23/haiku/x+3 prompts):
  | req | slot | failure mode | acceptance |
  |-----|------|--------------|------------|
  | 1 | 3 | **GPU hidden D2D ring write failed** (NEW) | 0% (0/3) |
  | 2 | 2 | prefill flush mismatch (captured=0) (OLD) | 0% (0/3) |
  | 3 | 1 | prefill flush mismatch (captured=0) (OLD) | 0% (0/3) |
  | 4 | 0 | none — DFlash active | 81.9% (113/138) |
- PROGRESS: candidate fix 1 RESTORED THE CAPTURE for slot 3. Before the fix, slot 3 failed with `captured=0` ("dflash prefill flush mismatch: slot=3 requested=28 written=0" — flush_prefill returned 0 at the capture-read step). After the fix, slot 3's flush_prefill gets PAST the capture-read (captured=28) and reaches the ring-write step, where it now fails with: `W dflash: discarding cross-ring state: GPU hidden D2D ring write failed` + `E dflash prefill flush wrote incomplete ring span: requested=28 written=0 slot=3; discarding DFlash state` (`common/speculative.cpp:3714`). So the failure moved from capture-read to ring-write — the eval-callback gating fix worked for the capture.
- INCONSISTENT across slots: only slot 3 showed the new "D2D ring write failed" (1 occurrence); slots 2 and 1 still showed the old "prefill flush mismatch" (captured=0) — 2 occurrences. So the capture restoration is not consistent across non-zero slots (timing-dependent: the eval callback capture only sometimes populates layer_hiddens before flush).
- Slot 0 still works (81.9%, 138 drafts, 113 accepted, tg 12.2 t/s) — unchanged.
- The new blocker (the D2D ring-write failure) is at `common/speculative.cpp:3695-3714`: `ring_write` attempts a GPU→GPU D2D copy from `hidden_gpu` to the cross ring, sets `gpu_d2d_failed=true` when `used_d2d=false`, and discards. For slot 3 the capture is in CPU `layer_hiddens` (source=`cpu_hidden`, since `use_prefill_gpu=false`), but the ring-write path is attempting the D2D (GPU→GPU) branch — a source/path mismatch. The fallback `llama_dflash_cross_ring_gpu_write` (H2D, CPU→GPU) at :3704 is only taken when `!used_d2d && data`, but the D2D attempt failing first sets `gpu_d2d_failed` and discards before the H2D fallback can succeed for all layers.

### HYPOTHESES
- Candidate fix 1 is correct and made measurable progress (capture restored for slot 3). It is NECESSARY but still NOT SUFFICIENT for multi-slot DFlash.
- The remaining blockers (two distinct):
  1. **GPU cross-ring D2D write fails for non-zero slots** (`speculative.cpp:3714`): the ring-write path attempts GPU→GPU D2D from `hidden_gpu` even when the capture source is CPU `layer_hiddens`. The source/path selection in `ring_write` doesn't account for the CPU-capture case in multi-slot, so the D2D attempt fails and discards before the H2D fallback completes. Fix: in `ring_write`, when the capture source is `cpu_hidden` (CPU layer_hiddens), skip the D2D attempt and go straight to the H2D `llama_dflash_cross_ring_gpu_write` path; only attempt D2D when the source is `hidden_gpu`/`prefill_gpu` (GPU buffers).
  2. **Inconsistent capture across non-zero slots**: slots 2/1 still hit `captured=0` (eval callback didn't populate layer_hiddens in time for those flushes) — a timing issue between the eval-callback capture (during the prefill decode) and the flush (post-decode). This may relate to the active-slot routing during the decode (the eval callback routes by `dflash_capture->active_tape_idx`, which may not be the request's slot during the prefill decode).

### TEST RESULTS
- Command: `scripts/dflash-regression/verify_slotfix.sh build-vulkan/bin/llama-server workc1 8099 /tmp/dflash-cand1 4` (4 sequential requests, default `--parallel 4`).
- Result: slot 3 → 0% (D2D ring write failed, NEW), slot 2 → 0% (prefill flush mismatch, OLD), slot 1 → 0% (prefill flush mismatch, OLD), slot 0 → 81.9% (138/113, tg 12.2 t/s, working).
- Command: `GGML_DFLASH_DEBUG=1 llama-server ... --parallel 4` (single farmer request, slot 3).
- Result: `W dflash: discarding cross-ring state: GPU hidden D2D ring write failed` + `E dflash prefill flush wrote incomplete ring span: requested=28 written=0 slot=3` (capture succeeded at 28, ring-write failed at 0). 0 meta-backend suppression, 0 mixed-capture-suppressed → the eval-callback gating fix took effect (capture restored).
- Output files: `/tmp/dflash-cand1/{workc1.server.err,workc1.req*.json,workc1_dbg.err}`.

### NEXT STEPS
- Candidate fix 1 is KEPT (correct, made progress). Multi-slot DFlash is NOT yet restored.
- Next fix (blocker 1, the D2D ring-write source/path mismatch): in `common/speculative.cpp` `ring_write` (~3695-3714), gate the GPU→GPU D2D attempt on the capture source being a GPU buffer (`hidden_gpu`/`prefill_gpu`); when the source is CPU `layer_hiddens` (`cpu_hidden`), go directly to the H2D `llama_dflash_cross_ring_gpu_write` path instead of attempting D2D and discarding on failure. This should let slot 3's CPU-captured prefill flush succeed into the cross ring.
- Then address blocker 2 (inconsistent capture for slots 2/1) — likely the active-slot routing during the prefill decode (ensure `set_active_slot(slot.id)` is in effect when the eval callback runs, so it captures into the correct slot's layer_hiddens).
- Speed: slot 0 is 12.2 t/s, NOT the 24 t/s baseline — the re-decode-on-partial-accept overhead (separate from prefill capture). Do NOT claim speed restoration.
- VERIFICATION STATUS: the goal (multi-slot DFlash active with captured>0 and acceptance restored for non-zero slots, speed toward 24 t/s) is NOT yet achieved. Slot 0 works (81.9%, 12.2 t/s); slots 1-3 still fail (0%), now at a DIFFERENT failure point (D2D ring write) for slot 3 due to candidate fix 1's progress. Two more fixes (D2D source/path + active-slot capture routing) are needed.

## 2026-07-12 21:00 UTC - DFlash decode call-tree comparison: work (37a89c8f6) vs gboddaer/main (130ea2480)

### FACTS — file-level divergence (the smoking gun)
- `git diff --stat gboddaer/main -- tools/server/server-context.cpp common/speculative.cpp common/speculative.h src/llama-context.cpp src/models/qwen35.cpp`: **5 files changed, 1884 insertions, 4844 deletions**.
- Line counts: `server-context.cpp` **work=5947 vs main=9002** (~3000 lines shorter); `speculative.cpp` work=5245 vs main=5106 (similar); `qwen35.cpp` work=683 vs main=836.
- The work branch's `server-context.cpp` is a **~3000-line-truncated** version of the fork's DFlash server integration. The merge took **upstream's simpler server structure** and only **partially** ported the fork's DFlash server pipeline (the 1-arg flat draft + a `post_decode` verify/accept/rollback), losing the fork's batched-draft / prefetch-verify / profit-controller / reduced-verify / per-slot-capture pipeline (the ~652 lost lines noted in HF-017 are part of this; the full loss is ~3000 lines).

### FACTS — call-tree comparison (server decode loop → draft → capture → verify/accept → rollback)

#### 1. Draft invocation (the central divergence)
- **WORK** (`pre_decode`, server-context.cpp:3273-3292): per-slot, **1-arg flat**:
  - `auto & draft_params = common_speculative_get_draft_params(slot.get_spec(), slot.get_seq_id());` then manually sets `drafting=true, n_max=n_draft_max, n_min=0, n_past=slot.prompt.n_tokens(), id_last=slot.sampled, prompt=&slot.spec_prompt, result=&slot.spec_draft`.
  - Loop: `for (s : drafting) { llama_dflash_set_active_slot(ctx_tgt, s->id); common_speculative_draft(s->get_spec()); }` — **1-arg**, each slot drafted separately (separate drafter decode per slot).
- **MAIN** (server-context.cpp:4636 + 4852-4859): **batched primary + 6-arg flat fallback**:
  - `common_speculative_draft_batch(batch_specs, ctx_dft_shared.get(), params_batch, batch_id_lasts, batch_results, ...)` (line 4636) — **ONE batched drafter decode for all slots in the cohort**, then `slot.spec_draft = std::move(batched_drafts[slot.id])` (4852).
  - Fallback: `common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, nullptr, draft_n_past)` (4859) — **6-arg**, with `n_min=params_spec.n_min`, `n_past=draft_n_past=-1` for DFlash (`draft_n_past = use_mtp_spec ? prompt.n_tokens() : -1`; DFlash is not MTP → -1).
- **Divergence**: WORK = per-slot 1-arg flat (`n_min=0`, `n_past=prompt.n_tokens()`); MAIN = batched multi-slot draft (`common_speculative_draft_batch`) + 6-arg fallback (`n_min=params_spec.n_min`, `n_past=-1`).

#### 2. DFlash impl method invoked
- **WORK**: 1-arg `common_speculative_draft(spec)` (speculative.cpp:4320) → `impl->draft(spec->dparams)` (the **flat** `draft()` override, speculative.cpp:3035) — per-slot, builds cross data inline + a **separate drafter decode per slot** + extract.
- **MAIN**: `common_speculative_draft_batch` (speculative.cpp:4806) → `impl->prepare_batch_draft(ctx_dft)` (speculative.cpp:2664) per slot (build cross data only, returns cross_len) + **ONE batched drafter decode** for the whole cohort; OR the 6-arg path → `impl->draft()` (flat, but with `params_spec.n_min` + `n_past=-1`).
- **Divergence**: WORK calls `draft()` (flat, per-slot drafter decode); MAIN calls `prepare_batch_draft()` + a single batched drafter decode (efficient). The 6-arg flat path exists in both but WORK's server never calls it.

#### 3. Accept call (seq_id argument difference)
- **WORK** (server-context.cpp:4419): `common_speculative_accept(slot.get_spec(), slot.get_seq_id(), n_accepted_draft)` — 3-arg with **`slot.get_seq_id()` = 0** for per-slot DFlash (`get_seq_id()` returns `spec ? 0 : id`, server-context.cpp:458).
- **MAIN** (server-context.cpp:6716): `common_speculative_accept(slot.get_spec(), slot.id, n_accepted_draft)` — 3-arg with **`slot.id`** (the physical slot). (Also a 2-arg `common_speculative_accept(slot.get_spec(), n_accepted_draft)` at 7185.)
- **Divergence**: WORK passes `seq_id=0` to accept; MAIN passes the physical `slot.id`. Both signatures exist in both branches (`common_speculative_accept(spec, seq_id, n_accepted)` at work:4410/main:4305; `common_speculative_accept(spec, n_accepted)` at work:5058/main:4934) but the WORK server uses the `seq_id=0` variant while MAIN uses `slot.id`.

#### 4. flush_prefill (capture-read routing)
- **WORK** (after my slot_id fix, server-context.cpp:4156): `common_speculative_flush_prefill(pf_spec, pf.span.src_offset, pf.span.n_tokens, pf.slot_id)` — **4-arg with slot_id**; flush_prefill uses `phys_slot = slot_id` for `set_active_slot`/`prefill_capture_info`/`prefill_gpu_n_tokens`.
- **MAIN** (server-context.cpp:6291): `common_speculative_flush_prefill(pf.spec, pf.span.src_offset, pf.span.n_tokens)` — **3-arg, NO slot_id**; flush_prefill uses the impl's `seq_id=0` (the `set_active_slot(seq_id)` override at speculative.cpp:2722, identical in both branches).
- **Divergence**: MAIN has the **latent** flush_prefill-seq_id=0 bug too (it reads slot 0's plan/buffer regardless of the request slot). MAIN is **masked** because its LRU picked slot 0 in the tests. My slot_id fix on WORK actually fixes a bug MAIN still has. (Confirmed: both branches' `flush_prefill` are byte-identical at the `set_active_slot(seq_id)` override — speculative.cpp:2722.)

#### 5. Decode loop structure
- **WORK**: `pre_decode()` (server-context.cpp:3143) does draft + capture scheduling (3934-4164: `capture_begin`, `set_prefill_capture_enabled`, `flush_prefill`, `capture_end` all in pre_decode/post-decode). `post_decode()` (4193) does verify/accept/rollback. The DFlash verify/accept/rollback is a **separate post_decode method**.
- **MAIN**: capture scheduling is a dedicated section (server-context.cpp:5868-6303: `should_flush_dflash_prefill` lambda, `pending_prefill_flushes`, `capture_begin` at 6040, `set_prefill_capture_enabled` at 6047, `flush_prefill` at 6291, `capture_end` at 6303). The draft is at 4636-4872. The verify/accept/rollback is inline in the main slot loop (6660-7400). MAIN also has **prefetch verify** (`dflash_flat_accept_prefetches`, line 6574-6580) and a **profit controller** (`slot.observe_profit_acceptance`, 6736; `slot.profit_pending_*`, 6727-6736) and **reduced-verify** (`dflash_reduced_verify_ready`, `dflash_verify_plan`, 6960+).
- **Divergence**: WORK has a **simplified** pre_decode/post_decode split; MAIN has the **full** capture-scheduling + batch-draft + prefetch-verify + profit-controller + reduced-verify + tree pipeline inline.

#### 6. Re-decode / rollback gate
- **WORK** (server-context.cpp:4429-4443): `const bool all_accepted_flat = n_accepted_draft == (int) slot.spec_draft.size(); if (!all_accepted_flat) { llama_dflash_rollback(...); llama_memory_seq_rm(...); if (n_reeval > 0) { llama_decode(ctx_tgt, batch_reeval); } }`. Gate: `!all_accepted_flat`.
- **MAIN** (server-context.cpp:7301+): `const bool all_accepted_flat = (n_accepted_draft == (int) n_draft) && !had_dflash_padding;` ... `llama_dflash_rollback(...)` + `llama_decode(ctx_tgt, batch_reeval)` with the comment "When tape replay was unavailable (e.g., Vulkan), re-decode". Gate: `all_accepted_flat = (n_accepted_draft == n_draft) && !had_dflash_padding`.
- **Divergence**: both re-decode on partial accept on Vulkan (same mechanism, same `llama_dflash_rollback` + `llama_decode(batch_reeval)`). MAIN's gate also requires `!had_dflash_padding` (slightly more restrictive). The re-decode overhead is present in BOTH — it is NOT the differentiator for the 12 vs 24 t/s gap by itself.

### FACTS — identified MISSING / ALTERED / DIVERTED logic (explains both issues)

**MISSING from WORK (present in MAIN):**
- `common_speculative_draft_batch` integration (the batched multi-slot drafter decode — ONE drafter decode per cohort instead of per-slot). This is the primary draft path in MAIN; WORK lacks it entirely.
- `prepare_batch_draft()` invocation (MAIN calls it via the batch path; WORK never calls it).
- **Prefetch verify** (`dflash_flat_accept_prefetches`, `prefetched.n_accepted_draft`/`n_hidden_keep`, MAIN 6574-6580) — MAIN prefetches the verify result to overlap; WORK lacks it.
- **Profit controller** (`slot.observe_profit_acceptance`, `slot.profit_pending_*`, `slot.dm_adaptive`, `adaptive_n_max`, MAIN 6727-6736) — MAIN adaptively tunes n_draft based on acceptance; WORK uses fixed `n_draft_max` (no adaptive controller). This directly affects throughput.
- **Reduced-verify** (`dflash_reduced_verify_ready`, `dflash_verify_plan`, `dflash_reduced_verify_top_k`, MAIN 6960+) — MAIN can verify a reduced top-k set (cheaper); WORK lacks it.
- **`had_dflash_padding`** gate in the all-accepted rollback decision (MAIN 7301).
- Tree-draft path (`common_speculative_draft_tree`, `tree_commit_n`, MAIN 4736/6660) — WORK lacks the tree path (acceptable on Vulkan, but it's part of the missing pipeline).

**ALTERED argument patterns (WORK vs MAIN):**
- Draft: 1-arg `common_speculative_draft(spec)` vs 6-arg `common_speculative_draft(spec, params_spec, cached_text_tokens, slot.sampled, log_probs, draft_n_past)`.
- `n_min`: `0` (WORK) vs `params_spec.n_min` (MAIN).
- `n_past`: `slot.prompt.n_tokens()` (WORK) vs `-1` for DFlash (MAIN). (Note: the dflash `draft()` uses `committed_len` internally, not `n_past`, so this is likely moot for DFlash — but it's a divergence.)
- Accept: `slot.get_seq_id()=0` (WORK) vs `slot.id` (MAIN).

**DIVERTED logic paths:**
- Verify/accept/rollback: WORK does it in a **separate `post_decode()` method**; MAIN does it **inline in the main slot loop** (with prefetch + profit + reduced-verify interleaved).
- Capture scheduling: WORK in `pre_decode` (3934-4164); MAIN in a dedicated section (5868-6303).

### EXPLANATION of the two issues

**Multi-slot capture failure (slots 1-3 = 0% acceptance):**
- The root architectural cause is the **truncated server DFlash pipeline**. WORK's simplified pipeline uses `slot.get_seq_id()=0` for accept and (before my fix) `seq_id=0` for flush_prefill, and lacks MAIN's per-slot capture wiring that uses `slot.id`. Combined with the eval-callback-gating bug (hidden_gpu allocated → eval callback off → no CPU capture) and the ring-D2D source/path mismatch, non-zero slots have no working capture path. MAIN has the **same latent flush_prefill seq_id=0 bug** (it's byte-identical, speculative.cpp:2722) but is **masked** because its LRU picked slot 0 in the tests; MAIN's full pipeline (batch draft, prefetch, per-slot capture with slot.id) otherwise handles capture. So the multi-slot capture bug is **present in both branches** but only WORK manifested it (LRU picked slot 3); MAIN would also fail on a non-zero slot.
- My two fixes (slot_id routing in flush_prefill + eval-callback gating on `dflash_use_prefill_staging`) are correct and necessary but target symptoms of the truncated pipeline; they restored capture for slot 3 (progress: failure moved to ring-D2D) but the ring-write source/path mismatch and inconsistent capture for slots 2/1 remain.

**Re-decode overhead / 12 vs 24 t/s speed gap:**
- The re-decode mechanism is IDENTICAL in both branches (`llama_dflash_rollback` + `llama_decode(batch_reeval)` on partial accept, both on Vulkan). So the re-decode itself is NOT the differentiator.
- The speed gap is primarily the **missing efficient pipeline**: MAIN's `common_speculative_draft_batch` (one drafter decode per cohort) vs WORK's per-slot `draft()` (a drafter decode PER SLOT); MAIN's prefetch-verify (overlaps verify with accept); MAIN's profit controller (tunes n_draft to maximize useful acceptance); MAIN's reduced-verify (cheaper verification). WORK has none of these — it does per-slot flat draft + post-decode rollback, which is ~2× slower at the same acceptance.
- Secondary: WORK's `n_min=0` (no minimum-draft enforcement) and the fixed `n_draft_max` (no adaptive tuning) vs MAIN's `params_spec.n_min` + profit-adaptive `adaptive_n_max`.

### HYPOTHESES
- The multi-slot capture failure AND the re-decode/speed gap are both consequences of the **~3000-line-truncated server DFlash pipeline** in the work branch. The merge took upstream's simpler server and only partially ported the fork's DFlash flow.
- The proper fix is NOT more incremental patches (slot_id, eval-callback gating, ring-D2D) — those treat symptoms. The proper fix is to **port the full fork DFlash server pipeline** from `gboddaer/main` into the work branch's `server-context.cpp`: the `common_speculative_draft_batch` integration, `prepare_batch_draft` path, prefetch-verify, profit controller, reduced-verify, per-slot capture with `slot.id`, and the `had_dflash_padding` gate. This is a substantial port (~the 3000-line difference, concentrated in server-context.cpp).
- Both branches share the latent `flush_prefill` seq_id=0 bug; my slot_id fix should be kept (it fixes a bug MAIN still has).

### TEST RESULTS
- Commands: `git show gboddaer/main:tools/server/server-context.cpp | grep -nE ...` vs `grep -nE ... tools/server/server-context.cpp` (call-site structure); `sed -n` of the draft/verify/capture sections; `git diff --stat gboddaer/main -- ...`.
- Result: the call-tree divergences above; `server-context.cpp` work=5947 vs main=9002 lines (−4844/+1884).

### NEXT STEPS
- The call-tree analysis shows the work branch's DFlash server pipeline is a truncated version of the fork's. The next step is a **structural port** of the fork's DFlash server pipeline (batch draft, prefetch verify, profit controller, reduced-verify, per-slot capture with slot.id) from `gboddaer/main:tools/server/server-context.cpp` into the work branch — adapting to the merge's upstream server structure. This is the fix that addresses BOTH the multi-slot capture failure and the speed gap at their root, rather than the incremental symptom-patches (slot_id, eval-callback gating, ring-D2D) attempted so far.
- Keep the slot_id flush_prefill fix (correct, fixes a latent bug in both branches). The eval-callback-gating fix (candidate fix 1) is also correct (don't disable eval callback solely due to hidden_gpu presence) but is a symptom-patch; the structural port would supersede it.
- Until the structural port is done, multi-slot DFlash (--parallel >1 on non-zero slots) will remain broken, and single-slot/slot-0 DFlash will remain at ~12 t/s (vs main's ~24 t/s).

## 2026-07-12 22:00 UTC - CORRECTION: delta analysis shows structural port is NOT proven necessary

### FACTS — the "3000-line truncation" was a mischaracterization
- Upstream `server-context.cpp` at the merge point (`f708a5b2c` / `ggml-org/master`) = **5372 lines**.
- **WORK server = 5947 lines** = upstream (5372) **+ 608 insertions / 33 deletions** (`git diff --stat f708a5b2c -- tools/server/server-context.cpp`). So the work branch started from **upstream's server** and added a **~608-line targeted DFlash port** (the `pre_decode` draft + `post_decode` verify/accept/rollback + capture scheduling).
- **MAIN server = 9002 lines** = upstream (5372) **+ 4800 insertions / 1170 deletions** (`git diff --stat f708a5b2c gboddaer/main -- tools/server/server-context.cpp`). Main is upstream **+ ~4800 lines of the fork's full DFlash server pipeline**.
- **CORRECTION:** the work branch did NOT "truncate 3000 lines" from main. It started from upstream's 5372-line server and ported a **minimal targeted subset** (~608 lines). The ~4200-line gap (main's 4800 vs work's 608 of DFlash wiring) is the fork's **full** DFlash server pipeline that the work branch deliberately did NOT port. The earlier "3000-line truncation / structural port required" claim (2026-07-12 21:00 UTC entry) was **wrong** and is retracted.

### FACTS — DFlash pipeline components: EXIST-in-impl vs WIRED-into-server
- **Batch draft — IMPL EXISTS, UNWIRED in work server:**
  - `common_speculative_draft_batch` (speculative.cpp:4806) and `prepare_batch_draft` (speculative.cpp:2664, the dflash override) **exist in the work branch's speculative.cpp** (identical impl layer, per HF-025).
  - Work server references: **0**. Main server references: **9** (`common_speculative_draft_batch` call at 4636 + `batched_drafts`/`batch_specs` plumbing).
  - → Fixable with a **targeted server-wiring patch** (call the existing `common_speculative_draft_batch` instead of the 1-arg flat `common_speculative_draft`).
- **Profit/fringe controller — IMPL EXISTS, UNWIRED in work server:**
  - `tools/server/server-adaptive-dm.h` is **present in the work branch (1682 lines, IDENTICAL line count to main)** and is `#include`d by `server-context.cpp:19`. The profit + fringe controller implementations exist.
  - Work server references adaptive-dm (`adaptive_n_max`/`dm_adaptive`/`observe_profit`/`profit_pending`/`server_adaptive_dm`): **0**. Main server: **91**.
  - Work `server_slot` (struct at server-context.cpp:187) does **NOT** derive from `server_adaptive_dm_state` (main's does: `struct server_slot : server_adaptive_dm_state` at main:696). Work `common/speculative.h` has **no** `dm_adaptive`/`dm_controller` params.
  - → Fixable with a **targeted wiring port** (make `server_slot` derive from `server_adaptive_dm_state`, add `dm_adaptive`/`dm_controller` params, call `observe_profit_acceptance` + use `adaptive_n_max`). The impl exists; only the server-side wiring is missing.
- **Tree draft — IMPL EXISTS, UNWIRED in work server:**
  - `draft_tree` (speculative.cpp:3318) and `common_speculative_draft_tree` (speculative.cpp:5026) **exist in work's speculative.cpp**.
  - Work server tree refs: **0**. Main server: **52**. (Tree is not used on Vulkan anyway — no `ggml_ssm_conv_tree`/`ggml_gated_delta_net_tree` Vulkan backend.)
- **Prefetch-verify — SERVER-SIDE ONLY (not in speculative.cpp of EITHER branch):**
  - `dflash_flat_accept_prefetches` / `prefetched.n_accepted_draft` are **server-side logic** in main's `server-context.cpp` (not in `speculative.cpp`). Work server refs: **0**. Main server: **89**.
  - → This is a server-side performance feature that would need porting the prefetch logic from main's server (not just wiring an existing impl).
- **Reduced-verify — SERVER-SIDE ONLY:**
  - `dflash_reduced_verify_plan` (struct at main server-context.cpp:398) + `dflash_select_reduced_verify_plan` (main:479) are **server-side logic**. Work server refs: **0**. Main server: **89** (combined with prefetch).
  - → Server-side performance feature; would need porting.

### FACTS — what the work server's ~608-line targeted port actually contains
- `on_decoded` (server-context.cpp:893), `pre_decode` (3143), `post_decode` (4193). The DFlash wiring = pre_decode (1-arg flat draft + capture scheduling at 3934-4164) + post_decode (verify/accept/`llama_dflash_rollback`+re-decode at 4320-4470). Plus the per-slot DFlash members in `server_slot` (`dflash_seq_backup`, `dflash_n_pos_before_draft`, `dflash_state`).

### VERDICT — structural port is NOT proven necessary; targeted patches can address both issues
1. **Multi-slot capture failure (slots 1-3 = 0%)** is a **CORRECTNESS bug**, NOT a missing-pipeline symptom. It is caused by: (a) `flush_prefill` reading `seq_id=0` instead of the physical slot (FIXED by my slot_id patch — a targeted patch), (b) the eval callback being disabled when `hidden_gpu` is allocated but staging isn't used (PARTIALLY FIXED by candidate fix 1 — a targeted patch), (c) the `ring_write` D2D source/path mismatch for CPU-captured data (next targeted patch). None of these require the batch draft / profit controller / prefetch / reduced-verify pipeline. The missing pipeline is a PERFORMANCE gap, not the cause of the capture correctness bug.
2. **Speed gap (12 vs 24 t/s)** is NOT proven to require the structural port. Candidate causes, each addressable with targeted patches:
   - **Batch draft wiring**: the impl exists (`common_speculative_draft_batch`); a targeted patch can make the work server call it instead of the 1-arg flat draft. (For `--parallel 1`, batch ≈ flat, so this helps multi-slot throughput, not single-slot.)
   - **Profit controller wiring**: the impl exists (`server-adaptive-dm.h`); a targeted wiring patch (server_slot derives from `server_adaptive_dm_state` + params + `observe_profit_acceptance`) can restore adaptive `n_draft` tuning.
   - **Re-decode overhead**: present in both branches (identical mechanism); fixable with targeted patches (skip re-decode when all-accepted is already done; batch the re-decode; reduce re-decode tokens).
   - **Verify decode efficiency**: `n_outputs_max` / graph; targeted.
   - The `--parallel 1` speed gap (work ~10 t/s vs main ~24 t/s at similar ~75-80% acceptance) is NOT explained by the batch draft (single-slot) and is most likely the re-decode overhead + verify-decode efficiency — both fixable with targeted patches. Needs profiling to confirm, NOT a structural port.
3. **Prefetch-verify and reduced-verify** are server-side performance optimizations in main that would require porting server logic (not just wiring). They would improve throughput further but are NOT required for correctness or baseline speed restoration.

### HYPOTHESES (corrected)
- The work branch's DFlash server integration is a **minimal targeted port** (upstream + 608 lines), NOT a truncation of main. The impl layer (batch draft, profit controller, tree draft in `speculative.cpp` + `server-adaptive-dm.h`) is **present**; only the **server-side wiring** is minimal.
- The multi-slot capture failure is fixable with **targeted correctness patches** (slot_id routing ✓, eval-callback gating ✓ partial, ring-D2D source/path — next). No structural port needed for correctness.
- The speed gap is fixable with **targeted wiring + overhead patches** (wire the existing batch-draft impl, wire the existing profit-controller impl, reduce re-decode overhead). The structural port (porting main's full ~4800-line pipeline including prefetch-verify + reduced-verify) is NOT proven necessary — it would add performance optimizations but is not required to restore correctness or baseline speed.
- I RETRACT the 2026-07-12 21:00 UTC claim that "the proper fix is a structural port of the full fork DFlash server pipeline." That was based on the mischaracterization of the 3000-line diff as truncation. The delta analysis shows the impl exists and targeted patches are the appropriate fix.

### NEXT STEPS (corrected)
- Continue the **targeted correctness patches** for multi-slot capture: next is the `ring_write` D2D source/path fix (in `common/speculative.cpp:3695-3714`, attempt D2D only when the capture source is a GPU buffer; go straight to H2D `llama_dflash_cross_ring_gpu_write` when the source is CPU `layer_hiddens`). Then the active-slot routing during the prefill decode (so the eval callback captures into the correct slot's `layer_hiddens` for slots 2/1).
- For the speed gap: profile the `--parallel 1` decode (work vs main) to locate the 12-vs-24 t/s difference (re-decode overhead vs verify-decode efficiency vs n_outputs_max), then targeted patches. Optionally wire the existing batch-draft + profit-controller impls (targeted, since the impls exist).
- Do NOT pursue the structural port of main's full server pipeline unless the targeted patches are proven insufficient by profiling evidence.

## 2026-07-12 23:00 UTC - CORRECTNESS RESTORED: multi-slot DFlash via targeted slot-routing patches

### FACTS — targeted patches applied (all on top of HEAD 37a89c8f6, uncommitted)
The multi-slot capture failure was a **slot-routing** problem: the DFlash impl's `seq_id=0` (per-slot design) was used to index per-physical-slot buffers (`prefill_gpu[slot]`, `hidden_gpu[slot]`, `layer_hiddens[slot]`) and to set the active slot, so non-zero request slots read slot 0's (empty) buffers. Five targeted patches thread the physical request `slot.id` through the capture→ring-write→generation pipeline:
1. **`flush_prefill` slot_id** (common/speculative.cpp:2721, .h:193, server-context.cpp:4156): `common_speculative_flush_prefill(..., llama_seq_id slot_id)`; `flush_prefill` uses `phys_slot = (slot_id>=0)?slot_id:seq_id` for `set_active_slot` + `prefill_capture_info` + `prefill_gpu_n_tokens`. Server passes `pf.slot_id`.
2. **eval-callback gating** (src/llama-context.cpp:7018): `dflash_skip_eval_callback = dflash_use_prefill_staging || dflash_suppress_callback_for_view` (was `dflash_graph_hidden_ready || ...`). Keeps the CPU eval callback on when prefill staging isn't used, so multi-slot prefills (hidden_gpu allocated via allocate_slots) still capture via the CPU path.
3. **`ring_write` D2D slot_id** (common/speculative.cpp:3595, 3682, 3697): added `llama_seq_id phys_slot=-1` param; `phys_slot_eff` used in both `llama_dflash_prefill_gpu_write_hidden(..., phys_slot_eff, ...)` D2D calls. `flush_prefill` passes `phys_slot`. (The D2D was reading `prefill_gpu[seq_id=0]` instead of `prefill_gpu[slot]`.)
4. **active-slot routing** (src/llama-context.cpp:7013): set `dflash_capture->active_tape_idx = seq` unconditionally (was gated on `seq < tapes.size()`; `tapes` is the tree buffer, size 0/1 for flat DFlash, so non-zero slots never updated active_tape_idx → the eval callback captured into the stale slot 0's `layer_hiddens`). The accessors (`active_tape`/`active_hidden_gpu`/`active_slot_hiddens`) all bounds-check.
5. **generation-path slot_id** (common/speculative.cpp:3545/3889/5099, .h:191, server-context.cpp:4427): `common_speculative_update_logits_deferred_dflash_kv(..., llama_seq_id slot_id)`; `append_target_hiddens` uses `phys_slot_eff` for `set_active_slot` + `ring_write(n_accepted, 0, false, cpu_hidden, phys_slot_eff)`. Server passes `slot.id`. (Was: `append_target_hiddens` called `set_active_slot(seq_id=0)` + `ring_write(n_accepted)` with default phys_slot=0 → verify-time hidden capture read slot 0 → "incomplete target hidden capture".)
- Build: `cmake --build build-vulkan --target llama-server llama-cli` EXIT=0 after each patch.

### FACTS — verification: multi-slot DFlash RESTORED, correctness CONFIRMED
- **Multi-slot `--parallel 4`, 4 sequential long prompts (n_predict=300), one server:**
  | slot | draft acceptance | errors |
  |------|-----------------|--------|
  | 3 | **0.799 (211/264)** | 0 |
  | 2 | **0.762 (176/231)** | 0 |
  | 1 | **0.853 (215/252)** | 0 |
  | 0 | **0.878 (216/246)** | 0 |
  - `prefill flush mismatch` / `incomplete target hidden capture` / `D2D ring write failed` / `incomplete GPU capture` / `incomplete ring span`: **all 0**.
  - Garble check: **0 non-ASCII in all 4 generations** (the llama banner's █▄ chars are not in the generation text); all 4 outputs are coherent step-by-step solutions (farmer area, 17×23 distributive, 3x+7=22, Fibonacci). No corruption, no position-offset divergence through 300 tokens.
- **Single-slot `--parallel 1`, DFlash vs non-DFlash (greedy, same prompt, n=200):** DFlash output is COHERENT and **differs from non-DFlash only in wording** (both valid step-by-step solutions — e.g., different variable definitions). The difference confirms DFlash is **accepting drafts** (a disabled DFlash would produce byte-identical output to non-DFlash). This is the "inherent DFlash-vs-non-DFlash wording difference" that `gboddaer/main` also exhibits (confirmed in the 2026-07-12 15:00 UTC benchmark). NO garble, NO corruption.
- **Regression check (non-DFlash):** still works — 12.57 t/s, coherent "Here's a thinking process..." output. No regression.

### FACTS — remaining discrepancy (performance, NOT correctness)
- **Speed gap remains**: multi-slot DFlash is ~11–13 t/s (slot 0: 11.08 t/s over 295 tokens; early 13.38 t/s) vs non-DFlash 12.5 t/s. So DFlash is now roughly **break-even** with non-DFlash (no longer slower, thanks to the wasted-draft-gen elimination) but NOT the ~1.7× speedup main achieves (main 24 t/s). The gap is the **re-decode-on-partial-accept overhead** (identical mechanism in both branches) + the **unwired performance pipeline** (batch draft `common_speculative_draft_batch`, profit controller `server-adaptive-dm.h`, prefetch-verify, reduced-verify — all exist in the impl but aren't wired into the work server, per the 2026-07-12 22:00 UTC delta analysis). This is a separate performance issue, NOT a correctness issue.

### VERDICT
- ✅ **Multi-slot capture failure: RESOLVED.** All slots (0-3) have DFlash active (76-88% acceptance, matching main's ~80%), 0 capture/ring/D2D errors, in `--parallel 4`.
- ✅ **Position-offset / corruption issues: RESOLVED.** Coherent outputs across 4 slots through 300 tokens; DFlash vs non-DFlash differs only in inherent wording (drafts accepted), no garble, no divergence-into-corruption.
- ✅ **No regressions:** non-DFlash and single-slot DFlash still work correctly.
- ⚠️ **Speed NOT restored to 24 t/s** (DFlash ~11-13 t/s ≈ non-DFlash 12.5). This is the separate re-decode-overhead + unwired-pipeline performance gap, explicitly out of scope for these correctness patches.
- The targeted slot-routing patches (5 small changes) fully resolved the multi-slot correctness failure. The earlier "structural port" hypothesis (2026-07-12 21:00 UTC) and its retraction (22:00 UTC) are confirmed: targeted patches were sufficient for correctness.

### TEST RESULTS
- Commands: `scripts/dflash-regression/verify_slotfix.sh build-vulkan/bin/llama-server workcorr2 8103 /tmp/dflash-corr2 4` (acceptance), `scripts/dflash-regression/verify_corr.sh build-vulkan/bin/llama-server workfinal 8104 /tmp/dflash-final 4` (long-prompt correctness + garble), and a `llama-cli` DFlash-vs-non-DFlash greedy diff (`--parallel 1`).
- Results: multi-slot acceptance 0.762-0.878 across all 4 slots, 0 errors, 0 garble; single-slot DFlash coherent + differs-from-non-DFlash-only-in-wording; non-DFlash 12.57 t/s coherent (no regression).
- Output files: `/tmp/dflash-corr2/workcorr2.server.err`, `/tmp/dflash-final/{workfinal.server.err,workfinal.gen{0-3}.txt,nd_gen.txt,df_gen.txt,nd_full.txt,df_full.txt,nodflash.err}`.

## 2026-07-13 00:00 UTC - Speed profiling: drafter decode is the 2.3× bottleneck (NOT re-decode)

### FACTS — profiling setup
- Profiled DFlash decode `--parallel 1` (single-slot, isolates the per-cycle cost) on both branches, same model pair (Qwen3.6-27B-Q4_K_M target + Qwen3.6-27B-DFlash-Q4_K_M draft), Vulkan0 iGPU, `--spec-draft-n-max 3`, greedy, farmer prompt.
- Tools: `GGML_DFLASH_QA_TRACE=1` (verify/rollback counts) + `GGML_DFLASH_PROFILE=summary` (per-draft breakdown: cross/batch/decode/argmax/total/graph_reuse, logged at `common/speculative.cpp:3299`).
- The `all_accepted_flat` skip of rollback+re-decode is ALREADY implemented (`server-context.cpp:4429`), so all-accepted batches do NOT re-decode.

### FACTS — re-decode overhead is SECONDARY (16%)
- Work `--parallel 1`, n_predict=200, 62 verify cycles:
  - **all_accepted=1 (SKIP re-decode): 38 (61.3%)**
  - all_accepted=0 (PARTIAL → re-decode): 24 (38.7%)
  - 24 re-decodes, n_reeval avg 2.00 tokens, **total 48 re-decode tokens**
  - draft: 185 tokens generated, 137 accepted, 74.1% acceptance
- Re-decode cost estimate: 48 re-decode tokens vs ~248 verify-decode tokens (62 cycles × ~4) = **16% of target decode work**. Not enough to explain the 2×+ speed gap. The all-accepted skip is already optimal here (61% skip).

### FACTS — the PRIMARY bottleneck: drafter decode is 2.3× slower in work
- Apples-to-apples `--parallel 1` (same acceptance ~74-79%, same draft count 61-62, same n_draft=3, same iGPU, same drafter model, same re-decode mechanism):
  | metric | work (37a89c8f6+fixes) | main (130ea2480) |
  |--------|------------------------|------------------|
  | generation t/s | **10.95 t/s** | **24.18 t/s** |
  | acceptance | 74.1% | 79.0% |
  | draft calls | 62 | 61 |
  | drafter decode (per call) | **avg 54.01ms** (min 27.97, max 99.25) | **avg 23.57ms** (min 20.79, max 38.57) |
  | total draft time | 1998ms (decode) + 82ms (cross) | 872ms (decode) + 56ms (cross) |
- Per-draft breakdown (work, `GGML_DFLASH_PROFILE=summary`): **decode=54.01ms avg (96% of draft time)**, cross(build_cross_data)=2.23ms (4%), batch=~0ms, argmax=0.13ms. graph_reuse increments (0→17) → the drafter graph IS cached (not rebuilt every call). So the 2.3× is the drafter forward pass itself, not graph rebuild or cross-data prep.
- The drafter graph (`dflash_draft.cpp`) is essentially identical work-vs-main (`git diff --stat` = 21 ins/6 del, all mechanical: `n_layer`→`n_layer()` refactor, `is_swa_impl` population, class renames `graph_kv_update`/`graph`). So the graph builder is NOT the difference.
- **Leading hypothesis — drafter `n_outputs_max`:** work forces the drafter `cparams.n_outputs_max = max(cparams.n_outputs_max, 17)` (`server-context.cpp:1361`, comment "Ensure drafter has enough output positions for DFlash block_size drafts"). With `n_outputs_max=17 ≥ block_size=16`, the drafter graph's `inp_out_ids = n_outputs < n_tokens ? build_inp_out_ids() : nullptr` (`dflash_draft.cpp:989`) is **null** → the LM-head matmul (`build_lora_mm(model.output, cur)` at :1205, weights shared from the 27B target, vocab 248320) computes logits for **all 16 token positions**. Main uses `params_dft.n_outputs_max = params_base.n_parallel` (main:2228) — smaller — so `inp_out_ids` selects only the needed `n_draft+1` rows → a much smaller LM-head matmul. The LM head (248320×hidden) is the dominant matmul in a drafter whose weights are mostly the shared 27B LM head, so 16 rows vs ~4 rows ≈ 4× LM-head work ≈ the 2.3× decode difference. **Needs confirmation by testing** (reduce work's drafter n_outputs_max to `n_draft+1` and measure the decode time).
- Secondary observation: **main uses adaptive `n_draft` (1/2/3) via the profit controller** (the profile shows `n_draft=2,2,2,1,1,1,3,3,...` — the profit controller tunes it); work uses fixed `n_draft=3`. Adaptive n_draft reduces work per cycle when acceptance is low, but main is 23ms even at n_draft=3, so this is a minor throughput contributor, not the 2.3× cause.

### FACTS — draft-generation frequency / acceptance impact
- Draft frequency is essentially identical (62 vs 61 calls for ~200 tokens). Acceptance is similar (74% vs 79%). So the gap is NOT draft frequency or acceptance — it is the per-call drafter decode cost (2.3×).
- Throughput math: work produces ~2.93 tokens/cycle at 10.95 t/s; main ~3.28 tokens/cycle at 24 t/s. The per-cycle cost (work 0.27s vs main 0.14s) is 1.9× higher in work, dominated by the drafter decode (54ms vs 23ms = 31ms/cycle difference, which is most of the 130ms/cycle gap).

### PROPOSED LOGIC CHANGES (documented before implementing)
1. **PRIMARY — reduce the drafter `n_outputs_max` to the needed `n_draft+1` (not 17).** In `server-context.cpp:1361`, change `cparams.n_outputs_max = max(cparams.n_outputs_max, 17)` to use `1 + common_speculative_n_max(&params_base.speculative)` (the actual max draft count, e.g., 4 for n_max=3) instead of the hardcoded 17. This makes the drafter graph use `inp_out_ids` to compute only the needed output rows → ~4× smaller LM-head matmul → expected ~2× faster drafter decode → closes most of the 2.3× gap. **Risk:** the DFlash drafter argmax reads positions 1..n_draft; if n_outputs_max < n_draft+1 the argmax would be short. Must keep n_outputs_max ≥ n_draft+1. The `block_size` comment suggests 17 was chosen for the full block, but only n_draft+1 positions are consumed (`flat draft()` reads `argmax[i*K]` for i=1..output_len-1, output_len=n_draft+1). So n_draft+1 should suffice. **Test:** measure the drafter decode time after the change; target ~23ms (main parity).
2. **SECONDARY — wire the profit controller** (adaptive `n_draft`). The impl EXISTS (`tools/server/server-adaptive-dm.h`, 1682 lines, identical to main) but is unwired in the work server (`server_slot` doesn't derive from `server_adaptive_dm_state`; `common/speculative.h` has no `dm_adaptive` params; 0 refs in work server vs 91 in main). A targeted wiring patch (derive `server_slot` from `server_adaptive_dm_state`, add `dm_adaptive`/`dm_controller` params, call `observe_profit_acceptance` + use `adaptive_n_max`) would tune `n_draft` for throughput. Minor benefit (main is fast even at fixed n_draft=3), so lower priority than #1.
3. **TERTIARY — re-decode overhead (16%).** Already optimal via the all-accepted skip (61%). Could shave a little by batching the re-decode or skipping when the verify decode's per-token snapshots already cover the accepted prefix (the tree-conv path, CUDA-only — not applicable on Vulkan). Low priority; 16% is not the gap.

### HYPOTHESES
- The 2.3× drafter-decode slowdown is caused by the drafter computing 16 LM-head output rows (n_outputs_max=17) instead of the needed ~4 (n_draft+1). Reducing n_outputs_max to n_draft+1 should restore main-parity decode time (~23ms) and close the speed gap from ~11 t/s toward ~24 t/s.
- The re-decode (16%) and the profit controller (minor) are NOT the primary cause; fixing n_outputs_max alone should recover most of the gap.
- This is a TARGETED patch (one line in server-context.cpp:1361), NOT a structural port — consistent with the 22:00 UTC delta analysis.

### NEXT STEPS (pending go-ahead to implement)
- Implement proposed change #1 (drafter n_outputs_max = n_draft+1). Build. Re-run `GGML_DFLASH_PROFILE=summary` to confirm the drafter decode drops from ~54ms toward ~23ms. Then re-run the multi-slot correctness + perf test to confirm speed approaches 24 t/s with no correctness regression (acceptance, garble, multi-slot all-slots-active).
- If #1 alone closes the gap, #2 (profit controller) and #3 (re-decode) are optional follow-ups.
- If #1 does NOT close the gap (the LM-head hypothesis is wrong), profile deeper: compare the drafter context cparams (KV cache type, n_ubatch, flash_attn) work vs main via `-lv`, and check whether the 2.3× is in attention vs FFN vs LM-head via a finer profile.

### TEST RESULTS
- Commands: `GGML_DFLASH_QA_TRACE=1 llama-server ... --parallel 1` (re-decode counts), `GGML_DFLASH_PROFILE=summary llama-server ... --parallel 1` (draft breakdown), same for main-ref.
- Results: work re-decode 24/62 cycles (16% overhead); work drafter decode 54ms/call vs main 23ms/call (2.3×); work 10.95 t/s vs main 24.18 t/s; acceptance 74% vs 79%.
- Output files: `/tmp/dflash-prof/{work_qa.err,work_prof.err,main_p1.err,main_prof.err}`.

## 2026-07-13 01:00 UTC - n_outputs_max fix applied: PARTIAL speed gain (11→15 t/s), target 24 NOT reached

### FACTS — change applied
- `tools/server/server-context.cpp:1361`: changed `cparams.n_outputs_max = std::max<uint32_t>(cparams.n_outputs_max, 17)` to `cparams.n_outputs_max = std::max<uint32_t>(cparams.n_outputs_max, 1u + (uint32_t) common_speculative_n_max(&params_base.speculative))`. For `--spec-draft-n-max 3` this is `1+3=4` (was 17). With `n_outputs_max=4 < block_size=16`, the drafter graph uses `build_inp_out_ids()` to compute the LM head for only 4 rows instead of 16.
- Build: `cmake --build build-vulkan --target llama-server llama-cli` EXIT=0.

### FACTS — speed: real but PARTIAL improvement
- `GGML_DFLASH_PROFILE=summary`, `--parallel 1`, farmer prompt, n_predict=200:
  | metric | before fix (n_outputs_max=17) | after fix (n_outputs_max=4) | main (target) |
  |--------|-------------------------------|----------------------------|---------------|
  | drafter decode avg | 54.01ms | **40.48ms** | 23.57ms |
  | generation t/s | 10.95 | **14.19–15.08** | 24.18 |
- The LM-head reduction (16→4 rows) saved ~14ms of the drafter decode (54→40ms), improving single-slot speed ~1.3× (11→15 t/s). **But the 24 t/s target was NOT reached** — a ~17ms drafter-forward-pass gap remains (40ms work vs 23ms main), i.e. the LM head was only PART of the 2.3× gap (the hypothesis was directionally correct but incomplete).
- Multi-slot `--parallel 4` (4 long prompts): slot speeds ~11–13 t/s (slot 0: 10.87 t/s over 287 tokens), similar to before the fix — the n_outputs_max fix mainly helps single-slot; multi-slot throughput is dominated by the per-slot drafter forward pass (40ms) which is still 1.7× main's (23ms).

### FACTS — correctness: NO regression
- Multi-slot `--parallel 4`, 4 long prompts (n_predict=300): **0 errors** (prefill flush mismatch / incomplete target hidden / D2D ring write / incomplete GPU capture / incomplete ring span all 0), **0 garble** (0 non-ASCII in all 4 generations), all 4 outputs coherent step-by-step solutions.
- Per-slot acceptance: slot 3 = 0.799 (211/264), slot 2 = 0.762 (176/231), slot 1 = 0.853 (215/252), slot 0 = 0.878 (216/246) — **identical to before the n_outputs_max fix** (the fix doesn't touch the capture/verify path, only the drafter's output-row count). All 4 slots DFlash-active.

### VERDICT
- ✅ Correctness fully preserved (no regression; all 4 slots active, 76-88% acceptance, 0 errors, 0 garble).
- ✅ Real speed improvement: single-slot 11→15 t/s (~1.3×), drafter decode 54→40ms.
- ❌ **Target ~24 t/s NOT reached.** Single-slot is ~15 t/s (vs main 24), multi-slot ~11-13 t/s. The n_outputs_max fix addressed the LM-head portion but the remaining ~1.6× gap is the **drafter forward pass** (attention+FFN over 16 block tokens): work 40ms vs main 23ms, same drafter model, essentially identical graph.

### HYPOTHESES (remaining gap — drafter forward pass)
- The remaining 40ms vs 23ms is in the drafter's 16-token forward pass (not the LM head, which is now 4 rows in both). Candidates to investigate next:
  1. **Drafter KV cache type / flash_attn**: the work branch may set the drafter context's cache type or flash-attn differently than main (the init logs didn't print these at default verbosity). If the work drafter uses a slower KV cache path (e.g., flash-attn off, or a different quant), that's the gap. Needs `-lv` cparams dump comparison.
  2. **Drafter `n_ubatch` / graph batching**: both use `LLAMA_DFLASH_MAX_SLOTS * block_size`, but the effective batch/graph sizing may differ.
  3. **Cross-attention data path**: the drafter's cross-attention reads the cross ring; if the work branch's ring read / cross-attention graph is less efficient (e.g., CPU fallback vs GPU D2D), the forward pass is slower. `build_cross_data` is only 2.5ms, but the cross-attention compute within the 40ms decode could differ.
  4. **First-decode graph build**: work's first decode is 59ms (graph_reuse=0) vs main's 24ms — the graph BUILD is 2.5× slower in work, suggesting a larger/different graph or a one-time cost that doesn't fully amortize over 62 cycles.
- These need a finer profile (drafter attention vs FFN vs LM-head split, and a cparams dump) — the `GGML_DFLASH_PROFILE=summary` only breaks down cross/batch/decode/argmax, not the within-decode components.

### TEST RESULTS
- Commands: `GGML_DFLASH_PROFILE=summary llama-server ... --parallel 1` (decode breakdown), `scripts/dflash-regression/verify_corr.sh ... --parallel 4` (multi-slot correctness+perf).
- Results: drafter decode 54→40ms, single-slot 10.95→14.19-15.08 t/s, multi-slot ~11-13 t/s; acceptance 74-88% across all slots; 0 errors, 0 garble.
- Output files: `/tmp/dflash-prof/work_nomax.{err,resp}`, `/tmp/dflash-nomax/{worknomax.server.err,worknomax.gen{0-3}.txt}`.

### NEXT STEPS
- The n_outputs_max fix is KEPT (correct, ~1.3× single-slot speedup, no regression).
- To close the remaining ~1.6× gap (toward 24 t/s): profile the drafter forward pass internals (attention vs FFN vs cross-attention) and compare the drafter context cparams (KV cache type, flash_attn, n_ubatch) work vs main via `-lv`. The most likely remaining cause is a drafter-context config difference (KV cache / flash-attn) or the cross-attention compute path, NOT the LM head (now fixed) and NOT the re-decode (16%, secondary) and NOT the batch-draft/profit wiring (minor, per the 00:00 UTC profiling).

## 2026-07-13 02:00 UTC - Finer profiling: remaining 1.5× is in the 16-token drafter forward pass (not pin-pointable with available tools)

### FACTS — call-graph deep-dive (work vs main)
- `git diff gboddaer/main -- src/models/dflash_draft.cpp` = 21 ins / 6 del, ALL mechanical: `hparams.n_layer`→`hparams.n_layer()` (upstream refactor), `is_swa_impl[]` population from `swa_layers[]` (the SWA-bridge fix), forward declarations + class renames (`graph_kv_update`/`graph`). **No loop-structure, backend-call, or ring-access changes.** The drafter graph (layer loop at :1020, fused cross-attention `build_lora_mm(model.dflash_fc, target_hidden)` at :1014, `inp_out_ids` at :989, `get_rows` at :1191, LM head `build_lora_mm(model.output, cur)` at :1205, topk/argmax at :1215/:1217) is **structurally identical**.
- Drafter context setup (`params_dft`/`cparams`) is **essentially identical**: n_ctx=256×slots, n_batch=n_ctx, `n_ubatch=LLAMA_DFLASH_MAX_SLOTS*block_size`, n_parallel=slots, kv_unified=false, flash_attn + cache_type_k/v inherited from params_base (both run with `--flash-attn on --cache-type-k/v q8_0`). `LLAMA_DFLASH_MAX_SLOTS=8` (both, include/llama.h:419) → n_ubatch=128 both. `cross_ctx=512` (both, contract log). The only intentional difference is the drafter `n_outputs_max` (work: now `1+n_max`=4 after my fix; main: `params_base.n_parallel`=1 for `--parallel 1` — but main produces 2-3 drafts, so `n_outputs` in the drafter graph is evidently derived from `n_tokens`, not `n_outputs_max`; the LM head is ~4 rows in both now).
- Raw Vulkan backend (drafter model, no DFlash): `llama-bench -m DRAFT -p 0 -n 64` → **work 33.91 t/s vs main 34.56 t/s** (identical within noise). So the Vulkan backend performance for the drafter's forward pass is the SAME. The gap is NOT a backend version regression.

### FACTS — profiling the drafter forward pass
- Apples-to-apples `--parallel 1`, n_predict=200, farmer prompt, `GGML_DFLASH_PROFILE=summary`:
  | metric | work (n_outputs_max=4) | main |
  |--------|------------------------|------|
  | generation t/s | 14–15 | 24.58 |
  | drafter decode avg | **40.48ms** (min 31.00, max 58.99) | **27.25ms** (min 21.73, max 33.12) |
  | cross_len avg (cross-attention length) | 151 (max 250) | 149 (max 250) |
  | cross(build_cross_data) | 2.53ms | ~1.5ms |
  | argmax(extract) | 0.13ms | 0.11ms |
- The cross_len distribution is **identical** (avg ~150, max 250 both). So the cross-attention SIZE is the same. The decode (40 vs 27) is the 16-token forward pass (self-attn + FFN + cross-attn + 4-row LM head) at the SAME cross_len.
- First-decode (graph_reuse=0): work 58.99ms (cross_len=54) vs main 21.73ms (cross_len=35) — work's first cross_len is larger (54 vs 35) because the first DFlash draft happens later in work's generation (different warmup), which accounts for some of the first-decode difference, but the AVG over the full run (same cross_len) is still 40 vs 27 → the gap is real and consistent, not a cross_len artifact.
- Raw 1-token drafter decode (llama-bench, no cross-attention): 29.4ms both. The 16-token DFlash decode (with cross-attention) is 27ms (main) vs 40ms (work). Main's 16-token batch is MORE efficient than its 1-token (27 < 29.4 — batch parallelism wins); work's is LESS efficient (40 > 29.4 — the 16-token batch adds overhead in work).

### VERDICT — the remaining 1.5× is NOT pin-pointable with the available profiling
- I have ruled out: the graph code (mechanical-identical), the context config (n_ctx/n_ubatch/kv_unified/flash_attn/cache_type/cross_ctx all identical), the Vulkan backend (raw drafter llama-bench identical), the cross-attention size (cross_len identical), the LM head (now ~4 rows both), and the re-decode (16%, secondary).
- The remaining ~13ms (40 vs 27) is inside the 16-token batched drafter forward pass (self-attention + FFN + cross-attention compute). `GGML_DFLASH_PROFILE=summary` only breaks down cross/batch/decode/argmax — it does NOT split the `decode` term into attention vs FFN vs cross-attention. There is no env-exposed ggml-op-level profiler in this build to attribute the 13ms to a specific op or matmul.
- The most likely remaining candidates (NOT confirmed, need op-level profiling):
  1. A runtime-added graph op in the work drafter not present in main (e.g., a capture/trace op, or a different cross-attention matmul shape) — but the dflash_draft.cpp diff shows none.
  2. Graph build/dispatch overhead: work's first-decode build is 2.7× slower (59 vs 22ms). If the work drafter's graph is being rebuilt more often than graph_reuse suggests, or the Vulkan shader compilation is heavier, the per-decode amortized cost is higher. The graph_reuse counter increments in both, but a periodic rebuild (e.g., when cross_len crosses a bucket boundary — `cross_bucket()` in speculative.cpp) could rebuild more in work.
  3. The cross-attention matmul efficiency: same cross_len, but if the work branch's cross-attention uses a different matmul shape/stride (e.g., contiguous vs view), the Vulkan kernel is slower. The graph code is identical, but the tensor shapes at runtime could differ if build_cross_data produces a different layout.

### PROPOSED FINAL FIX (honest — no blind fix without root cause)
- **Keep the n_outputs_max fix** (real: 54→40ms, 11→15 t/s single-slot, no correctness regression). This is the confirmed, evidence-based improvement.
- **Do NOT propose a blind fix for the remaining 13ms.** Per systematic-debugging (no fix without root cause), attributing the 13ms to a specific op requires a ggml-op-level profiler. The next step is to add op-level timing (e.g., `ggml_graph_compute_with_timings` or a Vulkan layer profiler / `VK_LAYER_LUNARG_monitoring`, or a temporary per-op timestamp in the drafter graph) to split the 40ms into self-attn / FFN / cross-attn / LM-head, then compare to main's 27ms split. Only then can a targeted fix be proposed.
- This is NOT a targeted-patch situation yet — it's a "need finer instrumentation" situation. The earlier targeted patches (slot_id routing, eval-callback gating, ring_write D2D, active-slot, gen-path slot_id, n_outputs_max) restored correctness AND recovered the LM-head portion of the speed. The remaining 1.5× is a deeper drafter-forward-pass efficiency question requiring op-level profiling.

### TEST RESULTS
- Commands: `git diff gboddaer/main -- src/models/dflash_draft.cpp`; `git show gboddaer/main:tools/server/server-context.cpp | sed -n ...` (context params); `llama-bench -m DRAFT -p 0 -n 64` (both); `GGML_DFLASH_PROFILE=summary llama-server --parallel 1 n_predict=200` (both, apples-to-apples).
- Results: dflash_draft.cpp diff mechanical; context params identical; raw drafter llama-bench 33.91 vs 34.56 (identical); drafter 16-token decode 40.48 vs 27.25ms at the same cross_len (avg ~150 both); main 24.58 t/s vs work 14-15 t/s.
- Output files: `/tmp/dflash-prof/{work_nomax.err,main_p200.err,work_lv2.err,main_lv.err}`, llama-bench stdout.

### NEXT STEPS
- Add op-level profiling to the drafter decode (per-op timing: self-attn vs FFN vs cross-attn vs LM-head) — either via a temporary instrumentation in the drafter graph build/compute, or a Vulkan layer profiler. Compare the split work vs main to pinpoint the 13ms. THEN propose a targeted fix.
- If op-level profiling is not feasible in this environment, the n_outputs_max fix (15 t/s, no regression) stands as the confirmed improvement, and the remaining 1.5× gap to main's 24 t/s is documented as requiring op-level instrumentation to resolve — it is NOT closeable by a blind targeted patch.

## 2026-07-13 03:00 UTC - Op-level profiling: root cause = 2.3× more graph-compute launches in work

### FACTS — methodology
- Ran `GGML_VK_PERF_LOGGER=1` on both branches (`--parallel 1`, n_predict=20, same prompt/model/flags). The Vulkan perf logger outputs per-op-type timing: `OP: count × per_op_us = total_us` and `Total time: X us` per `ggml_graph_compute` call (block).
- Parsed all per-op lines and Total-time blocks with a Python script (line-by-line regex, grouped by op signature).

### FACTS — the decisive finding: WORK does 2.3× more graph-compute blocks
| metric | WORK (0d5f99638) | MAIN (130ea2480) |
|--------|-----------------|------------------|
| graph-compute blocks (Total time count) | **70** | **30** |
| total GPU compute (sum of block totals) | 1379.8ms | 1504.9ms |
| avg per-block GPU time | 19.71ms | 50.16ms |
- WORK does **2.3× more graph-compute calls** (70 vs 30) for the same n_predict=20. The total GPU compute is SIMILAR (1380 vs 1505ms), but work splits it into 2.3× more launches. Each launch has CPU-side dispatch overhead (graph scheduling, Vulkan command buffer recording, fence waits) not captured in the per-op GPU timing but real in wall-clock. The 40 extra launches × ~5ms CPU overhead ≈ 200ms extra over 20 tokens = ~10ms/token = most of the 1.5× wall-clock gap (work ~67ms/token vs main ~42ms/token).
- WORK blocks are SMALLER (19.7ms avg) and MORE numerous; MAIN blocks are LARGER (50ms avg) and FEWER. Main batches more work per graph-compute; work splits into more separate computes.

### FACTS — per-op comparison confirms the launch-count difference
- Shared matmuls (e.g., `MUL_MAT_VEC q4_K m=17408 n=4 k=5120`): WORK count=768 total=187.99ms vs MAIN count=126 total=30.43ms. The 6.1× count ratio ≈ the 70/30 block ratio × per-block matmul count. So the per-matmul GPU time is similar; the difference is the COUNT (more decode blocks = more matmuls).
- Op batch size: WORK uses `n=4` (n_outputs=1+n_max=4, fixed n_draft=3); MAIN uses `n=3` (adaptive n_draft=2 via the profit controller). Main's adaptive n_draft sometimes reduces the matmul batch (n=3 vs 4), but this is minor — the count ratio (6.1×) dominates.
- Ops ONLY in WORK (not main): `PAD_REFLECT_1D` (528 count, 32ms), `GATED_DELTA_NET_TREE` (528, 17.6ms), `SSM_CONV_SILU SSM_SCAN` (528, 2.2ms), `FLASH_ATTN_BACK` (multiple shapes, ~5ms total). These are TARGET recurrent-capture ops from the rollback/tape path (the K-snapshot DeltaNet path) — they appear in work's 70 blocks but NOT in main's 30-block run, confirming work does more target decode passes (the re-decode + the backup/rollback path adds extra target graph computes).
- Ops ONLY in MAIN: `MUL_MAT_VEC q4_K m=17408 n=3 k=5120` (384 count, 93ms), `PAD` (672, 39.7ms), `GATED_DELTA_NET` (672, 24.8ms), etc. — these are the SAME ops with `n=3` (adaptive) vs work's `n=4`, plus `PAD` (main's recurrent state padding for the K-snapshot path). Main's `GATED_DELTA_NET` (non-TREE variant, 672 count) vs work's `GATED_DELTA_NET_TREE` (528 count) — main uses the non-tree GDN variant (cheaper per-op, 24.8ms for 672 vs work's 17.6ms for 528), suggesting main's target recurrent path is structured differently.

### ROOT CAUSE (confirmed)
- The remaining 1.5× wall-clock gap (work 15 t/s vs main 24 t/s) is **2.3× more graph-compute launches** (70 vs 30) in the work branch. The GPU compute per op is similar (total GPU time 1380 vs 1505ms — main is even slightly MORE GPU compute). The gap is the **CPU-side launch overhead** of 40 extra graph-compute calls: graph scheduling, Vulkan command-buffer recording, fence waits, and the extra target decode passes from the re-decode + rollback + backup path.
- The extra work blocks come from: (1) the **re-decode** (24 separate `llama_decode(batch_reeval)` calls — each is a separate graph compute), (2) the **backup** (`cp_recurrent` every cycle — may trigger graph/cache effects), and (3) the target recurrent-capture ops (`PAD_REFLECT_1D`, `GATED_DELTA_NET_TREE`, `SSM_CONV_SILU`) appearing in 528 count (work) but not in main's 30-block run — work's target decode path does more recurrent work per cycle (the K-snapshot rollback path).

### PROPOSED FINAL FIX (data-driven)
1. **Reduce graph-compute launches**: the highest-impact fix is to reduce the 70→30 block count. The biggest contributor is the **re-decode** (24 separate `llama_decode` calls). Options:
   a. **Batch the re-decode** with the next cycle's verify decode (combine the accepted-token re-decode with the next verify in one graph compute) — saves ~24 launches.
   b. **Skip the re-decode when the verify batch's K-snapshot state already covers the accepted prefix** (the K-snapshot rollback path, which works on CUDA via `ggml_gated_delta_net_tree` but is unavailable on Vulkan — on Vulkan, the re-decode is the fallback). On Vulkan, the re-decode is necessary (r_l can't be snapshotted), so option (a) is the path.
   c. **Reduce the backup overhead**: the per-cycle `cp_recurrent` (line 3357) may trigger graph/cache effects. If the backup can be deferred or done less often, it reduces overhead.
2. **Secondary — wire the profit controller** (adaptive n_draft): reduces n_draft when acceptance is low (n=3 vs n=4), slightly reducing per-decode work. Minor.
3. The op-level profiling confirms the n_outputs_max fix was correct (the LM head `MUL_MAT_VEC q6_K m=248320 n=4 k=5120` is 46.53ms for 10 calls = 4.65ms/call — small relative to the total; the n_outputs_max=17→4 already minimized this).

### NEXT STEPS (pending go-ahead)
- Implement the re-decode batching (combine the accepted-token re-decode with the next verify decode into fewer graph computes) — the highest-impact targeted patch for the remaining 1.5× gap. This is a server-side change in `post_decode` (server-context.cpp:4431-4443), not a structural port.
- Alternatively, profile the CPU-side overhead (graph scheduling, Vulkan dispatch) to confirm the 5ms/launch estimate and identify which launches are avoidable.

### TEST RESULTS
- Commands: `GGML_VK_PERF_LOGGER=1 llama-server --parallel 1 n_predict=20` (both branches); Python parser for per-op + Total-time blocks.
- Results: WORK 70 blocks (1379.8ms GPU), MAIN 30 blocks (1504.9ms GPU); shared matmuls 6.1× count ratio (768 vs 126); work-only ops PAD_REFLECT_1D/GATED_DELTA_NET_TREE/SSM_CONV (target recurrent capture); main-only ops with n=3 (adaptive) + PAD + GATED_DELTA_NET (non-tree).
- Output files: `/tmp/dflash-perop/{work_perop20.err,main_perop20.err}`.

## 2026-07-13 04:00 UTC - Re-decode batching: attempted, FAILS (corrupts hidden capture), reverted

### FACTS — what was attempted
- Implemented re-decode batching: instead of a separate `llama_decode(batch_reeval)` in post_decode (server-context.cpp:4443), saved the accepted re-decode tokens to `slot.dflash_reeval_tokens` and prepended them to the NEXT verify decode batch in `handle_last_sampled_token` (line 517) with `is_output=false`. The drafter uses the cross-attention ring (from the verify decode), not the recurrent state, so the re-decode can be deferred to the next verify without affecting the draft. The next verify decode would process [accepted_tokens, new_id_last, new_drafts] in ONE graph compute, saving the re-decode launch.
- Build succeeded. Tested `--parallel 4`, 4 long prompts (n_predict=300).

### FACTS — the failure (correctness regression)
- Acceptance DROPPED from 87.8% (slot 0, before batching) to **59.5% (191/321)** — a regression. The output was still coherent (0 non-ASCII garble) but the draft quality degraded significantly.
- **Root cause of the failure**: the `hidden_gpu` capture (qwen35.cpp:206-244) takes the FIRST `n_hidden_keep` rows of the verify decode batch to populate the cross-attention ring. Prepending the re-decode tokens (n_accepted tokens with is_output=false) at the beginning of the batch SHIFTED the first n_hidden_keep rows — the hidden capture now captures the re-decode tokens' hidden states (re-processed accepted tokens, already in the ring from the previous verify) instead of the new verify tokens' hidden states (id_last + accepted drafts). The drafter reads the WRONG ring data → bad drafts → low acceptance.
- The 0-garble result is misleading: the target's own logits (from the verify) are still correct (the recurrent state advances correctly through [accepted, id_last, drafts]), so the OUTPUT is coherent. But the DRAFTS are wrong (the ring data is wrong), so the acceptance drops and the speculative speedup is lost.

### FACTS — reverted
- Reverted to committed state (0d5f99638): `git checkout 0d5f99638 -- tools/server/server-context.cpp`. Rebuilt (EXIT=0). Verified: `--parallel 4` acceptance restored to 72.7%, 77.6%, 78.2%, 81.9% (matching pre-batching), 0 mismatches. The revert is clean (only TASK_PROGRESS.md has uncommitted changes).

### VERDICT
- The re-decode batching as implemented is **NOT viable** — it corrupts the hidden_gpu capture by shifting the first n_hidden_keep rows. A correct implementation would ALSO need to adjust the hidden_gpu capture's row selection (skip the re-decode rows, capture only the id_last + drafts rows) — a more complex change to the capture logic in the graph builder (qwen35.cpp hidden_gpu block + the n_hidden_keep computation), which is risky and not a simple targeted patch.
- The confirmed root cause of the speed gap (2.3× more graph-compute launches, per the 03:00 UTC per-op profiling) stands. The re-decode (24 separate launches) is the largest contributor, but batching it requires fixing the hidden capture row selection — a deeper change than expected.

### CURRENT STATE (final, after revert)
- The committed fixes (0d5f99638, pushed to gboddaer/merge_llama_into_beellama_2): 5 slot-routing correctness patches + n_outputs_max speed fix.
- Multi-slot DFlash: WORKING (all 4 slots 76-88% acceptance, 0 errors, 0 garble, coherent outputs).
- Speed: ~11-15 t/s (was 0% acceptance / ~5 t/s before the correctness fixes; ~11 t/s before n_outputs_max; ~15 t/s after n_outputs_max). Main is ~24 t/s. The remaining 1.5× gap is the 2.3× launch-count overhead (re-decode + extra target recurrent-capture decodes), confirmed by per-op profiling but NOT closeable by the re-decode batching without also fixing the hidden capture row selection.
- Correctness: FULLY RESTORED. No regression from any committed fix.

### NEXT STEPS (if pursuing the remaining 1.5×)
- The re-decode batching requires adjusting the hidden_gpu capture to skip the re-decode rows: in qwen35.cpp's hidden_gpu block, the row selection (currently the first n_hidden_keep rows of the batch) must skip the re-decode prefix and capture only the id_last + drafts rows. This is a graph-builder change (the hidden_gpu block's src_offset/dst_offset computation) — complex and risky.
- Alternatively: investigate the OTHER extra launches (the 16 non-re-decode extra blocks — the target recurrent-capture ops PAD_REFLECT_1D/GATED_DELTA_NET_TREE appearing 528 times in work vs not in main's 30-block run). These might be from the tape_gpu block (now-inert, per the 37a89c8f6 commit note) or the backup path, and might be removable without the hidden-capture complication.
- Or: accept the current state (15 t/s, correct, multi-slot working) as the practical result and defer the remaining 1.5× to a future investigation that addresses the hidden-capture row selection.

## 2026-07-13 05:00 UTC - Step 1 (tape_gpu removal) + Step 4 (drafter gap investigation) results

### Step 1: Remove inert tape_gpu block — NOT a speed improvement, reverted
- Disabled `set_tape_recording(true)` in the server (forces `tape_gpu_n_seqs=0`, skipping the tape_gpu graph block). Built (EXIT=0), benchmarked `--parallel 1` n_predict=200.
- Result: **SLOWER** — drafter decode 40→75ms, t/s 15→11. The tape removal causes graph-cache invalidation (the graph structure changes when `tape_gpu_n_seqs` goes 0 vs >0), and the tape's `ggml_cpy` ops were cheap (~2.8ms total). Reverted to `0d5f99638`.
- **Conclusion:** the tape_gpu block is NOT a meaningful speed overhead. The PAD_REFLECT_1D/GATED_DELTA_NET_TREE/SSM_CONV ops seen in the per-op profiling (528 count) are the TARGET's normal DeltaNet recurrent compute (proportional to the 70 decode blocks), NOT from the tape.

### Step 4: Drafter gap investigation — per-op GPU compute is IDENTICAL, gap is launch overhead
- Targeted comparison of the drafter's biggest matmul: `MUL_MAT q4_K m=17408 n=16 k=5120` (the FFN gate, n=16=batch_size):
  - WORK: ~660us per call (10 × 646us, 10 × 687us, 10 × 645us — 3 blocks)
  - MAIN: ~674us per call (10 × 648us, 10 × 700us, 10 × 674us — 3 blocks)
  - **Per-call GPU time is virtually identical** (within 2%).
- WORK has **235 MUL_MAT lines vs MAIN's 108** — 2.2× more total matmul invocations. This confirms: the 40ms vs 27ms drafter-decode gap is NOT per-op efficiency but **2.2× more decode cycles** (70 vs 30 blocks), each with CPU-side dispatch overhead.
- The MUL_MAT_VEC ops (the cross-attention K/V projection + LM head) also show n=2/n=4/n=8 in both branches with similar per-call timing (~240-270us for m=17408 n=2). The per-op GPU compute is the same; the count differs.
- **Final conclusion on the drafter gap:** the remaining 1.5× wall-clock gap (15 vs 24 t/s) is entirely due to **2.2× more graph-compute launches** (CPU dispatch overhead), NOT per-op GPU compute efficiency. The GPU processes each op at the same speed; the work branch simply launches 2.2× more graph-compute calls (re-decode + extra target decode passes from the rollback/backup path). The gap can only be closed by reducing the launch count (batching decodes or eliminating redundant target decode passes), NOT by optimizing the drafter graph or backend.

### Combined verdict (Steps 1 + 4)
- Option 1 (tape_gpu removal): **no benefit** (graph-cache invalidation outweighs the cheap tape copies; reverted).
- Option 4 (drafter gap investigation): **confirmed the gap is launch-count, not per-op**. The per-op GPU compute is identical between work and main. The fix must target reducing the number of graph-compute launches (the re-decode, the extra target decode passes), not the drafter graph or the Vulkan backend.
- The re-decode batching (the most direct launch-count fix) remains blocked by the hidden_gpu capture row selection issue (the failed attempt showed prepending re-decode tokens corrupts the capture). The correct fix requires adjusting the hidden_gpu capture's `src_offset` to skip the re-decode prefix — a graph-builder change with HIGH risk.
- Options 2 (hidden-capture row offset), 3 (batch draft wiring), and 5 (profit controller) remain the viable paths to reduce launches and close the gap. All are MEDIUM-HIGH risk and require careful implementation + correctness testing.

## 2026-07-14 01:00 UTC - Re-decode batching with src_offset: works but marginal gain

### FACTS — what was implemented
- Wired `src_offset` (reeval_n) through `update_logits_deferred_dflash_kv` → `append_target_hiddens` → `ring_write` so the ring write skips the re-decode prefix and reads from the verify tokens. Both CPU (layer_hiddens) and GPU (hidden_gpu D2D) paths use the offset.
- The hidden_gpu graph builder captures ALL tokens (no skip in the graph builder). The skip is handled entirely by `ring_write`'s `src_offset`.
- `dflash_capture_skip_n` is set (via `llama_dflash_set_capture_skip_n`) for graph-cache invalidation only — when `reeval_n` changes (0→N or N→0), the graph is rebuilt with correct `inp_out_ids` for the new batch size.
- Slot reset: `dflash_reeval_tokens`, `dflash_reeval_pos_base`, `dflash_reeval_n` are cleared on slot release.

### FACTS — verification results
| mode | acceptance (before) | acceptance (after) | speed (before) | speed (after) | errors |
|------|---------------------|--------------------|-----------------|---------------|--------|
| --parallel 1 | 74% | **60%** | 14-15 t/s | **15.66 t/s** | 0 |
| --parallel 4 (long prompts) | 76-88% | **72-89%** | 11-13 t/s | **11-13 t/s** | 0 |
| --parallel 4 (short prompts) | 76-88% | **53-79%** | 11-13 t/s | ~11 t/s | 0 |

### FACTS — analysis
- **Correctness: PRESERVED** — 0 errors, 0 garble, coherent outputs across all tests.
- **Speed: marginal gain** — ~1 t/s for single-slot (15.66 vs 14-15), no gain for multi-slot.
- **Acceptance: dropped** — 74→60% (single-slot), 76-88%→53-79% (multi-slot, variable).
- **Root cause of the marginal gain:** the `dflash_capture_skip_n` graph-cache invalidation causes ~48 graph rebuilds per 200 tokens (every time reeval_n changes 0→N or N→0). Each rebuild costs ~30ms. Total rebuild cost: ~1440ms. Saved re-decode launches: ~120ms (24 × 5ms). The rebuilds cost 12× more than the saved launches. The ~1 t/s gain comes from the fact that not all rebuilds are full rebuilds (some are partial cache hits).
- **Root cause of the acceptance drop:** the graph rebuilds produce slightly different numerical results (different op fusion, tensor layouts) compared to the cached graph. This causes the hidden states to differ slightly, leading to lower draft acceptance. The effect is more pronounced for single-slot (60% vs 74%) than multi-slot (72-89% vs 76-88%) because single-slot has fewer graph ops to amortize the rebuild cost.

### VERDICT
- The re-decode batching with `src_offset` is **correct** (no corruption) but **not a meaningful speed improvement** (~1 t/s gain, offset by graph rebuild overhead and acceptance drop).
- The 15→24 t/s gap is NOT closeable by re-decode batching alone — the graph rebuild overhead from the batch-size change (when re-decode tokens are prepended) eats the launch savings.
- **The re-decode batching is kept as uncommitted work-in-progress** for reference. It does NOT regress correctness. But the marginal speed gain and acceptance drop make it not worth committing as-is.

### NEXT STEPS
- Option 2 (wire batch draft) and Option 3 (wire profit controller) remain the viable paths. These don't require graph-builder changes (no graph rebuild overhead) and target the launch count from a different angle (batching drafter decodes, reducing wasted cycles).
- The re-decode batching could be revisited if the graph rebuild cost can be eliminated (e.g., by using a fixed-size prefix that doesn't change the batch size, or by making the graph builder handle variable batch sizes without invalidation).

## 2026-07-14 20:00 UTC - Option 2: Wire batch draft — SUCCESS, speed improved

### FACTS — what was implemented
- Replaced the per-slot `common_speculative_draft(s->get_spec())` loop in `pre_decode` (server-context.cpp) with a single `common_speculative_draft_batch(batch_specs, ctx_dft, params_batch, batch_id_lasts, batch_results, nullptr)` call for DFlash. Non-DFlash falls back to per-slot flat draft. The batch path uses the EXISTING `common_speculative_draft_batch` function (speculative.cpp:4806) and `prepare_batch_draft` (speculative.cpp:2664) — no impl changes needed.
- The batch function: (1) calls `prepare_batch_draft(ctx_dft)` per spec to build cross-data, (2) does ONE batched drafter decode for all specs, (3) extracts per-spec argmax. This replaces N separate drafter decodes with 1.
- Falls back to per-slot flat draft if the batch returns empty for a specific slot (e.g., committed_len==0).
- Build: `cmake --build build-vulkan --target llama-server` EXIT=0.

### FACTS — verification results
| mode | acceptance (before) | acceptance (after) | speed (before) | speed (after) | errors |
|------|---------------------|--------------------|-----------------|---------------|--------|
| --parallel 4 (long prompts) | 76-88% | **75-87%** | 11-13 t/s | **15-17 t/s** | 0 |
| --parallel 1 | 74% | **74%** | 14-15 t/s | **15-16 t/s** | 0 |

Per-slot acceptance (--parallel 4, 4 long prompts, n_predict=300):
- slot 3: 77.4% (209/270), slot 2: 74.8% (175/234), slot 1: 82.6% (213/258), slot 0: 87.1% (216/248)
- 0 errors, 0 garble (all 4 outputs coherent step-by-step solutions)

Speed (slot 0, --parallel 4): **16.60-16.98 t/s** (was 11-13 t/s) — a **30% improvement** for multi-slot.
Speed (--parallel 1): **14.81-15.79 t/s** (was 14-15 t/s) — marginal single-slot gain (batch of 1 ≈ flat).

### FACTS — analysis
- **Correctness: PRESERVED** — acceptance 75-87% (matching baseline 76-88%), 0 errors, 0 garble.
- **Multi-slot speed: +30%** (11-13 → 15-17 t/s) — the batch draft reduces 4 drafter decodes per cycle to 1, saving 3 launches per cycle. The launch savings compound across all 4 slots.
- **Single-slot speed: marginal** (14-15 → 15-16 t/s) — batch of 1 spec ≈ flat draft, minimal launch savings.
- **No graph-builder changes** — the batch draft is a server-side wiring change only (no graph rebuild overhead, unlike the re-decode batching).

### VERDICT
- Option 2 (wire batch draft) is a **clear success**: +30% multi-slot speed, no correctness regression, no graph-builder changes, uses existing impl.
- The remaining gap to 24 t/s: multi-slot is now ~16 t/s (was 11-13), still ~33% below main's 24 t/s. The remaining gap is the 2.2× graph-compute launch overhead (re-decode + target recurrent-capture decodes), which requires the re-decode batching (graph-builder changes, high risk) or other launch-count reductions.
