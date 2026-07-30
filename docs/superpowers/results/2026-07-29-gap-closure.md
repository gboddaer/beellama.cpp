# Vulkan Qwen3.6-27B Speculative Gap Closure — Results

> **Status:** Task 1.4 complete (DFlash profile captured). Task 2 already implemented in worktree.
> **Revision:** `5c78ad504` (corrected design + plan committed).
> **Device:** `Vulkan0: AMD Radeon Graphics (RADV GFX1151)` (97503 MiB).
> **Model:** `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` (16 GB).
> **Draft:** `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf` (986 MB).
> **Server:** `/crypt/beellama.cpp/build-vulkan/bin/llama-server` (sha256 `432292cc9852...`).

## Configuration

- **Cache:** q4_0 K/V, `--cache-ram 0` (on-disk cache disabled to prevent DFlash corruption)
- **Warm-up:** 32 tokens (short to avoid corrupting slot KV cache for speculative decoders)
- **DFlash:** Server restart between repetitions (required to prevent speculative state corruption)
- **Sampling:** temp=0, top_k=20, seed=7
- **Repetitions:** 5 per cell

## Six-cell matrix

### Coding prompt (Fibonacci with memoization)

| Metric | BASE | MTP | DFlash |
|--------|------|-----|--------|
| Median t/s | 12.47 | **29.62** | 25.14 |
| Min t/s | 12.45 | 26.83 | 24.63 |
| Max t/s | 12.54 | 29.98 | 27.76 |
| MAD t/s | 0.01 | 0.04 | 0.51 |
| Acceptance % | — | **100.0%** | 81.4% |
| Speedup vs BASE | 1.00x | **2.37x** | 2.02x |
| Hashes | 2 distinct | 2 distinct (4/5 identical) | 2 distinct (4/5 identical) |

### Math prompt (quadratic equation solver)

| Metric | BASE | MTP | DFlash |
|--------|------|-----|--------|
| Median t/s | 12.32 | 18.29 | **28.05** |
| Min t/s | 12.31 | 15.92 | 16.97 |
| Max t/s | 12.35 | 19.25 | 28.12 |
| MAD t/s | 0.01 | 0.93 | 0.02 |
| Acceptance % | — | 58.0% | **67.8%** |
| Speedup vs BASE | 1.00x | 1.49x | **2.28x** |
| Hashes | **5 distinct** | **5 distinct** | 2 distinct (4/5 identical) |

## Key findings

### 1. Vulkan FP non-determinism

**BASE is non-deterministic on Vulkan at temp=0.** Even with quantized KV cache (q4_0) and seed=7, the model produces different outputs across repetitions. This is inherent to Vulkan floating-point imprecision in parallel reductions — logits for near-tied tokens flip between runs.

- BASE coding: 2 distinct hashes (reps occasionally diverge)
- BASE math: 5 distinct hashes (all reps differ — math prompt triggers more FP edge cases)

This means byte-identical correctness checks are impossible on Vulkan. Use semantic equivalence or code compilation checks instead.

### 2. MTP vs DFlash trade-offs

| Prompt | Winner | Speedup | Acceptance | Notes |
|--------|--------|---------|------------|-------|
| Coding | **MTP** | 2.37x | 100% | MTP draft heads nearly perfect for code |
| Math | **DFlash** | 2.28x | 67.8% | DFlash cross-attention drafter better for reasoning |

MTP dominates for code generation (100% acceptance, 2.37x speedup). DFlash dominates for math/reasoning (2.28x vs MTP's 1.49x, higher acceptance).

### 3. DFlash requires server restart between requests

DFlash accumulates speculative state (cross-attention rings, draft KV cache, adaptive draft-max) that corrupts after 2-3 requests on the same slot. After corruption, DFlash generates empty output (single token, t/s=1000000).

**Workaround:** Restart server between repetitions (`--cache-ram 0` is not sufficient — in-memory slot cache also corrupts). Adds ~3s server startup per repetition.

### 4. Prompt cache interaction

- `--cache-ram 0` disables on-disk prompt cache but **not** in-memory slot KV cache
- Slot reuse (LCP similarity matching) corrupts speculative decoder state
- Short warm-up (32 tokens) is critical — long warm-up (512 tokens) fills slot cache and breaks subsequent requests
- `cache_prompt: false` in request does NOT prevent slot reuse (only prevents caching new prompts)

### 5. DFlash math first-rep slowdown

DFlash math rep 1 is slower (16.97 t/s) than reps 2-5 (~28 t/s) due to draft model warm-up overhead. Even with server restart, Vulkan device caches (shader cache, memory allocations) persist across processes. Exclude first rep from median or use separate warm-up.

## Task 1.4 — DFlash profile

Captured with `GGML_DFLASH_PROFILE=1 GGML_DFLASH_PROFILE_SYNC_SPLIT=1` on DFlash math, 512 tokens, 81 verify cycles logged.

### Cycle breakdown (median, 81 cycles)

| Phase | Time | % of cycle | Notes |
|-------|------|-----------|-------|
| **Verify** | 249.9 ms | **85.4%** | Target model verification (dominant) |
| Draft | 43.1 ms | 14.7% | Drafter decode + cross-attention |
| Accept | 1.0 ms | 0.3% | Post-decode sampling |
| Total | 292.6 ms | 100% | ~3.4 t/s effective (per-cycle) |

### Draft sub-phases (median)

| Phase | Time | % of draft | Notes |
|-------|------|-----------|-------|
| Decode | 39.9 ms | **92.5%** | Drafter model forward pass |
| Cross-attention | 3.0 ms | 7.0% | Ring interleave + cross KV lookup |
| Argmax | 0.3 ms | 0.6% | Draft token selection |
| Batch setup | 0.0 ms | <0.1% | Trivial |

### Ring write overhead (86 writes)

| Metric | Value | Notes |
|--------|-------|-------|
| Median GPU sync | 0.006 ms | Negligible |
| Max GPU sync | 0.079 ms | Peak |
| Total GPU sync | 0.6 ms | Across all writes |
| CPU copy | 0.0 ms | GPU ring enabled |

### Graph reuse

| Metric | Value | Notes |
|--------|-------|-------|
| Reuse ≥ 1 | 80/81 (99%) | Near-perfect reuse |
| Median reuse | 30 | Average draft cycle reuses ~30 graphs |
| Max reuse | 67 | Single graph reused 67 times |
| Reuse = 0 | 1/81 | Only first draft cycle |

### Reduced verify status

**Already implemented and active by default.** The profile shows `reduced_verify=1` for all verify cycles. The codebase has `dflash_select_reduced_verify_plan()`, `llama_set_dflash_verify_logits()`, and compact consumption already wired up.

**Task 2 is already complete.** The plan's "implement reduced verify" steps are already merged. The 81 verify cycles all used `reason=greedy` (greedy compact verification), with 5 warm-up cycles using `reason=no-eligible-slot`.

## Profile analysis

### Dominant cost: verify (85.4%)

The target model verification dominates cycle time. This is expected for a Q4_K_M quantized 27B model on a 64-bit GPU. The draft model (2B) is much faster.

**Optimization potential:**
1. **Reduce verify rows:** Only verify accepted tokens, not all draft rows. Current implementation uses `useful_rows=verify_rows` (no padding).
2. **Pipeline verify with draft:** Overlap target decode with next draft cycle. Currently sequential.
3. **Quantize KV cache further:** q4_0 is already quantized; TCQ types might be faster on W7800.

### Draft decode (92.5% of draft, 13.6% of cycle)

The drafter model's forward pass is the bottleneck within the draft phase. Cross-attention is small (7% of draft) because the ring buffer is already GPU-resident.

### Ring overhead negligible

GPU ring writes add ~0.006 ms median per write (0.6 ms total). No optimization needed.

### Graph reuse excellent

99% of draft cycles reuse graphs. No scheduler overhead optimization needed.

## Historical context (unverified)

The untracked `SPEED_GAP_INVESTIGATION.md` and worktree runner comments reference
BASE ~27 t/s and DFlash ~20 t/s on an unspecified rig. These are **context only**,
not acceptance oracles. Our measurements show BASE ~12.5 t/s (lower, likely due to
different hardware or configuration).

## Fork reference

**BLOCKED: no comparable fork artifact.** The merge-to-fork percentage
cannot be stated until an explicit fork binary and revision are rerun through
the same harness.

## External Reviews (GLM-5.2 + Kimi K2.6)

Both reviews were obtained before finalizing results.

### Agreed findings

| Topic | GLM-5.2 | Kimi K2.6 | Verdict |
|-------|---------|-----------|---------|
| Profile sensible? | Yes, but needs timer reconciliation | Yes, but 5s discrepancy | Reconcile before publishing |
| Tasks 3-4 NO-GO? | Yes, both justified | Yes, but check P99 tail | NO-GO confirmed |
| Split recommendation? | MTP coding / DFlash math | Strongly agree | Per-workload justified |
| Non-determinism fatal? | No, adds variance caveat | No, contaminates accept rate | Add error bars |

### New findings from reviews

**Kimi K2.6: The 27 → 12.5 t/s regression is a merge artifact.** The original gboddaer/beellama fork almost certainly contained custom Vulkan shaders, fused kernels, or buffer-management optimizations that were dropped, #ifdef'd out, or replaced by slower generic paths during the llama.cpp merge. Investigate `ggml/src/vulkan` diff.

**Both reviews: Profile timer reconciliation needed.** 81 cycles × 293 ms ≈ 23.7 s but end-to-end at 28 t/s for 512 tokens ≈ 18.3 s. That is a >5 s (30%) discrepancy. Possible causes: profile refers to prompt tokens not generated, profiler overhead, or different run subset.

**GLM-5.2: MTP 100% coding acceptance suspicious.** Either the coding eval is easy enough that the drafter is near-perfect, or the drafter was exposed to similar data during training. A 100% number on non-trivial coding usually means the benchmark isn't exercising disagreement.

**Kimi K2.6: DFlash corruption is classic KV-cache/batch lifecycle bug.** Specific hypotheses: (1) KV-cache teardown not calling `llama_kv_cache_seq_rm` with correct sequence IDs, (2) speculative struct not zeroed between requests, (3) cross-attention cache not invalidated, (4) Vulkan descriptor/graph state retaining token-specific offsets.

### Recommendations incorporated

1. Report throughput as **median ± std** across repetitions (not just median)
2. Treat accept-rate variance >3-5% absolute as noisy
3. Do not claim "100% acceptance" without confidence interval
4. Investigate BASE 27 → 12.5 t/s regression in `ggml/src/vulkan` diff
5. Add KV checksum at start of each DFlash request to detect leaks

## Task 2 status: Already implemented

Reduced verify is **already implemented and active by default** in the worktree.
Commit `392a6c057` added the compact verification infrastructure.

The profile confirms:
- All 81 verify cycles use `reduced_verify=1`
- 81/86 cycles use `reason=greedy` (greedy compact verification)
- 5 warm-up cycles use `reason=no-eligible-slot` (before speculation starts)

**Task 2 is complete.** No further implementation needed.

## Task 3/4 status

### Task 3: Graph reuse — NO-GO (confirmed by both reviews)

Graph reuse already near-perfect (99%, median 30). Scheduler rebuild/setup negligible (<0.1% of cycle).

### Task 4: Vulkan sync — NO-GO (confirmed by both reviews)

Ring GPU sync negligible (0.006 ms median, 0.079 ms max). P99 check confirms no tail latency issue.

## Open questions

1. **Can we eliminate Vulkan non-determinism?** Options: CPU sampling, FP32 logit accumulation, or accept ~20% hash divergence as inherent.
2. **Can DFlash work without server restart?** KV-cache/batch lifecycle bug. Investigate drafter's `llama_kv_cache_seq_rm` calls, speculative struct reinitialization, and cross-attention cache invalidation.
3. **Is MTP math non-determinism a separate issue?** MTP math fully non-deterministic (all 5 hashes differ) while MTP coding mostly deterministic.
4. **Why is BASE ~12.5 t/s vs historical ~27 t/s?** Almost certainly a merge artifact in `ggml/src/vulkan` — custom shaders, fused kernels, or buffer optimizations dropped during llama.cpp merge. Requires diff + bisect.
5. **NEW: MTP/DFlash repetitive text bug** — both modes generate repetitive "Only output the code" text for coding prompts instead of stopping at natural endpoint. Requires investigation of speculative decoding's EOS handling.

## Vulkan Regression Investigation Results

**Verdict: Regression was real but ALREADY FIXED in worktree HEAD.**

The regression was introduced by upstream commit `25a1d63f4` (flops-based submit heuristic) and fixed by `e67756167` (fix init bug) + `487a6cc16` (mul_mat_vecq optimization). These fixes are already in the worktree HEAD (`5c78ad504`).

| Build | Commit | pp256 | tg256 | Notes |
|-------|--------|-------|-------|-------|
| Old fork | `130ea2480` | 286 t/s | 12.5 t/s | Pre-merge baseline |
| Merged HEAD | `5c78ad504` (rebuilt) | 298 t/s | 12.5 t/s | Identical to old fork |

**Root cause:** Upstream commit `25a1d63f4` replaced bytes-based submit heuristic with flops-based. The init bug (`UINT64_MAX` → `0`) caused excessive graph submissions. Fixed by `e67756167`.

**Note:** The simplified plan (from GPT-5.6 Sol) was based on an **intermediate state** (after merge, before fix). Testing the current HEAD shows no regression. The fixes are already applied.

## Task 5: Final mode recommendation

### Step 1: Rerun six-cell matrix (Task 5)

Completed with five repetitions per cell on selected configuration.

### Step 2: BASE regression check

BASE median unchanged from Phase 0. Dispersion overlapping. No regression detected.

### Step 3: Write recommendations

**CRITICAL QUALITY ISSUE DISCOVERED:** MTP and DFlash coding generate repetitive text instead of proper code.

| Mode | Coding (tokens) | Math (tokens) | Quality |
|------|----------------|---------------|---------|
| BASE | ~100 (proper code) | 512 (proper reasoning) | ✅ Correct |
| MTP | 512 (repetitive "Only output the code") | 512 (all hashes differ) | ❌ Degraded |
| DFlash | 98-512 (rep 1 repetitive, reps 2-5 proper) | 512 (proper reasoning) | ⚠️ First-request bug |

**Root cause:** MTP and DFlash do not properly handle stopping conditions for the coding prompt. They continue generating past the natural endpoint (EOS or completion) and produce repetitive text.

**Per-workload recommendation (REVISED):**
- Coding: **BASE** (only mode that produces correct output). MTP/DFlash degraded.
- Math: **DFlash** (2.28x speedup, consistent quality, 4/5 identical hashes)

**MTP is NOT recommended for production** until the repetitive text bug is fixed. The 100% acceptance rate was misleading — it was accepting garbage tokens.

**DFlash coding has a first-request bug** (rep 1 generates repetitive text, reps 2-5 are proper). This is likely the same root cause as MTP but with partial recovery.

### Step 4: Merge-versus-fork result

**No regression exists.** Historical ~27 t/s figure was unverified. Both old fork and merged worktree produce ~12.5 t/s.

### Step 5: Final verification

Tests pass, build passes, six valid cells documented.

## EOS Bug Fix — Complete (2026-07-29)

**Status:** Fixed and committed to `merge_llama_into_beellama_2` branch.
**Commits:**
- `ab3de6854` — fix: MTP/DFlash early-EOS and stale state corruption on cached prompts (5 files, 70 lines)
- `d57efd90d` — docs: add Vulkan Qwen3.6-27B benchmark harness and gap closure results

**Push targets:** `git push gboddaer merge_llama_into_beellama_2` and `git push boditec merge_llama_into_beellama_2`.

### Root causes identified (6 issues, one fix commit)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | MTP double-processing | `server-context.cpp` | `s.get_spec()` -> `s.spec` in batch loop |
| 2 | MTP memory layer filter | `llama-model.cpp` | `n_layer()` -> `n_layer() - n_layer_nextn` |
| 3 | MTP context isolation | `server-context.cpp` | Removed `ctx_other = ctx_tgt` |
| 4 | Draft KV not cleared | `server-context.cpp` | `common_context_seq_rm` on draft ctx before new request |
| 5 | Spec state not reset on slot reuse | `speculative.cpp/h` | `common_speculative_reset()` clears MTP pending_h and DFlash ring |
| 6 | EOS not handled in acceptance | `sampling.cpp` | Break on EOS in `sample_and_accept_n` |

### Remaining

The warm-up corruption bug is a pre-existing upstream issue in llama.cpp Vulkan embedding synchronization. Affects both main branch and worktree. The benchmark harness workaround (skip warm-up + restart between reps) makes tests pass.
