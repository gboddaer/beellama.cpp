# Handoff: Vulkan DFlash — Tasks 0-5 Complete, Performance Recovery Next

## Start here

Read the plan:
`docs/superpowers/plans/2026-07-30-vulkan-speculative-correctness-performance-recovery.md`

Continue from **Task 6** (Build clean performance references). Tasks 0-5 are done and committed.

## What was accomplished

| Task | Status | Commit |
|------|--------|--------|
| 0: Freeze audit baseline, correct docs | ✅ | `d3f388727` |
| 1: Benchmark harness detects failures | ✅ | `e6211d0ec` |
| 2: Restore CLI, fix null safety, MTP layer boundary | ✅ | `0c1415750` |
| 3: Fix prompt echo (coordinated cache reset) | ✅ | `fe92d8a7f` + `2b85a8780` |
| 4: Fix DFlash -np 2 crash | ✅ | `294a9ff7b` |
| 5: EOS behavior verified | ✅ | (no code change needed) |

## Task 4 fix detail (the last correctness fix)

**Commit:** `294a9ff7b` — `llama: preserve output state during DFlash rollback re-eval`

**File:** `src/llama-context.cpp` (27 lines, 2 changes)

**Root cause:** DFlash rollback re-eval calls `llama_decode()` with `logits=false`. This clobbered `output_ids[]` (filled with -1 by `output_reserve()`) and `n_outputs` (set to 0). Under `-np 2`, slot 0's rollback re-eval corrupted this state before slot 1's `post_decode` ran, causing `llama_get_logits_ith(ctx_tgt, 9)` to fail.

**Fix:**
1. `decode()` saves `n_outputs` before decode and restores it after, for batches with no logits requests
2. `output_reserve()` skips `std::fill(output_ids, -1)` when `n_outputs == 0`

Both needed: fix 1 alone leaves `output_ids` clobbered; fix 2 alone leaves `n_outputs=0` causing "corrupt output buffer".

**Verification:**
- `-np 2` concurrent DFlash smoke: ✅ both requests OK, nonzero draft_n
- `-np 1` repeated requests: ✅ 3/3, draft_n=105, identical output
- Harness strict tests: ✅ 10/10
- CTest: ✅ 6/6

## Git state

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- HEAD: `294a9ff7b` (`llama: preserve output state during DFlash rollback re-eval`)
- Working tree: clean (no uncommitted changes)
- Previous pushed tip: `1ce1505d3` (not yet pushed: `294a9ff7b`)
- Push targets: `gboddaer` and `boditec` (both need `294a9ff7b` pushed)

## What's next: Task 6 (Performance References)

The plan's Tasks 6-8 are performance work:

1. **Task 6:** Build clean reference worktree at `adb92b36af`, configure identical builds, run BASE/DFlash benchmarks on both reference and HEAD, verify provenance
2. **Task 7:** Activate compact reduced verification safely (opt-in, gated by `GGML_DFLASH_REDUCED_VERIFY=1`)
3. **Task 8:** Profile and fix remaining DFlash performance gap (target: DFlash ≥ 90% of reference, ≥ 1.5x HEAD BASE)

**Key reference numbers from the old handoff:**
- Old-fork DFlash: ~31.39 t/s, 84.6% acceptance
- HEAD DFlash: ~6.46 t/s, 54.7% acceptance
- HEAD BASE: ~12.5 t/s
- Old-fork reference commit: `adb92b36af0353870a1ab53515ec0b19d5d3618e`

**Performance gates:**
- HEAD BASE within 5% of reference BASE
- DFlash ≥ 90% of reference DFlash median
- DFlash ≥ 1.5x HEAD BASE
- If reference DFlash ≥ 26.7 t/s, HEAD must be ≥ 24.0 t/s

## Important notes

- Do not push without explicit human approval
- When approved, push to BOTH: `git push gboddaer merge_llama_into_beellama_2` and `git push boditec merge_llama_into_beellama_2`
- All commits must include `Assisted-by: <model>` trailer
- Do not optimize before Tasks 6-8 profiling is complete
- Do not stack fixes; one hypothesis, one A/B at a time
- The GLM-5.2 review's recommendation to add a server-side guard rejecting `--spec-type dflash` with `-np > 1` is **moot** — Task 4 fixed the crash. Multi-slot DFlash now works.

## Verification commands

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2

# Verify clean state
git status --short
git log --oneline -3

# Build
cmake --build build-vulkan --target llama-server -j"$(nproc)"

# Run correctness smokes
bash /tmp/task4-np2-smoke.sh
bash /tmp/task4-np1-smoke.sh

# Harness tests
python3 -m unittest bench.vulkan-gap.test_harness_strict -v

# CTest
ctest --test-dir build-vulkan -R 'test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode'
```

## Related documents

- Plan: `docs/superpowers/plans/2026-07-30-vulkan-speculative-correctness-performance-recovery.md`
- Original handoff: `docs/superpowers/HANDOFF-vulkan-speculative-correctness-performance-recovery.md`
- Task 4 root cause (detailed): `docs/superpowers/HANDOFF-vulkan-speculative-task4-root-cause.md`
- Task 4 checkpoint (pre-fix): `docs/superpowers/HANDOFF-vulkan-speculative-task4-checkpoint.md`
- Review: `docs/superpowers/reviews/2026-07-30-vulkan-speculative-execution-review.md`