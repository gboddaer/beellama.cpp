# Handoff: Vulkan Qwen3.6-27B Speculative Gap Closure — Corrected

## Repository / state

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- Reviewed HEAD: `392a6c057`
- Production code changed by this review: **none**
- Design and plan are untracked and corrected in place; do not commit unless the
  user explicitly authorizes it.

## Purpose

Use reproducible W7800 Vulkan measurements to choose BASE, MTP, or DFlash for
Qwen3.6-27B, then close only demonstrated speculative-decoding gaps with fixes
that preserve correctness.

The work is not a mandate to make DFlash beat BASE on W7800, and historical
runner comments are not acceptance baselines.

## Primary artifacts

- Corrected design:
  `docs/superpowers/specs/2026-07-29-vulkan-qwen36-spec-gap-closure-design.md`
- Corrected implementation plan:
  `docs/superpowers/plans/2026-07-29-vulkan-qwen36-spec-gap-closure.md`
- Historical investigation (partly stale): `SPEED_GAP_INVESTIGATION.md`

Read the corrected design before executing the corrected plan.

## Review result

The original direction—measure BASE/MTP/DFlash first, isolate changes, and
recommend per workload—was sound. The original Phase 1 implementation track was
not safe or sufficiently evidenced.

### Load-bearing corrections

1. **Reduced verify is an output-transfer/sampling optimization, not removal of
   the vocabulary projection.** GPU top-k still consumes `t_logits`. The prior
   design overstated both mechanism and expected speedup.
2. **The original reduced-verify server pseudocode was unsafe.** It used static
   context state, ignored per-request/multi-slot row offsets, allowed selector
   top-k values above the context's 64-value clamp, and attempted an impossible
   full-logits fallback after full logits had been omitted.
3. **Fixed verify padding is not “without padding waste.”** Dummy tokens still
   run target-model compute. Cleanup must also restore Qwen3.6 recurrent state,
   not merely delete KV positions. The old single-slot cleanup with `break` is
   not production-safe. Padding is removed from the executable plan pending a
   separate evidence-backed design.
4. **The proposed Vulkan timeline wait was a false optimization.** Signaling an
   empty submit after prior queue work, waiting in another empty submit, then
   CPU-waiting on a fence drains prior work. A valid design must signal the
   actual producer submission and wait in the actual consumer submission, with
   no CPU fence on the fast path. Vulkan changes are now an investigation gate,
   not ready code.
5. **Historical 19.75/27.23 t/s values are unverified comments.** A fork ratio is
   reported only when an explicit fork binary/revision is rerun through the same
   harness.
6. **The old benchmark was too weak.** It lacked robust startup/request failure
   handling, bounded cleanup, full provenance, repeated statistics, and
   multi-request/multi-slot correctness checks.
7. **Kill-switches must start old-path/default-off.** A default flip is a
   separate review after correctness and performance gates pass.

## Correct execution track

1. **Task 0:** build the provenance-recording harness and its unit tests.
2. **Task 1:** measure current BASE/MTP/DFlash, capture existing profile counters,
   and rerun a fork only if an explicit artifact is supplied.
3. **Task 2:** implement only opt-in reduced verify with per-view activation,
   consume-state cleanup, top-k 1..64 validation, and per-slot compact-row
   gathering. Test sequential requests and `-np 2`.
4. **Task 3:** consider graph-shape work only if graph setup exceeds the plan's
   measured gate; write a separate recurrent-state-safe design first.
5. **Task 4:** consider Vulkan synchronization only if the wait remains material;
   trace producer/consumer queues and write a separate design first.
6. **Task 5:** rerun the six-cell table and recommend the fastest correct mode.

Do **not** execute earlier versions of Tasks 1.2 (padding) or 1.3 (empty-submit
timeline semaphore).

## Exact current-source anchors

- Verify compact path disabled:
  `tools/server/server-context.cpp:4423-4433`
- Reduced-plan selector and batch validator:
  `tools/server/server-context.cpp:1012-1194`
- Full-logits speculative sampling:
  `tools/server/server-context.cpp:4696-4728`
- Current exact-size verify batch:
  `tools/server/server-context.cpp:514-548`
- Reduced-consumer setter and top-k clamp:
  `src/llama-context.cpp:1793-1808`
- Compact output readback versus raw-logits transfer:
  `src/llama-context.cpp:7260-7331`
- Output-buffer suppression:
  `src/llama-context.cpp:7496-7526`
- Unconditional DFlash scheduler sync:
  `src/llama-context.cpp:7141-7173`
- Reduced sampler rejects unsupported sampler state with an empty result:
  `common/sampling.cpp:747-833`

Line numbers are anchors for reviewed HEAD and must be reconfirmed after edits.

## Independent review

A local GLM second opinion agreed on the unsafe reduced-verify lifecycle,
impossible empty-result fallback, invalid timeline-semaphore optimization, and
weak benchmark statistics. It also warned about padding cleanup. One GLM detail
was rejected after source verification: current `pos0` is incremented past the
real verify tokens before old padding used `pos0 + i`; it was not the root-token
position. The broader recurrent-state and cleanup objections remain valid.

## Open blockers / decisions

- A reproducible fork binary and revision are not currently identified. Without
  them, the merge-to-fork ratio remains blocked rather than inferred from
  comments.
- MTP may stall or be unsupported. Record that as a result with logs; do not let
  it silently disappear from the matrix.
- No default-on reduced verify until deterministic, sequential-request,
  multi-slot, supported-stochastic, and performance gates pass.
- No production padding or Vulkan synchronization change without a new approved
  design.

## Suggested skills for execution

1. `superpowers:executing-plans` or `superpowers:subagent-driven-development`
2. `superpowers:test-driven-development` for harness and reduced-verify work
3. `superpowers:systematic-debugging` on mismatches, stalls, or Vulkan errors
4. `superpowers:verification-before-completion` before any success claim
5. `context-mode` for build, benchmark, and profile logs

The worktree already exists; do not create another unless the user requests it.
