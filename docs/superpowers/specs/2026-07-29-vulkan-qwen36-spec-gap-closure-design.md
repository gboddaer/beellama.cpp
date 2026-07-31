# Vulkan Qwen3.6-27B Speculative Gap Closure — Corrected Design

> **Status:** Corrected after source-level review of
> `merge_llama_into_beellama_2` at `392a6c057`.
> **Plan:** `docs/superpowers/plans/2026-07-29-vulkan-qwen36-spec-gap-closure.md`.

## Purpose

Determine, with reproducible measurements, whether BASE, MTP, or DFlash is the
best mode for Qwen3.6-27B on the W7800 Vulkan rig, then close only performance
gaps whose causes are demonstrated and whose fixes preserve speculative-decoding
correctness.

The work must not assume that every historical merge-versus-fork difference is
still reproducible or that DFlash must beat BASE on this GPU.

## Evidence status

### Verified in the current source

1. `llama_set_dflash_consume_reduced()` exists, but the server does not activate
   it. The verify path sets `dflash_compact_verify_active = false` in
   `tools/server/server-context.cpp:4423-4433`.
2. Compact verify can avoid allocation and host transfer of the full logits
   buffer. It does **not** avoid the vocabulary projection itself: GPU top-k is
   derived from `t_logits`. Therefore the expected win is reduced output-buffer
   traffic and CPU sampling work, not elimination of the logits matmul.
3. The current verify batch has `1 + spec_draft.size()` tokens, so `n_tokens`
   and `n_outputs` vary. `llm_graph_result::can_reuse()` keys on those values.
   This makes graph reuse less likely, but the net cost is not yet measured.
4. `src/llama-context.cpp:7166-7173` performs an unconditional scheduler
   synchronization whenever DFlash capture state exists.
5. Existing DFlash profiling already reports drafter preparation/decode/argmax,
   ring-copy work, graph reuse, and the verify synchronization split. New
   instrumentation should be added only for a measured residual.

### Historical, not yet a valid acceptance baseline

The untracked main-tree runner records W7800 comments of BASE 27.23 t/s and
DFlash 19.75 t/s. `SPEED_GAP_INVESTIGATION.md` records different values from a
slower UMA rig. Neither record identifies a complete reproducible tuple of
commit, binary, prompt, request payload, sampling settings, warm-up policy, and
repetitions. These numbers are context, not a T1 oracle.

A merge-versus-fork target is valid only after both revisions are rerun through
the same harness on the same rig.

## Corrections to the previous design

### Reduced verify

Reduced verify is a credible first experiment, but the prior implementation
outline was unsafe:

- A function-local `static` initialization guard is not context-safe and cannot
  represent per-request sampling changes.
- The decision must use every active slot's request sampling parameters and a
  single top-k compatible with the whole decode view.
- `llama_context::set_dflash_verify_logits()` clamps top-k to 64, while the
  selector currently admits values through 256. Activation must reject values
  above 64 until those interfaces agree.
- Compact rows are batch-global. Multi-slot sampling must map each
  `slot.spec_i_batch` index into the current `batch_view`; every slot cannot read
  from row zero.
- An empty reduced-sampler result cannot fall back to full-logits sampling after
  full-logits allocation/readback was skipped. Unsupported views must be
  rejected before decode; an unexpected post-decode failure is an error or
  requires a fresh full-logits re-decode.
- The consume flag must be reset on success, retry, and exception paths.

Reduced verify therefore ships opt-in first and is enabled by default only in a
separate, reviewed change after correctness and performance gates pass.

### Verify padding and graph reuse

The previous design incorrectly called fixed-size padding "without padding
waste." Padding to `1 + n_draft_max` still runs target-model computation for the
dummy tokens. Reduced verify only avoids full-logits output transfer; it does
not make those target tokens free.

The old code's `pos0` was already advanced past the real verify tokens, so
`pos0 + i` was not inherently a duplicate-position bug. The real unresolved
issues are:

- target compute cost versus graph-build savings;
- exact per-slot dummy ranges;
- cleanup on all-accepted, rollback, retry, and error paths;
- recurrent-state effects in Qwen3.6's hybrid memory, which are not repaired by
  deleting KV positions alone; and
- multi-slot correctness.

Fixed-shape token padding is therefore removed from the committed Phase 1
implementation. It may be prototyped only after profiling proves graph rebuilds
are material and after a separate recurrent-state-safe design is approved.

### Vulkan synchronization

The previous timeline-semaphore pseudocode was not a valid optimization. It
submitted an empty signal after prior compute work, submitted an empty wait, and
then waited for a fence on the CPU. On one queue this completes after all prior
work and is effectively another queue drain. It neither identifies nor orders
the exact producer and consumer operations.

A valid asynchronous design must:

1. identify the queue/submission that produces captured hidden/tape data;
2. signal from that producer submission;
3. make the actual consumer queue submission wait on the signal; and
4. avoid a CPU fence wait on the fast path.

If producer and consumer use the same Vulkan queue, queue order may already be
the dependency and the scheduler sync may be removable for that case. If they
use different queues, a GPU-to-GPU semaphore dependency is required. This needs
a source trace and validation-layer proof before implementation. The fabricated
empty-submit timeline change is removed from the plan.

## Architecture

### Phase 0 — Reproducible evidence

Create one harness that records the exact revision, binary, model paths, device,
full server command, request payload, environment, response timings, output
hash, server log, and repeated measurements. Run BASE, MTP, and DFlash for two
fixed prompts. A fork comparison is reported only when an explicit fork binary
and revision are supplied.

**Gate:** at least five measured repetitions after warm-up; report median,
minimum, maximum, and median absolute deviation. No optimization starts until
BASE and current DFlash complete correctly.

### Phase 1 — Safe reduced-verify experiment

Activate compact consumption only for a fully validated decode view. Keep the
existing full-logits path authoritative and make compact consumption opt-in.
Test greedy and supported top-k sampling, sequential requests, and multi-slot
row mapping. Measure output-transfer bytes/time and end-to-end t/s.

**Gate:** no token divergence for deterministic cases, no crash or stale state
across repeated requests, and a statistically visible end-to-end improvement.
Otherwise leave it off and record the result.

### Phase 2 — Profile-driven follow-ups

Use existing profile counters to rank the remaining costs.

- Investigate graph shape only if graph construction/reuse is material.
- Investigate Vulkan synchronization only if the unconditional sync remains
  material after reduced verify.
- Add instrumentation only for time still unaccounted after existing counters.

Each unresolved subsystem gets a separate design before code changes. This
prevents an unsafe padding or semaphore experiment from being treated as a
ready implementation task.

### Phase 3 — Mode recommendation

Produce the BASE/MTP/DFlash × coding/math table from the same harness. Recommend
the fastest correct mode per workload on this exact rig. Do not extrapolate a
W7800 result to UMA or other GPUs without measurements.

## Acceptance criteria

- **A1 — Reproducibility:** every result identifies commit, binary hash, device,
  model hashes/paths, command, payload, environment, and repetition statistics.
- **A2 — Correctness:** deterministic BASE/MTP/DFlash outputs are token- or
  byte-identical for the fixed requests; reduced verify also passes sequential
  request and `-np 2` row-isolation tests.
- **A3 — Reduced verify:** compact consumption is enabled only on supported
  views, restores state on every exit path, and either improves median DFlash
  t/s beyond run noise or remains disabled.
- **A4 — Regression claim:** a merge-versus-fork percentage is stated only from
  same-harness measurements. Historical comments are labeled historical.
- **A5 — BASE safety:** BASE median t/s changes by no more than 2% with
  overlapping run dispersion; otherwise investigate before recommending the
  branch.
- **A6 — Recommendation:** all six mode/prompt cells are either measured or
  explicitly marked unsupported/stalled with logs.

## Safety and rollout

New optimization switches default to the old, known path during development.
For the reduced-verify experiment:

- `GGML_DFLASH_REDUCED_VERIFY=0` — full-logits path (default initially);
- `GGML_DFLASH_REDUCED_VERIFY=1` — validated compact-consumer experiment.

A default flip is a separate task after all gates pass. There is no
`GGML_DFLASH_VERIFY_PAD_FIXED` or `GGML_DFLASH_VK_TIMELINE_WAIT` implementation
in this design because neither proposed implementation is currently safe.

## Non-goals

- Guaranteeing DFlash beats BASE on W7800.
- Claiming results for UMA, CUDA, Metal, multi-GPU, or another model.
- Fixing the Qwen3-Coder-Next deadlock.
- Reworking Qwen3.6 recurrent memory to support dummy verify tokens.
- Rewriting Vulkan queue synchronization without a producer/consumer trace.
