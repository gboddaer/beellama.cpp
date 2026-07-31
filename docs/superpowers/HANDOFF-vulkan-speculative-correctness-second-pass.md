# Handoff: Vulkan Speculative Correctness Second Pass

## Mission

Execute:

```text
docs/superpowers/plans/2026-07-31-vulkan-speculative-correctness-second-pass.md
```

The goal is full Vulkan correctness for BASE, MTP, and DFlash across fresh, persistent, and concurrent requests. Do not begin performance work.

## Required Skills

Use these before acting:

```text
superpowers:executing-plans
superpowers:systematic-debugging
superpowers:test-driven-development before production changes
superpowers:verification-before-completion
graphify
context-mode
```

Do not use subagent-driven development. This plan is written for one simpler agent executing tasks sequentially.

## Repository

```text
Worktree: /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
Branch: merge_llama_into_beellama_2
Expected HEAD: d0e3ca514248
Target model: /crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DFlash draft: /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
```

At handoff creation, the branch had already been pushed to `boditec`. Do not commit or push again without explicit human authorization.

## Read First

Read completely, in this order:

```text
/crypt/beellama.cpp/AGENTS.md
docs/superpowers/specs/2026-07-31-vulkan-speculative-correctness-second-pass-design.md
docs/superpowers/plans/2026-07-31-vulkan-speculative-correctness-second-pass.md
docs/superpowers/results/2026-07-31-vulkan-speculative-working-tree-correctness-recovery.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/hypotheses/root-cause.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/hypotheses/classification.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/milestones/M4-agent.md
```

The old root-cause and M4 documents are historical evidence, not accepted conclusions.

## Correct Starting Assessment

- BASE passed the observed 512-token persistent coding and math runs.
- MTP and DFlash still fail long persistent correctness.
- Later bad requests often diverge at generated token index 0.
- Token 0 is sampled from target prompt-final logits before speculative drafting.
- MTP has no non-consecutive recurrent-position warnings.
- DFlash has warnings even in request 1.
- DFlash warning tuples are same-position re-evaluations, such as `20 after 20 with 1 new token`.
- Existing QA chronology places those warnings inside rollback re-evaluation.
- Full recurrent `seq_rm(seq_id,-1,-1)` resets sequence metadata. Do not claim otherwise without contrary evidence.
- The `prompt_clear(false)` hunk in `d0e3ca514` failed the required 512-token gates. Treat it as experimental.
- The earlier 32-token GREEN checked only a short stop condition and did not prove content correctness.

## Defect Tracks

### Track A: Shared request-entry corruption

Affects MTP and DFlash. Investigate physical-slot reuse, process-global state, prompt-final output-row mapping, target memory initialization, and first-token sampler input.

### Track B: DFlash rollback

DFlash rollback re-evaluation submits tokens at positions already represented by restored recurrent state. Diagnose independently. This cannot explain MTP.

## Mandatory Diagnostic Order

Do not reorder:

1. Explicit `np=2` slot schedule `0,1,0,1` plus first-token tracing.
2. Temporary full target-memory data clear, paired against control, then exact restoration.
3. DFlash rollback/re-evaluation position diagnosis.
4. Bounded fresh-sequence diagnostic with `np=4` schedule `0,1,2,3,0,1,2,3`.
5. Root-cause decision gate.
6. One minimal shared-entry RED/GREEN fix.
7. One separate DFlash rollback RED/GREEN fix if required.
8. Fresh-build complete correctness closure.

Fresh sequence IDs are a diagnostic, not an approved production design. Recurrent-core modification is disallowed until direct evidence shows `seq_rm` or new-sequence initialization violates its postconditions.

## Plan Review

The final plan was independently reviewed by GLM-5.2 after two correction rounds. Final decision: `READY - no blocking contradiction or unsafe command`.

Review output for this session:

```text
/tmp/beellama-second-pass-plan-final-review.glm.md
```

## First Command

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git status --short --branch
git rev-parse HEAD
```

Then execute Task 0 exactly.

## Milestones

After every numbered task:

1. write the milestone report;
2. request GLM-5.2 review using the exact prompt in the plan;
3. verify GLM output and concrete concerns;
4. preserve GLM's continuation summary as authoritative;
5. use smart-compaction when actual context usage is high, focused on that summary and the exact next command.

If GLM is unavailable, retry the saved prompt up to three times. If all attempts fail, record the exact errors and stop at that milestone for human direction; do not claim a review occurred.

## Stop Conditions

Stop and report to the human if:

- model files or Vulkan device are unavailable;
- source cannot be restored byte-identically after a diagnostic;
- three production fixes have failed and architecture review is required;
- any action would require destructive Git operations;
- a commit or push is requested without explicit human authorization.

A failing correctness test is not a blocker. Return to the root-cause decision task with the exact first failing boundary.
