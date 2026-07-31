# Handoff: Vulkan Speculative Correctness and Performance Recovery

## Start here

Execute:

`docs/superpowers/plans/2026-07-30-vulkan-speculative-correctness-performance-recovery.md`

Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Follow tasks in order. Do not begin performance optimization until Tasks 1-5 pass.

## Goal

On branch `merge_llama_into_beellama_2`:

1. make MTP and DFlash correct across repeated requests without server restart;
2. prove DFlash isolation with `-np 2`;
3. repair benchmark provenance and validity checks;
4. recover DFlash Vulkan throughput against a clean old-fork reference;
5. publish auditable same-binary results.

## Repository state at handoff

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- HEAD/local/gboddaer/boditec: `2fa704760e3a0395c4afc19d3e74e76669bc5abf`
- Old-fork reference commit: `adb92b36af0353870a1ab53515ec0b19d5d3618e`
- Current worktree binary SHA-256: `cdf215a9c9599e7d6f9186e6704f7fdc10422ac704f678e823fa9819c0263c79`
- Old root binary SHA-256: `432292cc98525540e2093fa8ddd3711b54d587a1bf5978a8ecd16ea0bfbcac03`
- Untracked pre-existing scratch: `.graphifyignore`, `SPEED_GAP_INVESTIGATION.md`, `TASK_BRIEF.md`, `graphify-out/`
- New uncommitted deliverables from review:
  - `docs/superpowers/reviews/2026-07-30-vulkan-speculative-execution-review.md`
  - `docs/superpowers/plans/2026-07-30-vulkan-speculative-correctness-performance-recovery.md`
  - this handoff

Do not commit or push without explicit human approval. When approved, push the verified branch to BOTH:

```bash
git push gboddaer merge_llama_into_beellama_2
git push boditec merge_llama_into_beellama_2
```

## Last execution reviewed

Commits:

- `ab3de6854` - bundled six decoder/state changes
- `d57efd90d` - harness, results, and unrelated older plans
- `a5c8fb139` - incorrectly added branch push targets to `AGENTS.md`
- `2fa704760` - claimed EOS work complete in results

The EOS plan was temporary: `/tmp/eos-bug-gpt56-plan.md`. It was not committed and its investigation gates were not completed before implementation.

## Proven current behavior

Fresh tests used the current worktree binary, Qwen3.6-27B Q4_K_M, Vulkan0, temp 0, top-k 20, seed 7, and the same coding prompt twice on one `-np 1` server.

| Mode | Request 1 | Request 2 |
|------|-----------|-----------|
| BASE | stop, 106 tokens, 12.63 t/s, valid code | stop, 106, 12.51 t/s, same valid hash |
| MTP | stop, 98, 22.95 t/s, valid code | stop, 1 token, empty content, 1,000,000 t/s |
| DFlash | stop, 92, 6.80 t/s, valid code | length, 200, instruction echoed 39 times, 13.12 t/s |

Current DFlash first-request profile smoke: approximately 6.46 t/s, 54.7% acceptance.
Old-fork binary smoke: approximately 31.39 t/s, 84.6% acceptance.
BASE is approximately 12.5 t/s on both.

These are review smokes, not final publishable benchmarks. Build a clean reference worktree before accepting the exact ratio.

Temporary evidence files (if still present):

- `/tmp/eos-review-base.log`, `/tmp/eos-review-base-{1,2}.json`
- `/tmp/eos-review-mtp.log`, `/tmp/eos-review-mtp-{1,2}.json`
- `/tmp/eos-review-dflash.log`, `/tmp/eos-review-dflash-{1,2}.json`
- `/tmp/eos-review-dflash-old.log`
- `/tmp/eos-review-profile-head.log`, `/tmp/eos-review-profile-old.log`
- `/tmp/eos-independent-review-prompt.txt`

## Critical source findings

1. `common/speculative.cpp:4300`:

```cpp
if (!spec || !spec->impls.empty())
```

This dereferences null when `spec == nullptr`.

2. `src/llama-hparams.cpp:281-283` proves `n_layer()` already subtracts `n_layer_nextn`. `src/llama-model.cpp:2178` subtracts it again.

3. `tools/server/server-context.cpp:2313` clears draft KV with `slot.id`; per-slot DFlash currently reports sequence 0. Sequence ownership is inconsistent and `-np 2` is unproven.

4. The launch reset clears draft state while retaining target prompt/LCP state. This asymmetric policy is the leading repeated-request hypothesis.

5. `tools/server/server-context.cpp:4448` forces `dflash_compact_verify_active=false`; there is no server caller of `llama_set_dflash_consume_reduced()`. Reduced verification is not active.

6. The harness always passes `--spec-branch-budget` and `--spec-dflash-cross-ctx`, but the merged `common/arg.cpp` parser lacks both arguments. The worktree binary rejects them.

7. The old results document records worktree source revision but used server binary SHA `432292cc...`, which reports `adb92b36a`. No raw JSONL records remain.

8. `AGENTS.md` still contains the branch-specific push block. The user explicitly wants it removed from that global file and stored in work-specific plan/handoff docs instead.

## Verification already run

Fresh at HEAD:

```text
python3 -m unittest bench.vulkan-gap.test_harness -v
3 tests passed

cmake --build build-vulkan --target llama-server
passed

CTest: test-sampling, test-server-prompt-checkpoint, test-dflash-ring,
test-dflash-plumbing, test-dflash-decode
5/5 passed
```

Important: none of these tests covers the failing repeated-request behavior or the new reset code.

`git diff --check 5c78ad504..HEAD` currently fails on trailing whitespace in the committed async-decode plan.

## Independent reviewer

A separate GLM-5.2 review agreed that:

- the null condition is critical;
- completion claims are disproven;
- draft sequence identity is unsafe;
- DFlash performance results are invalid for this branch;
- repeated request, EOS, provenance, and `-np 2` tests are mandatory.

The coordinator independently found a stronger MTP issue: the commit's claim about `n_layer()` semantics is contradicted directly by `llama_hparams::n_layer()`.

## First implementation actions

1. Read the review and full recovery plan.
2. Verify local and both remote SHAs.
3. Remove the push block from `AGENTS.md` and mark old results superseded.
4. Repair the harness and add persistent/concurrent validity tests before decoder edits.
5. Restore the two documented DFlash CLI arguments with parser tests.
6. Fix null safety and the MTP layer boundary as separate reviewed changes.
7. Reproduce MTP/DFlash repeated-request failures with the repaired harness.
8. Trace target and draft positions; then implement coordinated cache reset.

## Commands for the first evidence checkpoint

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2

git fetch gboddaer merge_llama_into_beellama_2
git fetch boditec merge_llama_into_beellama_2
printf 'local    '; git rev-parse HEAD
printf 'gboddaer '; git rev-parse gboddaer/merge_llama_into_beellama_2
printf 'boditec  '; git rev-parse boditec/merge_llama_into_beellama_2

grep -n 'if (!spec ||' common/speculative.cpp
grep -n 'return n_layer_all - n_layer_nextn' src/llama-hparams.cpp
grep -n 'dflash_compact_verify_active = false' tools/server/server-context.cpp
grep -RIn 'llama_set_dflash_consume_reduced' tools/server common
```

## Do not trust or do

- Do not trust the old results as branch measurements.
- Do not use `/crypt/beellama.cpp/build-vulkan/bin/llama-server` for branch acceptance.
- Do not restart between repetitions for correctness.
- Do not discard empty/garbage responses before summarization.
- Do not use acceptance rate as a quality oracle.
- Do not optimize before persistent and `-np 2` correctness pass.
- Do not stack fixes; one hypothesis and one A/B at a time.
- Do not rewrite pushed commits unless the user explicitly requests history rewriting.

## Completion definition

The work is complete only when all plan release gates pass on a binary built from the branch HEAD, raw records and manifest are preserved, an independent final review has no Critical/Important findings, and local/gboddaer/boditec point to the same explicitly approved commit.
