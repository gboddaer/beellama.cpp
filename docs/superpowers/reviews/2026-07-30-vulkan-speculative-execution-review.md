# Independent Review: Vulkan Speculative Plan Execution

**Reviewed branch:** `merge_llama_into_beellama_2`
**Reviewed range:** `5c78ad504..2fa704760`
**Review date:** 2026-07-30
**Review status:** NOT READY for merge or performance claims

## Scope

This review covers:

- `ab3de6854` - EOS and speculative-state changes
- `d57efd90d` - benchmark harness and results
- `a5c8fb139` - push targets added to `AGENTS.md`
- `2fa704760` - EOS completion claims added to the results document
- the temporary EOS plan at `/tmp/eos-bug-gpt56-plan.md`
- the current worktree binary and the old-fork comparison binary

## Fresh evidence gathered

### Repository and build

- Local, `gboddaer`, and `boditec` branch tips all resolve to `2fa704760`.
- `build-vulkan/bin/llama-server` builds successfully at `2fa704760`.
- Five existing tests pass: `test-sampling`, `test-server-prompt-checkpoint`, `test-dflash-ring`, `test-dflash-plumbing`, and `test-dflash-decode`.
- Three Python harness tests pass.
- None of those tests exercises `common_speculative_reset()`, repeated requests on one slot, the MTP memory filter, or speculative EOS acceptance.
- `git diff --check 5c78ad504..HEAD` fails because the committed async-decode plan contains trailing whitespace.

### Current branch runtime: same coding request twice, one server, `-np 1`

Target model: `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`
Draft model: `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`
Backend: `Vulkan0`, temp 0, top-k 20, seed 7, 200-token cap.

| Mode | Request 1 | Request 2 | Verdict |
|------|-----------|-----------|---------|
| BASE | stop, 106 tokens, 12.63 t/s, valid code | stop, 106 tokens, 12.51 t/s, identical valid code | control passes |
| MTP | stop, 98 tokens, 22.95 t/s, valid code | stop, 1 token, empty content, reported 1,000,000 t/s | repeated-request failure |
| DFlash | stop, 92 tokens, 6.80 t/s, valid code | length, 200 tokens, prompt phrase echoed 39 times, 13.12 t/s | repeated-request failure |

The MTP log shows no additional draft generation on request 2. The DFlash log explicitly says it discarded the cross ring before request 2, then generated corrupted repetitive output.

### Performance comparison

Using the same model and request shape:

- Current worktree binary (`2fa704760`): DFlash request 1 = 6.46-6.80 t/s.
- Old-fork binary (`adb92b36a`, SHA-256 `432292cc9852...`): DFlash request 1 = 31.39-31.63 t/s.
- BASE is approximately 12.5 t/s on both lines of development.

The current branch therefore still has a DFlash-specific performance gap of approximately 4.9x in this short controlled test. A clean rebuilt reference is still required before publishing the exact ratio.

### Provenance failure

The published results document names revision `5c78ad504`, but its server path and SHA point to `/crypt/beellama.cpp/build-vulkan/bin/llama-server`, which reports version `adb92b36a`. The results measured the old-fork executable while recording the worktree source revision.

The current harness also cannot start DFlash with the current worktree binary because it unconditionally passes two documented Bee arguments that are missing from the merged argument parser:

- `--spec-branch-budget`
- `--spec-dflash-cross-ctx`

No raw JSONL records remain under `bench/vulkan-gap`; only 30 local server logs remain. The published statistics cannot be independently reconstructed.

## Findings

### Critical

#### C1. `common_speculative_reset()` dereferences null

**File:** `common/speculative.cpp:4299-4302`

```cpp
if (!spec || !spec->impls.empty()) {
    for (auto & impl : spec->impls) {
```

If `spec == nullptr`, the condition is true and the loop dereferences null. The API should return immediately for null and validate `seq_id` before indexing per-sequence vectors.

#### C2. The correctness fix is disproven

**Files:** `tools/server/server-context.cpp:2305-2320`, `common/speculative.cpp:4299-4323`

Both speculative modes fail the second request on the same slot while BASE passes. Server restart is still required by the harness. The branch is not correct for normal persistent-server use.

#### C3. Draft KV reset uses the wrong sequence abstraction

**File:** `tools/server/server-context.cpp:2313,2318`

The clear uses `slot.id`, while per-slot DFlash currently reports `get_seq_id() == 0`. The shared DFlash draft context and per-slot spec state do not use one consistent sequence identity. Slots above zero are especially unsafe. Historical commit `ad46a8703` explicitly recorded that multi-slot generated zero drafts and still needed investigation.

#### C4. The MTP layer-bound explanation is false

**Files:** `src/llama-hparams.cpp:281-283`, `src/llama-model.cpp:2173-2180`

`llama_hparams::n_layer()` already returns:

```cpp
return n_layer_all - n_layer_nextn;
```

The new code subtracts `n_layer_nextn` a second time. The commit message says `n_layer()` returns total layers, which contradicts the implementation. This can include the wrong cache layer and must be corrected or justified with exact model metadata and a model-backed test.

#### C5. Results were attributed to the wrong executable

**Files:** `bench/vulkan-gap/run.py`, `docs/superpowers/results/2026-07-29-gap-closure.md`

The source revision and executable revision were different. The current branch's DFlash throughput and correctness were not what the results document reported.

#### C6. Reduced verification is not active in the current branch

**File:** `tools/server/server-context.cpp:4438-4448`

The current code explicitly sets `dflash_compact_verify_active = false` and has no caller of `llama_set_dflash_consume_reduced()`. The results document's statement that all cycles used reduced verification came from the old executable, not the branch under review.

### Important

#### I1. Draft cache is cleared asymmetrically

The new launch code clears draft KV while preserving target prompt/KV reuse. If the target reuses an LCP but the draft is empty, target and draft positions diverge. A safe first fix should clear both sides for speculative requests, or implement coordinated target/draft prefix reuse. Clearing only one side is not a valid cache policy.

#### I2. `common_speculative_reset()` is incomplete and tightly coupled

It downcasts implementation types, resets selected fields, does not reset `impl_last` or draft parameters, and has no bounds checks. Reset ownership belongs in each implementation through a virtual request-reset hook or another explicit lifecycle API.

#### I3. The EOS plan was hypothesis-heavy but execution skipped its gates

The temporary plan correctly noticed that clearing in `begin()` would occur after prefill and revised the reset location. It did not prove the DFlash fallback hypothesis, did not provide a red/green regression test, and did not specify the four additional MTP changes later bundled into one commit. The implementation therefore cannot identify which change produced which effect.

#### I4. The harness hides the defect

`run.py` skips warm-up and restarts MTP/DFlash between repetitions. That turns a persistent-server correctness failure into apparently valid one-request samples. Restart must be an explicit diagnostic option, not the default verification path.

#### I5. Invalid outputs are counted as valid measurements

The harness accepts any positive token count and throughput. A one-token empty response at 1,000,000 t/s passes its current filter. `summarize.py` also treats it as valid. Finish reason, empty content, prompt echo, syntax, expected answer, and absurd timing values are not validated.

#### I6. No preserved raw evidence

There are no JSONL records from the reported matrix, no complete executable version string, and no model hashes. The results cannot be reproduced or audited.

#### I7. DFlash CLI contract regressed

The public docs advertise `--spec-branch-budget` and `--spec-dflash-cross-ctx`, and `common_params_speculative` still contains both fields, but `common/arg.cpp` no longer parses them. The harness and documented commands therefore fail on the branch binary.

#### I8. Multi-slot DFlash remains unverified

Per-slot speculative objects share one draft context. Current code uses local sequence 0 for every per-slot object to avoid a dparams bounds error. This avoids one crash but does not prove draft KV isolation. `-np 2` correctness and concurrency are release blockers.

#### I9. Documentation contradicts itself

The results document simultaneously says:

- the EOS bug is complete;
- server restart is required;
- the bug remains an open question;
- the workaround makes tests pass;
- the branch has no regression;
- the measured server is the old-fork binary.

The document must be marked superseded until new branch-native evidence exists.

#### I10. Branch-specific push instructions remain in `AGENTS.md`

Commit `a5c8fb139` remains at HEAD and the content is still present despite a later claim that it was reverted. The user has explicitly said this is the wrong location. Add a normal revert commit; do not rewrite shared history unless the user requests it.

### Minor / process

- The four commits have no `Assisted-by:` trailer despite the repository instruction for agent-authored commits.
- `d57efd90d` bundled the harness, results, and two older large plans, making review harder.
- The production comments are verbose and include task-specific diagnosis rather than concise invariants.
- The results use acceptance rate as a quality proxy even though accepted output was sometimes garbage.
- The independent GLM-5.2 review agreed on the null dereference, disproven completion claim, sequence-ID risk, invalid provenance, missing tests, and need for persistent-server and `-np 2` gates. Its concern about the shared-spec process-loop ordering was not independently confirmed; the current ownership model does justify processing owned per-slot specs first and the shared spec once afterward, but this still needs an integration counter/test.

## What appears sound

- Avoiding duplicate `common_speculative_process()` calls for a shared MTP spec is directionally correct.
- Stopping speculative acceptance when the target samples EOG is directionally correct and should remain, but requires a regression test.
- BASE Vulkan performance appears unchanged between the two tested binaries.
- The existing profiling, ring, plumbing, and sampling tests are useful, but insufficient for request lifecycle correctness.

## Review verdict

**Ready to merge:** No.

The branch builds, but the main correctness claim fails on the second request, the MTP layer change contradicts `n_layer()` semantics, the reset API contains a null dereference, DFlash multi-slot isolation is unproven, reduced verification is disabled, and the published performance results came from the wrong executable.

The recovery must proceed in this order:

1. correct the record and harness;
2. add persistent-request regression gates;
3. fix coordinated speculative request lifecycle;
4. prove `-np 2` isolation;
5. establish clean performance anchors;
6. enable compact reduced verification behind an opt-in gate;
7. profile and fix any remaining DFlash-specific gap one component at a time.
