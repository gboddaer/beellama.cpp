# Vulkan Speculative Correctness and Performance Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BASE, MTP, and DFlash correct across repeated and concurrent requests on `merge_llama_into_beellama_2`, then recover DFlash Vulkan throughput against a clean old-fork reference with auditable same-binary evidence.

**Architecture:** Correct the measurement system before changing decoding. Treat target KV, draft KV, MTP carry state, and DFlash ring state as one request-lifecycle unit. Require persistent-server correctness and `-np 2` isolation before optimizing. Enable compact reduced verification only as an opt-in, per-view state after correctness is proven; profile and change one remaining bottleneck at a time.

**Tech Stack:** C++17, llama.cpp server, Vulkan/RADV, Python 3 standard library, CMake/CTest, JSONL.

## Global Constraints

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`.
- Branch: `merge_llama_into_beellama_2`.
- Current reviewed HEAD: `2fa704760e3a0395c4afc19d3e74e76669bc5abf`.
- Clean old-fork reference commit: `adb92b36af0353870a1ab53515ec0b19d5d3618e`.
- Target: `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`.
- DFlash draft: `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`.
- Device: exact `Vulkan0: AMD Radeon Graphics (RADV GFX1151)` line selected by the harness.
- Endpoint: `/v1/completions`.
- Branch measurements MUST use `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build-vulkan/bin/llama-server` built from the checked-out HEAD.
- The old-fork binary is reference-only and must have a separate label, version, SHA-256, build directory, and JSONL file.
- Do not use server restart, skipped warm-up, or filtered-out failures to pass correctness gates.
- Record invalid responses as records with reasons; never silently drop them.
- Do not optimize before Tasks 1-5 pass.
- Do not combine independent hypotheses in one code change.
- Do not rewrite pushed history. Revert bad content with new commits.
- Do not commit or push without explicit human approval for that action. Agent-authored commits require `Assisted-by: <actual assistant model>`.
- When approved to push, push the same branch tip to both:
  - `git push gboddaer merge_llama_into_beellama_2`
  - `git push boditec merge_llama_into_beellama_2`

## Release gates

### Correctness gates

1. BASE, MTP, and DFlash each complete one warm-up plus five measured requests on one persistent `-np 1` server.
2. All five measured responses are non-empty, have finite throughput below 10,000 t/s, and pass prompt-specific validation.
3. Coding responses have `finish_reason=stop`, do not repeat `Only output the code`, and contain compilable Python defining `fibonacci`.
4. Math responses contain the correct answer `2.4` hours and do not echo the prompt.
5. MTP and DFlash generate draft tokens on every measured request (`draft_n > 0`).
6. Two concurrent distinct requests on `-np 2`, repeated three times, pass their own validators with no cross-slot text.
7. No HTTP 500, decode failure, empty output, 1,000,000 t/s sentinel, or server restart.

### Performance gates

1. HEAD BASE median generation t/s is within 5% of the clean reference BASE median.
2. MTP coding median is at least 1.5x HEAD BASE after correctness passes.
3. DFlash median is at least 90% of the clean reference DFlash median and at least 1.5x HEAD BASE.
4. If the clean reference DFlash median is at least 26.7 t/s, HEAD DFlash must be at least 24.0 t/s.
5. Five valid measured rows per cell; report median, min, max, MAD, finish reasons, validity, draft acceptance, and binary/model provenance.

---

## File Structure

**Create:**

- `docs/superpowers/reviews/2026-07-30-vulkan-speculative-execution-review.md` - independent evidence and findings.
- `docs/superpowers/HANDOFF-vulkan-speculative-correctness-performance-recovery.md` - fresh-context handoff.
- `bench/vulkan-gap/repeated_requests.py` - persistent-server and concurrent correctness driver.
- `bench/vulkan-gap/test_repeated_requests.py` - model-free tests for response validation and provenance.
- `bench/vulkan-gap/records/<run-id>/*.jsonl` - preserved raw records for accepted final runs.

**Modify:**

- `AGENTS.md` - remove the branch-specific push-target block added by `a5c8fb139`.
- `bench/vulkan-gap/run.py` - persistent server by default, provenance checks, complete records.
- `bench/vulkan-gap/summarize.py` - validity-aware summaries and cross-provenance rejection.
- `bench/vulkan-gap/test_harness.py` - validation, CLI, and invalid-record unit tests.
- `common/arg.cpp` - restore documented DFlash arguments.
- `common/sampling.cpp` - retain tested EOG termination only.
- `common/speculative.cpp` / `.h` - safe explicit request reset lifecycle.
- `src/llama-model.cpp` - correct MTP layer boundary.
- `tools/server/server-context.cpp` - coordinated target/draft lifecycle, sequence identity, compact verification.
- `tests/test-arg-parser.cpp` - public CLI contract tests.
- `tests/test-speculative.cpp` - null reset and lifecycle helper tests.
- `docs/superpowers/results/2026-07-29-gap-closure.md` - mark old results superseded, then replace with branch-native evidence.

---

### Task 0: Freeze the audit baseline and correct repository documentation

**Files:**
- Modify: `AGENTS.md:149-154`
- Modify: `docs/superpowers/results/2026-07-29-gap-closure.md`
- Review: `docs/superpowers/reviews/2026-07-30-vulkan-speculative-execution-review.md`

**Interfaces:**
- Consumes: reviewed HEAD `2fa704760` and the independent runtime evidence.
- Produces: an honest starting point; no document claims the bug is fixed.

- [ ] **Step 1: Verify both remotes and the local branch start at the reviewed SHA**

Run:

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git fetch gboddaer merge_llama_into_beellama_2
git fetch boditec merge_llama_into_beellama_2
printf 'local    '; git rev-parse HEAD
printf 'gboddaer '; git rev-parse gboddaer/merge_llama_into_beellama_2
printf 'boditec  '; git rev-parse boditec/merge_llama_into_beellama_2
```

Expected: all three print `2fa704760e3a0395c4afc19d3e74e76669bc5abf` before implementation begins.

- [ ] **Step 2: Remove the branch-specific push block from `AGENTS.md`**

Delete only the five lines beginning with `- **Push targets**`. Keep the push instructions in this plan and the handoff.

- [ ] **Step 3: Mark the old results superseded**

At the top of `docs/superpowers/results/2026-07-29-gap-closure.md`, replace the status with:

```markdown
> **SUPERSEDED:** The matrix and DFlash profile below used server binary
> `432292cc9852...` reporting commit `adb92b36a`, not the recorded worktree
> revision. Repeated-request correctness fails at `2fa704760`; do not use these
> numbers as branch acceptance evidence. See the 2026-07-30 recovery plan.
```

Change the EOS section status from `Fixed` to `Disproven by repeated-request testing` and link the review file.

- [ ] **Step 4: Confirm the review facts against source**

Run:

```bash
grep -n 'if (!spec ||' common/speculative.cpp
grep -n 'return n_layer_all - n_layer_nextn' src/llama-hparams.cpp
grep -n 'dflash_compact_verify_active = false' tools/server/server-context.cpp
grep -RIn 'llama_set_dflash_consume_reduced' tools/server common | cat
```

Expected: the first three matches exist and the last command finds no server caller.

- [ ] **Step 5: Request human approval for the documentation correction commit**

Proposed commit scope: `AGENTS.md`, review, results status, this plan, and handoff only. Do not commit yet without approval.

---

### Task 1: Make the benchmark harness detect failures instead of hiding them

**Files:**
- Modify: `bench/vulkan-gap/run.py`
- Modify: `bench/vulkan-gap/summarize.py`
- Modify: `bench/vulkan-gap/test_harness.py`
- Create: `bench/vulkan-gap/repeated_requests.py`
- Create: `bench/vulkan-gap/test_repeated_requests.py`

**Interfaces:**
- Consumes: `/v1/completions` JSON and server `--version` output.
- Produces: `measurement.valid: bool`, `measurement.invalid_reasons: list[str]`, `finish_reason`, provenance, and persistent/concurrent request records.

- [ ] **Step 1: Add failing model-free tests for response classification**

Add tests with these exact cases:

```python
def test_rejects_empty_one_token_sentinel(self):
    response = {
        "choices": [{"text": "", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
        "timings": {"predicted_per_second": 1_000_000.0},
    }
    got = MOD.extract_measurement(response, "coding")
    self.assertFalse(got["valid"])
    self.assertIn("empty_content", got["invalid_reasons"])
    self.assertIn("implausible_tps", got["invalid_reasons"])


def test_rejects_prompt_echo(self):
    text = ("Only output the code. " * 20).strip()
    response = {
        "choices": [{"text": text, "finish_reason": "length"}],
        "usage": {"completion_tokens": 100},
        "timings": {"predicted_per_second": 20.0, "draft_n": 80, "draft_n_accepted": 70},
    }
    got = MOD.extract_measurement(response, "coding")
    self.assertFalse(got["valid"])
    self.assertIn("prompt_echo", got["invalid_reasons"])


def test_accepts_compilable_fibonacci(self):
    text = "```python\ndef fibonacci(n, memo=None):\n    return n\n```"
    response = {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 20},
        "timings": {"predicted_per_second": 20.0},
    }
    got = MOD.extract_measurement(response, "coding")
    self.assertTrue(got["valid"])
```

Also add math cases that accept text containing `2.4` and reject a wrong answer.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python3 -m unittest bench.vulkan-gap.test_harness bench.vulkan-gap.test_repeated_requests -v
```

Expected: failures because `extract_measurement()` lacks `prompt_kind`, validity, and finish reason.

- [ ] **Step 3: Implement strict classification**

`extract_measurement(response, prompt_kind)` must record:

```python
{
    "tokens_predicted": int,
    "predicted_per_second": float,
    "draft_n": int,
    "draft_n_accepted": int,
    "draft_accept_pct": float | None,
    "finish_reason": str | None,
    "content_sha256": str,
    "content": str,
    "valid": bool,
    "invalid_reasons": list[str],
}
```

Invalid reasons:

- `empty_content` when `content.strip()` is empty;
- `too_few_tokens` when completion tokens <= 1;
- `invalid_tps` when t/s <= 0;
- `implausible_tps` when t/s >= 10,000;
- `prompt_echo` when the literal instruction phrase occurs more than twice;
- `coding_not_stopped` when coding finish reason is not `stop`;
- `coding_not_python` when fenced/unfenced Python cannot compile or lacks `def fibonacci`;
- `math_wrong_answer` when normalized text lacks `2.4`;
- `spec_no_drafts` when mode is MTP/DFlash and `draft_n == 0` after prompt processing.

Use `compile(code, "<model-output>", "exec")`; do not execute model-generated code.

- [ ] **Step 4: Preserve failed requests**

For HTTP errors, JSON errors, invalid output, and server exit, write a JSONL row with `valid=false`, error type/body, request index, server log path, and provenance. Never `continue` without writing a row.

- [ ] **Step 5: Make persistent-server behavior the default**

Remove the implicit `restart_between_reps = args.mode in (...)` policy. Add explicit debug flags:

```text
--restart-between-reps   default false
--skip-warmup            default false
```

Default sequence: start once, one 32-token warm-up, five measured requests, stop once.

- [ ] **Step 6: Add executable provenance checks**

Record full `server --version`, executable SHA-256, source HEAD, model/draft size and SHA-256, full command, environment, and device line. If the server path is inside the worktree, fail preflight unless the commit printed by `--version` is a prefix of source HEAD. External binaries require `VK_GAP_ALLOW_EXTERNAL_SERVER=1` and an explicit `reference_label`.

- [ ] **Step 7: Add concurrent correctness driver**

`repeated_requests.py` must support:

```bash
python3 bench/vulkan-gap/repeated_requests.py --mode dflash --np 2 --rounds 3
```

For each round, send coding and math requests concurrently using `concurrent.futures.ThreadPoolExecutor(max_workers=2)`. Store both results and reject coding output containing train/math phrases or math output containing `def fibonacci`.

- [ ] **Step 8: Make the summarizer validity-aware**

`summarize.py` must:

- report total, valid, and invalid counts;
- list invalid reasons;
- refuse to combine different executable SHA/revisions in one group unless `--allow-mixed-provenance` is passed;
- calculate throughput only from valid rows;
- exit nonzero if any required repetition is invalid;
- report finish reason counts and draft_n counts.

- [ ] **Step 9: Run tests and verify GREEN**

```bash
python3 -m unittest bench.vulkan-gap.test_harness bench.vulkan-gap.test_repeated_requests -v
```

Expected: all model-free tests pass.

- [ ] **Step 10: Verify process cleanup**

Run preflight with a missing model and interrupt a live smoke run. Verify no `llama-server` child remains and all log handles close. Use `try/finally` around the whole server lifecycle.

**Checkpoint:** ask for review before any decoder changes.

---

### Task 2: Restore the public CLI and remove known unsafe code

**Files:**
- Modify: `common/arg.cpp`
- Modify: `tests/test-arg-parser.cpp`
- Modify: `common/speculative.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `tools/server/server-context.cpp`
- Test: `tests/test-speculative.cpp`

**Interfaces:**
- Produces: documented DFlash CLI, null-safe reset, correct MTP layer boundary, one shared-spec process call.

- [ ] **Step 1: Add failing argument parser tests**

Append:

```cpp
argv = {"binary_name", "--spec-branch-budget", "4", "--spec-dflash-cross-ctx", "1024"};
assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
assert(params.speculative.branch_budget == 4);
assert(params.speculative.branch_budget_explicit);
assert(params.speculative.dflash_cross_ctx == 1024);
```

Build and run `test-arg-parser`; expected RED because both arguments are currently missing.

- [ ] **Step 2: Restore the two parser blocks from `adb92b36a`**

Use the existing fields and environment names:

```cpp
{"--spec-branch-budget"} -> LLAMA_ARG_SPEC_BRANCH_BUDGET
{"--spec-dflash-cross-ctx"} -> LLAMA_ARG_SPEC_DFLASH_CROSS_CTX
```

Set `branch_budget_explicit = true`, clamp branch budget to >= 0, and call `note_dflash_only_arg()` for both.

- [ ] **Step 3: Add and pass a null reset test**

In `tests/test-speculative.cpp` call:

```cpp
common_speculative_reset(nullptr, 0);
```

Expected after the fix: no crash. Implement:

```cpp
if (spec == nullptr) {
    return;
}
GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) spec->dparams.size());
```

Then iterate implementations. Add bounds assertions before indexing MTP vectors.

- [ ] **Step 4: Correct the MTP layer boundary**

Replace the double subtraction with:

```cpp
filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
```

Before changing it, record the model startup values for `n_layer_all`, `n_layer_nextn`, and `n_layer()` in the task log. The expected Qwen model invariant is `n_layer() + n_layer_nextn == n_layer_all`.

- [ ] **Step 5: Restore unproven MTP context setup to the reviewed base**

Re-add:

```cpp
cparams_mtp.ctx_other = ctx_tgt;
```

Source search currently shows Qwen35 does not consume `ctx_other`, so this should be behavior-neutral. Keeping the base setup removes an unsupported root-cause claim. If a focused A/B proves it changes behavior, document that evidence before choosing the final state.

- [ ] **Step 6: Keep but verify the shared-spec process-loop correction**

Keep `if (s.spec && ...)` for owned per-slot specs and process context-level `spec` once afterward. Add an environment-gated counter for one test run. Gate:

- one MTP process call per decoded batch, not `n_slots + 1`;
- one DFlash process call per active owned slot;
- remove the temporary counter after the assertion is captured.

- [ ] **Step 7: Build and run cheap tests**

```bash
cmake --build build-vulkan --target test-arg-parser test-speculative test-sampling llama-server -j"$(nproc)"
ctest --test-dir build-vulkan --output-on-failure -R '^(test-arg-parser|test-speculative|test-sampling)$'
build-vulkan/bin/llama-server --help | grep -E -- '--spec-branch-budget|--spec-dflash-cross-ctx'
```

Expected: all tests pass and both arguments appear.

**Checkpoint:** do not bundle Task 2 with the request-lifecycle fix. Request review first.

---

### Task 3: Fix single-slot request lifecycle with coordinated cache state

**Files:**
- Modify: `tools/server/server-context.cpp`
- Modify: `common/speculative.cpp` / `.h`
- Test: `bench/vulkan-gap/repeated_requests.py`

**Interfaces:**
- Consumes: target prompt cache, draft KV, MTP carry state, DFlash ring.
- Produces: one explicit `request_reset(seq_id, reason)` lifecycle and correct five-request persistence.

- [ ] **Step 1: Run the persistent regression and verify RED**

```bash
VK_GAP_SERVER=$PWD/build-vulkan/bin/llama-server \
VK_GAP_MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
VK_GAP_DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
python3 bench/vulkan-gap/repeated_requests.py --mode mtp --np 1 --rounds 5

python3 bench/vulkan-gap/repeated_requests.py --mode dflash --np 1 --rounds 5
```

Expected at reviewed HEAD: MTP returns an empty second response; DFlash echoes the prompt on the second response.

- [ ] **Step 2: Trace target/draft state at task launch, after prefill, and before draft**

Under `GGML_SPEC_REQUEST_TRACE=1`, log one line per boundary with:

```text
mode slot task request_phase target_pos_min target_pos_max draft_seq_id draft_pos_min draft_pos_max prompt_tokens pending_h_rows ring_filled committed_len
```

Run exactly two requests. Confirm or reject this hypothesis: the target keeps/reuses an LCP while draft KV and speculative carry state are reset to empty.

- [ ] **Step 3: Implement the safe no-reuse policy first**

For a new speculative request on an idle slot, clear the target prompt/KV and draft KV together before assigning the task:

```cpp
if (slot.can_speculate() && slot.prompt.n_tokens() > 0) {
    slot.prompt_clear(false);
}
common_context_seq_rm(slot.ctx_dft, slot.get_seq_id(), -1, -1);
common_speculative_reset(slot.get_spec(), slot.get_seq_id());
```

Do not clear only draft state while retaining target LCP state. Do not optimize speculative prompt-cache reuse in this task.

- [ ] **Step 4: Move implementation-specific reset ownership into implementations**

Replace external type-switch/downcast reset with a virtual implementation hook:

```cpp
virtual void reset_request(llama_seq_id seq_id, const char * reason) = 0;
```

MTP resets `pending_h`, batch bounds, verification rows, and last draft count for that sequence. DFlash calls `discard_cross_ring(reason)`. The common wrapper clears `impl_last[seq_id]` and sets `dparams[seq_id].drafting = false` without destroying caller-owned result pointers.

- [ ] **Step 5: Run persistent MTP and DFlash gates**

Expected for each mode: five valid rows, all `finish_reason=stop` for coding, no empty output, no echo, draft_n > 0 on every measured request, no restart.

- [ ] **Step 6: Remove request tracing or leave it strictly environment-gated**

Default logs must remain quiet.

**Decision gate:** only after this task passes may a later task implement coordinated speculative prompt-cache reuse. Correct full prefill is preferred over fast corrupted reuse.

---

### Task 4: Fix DFlash multi-slot sequence identity and prove isolation

**Files:**
- Modify: `tools/server/server-context.cpp`
- Modify: `common/speculative.cpp` / `.h` if sequence capacity assertions require it
- Test: `bench/vulkan-gap/repeated_requests.py`

**Interfaces:**
- Produces: unique draft-context sequence per server slot while preserving per-slot ring objects.

- [ ] **Step 1: Run `-np 2` and verify RED or no-draft behavior**

```bash
python3 bench/vulkan-gap/repeated_requests.py --mode dflash --np 2 --rounds 3
```

Record response validity, draft_n by slot, and draft-context positions by sequence.

- [ ] **Step 2: Give each per-slot DFlash spec enough sequence capacity**

Initialize each owned DFlash spec with `n_seq_in = params_base.n_parallel`:

```cpp
slot.spec.reset(common_speculative_init(
    params_base.speculative, slot.ctx_tgt, ctx_dft.get(), params_base.n_parallel));
```

- [ ] **Step 3: Use `slot.id` as the shared draft-context sequence ID**

Change the slot sequence accessor so DFlash operations use the physical server slot ID, not local zero. The larger dparams allocation from Step 2 prevents the old out-of-bounds failure.

Add assertions:

```cpp
GGML_ASSERT(slot.get_seq_id() >= 0);
GGML_ASSERT(slot.get_seq_id() < params_base.n_parallel);
```

All draft KV clears, rollback removes, begin, draft params, accept, and reset calls must use `slot.get_seq_id()` consistently.

- [ ] **Step 4: Verify two-slot isolation**

Run three concurrent rounds. Gates:

- coding validates as Python and never contains train/math text;
- math contains `2.4` and never contains `def fibonacci`;
- each slot has `draft_n > 0`;
- no out-of-range hidden capture warning;
- no sequence position warning;
- no cross-slot ring/capture routing warning.

- [ ] **Step 5: Re-run `-np 1`**

Single-slot correctness and performance must not regress by more than 5% from Task 3.

---

### Task 5: Prove EOS behavior end to end

**Files:**
- Modify: `common/sampling.cpp` only if the test exposes a defect
- Test: `bench/vulkan-gap/test_repeated_requests.py`
- Test: model-backed persistent runs

**Interfaces:**
- Produces: no sampling or emitted token after EOG for the same generation.

- [ ] **Step 1: Keep the EOG check after sampling/accept and before bonus sampling**

The target-sampled EOG is terminal whether it matches the draft or represents a rejection. Do not remove it merely because the earlier commit lacked tests.

- [ ] **Step 2: Add response-level EOS assertions**

For coding, all modes must finish with `stop` below the token cap. Repeat five times on one server.

- [ ] **Step 3: Run one QA-traced request per speculative mode**

With `GGML_DFLASH_QA_TRACE=1`, verify that once `sample_accept EOS` is logged for a task, no later `sample_accept i=` or `bonus_sampled=` line appears for that task.

- [ ] **Step 4: Verify no post-EOS prompt contamination**

Immediately send the same prompt again. It must produce a valid non-empty response and generate drafts. This couples EOS correctness to request lifecycle correctness.

---

### Task 6: Build clean performance references and preserve provenance

**Files:**
- Create: `bench/vulkan-gap/records/<run-id>/manifest.json`
- Create: separate JSONL per revision/mode/prompt
- Modify: results document only after runs finish

**Interfaces:**
- Produces: clean BASE and DFlash reference medians and branch medians under one protocol.

- [ ] **Step 1: Create an isolated reference worktree**

Use the `superpowers:using-git-worktrees` skill. Create:

```bash
git worktree add /crypt/beellama.cpp/.worktrees/dflash-reference-adb92 adb92b36a
```

Do not move HEAD in the active worktree.

- [ ] **Step 2: Configure both builds identically**

```bash
cmake -S /crypt/beellama.cpp/.worktrees/dflash-reference-adb92 \
      -B /crypt/beellama.cpp/.worktrees/dflash-reference-adb92/build-vulkan \
      -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /crypt/beellama.cpp/.worktrees/dflash-reference-adb92/build-vulkan \
      --target llama-server -j"$(nproc)"

cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target llama-server -j"$(nproc)"
```

- [ ] **Step 3: Verify binary provenance**

Record `--version`, SHA-256, `git status --short`, compiler, CMake options, Vulkan device line, Mesa/Vulkan driver, model hashes, and command. Reject a dirty tracked checkout.

- [ ] **Step 4: Run one warm-up and five persistent requests**

Run BASE and DFlash on the clean reference, then BASE/MTP/DFlash on HEAD. Use separate JSONL files; never append two executable SHAs to one file.

- [ ] **Step 5: Apply performance gates**

If BASE differs by more than 5%, stop and resolve environment/build parity. If any correctness row is invalid, stop and return to Tasks 3-5. Only then compare speculative throughput.

---

### Task 7: Activate compact reduced verification safely and opt-in

**Files:**
- Modify: `tools/server/server-context.cpp`
- Test: reduced-off/on persistent and `-np 2` runs

**Interfaces:**
- Consumes: `dflash_select_reduced_verify_plan()`, `dflash_batch_view_is_reduced_verify()`, `llama_set_dflash_verify_logits()`, `llama_set_dflash_consume_reduced()`, and `common_sampler_sample_reduced_and_accept_n()`.
- Produces: compact consumption only for a compatible whole decode view.

- [ ] **Step 1: Confirm current RED state**

```bash
grep -RIn 'llama_set_dflash_consume_reduced' tools/server common
```

Expected before implementation: no server caller; current results claiming reduced verify are invalid.

- [ ] **Step 2: Implement the reviewed per-view plan**

Follow Task 2 Steps 2-4 in `docs/superpowers/plans/2026-07-29-vulkan-qwen36-spec-gap-closure.md` exactly:

- require `GGML_DFLASH_REDUCED_VERIFY=1`;
- recompute eligibility for every decode view;
- require all speculative slots to support the same top-k in `[1,64]`;
- call `llama_set_dflash_consume_reduced(ctx_tgt, true)` only before an eligible decode;
- restore it to false immediately after decode and on every retry/error/destroy path;
- map compact rows using `absolute_index - off` for each slot;
- never fall back to full sampling after a decode that omitted full logits.

- [ ] **Step 3: Verify opt-out first**

Reduced-off output must pass Tasks 3-5 and match the pre-change semantic validators.

- [ ] **Step 4: Verify greedy reduced mode**

Five persistent coding and math requests must all be valid. Profile must state compact output was consumed for eligible verification views.

- [ ] **Step 5: Verify supported top-k and unsupported fallback**

Run temp 0.6/top-k 20 with fixed seed against reduced off/on. Run top-k 0 or a grammar request and verify compact mode is rejected before decode while full-logits mode succeeds.

- [ ] **Step 6: Verify `-np 2` row isolation**

Run the Task 4 concurrent gate with reduced mode on and off. Each prompt must match its own validator; no row-zero reuse across slots.

- [ ] **Step 7: Measure reduced off/on**

Keep it opt-in unless correctness passes and median DFlash throughput improves by at least 5%. Do not flip the default in this task.

---

### Task 8: Profile and fix the remaining DFlash performance gap

**Files:**
- Modify only the component proven by profiling
- Optionally restore environment-gated cycle timing from `adb92b36a:tools/server/server-context.cpp:7541`
- Update: results evidence

**Interfaces:**
- Produces: one confirmed bottleneck and one minimal fix per iteration.

- [ ] **Step 1: Capture phase medians on clean reference and HEAD**

Use:

```bash
GGML_DFLASH_PROFILE=1 GGML_DFLASH_PROFILE_SYNC_SPLIT=1 \
GGML_DFLASH_REDUCED_VERIFY=1 \
python3 bench/vulkan-gap/run.py dflash math --repetitions 5 --gen-tokens 512
```

Record per-cycle draft, verify, accept, rollback/reeval, scheduler sync, graph reuse, output rows, cross length, and total.

If current code lacks cycle summaries, port only the environment-gated summary format from the clean reference before optimizing.

- [ ] **Step 2: Compare against component gates**

Clean-reference review evidence was approximately 26.6 ms median draft, 120.3 ms verify, and 148.7 ms total per cycle. Recompute these from the clean rebuild. HEAD target for each component is <= 1.20x its clean-reference median.

- [ ] **Step 3: Use this decision tree**

- Verify >1.20x reference: inspect compact consume state, verify rows/output rows, full-logits transfer, and target scheduler synchronization.
- Draft >1.20x: inspect deferred drafter KV flush, `update_drafter_kv_cache`, graph reuse, `n_outputs_max`, projection-cache fallback, and cross length.
- Rollback/accept >1.20x: inspect tape/recurrent checkpoint restore and re-evaluation count.
- Unaccounted time >10%: trace `ggml_backend_sched_synchronize`, Vulkan queue waits, and CPU/GPU boundaries.
- `GGML_DFLASH_GPU_RING=0` changes total by >10%: isolate ring synchronization/copy path.

State one hypothesis in the task log, change one variable, rerun five samples, then accept or reject it.

- [ ] **Step 4: Bisect only if profiling does not identify the component**

`adb92b36a` is an ancestor of `5c78ad504` with 110 commits between them. Use a separate bisect worktree. The bisect script must:

- build the worktree binary;
- reject unsupported/unbuildable commits with exit 125;
- use only arguments supported at both endpoints;
- send one fixed math request after warm-up;
- classify good as valid output and >= 80% of the measured good-anchor t/s;
- save commit, binary SHA, response, and log for every step.

Do not bisect on the active worktree and do not classify corrupted high-acceptance output as good.

- [ ] **Step 5: Implement one minimal root-cause fix**

Create a failing test or A/B measurement before the change. Re-run correctness, `-np 2`, component profile, and five-repetition throughput afterward. Revert the change if its component median does not improve or correctness regresses.

- [ ] **Step 6: Repeat only if a second component still fails**

Do not stack speculative optimizations. Each accepted fix gets its own review boundary.

---

### Task 9: Final verification, results, and handoff closure

**Files:**
- Modify: `docs/superpowers/results/2026-07-29-gap-closure.md`
- Modify: handoff with final SHAs and open issues
- Preserve: final JSONL and manifest

- [ ] **Step 1: Run all model-free and C++ tests**

```bash
python3 -m unittest bench.vulkan-gap.test_harness bench.vulkan-gap.test_repeated_requests -v
cmake --build build-vulkan --target \
  test-arg-parser test-speculative test-sampling test-server-prompt-checkpoint \
  test-dflash-ring test-dflash-plumbing test-dflash-decode llama-server \
  -j"$(nproc)"
ctest --test-dir build-vulkan --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Run the final persistent six-cell matrix**

One warm-up plus five measured requests for BASE/MTP/DFlash x coding/math on the HEAD binary. No restart. Then run the `-np 2` concurrent gate three times.

- [ ] **Step 3: Verify every release gate line by line**

Do not claim completion merely because the build passes. Include exact valid/invalid counts, finish reasons, t/s stats, acceptance, and reference ratio.

- [ ] **Step 4: Replace superseded results with auditable results**

Link the raw JSONL files and manifest, include full binary commits/SHAs, and clearly distinguish cold-start, prompt processing, and generation throughput.

- [ ] **Step 5: Request independent code review**

Use the `superpowers:requesting-code-review` skill over the exact fix range. Resolve all Critical and Important findings before asking to merge.

- [ ] **Step 6: Request explicit commit and push approval**

When approved, use small scoped commits with `Assisted-by:` trailers. Push the same verified tip to `gboddaer` and `boditec`, then fetch and verify all three SHAs match.

---

## Self-review checklist

- Every current review finding maps to a task: yes.
- Persistent-server and `-np 2` gates precede performance work: yes.
- The plan does not treat acceptance as correctness: yes.
- The branch and reference binaries cannot be confused silently: yes.
- Reduced verify remains opt-in until measured and verified: yes.
- Root-cause profiling precedes performance fixes: yes.
- No server restart is allowed to hide correctness defects: yes.
- Push targets are recorded in this work-specific plan/handoff, not global `AGENTS.md`: yes.
