# Vulkan Speculative Correctness Second-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task-by-task. Also use `superpowers:systematic-debugging`, `superpowers:test-driven-development` before production changes, `superpowers:verification-before-completion`, `graphify`, and `context-mode`. Do not use subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and fix shared MTP/DFlash request-entry corruption and the separate DFlash rollback duplicate-position defect, then pass the complete Vulkan correctness suite.

**Architecture:** First separate physical-slot reuse, process-global state, and sequence-memory state using controlled slot schedules and first-token tracing. Next run a temporary full-memory-clear discriminator, then diagnose DFlash rollback independently. Only after those observations may the agent select one minimal RED/GREEN production fix at a time.

**Tech Stack:** C++17, CMake, Vulkan, Python 3, `unittest`, CTest, llama-server `/completion`, Qwen3.6-27B target GGUF, DFlash draft GGUF, GLM-5.2 through Ollama, context-mode, and smart-compaction.

## Global Constraints

- Work in `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`.
- Starting candidate is expected to be `d0e3ca514248`; if it differs, record the new SHA and revalidate assumptions.
- Treat the `prompt_clear(false)` hunk added by `d0e3ca514` as experimental, not accepted.
- Do not commit, push, create a PR, or begin performance optimization without new explicit human authorization.
- Do not run `git stash`, `git reset`, `git clean`, whole-tree restore, or destructive checkout.
- Use one hypothesis and one diagnostic variable at a time.
- Do not modify production code before an automated RED reproduction identifies the expected boundary.
- Restore temporary diagnostics from exact snapshots and verify hashes.
- Keep `GGML_DFLASH_REDUCED_VERIFY` unset.
- Use ASCII only in source, comments, evidence, and documentation.
- Use context-mode for builds, tests, logs, Git history, JSONL, and output that can exceed 20 lines.
- The authoritative target prompt uses temperature 0, top-k 20, and seed 7.
- The authoritative generation cap is 512 tokens. Short preflights cannot prove correctness.
- A row is not GREEN merely because `stop=True`; reapply content validators and raw-token checks.
- Do not infer cross-request staleness from DFlash non-consecutive warnings. Existing evidence places those warnings in request-1 rollback re-evaluation.
- Do not infer that recurrent `seq_rm` is defective. Its full-range path already resets sequence metadata; require direct contrary evidence.

## Required Reading

Read completely before execution:

```text
/crypt/beellama.cpp/AGENTS.md
docs/superpowers/specs/2026-07-31-vulkan-speculative-correctness-second-pass-design.md
docs/superpowers/plans/2026-07-31-vulkan-speculative-correctness-second-pass.md
docs/superpowers/results/2026-07-31-vulkan-speculative-working-tree-correctness-recovery.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/hypotheses/root-cause.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/hypotheses/classification.md
/crypt/tmp/beellama-working-correctness-20260731T105547Z/milestones/M4-agent.md
```

The earlier root-cause and M4 documents are evidence of a rejected conclusion, not current truth.

## Milestone Review and Compaction Protocol

Run this after every numbered task. Replace `MILESTONE` and `NEXT_TASK` exactly.

1. Write `$EVIDENCE/milestones/MILESTONE-agent.md` with:

```text
# Milestone MILESTONE

## Candidate
HEAD, status, source diff hash, binary hash.

## Commands and exits
Every build, test, and reproduction command with real exit code.

## Evidence
Record paths, row counts, validator counts, first divergence, first-token mapping, and log findings.

## Confirmed facts
Only observations directly supported by files or commands.

## Rejected hypotheses
Prediction, observation, decision, and restoration hash.

## Open risks
Unresolved facts only.

## Next task
NEXT_TASK and its exact first command.
```

2. Write `$EVIDENCE/reviews/MILESTONE.prompt.txt`. Include the complete milestone report after this prompt:

```text
You are GLM-5.2 independently reviewing a Vulkan speculative-decoding correctness milestone.

Goal: BASE, MTP, and DFlash must be correct for fresh, persistent, and concurrent Qwen3.6-27B requests.
Candidate repository: /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
Constraints: one hypothesis at a time, no performance work, no destructive Git operations, no commit or push.

Challenge causal claims. Distinguish target prompt-final corruption from speculative draft behavior. Do not treat DFlash rollback warnings as an explanation for MTP. Do not assume full recurrent seq_rm leaves metadata stale.

Return exactly:
## Decision
PROCEED or REVISIT, with one sentence.
## Verified facts
## Risks or contradictions
## Required correction before proceeding
Use None if no correction is required.
## Continuation summary for compaction
Include candidate SHA/status, accepted and rejected hypotheses, exact evidence paths, open loop, and exact next command.

MILESTONE EVIDENCE:
```

3. Run:

```bash
python3 /home/gbo/.pi/agent/skills/glm-ollama-reviewer/scripts/ollama_second_opinion.py \
  --host http://192.168.123.123:11434 \
  --model glm-5.2:cloud \
  --timeout 600 \
  --prompt-file "$EVIDENCE/reviews/MILESTONE.prompt.txt" \
  > "$EVIDENCE/reviews/MILESTONE-glm.md"
```

4. Require a nonempty file containing `## Decision` and `## Continuation summary for compaction`. If Ollama is unavailable, retry up to three times with the same saved prompt. If all attempts fail, record the exact errors and stop at the milestone for human direction; do not claim the milestone was reviewed.

5. Load only the GLM decision, contradictions, required correction, and continuation summary into context.

6. Verify every concrete GLM concern against source or evidence. If the decision is `REVISIT`, correct factual omissions before continuing.

7. Check actual context usage. When actual context usage is at least 60%, call:

```text
smart_compact(
  profile="balanced",
  focus="Preserve the GLM continuation summary in <absolute MILESTONE-glm.md path> and the exact next command",
  max_calls=8,
  max_latency_ms=120000
)
```

Do not trigger smart compaction from tool-output percentage. If context is below the compaction threshold, preserve the GLM summary and record that compaction was deferred.

---

### Task 0: Rebaseline and Withdraw the Previous Acceptance Claim

**Files:**
- Read: current repository and prior evidence
- Create outside Git: `/crypt/tmp/beellama-spec-correctness-second-pass-<timestamp>/`
- Create outside Git: `/tmp/beellama-spec-correctness-second-pass.env`

**Interfaces:**
- Produces immutable starting snapshots, hashes, hypothesis ledger, and environment used by every later task.

- [ ] **Step 1: Initialize evidence state**

```bash
WORK=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EVIDENCE=/crypt/tmp/beellama-spec-correctness-second-pass-$STAMP
MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
mkdir -p "$EVIDENCE"/{source-start,snapshots,records,logs,milestones,reviews,hypotheses,scripts,build}
HEAD_SHA=$(git -C "$WORK" rev-parse HEAD)
cat > /tmp/beellama-spec-correctness-second-pass.env <<EOF
export WORK=$WORK
export EVIDENCE=$EVIDENCE
export MODEL=$MODEL
export DRAFT=$DRAFT
export HEAD_SHA=$HEAD_SHA
EOF
```

- [ ] **Step 2: Verify candidate and provenance**

Record complete outputs and exit codes:

```bash
git -C "$WORK" status --short --branch
git -C "$WORK" rev-parse HEAD
git -C "$WORK" diff --check
git -C "$WORK" show --format= --unified=8 d0e3ca514 -- tools/server/server-context.cpp
sha256sum "$MODEL" "$DRAFT" "$WORK/build-vulkan/bin/llama-server"
"$WORK/build-vulkan/bin/llama-server" --version
```

Expected initial status: clean except the new second-pass design/plan/handoff documents created before execution. Record those files rather than hiding them.

- [ ] **Step 3: Preserve exact source**

```bash
git -C "$WORK" diff --binary > "$EVIDENCE/source-start/working.patch"
cp -a "$WORK/tools/server/server-context.cpp" "$EVIDENCE/source-start/server-context.cpp"
cp -a "$WORK/src/llama-memory-recurrent.cpp" "$EVIDENCE/source-start/llama-memory-recurrent.cpp"
cp -a "$WORK/src/llama-context.cpp" "$EVIDENCE/source-start/llama-context.cpp"
cp -a "$WORK/common/speculative.cpp" "$EVIDENCE/source-start/speculative.cpp"
cp -a "$WORK/bench/vulkan-gap/run.py" "$EVIDENCE/source-start/run.py"
cp -a "$WORK/bench/vulkan-gap/repeated_requests.py" "$EVIDENCE/source-start/repeated_requests.py"
find "$EVIDENCE/source-start" -type f -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/source-start/SHA256SUMS"
```

- [ ] **Step 4: Write the corrected hypothesis ledger**

Create `$EVIDENCE/hypotheses/ledger.tsv`:

```text
id	track	hypothesis	prediction	command	result	decision	restored_sha256
A1	shared-entry	corruption is physical-slot-local	unused slot passes and reused slot fails	pending	pending	pending	none
A2	shared-entry	corruption is process-global/shared	second request fails even on unused slot	pending	pending	pending	none
A3	shared-entry	target memory data survives sequence removal	full data clear restores BASE-equivalent first token	pending	pending	pending	none
B1	dflash-rollback	Vulkan rollback re-evaluates an already represented position	warning position equals restored tail and disappears when position/count is corrected	pending	pending	pending	none
C1	sequence-id	fresh target sequence identity isolates corruption	first-use IDs pass while reused IDs fail	pending	pending	pending	none
```

- [ ] **Step 5: Record the withdrawal of the previous acceptance**

Write `$EVIDENCE/hypotheses/prior-fix-disposition.md`:

```text
The prompt_clear(false) hunk in d0e3ca514 is experimental.
It passed only a short stop-condition check and failed the required 512-token persistent correctness gates.
The exact 512-token RED reproduction was not proven GREEN.
No current production fix is accepted.
```

- [ ] **Step 6: Build and run cheap controls**

```bash
cmake --build "$WORK/build-vulkan" -j"$(nproc)"
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir "$WORK/build-vulkan" --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git -C "$WORK" diff --check
```

All setup and model-free commands must exit 0 before model-backed diagnosis.

- [ ] **Step 7: Run Milestone M0 protocol**

Next task: `Task 1: Explicit slot isolation and first-token tracing`.

---

### Task 1: Explicit Slot Isolation and First-Token Tracing

**Files:**
- Create: `bench/vulkan-gap/slot_isolation.py`
- Modify: `bench/vulkan-gap/test_repeated_requests.py`
- Create evidence: `$EVIDENCE/records/slot-isolation/`, `$EVIDENCE/logs/slot-isolation/`
- Do not modify production C++.

**Interfaces:**
- `slot_schedule(n_slots: int, rounds: int) -> list[int]`
- `first_divergence(reference: list[int], candidate: list[int]) -> int | None`
- Driver sends `/completion` requests with explicit `id_slot` and records raw tokens.

- [ ] **Step 1: Add model-free RED tests**

At the top of `test_repeated_requests.py`, load the new driver independently from the existing `MOD` binding:

```python
SLOT_SPEC = importlib.util.spec_from_file_location("vk_gap_slot_isolation", HERE / "slot_isolation.py")
SLOT_MOD = importlib.util.module_from_spec(SLOT_SPEC)
SLOT_SPEC.loader.exec_module(SLOT_MOD)
```

Add tests for:

```python
def test_slot_schedule_two_slots_two_rounds(self):
    self.assertEqual(SLOT_MOD.slot_schedule(2, 2), [0, 1, 0, 1])


def test_first_divergence(self):
    self.assertEqual(SLOT_MOD.first_divergence([1, 2, 3], [1, 9, 3]), 1)
    self.assertIsNone(SLOT_MOD.first_divergence([1, 2], [1, 2]))
    self.assertEqual(SLOT_MOD.first_divergence([1, 2], [1, 2, 3]), 2)
```

Before creating `slot_isolation.py`, run the named test module and require RED because the module does not exist. After creating an empty module, require RED because the helpers do not exist. This distinguishes missing-file setup from missing behavior.

- [ ] **Step 2: Implement the minimal driver**

The driver must:

- import `build_server_command`, `extract_measurement`, `find_device`, `wait_ready`, and `open_output_exclusive` from `run.py` through `importlib.util`;
- launch exactly one server;
- accept positional `mode` and options `--np`, `--rounds`, `--prompt`, `--gen-tokens`, `--output`, `--port`;
- create output exclusively;
- send deterministic payloads with `temperature=0`, `top_k=20`, `seed=7`, `return_tokens=true`, `cache_prompt=false`, and explicit `id_slot`;
- record `request_ordinal`, `id_slot`, full response measurement, raw token IDs, first token ID, and provenance;
- terminate the server in `finally`;
- never classify a row from `stop=True` alone.

The schedule implementation is exactly:

```python
def slot_schedule(n_slots, rounds):
    if n_slots <= 0 or rounds <= 0:
        raise ValueError("n_slots and rounds must be positive")
    return [slot for _ in range(rounds) for slot in range(n_slots)]
```

The divergence implementation is exactly:

```python
def first_divergence(reference, candidate):
    for i, (lhs, rhs) in enumerate(zip(reference, candidate)):
        if lhs != rhs:
            return i
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate))
```

- [ ] **Step 3: Run model-free GREEN tests**

```bash
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
```

Require all tests to pass and record the exact count.

- [ ] **Step 4: Set deterministic environment**

```bash
source /tmp/beellama-spec-correctness-second-pass.env
export VK_GAP_SERVER="$WORK/build-vulkan/bin/llama-server"
export VK_GAP_MODEL="$MODEL"
export VK_GAP_DRAFT="$DRAFT"
export VK_GAP_TEMP=0
export VK_GAP_TOP_K=20
export VK_GAP_BATCH=512
export VK_GAP_UBATCH=128
export VK_GAP_CTX_SIZE=2048
unset GGML_DFLASH_REDUCED_VERIFY
```

Enable existing first-token trace:

```bash
export GGML_NODFLASH_TOKEN_TRACE=1
```

- [ ] **Step 5: Run authoritative `np=2` schedule**

For BASE, MTP, and DFlash coding:

```bash
python3 bench/vulkan-gap/slot_isolation.py "$MODE" \
  --np 2 --rounds 2 --prompt coding --gen-tokens 512 --port "$PORT" \
  --output "$EVIDENCE/records/slot-isolation/${MODE}-coding-np2.jsonl"
```

Use unique ports per mode. Capture full server logs. Do not restart between rows within a mode.

- [ ] **Step 6: Parse first-token and reuse behavior independently**

Write an evidence-side parser that reports for each mode:

```text
slot 0 first use: valid, first token, token hash
slot 1 first use: valid, first token, token hash
slot 0 reuse: valid, first token, first divergence from slot 0 first use
slot 1 reuse: valid, first token, first divergence from slot 1 first use
target argmax from server trace for each request
```

Classification:

```text
both unused slots pass, reused slots fail -> physical-slot-local state supported
unused slot 1 fails after slot 0 -> process-global/shared state supported
all rows pass -> current failure is not reproduced; rerun the original np=1 5-row cell
BASE fails -> stop and diagnose harness/target before speculative modes
```

- [ ] **Step 7: Prove where token 0 is sampled**

From logs, align each request's prompt completion with `[NODFLASH_TOK]`. Record `tok_idx`, `sampled`, and `target_argmax`. State explicitly whether the wrong first token exists before `common_speculative_draft()` is called.

If existing trace lacks prompt-final batch/position details, copy `server-context.cpp` under `$EVIDENCE/snapshots/task1-entry-trace/` and write `SHA256SUMS.before` before adding one temporary environment-gated trace. Gate it with `GGML_SPEC_ENTRY_TRACE=1` and print:

```text
[SPEC_ENTRY] request=<task> slot=<id> state=<state> i_batch=<absolute> off=<off> tok_idx=<relative> pos_min=<min> pos_max=<max> sampled=<token> argmax=<token>
```

Save the diagnostic patch, restore the file, verify its SHA-256 equals the pre-diagnostic value, rebuild, and rerun model-free tests before leaving Task 1.

- [ ] **Step 8: Update ledger**

Decide A1 versus A2 from the explicit schedule. Do not mention recurrent memory unless the result directly tests it.

- [ ] **Step 9: Run Milestone M1 protocol**

Next task: `Task 2: Full target-memory clear discriminator`.

---

### Task 2: Full Target-Memory Clear Discriminator

**Files:**
- Temporarily modify: `tools/server/server-context.cpp`
- Create: `$EVIDENCE/scripts/full-clear-control.sh`
- Restore source before milestone completion.

**Interfaces:**
- Environment diagnostic `GGML_SPEC_TEST_FULL_TARGET_CLEAR=1` is temporary and must not remain in source.

- [ ] **Step 1: Preserve the exact source**

```bash
mkdir -p "$EVIDENCE/snapshots/task2-full-clear/tools/server"
cp -a "$WORK/tools/server/server-context.cpp" \
  "$EVIDENCE/snapshots/task2-full-clear/tools/server/server-context.cpp"
sha256sum "$WORK/tools/server/server-context.cpp" \
  > "$EVIDENCE/snapshots/task2-full-clear/SHA256SUMS.before"
```

- [ ] **Step 2: State the single hypothesis**

Append A3 prediction to the ledger:

```text
If per-sequence removal leaves target memory data that contaminates a new prompt, a full target-memory data clear before request launch will make requests 2-5 match the fresh first-token oracle in both MTP and DFlash.
```

- [ ] **Step 3: Verify request-entry ordering, then add the minimal temporary diagnostic**

First verify from source that `launch_slot_with_task()` runs while the selected slot is idle, before `slot.task` is assigned and before any new-request prompt token is added to a decode batch. Record the exact source lines showing that `slot.prompt.tokens` is not populated from the incoming request at this point and that the new prompt remains in `task.tokens`. The current source names `slot.ctx_tgt` and `slot.prompt.tokens`; verify those exact members before editing. If ordering or member names differ, do not insert the diagnostic and stop at the milestone with the mismatch.

Insert this block as the first operation for the speculative-slot diagnostic inside `launch_slot_with_task()`, before the existing speculative reset and sampler initialization. BASE remains untouched because the block is guarded by `slot.can_speculate()`:

```cpp
if (slot.can_speculate()) {
    if (const char * env = std::getenv("GGML_SPEC_TEST_FULL_TARGET_CLEAR"); env && std::atoi(env) != 0) {
        GGML_ASSERT(slots.size() == 1);
        GGML_ASSERT(!slot.is_processing());
        llama_memory_clear(llama_get_memory(slot.ctx_tgt), true);
        slot.prompt.tokens.clear();
    }
}
```

The placement is deliberately before new-request prefill. `slot.prompt.tokens.clear()` removes only prior-request bookkeeping; it must not be moved after `slot.task` assignment or prompt batching. If source inspection contradicts this precondition, do not insert the block; return to the milestone with the mismatch.

This is diagnostic-only. Do not add explanatory production comments or combine another change.

- [ ] **Step 4: Build and run paired controls**

For MTP and DFlash coding, run the same 5-request, 512-token `np=1` cell twice:

```text
control: GGML_SPEC_TEST_FULL_TARGET_CLEAR unset
experiment: GGML_SPEC_TEST_FULL_TARGET_CLEAR=1
```

Use unique output paths, ports, and logs. Reuse the same built binary without rebuilding between control and experiment. Require identical binary hash; only the environment variable differs.

- [ ] **Step 5: Compare exact outcomes**

Report:

- content-valid count;
- first token per repetition;
- first divergence from the fresh BASE oracle;
- token hashes;
- draft counts;
- forbidden log counts.

Decision:

```text
experiment fixes token-0 divergence in both modes -> target memory data/init supported
experiment changes neither mode -> reject target memory data persistence
only DFlash changes -> DFlash-specific target memory/rollback lead, not shared cause
only MTP changes -> MTP context/state lead
```

- [ ] **Step 6: Preserve the diagnostic patch, then restore source exactly**

```bash
git -C "$WORK" diff -- tools/server/server-context.cpp \
  > "$EVIDENCE/hypotheses/task2-full-clear-diagnostic.patch"
test -s "$EVIDENCE/hypotheses/task2-full-clear-diagnostic.patch"
cp -a "$EVIDENCE/snapshots/task2-full-clear/tools/server/server-context.cpp" \
  "$WORK/tools/server/server-context.cpp"
sha256sum -c "$EVIDENCE/snapshots/task2-full-clear/SHA256SUMS.before"
cmake --build "$WORK/build-vulkan" --target llama-server -j"$(nproc)"
```

- [ ] **Step 7: Run Milestone M2 protocol**

Next task: `Task 3: DFlash rollback and re-evaluation diagnosis`.

---

### Task 3: DFlash Rollback and Re-evaluation Diagnosis

**Files:**
- Analyze: `tools/server/server-context.cpp:4705-4930`
- Analyze: `src/llama-context.cpp:4139-4210`
- Analyze: `src/llama-memory-recurrent.cpp:930-1165`
- Temporary diagnostics only if existing QA output is insufficient.

**Interfaces:**
- Produces exact restored position and re-evaluation position for every partial acceptance.

- [ ] **Step 1: Preserve a fresh-request QA reproduction**

Run one fresh DFlash coding request with:

```bash
export GGML_DFLASH_QA_TRACE=1
export GGML_DFLASH_TOKEN_TRACE=1
export GGML_DFLASH_SAMPLE_TRACE=1
```

Use 512 tokens and save raw response and server log. Confirm duplicate-position warnings occur before any request reuse.

- [ ] **Step 2: Parse rollback chronology**

For every rollback cycle, emit one row:

```text
verify_pre n_before pos_next id_last n_draft
verify_post accepted n_rollback
rollback n_hidden_keep n_reeval
warning last_pos previous_pos n_new
```

Require the parser to preserve log order. Do not infer chronology from aggregate counts.

- [ ] **Step 3: Add position trace only if needed**

Snapshot touched files. Under `GGML_DFLASH_QA_TRACE`, record:

```text
backup_pos
active_pos_before_restore
active_pos_after_restore
n_past_before
n_accepted
n_reeval
first_reeval_pos
last_reeval_pos
```

Use `llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id)` at each boundary. Restore temporary diagnostics after capture.

- [ ] **Step 4: Verify and test existing variables one at a time**

First require source matches for both variables:

```bash
git -C "$WORK" grep -n 'GGML_DFLASH_FORCE_CKPT_ROLLBACK\|GGML_DFLASH_FORCE_REDECODE' -- \
  tools/server/server-context.cpp src/llama-context.cpp
```

If a variable is absent, record `variable absent` and skip that condition; absence is not diagnostic evidence.

Run one fresh 512-token DFlash coding request for each available independent condition:

```text
control: all diagnostic variables unset
GGML_DFLASH_FORCE_CKPT_ROLLBACK=1
GGML_DFLASH_FORCE_REDECODE=1
```

Never stack variables. For each condition compare:

- duplicate-position warning count;
- first warning cycle;
- first token divergence from fresh BASE;
- full token hash and content validity;
- draft count.

- [ ] **Step 5: Decide B1**

B1 is confirmed only if restored position and first re-evaluation position are equal when one new token is submitted, and a single diagnostic variable moves or removes that boundary as predicted.

If B1 is not confirmed, add one narrower observation. Do not write a fix.

- [ ] **Step 6: Restore and verify diagnostics**

Restore every touched file from snapshots, verify SHA-256, rebuild, and save diagnostic patches under `$EVIDENCE/hypotheses/`.

- [ ] **Step 7: Run Milestone M3 protocol**

Next task: `Task 4: Bounded fresh-sequence diagnostic`.

---

### Task 4: Bounded Fresh-Sequence Diagnostic

**Files:**
- Use: `bench/vulkan-gap/slot_isolation.py`
- No production changes.

**Interfaces:**
- Uses physical slots as bounded fresh target sequence identities without inventing a sequence allocator.

- [ ] **Step 1: Run `np=4` first-use and reuse schedule**

For MTP and DFlash coding:

```bash
python3 bench/vulkan-gap/slot_isolation.py "$MODE" \
  --np 4 --rounds 2 --prompt coding --gen-tokens 512 --port "$PORT" \
  --output "$EVIDENCE/records/slot-isolation/${MODE}-coding-np4.jsonl"
```

Schedule must be `0,1,2,3,0,1,2,3`.

- [ ] **Step 2: Compare first-use IDs with reused IDs**

Report validity, first target token, target argmax, first divergence, and drafts for each slot.

Decision:

```text
all first-use IDs pass and all reused IDs fail -> bounded sequence/slot isolation supported
later first-use IDs fail before reuse -> process-global state; Path C rejected
mixed behavior -> correlate by physical slot and request ordinal; no production conclusion
```

- [ ] **Step 3: Assess production feasibility without implementing**

Write `$EVIDENCE/hypotheses/path-c-feasibility.md` answering with source references:

- sequence capacity and `seq_id >= size` behavior;
- every place target identity is assumed equal to `slot.id`;
- DFlash backup ID collision rules;
- prompt checkpoint and sampler identity;
- DFlash physical hidden/ring slot routing;
- bounded reuse cleanup requirement;
- concurrency impact.

- [ ] **Step 4: Apply the Path C gate**

Path C remains rejected unless all five requirements in the design document are supported. A fresh physical slot passing is necessary but not sufficient.

- [ ] **Step 5: Run Milestone M4 protocol**

Next task: `Task 5: Root-cause decision gate`.

---

### Task 5: Root-Cause Decision Gate

**Files:**
- Create: `$EVIDENCE/hypotheses/decision.md`
- Do not modify production code.

**Interfaces:**
- Produces one confirmed shared-entry hypothesis and, independently, one confirmed or rejected DFlash rollback hypothesis.

- [ ] **Step 1: Build the evidence matrix**

Include rows for:

```text
np2 unused slot
np2 reused slot
full target-memory clear
DFlash checkpoint rollback
DFlash forced re-decode
np4 first-use IDs
np4 reused IDs
```

Columns:

```text
mode, request, slot, first token, target argmax, first divergence, valid, draft_n, warning count, binary hash
```

- [ ] **Step 2: Select exactly one shared-entry hypothesis**

Use these rules:

```text
unused slots fail -> process-global/shared target output or speculative state
unused slots pass, reused slots fail, full clear fixes -> target memory/initialization
unused slots pass, reused slots fail, full clear does not fix -> non-memory slot-local state
wrong target argmax at prompt completion -> investigate target prefill/output row before drafting
correct target argmax but wrong returned token -> sampler/response path
```

- [ ] **Step 3: Select DFlash rollback disposition separately**

Do not merge Track B into Track A. State whether duplicate re-evaluation is confirmed causal, confirmed separate, or unresolved.

- [ ] **Step 4: Reject unsupported production options**

Explicitly reject:

- fresh sequence IDs if Path C gate is incomplete;
- recurrent-core changes if full-clear and metadata evidence do not support them;
- accepting the partial fix while any correctness gate fails.

- [ ] **Step 5: State the exact RED production boundary**

Write:

```text
I think <state transition> is the shared root cause because <single experiment> changed first divergence from <old> to <new>, while <control> did not.
Failing command: <exact command>
Expected RED: <exact first token or validator failure>
Source boundary: <file:function>
Minimal fix surface: <one or two files>
```

Write the same separately for DFlash rollback if confirmed.

- [ ] **Step 6: Run Milestone M5 protocol**

GLM must challenge whether each root cause is causal rather than correlated. Do not proceed if evidence does not answer a concrete contradiction.

Next task: `Task 6: Shared-entry RED/GREEN fix`.

---

### Task 6: Shared-Entry RED/GREEN Fix

**Files:**
- Modify only the minimal source boundary selected in Task 5.
- Modify nearest focused test.
- Treat `d0e3ca514` prompt-clear hunk as removable experimental state.

**Interfaces:**
- Produces a shared MTP/DFlash request-entry fix that removes token-0 divergence without relying on fresh sequence allocation.

- [ ] **Step 1: Isolate the experimental d0 hunk**

Snapshot `server-context.cpp`, record SHA-256, and save the hunk's exact diff. If Task 5 does not confirm it as required, remove only the five-line production hunk and its comment, preserving harness/docs. Treat removal like any production attempt: rebuild, rerun the exact RED to verify the same boundary remains RED, and restore the snapshot if the boundary changes unexpectedly.

- [ ] **Step 2: Snapshot proposed fix files**

Preserve paths and SHA-256 under `$EVIDENCE/snapshots/task6-attempt-1/`.

- [ ] **Step 3: Add focused RED test**

Use the nearest model-free test if possible, but the model-backed 512-token reproduction is mandatory. For output-row mapping, use `test-server-prompt-checkpoint` or `test-sampling`. For speculative reset state, use `test-speculative` and the explicit slot driver.

- [ ] **Step 4: Implement one minimal fix**

One hypothesis, one production boundary, no cleanup refactor, no performance code, no Path C allocator unless Task 5 explicitly passed every Path C gate.

- [ ] **Step 5: Run focused GREEN**

Require:

- exact RED command exits 0;
- MTP and DFlash wrong first-token boundary is removed;
- fresh BASE remains valid;
- first-use and reused slots behave identically;
- no zero-draft speculative rows;
- no new forbidden logs.

- [ ] **Step 6: Reject or retain**

If the first divergence merely moves later, reject. Save patch, restore exact snapshot, verify hash, and return to Task 5 with one new observation. After three failed fixes, stop and request architecture review before attempt four.

- [ ] **Step 7: Preserve accepted shared fix**

Save patch and file list under `$EVIDENCE/hypotheses/shared-entry-accepted-*`. Run `git diff --check`.

- [ ] **Step 8: Run Milestone M6 protocol**

Next task: `Task 7: DFlash rollback RED/GREEN fix` if Track B remains failing; otherwise Task 8.

---

### Task 7: DFlash Rollback RED/GREEN Fix

**Files:**
- Modify only the minimal Track B boundary from Task 5, likely `tools/server/server-context.cpp` or `src/llama-context.cpp`.
- Test: nearest DFlash decode/recurrent test plus model-backed warning reproduction.

**Interfaces:**
- Produces no duplicate-position re-evaluation and preserves exact accepted-token semantics.

- [ ] **Step 1: Write automated RED parser**

The script exits nonzero when any DFlash warning satisfies:

```text
last_pos == previous_pos and n_new > 0
```

It must run a fresh 512-token request and preserve response and log.

- [ ] **Step 2: Add focused test before source change**

Extend `test-dflash-decode` or the nearest recurrent-memory test to represent restored-position plus re-evaluation position semantics. Confirm RED for the expected off-by-one/count behavior.

- [ ] **Step 3: Implement one minimal correction**

Correct exactly one of backup position, replay count, or re-evaluation start/count according to Task 3 evidence. Do not change all three.

- [ ] **Step 4: Verify GREEN**

Require:

- zero duplicate-position warnings;
- DFlash raw token equality to fresh BASE for fresh deterministic coding;
- 5/5 persistent coding and math validity;
- nonzero drafts;
- no regression in MTP or BASE.

- [ ] **Step 5: Reject or retain**

Use the same restoration and three-attempt architecture-review rule as Task 6.

- [ ] **Step 6: Run Milestone M7 protocol**

Next task: `Task 8: Full correctness closure`.

---

### Task 8: Full Correctness Closure

**Files:**
- Create final records under `$EVIDENCE/final/`
- Update results document only after all gates pass.

**Interfaces:**
- Re-runs the complete original Tasks 5-8 from a fresh Release build.

- [ ] **Step 1: Fresh build**

```bash
rm -rf "$WORK/build-vulkan-correctness-final"
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
  cmake -S "$WORK" -B "$WORK/build-vulkan-correctness-final" \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON
cmake --build "$WORK/build-vulkan-correctness-final" -j"$(nproc)"
```

- [ ] **Step 2: Model-free and CTest gates**

```bash
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir "$WORK/build-vulkan-correctness-final" --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git -C "$WORK" diff --check
```

- [ ] **Step 3: Single-slot matrices**

Run 5 persistent 512-token rows for BASE, MTP, and DFlash, coding and math, under:

```text
smoke:  b=512,  ub=128, ctx=2048
matrix: b=2048, ub=512, ctx=8192
```

The count is exactly `3 modes x 2 prompts x 2 configurations x 5 rows = 60`. Require 60/60 valid rows and nonzero drafts for all speculative rows.

- [ ] **Step 4: Fresh token equivalence**

For each configuration, use BASE fresh coding repetition 1 as the sole oracle. Require fresh MTP and DFlash coding `token_ids` to equal it exactly.

- [ ] **Step 5: Persistent and concurrent isolation**

For both configurations and speculative modes:

- 5 persistent coding requests;
- 5 persistent math requests;
- three `np=2` concurrent rounds, each submitting coding and math together.

Require no cross-slot content and nonzero drafts.

- [ ] **Step 6: Forbidden log scan**

Require zero:

```text
GGML_ASSERT
invalid logits
corrupt output buffer
decode error
non-consecutive token position
segmentation fault
abort
unexpected server exit
DFlash sequence/ring/hidden routing warning
```

- [ ] **Step 7: Final diff audit**

Every production hunk must have RED/GREEN evidence. Temporary diagnostics and unsupported Path C/recurrent experiments must be absent. The starting d0 prompt-clear hunk remains only if Task 5 and Task 6 evidence independently justified it.

- [ ] **Step 8: Update final result document**

Only after all gates pass, update:

```text
docs/superpowers/results/2026-07-31-vulkan-speculative-working-tree-correctness-recovery.md
```

Include exact hashes, commands, valid counts, token equivalence, persistent/concurrent results, rejected hypotheses, and remaining risks. Do not make performance claims.

- [ ] **Step 9: Run Milestone M8 protocol and stop**

The continuation summary may state `CORRECTNESS GOAL ACHIEVED` only when every gate passes. Otherwise return to Task 5 with the exact first failing gate.

## Plan Self-Review Checklist

- [x] Explicit slot isolation precedes memory or sequence-ID changes.
- [x] Full data clear is diagnostic and restored before continuation.
- [x] DFlash rollback is independent from MTP request-entry corruption.
- [x] Fresh sequence identities are tested before any allocator design.
- [x] Recurrent core changes require direct postcondition evidence.
- [x] The partial fix cannot be accepted while long tests fail.
- [x] Every production fix requires exact 512-token RED/GREEN evidence.
- [x] Every milestone includes GLM review and an authoritative compaction summary.
- [x] Full smoke, matrix, persistent, concurrent, and fresh-build gates remain mandatory.
