# Vulkan Speculative Working-Tree Correctness Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task-by-task. Do not use subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2` correct for BASE, MTP, and DFlash generation on Vulkan across fresh, persistent, and concurrent requests.

**Architecture:** Treat the actual dirty worktree as `Work`, preserve its starting state, repair the benchmark harness, classify the failure by request boundary and exact token divergence, then run a one-hypothesis TDD loop until the minimal production fix passes every correctness gate. Remove the unrelated reduced-verification/profiling experiment before final verification. Performance work is out of scope until this plan is fully green.

**Tech Stack:** C++17, CMake, Vulkan, Python 3 `unittest`, CTest, llama-server HTTP completion API, Qwen3.6-27B target GGUF, DFlash draft GGUF, context-mode tools, GLM-5.2 through Ollama, and `smart_compact`.

## Global Constraints

- Work only in `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2` unless a step names an evidence path outside Git.
- Starting HEAD must be `fb7e5d080f257bb8d15ee68b850cfafe593a7899`. If it changed, record the new SHA and revalidate every starting assumption before editing.
- `Work` includes committed HEAD plus the starting uncommitted `tools/server/server-context.cpp` patch.
- Never run `git stash`, `git reset`, `git clean`, `git checkout --`, or a whole-tree `git restore`.
- Never apply an old stash wholesale.
- Never commit, push, or create a PR. A later human may authorize a commit separately.
- Preserve the starting dirty patch before any edit. Restore rejected experiments from exact file snapshots, not from Git.
- Keep `GGML_DFLASH_REDUCED_VERIFY` unset for correctness unless a step explicitly tests that one variable.
- Do not begin performance optimization. Throughput is used only as a sanity field: finite, positive, and below 10,000 t/s.
- Use one hypothesis and one production change at a time. Revert a failed experiment before trying the next.
- Production code may change only after an automated reproduction is RED for the expected reason.
- Do not trust a stored `measurement.valid` bit without reapplying the current validators.
- Do not pipe a build or test through `tail`, `head`, or `grep`; doing so can hide the real exit code. Use context-mode to summarize complete output.
- Use ASCII only in source, comments, plans, results, and handoffs.
- Use context-mode for builds, tests, logs, Git history, JSONL, profiler output, and any output that may exceed 20 lines.
- Use `ctx_execute_file` to analyze a file. Use `read` only for the narrow exact region needed for an edit, then use `edit` for precise replacements.
- Continue automatically from one task to the next. Do not ask the human to say `go` at milestone boundaries.
- Stop only for an external blocker that prevents execution: missing model file, missing Vulkan device, filesystem failure, or inability to restore a rejected experiment. A failing correctness test is not a blocker; it enters the root-cause loop.

## End Goal and Required Gates

The worktree is correct only when all of the following pass from a fresh Release build:

1. Harness model-free tests pass.
2. Selected CTests pass.
3. Fresh deterministic BASE, MTP, and DFlash coding requests return raw token IDs.
4. Fresh MTP and DFlash coding token IDs equal fresh BASE coding token IDs with temperature 0, top-k 20, and seed 7.
5. Coding persistent matrix: 5/5 rows per mode have `finish_reason=stop`, compilable Python, `def fibonacci`, no prompt echo, and no empty content.
6. Math persistent matrix: 5/5 rows per mode contain `2.4`, contain no coding cross-contamination, and have no prompt echo.
7. Every MTP and DFlash row has `draft_n > 0`.
8. Five persistent requests pass without server restart for both prompts and both speculative modes.
9. Three concurrent `-np 2` rounds pass for MTP and DFlash. Each round runs coding and math simultaneously and shows no cross-slot content.
10. All correctness gates pass under both server configurations:
    - smoke: `-b 512 -ub 128 --ctx-size 2048`;
    - matrix: `-b 2048 -ub 512 --ctx-size 8192`.
11. Server logs contain none of:
    - `GGML_ASSERT`;
    - `invalid logits`;
    - `corrupt output buffer`;
    - `decode error`;
    - `non-consecutive token position`;
    - segmentation, abort, or unexpected server exit;
    - DFlash sequence, ring, or hidden-capture routing warnings.
12. The final source diff contains the accepted correctness fix and related tests/harness work only. The starting reduced-verification and cycle-profiling experiment is removed from the final candidate and remains preserved in evidence.

## Known Starting Evidence

- Existing quarantine: `/crypt/tmp/beellama-task8-quarantine-20260731T064446Z/`
- Starting source diff SHA-256 at the last check: `13b2b9298b40d28b7407aeb5372a9d521f03da9a2b9dc8c81962ca1d74ddb1cc`
- Dirty working binary tested at SHA-256: `bd904891a6854498e788a8872e1016b0f974510594676de0b048c21e6e5a6416`
- Working-tree records:
  `/crypt/tmp/beellama-task8-quarantine-20260731T064446Z/correctness/working-tree/`
- Test-only harness patch from the previous session:
  `/crypt/tmp/beellama-task8-quarantine-20260731T064446Z/harness-test-only-fixes.patch`
- That patch is evidence only. Do not apply it blindly; implement each harness fix test-first.

## Tool Routing

Use these tools consistently:

- Repository status or one guaranteed-short command: `bash` is acceptable.
- Build/test/large Git output/log processing: `ctx_execute`.
- JSONL or log analysis: `ctx_execute_file` or a Python program run through `ctx_execute`.
- Source analysis without editing: `ctx_execute_file`.
- Exact source inspection for editing: `read` with a narrow offset/limit.
- Source mutation: `edit` with the smallest unique exact replacement.
- New evidence or helper file: `write`.
- Independent review: `glm-ollama-reviewer` helper with `--model glm-5.2:cloud`.
- Context reduction: GLM writes the authoritative continuation summary; then use `smart_compact` when actual context usage is high.

## Milestone Review and GLM-Guided Compaction Protocol

Run this protocol after every numbered task. Replace `MILESTONE` and `NEXT_TASK` with exact values.

1. Write `$EVIDENCE/milestones/MILESTONE-agent.md` with these headings:

```text
# Milestone MILESTONE

## Candidate
HEAD, source status, source diff hash, binary hash.

## Commands and exits
Every build/test/reproduction command and its exit code.

## Evidence
Record paths, valid counts, first divergence, and log findings.

## Accepted changes
Exact files and why each change remains.

## Rejected changes
Hypothesis, observed result, and proof of restoration.

## Open risks
Only facts not yet resolved.

## Next task
NEXT_TASK and its first command.
```

2. Write `$EVIDENCE/reviews/MILESTONE.prompt.txt`:

```text
You are GLM-5.2 reviewing a Vulkan speculative-decoding correctness recovery milestone.

Goal: BASE, MTP, and DFlash must be correct for fresh, persistent, and concurrent Qwen3.6-27B requests.
Repository: /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
Constraints: no performance work, no stash/reset/clean, one hypothesis at a time, no commit or push.

Read the milestone evidence pasted below. Check whether conclusions follow from evidence, identify missed correctness risks, and decide whether the next task is justified. Do not invent facts.

Return exactly these headings:
## Decision
PROCEED or REVISIT, with one sentence.
## Verified facts
## Risks or contradictions
## Required correction before proceeding
Use `None` if no correction is required.
## Continuation summary for compaction
Include candidate SHA/status, accepted changes, rejected hypotheses, test evidence, open loop, and exact next command.

MILESTONE EVIDENCE:
```

Append the complete milestone report to the prompt.

3. Run GLM through context-mode, saving stdout without flooding context:

```bash
python3 /home/gbo/.pi/agent/skills/glm-ollama-reviewer/scripts/ollama_second_opinion.py \
  --host http://192.168.123.123:11434 \
  --model glm-5.2:cloud \
  --timeout 600 \
  --prompt-file "$EVIDENCE/reviews/MILESTONE.prompt.txt" \
  > "$EVIDENCE/reviews/MILESTONE-glm.md"
```

4. Verify the GLM file is nonempty and contains `## Decision` and `## Continuation summary for compaction`. If Ollama is unavailable, record the error in the milestone and continue using the agent report.

5. Load only the GLM decision and continuation summary into context with `ctx_execute_file`.

6. If GLM says `REVISIT`, verify its concern against repository evidence. Correct a factual omission before continuing. Do not accept speculative advice without verification.

7. The GLM continuation summary is the authoritative compact guide. When actual context usage is high, call:

```text
smart_compact(
  profile="balanced",
  focus="Preserve the GLM continuation summary in <absolute MILESTONE-glm.md path> and the exact next command",
  max_calls=8,
  max_latency_ms=120000
)
```

`smart_compact` does not expose a model selector. GLM-5.2 therefore creates the compact guide, and `smart_compact` applies context reduction around that guide. If actual context usage is not high, keep the guide and continue without forcing compaction.

---

### Task 0: Freeze the Actual Working Candidate and Initialize Evidence

**Files:**
- Read only: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/`
- Create outside Git: `/crypt/tmp/beellama-working-correctness-<timestamp>/`
- Create outside Git: `/tmp/beellama-working-correctness-state.env`
- Preserve: current source diff, untracked docs list, stash inventory, binary/model hashes, and exact source snapshots

**Interfaces:**
- Produces `EVIDENCE`, `WORK`, `HEAD_SHA`, model paths, and immutable pre-recovery snapshots used by all later tasks.

- [ ] **Step 1: Read required instructions completely**

Read:

```text
/crypt/beellama.cpp/AGENTS.md
/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/docs/superpowers/specs/2026-07-31-vulkan-speculative-working-tree-correctness-design.md
/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/docs/superpowers/plans/2026-07-31-vulkan-speculative-working-tree-correctness-recovery.md
```

Invoke `superpowers:executing-plans`, `superpowers:systematic-debugging`, and `context-mode`. Do not invoke subagent-driven development.

- [ ] **Step 2: Initialize state**

```bash
WORK=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EVIDENCE=/crypt/tmp/beellama-working-correctness-$STAMP
MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
mkdir -p "$EVIDENCE"/{source-start,snapshots,records,logs,milestones,reviews,hypotheses,scripts,build}
HEAD_SHA=$(git -C "$WORK" rev-parse HEAD)
cat > /tmp/beellama-working-correctness-state.env <<EOF
export WORK=$WORK
export EVIDENCE=$EVIDENCE
export MODEL=$MODEL
export DRAFT=$DRAFT
export HEAD_SHA=$HEAD_SHA
EOF
```

Every later shell command begins with:

```bash
source /tmp/beellama-working-correctness-state.env
```

- [ ] **Step 3: Verify the starting candidate**

Run through `ctx_execute` and record output in `$EVIDENCE/source-start/repository.txt`:

```bash
git -C "$WORK" status --short --branch
git -C "$WORK" show -s --format='%H %D%n%s' HEAD
git -C "$WORK" rev-list --left-right --count gboddaer/main...HEAD
git -C "$WORK" stash list --date=iso
git -C "$WORK" diff --check
```

Expected:

- HEAD is `fb7e5d080f257bb8d15ee68b850cfafe593a7899` unless a human moved it after plan creation;
- `tools/server/server-context.cpp` is modified;
- the two Task 4 handoffs and old recovery plan are untracked;
- `git diff --check` exits 0.

If HEAD changed, do not guess. Re-run the stash/current-code comparison from the design evidence and update `$EVIDENCE/source-start/repository.txt` before continuing.

- [ ] **Step 4: Preserve exact starting source**

```bash
git -C "$WORK" diff --binary > "$EVIDENCE/source-start/original-working.patch"
git -C "$WORK" status --short --branch > "$EVIDENCE/source-start/status.txt"
cp -a "$WORK/tools/server/server-context.cpp" "$EVIDENCE/source-start/server-context.cpp"
cp -a "$WORK/src/llama-context.cpp" "$EVIDENCE/source-start/llama-context.cpp"
cp -a "$WORK/common/speculative.cpp" "$EVIDENCE/source-start/speculative.cpp"
cp -a "$WORK/common/sampling.cpp" "$EVIDENCE/source-start/sampling.cpp"
cp -a "$WORK/bench/vulkan-gap/run.py" "$EVIDENCE/source-start/run.py"
cp -a "$WORK/bench/vulkan-gap/repeated_requests.py" "$EVIDENCE/source-start/repeated_requests.py"
find "$EVIDENCE/source-start" -type f -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/source-start/SHA256SUMS"
```

Verify `original-working.patch` is nonempty.

- [ ] **Step 5: Record machine, model, and binary provenance**

Configure the build directory if needed, then verify its Vulkan setting before building:

```bash
if [ ! -f "$WORK/build-vulkan/CMakeCache.txt" ]; then
  env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    cmake -S "$WORK" -B "$WORK/build-vulkan" \
      -DGGML_VULKAN=ON \
      -DGGML_NATIVE=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON
fi
grep '^GGML_VULKAN:BOOL=ON$' "$WORK/build-vulkan/CMakeCache.txt"
cmake --build "$WORK/build-vulkan" --target llama-server -j"$(nproc)"
```

Use context-mode and require every command to exit 0. Then record:

```bash
{
  date -u --iso-8601=seconds
  uname -a
  vulkaninfo --summary
  sha256sum "$MODEL" "$DRAFT"
  sha256sum "$WORK/build-vulkan/bin/llama-server"
  "$WORK/build-vulkan/bin/llama-server" --version
  git -C "$WORK" diff --binary | sha256sum
  ps -eo pid,comm,args | grep -E 'llama-(server|bench)' || true
} > "$EVIDENCE/source-start/machine-model-binary.txt" 2>&1
```

No unrelated llama server may remain on the Vulkan device.

- [ ] **Step 6: Initialize the hypothesis ledger**

Create `$EVIDENCE/hypotheses/ledger.tsv` with exactly this header:

```text
id	category	hypothesis	prediction	command	result	decision	restored_sha256
```

- [ ] **Step 7: Run Milestone M0 protocol**

Next task: `Task 1: Repair and prove the harness`. Continue automatically.

---

### Task 1: Repair and Prove the Harness Before Model-Backed Diagnosis

**Files:**
- Modify: `bench/vulkan-gap/run.py`
- Modify: `bench/vulkan-gap/repeated_requests.py`
- Modify: `bench/vulkan-gap/test_harness_strict.py`
- Modify: `bench/vulkan-gap/test_repeated_requests.py`

**Interfaces:**
- `extract_measurement(response, prompt_kind, mode)` validates prompt content and speculative draft use separately.
- `build_server_command(...) -> list[str]` is the single command builder used by `run.py`.
- Both runners consume `VK_GAP_BATCH`, `VK_GAP_UBATCH`, and `VK_GAP_CTX_SIZE`.
- Both request raw token IDs with `return_tokens=true` and record them.
- Output JSONL creation is exclusive: an existing output path is an error, never an append target.

- [ ] **Step 1: Add RED tests for mode-aware validation**

Change existing speculative tests to call:

```python
got = MOD.extract_measurement(response, "coding", "dflash")
```

and:

```python
got = MOD.extract_measurement(response, "coding", "mtp")
```

Add a BASE control:

```python
got = MOD.extract_measurement(response, "coding", "base")
self.assertNotIn("spec_no_drafts", got["invalid_reasons"])
```

Run:

```bash
python3 -m unittest bench.vulkan-gap.test_harness_strict -v
```

Expected RED: the current function does not accept the separate mode argument.

- [ ] **Step 2: Add RED tests for unclosed fences**

Add:

```python
def test_unclosed_python_fence_is_invalid_not_exception(self):
    response = {
        "choices": [{"text": "```python\ndef fibonacci(n):\n    if n < 2:\n        return n", "finish_reason": "length"}],
        "usage": {"completion_tokens": 20},
        "timings": {"predicted_per_second": 20.0, "draft_n": 8},
        "tokens": [1, 2, 3],
    }
    got = MOD.extract_measurement(response, "coding", "dflash")
    self.assertFalse(got["valid"])
    self.assertIn("coding_not_stopped", got["invalid_reasons"])
```

Expected RED: current `_extract_python_code()` raises `ValueError`.

- [ ] **Step 3: Add RED tests for command construction**

Add a test for the planned helper:

```python
def test_server_command_has_no_removed_z_argument(self):
    got = MOD.build_server_command(
        server_path="/tmp/llama-server",
        model_path="/tmp/target.gguf",
        draft_path="/tmp/draft.gguf",
        port=8099,
        n_parallel=1,
        device_id="Vulkan0",
        mode="dflash",
        batch_size=512,
        ubatch_size=128,
        ctx_size=2048,
    )
    self.assertNotIn("--z", got)
    self.assertEqual(got[got.index("-b") + 1], "512")
    self.assertEqual(got[got.index("-ub") + 1], "128")
    self.assertEqual(got[got.index("--ctx-size") + 1], "2048")
```

Expected RED: `build_server_command` does not exist.

- [ ] **Step 4: Add RED tests for external HEAD and token recording**

Add pure helper tests:

```python
def test_external_head_is_the_preflight_expected_head(self):
    self.assertEqual(MOD.expected_source_head("abc123", "def456"), "abc123")

def test_worktree_head_is_used_without_external_override(self):
    self.assertEqual(MOD.expected_source_head(None, "def456"), "def456")
```

Add token assertion to a valid response test:

```python
response["tokens"] = [10, 20, 30]
got = MOD.extract_measurement(response, "coding", "base")
self.assertEqual(got["token_ids"], [10, 20, 30])
```

Expected RED: helper and `token_ids` do not exist.

- [ ] **Step 5: Add RED tests for exclusive output creation**

Define the desired helper through a temporary directory:

```python
def test_existing_output_is_rejected(self):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rows.jsonl"
        path.write_text("old\n")
        with self.assertRaises(FileExistsError):
            with MOD.open_output_exclusive(path):
                pass
```

Expected RED: helper does not exist.

- [ ] **Step 6: Inventory and lock the CLI surface used by later tasks**

Run both help commands and save them under `$EVIDENCE/source-start/`:

```bash
python3 bench/vulkan-gap/run.py --help > "$EVIDENCE/source-start/run-help.txt"
python3 bench/vulkan-gap/repeated_requests.py --help > "$EVIDENCE/source-start/repeated-help.txt"
```

Verify the `run.py` usage line contains positional mode and prompt choices in this order:

```text
{base,mtp,dflash} {coding,math}
```

Verify `run.py` also accepts these existing options:

```text
--repetitions --gen-tokens --output --warmup-tokens
--restart-between-reps --skip-warmup
--reference-label --external-source-head
```

Verify `repeated_requests.py` accepts positional mode plus:

```text
--np --rounds --prompt --gen-tokens --concurrent
```

Task 1 adds `--output` to `repeated_requests.py`; add a RED argparse test for it before implementation. Do not add `--mode`, because mode is intentionally positional. After implementation, rerun both help commands and require every later-plan option to be present.

- [ ] **Step 7: Implement the minimal harness APIs**

Implement:

```python
def extract_measurement(response, prompt_kind="coding", mode="base"):
```

Use `mode` only for the draft requirement:

```python
if mode in ("mtp", "dflash") and draft_n == 0:
    invalid_reasons.append("spec_no_drafts")
```

Make fenced extraction use `find` and return the remaining text when the closing fence is absent. It must never raise for model output.

Add:

```python
def expected_source_head(external_source_head, worktree_head):
    return external_source_head or worktree_head


def open_output_exclusive(path):
    return open(path, "x", encoding="utf-8")
```

Extract the existing embedded command list into `build_server_command(...)`. Remove `--z`. Read batch settings in `main()`:

```python
batch_size = int(os.environ.get("VK_GAP_BATCH", "512"))
ubatch_size = int(os.environ.get("VK_GAP_UBATCH", "128"))
ctx_size = int(os.environ.get("VK_GAP_CTX_SIZE", "2048"))
```

Pass these values to the helper and record them in provenance through the existing `VK_GAP_` environment capture.

Set request payload:

```python
"return_tokens": True,
```

Record:

```python
"token_ids": [int(token) for token in response.get("tokens", [])],
"tokens_sha256": hashlib.sha256(
    json.dumps(response.get("tokens", []), separators=(",", ":")).encode("utf-8")
).hexdigest(),
```

Call:

```python
measurement = extract_measurement(response, args.prompt, args.mode)
```

Use `open_output_exclusive(output_path)` instead of append mode.

For binary preflight, compare the parsed binary commit to:

```python
expected_head = expected_source_head(args.external_source_head, worktree_head)
```

- [ ] **Step 8: Give `repeated_requests.py` the same configuration and evidence contract**

Use the same three environment settings, add `return_tokens=true`, and include `token_ids` in each result. Add `--output PATH`; when supplied, open it exclusively and write one JSON object per response. Add the exact launched command to the first provenance object.

Extract this pure helper and use it in `run_concurrent()`:

```python
def concurrent_request_specs(prompts, gen_tokens):
    return [
        ("coding", prompts["coding"], gen_tokens),
        ("math", prompts["math"], gen_tokens),
    ]
```

Under `--concurrent`, `--prompt` is intentionally ignored because the gate always submits one coding and one math request. Add a model-free test that passes `prompt="coding"` but asserts the helper still returns exactly one coding and one math request and that both use `args.gen_tokens`. This proves `--concurrent` is not a vacuous single-prompt test.

Do not change validation thresholds in this step.

- [ ] **Step 9: Run model-free GREEN tests**

```bash
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
```

Expected: all tests exit 0. Record exact count. Then rerun both `--help` commands and assert the complete option inventory from Step 6, including the newly added `repeated_requests.py --output`, is present.

- [ ] **Step 10: Prove command lines against both server binaries**

Use the working binary's `--list-devices` output and the existing `find_device()` helper to record the exact device ID selected. Verify `llama-server --help` contains `--device` and that the selected ID is passed unchanged by `build_server_command()`.

Then run a one-request startup preflight with `--skip-warmup --repetitions 1 --gen-tokens 8` for BASE only. The content may be invalid because the cap is 8; startup must succeed and the output must contain exactly one row. Capture the expected nonzero runner exit without treating it as a server-start failure. Confirm the server log does not contain `invalid argument`.

Do not require a second binary in this harness task; cross-revision benchmarking is outside this correctness plan.

- [ ] **Step 11: Run `git diff --check` and GLM review**

The GLM review must focus on false-valid risks, exit-code preservation, token recording, and stale output prevention. Correct any verified issue before M1.

- [ ] **Step 12: Run Milestone M1 protocol**

Next task: `Task 2: Classify fresh versus persistent corruption`. Continue automatically.

---

### Task 2: Classify Fresh Versus Persistent Corruption Without Production Changes

**Files:**
- Do not modify production C++.
- Create: `$EVIDENCE/records/classification/`
- Create: `$EVIDENCE/logs/classification/`
- Create: `$EVIDENCE/hypotheses/classification.md`

**Interfaces:**
- Produces a classification for each speculative mode: first-request, later-request, or no corruption.
- Produces exact raw token IDs and first divergence against BASE.

- [ ] **Step 1: Rebuild after harness-only changes**

```bash
cmake --build "$WORK/build-vulkan" --target llama-server -j"$(nproc)"
```

Record binary SHA-256. Production C++ binary may be unchanged; record the fact rather than assuming it.

- [ ] **Step 2: Set the common deterministic environment**

```bash
export VK_GAP_SERVER="$WORK/build-vulkan/bin/llama-server"
export VK_GAP_MODEL="$MODEL"
export VK_GAP_DRAFT="$DRAFT"
export VK_GAP_PORT=8099
export VK_GAP_TEMP=0
export VK_GAP_TOP_K=20
export VK_GAP_NP=1
export VK_GAP_BATCH=512
export VK_GAP_UBATCH=128
export VK_GAP_CTX_SIZE=2048
unset GGML_DFLASH_REDUCED_VERIFY
unset GGML_DFLASH_FORCE_CKPT_ROLLBACK
unset GGML_DFLASH_FORCE_REDECODE
unset GGML_DFLASH_FORCE_CPU_CROSS
unset GGML_DFLASH_DISABLE_KV_CACHE
unset GGML_DFLASH_GPU_RING
```

- [ ] **Step 3: Run fresh-server coding controls**

For each mode, remove only the named new output if it does not yet contain evidence. Never remove an old evidence file silently.

```bash
for MODE in base mtp dflash; do
  python3 bench/vulkan-gap/run.py "$MODE" coding \
    --repetitions 5 --gen-tokens 512 \
    --restart-between-reps --skip-warmup \
    --reference-label working-tree-fresh-smoke \
    --external-source-head "$HEAD_SHA" \
    --output "$EVIDENCE/records/classification/${MODE}-coding-fresh-smoke.jsonl"
done
```

These commands may exit nonzero while RED. Capture each real exit code and continue through all three modes.

- [ ] **Step 4: Run persistent coding controls**

```bash
for MODE in base mtp dflash; do
  python3 bench/vulkan-gap/run.py "$MODE" coding \
    --repetitions 5 --gen-tokens 512 --skip-warmup \
    --reference-label working-tree-persistent-smoke \
    --external-source-head "$HEAD_SHA" \
    --output "$EVIDENCE/records/classification/${MODE}-coding-persistent-smoke.jsonl"
done
```

- [ ] **Step 5: Run a warm-up state-transition control**

For MTP and DFlash, run one two-repetition persistent coding cell with the normal 32-token warm-up enabled:

```bash
for MODE in mtp dflash; do
  python3 bench/vulkan-gap/run.py "$MODE" coding \
    --repetitions 2 --gen-tokens 512 --warmup-tokens 32 \
    --reference-label working-tree-warmup-control \
    --external-source-head "$HEAD_SHA" \
    --output "$EVIDENCE/records/classification/${MODE}-coding-warmup-control.jsonl"
done
```

Compare this with the no-warm-up persistent result. If no-warm-up repetition 1 passes but the first measured row after warm-up fails, classify the warm-up completion as a prior request and follow the request-reset branch.

- [ ] **Step 6: Reapply validators and classify**

The classification BASE oracle is repetition 1 in:

```text
$EVIDENCE/records/classification/base-coding-fresh-smoke.jsonl
```

Write an evidence-side Python script that reads every classification JSONL and prints:

- row count;
- valid count;
- finish-reason counts;
- unique content hashes;
- unique token hashes;
- draft min/max;
- invalid reasons by repetition;
- first token divergence from BASE fresh repetition 1.

Classification rules:

```text
BASE fails fresh -> stop speculative diagnosis and investigate target/build/harness first.
Spec fresh fails at repetition 1 -> within-request corruption.
Spec fresh passes but persistent repetition 2 or later fails -> request-reset corruption.
MTP and DFlash fail at the same request boundary -> common speculative orchestration/sampling lead.
Only DFlash fails -> DFlash ring/cross/rollback lead.
```

Write the result to `$EVIDENCE/hypotheses/classification.md`.

- [ ] **Step 7: Check logs for hard errors**

Parse every server log referenced by the records. Count and print exact matching lines for the forbidden patterns in the End Goal. Do not dump full logs into context.

- [ ] **Step 8: Record the first source-level hypothesis**

Append exactly one row to `ledger.tsv`. The prediction must identify which request and token boundary will change under one diagnostic variable. Do not write a fix yet.

- [ ] **Step 9: Run Milestone M2 protocol**

Next task: `Task 3: Trace and isolate the first bad state transition`. Continue automatically.

---

### Task 3: Trace and Isolate the First Bad State Transition

**Files:**
- Analyze: `tools/server/server-context.cpp`
- Analyze: `common/speculative.cpp`
- Analyze: `common/sampling.cpp`
- Analyze: `src/llama-context.cpp`
- Create evidence only: `$EVIDENCE/records/traces/`, `$EVIDENCE/logs/traces/`, `$EVIDENCE/hypotheses/`
- Do not make a production fix in this task.

**Interfaces:**
- Produces one confirmed root cause boundary and a RED command suitable for Task 4.

- [ ] **Step 1: Freeze the classification BASE token oracle**

Use repetition 1 from:

```text
$EVIDENCE/records/classification/base-coding-fresh-smoke.jsonl
```

Copy that single row to:

```text
$EVIDENCE/records/traces/base-coding-oracle.jsonl
```

Do not generate a second oracle in this task. Verify the copied row is valid, has raw token IDs, and came from the same working binary, configuration, prompt, and sampling settings used for the speculative trace.

- [ ] **Step 2: Capture one traced request for each failing speculative mode**

Before running a trace, verify current source reads each requested environment variable:

```bash
git -C "$WORK" grep -n 'GGML_DFLASH_QA_TRACE\|GGML_DFLASH_TOKEN_TRACE\|GGML_DFLASH_SAMPLE_TRACE' -- \
  tools/server/server-context.cpp common/speculative.cpp common/sampling.cpp src/llama-context.cpp
```

If a variable is absent, record it and do not rely on that trace family. Use the temporary-diagnostic snapshot path for the missing observation instead.

For DFlash:

```bash
export GGML_DFLASH_QA_TRACE=1
export GGML_DFLASH_TOKEN_TRACE=1
export GGML_DFLASH_SAMPLE_TRACE=1
python3 bench/vulkan-gap/run.py dflash coding \
  --repetitions 1 --gen-tokens 512 --skip-warmup \
  --reference-label working-tree-dflash-trace \
  --external-source-head "$HEAD_SHA" \
  --output "$EVIDENCE/records/traces/dflash-coding-trace.jsonl"
unset GGML_DFLASH_QA_TRACE GGML_DFLASH_TOKEN_TRACE GGML_DFLASH_SAMPLE_TRACE
```

Run the equivalent MTP trace if MTP failed classification. Existing QA logging may be shared by the speculative sampler; record which trace families actually appear.

- [ ] **Step 3: Produce first-divergence reports**

For each failing mode, compare `measurement.token_ids` to BASE and write:

```text
mode=<mode>
base_tokens=<count>
spec_tokens=<count>
first_divergence_index=<zero-based index or NONE>
base_window=<8 token ids before and after>
spec_window=<8 token ids before and after>
content_at_divergence=<escaped short text window>
```

Save under `$EVIDENCE/hypotheses/<mode>-first-divergence.txt`.

- [ ] **Step 4: Align divergence with QA trace**

Extract, in order, only these lines from the DFlash log:

```text
[DFLASH_QA] verify_pre
[DFLASH_QA] sample_accept
[DFLASH_QA] verify_post
[DFLASH_QA] verify slot=
[DFLASH_QA] rollback
[DFLASH_QA] rollback_reeval
```

Identify whether the first wrong emitted token occurs:

- before any rollback;
- on the first partial acceptance;
- immediately after rollback or re-evaluation;
- only after a new request reset.

Do not infer from throughput or aggregate acceptance alone.

- [ ] **Step 5: Cross-check and record branch selection**

Write `$EVIDENCE/hypotheses/branch-selection.md`. The selected branch must follow this exact mapping:

```text
Task 2 says later-request or warm-up-dependent -> Branch A only.
Task 2 says DFlash first-request failure and Task 3 places divergence after rollback -> Branch B only.
Task 2 says DFlash first-request failure and Task 3 places divergence before rollback -> Branch C only.
Task 2 says MTP and DFlash first-request failure at the same acceptance/logits boundary -> Branch D only.
```

Include the classification record path and QA line that justify the selection. If evidence fits two branches, gather one more observation that distinguishes them; do not choose by intuition.

If Task 2 shows no corruption for BASE, MTP, and DFlash in both fresh and persistent classification runs, write `No corruption remains at classification gate` and skip directly to Task 5.

- [ ] **Step 6: Follow the selected decision-tree branch**

Use exactly one branch.

**Branch A: later request only**

Trace:

- `launch_slot_with_task()` in `tools/server/server-context.cpp`;
- draft-context sequence removal;
- `common_speculative_reset()`;
- MTP `pending_h`, batch bounds, and verify state;
- DFlash `discard_cross_ring()` and per-slot sequence identity.

Capture request number, slot id, `n_tokens`, sequence id, reset reason, and draft-context position before and after reset. Add temporary diagnostics only if existing logs cannot show these values. Before adding diagnostics, copy every touched file to `$EVIDENCE/snapshots/task3-diagnostics/` and record SHA-256. Temporary diagnostics must be environment-gated by `GGML_DFLASH_QA_TRACE`.

**Branch B: DFlash first request, first divergence after rollback**

Test existing diagnostic variables one at a time in this order:

1. `GGML_DFLASH_FORCE_CKPT_ROLLBACK=1`
2. `GGML_DFLASH_FORCE_REDECODE=1`

For each variable, run one fresh request. If it becomes token-identical to BASE, rerun five persistent requests with only that variable. Record the result, unset the variable, and do not stack the second variable.

**Branch C: DFlash first request, divergence before rollback**

Test one at a time, stopping when one changes the first-divergence boundary:

1. `GGML_DFLASH_GPU_RING=0`
2. `GGML_DFLASH_FORCE_CPU_CROSS=1`
3. `GGML_DFLASH_DISABLE_KV_CACHE=1`

A speed change without token-boundary change does not confirm a correctness root cause.

**Branch D: MTP and DFlash first request fail at the same acceptance boundary**

Trace:

- `slot.spec_i_batch` construction;
- batch `logits[]` rows;
- sub-batch offset `off`;
- `common_sampler_sample_and_accept_n()`;
- accepted and bonus token handling;
- sampler clone/restore.

Verify every sampled index maps to a requested target-logits row. Do not add a null-logits fallback.

- [ ] **Step 7: Remove temporary diagnostics**

Before leaving Task 3, restore every temporary diagnostic file from `$EVIDENCE/snapshots/task3-diagnostics/` and verify its SHA-256 matches the pre-diagnostic value. Preserve the diagnostic diff under `$EVIDENCE/hypotheses/task3-diagnostics.patch`. If no diagnostics were added, write `No temporary diagnostics added` to that path.

- [ ] **Step 8: State one confirmed root cause**

Write `$EVIDENCE/hypotheses/root-cause.md` with exactly:

```text
I think <specific state transition or index mapping> is the root cause because <diagnostic variable or trace> moved the first divergence from <old index> to <new result>, while <control> did not change.

Failing command: <exact command>
Expected failure before fix: <exact validator reason or token divergence>
Source boundary: <file, function, and relevant state>
Minimal fix surface: <one or two files>
```

If no branch confirms a boundary, do not guess. Ask GLM-5.2 for an architecture review with the token divergence and QA lines, add one environment-gated observation at the narrowest unknown boundary, and repeat Task 3. If Ollama is unavailable, record the failure, use the same evidence to choose one additional observation point, and continue; GLM unavailability is not permission to invent a source fix.

- [ ] **Step 9: Run Milestone M3 protocol**

GLM must specifically challenge whether the claimed root cause is causal rather than correlated. Next task: `Task 4: RED-GREEN minimal production fix`. Continue automatically only after repository evidence answers any concrete GLM contradiction.

---

### Task 4: RED-GREEN Minimal Production Fix Loop

**Files:**
- Remove from active source: the exact starting patch preserved at `$EVIDENCE/source-start/original-working.patch`.
- Modify only the minimal source boundary named in `root-cause.md`.
- Test with the exact model-backed RED command from Task 3.
- Add or modify the nearest existing unit test when the defect can be represented without model files.

**Interfaces:**
- Produces one accepted correctness patch and a GREEN reproduction on a source base without the reduced-verification/profiling experiment.
- Produces `$EVIDENCE/hypotheses/accepted-fix.patch` relative to the clean pre-fix source snapshot.

- [ ] **Step 1: Remove the starting experiment before production implementation**

Task 3 has already tested the actual dirty candidate and restored all temporary diagnostics. Before any correctness fix, verify `git diff --name-only` at Task 0 showed that the starting tracked patch modified only `tools/server/server-context.cpp`. If the starting patch touched any second tracked file, update the removal proof for every listed file before proceeding.

Remove the exact starting experiment while no accepted fix can overlap it:

```bash
git -C "$WORK" apply --reverse --check "$EVIDENCE/source-start/original-working.patch"
git -C "$WORK" apply --reverse "$EVIDENCE/source-start/original-working.patch"
git -C "$WORK" diff --check
```

Verify `tools/server/server-context.cpp` is byte-identical to committed HEAD:

```bash
git -C "$WORK" show HEAD:tools/server/server-context.cpp > "$EVIDENCE/build/server-context-head.cpp"
cmp "$EVIDENCE/build/server-context-head.cpp" "$WORK/tools/server/server-context.cpp"
```

If reverse-check or comparison fails, do not force it. Restore any Task 3 diagnostics, compare against `$EVIDENCE/source-start/server-context.cpp`, and resolve the mismatch before continuing.

- [ ] **Step 2: Preserve and rerun the automated RED test after experiment removal**

Copy the exact command and environment from `root-cause.md` into:

```text
$EVIDENCE/scripts/red-root-cause.sh
```

The script must:

- source `/tmp/beellama-working-correctness-state.env`;
- use a unique port;
- remove no pre-existing evidence;
- write a new JSONL and server log under `$EVIDENCE/records/red/`;
- exit nonzero for the expected validator or token-equivalence failure.

Build the source without the experiment, then run the script. It must remain RED for the same first-divergence or request-boundary reason. If removal makes it GREEN or changes the boundary, return to Task 3 and isolate which starting hunk affected correctness.

- [ ] **Step 3: Snapshot every file in the proposed fix surface**

For each file, preserve its directory structure under:

```text
$EVIDENCE/snapshots/task4-attempt-<N>/
```

Record SHA-256 before editing. These are now clean production-source snapshots plus accepted harness changes.

- [ ] **Step 4: Add the nearest focused regression test before production code**

Use this mapping:

```text
request reset/state -> bench/vulkan-gap/test_repeated_requests.py and test-server-prompt-checkpoint
spec_i_batch/logits mapping -> test-sampling or test-dflash-decode
DFlash rollback/re-eval -> test-dflash-decode
DFlash ring/cross routing -> test-dflash-ring or test-dflash-plumbing
MTP layer/state handling -> test-speculative and bench/vulkan-gap/test_repeated_requests.py
```

The model-backed RED script remains mandatory even when a model-free unit test is possible. Before running a C++ test, build all configured targets so its executable exists:

```bash
cmake --build "$WORK/build-vulkan" -j"$(nproc)"
```

Run the focused unit test and confirm it fails for the expected missing behavior, not a missing executable, syntax error, or setup error.

- [ ] **Step 5: Implement one minimal fix**

Change only the source state identified in `root-cause.md`. Do not include reduced verification, profiling, batching, adaptive draft max, or cleanup refactors.

Comments must explain only a non-obvious invariant. Do not mention the task or user.

- [ ] **Step 6: Build and run focused GREEN**

```bash
cmake --build "$WORK/build-vulkan" --target llama-server -j"$(nproc)"
bash "$EVIDENCE/scripts/red-root-cause.sh"
```

Expected GREEN: exit 0, token-equivalence or request-boundary failure removed, no new forbidden log line. Run the focused unit test and require GREEN.

- [ ] **Step 7: Decide retain or reject**

Retain only if all are true:

- exact RED command is now GREEN;
- BASE control remains valid;
- first divergence is removed, not merely delayed into invalid output;
- no new assertion, warning, HTTP failure, or zero-draft row appears.

If rejected:

1. copy failed delta to `$EVIDENCE/hypotheses/attempt-<N>-rejected.patch`;
2. restore each touched file from the exact attempt snapshot with `cp`;
3. verify restored SHA-256 equals the pre-attempt value;
4. append rejection to `ledger.tsv`;
5. return to Task 3 with one new hypothesis.

- [ ] **Step 8: Handle three failed source hypotheses**

After three rejected production attempts, do not try attempt four immediately. Ask GLM-5.2 to review the root-cause trace, all three diffs, observed results, restoration hashes, and current architecture. If Ollama is unavailable, record that fact and use the same evidence to add one new observation point before forming another hypothesis. Return to Task 3 and continue until one fix is accepted or an external blocker is proven.

- [ ] **Step 9: Preserve the accepted fix delta**

Diff the accepted files against their clean Task 4 pre-fix snapshots and save:

```text
$EVIDENCE/hypotheses/accepted-fix.patch
$EVIDENCE/hypotheses/accepted-fix-files.txt
```

Run `git diff --check` and verify none of the symbols or comments from `original-working.patch` reappeared.

- [ ] **Step 10: Run Milestone M4 protocol**

Next task: `Task 5: Full single-slot correctness`. Continue automatically.

---

### Task 5: Full Single-Slot Correctness Under Both Configurations

**Files:**
- Do not change production code unless a gate exposes a new root cause.
- Create records: `$EVIDENCE/records/single-slot/<config>/`

**Interfaces:**
- Produces 60 measured rows: 2 configurations x 3 modes x 2 prompts x 5 rows.

- [ ] **Step 1: Build test executables and run cheap regression tests**

A target-scoped `llama-server` build does not build CTest executables. Build all configured targets first:

```bash
cmake --build "$WORK/build-vulkan" -j"$(nproc)"
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir "$WORK/build-vulkan" --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git -C "$WORK" diff --check
```

All must exit 0.

- [ ] **Step 2: Run the smoke configuration matrix**

Set:

```bash
export VK_GAP_BATCH=512
export VK_GAP_UBATCH=128
export VK_GAP_CTX_SIZE=2048
export VK_GAP_NP=1
unset GGML_DFLASH_REDUCED_VERIFY
```

Run all modes and prompts with one persistent server per cell:

```bash
for MODE in base mtp dflash; do
  for PROMPT in coding math; do
    python3 bench/vulkan-gap/run.py "$MODE" "$PROMPT" \
      --repetitions 5 --gen-tokens 512 --skip-warmup \
      --reference-label working-tree-correct-smoke \
      --external-source-head "$HEAD_SHA" \
      --output "$EVIDENCE/records/single-slot/smoke/${MODE}-${PROMPT}.jsonl"
  done
done
```

Every command must exit 0.

- [ ] **Step 3: Run the matrix configuration**

Set:

```bash
export VK_GAP_BATCH=2048
export VK_GAP_UBATCH=512
export VK_GAP_CTX_SIZE=8192
```

Repeat all six cells into `$EVIDENCE/records/single-slot/matrix/`.

- [ ] **Step 4: Reapply final validators independently**

Use an evidence-side Python script, not the stored validity bit. Require every End Goal content, draft, timing, and row-count gate.

For fresh coding repetition 1, compare raw MTP and DFlash token IDs with BASE under the same configuration. Require exact equality.

- [ ] **Step 5: Parse all server logs**

Require zero forbidden patterns. A `non-consecutive token position` warning is a failure even if HTTP responses validate.

- [ ] **Step 6: Route any failure back to root-cause loop**

Reproduce only the failing mode/prompt/configuration. Add it to `ledger.tsv`, return to Task 3, then rerun all of Task 5 after the next accepted fix. Do not continue to multi-slot with any single-slot failure.

- [ ] **Step 7: Run Milestone M5 protocol**

Next task: `Task 6: Persistent and concurrent isolation`. Continue automatically.

---

### Task 6: Persistent and Concurrent Multi-Slot Isolation

**Files:**
- Use: `bench/vulkan-gap/repeated_requests.py`
- Create records: `$EVIDENCE/records/isolation/`

**Interfaces:**
- Proves request reuse and `-np 2` isolation for MTP and DFlash.

- [ ] **Step 1: Run explicit persistent gates**

For each configuration, mode, and prompt:

```bash
python3 bench/vulkan-gap/repeated_requests.py "$MODE" \
  --np 1 --rounds 5 --prompt "$PROMPT" --gen-tokens 512 \
  --output "$EVIDENCE/records/isolation/${CONFIG}-${MODE}-${PROMPT}-persistent.jsonl"
```

Run MTP and DFlash, coding and math, under smoke and matrix configurations. Every command exits 0 and every speculative row has drafts.

- [ ] **Step 2: Run three concurrent rounds**

For each configuration and mode:

```bash
for ROUND in 1 2 3; do
  python3 bench/vulkan-gap/repeated_requests.py "$MODE" \
    --np 2 --rounds 1 --gen-tokens 512 --concurrent \
    --output "$EVIDENCE/records/isolation/${CONFIG}-${MODE}-concurrent-${ROUND}.jsonl"
done
```

The concurrent driver must submit coding and math together. Require:

- coding output validates and contains no train/math answer text;
- math contains `2.4` and no `def fibonacci`;
- both responses have nonzero drafts;
- no server exit or cross-slot warning.

- [ ] **Step 3: Inspect slot and logits safety**

Parse logs for:

```text
invalid logits
corrupt output buffer
output_ids
sequence position
out-of-range hidden capture
cross-ring routing
GGML_ASSERT
```

Zero matches required, except informational lines that do not state a failure. Record exact disposition for every match.

- [ ] **Step 4: Route a failure back to Task 3**

A multi-slot-only failure is classified as shared target-view, output-row preservation, slot sequence identity, or per-view flag lifetime. Add one hypothesis and continue the same RED-GREEN loop. After a fix, rerun all Tasks 5 and 6.

- [ ] **Step 5: Run Milestone M6 protocol**

Next task: `Task 7: Audit and freeze the final source candidate`. Continue automatically.

---

### Task 7: Audit and Freeze the Final Source Candidate

**Files:**
- Read and verify all tracked modifications.
- Preserve: `$EVIDENCE/source-start/original-working.patch`
- Preserve: `$EVIDENCE/hypotheses/accepted-fix.patch`

**Interfaces:**
- Produces a frozen final source diff without the starting reduced-verification/profiling experiment or temporary diagnostics.

- [ ] **Step 1: Save the fully passing source state**

Copy every modified source and test file to `$EVIDENCE/snapshots/task7-passing-final/` and record hashes. Save full diff as:

```text
$EVIDENCE/snapshots/task7-passing-final.patch
```

- [ ] **Step 2: Prove the starting experiment is absent**

Parse `$EVIDENCE/source-start/original-working.patch` for every added symbol and marker, then verify none remains in current source unless also listed in `accepted-fix.patch`. Also compare every current production file against its matching Task 0 snapshot when present; for a production file first touched in Task 4, compare against its Task 4 pre-fix snapshot and committed HEAD. Every changed production hunk must appear in `accepted-fix.patch`; harness and test files are reviewed separately. At minimum require zero source matches for these known experiment identifiers:

```text
dflash_sample_reduced_verify
dflash_reduced_verify_active
dflash_reduced_verify_top_k
dflash_profile_enabled
t_dflash_cycle_start
t_dflash_draft_total
t_dflash_verify_total
t_dflash_accept_total
Task 7: Compact reduced verification
Task 8: DFlash cycle profiling
```

The patch must remain preserved in evidence. Do not delete or reapply it.

- [ ] **Step 3: Prove temporary diagnostics are absent**

Compare all files listed in `$EVIDENCE/hypotheses/task3-diagnostics.patch` with current source. Require zero temporary diagnostic additions. Environment-gated diagnostics that predated this recovery are allowed only when present in the Task 0 source snapshot.

- [ ] **Step 4: Inspect final diff scope**

List every modified tracked file and assign it one category:

```text
harness correctness
regression test
accepted production correctness fix
documentation or execution plan
```

Compare production files to `accepted-fix-files.txt`. Any production file outside that list is rejected unless it is a harness or test file named by Task 1 or Task 4.

- [ ] **Step 5: Freeze final hashes and run a focused smoke**

Write hashes to `$EVIDENCE/snapshots/task7-final-SHA256SUMS`. Build, run model-free tests, selected CTests, one fresh token-equivalence set, five DFlash persistent coding requests, and one DFlash concurrent round under matrix settings.

If any result differs from the passing Task 5/6 state, restore from `$EVIDENCE/snapshots/task7-passing-final/` and identify the unexpected mutation before proceeding.

- [ ] **Step 6: Run Milestone M7 protocol**

Next task: `Task 8: Fresh final verification and result handoff`. Continue automatically.

---

### Task 8: Fresh Final Verification and Result Handoff

**Files:**
- Create ignored build directory: `build-vulkan-correctness-final/`
- Create: `docs/superpowers/results/2026-07-31-vulkan-speculative-working-tree-correctness-recovery.md`
- Create outside Git: `$EVIDENCE/final/`

**Interfaces:**
- Produces final auditable evidence and a worktree ready for human review.

- [ ] **Step 1: Configure from an empty Release build directory**

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

This unscoped build is intentional: it builds `llama-server`, `llama-bench`, and every configured CTest executable. Require exit 0.

- [ ] **Step 2: Record final provenance**

Record:

- HEAD and complete source diff hash;
- final server version and SHA-256;
- every `libllama*` and `libggml*` SHA-256;
- CMake flags;
- model hashes;
- Vulkan summary;
- exact command and environment for every final cell.

Point `VK_GAP_SERVER` to the fresh final binary.

- [ ] **Step 3: Run final model-free and CTest gates**

```bash
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir "$WORK/build-vulkan-correctness-final" --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git -C "$WORK" diff --check
```

All exit 0.

- [ ] **Step 4: Repeat the complete final correctness suite**

Using the fresh final binary, repeat:

- both six-cell single-slot matrices from Task 5;
- all persistent gates from Task 6;
- three `-np 2` rounds per speculative mode and configuration;
- independent validation and forbidden-log scans.

For each configuration, designate BASE coding repetition 1 from this final-binary run as the oracle. Record its exact JSONL path and token hash. Require MTP and DFlash fresh coding repetition 1 to have exactly equal `token_ids`. Both speculative modes must use the same BASE oracle for that configuration.

Do not reuse Task 5/6 records as final evidence.

- [ ] **Step 5: Apply the release correctness gate**

Pass only when every End Goal item is satisfied. If any row fails, return to Task 3 and continue. Do not write a completion result document around an open failure.

- [ ] **Step 6: Request final GLM-5.2 review**

Ask GLM to verify:

1. final records came from the fresh final binary;
2. raw token equivalence uses the correct BASE oracle;
3. no invalid row entered a passing count;
4. request reuse and multi-slot isolation are both covered;
5. the original reduced/profiling experiment is absent;
6. each retained production change has RED/GREEN evidence;
7. no performance claim is made.

Verify every concrete concern before changing the result.

- [ ] **Step 7: Write the final result document**

Include:

- exact starting and final source state;
- accepted root cause and fix;
- rejected hypotheses;
- harness defects corrected;
- commands and exit codes;
- all record and log paths;
- valid counts per cell;
- token-equivalence results;
- persistent and concurrent results;
- binary/library/model hashes;
- GLM reviews and any disagreement;
- remaining risks;
- explicit statement that no commit, push, PR, or performance closure occurred.

- [ ] **Step 8: Run Milestone M8 protocol and stop**

The continuation summary must state `CORRECTNESS GOAL ACHIEVED` only if every gate passed. Otherwise it must state the exact first failing gate and return to Task 3.

After success, stop for the human to inspect the diff and decide whether to authorize a commit. Do not commit or push.

## Self-Review Checklist

Before handing this plan to the execution agent, verify:

- [ ] Every task names exact files and evidence paths.
- [ ] Harness correctness precedes model-backed conclusions.
- [ ] Fresh and persistent requests are separated.
- [ ] BASE is validated before serving as token oracle.
- [ ] Every production change requires RED first.
- [ ] Failed experiments are restored from snapshots.
- [ ] Three failed fixes trigger architecture review, not a fourth guess.
- [ ] Single-slot gates precede multi-slot gates.
- [ ] Both server configurations are covered.
- [ ] The dirty experiment is preserved, tested, then removed.
- [ ] Final verification uses a fresh Release build.
- [ ] GLM review and GLM-authored continuation summary occur at every milestone.
- [ ] Smart compaction is focused on the GLM summary when actual context usage is high.
- [ ] No task asks for routine human confirmation.
- [ ] No task commits, pushes, or creates a PR.
