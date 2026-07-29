# Vulkan Qwen3.6-27B Speculative Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reproducible BASE/MTP/DFlash evidence on W7800 Vulkan,
implement a safe opt-in reduced-verify consumer, and use measured profiles—not
assumptions—to choose any follow-up optimization and the recommended mode.

**Architecture:** A provenance-recording harness is the first deliverable. The
only pre-approved optimization is reduced verify, activated per decode view and
sampled with correct multi-slot row mapping. Verify padding and Vulkan timeline
work are investigation gates, not implementation tasks, because the previous
pseudocode was unsafe.

**Tech Stack:** C++17, llama.cpp server, Vulkan/RADV, Python 3 standard library,
CMake, JSONL benchmark records.

## Global Constraints

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`.
- Reviewed baseline revision: `392a6c057`.
- Target: `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`.
- DFlash draft: `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf`.
- Device under test: the Vulkan device whose `--list-devices` line contains
  `NAVI31`; record the exact selected line.
- Do not use the historical 19.75 or 27.23 t/s comments as acceptance oracles.
- Run at least five measured requests after one warm-up request. Report median,
  min, max, and median absolute deviation (MAD).
- A new optimization flag defaults to the old path until a separate default-flip
  task is approved.
- Never fall back to full-logits sampling after a decode that intentionally
  omitted the full-logits buffer. Reject the compact view before decode or
  re-decode it through the full path.
- Do not implement verify padding in this plan.
- Do not implement the previous empty-submit timeline-semaphore pseudocode.
- Do not commit unless the user explicitly authorizes commits. Task boundaries
  list files to stage, but no task runs `git commit` by default.

---

## File Structure

**Create:**

- `bench/vulkan-gap/prompts/coding.txt` — fixed coding prompt.
- `bench/vulkan-gap/prompts/math.txt` — fixed math prompt.
- `bench/vulkan-gap/run.py` — server lifecycle, request, provenance, JSONL data.
- `bench/vulkan-gap/summarize.py` — median/min/max/MAD and correctness summary.
- `bench/vulkan-gap/test_harness.py` — unit tests for response parsing and stats.
- `docs/superpowers/results/2026-07-29-gap-closure.md` — reviewed result tables.

**Modify for reduced verify:**

- `tools/server/server-context.cpp` — per-view activation, cleanup, and compact
  row mapping.
- `docs/superpowers/results/2026-07-29-gap-closure.md` — before/after evidence.

**Explicitly unchanged in this plan:**

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — no speculative semaphore code.
- Verify-padding code in `server_slot::handle_last_sampled_token()` — remains
  exact-size.

---

### Task 0: Build a testable benchmark harness

**Files:**
- Create: `bench/vulkan-gap/prompts/coding.txt`
- Create: `bench/vulkan-gap/prompts/math.txt`
- Create: `bench/vulkan-gap/run.py`
- Create: `bench/vulkan-gap/summarize.py`
- Test: `bench/vulkan-gap/test_harness.py`
- Create: `docs/superpowers/results/2026-07-29-gap-closure.md`

**Interfaces:**
- Produces: `run.py MODE PROMPT --repetitions N --output FILE` where `MODE` is
  `base`, `mtp`, or `dflash`.
- Produces one JSON object per measured request with revision, binary hash,
  device, command, request, response timing, output hash, and log path.
- `summarize.py FILE` prints grouped median/min/max/MAD and output-hash sets.

- [ ] **Step 1: Write failing harness tests**

Create `bench/vulkan-gap/test_harness.py`:

```python
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("vk_gap", HERE / "run.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_extract_timing_and_acceptance():
    response = {
        "content": "ok",
        "tokens_predicted": 200,
        "timings": {
            "predicted_per_second": 19.5,
            "draft_n": 100,
            "draft_n_accepted": 45,
        },
    }
    got = MOD.extract_measurement(response)
    assert got["tokens_predicted"] == 200
    assert got["predicted_per_second"] == 19.5
    assert got["draft_accept_pct"] == 45.0


def test_extract_acceptance_is_null_for_base():
    got = MOD.extract_measurement({
        "content": "ok",
        "tokens_predicted": 10,
        "timings": {"predicted_per_second": 20.0},
    })
    assert got["draft_accept_pct"] is None


def test_summary_stats():
    assert MOD.summary_stats([10.0, 11.0, 30.0]) == {
        "median": 11.0,
        "min": 10.0,
        "max": 30.0,
        "mad": 1.0,
    }
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
python3 -m unittest bench/vulkan-gap/test_harness.py -v
```

Expected: import failure because `bench/vulkan-gap/run.py` does not exist.

- [ ] **Step 3: Create the fixed prompts**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
mkdir -p bench/vulkan-gap/prompts docs/superpowers/results
printf '%s\n' 'Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.' > bench/vulkan-gap/prompts/coding.txt
printf '%s\n' 'Solve step by step: A train leaves station A at 60 km/h heading east. Another leaves station B, 240 km east of A, at 40 km/h heading west. After how many hours do they meet? Show the reasoning then the answer.' > bench/vulkan-gap/prompts/math.txt
```

- [ ] **Step 4: Implement the pure parsing/statistics helpers first**

Start `bench/vulkan-gap/run.py` with:

```python
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def extract_measurement(response):
    timings = response.get("timings") or {}
    draft_n = int(timings.get("draft_n") or 0)
    accepted = int(timings.get("draft_n_accepted") or 0)
    return {
        "tokens_predicted": int(response.get("tokens_predicted") or 0),
        "predicted_per_second": float(timings.get("predicted_per_second") or 0.0),
        "draft_accept_pct": (100.0 * accepted / draft_n) if draft_n else None,
        "content_sha256": hashlib.sha256(
            response.get("content", "").encode("utf-8")
        ).hexdigest(),
        "content": response.get("content", ""),
    }


def summary_stats(values):
    med = statistics.median(values)
    deviations = [abs(value - med) for value in values]
    return {
        "median": med,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(deviations),
    }
```

- [ ] **Step 5: Run the helper tests**

Run the Step 2 command again.

Expected: all three tests pass.

- [ ] **Step 6: Complete `run.py` with bounded server lifecycle**

Add these behaviors; implement each exactly rather than using `pkill` or an
unbounded curl loop:

```python
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(proc, port, timeout_s=180):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        try:
            http_json(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


def find_device(server):
    text = subprocess.run(
        [str(server), "--list-devices"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    for line in text.splitlines():
        if "NAVI31" in line:
            return line.split(":", 1)[0].strip(), line.strip()
    raise RuntimeError("no Vulkan NAVI31 device found")


def mode_args(mode, draft):
    if mode == "base":
        return ["--spec-type", "none"]
    if mode == "mtp":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"]
    if mode == "dflash":
        return [
            "--spec-type", "dflash",
            "--spec-draft-model", str(draft),
            "--spec-draft-n-max", "8",
            "--spec-branch-budget", "0",
            "--spec-dflash-cross-ctx", "512",
        ]
    raise ValueError(mode)
```

The CLI/main function must:

1. read `VK_GAP_SERVER` (default `ROOT/build/bin/llama-server`),
   `VK_GAP_MODEL`, `VK_GAP_DRAFT`, `VK_GAP_PORT` (default 8099),
   `VK_GAP_TEMP` (default 0), and `VK_GAP_TOP_K` (default 20);
2. reject missing binaries/models before starting;
3. launch one server with an argument list, not an interpolated shell string;
4. write stderr/stdout to `bench/vulkan-gap/logs/<timestamp>-<mode>-<prompt>.log`;
5. use `try/finally` to terminate only the PID it started, then kill it after a
   five-second bounded wait if necessary;
6. send one 32-token warm-up request, followed by the requested number of
   measured requests with `cache_prompt: false`;
7. use a 600-second request timeout and treat an HTTP/JSON failure as a failed
   record, not a zero-t/s result; and
8. append JSONL records only after a valid response with
   `tokens_predicted > 0` and `predicted_per_second > 0`.

Use this exact common server argument list:

```python
common = [
    str(server), "-m", str(model),
    "--port", str(port), "-np", os.environ.get("VK_GAP_NP", "1"),
    "--kv-unified", "-ngl", "all", "-b", "2048", "-ub", "512",
    "--ctx-size", "8192", "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0", "--flash-attn", "on",
    "--device", device, "--jinja", "--reasoning", "off",
    "--no-mmap", "--no-host", "--host", "127.0.0.1",
]
```

Use this request shape for measured runs:

```python
payload = {
    "prompt": prompt_text,
    "n_predict": args.gen_tokens,
    "temperature": temp,
    "top_k": top_k,
    "top_p": 1.0,
    "min_p": 0.0,
    "seed": 7,
    "stream": False,
    "cache_prompt": False,
}
```

Each JSONL record must include:

```python
record = {
    "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "server": str(server),
    "server_sha256": sha256_file(server),
    "device": device_line,
    "model": str(model),
    "draft": str(draft) if mode == "dflash" else None,
    "mode": mode,
    "prompt": args.prompt,
    "repetition": repetition,
    "command": common + mode_args(mode, draft),
    "request": payload,
    "measurement": extract_measurement(response),
    "log": str(log_path),
    "environment": {
        key: value for key, value in os.environ.items()
        if key.startswith("GGML_DFLASH_") or key.startswith("VK_GAP_")
    },
}
```

- [ ] **Step 7: Implement `summarize.py`**

Group valid JSONL rows by `(revision, server_sha256, mode, prompt,
environment)`. Print count, t/s median/min/max/MAD, median acceptance, and the
set of content hashes. Exit nonzero if a group has fewer than five rows or any
deterministic group has more than one output hash.

- [ ] **Step 8: Test failure handling before loading the model**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
VK_GAP_SERVER=/does/not/exist python3 bench/vulkan-gap/run.py base coding --repetitions 1 --output /tmp/never.jsonl
```

Expected: nonzero exit with `missing server`; no JSONL row and no lingering
server process.

- [ ] **Step 9: Build and smoke-test BASE**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
cmake --build build -j
python3 bench/vulkan-gap/run.py base coding --repetitions 1 --gen-tokens 64 --output /tmp/vk-gap-smoke.jsonl
python3 bench/vulkan-gap/summarize.py /tmp/vk-gap-smoke.jsonl || true
```

Expected: one valid record; summarizer complains only that five repetitions are
required.

- [ ] **Step 10: Initialize the results document**

Record revision, binary hash, device line, model paths, harness command, and an
empty six-cell table. Label historical runner-comment values as unverified
context, not baselines.

**Review boundary:** inspect the harness and smoke record. If authorized to
commit, stage only `bench/vulkan-gap/` and the results document.

---

### Task 1: Establish current and reference measurements

**Files:**
- Modify: `docs/superpowers/results/2026-07-29-gap-closure.md`

**Interfaces:**
- Consumes Task 0's runner and summarizer.
- Produces the current six-cell matrix and, only when supplied, a same-harness
  fork reference.

- [ ] **Step 1: Run deterministic correctness baselines**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
rm -f /tmp/vk-gap-correct.jsonl
VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py base   coding --repetitions 3 --gen-tokens 200 --output /tmp/vk-gap-correct.jsonl
VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py mtp    coding --repetitions 3 --gen-tokens 200 --output /tmp/vk-gap-correct.jsonl
VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash coding --repetitions 3 --gen-tokens 200 --output /tmp/vk-gap-correct.jsonl
```

Expected: one output hash across all deterministic BASE rows; each speculative
mode must have the same hash. A stall, crash, or mismatch is recorded as a
pre-existing blocker before optimization.

- [ ] **Step 2: Run the current six-cell benchmark**

Run each pair below with a fresh output file:

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
rm -f /tmp/vk-gap-current.jsonl
for mode in base mtp dflash; do
  for prompt in coding math; do
    VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py "$mode" "$prompt" \
      --repetitions 5 --gen-tokens 512 --output /tmp/vk-gap-current.jsonl
  done
done
python3 bench/vulkan-gap/summarize.py /tmp/vk-gap-current.jsonl
```

Expected: six groups with five rows each, or an explicit failed mode with logs.

- [ ] **Step 3: Run a fork reference only from an explicit binary**

Do not infer performance from
`/crypt/beellama.cpp/startvulkan_qwen3.6-27b.sh`. Use the current main worktree
as an explicit reference artifact only after verifying that it supports the same
model and command:

```bash
export VK_GAP_FORK_SERVER=/crypt/beellama.cpp/build-vulkan/bin/llama-server
export VK_GAP_FORK_REVISION="$(git -C /crypt/beellama.cpp rev-parse HEAD)"
test -x "$VK_GAP_FORK_SERVER"
"$VK_GAP_FORK_SERVER" --version
sha256sum "$VK_GAP_FORK_SERVER"
```

Archive the revision, `--version`, and SHA-256 output. Then rerun DFlash with
`VK_GAP_SERVER="$VK_GAP_FORK_SERVER"`. If that binary is absent or rejects the
same command/model, mark the merge-to-fork percentage **BLOCKED: no comparable
fork artifact** rather than substituting historical comments.

- [ ] **Step 4: Capture the existing profile before changing code**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
GGML_DFLASH_PROFILE=1 GGML_DFLASH_PROFILE_SYNC_SPLIT=1 \
  VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash math \
  --repetitions 1 --gen-tokens 512 --output /tmp/vk-gap-profile.jsonl
```

Extract existing profile lines for target decode/output transfer, verify sync,
drafter prepare/decode/argmax, ring copy, rollback, and graph reuse. Compute each
as a percentage of measured generation time. Do not add timers yet.

- [ ] **Step 5: Write the Phase 0 result and ranked measured costs**

Every claimed root cause must cite a profile counter or A/B result. Replace
"root cause" with "hypothesis" when evidence is absent.

**Review boundary:** no production code has changed. Confirm whether reduced
verify remains the largest safe experiment.

---

### Task 2: Implement opt-in reduced verify safely

**Files:**
- Modify: `tools/server/server-context.cpp`
- Test: the deterministic, sequential-request, supported-sampling, unsupported-
  sampling, and `-np 2` integration commands below.
- Modify: `docs/superpowers/results/2026-07-29-gap-closure.md`

**Interfaces:**
- Consumes: `dflash_select_reduced_verify_plan()`,
  `dflash_batch_view_is_reduced_verify()`,
  `llama_set_dflash_verify_logits()`, `llama_set_dflash_consume_reduced()`, and
  `common_sampler_sample_reduced_and_accept_n()`.
- Produces: compact consumption only when every speculative slot in the current
  view is compatible and has the same top-k (1..64).

- [ ] **Step 1: Add a failing multi-slot integration check**

Run the current full-logits path with `VK_GAP_NP=2` and two concurrent distinct
prompts. Save each response hash. Repeat three times. This establishes that the
test itself detects slot cross-contamination before compact consumption is
changed.

- [ ] **Step 2: Replace one-time static activation with per-view state**

In `server_context_impl`, keep batch-level state because one target context and
one output allocation serve the whole decode view:

```cpp
bool dflash_compact_verify_active = false;
int  dflash_compact_verify_top_k  = 0;
```

Before `llama_decode(ctx_tgt, batch_view)`:

1. default both fields to false/zero;
2. require `GGML_DFLASH_REDUCED_VERIFY=1` explicitly;
3. inspect every generating slot with a non-empty `spec_draft` using
   `slot.task->params.sampling`;
4. require `common_sampler_supports_reduced(slot.smpl.get())`;
5. require every slot plan to be enabled and have the same top-k;
6. reject top-k outside `[1, 64]` (the context setter clamps to 64);
7. call `dflash_batch_view_is_reduced_verify()` with the current `off`,
   `batch_view.n_tokens`, and chosen top-k; and
8. only then call:

```cpp
llama_set_dflash_verify_logits(ctx_tgt, true, chosen_top_k);
llama_set_dflash_consume_reduced(ctx_tgt, true);
dflash_compact_verify_active = true;
dflash_compact_verify_top_k = chosen_top_k;
```

Otherwise call `llama_set_dflash_consume_reduced(ctx_tgt, false)` and preserve
full-logits sampling. Do not use a function-local `static` guard.

- [ ] **Step 3: Guarantee consume-state restoration**

Immediately after `llama_decode()` returns—before testing `ret`—execute:

```cpp
llama_set_dflash_consume_reduced(ctx_tgt, false);
```

Also set it false at the beginning of `server_context_impl::destroy()` and any
retry entry that can bypass the normal post-decode path. The verify-logits flag
may remain configured because it only emits compact output alongside full
logits when the consumer flag is false.

- [ ] **Step 4: Map compact rows per slot**

In `post_decode(n_batch_tokens, off, batch_view)`, do not pass the global compact
buffer base to every slot. For each slot:

```cpp
const int32_t k = llama_get_logits_argmax_k(slot.ctx_tgt);
std::vector<llama_token> slot_ids;
std::vector<float> slot_logits;
slot_ids.reserve(slot.spec_i_batch.size() * (size_t) k);
slot_logits.reserve(slot.spec_i_batch.size() * (size_t) k);

const llama_token * all_ids = llama_get_logits_argmax(slot.ctx_tgt);
const float * all_logits = llama_get_logits_argmax_probs(slot.ctx_tgt);
for (int32_t absolute_index : slot.spec_i_batch) {
    const int32_t row = absolute_index - off;
    GGML_ASSERT(row >= 0 && row < n_batch_tokens);
    slot_ids.insert(slot_ids.end(), all_ids + (size_t) row * k,
                    all_ids + (size_t) (row + 1) * k);
    slot_logits.insert(slot_logits.end(), all_logits + (size_t) row * k,
                       all_logits + (size_t) (row + 1) * k);
}
```

Then call `common_sampler_sample_reduced_and_accept_n()` with those contiguous
per-slot arrays and `slot.spec_draft.size() + 1` rows.

Before the call, assert `common_sampler_supports_reduced()` again. If it is
false, throw a descriptive internal error; do **not** call
`common_sampler_sample_and_accept_n()` against a missing logits buffer. A future
recovery path may re-decode with the consume flag off, but that is not part of
this minimal change.

- [ ] **Step 5: Build and verify the opt-out path first**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
cmake --build build -j
GGML_DFLASH_REDUCED_VERIFY=0 VK_GAP_TEMP=0 \
  python3 bench/vulkan-gap/run.py dflash coding --repetitions 3 \
  --gen-tokens 200 --output /tmp/reduced-off.jsonl
```

Expected: same deterministic output hash as Task 1 BASE and no null-logits
assertion.

- [ ] **Step 6: Verify greedy compact consumption across sequential requests**

```bash
GGML_DFLASH_REDUCED_VERIFY=1 VK_GAP_TEMP=0 \
  python3 bench/vulkan-gap/run.py dflash coding --repetitions 5 \
  --gen-tokens 200 --output /tmp/reduced-on-greedy.jsonl
python3 bench/vulkan-gap/summarize.py /tmp/reduced-on-greedy.jsonl
```

Expected: one hash, equal to BASE; five successful rows; profile reports compact
output use and no full-logits transfer for eligible verify views.

- [ ] **Step 7: Verify supported top-k sampling and unsupported fallback**

For supported stochastic sampling:

```bash
GGML_DFLASH_REDUCED_VERIFY=1 VK_GAP_TEMP=0.6 VK_GAP_TOP_K=20 \
  python3 bench/vulkan-gap/run.py dflash math --repetitions 5 \
  --gen-tokens 200 --output /tmp/reduced-on-topk20.jsonl
```

Compare against reduced-off with the same seed and payload. If exact hashes are
not stable because RNG consumption differs, compare both paths with a focused
sampler test using identical top-20 rows and cloned sampler state before making
a correctness claim.

For an unsupported request (`top_k=0` or active grammar), logs must show compact
verify rejected before decode and the request must complete through full logits.

- [ ] **Step 8: Run the `-np 2` row-isolation check**

Run two distinct concurrent requests with `VK_GAP_NP=2`, reduced verify on, three
times. Each prompt must match its own reduced-off deterministic hash. This gate
specifically catches the old plan's row-zero-for-every-slot bug.

- [ ] **Step 9: Measure reduced verify on/off**

Run both prompts, five repetitions each, on one binary:

```bash
GGML_DFLASH_REDUCED_VERIFY=0 VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash coding --repetitions 5 --gen-tokens 512 --output /tmp/reduced-ab.jsonl
GGML_DFLASH_REDUCED_VERIFY=1 VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash coding --repetitions 5 --gen-tokens 512 --output /tmp/reduced-ab.jsonl
GGML_DFLASH_REDUCED_VERIFY=0 VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash math   --repetitions 5 --gen-tokens 512 --output /tmp/reduced-ab.jsonl
GGML_DFLASH_REDUCED_VERIFY=1 VK_GAP_TEMP=0 python3 bench/vulkan-gap/run.py dflash math   --repetitions 5 --gen-tokens 512 --output /tmp/reduced-ab.jsonl
python3 bench/vulkan-gap/summarize.py /tmp/reduced-ab.jsonl
```

Keep the implementation opt-in if correct. Remove it if it has no measurable
benefit and adds maintenance cost. Do not flip the default in this task.

**Review boundary:** stage only server reduced-verify changes and result updates
if the user authorizes a commit.

---

### Task 3: Decide whether graph reuse deserves a separate design

**Files:**
- Modify only: `docs/superpowers/results/2026-07-29-gap-closure.md`

**Interfaces:**
- Consumes existing `llama_perf_context(...).n_reused` and profile timing.
- Produces a go/no-go decision; no verify-padding code.

- [ ] **Step 1: Quantify graph reuse and build/setup time**

Use the Task 1 profile and reduced-verify on/off runs. Record verify cycles,
`n_reused`, target decode time, scheduler setup/build time if already exposed,
and total generation time.

- [ ] **Step 2: Apply the gate**

Stop this line of work when graph rebuild/setup is below 5% of generation time
or when the maximum possible saving is smaller than run MAD.

If it exceeds the gate, write a separate design that addresses target compute
for dummy tokens, per-slot exact ranges, Qwen3.6 recurrent-state restoration,
all-accepted cycles, rollback/retry/error cleanup, and `-np 2`. Do not copy the
old `break; // single-slot benchmark` cleanup into production code.

---

### Task 4: Decide whether Vulkan synchronization deserves a separate design

**Files:**
- Modify only: `docs/superpowers/results/2026-07-29-gap-closure.md`

**Interfaces:**
- Consumes `profile_verify_sync_split_us` and a source trace of producer and
  consumer submissions.
- Produces a go/no-go decision and, if needed, a separate synchronization spec.

- [ ] **Step 1: Quantify the scheduler sync**

Record median verify-sync time and percentage of generation time after reduced
verify. Confirm whether the profiled wait is GPU completion required for output
readback, hidden/tape capture, both, or unrelated queued work.

- [ ] **Step 2: Trace queues and submissions**

Document exact functions and queues for:

1. target graph submission;
2. hidden/tape capture writes;
3. ring/interleave D2D submission; and
4. drafter consumption.

Use Vulkan validation layers for any later prototype.

- [ ] **Step 3: Apply the gate**

If the sync is below 5% of generation time or hidden by required output
readback, stop. Otherwise write a separate design whose fast path signals the
actual producer submission and waits in the actual consumer submission without
a CPU fence. An empty signal submit followed by an empty wait/fence is expressly
rejected because it drains prior queue work.

---

### Task 5: Produce the final mode recommendation

**Files:**
- Modify: `docs/superpowers/results/2026-07-29-gap-closure.md`
- Modify: `docs/beellama-features.md` only if the user requests public docs.

**Interfaces:**
- Consumes all accepted Task 1/2 results.
- Produces the six-cell recommendation and an honest blocked-items list.

- [ ] **Step 1: Rerun the six-cell matrix on the selected configuration**

Use five repetitions per cell and the exact same payloads as Phase 0. Keep
reduced verify opt-in unless a separate default-flip review has occurred.

- [ ] **Step 2: Check BASE regression**

Compare BASE median and dispersion against Phase 0. A change greater than 2%
with non-overlapping dispersion blocks completion.

- [ ] **Step 3: Write recommendations**

For each BASE/MTP/DFlash × coding/math cell, report median t/s, min/max/MAD,
acceptance when applicable, correctness status, revision, and selected mode.
Mark a stalled mode `STALL` with its log path; do not omit the cell.

- [ ] **Step 4: State the merge-versus-fork result correctly**

If an explicit fork artifact was rerun, report the same-harness ratio. If not,
state that the historical regression could not be closed or disproved because a
reproducible fork baseline was unavailable.

- [ ] **Step 5: Final verification**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
python3 -m unittest bench/vulkan-gap/test_harness.py -v
cmake --build build -j
python3 bench/vulkan-gap/summarize.py /tmp/vk-gap-final.jsonl
git diff --check
```

Expected: tests and build pass, six valid/explicitly failed cells, no whitespace
errors. Do not claim success without attaching the summarized evidence.

## Execution Notes

- The safe order is evidence → reduced verify → re-profile → optional new specs.
- Do not execute old Tasks 1.2 or 1.3 from earlier versions of this document.
- A failed reduced-verify request is debugged with
  `GGML_DFLASH_REDUCED_VERIFY=0`; do not continue to later tasks while its state
  lifecycle or row mapping is uncertain.
- The design intentionally narrows Phase 1. It is better to leave a measured
  bottleneck in place than to land an incorrect recurrent-memory or Vulkan-sync
  optimization.
