# Vulkan Speculative Clean Baseline Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-establish trustworthy correctness and performance baselines for `merge_llama_into_beellama_2`, then recover DFlash Vulkan performance to at least 90% of current `gboddaer/main` without weakening correctness.

**Architecture:** Preserve the current dirty investigation as evidence, but do all measurements in clean detached worktrees. Gate performance work behind persistent single-slot and concurrent multi-slot correctness. Compare clean builds with identical configuration, localize the gap with existing profiling, and test one hypothesis per iteration.

**Tech Stack:** C++17, CMake, Vulkan, Python `unittest`, CTest, BeeLlama benchmark harness, Qwen3.6-27B target and DFlash GGUFs.

## Global Constraints

- Do not modify, reset, clean, stash, commit, or push the existing dirty worktree while preserving evidence.
- Do not use `/crypt/beellama.cpp/bench/vulkan-gap/run_persistent.py`; it is untracked and its records did not enforce the strict validators.
- Use the tracked `bench/vulkan-gap/run.py` from the work branch for every publishable run.
- Use current `gboddaer/main`, not local `main` and not `adb92b36a`, as the performance target.
- Build both revisions from empty build directories with identical flags and an empty `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, and `LDFLAGS` environment.
- Correctness must pass before performance changes begin.
- Reduced verification remains disabled on the work baseline. Do not use or commit the current uncommitted Task 7 implementation.
- Change one variable per experiment. Revert failed experiments in the disposable experiment worktree before trying another.
- Do not call a benchmark row valid merely because its stored `measurement.valid` is true. Reapply the current validators to the response fields.
- Report proper median and median absolute deviation (MAD). Do not use `sorted(values)[len(values)//2]` for an even-sized sample.
- A performance change is retained only if all correctness gates pass and both coding and math improve by at least 5% or remove a confirmed component bottleneck.
- Final performance gate: work DFlash median throughput is at least 90% of current `gboddaer/main` for both coding and math.
- No commit without explicit human approval. The human writes or approves the commit message. Never push or create a PR.

---

## Current Evidence That This Plan Must Treat As Suspect

- Local work HEAD at plan creation: `fb7e5d080f257bb8d15ee68b850cfafe593a7899`.
- Fresh `gboddaer/main` at plan creation: `130ea2480bee8268907a102bd814e276f4e88bdf`.
- The work branch contains `gboddaer/main` and is 468 commits ahead of it.
- The worktree is dirty in `tools/server/server-context.cpp` and has two untracked Task 4 handoffs.
- The dirty source mixes Task 7 reduced verification with Task 8 profiling.
- The reduced-verification state is restored inside the first slot's `post_decode` path. A second active slot would then try full-logits sampling after an argmax-only decode.
- Task 7 was called complete even though its required reduced-on `-np 2` gate was marked pending.
- The Task 8 profiler does not measure wall-clock cycle time. It reports the sum of partial draft, `llama_decode`, and `common_speculative_accept` timers, while sampling synchronization and rollback/re-eval are omitted.
- Existing Task 6 records used `adb92b36a`, not current `gboddaer/main`, and were produced by the untracked runner.
- Existing Task 6 DFlash coding records contain four `finish_reason=length` rows but label all five rows valid. They therefore fail the current coding validator.
- Existing Task 6 statistics called an upper middle value the median for even-sized subsets.
- Historical `TASK_PROGRESS.md` already identified DFlash draft output sizing and graph-compute launch count as performance leads. Recheck those findings before inventing a new subsystem.

---

### Task 1: Freeze Evidence and Create Clean Measurement Worktrees

**Files:**
- Preserve outside Git: `/crypt/tmp/beellama-task8-quarantine-<timestamp>/`
- Create disposable worktrees: `/crypt/tmp/worktrees/beellama-baseline-work-<sha>/`, `/crypt/tmp/worktrees/beellama-baseline-main-<sha>/`
- Do not modify: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/`

**Interfaces:**
- Produces exact `WORK_SHA`, `MAIN_SHA`, a dirty patch, copied records, and two clean source trees.

- [ ] **Step 1: Fetch and record the current refs**

```bash
ROOT=/crypt/beellama.cpp
WORK_BRANCH=merge_llama_into_beellama_2

git -C "$ROOT" fetch --no-tags gboddaer main "$WORK_BRANCH"
WORK_SHA=$(git -C "$ROOT" rev-parse "$WORK_BRANCH")
MAIN_SHA=$(git -C "$ROOT" rev-parse gboddaer/main)
REMOTE_WORK_SHA=$(git -C "$ROOT" rev-parse gboddaer/"$WORK_BRANCH")
printf 'WORK_SHA=%s\nMAIN_SHA=%s\nREMOTE_WORK_SHA=%s\n' "$WORK_SHA" "$MAIN_SHA" "$REMOTE_WORK_SHA"
git -C "$ROOT" merge-base --is-ancestor "$MAIN_SHA" "$WORK_SHA"
```

Expected: fetch exits 0 and the ancestor check exits 0. Record any SHA change from the values in this plan.

- [ ] **Step 2: Preserve the dirty investigation without stashing**

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
Q=/crypt/tmp/beellama-task8-quarantine-$STAMP
DIRTY=/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
mkdir -p "$Q"

git -C "$DIRTY" status --short --branch > "$Q/worktree-status.txt"
git -C "$DIRTY" diff --binary > "$Q/server-context-and-other-dirty.patch"
cp -a /crypt/beellama.cpp/docs/superpowers/HANDOFF-vulkan-speculative-task6-complete.md "$Q/"
cp -a /crypt/beellama.cpp/docs/superpowers/HANDOFF-vulkan-speculative-task7-complete.md "$Q/"
cp -a /crypt/beellama.cpp/docs/superpowers/HANDOFF-vulkan-speculative-task8-*.md "$Q/"
cp -a /crypt/beellama.cpp/bench/vulkan-gap/records "$Q/records"
cp -a /crypt/beellama.cpp/bench/vulkan-gap/run_persistent.py "$Q/"
find "$Q" -type f -print0 | sort -z | xargs -0 sha256sum > "$Q/SHA256SUMS"
```

Expected: the patch is non-empty and the existing worktree status is unchanged.

- [ ] **Step 3: Create clean detached worktrees**

```bash
WORK_WT=/crypt/tmp/worktrees/beellama-baseline-work-${WORK_SHA:0:9}
MAIN_WT=/crypt/tmp/worktrees/beellama-baseline-main-${MAIN_SHA:0:9}

git -C "$ROOT" worktree add --detach "$WORK_WT" "$WORK_SHA"
git -C "$ROOT" worktree add --detach "$MAIN_WT" "$MAIN_SHA"
test -z "$(git -C "$WORK_WT" status --porcelain)"
test -z "$(git -C "$MAIN_WT" status --porcelain)"
```

Expected: both status checks exit 0.

- [ ] **Step 4: Record the repository topology**

```bash
git -C "$ROOT" rev-list --left-right --count "$MAIN_SHA...$WORK_SHA"
git -C "$ROOT" log --first-parent --oneline -30 "$WORK_SHA"
git -C "$ROOT" log --oneline gboddaer/"$WORK_BRANCH".."$WORK_SHA"
```

Expected at plan creation: `0 468` against main and two local commits ahead of the remote work branch. If different, use the new facts in all later manifests.

- [ ] **Step 5: Stop for evidence review**

Report the quarantine path, SHAs, worktree cleanliness, and branch divergence. Do not continue if either baseline worktree is dirty.

---

### Task 2: Build Identical Release Binaries and Complete Provenance

**Files:**
- Create in each disposable worktree: `build-vulkan-release/`
- Create: `$Q/work-build-manifest.txt`, `$Q/main-build-manifest.txt`

**Interfaces:**
- Produces two clean binaries whose executable, shared libraries, build flags, and source revisions are recorded.

- [ ] **Step 1: Configure both builds from empty directories**

```bash
for WT in "$WORK_WT" "$MAIN_WT"; do
    rm -rf "$WT/build-vulkan-release"
    env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
      cmake -S "$WT" -B "$WT/build-vulkan-release" \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=ON
done
```

Expected: both configurations exit 0. Do not reuse `build-vulkan` from the old investigation.

- [ ] **Step 2: Build the baseline targets**

```bash
cmake --build "$WORK_WT/build-vulkan-release" --target llama-server llama-bench -j"$(nproc)"
cmake --build "$MAIN_WT/build-vulkan-release" --target llama-server llama-bench -j"$(nproc)"
```

Expected: both builds exit 0.

- [ ] **Step 3: Record all runtime artifacts, not just the launcher**

```bash
for LABEL in work main; do
    if [ "$LABEL" = work ]; then WT=$WORK_WT; else WT=$MAIN_WT; fi
    OUT="$Q/${LABEL}-build-manifest.txt"
    {
        git -C "$WT" rev-parse HEAD
        "$WT/build-vulkan-release/bin/llama-server" --version
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|GGML_VULKAN|GGML_NATIVE):' \
          "$WT/build-vulkan-release/CMakeCache.txt"
        find "$WT/build-vulkan-release/bin" -maxdepth 1 -type f \
          \( -name 'llama-server' -o -name 'libllama*' -o -name 'libggml*' \) \
          -print0 | sort -z | xargs -0 sha256sum
    } > "$OUT" 2>&1
done
diff -u <(grep -E 'CMAKE|GGML' "$Q/work-build-manifest.txt") \
        <(grep -E 'CMAKE|GGML' "$Q/main-build-manifest.txt") || true
```

Expected: source revisions differ, but common build flags match. Shared-library hashes are expected to differ.

- [ ] **Step 4: Capture the machine state**

```bash
{
    date -u --iso-8601=seconds
    uname -a
    vulkaninfo --summary
    sha256sum /crypt/models/Qwen3.6-27B-Q4_K_M.gguf
    sha256sum /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
    ps -eo pid,comm,args | grep -E 'llama-(server|bench)' || true
} > "$Q/machine-and-model-manifest.txt" 2>&1
```

Expected: model hashes are recorded and no unrelated benchmark server is consuming the Vulkan device.

---

### Task 3: Prove the Harness and Re-audit the Old Records

**Files:**
- Test: `$WORK_WT/bench/vulkan-gap/test_harness_strict.py`
- Test: `$WORK_WT/bench/vulkan-gap/test_repeated_requests.py`
- Read only: `$Q/records/*.jsonl`
- Create: `$Q/old-record-audit.txt`

**Interfaces:**
- Produces evidence that the strict harness rejects known bad rows and a list of invalid old claims.

- [ ] **Step 1: Run the tracked model-free tests**

```bash
cd "$WORK_WT"
python3 -m unittest \
  bench.vulkan-gap.test_harness_strict \
  bench.vulkan-gap.test_repeated_requests -v
```

Expected: exit 0. Record the exact test count.

- [ ] **Step 2: Re-audit old records without trusting their stored validity bit**

```bash
python3 - "$Q/records" > "$Q/old-record-audit.txt" <<'PY'
import collections
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = []
    tps = []
    for row in rows:
        m = row.get('measurement', {})
        if row.get('prompt') == 'coding' and m.get('finish_reason') != 'stop':
            bad.append(f"rep{row.get('repetition')}:coding_finish={m.get('finish_reason')}")
        if not m.get('content_sha256'):
            bad.append(f"rep{row.get('repetition')}:empty_hash")
        value = m.get('predicted_per_second')
        if isinstance(value, (int, float)) and 0 < value < 10000:
            tps.append(float(value))
    if rows:
        med = statistics.median(tps) if tps else None
        mad = statistics.median(abs(x - med) for x in tps) if tps else None
        print(path.name, f"rows={len(rows)}", f"median={med}", f"mad={mad}", f"gate_failures={bad}")
PY
```

Expected: the old DFlash coding records show `finish_reason=length` failures. The audit's proper medians differ from any upper-middle calculation on an even subset.

- [ ] **Step 3: Confirm the publishable runner and its warm-up behavior**

```bash
cd "$WORK_WT"
python3 bench/vulkan-gap/run.py --help
python3 bench/vulkan-gap/repeated_requests.py --help
```

Expected: `run.py` defaults to one warm-up plus five measured repetitions, keeps one server unless explicitly told to restart, and writes a manifest.

- [ ] **Step 4: Stop if the harness cannot reject known bad coding rows**

If a `finish_reason=length` coding row is accepted by the current tracked harness, add a failing model-free test first, make the smallest validator fix, and rerun Step 1. Do not start a model-backed benchmark until the test is green.

---

### Task 4: Establish Fresh Correctness Baselines

**Files:**
- Create outside Git: `$Q/correctness/<revision>/<mode>-<prompt>.jsonl`
- Create outside Git: `$Q/correctness/<revision>/*.manifest.json`

**Interfaces:**
- Produces strict BASE, MTP, and DFlash correctness results for work and current main.

- [ ] **Step 1: Define the common environment**

```bash
HARNESS_ROOT=$WORK_WT
MODEL=/crypt/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
mkdir -p "$Q/correctness/work" "$Q/correctness/main"
```

- [ ] **Step 2: Run the six-cell persistent matrix on clean work HEAD**

```bash
cd "$HARNESS_ROOT"
for MODE in base mtp dflash; do
  for PROMPT in coding math; do
    VK_GAP_SERVER="$WORK_WT/build-vulkan-release/bin/llama-server" \
    VK_GAP_MODEL="$MODEL" VK_GAP_DRAFT="$DRAFT" VK_GAP_PORT=8099 \
    VK_GAP_TEMP=0 VK_GAP_TOP_K=20 VK_GAP_NP=1 \
      python3 bench/vulkan-gap/run.py "$MODE" "$PROMPT" \
        --repetitions 5 --gen-tokens 512 \
        --reference-label work-clean \
        --external-source-head "$WORK_SHA" \
        --output "$Q/correctness/work/${MODE}-${PROMPT}.jsonl"
  done
done
```

Expected: every command exits 0. Each file has five measured rows plus a separate warm-up that is not counted.

- [ ] **Step 3: Run the same matrix on current `gboddaer/main`**

```bash
cd "$HARNESS_ROOT"
for MODE in base mtp dflash; do
  for PROMPT in coding math; do
    VK_GAP_SERVER="$MAIN_WT/build-vulkan-release/bin/llama-server" \
    VK_GAP_MODEL="$MODEL" VK_GAP_DRAFT="$DRAFT" VK_GAP_PORT=8100 \
    VK_GAP_TEMP=0 VK_GAP_TOP_K=20 VK_GAP_NP=1 \
      python3 bench/vulkan-gap/run.py "$MODE" "$PROMPT" \
        --repetitions 5 --gen-tokens 512 \
        --reference-label gboddaer-main-clean \
        --external-source-head "$MAIN_SHA" \
        --output "$Q/correctness/main/${MODE}-${PROMPT}.jsonl"
  done
done
```

Expected: every command exits 0. If current main fails a strict gate, record it and do not use invalid rows as a performance oracle.

- [ ] **Step 4: Run persistent repeated-request gates on work HEAD**

```bash
cd "$HARNESS_ROOT"
for MODE in mtp dflash; do
  for PROMPT in coding math; do
    VK_GAP_SERVER="$WORK_WT/build-vulkan-release/bin/llama-server" \
    VK_GAP_MODEL="$MODEL" VK_GAP_DRAFT="$DRAFT" VK_GAP_PORT=8099 \
      python3 bench/vulkan-gap/repeated_requests.py "$MODE" \
        --np 1 --rounds 5 --prompt "$PROMPT" --gen-tokens 512
  done
done
```

Expected: five valid responses per mode/prompt from one persistent server, with no restart.

- [ ] **Step 5: Run the concurrent `-np 2` gate three times**

```bash
cd "$HARNESS_ROOT"
for MODE in mtp dflash; do
  for ROUND in 1 2 3; do
    VK_GAP_SERVER="$WORK_WT/build-vulkan-release/bin/llama-server" \
    VK_GAP_MODEL="$MODEL" VK_GAP_DRAFT="$DRAFT" VK_GAP_PORT=8099 \
      python3 bench/vulkan-gap/repeated_requests.py "$MODE" \
        --np 2 --rounds 1 --prompt coding --gen-tokens 512 --concurrent
  done
done
```

Expected: both concurrent responses pass their own validator, no cross-slot text appears, DFlash/MTP draft counts are non-zero, and no server exits.

- [ ] **Step 6: Apply the correctness gate**

The work baseline passes only when:

1. All five coding rows have `finish_reason=stop`, compilable Python, a `fibonacci` definition, and no prompt echo.
2. All five math rows contain `2.4` hours and no prompt echo.
3. MTP and DFlash have `draft_n > 0` on every measured row.
4. No row has an HTTP error, empty output, decode error, non-finite timing, or throughput at or above 10,000 t/s.
5. The repeated and concurrent gates pass without restarting the server.

If any work row fails, stop here. Reproduce that single failure, trace target and draft state, add a failing regression test, and make one minimal correctness fix. Rerun all of Task 4 before performance work.

---

### Task 5: Establish the Clean Performance Baseline

**Files:**
- Read: `$Q/correctness/{work,main}/*.jsonl`
- Create: `$Q/performance-baseline.txt`

**Interfaces:**
- Produces proper medians, MADs, ratios, acceptance, and an explicit component gap.

- [ ] **Step 1: Summarize only strictly valid measured rows**

```bash
python3 - "$Q/correctness" > "$Q/performance-baseline.txt" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
for revision in ('work', 'main'):
    for path in sorted((root / revision).glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        valid = [r for r in rows if r.get('measurement', {}).get('valid') is True]
        values = [float(r['measurement']['predicted_per_second']) for r in valid]
        accepts = [float(r['measurement']['draft_accept_pct']) for r in valid
                   if r['measurement'].get('draft_accept_pct') is not None]
        if len(valid) != 5:
            print(revision, path.name, f"INVALID valid={len(valid)}/5")
            continue
        med = statistics.median(values)
        mad = statistics.median(abs(x - med) for x in values)
        accept = statistics.median(accepts) if accepts else None
        print(revision, path.name, f"median_tps={med:.3f}", f"mad={mad:.3f}",
              f"accept_median={accept}", f"min={min(values):.3f}", f"max={max(values):.3f}")
PY
cat "$Q/performance-baseline.txt"
```

Expected: every publishable cell has exactly five valid rows.

- [ ] **Step 2: Check baseline stability**

For any DFlash cell with `MAD / median > 0.05`, repeat that cell twice, alternating execution order:

1. work then main;
2. main then work.

Use the median of all 15 valid measurements only if all three five-row runs pass correctness. Do not discard slow valid rows.

- [ ] **Step 3: Apply the baseline gates**

Record:

- BASE work/main ratio, target range 0.95 to 1.05.
- DFlash work/main ratio for coding and math, target at least 0.90.
- DFlash work/BASE ratio, target at least 1.50.
- DFlash acceptance medians for both revisions.

A BASE ratio outside 0.95 to 1.05 means the problem is broader than speculative orchestration. Investigate build or backend differences before DFlash-specific work.

- [ ] **Step 4: Stop for human review of the baseline**

Report exact SHAs, artifact hashes, valid counts, medians, MADs, ratios, acceptance, finish reasons, and the quarantine path. Do not optimize until the human confirms the baseline is usable.

---

### Task 6: Localize the DFlash Gap With Existing Instrumentation

**Files:**
- Create: `$Q/profile/work/`, `$Q/profile/main/`
- Read: `common/speculative.cpp`, `src/llama-context.cpp`, `tools/server/server-context.cpp`, `TASK_PROGRESS.md`
- Do not modify production code in this task.

**Interfaces:**
- Produces one ranked bottleneck with matching metrics from both clean revisions.

- [ ] **Step 1: Capture existing DFlash summary and verify profiles**

Run one warm-up and one 512-token math request on each clean binary with:

```bash
GGML_DFLASH_PROFILE=summary,verify \
GGML_DFLASH_PROFILE_SYNC_SPLIT=1 \
GGML_DFLASH_QA_TRACE=1 \
VK_GAP_SERVER="$WORK_WT/build-vulkan-release/bin/llama-server" \
VK_GAP_MODEL="$MODEL" VK_GAP_DRAFT="$DRAFT" VK_GAP_PORT=8099 \
  python3 "$HARNESS_ROOT/bench/vulkan-gap/run.py" dflash math \
    --repetitions 1 --gen-tokens 512 \
    --reference-label work-profile --external-source-head "$WORK_SHA" \
    --output "$Q/profile/work/math.jsonl"
```

Repeat with the main binary, port 8100, `main-profile`, and `MAIN_SHA`.

Expected: both requests are valid. Preserve complete server logs.

- [ ] **Step 2: Compare matching component metrics**

Extract for each revision:

- end-to-end generation t/s;
- draft calls and accepted/generated draft tokens;
- draft cross, batch, decode, argmax, and total timing;
- target verify timing and synchronization split;
- rollback/re-eval count and tokens;
- graph reuse count and reuse rate;
- output rows per decode;
- DFlash cross length distribution;
- draft context `n_outputs_max`, configured draft max, active draft max, and block size.

Do not compare a wall-clock number from one revision to a synthetic sum from the other.

- [ ] **Step 3: Recheck the historical output-sizing lead first**

The current evidence showed this work log with CLI `--spec-draft-n-max 8`:

```text
[DFLASH] draft ctx: n_ctx=256 n_outputs_max=17 n_max=16 block_size=16
```

Check the same line on current main and trace these fields:

```bash
git -C "$WORK_WT" grep -n 'common_speculative_n_max\|spec-draft-n-max\|n_outputs_max' -- \
  common/arg.cpp common/speculative.cpp tools/server/server-context.cpp
git -C "$MAIN_WT" grep -n 'common_speculative_n_max\|spec-draft-n-max\|n_outputs_max' -- \
  common/arg.cpp common/speculative.cpp tools/server/server-context.cpp
```

Hypothesis to test: merged parameter semantics make DFlash sizing use `speculative.n_max` while the public CLI writes `speculative.draft.n_max`, so the draft context is sized for 16 instead of the configured 8. Do not call this root cause until the clean profile and parameter trace confirm it.

- [ ] **Step 4: Use Vulkan per-op logging only if summary timing is insufficient**

```bash
GGML_VK_PERF_LOGGER=1 GGML_DFLASH_PROFILE=summary \
  <same clean one-request command>
```

Parse the saved logs offline. Compare graph-compute launch counts, total GPU time, and per-op counts. Do not print the full per-op log into the agent context.

- [ ] **Step 5: Choose exactly one first experiment**

Priority order:

1. Incorrect DFlash draft/output sizing confirmed by Step 3.
2. Missing whole-view reduced verification compared with current main.
3. Excess graph-compute launches or rollback/re-eval.
4. Shared draft batching or adaptive draft-max wiring.

Write one sentence: `I think X is the first root cause because Y metric differs by Z.` Stop for human confirmation before changing code.

---

### Task 7: Run One-Variable Performance Experiments

**Files:**
- Create disposable experiment worktree: `/crypt/tmp/worktrees/beellama-experiment-<sha>/`
- Modify only the component confirmed by Task 6.
- Add the smallest relevant test before production changes.

**Interfaces:**
- Produces one independently reviewable experiment with red/green test evidence and clean A/B results.

- [ ] **Step 1: Create a clean experiment worktree from `WORK_SHA`**

```bash
EXP_WT=/crypt/tmp/worktrees/beellama-experiment-${WORK_SHA:0:9}
git -C "$ROOT" worktree add --detach "$EXP_WT" "$WORK_SHA"
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
  cmake -S "$EXP_WT" -B "$EXP_WT/build-vulkan-release" \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON
```

Expected: clean detached worktree and a successful configuration. Do not copy the quarantined Task 7/8 patch into it.

- [ ] **Step 2: Write and run the failing test**

If output sizing is confirmed, test the invariant that DFlash draft context output capacity follows the configured active draft horizon and remains below or equal to the model block limit. The test must cover CLI draft max 8 with block size 16 and must fail against clean `WORK_SHA`.

If reduced verification is the confirmed first bottleneck, first add model-free tests for:

- whole-view row coverage;
- same top-k across all active speculative slots;
- reduced state restored after the decode view, not after the first slot;
- reduced mode rejected before decode for unsupported sampling;
- two-slot compact row mapping using `absolute_index - view_start`.

Run the exact test and record the expected failure. Do not continue if it passes unexpectedly.

- [ ] **Step 3: Implement the smallest coherent change**

Rules:

- Do not use the failed `n_outputs_max=block_size-1` experiment as a final design. Derive the value from the configured active DFlash horizon and model invariant.
- Do not port only the first-slot reduced sampler from the dirty patch. Reduced mode is a property of one complete decode view and its shared target context.
- Restore reduced-consume state after the complete view and on retry/error/destroy paths.
- Do not combine output sizing, reduced verification, profiling, batching, and adaptive control in one patch.

- [ ] **Step 4: Run the focused test and cheap regression suite**

```bash
cmake --build "$EXP_WT/build-vulkan-release" --target llama-server -j"$(nproc)"
cd "$EXP_WT"
python3 -m unittest bench.vulkan-gap.test_harness_strict bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir build-vulkan-release --output-on-failure \
  -R '^(test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Rerun all correctness gates before measuring speed**

Repeat Task 4 for the experiment binary, including three `-np 2` concurrent rounds. Any correctness failure rejects the experiment.

- [ ] **Step 6: Run an alternating clean A/B performance comparison**

Run five measured requests per prompt in this order:

1. clean work baseline;
2. experiment;
3. experiment;
4. clean work baseline.

Require all 20 rows per prompt to be valid. Report proper median and MAD for each block and the combined baseline/experiment ratio.

- [ ] **Step 7: Apply the experiment gate**

Retain the experiment only when:

- all Task 4 correctness gates pass;
- coding and math improve at least 5%, or the confirmed failing component reaches within 1.20x of main without worsening end-to-end throughput;
- acceptance does not regress by more than 5 percentage points;
- no new graph rebuild, retry, or synchronization pathology appears.

Otherwise revert the experiment in the disposable worktree, record the failed hypothesis, and return to Task 6. Do not stack another change on top.

- [ ] **Step 8: Stop for human review and commit authorization**

Present the focused diff, red/green evidence, full correctness matrix, A/B statistics, and remaining gap. Do not commit until explicitly authorized.

---

### Task 8: Final Clean Verification and Performance Closure

**Files:**
- Create a fresh detached verification worktree at the human-approved candidate SHA.
- Create: `docs/superpowers/results/2026-07-31-vulkan-speculative-clean-baseline-recovery.md` only after results exist.

**Interfaces:**
- Produces final auditable release evidence. It does not push or create a PR.

- [ ] **Step 1: Build the approved candidate from a fresh worktree**

Use the exact Task 2 configure command and an empty build directory. Record executable and shared-library hashes.

- [ ] **Step 2: Run the complete cheap test suite**

```bash
python3 -m unittest bench.vulkan-gap.test_harness_strict bench.vulkan-gap.test_repeated_requests -v
ctest --test-dir build-vulkan-release --output-on-failure \
  -R '^(test-arg-parser|test-speculative|test-sampling|test-server-prompt-checkpoint|test-dflash-ring|test-dflash-plumbing|test-dflash-decode)$'
git diff --check
```

Expected: all commands exit 0 and the verification worktree has no source changes.

- [ ] **Step 3: Run the final work/main six-cell matrix**

Repeat Task 4 with one warm-up plus five measured requests for BASE, MTP, and DFlash on coding and math. Repeat any cell whose `MAD / median` exceeds 0.05.

- [ ] **Step 4: Run final persistent and concurrent gates**

Repeat five persistent requests for MTP and DFlash on coding and math, then three concurrent `-np 2` rounds for each speculative mode.

- [ ] **Step 5: Apply all release gates**

Pass only when:

- every correctness condition in Task 4 passes;
- BASE work/main is within 5%;
- DFlash work/main is at least 90% for coding and math;
- DFlash is at least 1.5x work BASE for coding and math;
- every speculative row has non-zero draft tokens;
- performance records contain exact source SHA, server version, executable and shared-library hashes, model hashes, command, environment, device, finish reason, validity reasons, acceptance, median, and MAD.

- [ ] **Step 6: Request independent review**

Ask the reviewer to check:

1. no result uses the quarantined dirty binary or untracked runner;
2. all measurements come from clean builds at recorded SHAs;
3. multi-slot state is per-slot where required and shared target flags are per-view;
4. no invalid row enters performance statistics;
5. each retained optimization has an isolated A/B result.

- [ ] **Step 7: Write the final results document**

Include exact commands, SHAs, hashes, raw record paths, valid/invalid counts, medians, MADs, ratios, acceptance, open risks, and rejected experiments. Do not claim completion if any gate remains open.

- [ ] **Step 8: Stop for explicit human decision**

The human decides whether to commit any remaining documentation and whether the branch is ready for later manual push. The agent must not push or create a PR.
