# DFlash Main Corrected Reference Design

Date: 2026-08-02

## Purpose

Create and freeze `main-corrected-reference` from `130ea2480` by backporting only the minimum speculative correctness behavior needed for valid persistent and concurrent BASE, MTP, and DFlash measurements. Preserve `130ea2480` as the immutable historical performance anchor and do not import unrelated work-branch performance optimizations.

## Branches and Anchors

```text
historical reference: 130ea2480bee8268907a102bd814e276f4e88bdf
corrected branch: main-corrected-reference
corrected worktree: /crypt/beellama.cpp/.worktrees/main-corrected-reference
work control: 986ce7aed9b30a00145422c1cbb75da9a6557d05
active work branch: merge_llama_into_beellama_2
```

The corrected branch is pushed separately to `gboddaer` and `boditec`. No PR is created.

## Problem

At `130ea2480`, first-request MTP and DFlash output is valid, but the persistent harness sends a warmup on the same server before measured requests. Stale speculative prompt, KV, ring, embedding, and recurrent state then corrupt later requests into prompt echo. This makes the pinned reference invalid for persistent and concurrent comparison.

Fresh diagnostic evidence under the fixed protocol showed:

```text
reference MTP first request:    valid, 22.62 t/s
reference DFlash first request: valid, 30.85 t/s
reference MTP after warmup:     invalid prompt_echo
reference DFlash after warmup:  invalid prompt_echo
```

## Experimental Roles

### Historical reference

`130ea2480` remains unchanged and records the original fast first-request behavior. It is not used as a persistent correctness reference.

### Corrected reference

`main-corrected-reference` is the operational independent target. It must pass first-request, persistent, and concurrent correctness before any result is accepted.

### Work control

Clean `986ce7aed` is used only to confirm the intended corrected behavior and deterministic output. It is not the independent performance target.

## Backport Strategy

Use a staged minimal backport instead of merging all 471 intervening commits.

1. Reproduce the persistent failure on the unmodified reference.
2. Add the minimum coordinated request reset and prompt-cache bypass needed to prevent stale state reuse.
3. Reset draft KV and per-request MTP/DFlash state at speculative launch.
4. Add target-context idle serialization and target memory clearing only if concurrent or repeated requests still contaminate state.
5. Port EOS handling or rollback failure propagation only when a targeted failing test demonstrates that the corrected reference needs it.

Each stage is correctness-gated. Stop adding behavior as soon as every qualification gate passes. Do not port DFlash graph, copy, replay, verifier, scheduler, Vulkan queue, or draft-performance changes.

## Allowed Production Files

The expected minimal source surface is:

```text
common/speculative.cpp
common/speculative.h
tools/server/server-context.cpp
```

The following files may change only if a targeted failure proves they are required:

```text
common/sampling.cpp
src/llama-context.cpp
src/llama-model.cpp
```

Any other production file requires fresh human approval.

## Fixed Protocol

```text
target model: /crypt/models/Qwen3.6-27B-Q4_K_M.gguf
draft model: /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf
device: Vulkan0 / RADV GFX1151
batch: 512
ubatch: 128
context: 2048
parallel slots for throughput: 1
temperature: 0
top_k: 20
seed: 7
cache K/V: q4_0 / q4_0
DFlash draft max: 8
DFlash cross context: 512
branch budget: 0
GPU ring: disabled
request cache_prompt: false
server prompt cache: enabled
throughput generation ceiling: 256
concurrent correctness generation ceiling: 512
```

No protocol tuning is allowed during qualification.

## Qualification Matrix

### Build and model-free gates

- `git diff --check` passes.
- Vulkan0/GFX1151 is selected.
- `llama-server`, `test-dflash-ring`, and `test-dflash-plumbing` build.
- Every DFlash CTest present at `130ea2480` passes.
- Strict harness and repeated-request unit tests pass from the work-control harness.

### Historical first-request control

Reproduce valid MTP and DFlash first requests from unmodified `130ea2480`. Record original binary and output hashes.

### Corrected first-request and persistent gates

For corrected BASE, MTP, and DFlash:

- five independent first-request coding controls;
- five measured coding requests on one persistent server after warmup;
- every row valid;
- nonzero speculative draft counts for MTP and DFlash;
- corrected first-request token and content hashes are deterministic within each mode;
- persistent rows are all valid and rows 2-5 are token/content-hash-identical within each mode;
- first persistent row transitions and historical/corrected hash differences are reported and are not hidden by length or duration filtering;
- no prompt echo, early EOS failure, decode error, recurrent-removal failure, or server crash.

### Concurrent gates

For corrected MTP and DFlash:

- `np=2`, three concurrent coding-plus-math rounds;
- six structurally valid responses per mode;
- no cross-request text or token contamination;
- nonzero speculative draft counts;
- no target-state corruption or server failure.

Math semantic validity remains separately reported. Hash-equivalent nonempty math output is state-equivalence evidence and does not override the final all-output-valid objective.

### Performance gates

Record medians for historical first-request, corrected first-request, and corrected persistent BASE/MTP/DFlash.

The corrected reference is accepted only if:

```text
corrected/historical BASE ratio is between 0.95 and 1.05
corrected/historical MTP ratio is between 0.95 and 1.05
corrected/historical DFlash ratio is between 0.95 and 1.05
corrected DFlash median >= 28.457 t/s
corrected DFlash median >= 1.5x corrected BASE
corrected BASE, MTP, and DFlash persistent outputs are valid
corrected MTP and DFlash concurrent outputs are contamination-free
```

Historical and corrected control medians use five independent first-request processes. Corrected operational medians use five measured requests after warmup on one persistent server and must independently satisfy the absolute DFlash gates. Any gate miss stops the freeze for human review.

## Freeze Artifacts

The qualification evidence records:

```text
historical and corrected commit SHAs
source diff and SHA-256
server and shared-library SHA-256 values
model and draft SHA-256 values
compiler and Vulkan device provenance
all raw JSONL, manifests, and server logs
correctness summaries
throughput medians and ratios
paired token/content hashes
concurrent contamination checks
final git status
```

The corrected branch receives one concise source commit and one qualification-record commit only after all gates pass. Both commits use `Assisted-by: OpenAI`.

## Push Rules

After fresh verification and explicit gate success:

```text
git push gboddaer main-corrected-reference
git push boditec main-corrected-reference
```

Do not push to `origin`. Do not create a PR.

## Stop Conditions

Stop without commit or push if:

- the historical or work-control SHA changes;
- source changes exceed the approved production-file surface;
- Vulkan0/GFX1151 is not selected;
- build or model-free tests fail;
- any corrected BASE, MTP, or DFlash coding row is invalid;
- deterministic coding hashes diverge;
- concurrent contamination appears;
- a decode, rollback, or recurrent-state operation fails;
- corrected BASE leaves the 0.95 to 1.05 historical ratio;
- corrected DFlash performance materially regresses without human review;
- the branch contains unrelated source changes.

## Final Objectives Preserved

```text
work DFlash median >= 28.457 t/s
work DFlash median >= 1.5x work BASE
work/corrected-reference BASE ratio between 0.95 and 1.05
BASE and MTP correctness and performance preserved
all measured BASE, MTP, and DFlash outputs valid
final BASE, MTP, and DFlash persistent gates pass on one server without restart
final MTP and DFlash concurrent gates pass without cross-request text or token contamination
```
