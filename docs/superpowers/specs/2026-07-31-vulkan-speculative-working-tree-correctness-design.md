# Vulkan Speculative Working-Tree Correctness Recovery Design

## Goal

Make `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2` correct for Qwen3.6-27B BASE, MTP, and DFlash generation on Vulkan before any performance work resumes.

Correct means that fresh, persistent, and concurrent requests produce valid content, speculative modes use nonzero drafts, speculative token streams preserve target-model behavior at deterministic sampling, and server logs contain no decode, logits, sequence-position, or cross-slot corruption errors.

## Candidate Definition

`Work` is the actual target worktree, not only its committed HEAD:

- Path: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- Starting HEAD: `fb7e5d080f257bb8d15ee68b850cfafe593a7899`
- Starting source modification: `tools/server/server-context.cpp`
- Starting untracked evidence documents: the two Task 4 handoffs and the clean-baseline recovery plan

The starting source diff mixes an opt-in reduced-verification experiment with cycle profiling. It remains present during diagnosis so the working candidate is tested as requested. Correctness runs keep `GGML_DFLASH_REDUCED_VERIFY` unset unless that single variable is under test.

The old source diff was quarantined at:

`/crypt/tmp/beellama-task8-quarantine-20260731T064446Z/`

No stash, reset, clean, commit, or push is part of this recovery.

## Evidence Corrections

The following committed fixes are present and must not be reimplemented blindly:

- `ab3de6854`: early-EOS and stale speculative-state fixes
- `fe92d8a7f`: coordinated cache reset
- `2b85a8780`: unconditional speculative reset through `reset_request()`
- `294a9ff7b`: preserve target output rows during DFlash rollback re-evaluation

The relevant old stashes do not contain a later missing correction:

- `stash@{1}`: rollback/tape/tree changes already present in HEAD
- `stash@{2}`: unconditional diagnostics only
- `stash@{3}`: old Vulkan fallback, superseded by the current D2D cross-ring path

Prior completion claims proved narrower properties than the required gate. The saved `-np 2` script used 32 output tokens and proved no crash plus nonzero drafts. The saved `-np 1` script used 64 output tokens and printed results without failing on invalid coding output. Neither is evidence for five coding requests ending with `finish_reason=stop` and compilable Fibonacci code.

Fresh tests of the actual working-tree binary produced:

- `b512/ub128/ctx2048`, reduced verify off: 1/5 valid DFlash coding
- `b2048/ub512/ctx8192`, reduced verify off: 0/5 valid DFlash coding
- `b2048/ub512/ctx8192`, reduced verify on: 0/5 valid DFlash coding

These records are under:

`/crypt/tmp/beellama-task8-quarantine-20260731T064446Z/correctness/working-tree/`

## Recovery Architecture

The recovery has four layers.

### 1. Trustworthy Harness

Repair model-free defects before another result is trusted:

- remove the invalid `--z` server argument;
- separate prompt kind from speculative mode in validation;
- handle unclosed Markdown fences without raising `ValueError`;
- validate external binary HEAD against `--external-source-head`;
- prevent JSONL append contamination;
- record raw generated token IDs using `return_tokens=true`;
- expose batch, ubatch, and context size as recorded environment settings;
- preserve real process exit codes by never piping benchmark commands through `tail`.

### 2. Staged Reproduction

Do not begin with a large matrix. For BASE, MTP, and DFlash, run:

1. one first request on a fresh server with no warm-up;
2. a second request on that same server;
3. five requests on one persistent server;
4. fresh-server repetitions;
5. concurrent `-np 2` only after single-slot correctness passes.

This separates within-request corruption from request-reset corruption.

### 3. Token-Level Root Cause

Use deterministic sampling (`temperature=0`, `top_k=20`, `seed=7`) and `return_tokens=true`. BASE is the target oracle. Find the first MTP or DFlash token that differs from BASE and align it with existing QA traces:

- draft token;
- target argmax;
- sampled token;
- accepted count;
- rollback count;
- request and slot identity;
- target batch/logits row mapping.

Classify the defect before changing production code:

- first request already diverges: within-request verification, state, or rollback;
- first request passes and later requests diverge: reset or prompt-cache reuse;
- MTP and DFlash fail at the same boundary: common speculative orchestration or sampling;
- only DFlash fails: DFlash ring, cross-attention, rollback, or re-evaluation.

### 4. One-Fix TDD Loop

For each confirmed hypothesis:

1. save the exact pre-change files;
2. preserve an automated failing reproduction;
3. make one minimal production change;
4. build and rerun the focused reproduction;
5. retain the change only if the failure becomes green without a new error;
6. otherwise restore the exact saved files before testing another hypothesis.

Never stack failed experiments. After three failed source hypotheses, obtain a GLM-5.2 architecture review, update the hypothesis ledger, and continue from root-cause investigation rather than attempting a fourth guess.

## Final Source State

After root-cause diagnosis and before the first production fix:

- reverse-apply the exact quarantined starting patch while no accepted fix overlaps it;
- verify `tools/server/server-context.cpp` matches committed HEAD;
- implement the accepted correctness fix on that clean source base;
- build from an empty Release build directory after all gates pass;
- rerun all model-free, CTest, fresh, persistent, and concurrent gates.

The experimental patch remains recoverable from quarantine. No performance optimization, commit, push, or PR is included.

## Correctness Gates

Every final five-row cell must satisfy:

- coding: 5/5 `finish_reason=stop`, compilable Python, `def fibonacci`, no prompt echo;
- math: 5/5 contains `2.4`, no prompt echo, no coding cross-contamination;
- MTP and DFlash: `draft_n > 0` for every row;
- finite positive throughput below 10,000 t/s;
- no HTTP failure, empty output, decode error, invalid logits, assertion, server exit, or non-consecutive token-position warning;
- fresh deterministic MTP and DFlash token IDs equal fresh BASE token IDs for the same prompt and sampling settings;
- five persistent single-slot requests pass without server restart;
- three concurrent `-np 2` rounds pass for both MTP and DFlash with no cross-slot content.

Correctness must pass under both configurations:

- smoke: `-b 512 -ub 128 --ctx-size 2048`;
- matrix: `-b 2048 -ub 512 --ctx-size 8192`.

## Continuous Execution and Context Control

A Qwen3.6-27B implementation agent executes sequentially with `superpowers:executing-plans`. It does not stop at routine milestone reviews and does not wait for human permission between work packages.

At every milestone it:

1. writes a concise evidence report;
2. asks `glm-5.2:cloud` for an independent review and an authoritative continuation summary;
3. loads the GLM continuation summary into context;
4. calls `smart_compact` when actual context usage is high, focusing on that summary;
5. continues with the next package.

The GLM review is advisory. Repository evidence and tests remain authoritative. If Ollama is unavailable, the agent records the failure and continues.

The agent stops only for an unrecoverable external blocker: missing model files, unavailable Vulkan device, filesystem failure, or inability to restore a rejected experiment. A test failure is not a stop condition; it enters the documented investigation loop.
