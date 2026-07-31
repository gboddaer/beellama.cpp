# Vulkan Speculative Correctness Second-Pass Design

## Status

Approved diagnostic order from the human:

1. Return to root-cause diagnosis with explicit slot isolation and first-token tracing.
2. Compare per-sequence removal with a full target-memory data clear.
3. Diagnose DFlash rollback re-evaluation separately.
4. Test fresh sequence identities as a diagnostic, not as a production design.
5. Consider fresh sequence IDs as production only if the preceding evidence supports them.
6. Modify recurrent-memory reset behavior only if direct evidence proves it defective.
7. Do not accept the partial fix or begin performance work while correctness gates fail.

## Goal

Find and fix the first causal state transition that makes persistent MTP and DFlash requests diverge from a fresh BASE request, while separately removing DFlash duplicate-position rollback behavior. Finish with all fresh, persistent, and concurrent correctness gates passing on Vulkan.

## Corrected Starting Facts

- Candidate HEAD is `d0e3ca514248` on `merge_llama_into_beellama_2`.
- The worktree was clean when this design was written.
- `d0e3ca514` added `prompt_clear(false)` to speculative `server_slot::reset()`.
- That hunk is experimental. It did not pass the required 512-token persistent matrix.
- BASE passed the observed 512-token coding and math cells.
- MTP and DFlash still produced invalid later-request output.
- Bad later requests often diverged at generated token index 0.
- The first generated token is sampled from target prompt-final logits before speculative drafting begins.
- Full `llama_memory_recurrent::seq_rm(seq_id, -1, -1)` resets sequence metadata, rollback index, tail ownership, and empty-cell positions.
- DFlash duplicate-position warnings occur in request 1 and therefore are not proof of cross-request leakage.
- Existing QA chronology places DFlash duplicate-position warnings inside rollback re-evaluation.
- MTP has persistent corruption without those warnings.

## Problem Decomposition

### Track A: Shared request-entry corruption

MTP and DFlash can sample a wrong first token after prompt prefill on a reused server. Since this occurs before drafting, investigate the target prompt-final output path first:

- physical slot identity;
- target sequence identity;
- target recurrent and attention memory state;
- prompt-final batch and output-row mapping;
- sampler input and target argmax;
- target hidden/output buffers shared by speculative modes.

### Track B: DFlash rollback re-evaluation

DFlash restores a backup, then Vulkan fallback re-evaluates accepted positions. Existing traces show re-evaluation at a position equal to the restored recurrent tail position. Determine whether backup position, replay count, or re-evaluation start is off by one. Do not use this defect to explain MTP.

## Diagnostic Architecture

### Explicit slot schedule

Use one persistent server and send identical requests to a controlled physical-slot schedule:

```text
np=2: 0,1,0,1
np=4: 0,1,2,3,0,1,2,3
```

This distinguishes first use from reuse without introducing a production sequence allocator.

### First-token evidence

For every request record:

- request ordinal;
- physical slot ID;
- mode and prompt;
- first returned token ID;
- target argmax for prompt-final logits;
- resolved output row;
- target sequence position before prefill and after prefill;
- complete raw token IDs and validator result;
- draft count;
- exact binary, source, model, configuration, and server-log hashes.

### Full-clear discriminator

Under an environment-gated, `-np 1` diagnostic build, compare normal sequence removal with `llama_memory_clear(llama_get_memory(ctx_tgt), true)` before each request. Restore source exactly after the experiment.

A full clear becoming GREEN supports stale target memory data or initialization. A full clear remaining RED rejects recurrent/attention data persistence as the shared cause.

### DFlash rollback evidence

Record these positions for each partial acceptance:

- backup sequence position;
- active target position before restore;
- active target position after restore;
- `n_past_before`;
- `n_hidden_keep`;
- `n_reeval`;
- first and last re-evaluation positions;
- warning tuple and chronology.

## Production Decision Rules

A production change is allowed only when one diagnostic changes the first-divergence boundary exactly as predicted.

### Fresh sequence IDs

Do not implement fresh sequence IDs merely because unused slots pass. Consider them only if all are true:

1. every first-use sequence passes and every reused sequence fails;
2. full data clear restores fresh behavior;
3. non-memory speculative resets do not restore behavior;
4. the defect is proven to be keyed by target sequence identity;
5. a bounded mapping preserves `slot.id` routing, DFlash backup IDs, concurrency, and memory limits.

### Recurrent core

Do not modify `llama-memory-recurrent.cpp` unless metadata and state evidence shows `seq_rm` violates its documented postconditions or initializes a new sequence from nonzero data.

### Shared request-entry fix

Prefer a fix at prompt-final state or output-row initialization if tracing shows the first bad argmax is read from the wrong row or stale output state.

### DFlash rollback fix

Prefer a local rollback/re-evaluation correction after proving which position is duplicated. Do not combine it with the shared request-entry fix.

## Verification Gates

The final candidate must satisfy the original correctness plan:

- model-free harness tests pass;
- selected CTests pass;
- fresh MTP and DFlash coding tokens equal the same fresh BASE oracle at deterministic settings;
- 5/5 persistent coding and math rows validate for every mode;
- every speculative row has `draft_n > 0`;
- smoke and matrix configurations pass;
- three concurrent coding-plus-math rounds pass for MTP and DFlash;
- no forbidden log warning, including non-consecutive token positions;
- a fresh Release build reproduces all results;
- the final diff contains only accepted RED/GREEN changes and required tests/harness work.

## Process

- Use one hypothesis and one diagnostic variable at a time.
- Restore every rejected or temporary diagnostic from a byte-identical snapshot.
- Run an independent GLM-5.2 review after every milestone.
- Treat the GLM continuation summary as the authoritative compact guide.
- Use `smart_compact` after milestone review when actual context usage is high. Never use the tool-output percentage as a context estimate.
- Do not commit, push, or begin performance work without new explicit human authorization.
