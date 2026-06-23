# Enable and Fix DFlash on Vulkan — Various Models

This document is the canonical rationale for **enabling and fixing DFlash on the
Vulkan backend across the model set** tested on AMD Strix Halo (Radeon 8060S /
RADV GFX1151, Mesa 25.0.7, Vulkan 1.4.309, 96 GB UMA). It consolidates the
original Qwen3-Coder-Next enablement work with the later multi-model fixes
(Qwen3.5-122B-A10B GPU cross-ring, Qwen3.6-35B-A3B stochastic-sampling
corruption) and the per-model status/performance for Qwen3.6-27B and Gemma 4.

It replaces the scattered investigation notes that were created while debugging
the original 0% draft-token acceptance problem and the later qwen35moe
recurrent-state regressions.

## Summary

DFlash on Vulkan was enabled and stabilized across a range of target/drafter
pairs. Each model exposed a distinct issue; the fixes are layered and do not
conflict.

| Target model | Arch | Drafter | Original Vulkan status | Fix |
|---|---|---|---|---|
| Qwen3-Coder-Next | `qwen3next` (MoE) | qwen3-coder-next-dflash | 0% acceptance | Drafter graph target-KV branch gate + Qwen3Next recurrent-state tape capture + full-block flat drafting |
| Qwen3.6-27B | `qwen35` (dense) | Qwen3.6-27B-DFlash | worked | (no fix needed; performance characterized) |
| Qwen3.5-122B-A10B | `qwen35moe` | qwen35-122b-a10b-dflash | GPU cross-ring recurrent-state desync under `n_rs_seq=0` | Keep `n_rs_seq` for QWEN35MOE (RS partial rollback) instead of clamping to 0 |
| Qwen3.6-35B-A3B | `qwen35moe` | qwen36-35b-a3b-dflash | gibberish under stochastic sampling (0% acceptance) | Rotate DeltaNet RS snapshots for `n_seq_tokens < K` |
| Gemma 4 31B | `gemma4` | gemma4-31b-it-dflash | worked; "own own own" under greedy, coherent under `--reasoning on`/temp=1.0 | (no runtime fix needed; sampling guidance) |

The two decisive runtime fixes layered on top of the coder-next enablement are:

1. **Keep `n_rs_seq` for QWEN35MOE** (122B): under the GPU cross-ring,
   clamping `n_rs_seq=0` forced checkpoint-restore for draft rejection and the
   target's DeltaNet recurrent-state cells desynchronized from committed token
   positions (`non-consecutive token position` warnings) → wrong drafts accepted
   → repetition. Keeping `n_rs_seq = draft.n_max` (RS partial rollback) makes both
   ring=0 and ring=1 coherent.

2. **Rotate DeltaNet RS snapshots for `n_seq_tokens < K`** (35B-A3B): the
   `n_rs_seq>0` write-back loop in `build_recurrent_attn` wrote all `K`
   `gdn_out` snapshot slots into the recurrent-state tensor every step, but for a
   decode / short batch the gated_delta_net kernel only populates the last
   `min(n_seq_tokens, K)` slots; the leading slots are uninitialised. Under
   stochastic sampling the adaptive draft-max controller drives spec depth to 0
   once acceptance drops, so every step becomes a 1-token decode → snapshot slots
   `1..K-1` stay garbage → repeated draft-rejection rollbacks read stale
   recurrent state → output degrades to multilingual gibberish. The fix writes
   only the populated slots and rotates the older snapshots down.

## Test setup

- Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S, 96 GB UMA, Vulkan 1.4.309 (RADV).
- Build: Vulkan Release, `GGML_VULKAN=ON`. Vulkan has no DFlash GPU cross-ring
  and no TurboQuant/TCQ cache types, so tests use `--cache-type-k q4_0
  --cache-type-v q4_1` and `GGML_DFLASH_GPU_RING` to select ring mode.
- Server: `llama-server`, single-sequence DFlash (`-np 1`), `-ngl all`,
  `--flash-attn on`, `--no-cache-prompt`, `--device Vulkan0`.
- DFlash args: `--spec-type dflash --spec-draft-n-max 4 --spec-dflash-cross-ctx
  1024 --spec-branch-budget 0 --spec-draft-ngl all --spec-draft-device Vulkan0`.
- `GGML_DFLASH_GPU_RING=0` → CPU hidden capture (ring=0); `=1` → GPU cross-ring
  (ring=1), which on main falls back to CPU because main has no GPU ring code.

## Post-fix performance (greedy, 256 tokens, PR fixed, all rep=0)

| Target model | baseline tok/s | DFlash ring=0 (acc) | DFlash ring=1 (acc) |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B | 52.04 | 75.60 (0.827) | 76.88 (0.828) |
| Qwen3.6-27B (dense) | 12.47 | 29.80 (0.835) | 29.94 (0.816) |
| Qwen3-Coder-Next (MoE) | 53.26 | 34.07 (0.754) | 50.09 (0.621) |
| Qwen3.5-122B-A10B (MoE) | 13.92 | 8.20 (0.189) | 14.92 (0.256) |

Interpretation:

- **Qwen3.6-35B-A3B**: after the snapshot-rotation fix, DFlash is both coherent
  **and 1.45–1.48× faster** than baseline (52 → 76 tok/s) with 0.83 acceptance.
  ring=0 ≈ ring=1 on Vulkan for this model.
- **Qwen3.6-27B (dense)**: strongest DFlash win — **2.4×** (12.5 → 30 tok/s),
  acceptance 0.82. ring=0 ≈ ring=1.
- **MoE targets (Coder-Next, 122B)**: DFlash is net-slower to ~neutral, matching
  the known Vulkan/MoE pattern — cheap active decode means drafter overhead
  dominates. ring=1 is at or slightly above baseline; ring=0 is slower (CPU
  capture overhead). These were already coherent; the fixes preserve that and
  make their stochastic-sampling paths coherent too.
- **Gemma 4 31B** (`gemma4`, no DeltaNet recurrent state) is unaffected by the
  recurrent-state fixes: ring=0 ~26.5 tok/s, ring=1 ~21.4, both coherent under
  `--reasoning on`/temp=1.0. On Vulkan, ring=0 (CPU capture) beats ring=1 for
  Gemma 4's large `n_embd` (5376), consistent with the quickstart guidance.

## Qwen3-Coder-Next enablement (`qwen3next` MoE)

Qwen3-Coder-Next DFlash originally produced near-0% accepted draft tokens on
Vulkan. The failure was not caused by a generic DFlash verifier bug: Qwen3.6
DFlash worked, and Qwen3-Coder-Next could work under other implementations. The
issue was a chain of Qwen3Next-specific runtime and drafter-input mismatches.

The decisive Vulkan runtime fix is:

- Vulkan does not have the DFlash GPU cross ring.
- Without the GPU cross ring, the drafter's normal full-attention KV cache is not
  populated with target-context K/V.
- The Qwen3-Coder-Next drafter was still taking the full-attention KV-cache
  branch, reading empty/stale context.
- The DFlash drafter graph now only builds that branch when target KV is actually
  available. On Vulkan it falls back to projecting K/V freshly from CPU-captured
  `target_hidden`.

With the runtime fixes and a drafter trained for the live C++ position semantics,
in-distribution acceptance was observed at roughly 68-72% instead of 0%.

### User-facing command shape

The working Vulkan launch shape is:

```bash
build-vulkan/bin/llama-server \
  -m /crypt/models/Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf \
  --spec-draft-model /crypt/tmp/zlab_finetuned_pos1_f16.gguf \
  --spec-type dflash \
  --spec-dflash-cross-ctx 512 \
  --spec-draft-n-max 4 \
  --spec-branch-budget 0 \
  -np 1 \
  --kv-unified \
  -ngl all \
  --spec-draft-ngl all \
  -b 2048 \
  --ctx-size 8192 \
  --flash-attn on \
  --jinja \
  --temp 0.0 \
  --top-k 1
```

The exact drafter file matters. The runtime fix enables the correct Vulkan path,
but a drafter trained against the wrong input/position semantics can still show
poor acceptance.

### Root-cause chain

#### 1. Qwen3Next conv state was not advanced during DFlash rollback

Qwen3-Coder-Next has recurrent / convolutional state. During DFlash verification
and rollback, accepted tokens must advance the target recurrent state. The tape
replay path had to reconstruct both DeltaNet state and convolution state.

The Qwen3Next graph emits the pre-conv QKV tensor as:

```text
linear_attn_qkv_mixed-<layer>
```

with a legacy path named:

```text
qkv_mixed-<layer>
```

The DFlash tape map did not capture these names, so `tape_replay_conv()` skipped
every Qwen3Next recurrent layer. The conv state `r_l` remained frozen at the
pre-draft backup state. That could produce garbled target output even when no
draft tokens were accepted.

Fix:

```cpp
dflash_capture->tape_name_map["linear_attn_qkv_mixed-" + il_str] = {idx, DFLASH_TAPE_QKV};
dflash_capture->tape_name_map["qkv_mixed-" + il_str]             = {idx, DFLASH_TAPE_QKV};
```

This made the conv replay path see the Qwen3Next pre-conv data and advance `r_l`
correctly.

#### 2. Vulkan lacked a populated drafter full-attention KV cache

DFlash keeps recent target hidden states in a cross-attention ring. On CUDA, the
GPU cross ring can also populate the drafter-side full-attention KV cache with
target-context K/V.

On Vulkan, the GPU cross ring is unavailable, so DFlash uses CPU hidden capture:

```text
gpu_ring_handle == nullptr
```

In that case, `update_drafter_kv_cache()` returns early and the drafter
full-attention KV cache is not populated with target context.

However, the DFlash drafter graph still built `inp_attn_kv_full` whenever the
drafter had a memory context. For all-full-attention drafters, this selected the
branch that reads target context from the normal KV cache:

```cpp
if (inp_attn_kv_full && !hparams.is_swa(il)) {
    // read target context from drafter KV cache
}
```

On Vulkan that cache was empty or stale. The drafter therefore generated from
little or no real target context, producing repeated bad predictions and 0%
acceptance.

Fix: add an explicit context parameter:

```cpp
bool dflash_target_kv_available = false;
```

and a public setter:

```cpp
llama_set_dflash_target_kv_available(ctx_dft, gpu_ring_handle != nullptr);
```

Then build the full-attention KV input only when the target KV cache is actually
populated:

```cpp
llm_graph_input_attn_kv * inp_attn_kv_full = nullptr;
if (mctx && cparams.dflash_target_kv_available) {
    inp_attn_kv_full = dflash_build_base_attn_input(...);
}
```

When the flag is false, the graph uses the existing fresh-projection path:

```cpp
Kcur_ctx = wk(fused_target)
Vcur_ctx = wv(fused_target)
```

That path works with CPU-captured `target_hidden`, which is the correct Vulkan
fallback.

#### 3. Flat DFlash must keep the full query block

DFlash query-token attention is non-causal over the draft block. Shortening the
query block can change logits, including early positions.

The flat path previously used:

```cpp
batch_len = n_draft + 1;
```

Tree DFlash already used the full block. The flat and batched flat paths now
keep the trained/inferred block shape:

```cpp
batch_len  = block_size;
output_len = n_draft + 1;
```

Only `output_len` rows are marked for output and consumed by the sampler. This
preserves full-block drafter semantics while still limiting active draft horizon.

#### 4. Runtime parity required comparing exact live C++ inputs

The debugging path added several diagnostics that made it possible to prove where
the mismatch lived (see the diagnostics table below). The cross dump showed that
after the Vulkan KV branch fix, C++ and PyTorch forward passes matched for the
same live input. That narrowed the remaining 0% behavior to drafter
training/input semantics rather than a C++ graph mismatch.

#### 5. Drafter position semantics matter

C++ consumes DFlash predictions starting at row / position 1. Position 0 is the
known root / bonus token. A drafter trained to predict from position 0 can
appear valid in a standalone training loop while still producing poor live C++
acceptance, because C++ reads position 1. Training the live-consumed position
fixed the in-distribution acceptance collapse once the runtime graph was correct.

This document does not cover broader drafter generalization work; that is a
separate training/data problem.

## Qwen3.5-122B-A10B GPU cross-ring fix (`qwen35moe`)

### Symptom

122B DFlash under the GPU cross-ring (`GGML_DFLASH_GPU_RING=1`) produced
repetition (`rep=12`) with dozens of `non-consecutive token position N after N`
warnings in `llama-memory-recurrent.cpp`. ring=0 (CPU capture) was coherent.

### Root cause

`need_n_rs_seq() == draft.n_max` (e.g. 4) requests per-token recurrent-state
snapshots so the target can roll back after a partial draft rejection. An earlier
guard clamped `n_rs_seq=0` for QWEN35MOE ("DeltaNet RS-rollback snapshot bug
under split_equal"). With `n_rs_seq=0`, draft rejection falls back to
checkpoint-restore, but under the GPU cross-ring the target's DeltaNet
recurrent-state cells desynchronize from committed token positions → wrong drafts
accepted → repetition.

### Fix

In `src/llama-context.cpp`, do not clamp `n_rs_seq` to 0 for QWEN35MOE — keep it
as requested by DFlash (`need_n_rs_seq() == draft.n_max`). The QWEN35 dense path
(Qwen3.6-27B, `n_rs_seq=8`) was never clamped and already works.

### Caveat resolved by the 35B-A3B fix

The 122B fix's comment originally said the `n_rs_seq>0 → garbage` note was "not
reproduced in the single-sequence DFlash path". That was true for 122B under
greedy sampling but **false for Qwen3.6-35B-A3B under stochastic sampling**,
which exposed a separate DeltaNet snapshot write-back bug (see next section).
That bug is now fixed, so `n_rs_seq>0` is correct for QWEN35MOE under both greedy
and stochastic sampling.

## Qwen3.6-35B-A3B stochastic-sampling fix (`qwen35moe`)

### Symptom

35B-A3B DFlash produced multilingual gibberish (`"Here's a thinking
fertig是否经济的 whether puppermodele..."`) with **0.000 acceptance** and an HTTP
500 parse error under stochastic sampling (`--temp 1.0 --top-k 64 --top-p 0.95`,
the quickstart setting), for both `--reasoning on` and `--reasoning off`, and
for both ring=0 and ring=1. Greedy (`--temp 0`) was coherent.

### Root-cause investigation (evidence-based, gdb)

The 122B fix removed the QWEN35MOE `n_rs_seq` clamp, so 35B-A3B ran with
`n_rs_seq=4`. gdb on `llama_memory_recurrent::seq_rm` / `set_rs_idx` showed:

- Greedy (coherent): 10 rollbacks over 24 tokens, snapshot slot indices
  scattered (1, 3, 4, 2).
- Stochastic (garbage): 17 rollbacks over 32 tokens, **`set_rs_idx idx=2` on
  nearly every step** after the first rejection.

The differentiator is **stochastic sampling**, not reasoning mode, length, or
ring wrap (confirmed by isolation tests): greedy → coherent at 24/1024/2048
tokens; stochastic → garbage at 128 tokens.

### The bug

In `build_recurrent_attn` (`src/models/delta-net-base.cpp`), the `keep`
(`n_rs_seq>0`) write-back loop copies all `K` `gdn_out` snapshot slots into
`ssm_states_all` every step:

```cpp
for (int64_t k_i = 0; k_i < K; ++k_i) {
    const uint32_t cache_slot = (uint32_t) (K - 1 - k_i);
    ggml_tensor * src = ggml_view_4d(ctx0, gdn_out, ..., attn_score_elems + k_i * state_size_per_snap);
    ggml_tensor * dst = ggml_view_2d(ctx0, ssm_states_all, ..., cache_slot * ...);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
}
```

The gated_delta_net kernel only writes the last `min(n_seq_tokens, K)` snapshot
slots of `gdn_out` (slots `(K-n_seq_tokens) .. (K-1)`); the leading slots are
**uninitialised** (caller-owned). For a **decode / short batch**
(`n_seq_tokens < K`) the loop writes that uninitialised data into state slots
`1..K-1` on every step.

Under DFlash with stochastic sampling, acceptance drops → the adaptive draft-max
controller drives spec depth to 0 → **every step becomes a 1-token decode**
(`n_seq_tokens=1`) → snapshot slots `1..K-1` stay garbage → repeated
draft-rejection rollbacks (`idx=2`) read stale recurrent state → output degrades
to gibberish. Greedy / high-acceptance runs were unaffected because verify
batches (`n_seq_tokens = K`) overwrite the corruption with valid data.

### The fix

Write only the kernel-populated `gdn_out` slots (newest `n_pop` states → state
slots `0..n_pop-1`) and **rotate** the older snapshots down by `n_pop` slots
(old slot `s` → slot `s + n_pop`) so the recurrent-state history stays correct
for rollback:

```cpp
const int64_t n_pop = (n_seq_tokens < K) ? n_seq_tokens : K;
for (int64_t k_i = 0; k_i < K; ++k_i) {
    const uint32_t cache_slot = (uint32_t) (K - 1 - k_i);
    ggml_tensor * dst = ggml_view_2d(ctx0, ssm_states_all, ..., cache_slot * ...);
    if (k_i >= K - n_pop) {
        // gdn_out slot k_i holds the state after one of this batch's tokens.
        ggml_tensor * src = ggml_view_4d(ctx0, gdn_out, ..., attn_score_elems + k_i * state_size_per_snap);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
    } else {
        // gdn_out slot k_i is uninitialised (n_seq_tokens < K): rotate the older
        // snapshot down so slot history is preserved for rollback.
        const uint32_t old_slot = cache_slot - (uint32_t) n_pop;
        ggml_tensor * src = ggml_view_2d(ctx0, ssm_states_all, ..., old_slot * ...);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
    }
}
```

Iterating `k_i` ascending reads each old slot before it is overwritten, so the
in-place shift is safe (copies to the same persistent buffer are serialised by
the scheduler). For `n_seq_tokens >= K` the behaviour is unchanged (full verify
batch). The `!keep` path (`n_rs_seq=0`, e.g. Qwen3-Coder-Next) is untouched.

### Verification

Post-fix (PR `efb07137c`), all coherent (`rep=0`):

| model | arch | n_rs_seq | greedy r1 | stochastic r1 |
|---|---|---|---|---|
| 35B-A3B | qwen35moe | 4 | 0.788 acc ✓ | **0.000→0.711 acc FIXED** ✓ |
| 122B | qwen35moe | 4 | 0.229 acc ✓ (no reg) | **0.209 acc (bonus)** ✓ |
| qwen3.6-27B | qwen35 dense | 8 | 0.771 acc ✓ | 0.519 acc ✓ |
| Coder-Next | qwen3next | 0 | 0.621 acc ✓ | 0.579 acc ✓ (unaffected) |

The fix turns 35B-A3B from broken (gibberish under stochastic) into a 1.48×
speedup, and makes 122B's stochastic path coherent as a bonus, with **zero
regressions** across all models.

## Gemma 4 31B (`gemma4`)

Gemma 4 31B DFlash works on the PR (coherent C++ code under `--reasoning on`,
temp=1.0, top-k=64, top-p=0.95) and crashes on main (DFlash `std::out_of_range`
vocab index). It has no DeltaNet recurrent state, so it is unaffected by the
recurrent-state fixes. On Vulkan, ring=0 (CPU capture) beats ring=1 (GPU ring)
for Gemma 4's large `n_embd` (5376), consistent with the quickstart's "Vulkan:
not recommended for DFlash, falls back to CPU ring" guidance. The "own own own"
loop seen under greedy/raw-prompt sampling is model greedy behavior (identical on
main baseline and PR baseline), not a DFlash bug — use `temp>0` for coherent
output.

## Ruled-out or deprioritized hypotheses

### (coder-next) k_norm ordering

Suspected mismatch around whether target K normalization happened before or
after concatenating context/noise K. Not the cause of 0% acceptance; the
attempted k_norm fix was reverted.

### (coder-next) Basic cross-ring emptiness

Diagnostics showed the cross window was not generally empty. The real problem
was full-attention layers reading target context from an unpopulated drafter KV
cache on Vulkan instead of using the fresh `target_hidden` projection path.

### (coder-next) Generic DFlash verifier failure

Qwen3.6 DFlash worked, and the server verification path could reject all drafts
while still producing coherent target output when recurrent state was handled
correctly. A global verifier failure was unlikely.

### (coder-next) C++ vs PyTorch graph mismatch after KV-branch fix

After dumping the exact live hidden-state window and replaying it in PyTorch, C++
and PyTorch forward outputs matched. The C++ graph was not the remaining source
of the live acceptance issue after the Vulkan KV fix.

### (35B-A3B) 122B-style RS desync

The 122B smoking gun (`non-consecutive token position` warnings + `n_rs_seq=0`)
was **absent** in the broken 35B-A3B run (0 warnings, `n_rs_seq=4`) and **present**
in the working coder-next run (22 warnings, `n_rs_seq=0`). The RS rollback path
is clean in 35B-A3B; the corruption is in the snapshot write-back, not the
rollback/restore. Confirmed by gdb: no `RS-ROLLBACK-OVERFLOW`, no non-consec
warnings, but repeated `set_rs_idx idx=2` reading stale slots.

### (35B-A3B) Reasoning mode / generation length / ring wrap as the trigger

Isolation tests disproved all three: reasoning ON + greedy = coherent;
reasoning OFF + stochastic = garbage; 2048-token greedy (wraps past
`cross_ctx=1024`) = coherent. The sole trigger is **stochastic sampling +
`n_rs_seq>0`**, because it drives the adaptive controller to spec depth 0
(all-decode steps).

## Relevant code changes

### Public API

`include/llama.h`:

```cpp
LLAMA_API void llama_set_dflash_target_kv_available(struct llama_context * ctx, bool avail);
```

Debug-only recurrent state dump API:

```cpp
LLAMA_API void llama_dflash_dump_recurrent_state_dbg(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        const char *           tag);
```

### Context parameter

`src/llama-cparams.h`:

```cpp
bool dflash_target_kv_available = false;
```

### Flag propagation

`common/speculative.cpp`:

```cpp
llama_set_dflash_target_kv_available(ctx_dft, gpu_ring_handle != nullptr);
```

### Drafter graph branch gate

`src/models/dflash_draft.cpp`:

```cpp
if (mctx && cparams.dflash_target_kv_available) {
    inp_attn_kv_full = dflash_build_base_attn_input(...);
}
```

### Qwen3Next tape capture

`src/llama-context.cpp` registers the Qwen3Next pre-conv QKV names:

```cpp
linear_attn_qkv_mixed-<layer>
qkv_mixed-<layer>
```

### Keep n_rs_seq for QWEN35MOE (122B)

`src/llama-context.cpp` no longer clamps `n_rs_seq` to 0 for QWEN35MOE; it keeps
the DFlash-requested value (`need_n_rs_seq() == draft.n_max`).

### Rotate DeltaNet RS snapshots for n_seq_tokens < K (35B-A3B)

`src/models/delta-net-base.cpp`, `build_recurrent_attn` keep-path write-back loop
(see the 35B-A3B fix section above for the full code).

### Full-block flat drafting

`common/speculative.cpp` uses:

```cpp
const int batch_len  = block_size;
const int output_len = n_draft + 1;
```

and consumes only rows `1..output_len-1`.

## Diagnostics retained

The following environment variables are useful for future Vulkan / Qwen3Next /
qwen35moe DFlash debugging:

| Variable | Purpose |
|---|---|
| `GGML_DFLASH_RX_DIAG=1` | Logs cross-window, ring, and hidden-row availability. |
| `GGML_DFLASH_TOKEN_TRACE=1` | Logs sampled/drafted/verified token IDs. |
| `GGML_DFLASH_TOPK_TRACE=1` | Logs drafter top-k rows for C++ vs PyTorch comparison. |
| `GGML_DFLASH_CROSS_DUMP=/path/file.txt` | Dumps a live cross-attention hidden window for replay. |
| `GGML_DFLASH_FORCE_REDECODE=1` | Forces re-decode instead of tape replay, useful to isolate recurrent-state replay issues. |
| `GGML_DFLASH_DISABLE_KV_CACHE=1` | Disables the DFlash projection cache for isolation. |
| `GGML_DFLASH_GPU_RING` | `0` = CPU hidden capture; `1` = GPU cross-ring (Vulkan falls back to CPU on main). |
| `GGML_DFLASH_GPU_RING_DEBUG=1` | Logs GPU cross-ring D2D allocation/transfer traces. |

For recurrent-state rollback inspection, `gdb` on
`llama_memory_recurrent::seq_rm` and `llama_memory_recurrent::set_rs_idx` (both
in `src/llama-memory-recurrent.cpp`, symbols in `libllama.so`) shows the rollback
requests and the snapshot slot index selected.

## Final status

The Vulkan path enables DFlash across the model set by:

- Coder-Next: making the drafter graph choose the correct target-context source
  (GPU-ring → drafter KV cache; Vulkan/CPU-hidden → fresh `target_hidden`
  projection), advancing Qwen3Next recurrent state during rollback, and keeping
  the full query block.
- 122B: keeping `n_rs_seq` for QWEN35MOE so the GPU cross-ring can roll back
  DeltaNet recurrent state without desyncing.
- 35B-A3B: rotating DeltaNet RS snapshots for `n_seq_tokens < K` so stochastic
  sampling (which drives spec depth to 0 and turns every step into a decode)
  does not clobber the snapshot history.

Remaining low acceptance on out-of-distribution prompts is a drafter
training/generalization limitation, tracked separately from the Vulkan
enablement/fix rationale.

## Archived investigation documents

The following detailed investigation notes were consolidated into this document
and moved under:

```text
docs/archive/dflash-q3cn-0acceptance-investigation/
```

- `dflash-q3cn-vulkan-working.md`
- `dflash-q3cn-state-corruption-confirmed.md`
- `qwen3next-drafter-runtime-investigation.md`
- `dflash-drafter-debug-findings.md`
- `dflash-drafter-retrain-status.md`
- `dflash-cpp-vs-pytorch-graph-comparison.md`
- `dflash-rx-diag-results.md`
- `dflash-k-norm-fix-analysis.md`
- `dflash-k-norm-order-mismatch.md`
- `dflash-k-norm-reverted.md`
- `dflash-glm-state-review.md`
- `dflash-acceptance-diagnostic-report.md`
- `dflash-acceptance-diagnostic-report-v2.md`
- `dflash-acceptance-intermediate-report.md`
- `dflash-drafter-rootcause.md`
- `dflash-runtime-trace-results.md`
- `dflash-next-steps.md`
- `dflash-qwen3-coder-next-diagnostic-report.md`
- `vulkan-dflash-status.md`