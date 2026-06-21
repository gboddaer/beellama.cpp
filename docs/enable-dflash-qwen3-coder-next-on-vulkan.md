# Enable DFlash Qwen3-Coder-Next on Vulkan

This document is the canonical rationale for the Qwen3-Coder-Next DFlash Vulkan fix. It replaces the scattered investigation notes that were created while debugging the original 0% draft-token acceptance problem.

## Summary

Qwen3-Coder-Next DFlash originally produced near-0% accepted draft tokens on Vulkan. The failure was not caused by a generic DFlash verifier bug: Qwen3.6 DFlash worked, and Qwen3-Coder-Next could work under other implementations. The issue was a chain of Qwen3Next-specific runtime and drafter-input mismatches.

The decisive Vulkan runtime fix is:

- Vulkan does not have the DFlash GPU cross ring.
- Without the GPU cross ring, the drafter's normal full-attention KV cache is not populated with target-context K/V.
- The Qwen3-Coder-Next drafter was still taking the full-attention KV-cache branch, reading empty/stale context.
- The DFlash drafter graph now only builds that branch when target KV is actually available. On Vulkan it falls back to projecting K/V freshly from CPU-captured `target_hidden`.

With the runtime fixes and a drafter trained for the live C++ position semantics, in-distribution acceptance was observed at roughly 68-72% instead of 0%.

## User-facing command shape

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

The exact drafter file matters. The runtime fix enables the correct Vulkan path, but a drafter trained against the wrong input/position semantics can still show poor acceptance.

## Root-cause chain

### 1. Qwen3Next conv state was not advanced during DFlash rollback

Qwen3-Coder-Next has recurrent / convolutional state. During DFlash verification and rollback, accepted tokens must advance the target recurrent state. The tape replay path had to reconstruct both DeltaNet state and convolution state.

The Qwen3Next graph emits the pre-conv QKV tensor as:

```text
linear_attn_qkv_mixed-<layer>
```

with a legacy path named:

```text
qkv_mixed-<layer>
```

The DFlash tape map did not capture these names, so `tape_replay_conv()` skipped every Qwen3Next recurrent layer. The conv state `r_l` remained frozen at the pre-draft backup state. That could produce garbled target output even when no draft tokens were accepted.

Fix:

```cpp
dflash_capture->tape_name_map["linear_attn_qkv_mixed-" + il_str] = {idx, DFLASH_TAPE_QKV};
dflash_capture->tape_name_map["qkv_mixed-" + il_str]             = {idx, DFLASH_TAPE_QKV};
```

This made the conv replay path see the Qwen3Next pre-conv data and advance `r_l` correctly.

### 2. Vulkan lacked a populated drafter full-attention KV cache

DFlash keeps recent target hidden states in a cross-attention ring. On CUDA, the GPU cross ring can also populate the drafter-side full-attention KV cache with target-context K/V.

On Vulkan, the GPU cross ring is unavailable, so DFlash uses CPU hidden capture:

```text
gpu_ring_handle == nullptr
```

In that case, `update_drafter_kv_cache()` returns early and the drafter full-attention KV cache is not populated with target context.

However, the DFlash drafter graph still built `inp_attn_kv_full` whenever the drafter had a memory context. For all-full-attention drafters, this selected the branch that reads target context from the normal KV cache:

```cpp
if (inp_attn_kv_full && !hparams.is_swa(il)) {
    // read target context from drafter KV cache
}
```

On Vulkan that cache was empty or stale. The drafter therefore generated from little or no real target context, producing repeated bad predictions and 0% acceptance.

Fix: add an explicit context parameter:

```cpp
bool dflash_target_kv_available = false;
```

and a public setter:

```cpp
llama_set_dflash_target_kv_available(ctx_dft, gpu_ring_handle != nullptr);
```

Then build the full-attention KV input only when the target KV cache is actually populated:

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

That path works with CPU-captured `target_hidden`, which is the correct Vulkan fallback.

### 3. Flat DFlash must keep the full query block

DFlash query-token attention is non-causal over the draft block. Shortening the query block can change logits, including early positions.

The flat path previously used:

```cpp
batch_len = n_draft + 1;
```

Tree DFlash already used the full block. The flat and batched flat paths now keep the trained/inferred block shape:

```cpp
batch_len  = block_size;
output_len = n_draft + 1;
```

Only `output_len` rows are marked for output and consumed by the sampler. This preserves full-block drafter semantics while still limiting active draft horizon.

### 4. Runtime parity required comparing exact live C++ inputs

The debugging path added several diagnostics that made it possible to prove where the mismatch lived:

- `GGML_DFLASH_RX_DIAG=1` logs cross-ring / hidden-window availability.
- `GGML_DFLASH_TOKEN_TRACE=1` logs sampled token, drafted tokens, and verified tokens.
- `GGML_DFLASH_TOPK_TRACE=1` logs C++ drafter top-k predictions.
- `GGML_DFLASH_CROSS_DUMP=/path/file.txt` dumps the live cross-attention hidden window for PyTorch replay.

The cross dump showed that after the Vulkan KV branch fix, C++ and PyTorch forward passes matched for the same live input. That narrowed the remaining 0% behavior to drafter training/input semantics rather than a C++ graph mismatch.

### 5. Drafter position semantics matter

C++ consumes DFlash predictions starting at row / position 1:

```cpp
for (int i = 1; i < output_len; ++i) {
    // consume draft prediction rows
}
```

Position 0 is the known root / bonus token. The first actual draft token is position 1.

A drafter trained to predict from position 0 can appear valid in a standalone training loop while still producing poor live C++ acceptance, because C++ reads position 1. Training the live-consumed position fixed the in-distribution acceptance collapse once the runtime graph was correct.

This document does not cover broader drafter generalization work; that is a separate training/data problem. The Vulkan runtime issue is the empty target-KV branch and Qwen3Next recurrent-state handling described above.

## Ruled-out or deprioritized hypotheses

The investigation ruled out several plausible but ultimately non-decisive causes.

### k_norm ordering

There was a suspected mismatch around whether target K normalization happened before or after concatenating context/noise K. Further comparison showed this was not the cause of the 0% acceptance. The attempted k_norm fix was reverted.

### Basic cross-ring emptiness

Diagnostics showed the cross window was not generally empty. The real problem was more specific: full-attention layers were reading target context from an unpopulated drafter KV cache on Vulkan instead of using the fresh `target_hidden` projection path.

### Generic DFlash verifier failure

Qwen3.6 DFlash worked, and the server verification path could reject all drafts while still producing coherent target output when recurrent state was handled correctly. This made a global verifier failure unlikely.

### C++ vs PyTorch graph mismatch after KV-branch fix

After dumping the exact live hidden-state window and replaying it in PyTorch, C++ and PyTorch forward outputs matched. That eliminated the C++ graph as the remaining source of the live acceptance issue after the Vulkan KV fix.

### Hidden tensor layout/type corruption

Earlier diagnostics did not support basic hidden-state layout or tensor-type corruption as the persistent root cause.

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

### Full-block flat drafting

`common/speculative.cpp` uses:

```cpp
const int batch_len  = block_size;
const int output_len = n_draft + 1;
```

and consumes only rows `1..output_len-1`.

## Diagnostics retained

The following environment variables are useful for future Vulkan/Qwen3Next DFlash debugging:

| Variable | Purpose |
|---|---|
| `GGML_DFLASH_RX_DIAG=1` | Logs cross-window, ring, and hidden-row availability. |
| `GGML_DFLASH_TOKEN_TRACE=1` | Logs sampled/drafted/verified token IDs. |
| `GGML_DFLASH_TOPK_TRACE=1` | Logs drafter top-k rows for C++ vs PyTorch comparison. |
| `GGML_DFLASH_CROSS_DUMP=/path/file.txt` | Dumps a live cross-attention hidden window for replay. |
| `GGML_DFLASH_FORCE_REDECODE=1` | Forces re-decode instead of tape replay, useful to isolate recurrent-state replay issues. |
| `GGML_DFLASH_DISABLE_KV_CACHE=1` | Disables the DFlash projection cache for isolation. |

## Final status

The Vulkan path is enabled by making the drafter graph choose the correct target-context source:

- CUDA/GPU-ring path: full-attention layers may read target K/V from the populated drafter KV cache.
- Vulkan/CPU-hidden path: full-attention layers compute fresh K/V from `target_hidden`.

The remaining low acceptance on out-of-distribution prompts is not the same runtime bug. It is a drafter training/generalization limitation and should be tracked separately from the Vulkan enablement rationale.

## Archived investigation documents

The following detailed investigation notes were consolidated into this document and moved under:

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
