# Task: Optimize hidden_gpu copy ops in graph builder

## Context

The DFlash speed gap investigation (see SPEED_GAP_INVESTIGATION.md) has identified the root cause:

1. **Eval callback ON** → forces node-by-node graph execution → 15× scheduler overhead (3.9s vs 0.25s per 100 tokens)
2. **Eval callback OFF** → graph-embedded hidden_gpu copies add ~3.6s GPU overhead → net zero improvement (12.1 → ~11.4 t/s)
3. **The fork (adb92b36a) avoids both** — its graph-embedded hidden_gpu capture has negligible GPU overhead

The merge's hidden_gpu copy ops in `qwen35.cpp` are **identical** to the fork's (same `view_2d` → `cpy` → `build_forward_expand`, no extra `cont`). So the 2× GPU overhead isn't from the copy ops themselves.

## Key finding from diff

The fork has a `dflash_compact_verifier_only` mode that **skips building the full 5120-dim logits tensor** during DFlash verify — it only computes a top-k argmax. The merge always builds full logits:

**Fork (adb92b36a:324-339):**
```cpp
if (cparams.dflash_verify_logits) {
    res->t_logits_argmax = ggml_topk_ext(ctx0, cur, topk, ...);
    ggml_build_forward_expand(gf, res->t_logits_argmax);
}
if (!dflash_compact_verifier_only) {
    ggml_build_forward_expand(gf, cur);  // SKIPPED during DFlash verify
}
```

**Merge always:**
```cpp
ggml_build_forward_expand(gf, cur);  // always builds full logits
```

When the eval callback is off and the graph runs as batch, the merge wastes GPU work computing full logits every decode cycle when only argmax is needed.

## Current patch (already applied in working tree)

There is already an uncommitted patch in `src/models/qwen35.cpp` that ports the `dflash_compact_verifier_only` path from the fork. The infrastructure (`dflash_verify_logits`, `dflash_reduced_consumer_active`, `t_logits_argmax`, `ggml_topk_ext`, `ggml_argmax_ext`) all already exists in the merge — they were just never wired into the graph builder.

## Problem encountered

When benchmarking from a fresh `llama-server` instance (not the tmux warm session), we got `graphs reused = 0` — the graph was rebuilt every decode cycle. This may be a pre-existing issue with first-request benchmarks (cold graph cache), not caused by the patch. The tmux session's earlier benchmarks showed `graphs reused = 1` for the baseline.

## Tasks

1. **Build the patched binary** (already done — `build-vulkan/bin/llama-server` should have the patch):
   ```bash
   cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
   cmake --build build-vulkan --target llama-server -j$(nproc)
   ```

2. **Benchmark the patched binary** — use the same benchmark command that produces warm `graphs reused > 0` results. The known working command from TASK_PROGRESS.md:
   ```bash
   build-vulkan/bin/llama-server \
     --model /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
     --spec-type dflash \
     --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
     --spec-draft-n-max 4 \
     --device Vulkan0 \
     -ngl 999 \
     --ctx-size 8192 \
     --flash-attn on \
     --cache-type-k q8_0 \
     --cache-type-v q8_0 \
     --seed 7 \
     --temp 0 \
     -n 200 \
     --port 8899 \
     --host 127.0.0.1
   ```
   Then send a request that generates 200+ tokens. Use the prompt: "What is 2+2? First think step by step, then give the final answer on its own line as Answer: X."

3. **Check `graphs reused` count** — if 0, the graph is being rebuilt every cycle. Compare with the baseline (stash the patch, rebuild, benchmark, compare).

4. **Also try: disabling eval callback when hidden_gpu is ready** — in `src/llama-context.cpp` around line 7037, change:
   ```cpp
   dflash_skip_eval_callback =
       dflash_use_prefill_staging || dflash_suppress_callback_for_view;
   ```
   to:
   ```cpp
   dflash_skip_eval_callback =
       dflash_use_prefill_staging || dflash_suppress_callback_for_view ||
       dflash_graph_hidden_ready;
   ```
   This was tried before and caused `graphs reused = 0`, but that may have been a cold-cache issue. Test it with a warm session (send a short request first, then the benchmark request).

5. **If the compact verifier patch works (speed improves without regression)** — commit it:
   ```bash
   git add src/models/qwen35.cpp
   git commit -m "dflash(qwen35): port compact verifier-only mode from fork — skip full logits during DFlash verify"
   ```

6. **If the eval callback skip also works** — commit it separately:
   ```bash
   git add src/llama-context.cpp
   git commit -m "dflash(context): skip eval callback when hidden_gpu is ready for graph-embedded capture"
   ```

7. **Verify correctness** — check that:
   - Output is coherent (not garbled)
   - Draft acceptance rate is similar (78.4% baseline)
   - No errors/crashes
   - `graphs reused > 0` (graph caching works)

## Baseline numbers (from commit 48e9a1d75)
- Speed: 12.1 t/s (single-slot, Qwen3.6-27B DFlash)
- Acceptance: 78.4%
- graphs reused: 1 (from tmux session)
- Fork (main adb92b36a): 23.8 t/s