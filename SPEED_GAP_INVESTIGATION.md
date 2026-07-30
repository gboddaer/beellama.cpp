# Speed Gap Investigation: Merge vs Fork (Vulkan Backend)

## Summary

The merge branch (0920108be, Phases 1-4 complete) is **~3.8× slower** than the fork for DFlash speculative decoding on Vulkan: **~7.14 t/s vs ~27 t/s**.

## Performance History

| State | t/s | Eval Time | set_inputs avg | graph_compute avg | graphs_reused | Draft Accept |
|-------|-----|-----------|----------------|-------------------|---------------|-------------|
| Pre-Phase 1-4 (e67756167) | 5.89 | 33,948ms | 21.3ms | 8.8ms | 102 | 54.6% |
| **Post-Phase 1-4 (0920108be)** | **7.14** | **28,010ms** | **6.9ms** | **6.9ms** | **72** | **72.9%** |
| Fork target | ~27 | ~7,400ms | ~0.9ms | ~4.4ms | — | — |

Phases 1-4 improved performance by **21%** (5.89→7.14 t/s) and draft acceptance by **18.3pp** (54.6%→72.9%).

---

## Post-Phase 1-4 Profile (2026-07-24)

### Measured Breakdown (200-token DFlash benchmark, 28,010ms total eval time)

| Component | Calls | Avg (ms) | Total (ms) | % of Eval |
|-----------|-------|----------|------------|-----------|
| **wait_for_stream** (compute_queue.waitIdle) | 76 | 98.7 | 7,500 | **26.8%** |
| **Target decode** (llama_decode target) | 53 | 105.5 | 5,591 | **20.0%** |
| **Drafter decode** (llama_decode drafter) | 51 | 57.9 | 2,952 | **10.5%** |
| **Rollback reeval** (llama_decode on partial reject) | 23 | 102.7 | 2,363 | **8.4%** |
| **process_ubatch** (graph build+set_inputs+compute) | 130 | 14.0 | 1,815 | **6.5%** |
| → set_inputs | 130 | 6.9 | 895 | 3.2% |
| → graph_compute | 130 | 6.9 | 899 | 3.2% |
| **DFlash orchestration overhead** | — | — | ~70 | 0.2% |
| **Unaccounted (CPU orchestration)** | — | — | ~8,219 | 29.4% |

### Vulkan Backend Component Timers

| Component | Calls | Avg (ms) | Total (ms) | % of Eval |
|-----------|-------|----------|------------|-----------|
| wait_for_stream | 76 | 98.7 | 7,500 | 26.8% |
| end_batch (D2D submit+wait) | 30 | 0.46 | 13.8 | 0.0% |
| set_tensor_tensor (D2D→drafter) | 30 | 0.34 | 10.2 | 0.0% |
| interleave (ring→staging) | 60 | 0.09 | 5.3 | 0.0% |
| begin_batch | 30 | 0.02 | 0.5 | 0.0% |
| cross_sync (Phase 3 no-op) | 30 | 0.0 | 0.0 | 0.0% |
| interleave_wait (Phase 4 deferred) | 59 | 0.003 | 0.2 | 0.0% |
| write_d2d | 153 | 0.001 | 0.2 | 0.0% |

### Key Findings

1. **`wait_for_stream` is the #1 bottleneck** (26.8% of eval time, 7,500ms)
   - Calls `compute_queue.queue.waitIdle()` 76 times per benchmark
   - Each call blocks CPU for ~99ms waiting for GPU compute to drain
   - Despite Phase 1 narrowing from device.waitIdle() to queue.waitIdle(), the compute queue wait is still the dominant sync point

2. **Target decode** is the #2 bottleneck (20.0%, 5,591ms)
   - 105.5ms per decode, 53 decodes — this is the irreducible target compute cost

3. **Drafter decode** is the #3 bottleneck (10.5%, 2,952ms)
   - 57.9ms avg (max 116ms) — surprisingly expensive for a small drafter

4. **Rollback reeval** is the #4 bottleneck (8.4%, 2,363ms)
   - 23 of 51 draft cycles (45%) require partial rollback
   - Each rollback reeval costs 102.7ms — nearly as much as a full target decode

5. **Phase 3 is working**: cross_sync is a complete no-op (0.0ms in all 30 calls)
6. **Phase 4 is working**: interleave_wait is negligible (0.003ms avg)
7. **Phase 2 is working**: batch D2D overhead is minimal (end_batch 0.46ms avg)

---

## Pre-Phase 1-4 Profile (2026-07-23, for reference)

| Component | Calls | Avg | Total | % of eval |
|-----------|-------|-----|-------|-----------|
| **process_ubatch total** | 150 | 30.3ms | 4,538ms | **13.4%** |
| → set_inputs (within process_ubatch) | 150 | 21.3ms | 3,196ms | 9.4% |
| → graph_compute (within process_ubatch) | 150 | 8.8ms | 1,326ms | 3.9% |
| **Unaccounted (DFlash logic outside process_ubatch)** | — | — | ~29,410ms | **86.6%** |
| **Total eval time** | — | — | **33,948ms** | 100% |

**t/s**: 5.89 (200 tokens / 33.9s)
**Draft acceptance**: 136/249 = 54.6%, mean len 3.16

---

## Comparison with Fork

| Metric | Merge (Post-Phase 1-4) | Fork (original) | Ratio |
|--------|----------------------|-----------------|-------|
| t/s | 7.14 | 26.9 | 3.8× |
| set_inputs avg | 6.9ms | 0.9ms | 7.7× |
| graph_compute avg | 6.9ms | 4.4ms | 1.6× |
| process_ubatch total avg | 14.0ms | 5.3ms | 2.6× |
| Draft acceptance | 72.9% | — | — |
| Total eval time | 28,010ms | 7,400ms | 3.8× |

---

## Optimization Plan

### Completed Phases

- **Phase 1** ✅ (bca1d6ef5): Replace `device.waitIdle()` with `compute_queue.queue.waitIdle()` — 68% reduction in set_inputs time
- **Phase 2** ✅ (2686a91c6): Batch D2D ring-write copies into single submit+wait — negligible overhead (0.46ms per batch)
- **Phase 3** ✅ (d0f1ddfd5): Skip `waitIdle` in synchronize when no async work pending — cross_sync is now 0.0ms (complete no-op)
- **Phase 4** ✅ (0920108be): Deferred interleave wait with dedicated fence — interleave_wait is now 0.003ms (negligible)

### Recommended Next Phases

#### Phase 5: Timeline semaphore for `wait_for_stream` (PRIMARY TARGET — 26.8% of eval time)

**Problem**: `dflash_vk_backend_wait_for_stream` calls `compute_queue.queue.waitIdle()` 76 times, blocking the CPU for ~99ms each time. This is the single largest bottleneck.

**Solution**: Replace `vkQueueWaitIdle()` with a **timeline semaphore**:
1. After the target's graph compute (including hidden capture), signal a timeline semaphore
2. In `dflash_vk_backend_wait_for_stream`, wait on the semaphore value instead of draining the queue
3. This allows the transfer queue to proceed as soon as the hidden capture is complete
4. The CPU is no longer blocked waiting for unrelated compute work to finish

**Expected impact**: 5,000-7,000ms savings (18-25% of eval time → ~9-10 t/s)

#### Phase 6: Reduce rollback reeval cost (8.4% of eval time)

**Problem**: 45% of draft cycles require partial rollback, each costing 102.7ms.

**Solution**:
- Investigate why 45% of drafts are partially rejected
- Consider whether the reeval can reuse the verification computation
- Explore graph reuse for reeval batches (padding already implemented, check effectiveness)

**Expected impact**: 1,000-2,000ms savings (4-7% of eval time)

#### Phase 7: Reduce drafter decode time (10.5% of eval time)

**Problem**: Drafter forward pass takes 57.9ms avg (max 116ms) for 16-token batch with 5 cross-attention layers.

**Solution**:
- Profile drafter's process_ubatch separately (set_inputs vs graph_compute)
- Check if drafter graph reuse is working
- Investigate graph optimization (fewer nodes, better memory layout)

**Expected impact**: 500-1,500ms savings (2-5% of eval time)

#### Phase 8: Investigate the 29.4% "unaccounted" time

**Problem**: 8,219ms not captured by any timer.

**Solution**:
- Instrument `common_sampler_sample_and_accept_n` (verification+acceptance)
- Instrument `llama_dflash_rollback` (rollback logic)
- Instrument `llama_memory_seq_rm` and `llama_tape_replay_sync`
- Instrument checkpoint save/restore operations

**Expected impact**: Identify 2,000-5,000ms of additional optimization targets

#### Phase 9: Reduce set_inputs overhead (3.2% of eval time)

**Problem**: set_inputs still takes 6.9ms avg (895ms total) despite Phase 1 improvements.

**Solution**:
- Batch all input tensor writes into a single memcpy where possible
- Use a staging buffer with a single command buffer submission
- Investigate eliminating or combining input tensors

**Expected impact**: 300-500ms savings (1-2% of eval time)

---

## Methodology

Instrumentation added to:
- **ggml/src/ggml-vulkan/ggml-vulkan.cpp**: wait_for_stream, cross_sync, interleave, interleave_wait, set_tensor_tensor, write_d2d, begin_batch, end_batch
- **src/llama-context.cpp**: process_ubatch_total, set_inputs, graph_compute
- **common/speculative.cpp**: draft_cycle_total, draft_kv_prep, draft_build_cross, draft_batch_build, draft_wait_interleave, draft_drafter_decode, draft_argmax, ring_write_total, speculative_draft, speculative_process, dflash_accept, dflash_update_kv, dflash_rollback
- **tools/server/server-context.cpp**: target_decode, dflash_reeval

Timing uses `ggml_time_us()` with periodic printing every 25-50 calls. Two benchmark runs confirmed reproducibility within 2%.

Benchmark: 200-token DFlash generation with Qwen3.6-27B-Q4_K_M + DFlash-Q4_K_M, Vulkan0, seed 7, temp 0, ctx 8192, flash-attn on, k/v q8_0.

---

## 2026-07-29: Comprehensive 3-Way Benchmark

Full benchmark suite comparing **upstream llama.cpp**, **fork main** (original DFlash), and **merge** (compact verify) across two model pairs, two prompts, both with and without DFlash.

**Hardware**: AMD Strix Halo APU (Vulkan/RADV)  
**Settings**: ctx 4096, 4 threads, Flash Attention, temp 0.6, top-k 20, top-p 1.0

### Qwen3.6-27B (Q4_K_M, 46GB)

#### Coding Prompt (41 prompt tok, 2048 gen tok)

| Branch | Mode | Time | t/s | Draft Accept | Notes |
|--------|------|------|-----|--------------|-------|
| upstream | NO DFlash | — | — | — | Model not supported (MoD/SSM arch) |
| fork main | NO DFlash | 167s | 12.3 | — | Baseline |
| fork main | DFlash | 223s | 9.2 | 15% | 34% SLOWER, low acceptance |
| merge | NO DFlash | 167s | 12.3 | — | Same as fork main |
| merge | DFlash | 213s | 9.6 | 93% | 25% SLOWER, high acceptance |

#### Math Prompt (96 prompt tok, 2048 gen tok)

| Branch | Mode | Time | t/s | Draft Accept | Notes |
|--------|------|------|-----|--------------|-------|
| upstream | NO DFlash | — | — | — | Model not supported |
| fork main | NO DFlash | 167s | 12.3 | — | Baseline |
| fork main | DFlash | 99s | 20.7 | 44% | **40% FASTER** than baseline |
| merge | NO DFlash | 168s | 12.1 | — | Same as fork main |
| merge | DFlash | 160s | 12.8 | 90% | 5% FASTER than baseline |

### Qwen3-Coder-Next (Q4_K_M, 22GB)

#### Coding Prompt (39 prompt tok, 2048 gen tok)

| Branch | Mode | Time | t/s | Draft Accept | Notes |
|--------|------|------|-----|--------------|-------|
| upstream | NO DFlash | 39s | 52.8 | — | Baseline |
| fork main | NO DFlash | 39s | 52.2 | — | Same as upstream |
| fork main | DFlash | 46s | 45.4 | 30% | 14% SLOWER, TOPK issue |
| merge | NO DFlash | 39s | 53.2 | — | Same as upstream |
| merge | DFlash | STALL | — | — | **Deadlocks after prefill** |

#### Math Prompt (94 prompt tok, 2048 gen tok)

| Branch | Mode | Time | t/s | Draft Accept | Notes |
|--------|------|------|-----|--------------|-------|
| upstream | NO DFlash | 39s | 52.4 | — | Baseline |
| fork main | NO DFlash | 39s | 53.2 | — | Same as upstream |
| fork main | DFlash | 41s | 49.4 | 30% | 8% SLOWER, TOPK issue |
| merge | NO DFlash | 39s | 52.2 | — | Same as upstream |
| merge | DFlash | STALL | — | — | **Deadlocks after prefill** |

### Root Causes

#### 1. Vulkan TOPK bug (fork main DFlash)
```
DFlash reduced verify failed (no argmax output, top_k=20);
disabling reduced verify for this session (backend may not support TOPK)
```
Forces full-window K/V projection fallback, kills speculative efficiency. Affects fork main's DFlash on Qwen3-Coder-Next (30% draft accept, 8-14% slower).

#### 2. Merge branch DFlash stall (Qwen3-Coder-Next)
Server starts and loads fine, prefill completes, but curl request times out indefinitely. Ring buffer stalls. Same draft model works on fork main but deadlocks on merge. **Likely regression in compact verify path.**

#### 3. Prompt-dependent DFlash performance
Math prompts (96 tok) benefit more than coding prompts (39-41 tok) because longer context gives the draft model more room to accumulate accepted sequences before verification overhead dominates. Qwen3.6-27B math with fork main DFlash achieved **40% speedup** (20.7 vs 12.3 t/s).

### Key Takeaways

1. **NO DFlash performance is identical across all three branches** (within 1%) — the merge did not regress baseline performance.
2. **DFlash on Qwen3.6-27B is inconsistent**: 40% faster on math, 25-34% slower on coding. Prompt length is the key variable.
3. **DFlash on Qwen3-Coder-Next is broken on merge branch** and degraded on fork main. The draft model needs investigation.
4. **Compact verify (merge) improves draft acceptance** (93% vs 15% on fork main for Qwen3.6-27B coding) but does not translate to speed — the verification overhead dominates.
5. **The speed gap investigation from July 24** (7.14 vs 27 t/s) used different hardware/settings. New data shows the gap is more nuanced and prompt-dependent.