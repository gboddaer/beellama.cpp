# Phase 5 — Restructure DFlash Vulkan capture synchronization to eliminate redundant double-sync

**Date:** 2026-07-26  
**Branch:** `merge_llama_into_beellama_2` @ `0920108be`  
**Goal:** Measured improvement (15-18 t/s target), correctness preserved

## Problem

The per-draft-cycle decode loop performs a **redundant double-sync** on the Vulkan compute queue:

1. `dflash_wait_for_gpu_capture_stream()` → `dflash_vk_backend_wait_for_stream()` → `compute_queue.queue.waitIdle()` — **full compute queue drain**
2. `ggml_backend_sched_synchronize()` → `ggml_backend_synchronize(Vulkan)` → `ggml_vk_synchronize()` → waits for `ctx->fence` on compute queue — **full compute queue drain again**

Step 1 is the dominant per-cycle CPU stall. Step 2 always runs unconditionally (both branches of the `if (dflash_gpu_capture_stream_ready)` check in llama-context.cpp:7168-7176 call `ggml_backend_sched_synchronize`).

## Analysis

### Why `ggml_vk_synchronize()` is sufficient

`ggml_vk_synchronize()` (ggml-vulkan.cpp:15635):
1. Finalizes `ctx->compute_ctx` (if active) — ends the compute command buffer
2. Submits it to the compute queue
3. Submits an empty command buffer to the compute queue with `ctx->fence`
4. Calls `ggml_vk_wait_for_fence(ctx)` — blocks CPU until fence is signaled

All hidden capture ops (`ggml_cpy` into `hgpu->layers`) are graph-embedded and run on the compute queue's command buffer. They are fully ordered before `ctx->fence`. Therefore `ggml_vk_synchronize()` already guarantees hidden capture completion.

### Why the current `waitIdle()` is worse

`compute_queue.queue.waitIdle()` drains the **entire** queue, including any work submitted after the current graph batch. `ggml_vk_wait_for_fence` only waits for the specific fence, which is more targeted.

### CUDA comparison

CUDA's `dflash_cuda_backend_wait_for_stream` (ggml-cuda.cu:5003) records a CUDA event on the backend stream and makes `cudaStreamPerThread` wait on it — a GPU-side dependency that doesn't block the CPU. It's still redundant with `ggml_backend_sched_synchronize` (which does `cudaStreamSynchronize`), but the CPU cost is minimal. We leave CUDA alone.

### Backend registration

`dflash_capture_add_wait_backend()` (llama-context.cpp:187) looks up `"dflash_cuda_backend_wait_for_stream"` via `ggml_backend_reg_get_proc_address`. Both CUDA and Vulkan register a function under this name:
- CUDA: `dflash_cuda_backend_wait_for_stream` (event-based, lightweight)
- Vulkan: `dflash_vk_backend_wait_for_stream` (waitIdle, heavy)

The function pointer is stored in `capture_wait_backends` and called from `dflash_wait_for_gpu_capture_stream()`.

## Design

### Change

**Make `dflash_vk_backend_wait_for_stream()` return `true` immediately (no-op).** Remove the `compute_queue.queue.waitIdle()` call entirely.

This eliminates the redundant double-sync for Vulkan while preserving the correctness guarantee: `ggml_backend_sched_synchronize()` → `ggml_vk_synchronize()` is the single authoritative sync point.

### What stays the same

- CUDA's `dflash_cuda_backend_wait_for_stream` — event-based, lightweight, left untouched
- `ggml_backend_sched_synchronize()` — called unconditionally, provides the actual synchronization
- `dflash_wait_for_gpu_capture_stream()` — logic unchanged, just Vulkan returns true faster
- Ring write path (`begin_batch` → `write_d2d_tensor` → `end_batch` → `synchronize` → `interleave`) — unchanged
- Multi-GPU path — `dflash_wait_for_gpu_capture_stream()` returns false for `n_devices() > 1`, falls through to `ggml_backend_sched_synchronize()`

### Why this doesn't break other backends

- **CUDA**: Uses its own `dflash_cuda_backend_wait_for_stream` (event-based). No change.
- **CPU/other backends**: `dflash_capture_add_wait_backend` only registers backends that provide `"dflash_cuda_backend_wait_for_stream"`. If the function pointer is null, the backend is not added to `capture_wait_backends`. Unchanged.
- **Multi-device**: `dflash_wait_for_gpu_capture_stream()` returns false when `model.n_devices() > 1`, bypassing `capture_wait_backends` entirely.

### Files modified

1. **`ggml/src/ggml-vulkan/ggml-vulkan.cpp`** (~18195): Replace `dflash_vk_backend_wait_for_stream` body with `return true` (remove `waitIdle`)

That's it — a single-function change.

## Risks

- **Cross-queue visibility**: The `set_tensor_tensor` comment (ggml-vulkan.cpp:18273) notes a known Vulkan cross-queue memory visibility bug (transfer queue writes, compute queue reads, no semaphore). This bug exists regardless of phase 5 and is not affected by this change — `ggml_vk_synchronize()` still provides the full ordering guarantee between the two queues.
- **AMD iGPU specifics**: The queue family ordering on `single_queue` devices (same family) provides implicit ordering. On separate families, `ggml_vk_synchronize()` still drains the compute queue. Either way, correctness is preserved.

## Verification

- DFlash output must match non-DFlash (greedy, temp 0)
- Acceptance ≥75%
- No garble
- Multi-slot (`--parallel 4`) routing works
- Benchmark: compare t/s before/after on `build-vulkan/bin/llama-server`
