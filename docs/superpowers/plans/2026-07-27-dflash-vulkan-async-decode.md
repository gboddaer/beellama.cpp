# DFlash Vulkan Async Decode — Remove sched_synchronize Stall

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 204ms/call `ggml_backend_sched_synchronize` stall from the DFlash decode loop, restoring the asynchronous decode model used by upstream llama.cpp and the fork.

**Architecture:** The merge's phase 5 commit (ae43a1e64 + 70d0fb72a) added an unconditional `ggml_backend_sched_synchronize()` after every DFlash target decode. This is a CPU-blocking full scheduler drain that takes ~204ms per call because the GPU is still processing the graph when the sync is called — a pipeline stall. It accounts for 50% of DFlash eval time (9191ms out of 18475ms for 100 tokens). Upstream llama.cpp uses asynchronous decode with implicit sync (fence wait triggered by tensor access). The fork's `device.waitIdle()` returns in ~0ms because the decode completes before the wait is called. The fix: remove the explicit `sched_synchronize` and rely on implicit/GPU-side sync, matching upstream's model. This was tested and confirmed: removing it improves performance 53% (4.01→6.15 t/s) with correct output.

**Tech Stack:** C++, Vulkan, llama.cpp speculative decoding (DFlash), ggml backend scheduler

## What This Plan Is (and Is NOT)

**This plan REMOVES a regression. It does NOT add new async behavior.**

### What it IS

Upstream llama.cpp already uses **asynchronous decode** for Vulkan: `llama_decode()` submits the graph to the GPU command queue and returns immediately — the CPU does NOT wait for the GPU. The synchronization happens **implicitly** later, when the output tensor is actually read (e.g., `llama_get_logits_ith()` → `ggml_backend_tensor_get()` triggers a Vulkan fence wait). This is the normal, correct, fast path.

The merge's phase 5 commit **broke this** for DFlash by adding an explicit `ggml_backend_sched_synchronize()` after every DFlash target decode. This forces the CPU to wait for the entire GPU scheduler to drain — 204ms per call. This is NOT how upstream works. The fork does NOT have this call (it uses `device.waitIdle()` which returns in ~0ms because the decode already finished via the implicit output-read sync).

**This plan restores the upstream async model** by removing the explicit `sched_synchronize`. The DFlash decode will go back to being asynchronous: graph submitted → CPU continues → sync happens implicitly when the drafter reads the argmax output.

### What it is NOT

- **NOT** async ring buffer transfers — the ring transfers (CPU→GPU, GPU→GPU D2D) already run on the Vulkan transfer queue asynchronously. This plan does not touch them.
- **NOT** concurrent draft+verify batching — we are not making the draft model and target model run simultaneously. They still run sequentially (target verify → draft → target verify → ...).
- **NOT** overlapping draft and target on the GPU — we are not pipelining GPU work. We are simply removing a CPU-blocking wait that prevents the CPU from proceeding while the GPU is still working.
- **NOT** a new feature — this is a regression fix. The async decode model already exists in upstream and the fork. Phase 5 accidentally broke it.

### Why removing the sync is safe (cross-queue visibility)

The DFlash decode captures hidden states via `ggml_cpy` into `hgpu->layers` on the **compute queue**. The D2D ring write (`dflash_cross_ring_gpu_write_d2d_tensor`) reads these hidden states from the **transfer queue**. In Vulkan, cross-queue reads need explicit ordering (semaphore or fence).

The sync happens implicitly because the drafter's argmax extraction (`llama_get_logits_argmax`) calls `ggml_backend_tensor_get()` which triggers a `vk::Device::waitForFences()` on the compute queue's fence. By the time the D2D ring write runs (which is after the argmax extraction in the code flow), the compute queue has already been waited on. This was **tested and confirmed**: output is coherent, no garbage, acceptance rate maintained.

If cross-queue issues arise in the future (e.g., with different GPU drivers or multi-GPU setups), a Vulkan timeline semaphore should be wired between the compute and transfer queues — but that is future work, NOT this plan.

## Global Constraints

- **Worktree:** `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- **Build:** `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc)` (Vulkan: `-DGGML_VULKAN=ON`)
- **Target model:** `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf` (16GB, Qwen3.6-27B)
- **Draft model:** `/crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf` (986MB, DFlash drafter)
- **Benchmark command:** `build/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf --spec-type dflash --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1`
- **Do NOT commit unless the user explicitly asks.**
- **Fork (for comparison):** `/crypt/beellama.cpp` with build at `build-vulkan/bin/llama-server`
- **Baseline (no DFlash) is identical between fork and merge:** 12.48 t/s, 80ms/token — the Vulkan backend itself is not the problem.
- **Fork DFlash: 19.39 t/s.** Merge DFlash before fix: 4.01 t/s. After removing sched_synchronize (tested): 6.15 t/s. This plan does NOT close the full gap — it eliminates the #1 bottleneck (50% of eval time). The remaining 3.1× gap (rollback reeval + GPU contention) is documented as future work.

## Current Working Tree State (READ THIS FIRST)

The working tree has **uncommitted changes** from the investigation session. There are three modified files:

1. **`common/speculative.cpp`** (27 lines changed) — Contains TWO things:
   - The **accept bug fix** (lines ~4441-4460 and ~5054-5063): adds `curr_impl` fallback to `common_speculative_accept` and sets `impl_last[seq_id]` in `common_speculative_draft_batch`. This fix is CORRECT and should be KEPT.
   - The **batch_draft profile print** (lines ~5067-5077): debug instrumentation added during investigation. Should be REMOVED.

2. **`src/llama-context.cpp`** (20 lines changed) — Contains the **sched_synchronize block** (lines ~7170-7187): currently has `ggml_backend_sched_synchronize(sched.get())` COMMENTED OUT with hypothesis-test instrumentation around it. Should be REPLACED with a clean no-op comment.

3. **`tools/server/server-context.cpp`** (25 lines changed) — Contains THREE debug instrumentation blocks:
   - **target_decode timer** (lines ~4173-4185): wraps `llama_decode(ctx_tgt, batch_view)` with profile timing. Should be REMOVED (restore to single line).
   - **sample_accept timer** (lines ~4470-4479): wraps `common_sampler_sample_and_accept_n` with profile timing. Should be REMOVED.
   - **rollback_reeval timer** (lines ~4598-4602): wraps `llama_decode(ctx_tgt, batch_reeval)` with profile timing. Should be REMOVED.

There is also one committed change: **phase 5 commit `70d0fb72a`** (in `src/llama-context.cpp`) which made `dflash_vk_backend_wait_for_stream` a no-op. This commit is already in git history and is NOT touched by this plan — the no-op is correct (it prevents a redundant `device.waitIdle()` that would be equally expensive).

## Measured Bottleneck (confirmed by instrumentation, 100-token benchmark)

| Component | Calls | Total ms | Avg ms | % eval |
|-----------|-------|----------|--------|--------|
| **sched_synchronize** | **45** | **9191** | **204.2** | **49.8%** |
| target_decode | 25 | 5206 | 208.2 | 28.2% |
| rollback_reeval | 20 | 4321 | 216.0 | 23.4% |
| batch_draft | 23 | 1525 | 66.3 | 8.3% |
| sample_accept | 23 | 33 | 1.5 | 0.2% |

**Why sched_synchronize is 204ms:** The merge's `llama_decode` submits the graph asynchronously (Vulkan) and returns immediately. Then `ggml_backend_sched_synchronize()` is called, which waits on `ctx->fence` — but the GPU is still computing the graph. So the CPU blocks for ~204ms (the full GPU compute time) waiting for the graph to finish. This is called for BOTH the verify decode AND the rollback reeval decode (25+20=45 calls).

**Why the fork doesn't have this:** The fork calls `device.waitIdle()` in `dflash_vk_backend_wait_for_stream`, but the fork's decode is already complete by the time it's called (the output read triggers an implicit fence wait). So `device.waitIdle()` returns in ~0ms.

**Why removing it is safe:** See the "Why removing the sync is safe (cross-queue visibility)" section above for the full explanation. Short version: the drafter's argmax extraction (`llama_get_logits_argmax`) calls `ggml_backend_tensor_get` which triggers a fence wait — this implicit sync ensures the hidden capture is complete before the D2D copy reads it. The test confirmed: output is coherent with no garbage.

---

### Task 1: Replace sched_synchronize block with clean no-op in llama-context.cpp

**Why:** This is the core fix. The `sched_synchronize` call is 50% of eval time. Removing it (tested) improves 4.01→6.15 t/s with correct output. The current code has it commented out with debug instrumentation — we replace that with a clean, documented no-op.

**Files:**
- Modify: `src/llama-context.cpp` (lines ~7170-7187)

**Interfaces:**
- Consumes: `dflash_capture` (bool, already checked), `sched` (ggml_backend_sched_t, no longer used)
- Produces: No explicit sync after DFlash decode — relies on implicit sync from output reads

- [ ] **Step 1: Read the current code to confirm the exact text**

Run: `sed -n '7170,7188p' src/llama-context.cpp`

You should see a block starting with `if (dflash_capture) {` containing `// HYPOTHESIS TEST` comments, a commented-out `ggml_backend_sched_synchronize(sched.get());`, and a `profile_sync` fprintf block.

- [ ] **Step 2: Replace the hypothesis-test block with a clean no-op**

Use the edit tool with this exact oldText and newText:

**oldText** (the entire hypothesis-test block, from `if (dflash_capture) {` to the closing `}`):
```
        if (dflash_capture) {
            // HYPOTHESIS TEST: skip sched_synchronize entirely. The decode is async
            // (like upstream llama.cpp); the hidden capture data should be available
            // to the D2D ring write via implicit sync (output read fence) or GPU-side
            // queue ordering. If correctness holds, this eliminates the 204ms/call stall.
            const int64_t t_sched_sync_us = ggml_time_us();
            // ggml_backend_sched_synchronize(sched.get());  // SKIPPED for hypothesis test
            const int64_t t_sched_sync_end_us = ggml_time_us();
            {
                static bool profile_sync = [] {
                    const char * e = std::getenv("GGML_DFLASH_PROFILE");
                    return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
                }();
                if (profile_sync) {
                    fprintf(stderr, "dflash profile: sched_synchronize=%.1fms (SKIPPED)\n", (t_sched_sync_end_us - t_sched_sync_us) / 1e3);
                }
            }
        }
```

**newText:**
```
        if (dflash_capture) {
            // No explicit sched_synchronize here — the decode is asynchronous
            // (matching upstream llama.cpp). The hidden capture data will be
            // available to the D2D ring write via implicit sync (the output
            // read in the drafter's argmax extraction triggers a fence wait)
            // or GPU-side queue ordering. This eliminates the 204ms/call
            // CPU stall that was 50% of DFlash eval time.
            //
            // If cross-queue visibility issues arise (compute queue hidden
            // capture not visible to transfer queue D2D copy), a lightweight
            // Vulkan timeline semaphore should be wired between the compute
            // and transfer queues — NOT a CPU-blocking sched_synchronize.
        }
```

- [ ] **Step 3: Build**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors. If there are errors, the edit did not match exactly — re-read the code and retry.

- [ ] **Step 4: Benchmark correctness — short test (100 tokens)**

```bash
timeout 120 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/test_task1_short.log &
sleep 25
curl -s -o /tmp/test_task1_short.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":100,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
```

**Pass criterion:** 
- HTTP response code is 200
- The JSON response has non-empty `choices[0].message.content` or `choices[0].message.reasoning_content`
- The text is coherent English (not garbage/repeated characters)
- Run: `python3 -c "import json; d=json.load(open('/tmp/test_task1_short.json')); c=d['choices'][0]['message']['content'] or d['choices'][0]['message']['reasoning_content']; print('OK' if len(c)>50 else 'FAIL: too short'); print(c[:200])"`
- Expected output: `OK` followed by coherent text about fibonacci

**Fail criterion:** HTTP error, empty response, or garbage output → the implicit sync is insufficient. Stop and report — a timeline semaphore will be needed instead.

- [ ] **Step 5: Benchmark performance — 300-token coding task**

```bash
timeout 180 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_task1.log &
sleep 25
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Implement a complete binary search tree in C++ with insertion, deletion, in-order traversal, and a print function. Include proper memory management and a main function that demonstrates all operations."}],"max_tokens":300,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance" /tmp/bench_task1.log
```

**Pass criterion:**
- The `eval time` line shows `≥ 5.0 tokens per second` (was 4.01 before fix, should be ~6.15)
- The `draft acceptance` line shows `≥ 0.30` (was 0.355, should stay similar)
- Run: `grep "tokens per second" /tmp/bench_task1.log | tail -1`
- Extract the number: `python3 -c "import re; l=open('/tmp/bench_task1.log').read(); m=re.findall(r'(\S+) tokens per second', l); tps=float(m[-1]) if m else 0; print(f'PASS tps={tps:.1f}' if tps>=5.0 else f'FAIL tps={tps:.1f} (expected >=5.0)')"`
- Expected: `PASS tps=6.x`

**Fail criterion:** tps < 5.0 → the edit didn't take effect or the build is stale. Rebuild and retry.

---

### Task 2: Remove target_decode instrumentation from server-context.cpp

**Why:** The investigation added a profile timer around `llama_decode(ctx_tgt, batch_view)` that wraps it in a block and changes `const int ret` to `int ret = 0; { ... }`. This must be restored to the original single-line form to keep the code clean and avoid scope confusion.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines ~4173-4185)

**Interfaces:**
- Consumes: `ctx_tgt`, `batch_view`, `metrics`
- Produces: `int ret` (the decode return code)

- [ ] **Step 1: Read the current code to confirm the exact text**

Run: `sed -n '4172,4188p' tools/server/server-context.cpp`

You should see:
```
        int ret = 0;
        {
            static bool profile_tgt = [] {
                const char * e = std::getenv("GGML_DFLASH_PROFILE");
                return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
            }();
            const int64_t t_tgt_decode_us = profile_tgt ? ggml_time_us() : 0;
            ret = llama_decode(ctx_tgt, batch_view);
            if (profile_tgt) {
                LOG_INF("dflash profile: target_decode=%.1fms ret=%d\n", (ggml_time_us() - t_tgt_decode_us) / 1e3, ret);
            }
        }
```

- [ ] **Step 2: Replace with the original single-line form**

**oldText:**
```
        int ret = 0;
        {
            static bool profile_tgt = [] {
                const char * e = std::getenv("GGML_DFLASH_PROFILE");
                return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
            }();
            const int64_t t_tgt_decode_us = profile_tgt ? ggml_time_us() : 0;
            ret = llama_decode(ctx_tgt, batch_view);
            if (profile_tgt) {
                LOG_INF("dflash profile: target_decode=%.1fms ret=%d\n", (ggml_time_us() - t_tgt_decode_us) / 1e3, ret);
            }
        }
```

**newText:**
```
        const int ret = llama_decode(ctx_tgt, batch_view);
```

- [ ] **Step 3: Verify the edit applied correctly**

Run: `sed -n '4172,4176p' tools/server/server-context.cpp`

**Pass criterion:** You see `const int ret = llama_decode(ctx_tgt, batch_view);` followed by the `// NOTE: do NOT call set_tape_recording` comment. No `profile_tgt` or `int ret = 0` remaining.

Run: `grep -n "profile_tgt" tools/server/server-context.cpp`
**Pass criterion:** no output (the instrumentation is fully removed)

---

### Task 3: Remove sample_accept instrumentation from server-context.cpp

**Why:** The investigation added a profile timer around `common_sampler_sample_and_accept_n`. This must be restored to the original form.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines ~4470-4481)

**Interfaces:**
- Consumes: `slot.smpl`, `slot.ctx_tgt`, `slot.spec_i_batch`, `slot.spec_draft`
- Produces: `auto accepted` (the accepted tokens vector)

- [ ] **Step 1: Read the current code to confirm the exact text**

Run: `sed -n '4468,4482p' tools/server/server-context.cpp`

You should see:
```
                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                static bool profile_verify = [] {
                    const char * e = std::getenv("GGML_DFLASH_PROFILE");
                    return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
                }();
                const int64_t t_verify_us = profile_verify ? ggml_time_us() : 0;
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                if (profile_verify) {
                    LOG_INF("dflash profile: sample_accept=%.1fms accepted=%zu/%zu\n", (ggml_time_us() - t_verify_us) / 1e3, accepted.size(), slot.spec_draft.size());
                }
                slot.spec_i_batch.clear();
```

- [ ] **Step 2: Replace with the original form**

**oldText:**
```
                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                static bool profile_verify = [] {
                    const char * e = std::getenv("GGML_DFLASH_PROFILE");
                    return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
                }();
                const int64_t t_verify_us = profile_verify ? ggml_time_us() : 0;
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                if (profile_verify) {
                    LOG_INF("dflash profile: sample_accept=%.1fms accepted=%zu/%zu\n", (ggml_time_us() - t_verify_us) / 1e3, accepted.size(), slot.spec_draft.size());
                }
                slot.spec_i_batch.clear();
```

**newText:**
```
                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                slot.spec_i_batch.clear();
```

- [ ] **Step 3: Verify the edit applied correctly**

Run: `grep -n "profile_verify" tools/server/server-context.cpp`

**Pass criterion:** The only remaining occurrence of `profile_verify` is in the rollback_reeval timer (Task 4 will remove it). If there are ZERO occurrences, that's also fine (Task 4 may already be done or the reference was in the same block). Check: `grep -c "profile_verify" tools/server/server-context.cpp` should be `1` (the rollback_reeval one) or `0`.

---

### Task 4: Remove rollback_reeval instrumentation from server-context.cpp

**Why:** The investigation added a profile timer around the rollback `llama_decode(ctx_tgt, batch_reeval)`. This must be restored to the original form. Note: this timer references `profile_verify` (the static bool from Task 3's block), so if Task 3 removed that variable, this code will fail to compile. Both must be removed together.

**Files:**
- Modify: `tools/server/server-context.cpp` (lines ~4598-4602)

**Interfaces:**
- Consumes: `ctx_tgt`, `batch_reeval`, `n_reeval`
- Produces: `int ret_reeval` (the reeval decode return code)

- [ ] **Step 1: Read the current code to confirm the exact text**

Run: `grep -n -B2 -A4 "t_reeval_us" tools/server/server-context.cpp`

You should see:
```
                            const int64_t t_reeval_us = profile_verify ? ggml_time_us() : 0;
                            const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
                            if (profile_verify) {
                                LOG_INF("dflash profile: rollback_reeval=%.1fms n_reeval=%d ret=%d\n", (ggml_time_us() - t_reeval_us) / 1e3, n_reeval, ret_reeval);
                            }
```

- [ ] **Step 2: Replace with the original single-line form**

**oldText:**
```
                            const int64_t t_reeval_us = profile_verify ? ggml_time_us() : 0;
                            const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
                            if (profile_verify) {
                                LOG_INF("dflash profile: rollback_reeval=%.1fms n_reeval=%d ret=%d\n", (ggml_time_us() - t_reeval_us) / 1e3, n_reeval, ret_reeval);
                            }
```

**newText:**
```
                            const int ret_reeval = llama_decode(ctx_tgt, batch_reeval);
```

- [ ] **Step 3: Verify all profile instrumentation is gone from server-context.cpp**

Run: `grep -n "profile_tgt\|profile_verify\|t_reeval_us\|t_verify_us\|t_tgt_decode_us\|dflash profile:" tools/server/server-context.cpp`

**Pass criterion:** no output — all instrumentation is fully removed.

---

### Task 5: Remove batch_draft profile print from common/speculative.cpp

**Why:** The investigation added a profile-gated `LOG_INF` print in `common_speculative_draft_batch`. The `LOG_DBG` that was already there is sufficient for debug logging. The profile block must be removed to keep the code clean.

**Files:**
- Modify: `common/speculative.cpp` (lines ~5065-5079)

**Interfaces:**
- Consumes: `t0`, `t1`, `t2`, `t3` (timing variables), `n_ready` (int), `ctx_dft` (llama_context*)
- Produces: none (just removes a print)

- [ ] **Step 1: Read the current code to confirm the exact text**

Run: `sed -n '5064,5082p' common/speculative.cpp`

You should see:
```
    const int64_t t3 = ggml_time_us();

    {
        static bool profile_batch = [] {
            const char * e = std::getenv("GGML_DFLASH_PROFILE");
            return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
        }();
        if (profile_batch) {
            const llama_perf_context_data perf_dft = llama_perf_context(ctx_dft);
            LOG_INF("dflash profile: batch_draft n_ready=%d prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms graph_reuse=%d\n",
                    n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3, (int)perf_dft.n_reused);
        }
    }
    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
```

- [ ] **Step 2: Replace with the original form (just LOG_DBG)**

**oldText:**
```
    const int64_t t3 = ggml_time_us();

    {
        static bool profile_batch = [] {
            const char * e = std::getenv("GGML_DFLASH_PROFILE");
            return e && (strstr(e, "summary") || strstr(e, "1") || strstr(e, "all"));
        }();
        if (profile_batch) {
            const llama_perf_context_data perf_dft = llama_perf_context(ctx_dft);
            LOG_INF("dflash profile: batch_draft n_ready=%d prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms graph_reuse=%d\n",
                    n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3, (int)perf_dft.n_reused);
        }
    }
    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
```

**newText:**
```
    const int64_t t3 = ggml_time_us();

    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
```

- [ ] **Step 3: Verify the edit applied correctly**

Run: `grep -n "profile_batch" common/speculative.cpp`

**Pass criterion:** no output — the profile block is fully removed.

---

### Task 6: Build and verify all instrumentation is removed

**Why:** Tasks 1-5 made edits to three files. This task builds everything together and confirms no debug instrumentation remains. A clean build is required before benchmarking.

**Files:**
- No direct edits — build and grep only

- [ ] **Step 1: Build the project**

Run: `cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2 && cmake --build build -j$(nproc) 2>&1 | tail -5`

**Pass criterion:** Build succeeds with no errors. All targets built (llama-server, llama-cli, etc.).

**Fail criterion:** Compile errors → one of the edits in Tasks 1-5 didn't match. Read the error message, find the file and line, re-read the current code, and fix the edit.

- [ ] **Step 2: Verify no instrumentation remains in any file**

Run these greps and confirm each has NO output:
```bash
grep -n "profile_tgt\|profile_verify\|profile_sync\|profile_batch\|t_reeval_us\|t_verify_us\|t_tgt_decode_us\|t_sched_sync_us\|HYPOTHESIS TEST\|SKIPPED for hypothesis" \
  src/llama-context.cpp tools/server/server-context.cpp common/speculative.cpp
```

**Pass criterion:** no output — all instrumentation is fully removed from all three files.

- [ ] **Step 3: Verify the accept bug fix is still present**

Run: `grep -n "Fall back to curr_impl" common/speculative.cpp`
**Pass criterion:** shows line ~4447 with the comment.

Run: `grep -n "Also set impl_last" common/speculative.cpp`
**Pass criterion:** shows line ~5058 with the comment.

These two checks confirm the accept bug fix (from the investigation) was NOT accidentally removed.

---

### Task 7: Final correctness and performance benchmark

**Why:** Confirm that after all cleanup (Tasks 1-5), the build still produces correct output and the performance improvement from Task 1 (removing sched_synchronize) is preserved.

**Files:**
- No code changes — benchmark only

- [ ] **Step 1: Run 300-token DFlash benchmark on the merge**

```bash
timeout 180 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_merge_final.log &
sleep 25
curl -s -o /tmp/bench_merge_final.json http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Implement a complete binary search tree in C++ with insertion, deletion, in-order traversal, and a print function. Include proper memory management and a main function that demonstrates all operations."}],"max_tokens":300,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance\|graphs reused\|statistics.*dflash" /tmp/bench_merge_final.log
```

**Pass criterion:**
- `eval time` line shows `≥ 5.0 tokens per second` (expected ~6.0-6.2)
- `draft acceptance` shows `≥ 0.30`
- `statistics.*dflash` shows `#acc drafts` > 0 (accept fix working)
- Output JSON has coherent content (no garbage)

- [ ] **Step 2: Run 300-token DFlash benchmark on the fork for comparison**

```bash
timeout 180 /crypt/beellama.cpp/build-vulkan/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /crypt/models/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-draft-n-max 8 -ngl 999 -c 8192 --port 18080 --parallel 1 \
  2>/tmp/bench_fork_final.log &
sleep 25
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Implement a complete binary search tree in C++ with insertion, deletion, in-order traversal, and a print function. Include proper memory management and a main function that demonstrates all operations."}],"max_tokens":300,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time\|draft acceptance\|graphs reused\|statistics.*dflash" /tmp/bench_fork_final.log
```

**Expected:** fork ~19 t/s (this is the reference — it should NOT change since we didn't touch the fork)

- [ ] **Step 3: Run baseline (no DFlash) on the merge to confirm no regression**

```bash
timeout 60 /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2/build/bin/llama-server \
  -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 999 -c 8192 --port 18080 \
  2>/tmp/bench_merge_baseline.log &
sleep 20
curl -s -o /dev/null http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50,"temperature":0}'
kill $(lsof -ti :18080) 2>/dev/null
grep "eval time" /tmp/bench_merge_baseline.log
```

**Pass criterion:** `eval time` shows `≥ 11.0 tokens per second` (expected ~12.5). If it's significantly lower, the cleanup broke something in the non-DFlash path.

- [ ] **Step 4: Produce a comparison table**

Run:
```bash
echo "=== COMPARISON TABLE ==="
echo "| Metric | Merge (fixed) | Fork | Merge baseline |"
echo "|--------|---------------|------|-----------------|"
MERGE_TPS=$(grep "tokens per second" /tmp/bench_merge_final.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)')
FORK_TPS=$(grep "tokens per second" /tmp/bench_fork_final.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)')
BASE_TPS=$(grep "tokens per second" /tmp/bench_merge_baseline.log | tail -1 | grep -oP '[0-9.]+(?= tokens per second)')
echo "| Decode TPS | $MERGE_TPS | $FORK_TPS | $BASE_TPS |"
MERGE_ACC=$(grep "draft acceptance" /tmp/bench_merge_final.log | grep -oP '[0-9.]+(?= \()' | head -1)
FORK_ACC=$(grep "draft acceptance" /tmp/bench_fork_final.log | grep -oP '[0-9.]+(?= \()' | head -1)
echo "| Acceptance | $MERGE_ACC | $FORK_ACC | N/A |"
echo ""
echo "Merge improvement: 4.01 -> $MERGE_TPS t/s"
echo "Remaining gap: merge $MERGE_TPS vs fork $FORK_TPS = $(python3 -c "print(f'{$FORK_TPS/$MERGE_TPS:.1f}x')" 2>/dev/null || echo 'N/A')"
```

**Pass criterion:** The table prints without errors. Merge TPS ≥ 5.0. The remaining gap is documented for future work.

---

## Future Work (NOT in this plan — documented for reference)

This plan eliminates the #1 bottleneck (sched_synchronize = 50% of eval time) and improves 4.01→~6.15 t/s. The remaining 3.1× gap (6.15 vs 19.39 t/s) has three identified causes:

### 1. 80% rollback reeval rate (28% of eval time)
The merge does `llama_dflash_rollback` + `llama_decode` reeval for every partial draft acceptance. 60 of 75 verify cycles need reeval, each costing ~231ms. The fork also does reeval but its reeval is faster (no sync overhead). Possible fix: skip reeval for RS (Recurrent State) contexts where `n_rollback <= llama_n_rs_seq(ctx_tgt)`, matching the fork's MTP path (`use_ckpt_tgt` check).

### 2. GPU contention from pipelining (batch_draft 54→135ms)
Without the explicit sync, the GPU processes the target decode and draft decode back-to-back, causing contention that slows the draft from 54ms to 135ms. Fix: wire a Vulkan timeline semaphore between the compute queue (hidden capture) and transfer queue (D2D ring write) — this orders work GPU-side without blocking the CPU.

### 3. Fork per-cycle profiling needed
We don't have the fork's per-component profile. The fork's cycle is 170ms with 3.3 tokens/cycle. We need to instrument the fork with the same timers to understand how it achieves 19.39 t/s. Key questions: is the fork's target decode faster (different graph)? Does the fork pipeline better? Does the fork have fewer rollback reevals?

### 4. Phase 5 no-op (dflash_vk_backend_wait_for_stream)
The phase 5 commit (ae43a1e64) made `dflash_vk_backend_wait_for_stream` a no-op. This is correct now that `sched_synchronize` is removed — the no-op prevents a redundant `device.waitIdle()`. In the future, this should be replaced with a proper Vulkan timeline semaphore for GPU-side cross-queue ordering.