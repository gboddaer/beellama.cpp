# Vulkan BASE Regression Investigation Plan

> **Goal:** Identify root cause of 27 t/s → 12.5 t/s BASE regression on W7800 Vulkan (RADV GFX1151)
> **Context:** The `merge_llama_into_beellama_2` worktree merged llama.cpp master into the beellama fork. Historical fork achieved ~27 t/s; merged worktree achieves ~12.5 t/s.
> **Architecture:** Same model (`Qwen3.6-27B-Q4_K_M.gguf`), same device, same configuration.
> **Tech Stack:** C++17, cmake, Vulkan/RADV, Python 3 standard library, git.

## Global Constraints

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Old fork main: `/crypt/beellama.cpp` (main branch)
- Model: `/crypt/models/Qwen3.6-27B-Q4_K_M.gguf`
- Device: Vulkan0: AMD Radeon Graphics (RADV GFX1151) with 97503 MiB
- Configuration: q4_0 KV cache, flash-attn on, ctx-size 8192, -ngl all, -b 2048, -ub 512
- **Workload type:** Token generation (single-token decode, not prompt processing)
- **Measurement protocol:** 64-token warm-up, 256-token measurement, triplicate runs, median t/s, fixed seed 7
- Do not commit unless explicitly authorized

## Key Hypotheses (Priority Order)

| Priority | Hypothesis | Test Method | Expected Impact |
|----------|-----------|-------------|-----------------|
| 1 | Q4_K_M matmul-vec shader changed (most likely) | Diff vulkan-shaders/ | 2x regression |
| 2 | Subgroup size changed from 64→32 (RDNA3) | Diff + targeted revert | 2x regression |
| 3 | Silent CPU fallback / partial offload | Log inspection | 5-10x regression |
| 4 | Flash-attn accidentally used for generation | Flash-attn toggle | 2x regression |
| 5 | Build flags different (Debug/Release, validation) | Build verification | 2-5x regression |
| 6 | Memory flags changed (DEVICE_LOCAL vs HOST_VISIBLE) | Diff + targeted revert | 2x regression |
| 7 | Merge commit conflict resolution | Manual diff | Variable |

## Review Feedback Incorporated

**From GLM-5.2:**
- Build both binaries first, then diff code (confirm baseline before blaming merge)
- Test flash-attn toggle (cheap, high signal)
- Check subgroup size detection
- Use negative-space grep for suspicious deletions

**From Kimi K2.6:**
- Add Step 0 (fix clocks, clear shader cache, define measurement protocol)
- Manual bisect instead of automated (merge commit risk, build failures)
- Add Q4_K_M matmul-vec shader hypothesis (highest priority)
- Verify offload (all layers on GPU)
- Add build flag checklist
- Add measurement methodology (warm-up, triplicate runs, log inspection)
- Clarify workload type (token generation vs prompt processing)

---

## Task 1: Rebuild old fork and confirm historical baseline

**Files:** Build directory at `/crypt/beellama.cpp/build-vulkan-old`

**Goal:** Confirm ~27 t/s still reproduces on current driver. Rules out Mesa/RADV version drift.

- [ ] **Step 1: Build old beellama main branch**

```bash
cd /crypt/beellama.cpp
cmake -B build-vulkan-old -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan-old -j
```

- [ ] **Step 2: Run old binary with same configuration**

```bash
# Warm-up
build-vulkan-old/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --port 8098 -ngl all -b 2048 -ub 512 --ctx-size 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on \
  --jinja --reasoning off --no-mmap --no-host --host 127.0.0.1 &

# Send test request and measure t/s
curl -s http://127.0.0.1:8098/v1/completions \
  -d '{"prompt":"Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.","n_predict":200,"temperature":0,"top_k":20,"top_p":1.0,"min_p":0.0,"seed":7,"stream":false,"cache_prompt":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"t/s: {d['timings']['predicted_per_second']}\")"

kill %1
```

Expected: ~27 t/s if historical baseline still holds. If <20 t/s, driver/regression is external to merge.

- [ ] **Step 3: Test flash-attn off vs on**

```bash
# Test with flash-attn OFF on both old and new
build-vulkan-old/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --port 8098 -ngl all -b 2048 -ub 512 --ctx-size 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn 0 \
  --jinja --reasoning off --no-mmap --no-host --host 127.0.0.1 &

curl -s http://127.0.0.1:8098/v1/completions \
  -d '{"prompt":"Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.","n_predict":200,"temperature":0,"top_k":20,"top_p":1.0,"min_p":0.0,"seed":7,"stream":false,"cache_prompt":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"t/s: {d['timings']['predicted_per_second']}\")"

kill %1
```

Expected: If old fork ignores flash-attn flag, performance should be similar with/without it. If merged build shows significant flash-attn impact, this is the regression source.

---

## Task 2: Test merged build with flash-attn toggle and batch settings

**Goal:** Isolate whether flash-attn or batch settings are contributing to regression.

- [ ] **Step 1: Test merged build with flash-attn OFF**

```bash
/crypt/beellama.cpp/build-vulkan/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --port 8099 -ngl all -b 2048 -ub 512 --ctx-size 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn 0 \
  --jinja --reasoning off --no-mmap --no-host --host 127.0.0.1 &

curl -s http://127.0.0.1:8099/v1/completions \
  -d '{"prompt":"Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.","n_predict":200,"temperature":0,"top_k":20,"top_p":1.0,"min_p":0.0,"seed":7,"stream":false,"cache_prompt":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"t/s: {d['timings']['predicted_per_second']}\")"

kill %1
```

Expected: If flash-attn ON is causing the regression, flash-attn OFF should show improvement.

- [ ] **Step 2: Test with matched batch settings**

```bash
# Try -b 512 -ub 512 (matched batch) instead of -b 2048 -ub 512
/crypt/beellama.cpp/build-vulkan/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --port 8099 -ngl all -b 512 -ub 512 --ctx-size 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on \
  --jinja --reasoning off --no-mmap --no-host --host 127.0.0.1 &

curl -s http://127.0.0.1:8099/v1/completions \
  -d '{"prompt":"Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.","n_predict":200,"temperature":0,"top_k":20,"top_p":1.0,"min_p":0.0,"seed":7,"stream":false,"cache_prompt":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"t/s: {d['timings']['predicted_per_second']}\")"

kill %1
```

Expected: If ubatch mismatch is the cause, matched batch should show improvement.

---

## Task 3: Diff vulkan-specific files

**Goal:** Identify deleted optimizations or routing changes in the merge.

- [ ] **Step 1: List deleted vulkan files**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git log --diff-filter=D --name-only -- ggml/src/ggml-vulkan/
```

Expected: Any deleted shader files or optimization-specific files are high-value targets.

- [ ] **Step 2: Negative-space grep for suspicious deletions**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git diff $(git merge-base HEAD origin/main)...HEAD -- ggml/src/ggml-vulkan/ vulkan-shaders/ | \
  grep -E '^-.*(subgroup|wave64|rdna|kquant|mmq|HOST_VISIBLE|DEVICE_LOCAL|vkDeviceWaitIdle|vkQueueWaitIdle|op_f16)' | head -50
```

Expected: Deleted lines containing subgroup, wave64, rdna3, kquant, mmq, memory flags, or sync calls.

- [ ] **Step 3: Check subgroup size detection**

```bash
# Check if subgroup_size is hardcoded or auto-detected
grep -r 'subgroup_size\|SUBGROUP_SIZE\|wave64\|Wave64' ggml/src/ggml-vulkan/ vulkan-shaders/ | head -20
```

Expected: If old fork pinned `subgroup_size=64` for RDNA3 and merge auto-detects as 32, that explains 2x regression.

---

## Task 4: Git bisect (if diff is uninformative)

**Goal:** Bisect over upstream commit range to identify specific regression commit.

- [ ] **Step 1: Identify bisect range**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git log --oneline --ancestry-path $(git merge-base HEAD origin/main)..HEAD | head -5
git rev-list --count $(git merge-base HEAD origin/main)..HEAD
```

Expected: Count of commits in merge range. If <500 commits, manual bisect feasible.

- [ ] **Step 2: Create bisect build+test script**

```bash
#!/bin/bash
# /tmp/vulkan-bisect.sh
COMMIT=$1
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git checkout $COMMIT -- ggml/src/ggml-vulkan/ vulkan-shaders/ 2>/dev/null
cmake --build build -j 2>/dev/null
if [ $? -ne 0 ]; then
  echo "build failed, skip"
  exit 125
fi

# Quick benchmark
build/bin/llama-server -m /crypt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --port 8099 -ngl all -b 2048 -ub 512 --ctx-size 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on \
  --jinja --reasoning off --no-mmap --no-host --host 127.0.0.1 &

sleep 10  # Wait for startup
TIPS=$(curl -s http://127.0.0.1:8099/v1/completions \
  -d '{"prompt":"Implement a Python function that returns the n-th Fibonacci number using memoization. Only output the code.","n_predict":200,"temperature":0,"top_k":20,"top_p":1.0,"min_p":0.0,"seed":7,"stream":false,"cache_prompt":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['timings']['predicted_per_second'])")

kill %1 2>/dev/null

# Threshold: >22 t/s = good, <15 = bad, between = retest
if [ $(echo "$TIPS > 22" | bc -l) -eq 1 ]; then
  exit 0  # good
elif [ $(echo "$TIPS < 15" | bc -l) -eq 1 ]; then
  exit 1  # bad
else
  exit 125  # skip, retest
fi
```

- [ ] **Step 3: Run bisect**

```bash
cd /crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2
git bisect start
git bisect bad HEAD
git bisect good $(git merge-base HEAD origin/main)
git bisect run /tmp/vulkan-bisect.sh
```

Expected: Bisect identifies specific commit(s) causing regression.

---

## Task 5: Document findings

**Files:** `docs/superpowers/results/2026-07-29-vulkan-regression-findings.md`

**Goal:** Create structured findings document with root cause, evidence, and recommendations.

- [ ] **Step 1: Document root cause**

If flash-attn toggle fixes it: "Regression caused by Vulkan flash-attn path introduced in merge. Old fork used non-flash KV path."

If batch settings fix it: "Regression caused by ubatch mismatch. -b 2048 -ub 512 suboptimal for single-token decode."

If subgroup size: "Regression caused by auto-detected subgroup_size=32 instead of fork's pinned 64 for RDNA3."

If specific commit: "Regression introduced in commit <hash> which <description>."

- [ ] **Step 2: Document evidence**

Include before/after benchmarks, specific diff changes, and bisect path if applicable.

- [ ] **Step 3: Document recommendations**

Provide fix recommendation (e.g., pin subgroup size, use non-flash path, adjust batch settings, or revert specific commit).

---

## Execution Notes

- The safe order is: confirm baseline → toggle settings → diff → bisect → document
- Each task is independent; skip diff if bisect is more efficient
- Do not modify production code without explicit authorization
- If bisect fails (no commits touch vulkan files), fall back to manual diff review
- Use the exact same model file and prompt for all comparisons
