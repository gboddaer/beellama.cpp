# Handoff: Vulkan DFlash Task 4 Checkpoint

## Scope

This handoff closes the current investigation stage. Do not make another speculative lifecycle change before returning to `superpowers:systematic-debugging` Phase 1 and tracing the invalid logits index to its source.

## Worktree state

- Worktree: `/crypt/beellama.cpp/.worktrees/merge_llama_into_beellama_2`
- Branch: `merge_llama_into_beellama_2`
- HEAD: `1ce1505d3` (`harness: remove debug prints from repeated_requests.py`)
- Last commits with the lifecycle work:
  - `2b85a8780` `server: unconditional speculative reset with virtual reset_request hook`
  - `0c1415750` `harness: strict validation, provenance, and error handling`
- Existing historical handoff: `docs/superpowers/HANDOFF-vulkan-speculative-correctness-performance-recovery.md`
- Do not commit or push without explicit user approval.

## Current uncommitted code change

Only `tools/server/server-context.cpp` is intentionally modified:

```cpp
if (slot.prompt.n_tokens() > 0) {
    common_speculative_reset(slot.get_spec(), slot.id);
}
```

This restores the original `n_tokens() > 0` guard around the new virtual reset dispatch introduced in `2b85a8780`. The draft KV sequence is still cleared for speculative slots. The virtual `reset_request()` hook and its MTP/DFlash implementations remain in the committed code.

Do not treat this uncommitted guard as a demonstrated fix. It is the requested rollback of the unconditional reset behavior.

## Fresh verification at this checkpoint

### Build and harness

Command:

```bash
cmake --build build-vulkan --target llama-server -j"$(nproc)"
python3 -m unittest bench.vulkan-gap.test_harness_strict -v
```

Result:

```text
build_exit=0
Ran 10 tests in 0.007s
OK
```

### DFlash `-np 1` queued-request smoke

A server with `-np 1`, DFlash, `--spec-draft-n-max 8`, `-b 512`, `-ub 128`, and context 2048 accepted two submitted requests per round for three rounds. All six HTTP completions returned normally with nonzero `draft_n`; their `finish_reason` was `length` because the smoke used `n_predict=32`.

This is an operational no-crash smoke only. It does not establish EOS behavior or output quality. It was run after restoring the guarded reset behavior.

### DFlash `-np 2` concurrent smoke: BLOCKED

Fresh command configuration:

```text
Qwen3.6-27B-Q4_K_M.gguf
Qwen3.6-27B-DFlash-Q4_K_M.gguf
--spec-type dflash --spec-draft-n-max 8 --kv-unified
-np 2 -b 512 -ub 128 --ctx-size 2048
--cache-type-k q4_0 --cache-type-v q4_0
-ngl all --flash-attn on --reasoning off
```

Two concurrent `/v1/completions` requests (coding and math prompts, `n_predict=32`, temperature 0, top-k 20) both failed with:

```text
RemoteDisconnected: Remote end closed connection without response
```

The server then exited on the known assertion:

```text
common/sampling.cpp:154: GGML_ASSERT(logits != nullptr) failed
get_logits_ith: invalid logits id 9, reason: batch.logits[9] != true
```

Evidence log: `/tmp/task4-guard-np2.log`.

Therefore Task 4 is not complete and no performance work should begin.

## Known non-results and corrections

- Do not call `finish_reason=length` at `n_predict=32` a correctness failure by itself. It only shows that the test reached its requested output cap.
- Do not use DFlash implementation counters (`#calls(b,g,a)` or `#acc drafts`) alone as an acceptance diagnosis. They are tied to `impl_last` accounting and are not a proof of target-token acceptance for this incident.
- The previous session's `-np 1`, context-4096 EOS check was reported as passing with `n_predict=256`, three requests, `finish_reason=stop`, and nonzero drafts. It was not rerun after this guard rollback, so preserve it as historical evidence rather than a current release gate.
- No `-np 2` test passed after this guard rollback.

## Next-stage investigation

Use `superpowers:systematic-debugging` before changing code.

1. Reproduce the failing `-np 2` smoke exactly from the configuration above.
2. Capture one concise trace at the batch-construction to target-logits boundary for both slots. At minimum record, per slot:
   - `slot.id`, sequence id, `slot.i_batch`, and complete `slot.spec_i_batch`.
   - target batch token count and each relevant `batch.logits[]` value.
   - sub-batch offset and token count passed to `post_decode`.
   - context `output_ids` mapping for each requested logits index.
3. Trace where index 9 is marked non-logits or otherwise removed. Do not add a null-logits fallback: it masks the correctness failure and corrupts sampling.
4. Compare the two-slot batch construction against the working one-slot path. Form one source-level hypothesis before applying a change.
5. Add a focused regression test or deterministic test hook before implementing a fix.
6. After a fix, rerun the exact `-np 2` concurrent smoke, then the `-np 1` EOS/repeated-request check with a sufficient `n_predict` cap. Only then continue the plan's performance gates.

## Cleanup

- The temporary smoke driver is `/tmp/task4-np2-smoke.sh`.
- The server started by that smoke was terminated by its cleanup trap.
- Keep `/tmp/task4-guard-np2.log` until the next investigator has extracted the required trace.
