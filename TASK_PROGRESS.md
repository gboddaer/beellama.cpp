
## Phase 5f: HIGH/MEDIUM Issues Resolution (2026-07-01)

### GLM Adversarial Review Findings
GLM-5.2 review of Phase 5d/5e revealed the integration was structurally
complete but functionally broken (8 critical issues).

### Critical Fixes (Phase 5f commit 1: 0ddf59a1d)
- C1: generate_draft() was dead code — fixed with seed parameter
- C5: rollback() hardcoded seq_id=0 — fixed with seq_id parameter
- C6: rollback() cleared active=false — fixed to keep active=true
- C7: rollback() position math wrong — fixed with absolute base_pos
- C8: pre-decode ran during prompt processing — gated on GENERATING
- H1: verify_draft() wrong batch index — fixed with draft_batch_idx
- H2: verify_draft() only checked one token — full chain walk
- C2: draft context KV never synced — added sync_draft_ctx()
- Added test-dflash-decode.cpp: 18 unit tests, all passing

### Major Architectural Fix (Phase 5f commit 2: f7588b7b1)
GLM Option B: Removed redundant dflash-server-utils decode path.

DISCOVERY: The merged server ALREADY has the complete real DFlash
cross-attention implementation via common_speculative framework:
- common_speculative_impl_dflash (common/speculative.cpp:2064)
- llama_set_dflash_capture(ctx_tgt, ...) sets up target hidden capture
- build_cross_data() feeds target hidden states to ctx_dft via
  llama_set_cross_data_seq() — REAL cross-attention
- GPU cross-attention ring (llama_dflash_cross_ring_gpu_set_cross)
- Ring buffer of target hidden states (captured via eval callback)

The dflash-server-utils decode path was a REDUNDANT parallel
implementation that bypassed cross-attention. Removed it; let
common_speculative handle DFlash entirely.

This resolves H4 (cross-attention) and H5 (seq_id sync) — the real
implementation handles both correctly.

### MEDIUM Fixes (Phase 5f commit 2: f7588b7b1)
- M1: Named constant DFLASH_DRAFT_CAP instead of magic 8
- M2: Renamed loop var n_draft_max to avoid shadowing state.n_draft
- M6: Added concurrency note about shared draft contexts

### Final Verification
- All 4 backends build: 0 errors (CUDA, ROCm, Vulkan, Debug)
- DFlash flow intact via common_speculative (8/8 components verified)
- Redundant hooks removed (4/4 verified removed)
- test-dflash-decode: 18/18 unit tests passing
- No test regressions

### Status: DFlash integration uses REAL cross-attention path
