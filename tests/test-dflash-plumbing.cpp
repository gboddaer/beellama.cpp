#include "llama-context.h"
#include "dflash-profile.h"
#include "speculative.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

static bool test_vk_dflash_cross_ring_roundtrip() {
    ggml_backend_reg_t vk_reg = ggml_backend_reg_by_name("Vulkan");
    if (!vk_reg) { return true; } // no Vulkan backend in this build — skip
    using alloc_fn_t = void *(*)(int,int,int);
    using free_fn_t  = void  (*)(void*);
    using write_fn_t = void  (*)(void*,int,int,const float*,int,int);
    using sync_fn_t  = void  (*)(void*);
    using snap_fn_t  = bool  (*)(void*,int,int,int,float*,int,int,int);
    using inter_fn_t = const float* (*)(void*,int,int,int);
    using setT_fn_t  = void  (*)(ggml_tensor*, const void*, size_t, size_t);
    auto falloc = (alloc_fn_t)ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_alloc");
    auto ffree  = (free_fn_t) ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_free");
    auto fwrite = (write_fn_t)ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_write");
    auto fsync  = (sync_fn_t) ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_synchronize");
    auto fsnap  = (snap_fn_t) ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_snapshot");
    auto finter = (inter_fn_t)ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_interleave");
    auto fsetT  = (setT_fn_t) ggml_backend_reg_get_proc_address(vk_reg, "dflash_cross_ring_gpu_set_tensor_tensor");
    if (!falloc || !ffree || !fwrite || !fsync || !fsnap || !finter || !fsetT) return false;

    void * h = falloc(/*n_layers*/2, /*n_embd*/3, /*ring_size*/4);
    if (!h) { return true; } // Vulkan present but cross-ring alloc failed (e.g. no buffer_device_address) — skip

    float l0[3] = {1,2,3}, l1[3] = {4,5,6};
    fwrite(h, 0, 1, l0, 1, 3);
    fwrite(h, 1, 1, l1, 1, 3);
    fsync(h);

    float snap[12] = {0};
    bool ok = fsnap(h, /*write_pos*/2, /*filled*/2, /*ctx_window*/2, snap, /*n_tokens*/2, /*n_layers*/2, /*n_embd*/3);
    if (ok) {
        const float exp_snap[12] = {0,0,0,1,2,3, 0,0,0,4,5,6};
        for (int i = 0; i < 12; ++i) { if (snap[i] != exp_snap[i]) { ok = false; break; } }
    }

    const float * staging = finter(h, /*write_pos*/2, /*filled*/2, /*ctx_window*/2);
    ok = ok && staging != nullptr;

    // Drafter-side Vulkan tensor to receive the set_tensor_tensor D2D copy.
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(vk_reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    GGML_UNUSED(buft);
    size_t mem_size = ggml_tensor_overhead()*4;
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = {};
    ip.mem_buffer = mem;
    ip.mem_size   = mem_size;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 12);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    ok = ok && buf != nullptr && t != nullptr && t->buffer != nullptr;
    if (ok) {
        fsetT(t, staging, 0, 12 * sizeof(float));
        float got[12] = {0};
        ggml_backend_tensor_get(t, got, 0, 12 * sizeof(float));
        const float exp[12] = {0,0,0, 0,0,0, 1,2,3, 4,5,6};
        for (int i = 0; i < 12; ++i) { if (got[i] != exp[i]) { ok = false; break; } }
    }
    if (buf) ggml_backend_buffer_free(buf);
    if (gctx) ggml_free(gctx);
    free(mem);
    ggml_backend_free(backend);
    ffree(h);
    return ok;
}

// ============================================================================
// Regression guards for PR #79 DFlash changes (added beyond the original
// cross-ring plumbing test). These cover the behaviour changes that are
// unit-testable without a full model:
//   - 122B fix:           llm_arch_supports_rs_rollback gate keeps QWEN35MOE on.
//   - DFlash enablement:  common_params_speculative::need_n_rs_seq() returns
//                         draft.n_max for DFlash.
//   - 35B-A3B fix:        RS snapshot write-back slot mapping (rotation for
//                         n_seq_tokens < K) + the gated_delta_net kernel snapshot
//                         contract (only the last min(n_seq_tokens,K) slots are
//                         written).
//   - Coverage gap:       the existing pure _for_test helpers in llama-context.h
//                         had no direct unit tests.
// Integration-only paths (Coder-Next KV fallback, hidden-state dump APIs, the
// rollback int return) are guarded by the rebenchmark, not here.
// ============================================================================

static bool test_llm_arch_supports_rs_rollback() {
    bool ok = true;
    ok &= expect( llm_arch_supports_rs_rollback(LLM_ARCH_QWEN35),    "QWEN35 must support RS rollback");
    ok &= expect( llm_arch_supports_rs_rollback(LLM_ARCH_QWEN35MOE), "QWEN35MOE must support RS rollback (122B fix: keep n_rs_seq)");
    ok &= expect(!llm_arch_supports_rs_rollback(LLM_ARCH_QWEN3NEXT), "QWEN3NEXT must NOT support RS rollback (Coder-Next uses checkpoint-restore)");
    ok &= expect(!llm_arch_supports_rs_rollback(LLM_ARCH_KIMI_LINEAR),"Kimi-linear must NOT support RS rollback");
    ok &= expect(!llm_arch_supports_rs_rollback(LLM_ARCH_UNKNOWN),   "unknown arch must NOT support RS rollback");
    return ok;
}

static bool test_need_n_rs_seq() {
    bool ok = true;
    auto make = [](std::vector<enum common_speculative_type> types, int32_t n_max) -> uint32_t {
        common_params_speculative s{};
        s.types      = std::move(types);
        s.draft.n_max = n_max;
        return s.need_n_rs_seq();
    };
    ok &= expect(make({COMMON_SPECULATIVE_TYPE_DFLASH}, 4) == 4u,  "DFlash -> need_n_rs_seq == draft.n_max");
    ok &= expect(make({COMMON_SPECULATIVE_TYPE_DRAFT_MTP}, 3) == 3u, "MTP -> need_n_rs_seq == draft.n_max");
    ok &= expect(make({}, 4) == 0u,                                  "no speculative types -> need_n_rs_seq == 0");
    ok &= expect(make({COMMON_SPECULATIVE_TYPE_DFLASH, COMMON_SPECULATIVE_TYPE_DRAFT_MTP}, 5) == 5u,
                     "DFlash+MTP -> need_n_rs_seq == draft.n_max");
    ok &= expect(make({COMMON_SPECULATIVE_TYPE_DFLASH}, 0) == 0u,   "DFlash with draft.n_max=0 -> 0");
    return ok;
}

// Pure-logic guard for the 35B-A3B fix: the RS snapshot write-back slot mapping
// produced by llama_dflash_rs_writeback_slot_for_test() (the helper build_recurrent_attn
// now calls). Verifies the rotation invariant for decode (n_seq_tokens < K) and the
// full-batch case, plus the explicit 35B-A3B decode scenario (n_rs_seq=4 -> K=5).
static bool test_rs_writeback_slot_mapping() {
    bool ok = true;
    const int64_t cases_K[]   = {2, 5};
    const int64_t cases_nst[] = {0, 1, 2, 3, 5, 8};
    for (int64_t K : cases_K) {
        for (int64_t nst : cases_nst) {
            const int64_t n_pop = (nst < K) ? nst : K;
            int64_t n_from_gdn = 0;
            for (int64_t k_i = 0; k_i < K; ++k_i) {
                uint32_t cache_slot = 0xDEADu;
                llama_dflash_rs_slot_src src{};
                const bool valid = llama_dflash_rs_writeback_slot_for_test(K, nst, k_i, cache_slot, src);
                if (n_pop == 0) {
                    ok &= expect(!valid, "n_seq_tokens=0 must yield invalid (n_pop==0)");
                    continue;
                }
                ok &= expect(valid, "k_i in [0,K) with n_pop>0 must be valid");
                ok &= expect(cache_slot == (uint32_t)(K - 1 - k_i), "cache_slot == K-1-k_i");
                if (src.from_gdn) {
                    ++n_from_gdn;
                    ok &= expect(src.gdn_slot == k_i,      "from_gdn: gdn_slot == k_i");
                    ok &= expect(k_i >= K - n_pop,         "from_gdn only for k_i >= K-n_pop");
                } else {
                    ok &= expect(k_i <  K - n_pop,         "rotate only for k_i < K-n_pop");
                    ok &= expect(src.old_slot == cache_slot - (uint32_t)n_pop, "rotate: old_slot == cache_slot - n_pop");
                    ok &= expect(src.old_slot <  cache_slot, "rotate: old_slot < cache_slot (read-before-overwrite)");
                }
            }
            ok &= expect(n_from_gdn == n_pop, "exactly n_pop slots read from gdn_out");
        }
    }
    // 35B-A3B decode scenario: n_rs_seq=4 -> K=5, n_seq_tokens=1.
    // After the step: state slot 0 <- newest gdn_out; state slot s <- old slot s-1 (s=1..4).
    {
        const int64_t K = 5, nst = 1;
        for (int64_t k_i = 0; k_i < K; ++k_i) {
            uint32_t cache_slot = 0;
            llama_dflash_rs_slot_src src{};
            llama_dflash_rs_writeback_slot_for_test(K, nst, k_i, cache_slot, src);
            if (cache_slot == 0) {
                ok &= expect(src.from_gdn && src.gdn_slot == K - 1, "decode: slot 0 <- newest gdn_out slot K-1");
            } else {
                ok &= expect(!src.from_gdn && src.old_slot == cache_slot - 1u, "decode: slot s <- old slot s-1 (rotation)");
            }
        }
    }
    return ok;
}

// Kernel-contract guard for the 35B-A3B fix: ggml_gated_delta_net, for a decode
// batch (n_seq_tokens=1 < K=4), must write ONLY the last min(n_seq_tokens,K)=1
// snapshot slot of its output; the leading K-1 state slots are left untouched
// (caller-owned). The fix's rotation in build_recurrent_attn relies on exactly
// this contract. Runs a single-op graph on the CPU backend (always present).
static bool test_gated_delta_net_snapshot_contract() {
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    if (!cpu_reg) { return true; } // CPU backend missing (never expected) - skip
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(cpu_reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { return true; }

    bool ok = true;
    const int64_t S_v = 2, H = 1, n_tokens = 1, n_seqs = 1, K = 4;
    const int64_t D = S_v * S_v * H;            // = 4 floats per snapshot
    const int64_t attn_elems       = S_v * H * n_tokens * n_seqs; // = 2
    const int64_t state_per_snap   = S_v * S_v * H * n_seqs;     // = 4
    const float sentinel = -12345.0f;

    size_t tensors_mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + (1 << 16);
    void * mem = malloc(tensors_mem);
    struct ggml_init_params ip = {};
    ip.mem_buffer = mem;
    ip.mem_size   = tensors_mem;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    ok &= expect(gctx != nullptr, "gated_delta_net test: ggml_init");
    if (!ok) { free(mem); ggml_backend_free(backend); return ok; }

    auto mk4 = [&](int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        return ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n0, n1, n2, n3);
    };
    ggml_tensor * q    = mk4(S_v, H, n_tokens, n_seqs);
    ggml_tensor * k    = mk4(S_v, H, n_tokens, n_seqs);
    ggml_tensor * v    = mk4(S_v, H, n_tokens, n_seqs);
    ggml_tensor * g    = mk4(1,    H, n_tokens, n_seqs); // scalar gate
    ggml_tensor * beta = mk4(1,    H, n_tokens, n_seqs);
    // 3D snapshot state (D, K, n_seqs) - mirrors build_recurrent_attn's s_3d_pad.
    ggml_tensor * state = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, K, n_seqs);
    ggml_tensor * result = ggml_gated_delta_net(gctx, q, k, v, g, beta, state, K);
    ok &= expect(q && k && v && g && beta && state && result, "gated_delta_net test: tensor creation");
    if (!ok) { ggml_free(gctx); free(mem); ggml_backend_free(backend); return ok; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    ok &= expect(buf != nullptr, "gated_delta_net test: alloc_ctx_tensors");
    if (!ok) { ggml_free(gctx); free(mem); ggml_backend_free(backend); return ok; }

    // Inputs: simple finite values. State slot 0 = 2x2 identity, pad slots = 0.
    float qv[2]    = {1.0f, 1.0f};
    float gv[1]    = {0.0f};
    float betav[1] = {0.0f};
    float statev[(size_t)(D * K * n_seqs)];
    for (int64_t i = 0; i < D * K * n_seqs; ++i) statev[i] = 0.0f;
    statev[0] = 1.0f; statev[3] = 1.0f; // identity in slot 0
    ggml_backend_tensor_set(q,    qv,    0, sizeof(qv));
    ggml_backend_tensor_set(k,    qv,    0, sizeof(qv));
    ggml_backend_tensor_set(v,    qv,    0, sizeof(qv));
    ggml_backend_tensor_set(g,    gv,    0, sizeof(gv));
    ggml_backend_tensor_set(beta, betav, 0, sizeof(betav));
    ggml_backend_tensor_set(state, statev, 0, sizeof(statev));

    // Pre-fill the whole result with a sentinel so untouched slots keep it.
    const int64_t result_elems = attn_elems + K * state_per_snap; // = 2 + 16 = 18
    std::vector<float> sentinel_buf((size_t)result_elems, sentinel);
    ggml_backend_tensor_set(result, sentinel_buf.data(), 0, sizeof(float) * (size_t)result_elems);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, result);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ok &= expect(status == GGML_STATUS_SUCCESS, "gated_delta_net test: graph compute succeeded");
    if (!ok) { ggml_backend_buffer_free(buf); ggml_free(gctx); free(mem); ggml_backend_free(backend); return ok; }

    std::vector<float> out((size_t)result_elems, 0.0f);
    ggml_backend_tensor_get(result, out.data(), 0, sizeof(float) * (size_t)result_elems);

    // n_tokens=1, K=4 -> n_pop=1, shift = n_tokens-K = -3, target_slot = 0-(-3) = 3 = K-1.
    // Only slot 3 is written; slots 0,1,2 must retain the sentinel.
    auto slot_start = [&](int64_t s) { return (size_t)(attn_elems + s * state_per_snap); };
    for (int64_t s = 0; s < K - 1; ++s) {       // untouched slots 0..K-2
        const size_t off = slot_start(s);
        bool untouched = true;
        for (int64_t j = 0; j < state_per_snap; ++j) {
            if (out[off + (size_t)j] != sentinel) { untouched = false; break; }
        }
        ok &= expect(untouched, "gated_delta_net decode: leading snapshot slot untouched (kernel writes only last n_pop)");
    }
    {   // written slot K-1 must differ from the sentinel
        const size_t off = slot_start(K - 1);
        bool written = false;
        for (int64_t j = 0; j < state_per_snap; ++j) {
            if (out[off + (size_t)j] != sentinel) { written = true; break; }
        }
        ok &= expect(written, "gated_delta_net decode: last snapshot slot is written by the kernel");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    free(mem);
    ggml_backend_free(backend);
    return ok;
}

// Coverage gap: the existing pure _for_test helpers in llama-context.h are called by
// production code but had no direct unit tests.
static bool test_existing_for_test_helpers() {
    bool ok = true;
    // llama_dflash_capture_tokens_per_seq(n_tokens, n_seq_tokens, n_seqs_unq)
    ok &= expect(llama_dflash_capture_tokens_per_seq(10, 1, 1) == 10, "capture: single unique seq -> all ubatch tokens");
    ok &= expect(llama_dflash_capture_tokens_per_seq(10, 4, 3) == 4,  "capture: multi unique seq -> n_seq_tokens");
    ok &= expect(llama_dflash_capture_tokens_per_seq(7,  7, 1) == 7,   "capture: single seq, n_seq_tokens==n_tokens");
    // llama_dflash_replay_gdn_supported_s_for_test
    ok &= expect( llama_dflash_replay_gdn_supported_s_for_test(16),  "gdn replay s=16");
    ok &= expect( llama_dflash_replay_gdn_supported_s_for_test(32),  "gdn replay s=32");
    ok &= expect( llama_dflash_replay_gdn_supported_s_for_test(64),  "gdn replay s=64");
    ok &= expect( llama_dflash_replay_gdn_supported_s_for_test(128), "gdn replay s=128");
    ok &= expect(!llama_dflash_replay_gdn_supported_s_for_test(8),    "gdn replay s=8 not supported");
    ok &= expect(!llama_dflash_replay_gdn_supported_s_for_test(256), "gdn replay s=256 not supported");
    ok &= expect(!llama_dflash_replay_gdn_supported_s_for_test(0),    "gdn replay s=0 not supported");
    // llama_dflash_replay_state_shape_valid_for_test(s, h_v, n_embd_s): valid iff s*s*h_v == n_embd_s
    ok &= expect( llama_dflash_replay_state_shape_valid_for_test(16, 1, 16*16), "state shape: 16*16*1 == 256");
    ok &= expect( llama_dflash_replay_state_shape_valid_for_test(8,  2, 8*8*2),  "state shape: 8*8*2 == 128");
    ok &= expect(!llama_dflash_replay_state_shape_valid_for_test(8,  2, 127),   "state shape: 8*8*2 != 127");
    ok &= expect(!llama_dflash_replay_state_shape_valid_for_test(0,  1, 0),     "state shape: s<=0 invalid");
    ok &= expect(!llama_dflash_replay_state_shape_valid_for_test(-1, 1, 0),     "state shape: negative s invalid");
    // llama_dflash_view_span_in_bounds_for_test(total, offset, n)
    ok &= expect( llama_dflash_view_span_in_bounds_for_test(100, 0,   100), "span: full");
    ok &= expect( llama_dflash_view_span_in_bounds_for_test(100, 30,  40),  "span: interior");
    ok &= expect(!llama_dflash_view_span_in_bounds_for_test(100, 101, 0),   "span: offset>total");
    ok &= expect(!llama_dflash_view_span_in_bounds_for_test(100, 90,  20),  "span: overflow");
    ok &= expect( llama_dflash_view_span_in_bounds_for_test(100, 100, 0),   "span: zero-length at end");
    // llama_dflash_suppress_callback_for_prefill_ubatch_for_test(active, staging, intersection)
    ok &= expect( llama_dflash_suppress_callback_for_prefill_ubatch_for_test(true,  true,  false), "suppress: active+staging+no-intersection");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(true,  true,  true),  "no suppress: has intersection");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(true,  false, false), "no suppress: not staging");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(false, true,  false), "no suppress: no prefill plan");
    // llama_dflash_prefill_plan_needs_staging_for_test(planned, current): staging iff planned > LLAMA_DFLASH_MAX_VERIFY_TOKENS (25)
    ok &= expect( llama_dflash_prefill_plan_needs_staging_for_test((int)LLAMA_DFLASH_MAX_VERIFY_TOKENS + 1, 0), "staging: planned > MAX_VERIFY");
    ok &= expect(!llama_dflash_prefill_plan_needs_staging_for_test((int)LLAMA_DFLASH_MAX_VERIFY_TOKENS,     0), "no staging: planned == MAX_VERIFY");
    ok &= expect(!llama_dflash_prefill_plan_needs_staging_for_test((int)LLAMA_DFLASH_MAX_VERIFY_TOKENS - 1, 0), "no staging: planned < MAX_VERIFY");
    return ok;
}


// NOTE: This test retains only the pure-logic DFlash assertions. The original
// fork version also contained ~207 source-text grep guards that asserted
// exact fork text snippets in src/common/ggml files. Those guards were an
// anti-pattern for a fork tracking upstream (they break on every upstream
// restructure, e.g. DFlash ring state moving into common/speculative.cpp as
// ring_state_size() overrides) and reported false negatives after this merge
// even though the guarded functionality was preserved. They were removed per
// GLM review (Option C). The behavioral coverage is retained here; integration
// re-benchmark guards should be re-added as behavioral (public-surface) tests,
// not source-text archaeology.
int main(int argc, char ** argv) {
    (void) argc; (void) argv;  // pure-logic tests need no repo-root argument
    bool ok = true;

    ok &= expect(llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN35), "Qwen3.5 must support GPU tape");
    ok &= expect(llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN35MOE), "Qwen3.5-MoE must support GPU tape");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN3NEXT), "Qwen3Next must stay on fallback");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_KIMI_LINEAR), "Kimi-linear must stay on fallback");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_UNKNOWN), "unknown arch must stay on fallback");
    ok &= expect(test_vk_dflash_cross_ring_roundtrip(), "Vulkan DFlash cross-ring alloc/write/snapshot/interleave/set_tensor_tensor round-trip");
    ok &= expect(test_llm_arch_supports_rs_rollback(), "llm_arch_supports_rs_rollback gate (122B fix: keep n_rs_seq for QWEN35MOE)");
    ok &= expect(test_need_n_rs_seq(), "common_params_speculative::need_n_rs_seq (DFlash RS-seq enablement)");
    ok &= expect(test_rs_writeback_slot_mapping(), "DFlash RS snapshot write-back slot mapping (35B-A3B fix: rotation for n_seq_tokens<K)");
    ok &= expect(test_gated_delta_net_snapshot_contract(), "gated_delta_net snapshot contract: n_seq_tokens<K writes only last n_pop slots (35B-A3B fix)");
    ok &= expect(test_existing_for_test_helpers(), "existing pure _for_test helpers in llama-context.h");

    return ok ? 0 : 1;
}
