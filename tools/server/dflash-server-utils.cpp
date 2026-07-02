#include "dflash-server-utils.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "src/dflash-profile.h"
#include "src/llama-ext.h"
#include "src/llama-memory.h"

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <filesystem>
#include <utility>

// DFlash server utility implementations

namespace dflash {

std::atomic<bool> g_enabled{false};

void init() {
    // Check if DFlash is enabled via params or env
    // For now, default to disabled until params are parsed
    g_enabled.store(false, std::memory_order_release);
}

void shutdown() {
    g_enabled.store(false, std::memory_order_release);
}

bool is_dflash_model(const llama_model * model) {
    (void)model;
    // TODO: Check GGUF metadata for dflash.drafter=true or dflash.target=true
    return false;
}

llama_model * load_dflash_drafter(const char * path, const llama_model_params & params) {
    (void)path;
    (void)params;
    // TODO: Load draft model with appropriate parameters
    return nullptr;
}

void init_slot_dflash(dflash_slot_state & state) {
    state.dft_ctx = nullptr;
    state.dft_batch = llama_batch_init(512, 0, 1);
    state.n_draft = 0;
    state.n_accepted = 0;
    state.active = false;
}

void cleanup_slot_dflash(dflash_slot_state & state) {
    if (state.dft_ctx) {
        llama_free(state.dft_ctx);
        state.dft_ctx = nullptr;
    }
    // Check if batch was allocated (token pointer non-null after init)
    if (state.dft_batch.token != nullptr) {
        llama_batch_free(state.dft_batch);
        state.dft_batch = llama_batch{};
    }
    state.n_draft = 0;
    state.n_accepted = 0;
    state.active = false;
}

llama_tokens generate_draft(llama_context * ctx_dft, dflash_slot_state & state,
                            llama_token seed, llama_seq_id seq_id, int n_draft) {
    state.draft_tokens.clear();
    state.draft_batch_idx.clear();
    state.n_draft = 0;
    state.n_accepted = 0;
    state.base_pos = -1;
    
    if (!ctx_dft || n_draft <= 0) {
        return state.draft_tokens;
    }
    
    // Cap n_draft to avoid batch overflow (TODO: make configurable)
    n_draft = std::min(n_draft, 8);
    
    // Seed the chain with the last accepted target token. The draft model decodes
    // from this token to predict the next ones. The seed itself is NOT a draft
    // token (it is already in the target KV); it is only fed to the draft context.
    llama_token cur = seed;
    if (cur < 0) {
        // No valid seed token available (e.g. before any target token exists)
        return state.draft_tokens;
    }
    
    // Autoregressive draft generation: loop n_draft times
    for (int i = 0; i < n_draft; ++i) {
        // Create a single-token batch for the draft context
        llama_batch b = llama_batch_get_one(&cur, 1);
        
        // Decode on the draft context
        int rc = llama_decode(ctx_dft, b);
        if (rc != 0) {
            // rc < 0: fatal; rc > 0: batch full / context exceeded — stop drafting
            if (rc < 0) {
                fprintf(stderr, "dflash: draft decode failed at step %d, rc=%d\n", i, rc);
            }
            break;
        }
        
        // Argmax over the draft vocab to get the next token.
        // llama_get_logits_argmax_ith returns int32_t* (pointer to argmax array).
        // This is a DFlash-specific API available in this fork.
        const int32_t * argmax_ptr = llama_get_logits_argmax_ith(ctx_dft, b.n_tokens - 1);
        if (!argmax_ptr) {
            // Argmax not available — stop drafting
            break;
        }
        state.draft_tokens.push_back((llama_token) *argmax_ptr);
        state.n_draft++;
        
        // The next draft step feeds the just-generated draft token
        cur = state.draft_tokens.back();
    }
    
    (void) seq_id; // reserved for draft-context KV seq management
    return state.draft_tokens;
}

void rollback(llama_context * ctx_tgt, dflash_slot_state & state, llama_seq_id seq_id) {
    // Remove rejected draft tokens from target KV cache.
    // Draft tokens were added at absolute positions [base_pos, base_pos + n_draft).
    // Accepted ones [base_pos, base_pos + n_accepted) stay; rejected ones are removed.
    int n_reject = state.n_draft - state.n_accepted;
    if (n_reject > 0 && ctx_tgt && state.base_pos >= 0) {
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        if (mem) {
            llama_pos p0 = state.base_pos + (llama_pos) state.n_accepted;
            llama_pos p1 = state.base_pos + (llama_pos) state.n_draft;
            llama_memory_seq_rm(mem, seq_id, p0, p1);
        }
    }
    
    // Reset per-cycle counters, but keep `active` true so drafting continues
    // for the lifetime of the slot's generation phase (GLM review C6).
    state.draft_tokens.clear();
    state.draft_batch_idx.clear();
    state.n_draft = 0;
    state.n_accepted = 0;
    state.base_pos = -1;
}

int verify_draft(llama_context * ctx_tgt, dflash_slot_state & state) {
    if (!ctx_tgt || state.draft_tokens.empty()) {
        return 0;
    }
    
    // Walk the draft chain and accept the longest prefix that matches the
    // target's argmax at the corresponding batch position.
    int n_accepted = 0;
    for (size_t i = 0; i < state.draft_tokens.size(); ++i) {
        if (i >= state.draft_batch_idx.size()) break;
        int32_t batch_idx = state.draft_batch_idx[i];
        if (batch_idx < 0) break;
        
        const int32_t * argmax_ptr = llama_get_logits_argmax_ith(ctx_tgt, batch_idx);
        if (!argmax_ptr) {
            break; // argmax unavailable — stop accepting
        }
        llama_token tgt_tok = (llama_token) *argmax_ptr;
        if (tgt_tok != state.draft_tokens[i]) {
            break; // mismatch — stop accepting here
        }
        n_accepted++;
    }
    
    state.n_accepted = n_accepted;
    return n_accepted;
}

bool sync_draft_ctx(llama_context * ctx_dft, llama_token token, llama_seq_id seq_id) {
    if (!ctx_dft || token < 0) return false;
    
    llama_batch b = llama_batch_get_one(&token, 1);
    int rc = llama_decode(ctx_dft, b);
    if (rc != 0) {
        fprintf(stderr, "dflash: draft context sync failed, rc=%d\n", rc);
        return false;
    }
    (void) seq_id;
    return true;
}

} // namespace dflash

namespace dflash_server_utils {

// Profile and trace enabled checks
bool is_profile_enabled(uint32_t flags) {
    return dflash_profile_enabled(flags);
}

bool is_crash_trace_enabled() {
    return false; // TODO: implement
}

bool is_rx_diag_enabled() {
    return false; // TODO: implement
}

bool is_token_trace_enabled() {
    return false; // TODO: implement
}

bool is_verify_padding_enabled() {
    return false; // TODO: implement
}

bool is_shared_drafter_batch_disabled() {
    return false; // TODO: implement
}

// Model and device checks
bool is_dflash_drafter(const llama_model * model) {
    if (!model) return false;
    // Check if model has DFlash draft metadata
    return false; // TODO: implement
}

bool is_backend_dev_type_gpu(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU;
}

bool is_backend_dev_gpu(ggml_backend_dev_t dev) {
    if (!dev) return false;
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
}

bool is_dflash_shared_output_compatible(ggml_backend_dev_t dev) {
    if (!dev) return false;
    // Check if device supports DFlash shared output tensors
    return is_backend_dev_gpu(dev);
}

bool supports_device_buffer(const llama_model * model, ggml_backend_dev_t dev) {
    if (!model || !dev) return false;
    // Check if model supports device buffer on this device
    return false; // TODO: implement
}

bool single_explicit_draft_device(const common_params & params, ggml_backend_dev_t & dev) {
    if (params.speculative.draft.devices.empty()) return false;
    if (params.speculative.draft.devices.size() > 1) return false;
    dev = params.speculative.draft.devices[0];
    return true;
}

bool is_tensor_split_experimental_enabled() {
    return false; // TODO: implement
}

// Slot and view management
bool slot_in_view(const void * slot, const llama_batch & view) {
    (void)slot;
    (void)view;
    // TODO: implement
    return false;
}

bool view_has_unexpected_prompt_logits(const void * slot) {
    (void)slot;
    // TODO: implement
    return false;
}

void log_reduced_verify_decision(const void * slot) {
    (void)slot;
    // TODO: implement
}

bool batch_view_is_reduced_verify(const llama_batch & view) {
    (void)view;
    // TODO: implement
    return false;
}

// Effective draft max calculations
int flat_effective_draft_max(llama_context * ctx_dft, int n_draft_max) {
    if (!ctx_dft) return n_draft_max;
    // Get effective block size from draft context
    // Effective max = block_size - 1
    return n_draft_max; // TODO: implement
}

int effective_adaptive_base_n_max(int base_n_max) {
    return base_n_max; // TODO: implement
}

// Reasoning and speculative checks
bool reasoning_budget_state_is_reasoning(int state) {
    return state == 1; // TODO: use proper enum
}

bool accept_info_is_reasoning(const void * info) {
    (void)info;
    // TODO: implement
    return false;
}

bool speculative_uses_fork_slot_impls(const common_params_speculative & speculative) {
    (void)speculative;
    return false; // TODO: implement
}

// Tool call marker checks
bool tail_pos_is_in_code_fence(const std::string & text, size_t pos) {
    if (pos >= text.size()) return false;
    // Check if position is inside a code fence
    // Look for ``` before pos and matching ``` after pos
    size_t fence_start = text.rfind("```", pos);
    if (fence_start == std::string::npos) return false;
    size_t fence_end = text.find("```", fence_start + 3);
    if (fence_end == std::string::npos) return false;
    return pos > fence_start && pos < fence_end;
}

bool tail_tool_marker_has_boundary(const std::string & text, size_t pos) {
    if (pos >= text.size()) return false;
    // Check for tool call marker boundaries
    return false; // TODO: implement
}

bool tail_has_tool_call_marker(const std::string & text, size_t scan_from) {
    if (scan_from >= text.size()) return false;
    // Check for tool call markers in text
    return false; // TODO: implement
}

// Speculative result checks
bool flat_result_has_bonus(const void * result) {
    (void)result;
    // TODO: implement
    return false;
}

// Reduced verify sampler chain check
bool reduced_sampler_chain_supported(const void * chain) {
    (void)chain;
    // TODO: implement
    return false;
}

} // namespace dflash_server_utils
