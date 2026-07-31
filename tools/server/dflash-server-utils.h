#pragma once

#include "common.h"
#include "llama.h"
#include "src/dflash-profile.h"
#include "src/llama-ext.h"
#include "src/llama-memory.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

// Thread-safe atomic flag — set once at init, never toggled mid-decode
extern std::atomic<bool> g_enabled;

// Cheap atomic check (~1ns) — use at every hook site
inline bool enabled() {
    return g_enabled.load(std::memory_order_acquire);
}

// Initialization (called once at server startup)
void init();

// Cleanup (called once at server shutdown)
void shutdown();

// Model loading
bool is_dflash_model(const llama_model * model);
llama_model * load_dflash_drafter(const char * path, const llama_model_params & params);

// Slot state management
struct dflash_slot_state {
    llama_context * dft_ctx = nullptr;
    llama_batch dft_batch;
    std::vector<llama_token> draft_tokens;
    // Batch index in the target batch where each draft token's logits land.
    // Populated by the pre-decode hook when draft tokens are appended to the batch.
    std::vector<int32_t> draft_batch_idx;
    // Absolute target KV position of the first draft token in this cycle.
    // Used by rollback() to remove the correct KV range.
    llama_pos base_pos = -1;
    int n_draft = 0;     // number of draft tokens generated this cycle
    int n_accepted = 0;  // number of draft tokens accepted this cycle
    bool active = false; // true for the lifetime of the slot's generation phase
};

void init_slot_dflash(dflash_slot_state & state);
void cleanup_slot_dflash(dflash_slot_state & state);

// Decode integration
// seed: the last accepted/sampled target token (the draft context's input).
//          The draft model decodes from this token to predict the next ones.
// seq_id: the sequence id used in the draft context's KV cache (typically slot.id).
llama_tokens generate_draft(llama_context * ctx_dft, dflash_slot_state & state,
                            llama_token seed, llama_seq_id seq_id, int n_draft = 1);

// Rollback rejected draft tokens from the target KV cache.
// seq_id: the target context sequence id (typically slot.id).
void rollback(llama_context * ctx_tgt, dflash_slot_state & state, llama_seq_id seq_id);

// Verify draft tokens against target logits.
// Returns the number of accepted draft tokens (0 to draft_tokens.size()).
// On full/partial acceptance the caller must advance the draft context KV with
// the accepted tokens. On rejection, rollback() must be called and the target
// logits at the rejection position provide the corrected token.
int verify_draft(llama_context * ctx_tgt, dflash_slot_state & state);

// Advance the draft context KV cache by feeding it a token (accepted or resampled).
// This keeps the draft model in sync with the target after each decode step.
bool sync_draft_ctx(llama_context * ctx_dft, llama_token token, llama_seq_id seq_id);

} // namespace dflash

// DFlash server utility functions
// These encapsulate DFlash-specific logic separate from core server code

namespace dflash_server_utils {

// Profile and trace enabled checks
bool is_profile_enabled(uint32_t flags);
bool is_crash_trace_enabled();
bool is_rx_diag_enabled();
bool is_token_trace_enabled();
bool is_verify_padding_enabled();
bool is_shared_drafter_batch_disabled();

// Model and device checks
bool is_dflash_drafter(const llama_model * model);
bool is_backend_dev_type_gpu(enum ggml_backend_dev_type type);
bool is_backend_dev_gpu(ggml_backend_dev_t dev);
bool is_dflash_shared_output_compatible(ggml_backend_dev_t dev);
bool supports_device_buffer(const llama_model * model, ggml_backend_dev_t dev);
bool single_explicit_draft_device(const common_params & params, ggml_backend_dev_t & dev);
bool is_tensor_split_experimental_enabled();

// Slot and view management
bool slot_in_view(const void * slot, const llama_batch & view);
bool view_has_unexpected_prompt_logits(const void * slot);
void log_reduced_verify_decision(const void * slot);
bool batch_view_is_reduced_verify(const llama_batch & view);

// Effective draft max calculations
int flat_effective_draft_max(llama_context * ctx_dft, int n_draft_max);
int effective_adaptive_base_n_max(int base_n_max);

// Reasoning and speculative checks
bool reasoning_budget_state_is_reasoning(int state);
bool accept_info_is_reasoning(const void * info);
bool speculative_uses_fork_slot_impls(const common_params_speculative & speculative);

// Tool call marker checks
bool tail_pos_is_in_code_fence(const std::string & text, size_t pos);
bool tail_tool_marker_has_boundary(const std::string & text, size_t pos);
bool tail_has_tool_call_marker(const std::string & text, size_t scan_from);

// Speculative result checks
bool flat_result_has_bonus(const void * result);

// Reduced verify sampler chain check
bool reduced_sampler_chain_supported(const void * chain);

} // namespace dflash_server_utils
