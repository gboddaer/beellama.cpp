#pragma once

#include "common.h"
#include "llama.h"
#include "src/dflash-profile.h"
#include "src/llama-ext.h"
#include "src/llama-memory.h"

#include <cstdint>
#include <string>
#include <vector>

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
