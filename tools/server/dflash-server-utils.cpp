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
