#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn_supported(const ggml_tensor * dst);

bool ggml_cuda_kvarn_fused_require_enabled();

void ggml_cuda_flash_attn_ext_kvarn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

const char * ggml_cuda_fa_build_policy();

bool ggml_cuda_fa_pair_compiled(ggml_type type_K, ggml_type type_V);
