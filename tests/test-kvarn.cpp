#include "llama-kvarn.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-kvarn: %s\n", msg);
        std::abort();
    }
}

static void test_type_table() {
    const int supported_bits[] = { 2, 3, 4, 5, 6, 8 };

    require(llama_kvarn_type_count() == 37, "unexpected KVarN type count");

    const llama_kvarn_type_desc * disabled = llama_kvarn_type_desc_from_name("off");
    require(disabled != nullptr, "disabled type name did not parse");
    require(disabled->type == LLAMA_KVARN_TYPE_DISABLED, "disabled type enum mismatch");
    require(disabled->key_bits == 0 && disabled->value_bits == 0, "disabled bits mismatch");
    require(disabled->group == 128, "disabled group mismatch");

    for (int key_bits : supported_bits) {
        for (int value_bits : supported_bits) {
            const std::string name = "kvarn_k" + std::to_string(key_bits) + "v" + std::to_string(value_bits) + "_g128";
            const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_name(name.c_str());
            require(desc != nullptr, "expected type name did not parse");
            require(desc->type != LLAMA_KVARN_TYPE_DISABLED && desc->type != LLAMA_KVARN_TYPE_INVALID, "parsed type enum mismatch");
            require(desc->key_bits == key_bits, "parsed key bits mismatch");
            require(desc->value_bits == value_bits, "parsed value bits mismatch");
            require(desc->group == 128, "parsed group mismatch");

            const llama_kvarn_type_desc * by_type = llama_kvarn_type_desc_from_type(desc->type);
            require(by_type != nullptr, "expected enum did not map to descriptor");
            require(std::string(by_type->name) == name, "enum descriptor name mismatch");
        }
    }

    require(llama_kvarn_type_desc_from_name("kvarn_k7v2_g128") == nullptr, "invalid type parsed");
}

static void test_tile_layout() {
    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_type(type);
        require(desc != nullptr, "layout type descriptor missing");

        const llama_kvarn_tile_layout layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
        require(layout.k_payload_bytes == size_t(2048 * desc->key_bits), "K payload bytes mismatch");
        require(layout.v_payload_bytes == size_t(2048 * desc->value_bits), "V payload bytes mismatch");
        require(layout.tile_bytes == size_t(2048 * (desc->key_bits + desc->value_bits) + 1536), "tile bytes mismatch");
        require(layout.k_s_col_off == layout.k_payload_off + layout.k_payload_bytes, "K scale offset mismatch");
        require(layout.v_payload_off == layout.k_s_row_off + 128 * sizeof(uint16_t), "V payload offset mismatch");
        require(layout.v_s_col_off == layout.v_payload_off + layout.v_payload_bytes, "V scale offset mismatch");
        require(layout.tile_bytes % 8 == 0, "tile bytes not 8-byte aligned");
    }
}

static void test_head_dimension_slicing() {
    require(llama_kvarn_head_slices(128) == 1, "128-dim head should use one KVarN slice");
    require(llama_kvarn_head_slices(256) == 2, "256-dim head should use two KVarN slices");
    require(llama_kvarn_head_slices(512) == 4, "512-dim head should use four KVarN slices");
    require(llama_kvarn_head_slices(384) == 3, "384-dim head should use three KVarN slices");
    require(llama_kvarn_head_slices(64)  == 0, "64-dim head is not KVarN slice-compatible");
    require(llama_kvarn_head_slices(513) == 0, "non-128-multiple head is not KVarN slice-compatible");
}

static void test_runtime_validation() {
    llama_kvarn_runtime_requirements supported = {};
    supported.attention_supported = true;
    supported.head_dims_supported = true;
    supported.n_seq_max = 1;
    supported.kv_unified = false;

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const auto params = llama_kvarn_params_for_type(type);
        require(llama_kvarn_validate_runtime(params, supported) == nullptr, "valid runtime rejected");
    }

    auto invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.key_bits = 3;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "mismatched preset bits accepted");

    invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.sink_tokens = 0;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "unsupported sink tokens accepted");

    auto requirements = supported;
    requirements.attention_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported attention accepted");

    requirements = supported;
    requirements.head_dims_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported head dimension accepted");

    requirements = supported;
    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified single-sequence runtime rejected");

    requirements = supported;
    requirements.n_seq_max = 2;
    requirements.kv_unified = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "non-unified multi-sequence runtime rejected");

    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified multi-sequence runtime rejected");
}

static void test_remove_policy() {
    require(llama_kvarn_can_remove_range(-1, 0, -1, 128), "empty sequence removal rejected");
    require(llama_kvarn_can_remove_range(783, -1, -1, 128), "full sequence removal with negative range rejected");
    require(llama_kvarn_can_remove_range(783, 0, -1, 128), "full sequence removal from zero rejected");
    require(llama_kvarn_can_remove_range(783, 0, 784, 128), "explicit full sequence range removal rejected");
    require(!llama_kvarn_can_remove_range(783, 0, 640, 128), "old compressed partial removal accepted");
    require(llama_kvarn_can_remove_range(783, 640, -1, 128), "current/previous tail removal rejected");
}

static void test_pack_roundtrip(int bits) {
    const int n = 257;
    std::vector<uint8_t> values(n);
    for (int i = 0; i < n; ++i) {
        values[i] = uint8_t((i * 7 + 3) & ((1 << bits) - 1));
    }

    std::vector<uint8_t> packed(llama_kvarn_packed_bytes(n, bits), 0);
    llama_kvarn_pack_bits(values.data(), n, bits, packed.data());

    for (int i = 0; i < n; ++i) {
        const uint8_t got = llama_kvarn_unpack_bits_value(packed.data(), i, bits);
        if (got != values[i]) {
            std::fprintf(stderr, "test-kvarn: %d-bit roundtrip mismatch at %d: got %u expected %u\n",
                    bits, i, unsigned(got), unsigned(values[i]));
            std::abort();
        }
    }
}

static void test_hadamard_roundtrip() {
    std::vector<float> values(128);
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
        values[i] = std::sin(float(i) * 0.19f) + float(i - 64) * 0.002f;
    }
    expected = values;

    llama_kvarn_hadamard_128(values.data());
    llama_kvarn_hadamard_128(values.data());

    for (int i = 0; i < 128; ++i) {
        require(std::fabs(values[i] - expected[i]) < 1e-5f, "Hadamard roundtrip mismatch");
    }
}

static float tile_rmse(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "RMSE shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(std::sqrt(sum / a.size()));
}

static void test_tile_quantization(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    require(desc != nullptr, "quantization type descriptor missing");

    const auto layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] =
                std::sin(float(r) * 0.071f) +
                std::cos(float(c) * 0.113f) +
                float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] =
                std::cos(float(r) * 0.057f) -
                std::sin(float(c) * 0.091f) +
                float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, desc->key_bits, layout, record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, desc->value_bits, layout, record.data());

    std::vector<float> k_dequant(k.size());
    std::vector<float> v_dequant(v.size());
    llama_kvarn_dequantize_k_tile(record.data(), desc->key_bits, layout, k_dequant.data());
    llama_kvarn_dequantize_v_tile(record.data(), desc->value_bits, layout, v_dequant.data());

    for (size_t i = 0; i < k.size(); ++i) {
        require(std::isfinite(k_dequant[i]), "K dequant produced non-finite value");
        require(std::isfinite(v_dequant[i]), "V dequant produced non-finite value");
    }

    const float max_rmse[] = { 0.0f, 0.0f, 0.40f, 0.22f, 0.12f, 0.08f, 0.05f, 0.0f, 0.025f };
    require(tile_rmse(k, k_dequant) < max_rmse[desc->key_bits], "K tile RMSE too high");
    require(tile_rmse(v, v_dequant) < max_rmse[desc->value_bits], "V tile RMSE too high");
}

static ggml_backend_t init_test_backend(enum ggml_backend_dev_type device_type, bool required) {
    const char * backend_name = std::getenv("GGML_KVARN_TEST_BACKEND");
    const bool use_named_gpu = backend_name != nullptr && backend_name[0] != '\0' && device_type == GGML_BACKEND_DEVICE_TYPE_GPU;

    ggml_backend_t backend = use_named_gpu ?
        ggml_backend_init_by_name(backend_name, nullptr) :
        ggml_backend_init_by_type(device_type, nullptr);
    if (backend == nullptr && !required) {
        return nullptr;
    }
    require(backend != nullptr, use_named_gpu ? "failed to initialize GGML_KVARN_TEST_BACKEND" : "failed to initialize requested backend");
    return backend;
}

static void set_kvarn_store_legacy_env(bool enabled) {
#ifdef _WIN32
    _putenv_s("GGML_KVARN_STORE_LEGACY", enabled ? "1" : "");
#else
    if (enabled) {
        setenv("GGML_KVARN_STORE_LEGACY", "1", 1);
    } else {
        unsetenv("GGML_KVARN_STORE_LEGACY");
    }
#endif
}

static void set_kvarn_mat_generic_env(bool enabled) {
#ifdef _WIN32
    _putenv_s("GGML_KVARN_MAT_GENERIC", enabled ? "1" : "");
#else
    if (enabled) {
        setenv("GGML_KVARN_MAT_GENERIC", "1", 1);
    } else {
        unsetenv("GGML_KVARN_MAT_GENERIC");
    }
#endif
}

static void test_cache_ops(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_tokens = 385;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, 0, 1, bits, false);
    if (device_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        // Regression guard: materialize live groups are per-execution state and
        // must be derived from the current indices tensor, not frozen op params.
        materialized->op_params[4] = 0;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN graph compute failed");

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

    double sink_error = 0.0;
    double compressed_error = 0.0;
    double previous_tail_error = 0.0;
    double live_tail_error = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * 128 + d]) - double(output[t * 128 + d]);
            if (t < 128) {
                sink_error += diff * diff;
            } else if (t < 256) {
                compressed_error += diff * diff;
            } else if (t < 384) {
                previous_tail_error += diff * diff;
            } else {
                live_tail_error += diff * diff;
            }
        }
    }
    sink_error = std::sqrt(sink_error / (128 * 128));
    compressed_error = std::sqrt(compressed_error / (128 * 128));
    previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
    live_tail_error = std::sqrt(live_tail_error / 128);
    require(sink_error < 0.01, "sink reconstruction error too high");
    require(compressed_error < 0.25, "compressed reconstruction error too high");
    require(previous_tail_error < 0.01, "previous tail reconstruction error too high");
    require(live_tail_error < 0.01, "live tail reconstruction error too high");

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    require(std::any_of(record_data.begin(), record_data.end(), [](uint8_t v) { return v != 0; }),
            "completed group was not flushed");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_cache_ops_multi_stream(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_stream = 2;
    constexpr int kv_size = 512;
    constexpr int n_groups_per_stream = kv_size / 128;
    constexpr int n_tokens_per_stream = 385;
    constexpr int n_tokens = n_tokens_per_stream * n_stream;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, indices, n_tokens_per_stream, 0, n_stream, bits, false);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate multi-stream KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                input[(s * n_tokens_per_stream + t) * 128 + d] =
                    std::sin(float(d) * 0.071f + float(s) * 0.31f) +
                    std::cos(float(t) * 0.037f + float(s) * 0.23f) +
                    float((d * 13 + t * 17 + s * 19) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            idx[s * n_tokens_per_stream + t] = int64_t(s * kv_size + t);
        }
    }
    std::vector<uint8_t> zeros(std::max(ggml_nbytes(stage), ggml_nbytes(records)), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multi-stream KVarN graph compute failed");

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

    for (int s = 0; s < n_stream; ++s) {
        double sink_error = 0.0;
        double compressed_error = 0.0;
        double previous_tail_error = 0.0;
        double live_tail_error = 0.0;
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                const size_t input_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const size_t output_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const double diff = double(input[input_off]) - double(output[output_off]);
                if (t < 128) {
                    sink_error += diff * diff;
                } else if (t < 256) {
                    compressed_error += diff * diff;
                } else if (t < 384) {
                    previous_tail_error += diff * diff;
                } else {
                    live_tail_error += diff * diff;
                }
            }
        }
        sink_error = std::sqrt(sink_error / (128 * 128));
        compressed_error = std::sqrt(compressed_error / (128 * 128));
        previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
        live_tail_error = std::sqrt(live_tail_error / 128);
        require(sink_error < 0.01, "multi-stream sink reconstruction error too high");
        require(compressed_error < 0.25, "multi-stream compressed reconstruction error too high");
        require(previous_tail_error < 0.01, "multi-stream previous tail reconstruction error too high");
        require(live_tail_error < 0.01, "multi-stream live tail reconstruction error too high");
    }

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t stream_record_bytes = size_t(record_bytes) * n_groups_per_stream * n_heads;
    for (int s = 0; s < n_stream; ++s) {
        const auto begin = record_data.begin() + ptrdiff_t(s * stream_record_bytes);
        const auto end = begin + ptrdiff_t(stream_record_bytes);
        require(std::any_of(begin, end, [](uint8_t v) { return v != 0; }),
                "multi-stream completed group was not flushed");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<uint8_t> test_store_records(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        bool           legacy,
        int            n_stream = 2,
        int            n_heads = 2,
        int            n_tokens_per_stream = 385,
        int            start_idx = 64,
        bool           discontinuous_indices = false,
        bool           seed_stage = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize legacy parity context");

    const int n_groups_per_stream = std::max(1, (start_idx + n_tokens_per_stream + (discontinuous_indices ? 1 : 0) + 127) / 128);
    const int n_tokens = n_tokens_per_stream * n_stream;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, value);
    stored->op_params[3] = n_tokens_per_stream;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate legacy parity tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < 128; ++d) {
                    input[((s * n_tokens_per_stream + t) * n_heads + h) * 128 + d] =
                        std::sin(float(d) * 0.071f + float(h) * 0.13f + float(s) * 0.31f) +
                        std::cos(float(t) * 0.037f + float(h) * 0.11f + float(s) * 0.23f) +
                        float((d * 13 + h * 7 + t * 17 + s * 19) % 31 - 15) * 0.01f;
                }
            }
        }
    }

    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            const int local_idx = start_idx + t + (discontinuous_indices && t >= n_tokens_per_stream / 2 ? 1 : 0);
            idx[s * n_tokens_per_stream + t] = int64_t(s * n_groups_per_stream * 128 + local_idx);
        }
    }

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    if (seed_stage) {
        std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
        for (size_t i = 0; i < stage_data.size(); ++i) {
            const float f = std::sin(float(i) * 0.017f) + std::cos(float(i) * 0.011f);
            stage_data[i] = ggml_fp32_to_fp16(f);
        }
        ggml_backend_tensor_set(stage, stage_data.data(), 0, ggml_nbytes(stage));
    } else {
        std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
        ggml_backend_tensor_set(stage, stage_zeros.data(), 0, ggml_nbytes(stage));
    }
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(records, record_zeros.data(), 0, ggml_nbytes(records));

    set_kvarn_store_legacy_env(legacy);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "legacy parity graph compute failed");
    set_kvarn_store_legacy_env(false);

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return record_data;
}

static std::vector<ggml_fp16_t> test_materialize_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        bool           generic,
        int            n_stream,
        int            n_heads,
        int            n_tokens_per_stream,
        int            start_idx) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize materialize parity context");

    const int n_groups_per_stream = std::max(4, (start_idx + n_tokens_per_stream + 127) / 128);
    const int n_tokens = n_tokens_per_stream * n_stream;
    const int n_kv = start_idx + n_tokens_per_stream;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, value);
    stored->op_params[3] = n_tokens_per_stream;
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_kv, 0, n_stream, bits, value);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate materialize parity tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < 128; ++d) {
                    input[((s * n_tokens_per_stream + t) * n_heads + h) * 128 + d] =
                        std::sin(float(d) * 0.071f + float(h) * 0.13f + float(s) * 0.31f) +
                        std::cos(float(t) * 0.037f + float(h) * 0.11f + float(s) * 0.23f) +
                        float((d * 13 + h * 7 + t * 17 + s * 19) % 31 - 15) * 0.01f;
                }
            }
        }
    }

    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            idx[s * n_tokens_per_stream + t] = int64_t(s * n_groups_per_stream * 128 + start_idx + t);
        }
    }

    std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
    for (size_t i = 0; i < stage_data.size(); ++i) {
        const float f = std::sin(float(i) * 0.017f) + std::cos(float(i) * 0.011f);
        stage_data[i] = ggml_fp32_to_fp16(f);
    }
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, stage_data.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    set_kvarn_mat_generic_env(generic);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "materialize parity graph compute failed");
    set_kvarn_mat_generic_env(false);

    std::vector<ggml_fp16_t> output(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, output.data(), 0, ggml_nbytes(materialized));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static void test_store_legacy_parity_gpu() {
    ggml_backend_t backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (backend == nullptr) {
        return;
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<uint8_t> modern = test_store_records(backend, bits, value, false);
            const std::vector<uint8_t> legacy = test_store_records(backend, bits, value, true);
            require(modern == legacy, "KVarN CUDA store records differ from legacy path");
        }
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<uint8_t> modern = test_store_records(
                    backend, bits, value, false, 1, 2, 16, 504, false, true);
            const std::vector<uint8_t> legacy = test_store_records(
                    backend, bits, value, true, 1, 2, 16, 504, false, true);
            require(modern == legacy, "KVarN CUDA direct-flush records differ from legacy path");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<uint8_t> modern = test_store_records(
                backend, 4, value, false, 1, 2, 385, 64, true, false);
        const std::vector<uint8_t> legacy = test_store_records(
                backend, 4, value, true, 1, 2, 385, 64, true, false);
        require(modern == legacy, "KVarN CUDA stale workspace hint fallback differs from legacy path");
    }

    set_kvarn_store_legacy_env(false);
    ggml_backend_free(backend);
}

static void test_materialize_generic_parity_gpu() {
    ggml_backend_t backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (backend == nullptr) {
        return;
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> fast = test_materialize_output(
                    backend, bits, value, false, 2, 2, 385, 64);
            const std::vector<ggml_fp16_t> generic = test_materialize_output(
                    backend, bits, value, true, 2, 2, 385, 64);
            require(fast == generic, "KVarN CUDA materialize fast path differs from generic path");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> fast = test_materialize_output(
                backend, 4, value, false, 1, 2, 17, 504);
        const std::vector<ggml_fp16_t> generic = test_materialize_output(
                backend, 4, value, true, 1, 2, 17, 504);
        require(fast == generic, "KVarN CUDA materialize tail fast path differs from generic path");
    }

    set_kvarn_mat_generic_env(false);
    ggml_backend_free(backend);
}

int main() {
    ggml_backend_load_all();

    test_type_table();
    test_tile_layout();
    test_head_dimension_slicing();
    test_runtime_validation();
    test_remove_policy();
    test_pack_roundtrip(2);
    test_pack_roundtrip(3);
    test_pack_roundtrip(4);
    test_pack_roundtrip(5);
    test_pack_roundtrip(6);
    test_pack_roundtrip(8);
    test_hadamard_roundtrip();

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type != LLAMA_KVARN_TYPE_DISABLED) {
            test_tile_quantization(type);
        }
    }

    for (int bits : { 3, 5, 6, 8 }) {
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_CPU, true, bits);
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_GPU, false, bits);
    }
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_CPU, true, 6);
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_GPU, false, 6);
    test_store_legacy_parity_gpu();
    test_materialize_generic_parity_gpu();

    std::printf("test-kvarn: all tests OK\n");
    return 0;
}
