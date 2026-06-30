#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "dflash-profile.h"
#include "ggml-alloc.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    cparams.embeddings_layer_inp.resize(hparams.n_layer(), false);
    embd_layer_inp.resize(hparams.n_layer());

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max = %u\n",   __func__, cparams.n_outputs_max);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails.
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid < model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd_nextn.data[i0*n_embd + k], embd_nextn.data[i1*n_embd + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_DEEPSEEK4) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return llm_graph_params{
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.tree_mask   =*/ nullptr,
        /*.tree_parent_ids =*/ nullptr,
        /*.tree_ssm_intermediates =*/ nullptr,
        /*.tree_n_recurrent_layers =*/ 0,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.no_fused_gdn                =*/ false,
        /*.samplers                     =*/ nullptr,
        /*.n_samplers                   =*/ 0,
        /*.dflash_n_slots               =*/ 0,
        /*.dflash_cross_ctx              =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        model->hparams.n_layer_nextn == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}






namespace {

enum class dflash_commit_slot_state {
    all_missing,
    all_present,
    invalid,
};

static const char * dflash_backend_dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "UNKNOWN";
}


static llama_kv_cache * dflash_get_base_kv_cache(llama_memory_t memory) {
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        return kv;
    }
    if (auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return kv_iswa->get_base();
    }
    return nullptr;
}


#ifndef GGML_CUDA_MAX_DEVICES
#define GGML_CUDA_MAX_DEVICES 16
#endif


extern "C" bool llama_dflash_allow_multi_gpu_tape(void) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_MULTI_GPU_TAPE");
        // simplified — always allow if env is not explicitly disabled
        if (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F')) {
            return false;
        }
        env = std::getenv("GGML_DFLASH_ALLOW_MULTI_GPU_TAPE");
        if (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F')) {
            return false;
        }
        return true;
    }();
    return enabled;
}

// DFlash helper structs and static functions
struct dflash_cross_ring_handle {
    void * gpu_ring;
    void   (*fn_free)(void *);
    void   (*fn_write)(void *, int, int, const float *, int, int);
    bool   (*fn_write_d2d)(void *, int, int, const void *, int, int);
    void   (*fn_synchronize)(void *);
    bool   (*fn_snapshot)(void *, int, int, int, float *, int, int, int);
    const float * (*fn_interleave)(void *, int, int, int);
    void   (*fn_set_tensor)(void *, const void *, size_t, size_t);
    // Tensor-variant fn pointers (Vulkan): resolve vk_buffer from ggml_tensor*.
    // Null on CUDA (CUDA uses the raw-ptr variants above); the raw variants are null on Vulkan.
    void   (*fn_set_tensor_tensor)(ggml_tensor *, const void *, size_t, size_t);
    bool   (*fn_write_d2d_tensor)(void *, int, int, ggml_tensor *, int, int, int);
};



static bool dflash_diagnostic_debug_enabled();

static llama_memory_recurrent * get_recurrent_mem(llama_memory_t mem) {
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem))      return h->get_mem_recr();
    if (auto * h = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return h->get_mem_recr();
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

static bool dflash_env_is_zero(const char * value) {
    return value && (
        value[0] == '\0' ||
        std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "no") == 0);
}


static ggml_backend_reg_t dflash_gpu_backend_reg() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        reg = ggml_backend_reg_by_name("ROCm");
    }
    if (!reg) {
        reg = ggml_backend_reg_by_name("Vulkan");
    }
    return reg;
}


static bool dflash_is_cuda_compatible_tensor(const ggml_tensor * t) {
    if (!t || !t->data || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    const char * name = ggml_backend_buffer_name(t->buffer);
    return name && (std::strncmp(name, "CUDA", 4) == 0 || std::strncmp(name, "ROCm", 4) == 0);
}


static bool dflash_tensor_span_in_bounds(const ggml_tensor * t, size_t offset_bytes, size_t n_bytes) {
    return t && llama_dflash_view_span_in_bounds_for_test(
            (uint64_t) ggml_nbytes(t),
            (uint64_t) offset_bytes,
            (uint64_t) n_bytes);
}


static bool dflash_backend_dev_type_is_gpu(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}


static bool dflash_backend_dev_is_gpu(ggml_backend_dev_t dev) {
    return dev && dflash_backend_dev_type_is_gpu(ggml_backend_dev_type(dev));
}


static bool dflash_context_has_meta_backend(const std::vector<ggml_backend_ptr> & backends) {
    for (const auto & backend : backends) {
        if (!backend) {
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_META) {
            return true;
        }
    }
    return false;
}


static bool dflash_multi_gpu_debug_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_MULTI_GPU_DEBUG");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}


static bool dflash_force_meta_callback_guard_for_test() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_FORCE_META_CALLBACK_GUARD");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}


static int dflash_gpu_ring_device_override() {
    const char * env = std::getenv("GGML_DFLASH_GPU_RING_DEVICE");
    if (!env || env[0] == '\0') {
        return -1;
    }
    return std::atoi(env);
}


static void dflash_log_backend_layout(
        const char * func,
        const char * label,
        const std::vector<ggml_backend_ptr> & backends) {
    if (!dflash_multi_gpu_debug_enabled() && !dflash_diagnostic_debug_enabled()) {
        return;
    }

    LLAMA_LOG_INFO("%s: dflash backend layout for %s: n_backends=%zu\n",
        func, label, backends.size());

    for (size_t i = 0; i < backends.size(); ++i) {
        ggml_backend_t backend = backends[i].get();
        ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
        const auto type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;

        LLAMA_LOG_INFO(
            "%s: dflash backend[%zu]: backend=%s dev=%s type=%s\n",
            func,
            i,
            backend ? ggml_backend_name(backend) : "<null>",
            dev ? ggml_backend_dev_name(dev) : "<null>",
            dev ? dflash_backend_dev_type_name(type) : "<null>");
    }
}


static ggml_backend_t dflash_backend_for_dev(
        const std::vector<ggml_backend_ptr> & backends,
        ggml_backend_dev_t want_dev) {
    for (const auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev == want_dev && dflash_backend_dev_is_gpu(dev)) {
            return backend.get();
        }
    }
    return nullptr;
}


static void dflash_capture_add_wait_backend(
        dflash_capture_data & cap,
        ggml_backend_t backend,
        ggml_backend_reg_t reg) {
    if (!backend || !reg) {
        return;
    }

    auto fn = (dflash_capture_data::sync_backend_to_stream_fn_t)
        ggml_backend_reg_get_proc_address(reg, "dflash_cuda_backend_wait_for_stream");
    if (!fn) {
        return;
    }

    for (const auto & wait_backend : cap.capture_wait_backends) {
        if (wait_backend.backend == backend) {
            return;
        }
    }

    cap.capture_wait_backends.push_back({ backend, fn });
    if (!cap.fn_sync_backend_to_stream) {
        cap.fn_sync_backend_to_stream = fn;
        cap.sync_backend_to_stream_backend = backend;
    }
}




static void dflash_read_tensor_to(struct ggml_tensor * t, float * dst, size_t n_floats) {
    if (ggml_is_contiguous(t)) {
        const size_t n_bytes = n_floats * sizeof(float);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst, t->data, n_bytes);
        } else {
            ggml_backend_tensor_get(t, dst, 0, n_bytes);
        }
        return;
    }

    // non-contiguous view: read each innermost-contiguous slice separately
    // for 4D [ne0, ne1, ne2, ne3], ne0*ne1 is contiguous if nb[1]==ne[0]*elem_size
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const size_t esz = ggml_element_size(t);

    // find the largest contiguous inner chunk
    size_t contig_elems = ne0;
    if (t->nb[1] == ne0 * esz) {
        contig_elems = ne0 * ne1;
        if (t->nb[2] == ne0 * ne1 * esz) {
            contig_elems = ne0 * ne1 * ne2;
        }
    }

    size_t dst_off = 0;
    size_t n_chunks = n_floats / contig_elems;
    const size_t chunk_bytes = contig_elems * sizeof(float);

    for (size_t i = 0; i < n_chunks; ++i) {
        // compute source offset by iterating through outer dimensions
        size_t src_off = 0;
        size_t idx = i;
        if (contig_elems == (size_t)(ne0)) {
            int64_t i1 = idx % ne1; idx /= ne1;
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        } else if (contig_elems == (size_t)(ne0 * ne1)) {
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i2 * t->nb[2] + i3 * t->nb[3];
        } else {
            int64_t i3 = idx;
            src_off = i3 * t->nb[3];
        }

        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst + dst_off, (const char *)t->data + src_off, chunk_bytes);
        } else {
            ggml_backend_tensor_get(t, dst + dst_off, src_off, chunk_bytes);
        }
        dst_off += contig_elems;
    }
}


static void dflash_read_tensor(struct ggml_tensor * t, std::vector<float> & dst, size_t n_floats) {
    dst.resize(n_floats);
    dflash_read_tensor_to(t, dst.data(), n_floats);
}


static bool dflash_diagnostic_debug_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DEBUG");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}


static bool dflash_crash_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_CRASH_TRACE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}


static bool dflash_profile_sync_split_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_PROFILE_SYNC_SPLIT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}


static void dflash_log_decode_seq_state(
        const char * where,
        const llama_ubatch & ubatch,
        const dflash_capture_data * cap,
        const llama_cparams & cparams) {
    if (!dflash_crash_trace_enabled() || cap == nullptr) {
        return;
    }

    const int n_log = std::min((int) ubatch.n_seqs_unq, 4);
    for (int s = 0; s < n_log; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];

        int tok_count = 0;
        int32_t pos_min = INT32_MAX;
        int32_t pos_max = INT32_MIN;

        for (uint32_t t = 0; t < ubatch.n_tokens; ++t) {
            for (int32_t k = 0; k < ubatch.n_seq_id[t]; ++k) {
                if (ubatch.seq_id[t][k] != seq) {
                    continue;
                }

                ++tok_count;
                pos_min = std::min(pos_min, (int32_t) ubatch.pos[t]);
                pos_max = std::max(pos_max, (int32_t) ubatch.pos[t]);
            }
        }

        auto * plan = cap->prefill_plan_for_seq(seq);

        dflash_hidden_gpu * hidden = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->hidden_gpu.size()) {
            hidden = cap->hidden_gpu[(size_t) seq].get();
        }

        dflash_hidden_gpu * prefill = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->prefill_gpu.size()) {
            prefill = cap->prefill_gpu[(size_t) seq].get();
        }

        dflash_tape_gpu * tape = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->tapes.size()) {
            tape = cap->tapes[(size_t) seq].get();
        }

        const int pos_begin = tok_count > 0 ? pos_min : -1;
        const int pos_end   = tok_count > 0 ? pos_max + 1 : -1;

        LLAMA_LOG_INFO(
            "%s: dflash decode seq[%d]: seq=%d tok_count=%d pos=[%d,%d) plan=%d cap=[%d,%d) written=%d "
            "hidden=%p htok=%d/%d prefill=%p ptok=%d/%d tape=%p ttok=%d/%d prefill_seq=%p src_off=%d dst_off=%d copy=%d\n",
            where,
            s,
            (int) seq,
            tok_count,
            pos_begin,
            pos_end,
            plan && plan->active ? 1 : 0,
            plan ? (int) plan->capture_begin : -1,
            plan ? (int) plan->capture_end : -1,
            plan ? (int) plan->n_written : -1,
            (void *) hidden,
            hidden ? hidden->n_tokens : -1,
            hidden ? hidden->max_tokens : -1,
            (void *) prefill,
            prefill ? prefill->n_tokens : -1,
            prefill ? prefill->max_tokens : -1,
            (void *) tape,
            tape ? tape->n_tokens : -1,
            tape ? tape->max_tokens : -1,
            (void *) cparams.prefill_gpu_seqs[s],
            cparams.dflash_prefill_src_offsets[s],
            cparams.dflash_prefill_dst_offsets[s],
            cparams.dflash_prefill_n_tokens_seqs[s]);
    }
}


static void dflash_clear_prefill_cparams(llama_cparams & cparams) {
    cparams.prefill_gpu_n_seqs = 0;
    cparams.dflash_prefill_capture_active = false;
    cparams.dflash_prefill_src_offset = 0;
    cparams.dflash_prefill_dst_offset = 0;
    cparams.dflash_prefill_n_tokens   = 0;
    for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        cparams.prefill_gpu_seqs[s] = nullptr;
        cparams.dflash_prefill_src_offsets[s] = 0;
        cparams.dflash_prefill_dst_offsets[s] = 0;
        cparams.dflash_prefill_n_tokens_seqs[s] = 0;
    }
}


static void dflash_profile_reset(dflash_capture_data & cap) {
    cap.profile_decode_us = 0;
    cap.profile_output_extract_us = 0;
    cap.profile_raw_logits_us = 0;
    cap.profile_raw_logits_bytes = 0;
    cap.profile_raw_logits_skipped = 0;
    cap.profile_reduced_logits_us = 0;
    cap.profile_reduced_logits_ids_us = 0;
    cap.profile_reduced_logits_probs_us = 0;
    cap.profile_reduced_logits_bytes = 0;
    cap.profile_verify_sync_split_us = 0;
    cap.profile_cb_ask = 0;
    cap.profile_cb_hidden_ask = 0;
    cap.profile_cb_tape_ask = 0;
    cap.profile_cb_qkv_ask = 0;
    cap.profile_cb_read = 0;
    cap.profile_cb_hidden_read = 0;
    cap.profile_cb_tape_read = 0;
    cap.profile_cb_qkv_read = 0;
    cap.profile_replay_wait_us = 0;
    cap.profile_replay_gdn_enqueue_us = 0;
    cap.profile_replay_gdn_wait_us = 0;
    cap.profile_replay_conv_enqueue_us = 0;
    cap.profile_replay_conv_wait_us = 0;
    cap.profile_replay_layers = 0;
    cap.profile_replay_sync_calls = 0;
    cap.profile_replay_direct_gpu = 0;
    cap.profile_replay_ggml_gpu = 0;
    cap.profile_replay_cpu_fallback = 0;
    cap.profile_conv_gpu_us = 0;
    cap.profile_conv_read_wait_us = 0;
    cap.profile_conv_cpu_us = 0;
    cap.profile_conv_write_wait_us = 0;
    cap.profile_cb_names.clear();
}


static void dflash_profile_cb_name(dflash_capture_data & cap, const ggml_tensor * t, const char * phase) {
    if (!cap.profile) {
        return;
    }
    cap.profile_cb_names[std::string(phase) + ":" + t->name] += 1;
}


static void dflash_profile_log(const dflash_capture_data & cap, const char * func, int32_t n_vocab) {
    if (!cap.profile) {
        return;
    }

    const uint64_t skipped_bytes_est =
        cap.profile_raw_logits_skipped * (uint64_t) std::max(0, n_vocab) * sizeof(float);

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_VERIFY)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: decode=%.3f ms output_extract=%.3f ms "
            "raw_logits=%.3f ms raw_logits_bytes=%.3f MiB raw_logits_skipped=%" PRIu64
            " raw_logits_skipped_bytes_est=%.3f MiB "
            "reduced_logits=%.3f ms reduced_logits_ids=%.3f ms reduced_logits_probs=%.3f ms "
            "reduced_logits_bytes=%.3f KiB verify_sync_split=%.3f ms "
            "cb ask=%" PRIu64 " hidden=%" PRIu64 " tape=%" PRIu64 " qkv=%" PRIu64
            " read=%" PRIu64 " hidden=%" PRIu64 " tape=%" PRIu64 " qkv=%" PRIu64 "\n",
            func,
            cap.profile_decode_us / 1000.0,
            cap.profile_output_extract_us / 1000.0,
            cap.profile_raw_logits_us / 1000.0,
            cap.profile_raw_logits_bytes / (1024.0 * 1024.0),
            cap.profile_raw_logits_skipped,
            skipped_bytes_est / (1024.0 * 1024.0),
            cap.profile_reduced_logits_us / 1000.0,
            cap.profile_reduced_logits_ids_us / 1000.0,
            cap.profile_reduced_logits_probs_us / 1000.0,
            cap.profile_reduced_logits_bytes / 1024.0,
            cap.profile_verify_sync_split_us / 1000.0,
            cap.profile_cb_ask,
            cap.profile_cb_hidden_ask,
            cap.profile_cb_tape_ask,
            cap.profile_cb_qkv_ask,
            cap.profile_cb_read,
            cap.profile_cb_hidden_read,
            cap.profile_cb_tape_read,
            cap.profile_cb_qkv_read);
    }

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_REPLAY) &&
        (cap.profile_replay_wait_us || cap.profile_replay_gdn_enqueue_us || cap.profile_replay_gdn_wait_us ||
        cap.profile_replay_conv_enqueue_us || cap.profile_replay_conv_wait_us ||
        cap.profile_conv_gpu_us || cap.profile_conv_read_wait_us ||
        cap.profile_conv_cpu_us || cap.profile_conv_write_wait_us ||
        cap.profile_replay_layers || cap.profile_replay_sync_calls ||
        cap.profile_replay_direct_gpu || cap.profile_replay_ggml_gpu || cap.profile_replay_cpu_fallback)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: replay_path=direct-gpu:%" PRIu64 " replay_path=ggml-gpu:%" PRIu64
            " replay_path=cpu-fallback:%" PRIu64 " replay_layers=%" PRIu64 " replay_sync_calls=%" PRIu64
            " gdn_enqueue=%.3f ms gdn_wait=%.3f ms conv_enqueue=%.3f ms conv_wait=%.3f ms "
            "legacy_replay_wait=%.3f ms legacy_conv_gpu_enqueue=%.3f ms "
            "legacy_conv_read_wait=%.3f ms legacy_conv_write_wait=%.3f ms conv_cpu=%.3f ms\n",
            func,
            cap.profile_replay_direct_gpu,
            cap.profile_replay_ggml_gpu,
            cap.profile_replay_cpu_fallback,
            cap.profile_replay_layers,
            cap.profile_replay_sync_calls,
            cap.profile_replay_gdn_enqueue_us / 1000.0,
            cap.profile_replay_gdn_wait_us / 1000.0,
            cap.profile_replay_conv_enqueue_us / 1000.0,
            cap.profile_replay_conv_wait_us / 1000.0,
            cap.profile_replay_wait_us / 1000.0,
            cap.profile_conv_gpu_us / 1000.0,
            cap.profile_conv_read_wait_us / 1000.0,
            cap.profile_conv_write_wait_us / 1000.0,
            cap.profile_conv_cpu_us / 1000.0);
    }

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_TRACE) && !cap.profile_cb_names.empty()) {
        LLAMA_LOG_INFO("%s: dflash profile: callback tensors:\n", func);
        for (const auto & hit : cap.profile_cb_names) {
            LLAMA_LOG_INFO("%s: dflash profile:   %" PRIu64 " %s\n", func, hit.second, hit.first.c_str());
        }
    }
}


static bool dflash_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (dflash_capture_data *) user_data;
    const llama_ubatch * ub = cap->ubatch;
    const uint32_t n_seqs_unq = ub ? ub->n_seqs_unq : 0;

    auto h_it = cap->hidden_name_idx.find(t->name);

    if (ask) {
        if (cap->profile) {
            cap->profile_cb_ask++;
        }
        if (h_it != cap->hidden_name_idx.end()) {
            if (cap->profile) {
                cap->profile_cb_hidden_ask++;
                dflash_profile_cb_name(*cap, t, "ask");
            }
            return true;
        }
        if (cap->tape_enabled) {
            if (dflash_diagnostic_debug_enabled()) {
                const char * nm = t->name ? t->name : "";
                if (strstr(nm, "qkv") || strstr(nm, "mixed")) {
                    fprintf(stderr, "[dflash-tape-cb] check name='%s' in_map=%d\n", nm, (int)cap->tape_name_map.count(t->name));
                }
            }
        }
        if (cap->tape_enabled && cap->tape_name_map.count(t->name)) {
            if (cap->profile) {
                cap->profile_cb_tape_ask++;
                dflash_profile_cb_name(*cap, t, "ask");
            }
            auto * active_tape = cap->active_tape();
            const bool gpu_tape_fits = active_tape && (!ub || (int) ub->n_seq_tokens <= active_tape->max_tokens);
            if (gpu_tape_fits) {
                // GPU tape captures K/V/gate/beta/QKV through graph-embedded per-seq copies.
                // No tape tensor needs the eval callback when the active GPU tape is present.
                // Multi-seq is supported; tensor stored with per-seq metadata.
                auto it = cap->tape_name_map.find(t->name);
                if (it != cap->tape_name_map.end() && it->second.second == DFLASH_TAPE_QKV && cap->profile) {
                    cap->profile_cb_qkv_ask++;
                }
                return false;
            }
            // CPU tape fallback: no multi-seq support
            if (n_seqs_unq > 1) {
                return false;
            }
            return true;
        }
        return false;
    }

    // ask=false: tensor data is ready, read it back. dflash_reset_hidden_capture()
    // (called at the top of decode()) zeroes buf.n_tokens for every slot before
    // the ubatch loop, so each slot's buffer accumulates only that slot's tokens
    // (in their ubatch order) across all ubatches in this llama_decode() call.
        if (h_it != cap->hidden_name_idx.end()) {
            if (cap->profile) {
                cap->profile_cb_read++;
                cap->profile_cb_hidden_read++;
                dflash_profile_cb_name(*cap, t, "read");
            }
            const int64_t new_embd = t->ne[0];
            const int64_t new_n = t->ne[1];
            const size_t h_idx = h_it->second;

            if (n_seqs_unq <= 1) {
            // single-seq fast path: route the whole tensor to one slot
            const int slot = ub ? ub->seq_id_unq[0] : -1;
            auto * sh = cap->slot_hiddens(slot);
            if (!sh) {
                return true; // no DFlash slot for this seq; skip capture
            }
            GGML_ASSERT(h_idx < sh->size());
            auto & buf = (*sh)[h_idx];
            buf.n_embd = new_embd;
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            const size_t add_elems = (size_t) new_n * (size_t) new_embd;
            buf.data.resize(old_elems + add_elems);
            dflash_read_tensor_to(t, buf.data.data() + old_elems, add_elems);
            // Capture token IDs for training data extraction
            if (ub) {
                const size_t old_n = buf.n_tokens;
                buf.token_ids.resize(old_n + new_n);
                for (int64_t ti = 0; ti < new_n; ++ti) {
                    buf.token_ids[old_n + ti] = ub->token[old_n + ti];
                }
            }
            buf.n_tokens += new_n;
            return true;
        }

        // multi-seq scatter: read full tensor once, count tokens per slot to
        // pre-reserve destination buffers, then append each token's hidden
        // vector to its owning slot's buffer in one pass.
        GGML_ASSERT(ub && (int64_t) ub->n_tokens == new_n);
        cap->scatter_buf.resize((size_t) new_embd * (size_t) new_n);
        dflash_read_tensor_to(t, cap->scatter_buf.data(), cap->scatter_buf.size());

        const int n_slots = cap->hiddens ? (int) cap->hiddens->size() : 0;
        for (uint32_t s = 0; s < n_seqs_unq; ++s) {
            const llama_seq_id seq = ub->seq_id_unq[s];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            buf.n_embd = new_embd;
            // Worst-case: all remaining tokens belong to this seq. Reserving
            // up to that bound costs at most one realloc per slot per ubatch
            // (vs one per token without reserve).
            buf.data.reserve((size_t) (buf.n_tokens + new_n) * (size_t) new_embd);
        }

        for (int64_t i = 0; i < new_n; ++i) {
            const llama_seq_id seq = ub->seq_id[i][0];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            buf.data.resize(old_elems + (size_t) new_embd);
            std::memcpy(buf.data.data() + old_elems,
                        cap->scatter_buf.data() + (size_t) i * (size_t) new_embd,
                        (size_t) new_embd * sizeof(float));
            // Capture token ID for training data extraction
            if (ub) {
                buf.token_ids.push_back(ub->token[i]);
            }
            buf.n_tokens += 1;
        }
        return true;
    }

    // tape recording
    if (cap->tape_enabled) {
        auto it = cap->tape_name_map.find(t->name);
        if (it != cap->tape_name_map.end()) {
            int layer_idx = it->second.first;
            int type      = it->second.second;
            auto & tape   = cap->tape_layers[layer_idx];

            if (cap->profile) {
                cap->profile_cb_read++;
                cap->profile_cb_tape_read++;
                if (type == DFLASH_TAPE_QKV) {
                    cap->profile_cb_qkv_read++;
                }
                dflash_profile_cb_name(*cap, t, "read");
            }

            // when GPU tape is active, all tape tensors are captured by graph-embedded copies
            auto * active_tape = cap->active_tape();
            const bool gpu_tape_fits = active_tape && (!ub || (int) ub->n_seq_tokens <= active_tape->max_tokens);
            if (gpu_tape_fits) {
                return true; // skip; already on GPU
            }

            size_t n_elem = ggml_nelements(t);

            switch (type) {
                case DFLASH_TAPE_K:
                    tape.S_k = t->ne[0];
                    tape.H_k = t->ne[1];
                    tape.n_tokens = (int) t->ne[2];
                    dflash_read_tensor(t, tape.k, n_elem);
                    break;
                case DFLASH_TAPE_V:
                    tape.S_v = t->ne[0];
                    tape.H_v = t->ne[1];
                    dflash_read_tensor(t, tape.v, n_elem);
                    break;
                case DFLASH_TAPE_GATE:
                    dflash_read_tensor(t, tape.gate, n_elem);
                    break;
                case DFLASH_TAPE_BETA:
                    dflash_read_tensor(t, tape.beta, n_elem);
                    break;
                case DFLASH_TAPE_QKV:
                    tape.conv_channels = t->ne[0];
                    tape.n_tokens = (int) t->ne[1]; // tokens per seq (ne[1] of 3D [ch, n_seq_tokens, n_seqs])
                    if (ub && n_seqs_unq > 1) {
                        tape.n_seqs = std::min((int) n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);
                        for (int s = 0; s < tape.n_seqs; ++s) {
                            tape.seq_ids[s] = ub->seq_id_unq[s];
                        }
                    } else {
                        tape.n_seqs = 1;
                        tape.seq_ids[0] = ub ? ub->seq_id_unq[0] : 0;
                    }
                    dflash_read_tensor(t, tape.qkv_mixed, n_elem);
                    break;
            }
            return true;
        }
    }

    return true;
}


static int64_t cross_bucket(int64_t n) {
    if (n <= 16) return 16;
    if (n <= 128) {
        int64_t b = 1;
        while (b < n) b <<= 1;
        return b;
    }
    const int64_t step = 128;
    return ((n + step - 1) / step) * step;
}


static int64_t dflash_max_cross_ctx() {
    static const int64_t max_ctx = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return max_ctx;
}

} // namespace



void llama_context::set_cross_data(const float * data, int64_t n_embd, int64_t n_tokens) {
    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && n_tokens > max_ctx) ? max_ctx : n_tokens;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd    = n_embd;
    cross.n_enc     = bucket;
    cross.n_enc_real = n_tokens;  // actual full data length (for windowing in set_input)
    cross.v_embd_gpu = nullptr;
    cross.v_embd_gpu_n_enc_real = 0;
    cross.fn_set_tensor_d2d = nullptr;
    cross.dflash_kv_cache = nullptr;
    cross.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(cross.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}

namespace {

static llama_ubatch dflash_make_commit_ubatch(
        uint32_t n_pos,
        llama_seq_id seq_id,
        llama_pos start_pos,
        uint32_t n_tokens) {
    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens, 0);
    data->pos.resize((size_t) n_tokens * n_pos, 0);
    data->n_seq_id.resize(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->seq_idx.resize(LLAMA_MAX_SEQ, -1);
    data->output.resize(n_tokens, 0);
    data->seq_id_data.resize(n_tokens, seq_id);
    data->seq_id_unq.push_back(seq_id);

    if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        data->seq_idx[(size_t) seq_id] = 0;
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->seq_id[i] = &data->seq_id_data[i];
        data->pos[i] = start_pos + (llama_pos) i;
    }
    if (!data->output.empty()) {
        data->output.back() = 1;
    }

    return {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens,
        /*.n_seqs       =*/ 1,
        /*.n_seqs_unq   =*/ 1,
        /*.n_pos        =*/ n_pos,
        /*.token        =*/ data->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ data->pos.data(),
        /*.n_seq_id     =*/ data->n_seq_id.data(),
        /*.seq_id       =*/ data->seq_id.data(),
        /*.seq_id_unq   =*/ data->seq_id_unq.data(),
        /*.seq_idx      =*/ data->seq_idx.data(),
        /*.output       =*/ data->output.data(),
        /*.data         =*/ std::move(data),
    };
}



static dflash_commit_slot_state dflash_collect_commit_slot_info(
        llama_kv_cache * kv,
        llama_seq_id seq_id,
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & sinfo) {
    if (!kv || seq_id < 0) {
        return dflash_commit_slot_state::invalid;
    }

    const uint32_t n_stream = kv->get_n_stream();
    const uint32_t stream = n_stream > 1 ? (uint32_t) seq_id : 0;
    if (stream >= n_stream) {
        return dflash_commit_slot_state::invalid;
    }

    sinfo = {};
    sinfo.s0 = stream;
    sinfo.s1 = stream;
    sinfo.resize(1);
    sinfo.strm[0] = stream;
    sinfo.idxs[0].reserve(ubatch.n_tokens);

    uint32_t n_missing = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        uint32_t cell_indices[2] = {};
        const llama_pos pos = ubatch.pos[i];
        const int n_cells = kv->cells_at_pos(seq_id, pos, cell_indices, 2);
        if (n_cells == 0) {
            ++n_missing;
            continue;
        }
        if (n_cells != 1) {
            return dflash_commit_slot_state::invalid;
        }
        sinfo.idxs[0].push_back(cell_indices[0]);
    }

    if (n_missing == ubatch.n_tokens) {
        return dflash_commit_slot_state::all_missing;
    }
    if (n_missing == 0 && sinfo.idxs[0].size() == ubatch.n_tokens) {
        return dflash_commit_slot_state::all_present;
    }
    return dflash_commit_slot_state::invalid;
}


static llama_memory_context_ptr dflash_make_base_commit_context(
        llama_kv_cache * base_kv,
        llama_seq_id seq_id,
        const llama_ubatch & ubatch) {
    if (!base_kv) {
        return nullptr;
    }

    llama_kv_cache::slot_info exact_sinfo;
    const dflash_commit_slot_state slot_state =
        dflash_collect_commit_slot_info(base_kv, seq_id, ubatch, exact_sinfo);
    if (slot_state == dflash_commit_slot_state::invalid) {
        return nullptr;
    }

    std::vector<llama_ubatch> ubatches = { ubatch };
    llama_kv_cache::slot_info_vec_t sinfos;

    if (slot_state == dflash_commit_slot_state::all_missing) {
        sinfos = base_kv->prepare(ubatches);
        if (sinfos.empty()) {
            return nullptr;
        }
    } else {
        sinfos.push_back(std::move(exact_sinfo));
    }

    return std::make_unique<llama_kv_cache_context>(base_kv, std::move(sinfos), std::move(ubatches));
}

} // namespace



bool llama_context::dflash_target_kv_cache_update_gpu(
        llama_seq_id seq_id,
        llama_pos start_pos,
        const void * d_hidden,
        int n_tokens,
        int n_layers,
        int n_embd_layer,
        set_tensor_d2d_fn_t fn_d2d,
        set_tensor_d2d_tensor_fn_t fn_d2d_tensor) {
    if (!d_hidden || n_tokens <= 0 || n_layers <= 0 || n_embd_layer <= 0 || (!fn_d2d && !fn_d2d_tensor)) {
        return false;
    }
    if (!llm_arch_is_dflash_drafter(model.arch) || !memory) {
        return false;
    }
    if (model.n_devices() > 1) {
        return false;
    }
    dflash_kv_cache_set_active_seq(seq_id);

    const int64_t n_target_features = (int64_t) n_layers * n_embd_layer;
    if (n_target_features != model.hparams.dflash_n_target_features) {
        return false;
    }

    llama_kv_cache * base_kv = dflash_get_base_kv_cache(memory.get());
    if (!base_kv) {
        return false;
    }

    // We are refreshing the accepted tail of the drafter prefix with target-hidden
    // K/V. Overwriting occupied cells in-place makes apply_ubatch() purge the whole
    // earlier prefix to preserve its generic overwrite invariants. Drop only the
    // tail we are about to replace, then allocate that suffix again so older
    // accepted-prefix cells stay live.
    base_kv->seq_rm(seq_id, start_pos, -1);

    const auto ubatch = dflash_make_commit_ubatch(model.hparams.n_pos_per_embd(), seq_id, start_pos, (uint32_t) n_tokens);
    auto sinfo = base_kv->find_slot(ubatch, false);
    if (sinfo.empty()) {
        return false;
    }
    base_kv->apply_ubatch(sinfo, ubatch);

    std::vector<llama_ubatch> ubatches = { ubatch };
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.push_back(std::move(sinfo));
    auto mctx = std::make_unique<llama_kv_cache_context>(base_kv, std::move(sinfos), std::move(ubatches));

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }

    const size_t max_nodes = graph_max_nodes(n_tokens);
    auto res = std::make_unique<llm_graph_result>(max_nodes);

    const auto gparams = graph_params(res.get(), ubatch, mctx.get(), LLM_GRAPH_TYPE_DFLASH_KV_UPDATE);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (!gf) {
        return false;
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(res->get_ctx(), gpu_buft);

    dflash_kv_cache_data * dflash_kv_cache = dflash_kv_cache_active();
    ggml_backend_buffer_t alloc_buf = nullptr;
    const bool reuse_update_buf = dflash_kv_cache != nullptr;
    if (reuse_update_buf) {
        if (needed > dflash_kv_cache->update_buf_size) {
            if (dflash_kv_cache->update_buf) {
                ggml_backend_buffer_free(dflash_kv_cache->update_buf);
            }
            dflash_kv_cache->update_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            dflash_kv_cache->update_buf_size = dflash_kv_cache->update_buf
                ? ggml_backend_buffer_get_size(dflash_kv_cache->update_buf) : 0;
        }
        alloc_buf = dflash_kv_cache->update_buf;
    } else {
        alloc_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
    }
    if (!alloc_buf) {
        return false;
    }

    {
        struct ggml_tallocr talloc = ggml_tallocr_new(alloc_buf);
        struct ggml_tensor * t = ggml_get_first_tensor(res->get_ctx());
        while (t) {
            if (t->data == nullptr && t->view_src == nullptr) {
                ggml_tallocr_alloc(&talloc, t);
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(res->get_ctx(), t);
        }
    }

    cross.dflash_kv_update_gpu = d_hidden;
    cross.dflash_kv_update_n_embd = n_target_features;
    cross.dflash_kv_update_n_enc_real = n_tokens;
    cross.dflash_kv_update_fn_set_tensor_d2d = fn_d2d;
    cross.dflash_kv_update_fn_set_tensor_d2d_tensor = fn_d2d_tensor;

    const ggml_status status = [&] {
        res->set_inputs(&ubatch);
        return ggml_backend_graph_compute_async(gpu_backend, gf);
    }();

    cross.dflash_kv_update_gpu = nullptr;
    cross.dflash_kv_update_n_embd = 0;
    cross.dflash_kv_update_n_enc_real = 0;
    cross.dflash_kv_update_fn_set_tensor_d2d = nullptr;
    cross.dflash_kv_update_fn_set_tensor_d2d_tensor = nullptr;

    // Reused commit scratch lives on cudaStreamPerThread, while the graph runs on
    // the backend stream. Order those streams before this function returns so the
    // next accepted-prefix commit cannot overwrite update_buf while the prior
    // async commit is still reading from it.
    bool synchronized = false;
    if (status == GGML_STATUS_SUCCESS) {
        const bool ordered =
            reuse_update_buf &&
            dflash_kv_cache &&
            dflash_kv_cache->fn_wait_backend_stream &&
            dflash_kv_cache->fn_wait_backend_stream(gpu_backend);
        if (!ordered) {
            ggml_backend_synchronize(gpu_backend);
            synchronized = true;
        }
    }

    if (!reuse_update_buf) {
        if (!synchronized) {
            ggml_backend_synchronize(gpu_backend);
        }
        ggml_backend_buffer_free(alloc_buf);
    }

    return status == GGML_STATUS_SUCCESS;
}










static bool dflash_gpu_hidden_span_in_bounds(
        const ggml_tensor * tensor,
        int                 src_offset,
        int                 n_tokens,
        int                 n_embd,
        const char *        where) {
    if (!tensor || !tensor->data || src_offset < 0 || n_tokens <= 0 || n_embd <= 0) {
        LLAMA_LOG_WARN("%s: invalid DFlash hidden tensor span request tensor=%p src_offset=%d n_tokens=%d n_embd=%d\n",
            where, (const void *) tensor, src_offset, n_tokens, n_embd);
        return false;
    }
    if (tensor->type != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("%s: invalid DFlash hidden tensor type %s, expected f32\n",
            where, ggml_type_name(tensor->type));
        return false;
    }

    const size_t n_embd_size = (size_t) n_embd;
    if (n_embd_size > std::numeric_limits<size_t>::max() / sizeof(float)) {
        LLAMA_LOG_WARN("%s: DFlash hidden tensor row-size overflow n_embd=%d\n",
            where, n_embd);
        return false;
    }

    const size_t row_bytes = n_embd_size * sizeof(float);
    if ((size_t) src_offset > std::numeric_limits<size_t>::max() / row_bytes ||
            (size_t) n_tokens > std::numeric_limits<size_t>::max() / row_bytes) {
        LLAMA_LOG_WARN("%s: DFlash hidden tensor span overflow src_offset=%d n_tokens=%d n_embd=%d\n",
            where, src_offset, n_tokens, n_embd);
        return false;
    }

    const size_t src_offset_bytes = (size_t) src_offset * row_bytes;
    const size_t n_bytes = (size_t) n_tokens * row_bytes;
    if (src_offset_bytes > std::numeric_limits<size_t>::max() - n_bytes ||
            !(src_offset_bytes + n_bytes <= ggml_nbytes(tensor))) {
        LLAMA_LOG_WARN("%s: DFlash hidden tensor span out of bounds offset=%zu size=%zu tensor_bytes=%zu ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
            where, src_offset_bytes, n_bytes, ggml_nbytes(tensor),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        return false;
    }

    return true;
}



// DFlash-specific functions

bool llama_context::allocate_prefill_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture || dflash_capture->layer_ids.empty()) {
        return false;
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (!dflash_capture->gpu_capture_enabled) {
        return false;
    }
    if (model.n_devices() > 1 && !llama_dflash_allow_multi_gpu_tape()) {
        return false;
    }
    if (!llama_dflash_gpu_hidden_supported_arch(model.arch)) {
        return false;
    }

    // Already allocated with sufficient capacity — reuse.
    if (!dflash_capture->prefill_gpu.empty() &&
        (int) dflash_capture->prefill_gpu.size() >= n_slots &&
        dflash_capture->prefill_gpu_max_tokens >= max_tokens) {
        // Update n_tokens on existing buffers for the new batch size.
        for (int s = 0; s < n_slots && s < (int) dflash_capture->prefill_gpu.size(); ++s) {
            dflash_capture->prefill_gpu[s]->n_tokens = 0;
        }
        return true;
    }

    const int n_layers = (int) dflash_capture->layer_ids.size();
    const int64_t n_embd = model.hparams.n_embd;

    dflash_capture->prefill_gpu.clear();
    dflash_capture->prefill_gpu.reserve(n_slots);

    size_t total_size = 0;
    for (int slot = 0; slot < n_slots; ++slot) {
        auto hidden = std::make_unique<dflash_hidden_gpu>();
        hidden->layers.resize(n_layers);
        hidden->layer_ids = dflash_capture->layer_ids;
        hidden->n_embd = n_embd;
        hidden->max_tokens = max_tokens;
        hidden->n_tokens = 0;

        for (int i = 0; i < n_layers; ++i) {
            const int il = dflash_capture->layer_ids[i];
            ggml_backend_dev_t layer_dev = model.dev_layer(il);
            ggml_backend_t layer_backend = dflash_backend_for_dev(backends, layer_dev);
            if (!layer_backend) {
                LLAMA_LOG_WARN("%s: no GPU backend for prefill layer %d device %s; using callback fallback\n",
                    __func__, il, layer_dev ? ggml_backend_dev_name(layer_dev) : "<null>");
                dflash_capture->prefill_gpu.clear();
                dflash_capture->prefill_gpu_max_tokens = 0;
                return false;
            }

            const size_t ctx_mem = ggml_tensor_overhead() * 2;
            struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
            struct ggml_context * hidden_ctx = ggml_init(ctx_params);
            if (!hidden_ctx) {
                LLAMA_LOG_WARN("%s: failed to create prefill GPU context for slot %d layer %d; using callback fallback\n",
                    __func__, slot, il);
                dflash_capture->prefill_gpu.clear();
                dflash_capture->prefill_gpu_max_tokens = 0;
                return false;
            }
            hidden->ctxs.push_back(hidden_ctx);

            hidden->layers[i] = ggml_new_tensor_2d(hidden_ctx, GGML_TYPE_F32, n_embd, (int64_t) max_tokens);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(hidden_ctx, layer_backend);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate prefill GPU buffer for slot %d layer %d; using callback fallback\n",
                    __func__, slot, il);
                dflash_capture->prefill_gpu.clear();
                dflash_capture->prefill_gpu_max_tokens = 0;
                return false;
            }
            hidden->bufs.push_back(buf);
            dflash_capture_add_wait_backend(*dflash_capture, layer_backend, ggml_backend_dev_backend_reg(layer_dev));
            total_size += ggml_backend_buffer_get_size(buf);
        }

        dflash_capture->prefill_gpu.push_back(std::move(hidden));
    }

    dflash_capture->prefill_gpu_max_tokens = max_tokens;

    LLAMA_LOG_INFO("%s: allocated prefill GPU staging buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_layers, max_tokens);
    return true;
}


bool llama_context::cross_ring_gpu_write_hidden(void * handle, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!handle || !dflash_capture) {
        return false;
    }
    auto * hgpu = dflash_capture->active_hidden_gpu();
    if (!hgpu || layer < 0 || layer >= (int) hgpu->layers.size()) {
        return false;
    }
    if (src_offset < 0 || n_tokens <= 0 || n_embd != hgpu->n_embd ||
            src_offset + n_tokens > hgpu->n_tokens) {
        return false;
    }

    auto * tensor = hgpu->layers[layer];
    if (!tensor || !tensor->data) {
        return false;
    }
    if (!dflash_gpu_hidden_span_in_bounds(tensor, src_offset, n_tokens, n_embd, __func__)) {
        return false;
    }

    auto * h = (dflash_cross_ring_handle *)handle;
    // Vulkan: tensor-variant D2D resolves the target hidden vk_buffer from ggml_tensor*
    // and takes src_offset explicitly (the raw variant bakes the offset into the pointer,
    // which is a Vulkan sentinel and cannot be used). CUDA leaves fn_write_d2d_tensor null.
    if (h->fn_write_d2d_tensor &&
        h->fn_write_d2d_tensor(h->gpu_ring, layer, ring_pos, tensor, src_offset, n_tokens, n_embd)) {
        return true;
    }
    // CUDA: raw-ptr D2D (offset baked into the pointer; tensor->data is a real device address).
    const size_t src_offset_bytes = (size_t) src_offset * (size_t) n_embd * sizeof(float);
    const void * src = (const char *) tensor->data + src_offset_bytes;
    if (h->fn_write_d2d && h->fn_write_d2d(h->gpu_ring, layer, ring_pos, src, n_tokens, n_embd)) {
        return true;
    }

    static bool warned_d2h_fallback = false;
    if (!warned_d2h_fallback) {
        LLAMA_LOG_WARN("%s: GPU hidden D2D ring write unavailable; falling back to GPU readback + H2D ring upload\n",
            __func__);
        warned_d2h_fallback = true;
    }

    const size_t n_bytes = (size_t) n_tokens * (size_t) n_embd * sizeof(float);
    std::vector<float> staging((size_t) n_tokens * (size_t) n_embd);
    ggml_backend_tensor_get(tensor, staging.data(), src_offset_bytes, n_bytes);
    h->fn_write(h->gpu_ring, layer, ring_pos, staging.data(), n_tokens, n_embd);
    h->fn_synchronize(h->gpu_ring);
    return true;
}


bool llama_context::dflash_hidden_capture_available() const {
    return dflash_capture_valid_last_decode;
}


bool llama_context::dflash_kv_cache_init(int ctx_size) {
    if (ctx_size <= 0 || !llm_arch_is_dflash_drafter(model.arch)) {
        return false;
    }
    if (model.n_devices() > 1) {
        dflash_kv_caches.clear();
        cross.dflash_kv_cache = nullptr;
        if (!dflash_kv_cache_multi_gpu_fallback_logged) {
            LLAMA_LOG_INFO("%s: multi-device DFlash drafter detected (%zu devices); disabling DFlash drafter K/V projection cache. "
                "Use a single --spec-draft-device to keep the cache enabled.\n",
                __func__, model.n_devices());
            dflash_kv_cache_multi_gpu_fallback_logged = true;
        }
        return false;
    }
    dflash_kv_cache_data * active_cache = dflash_kv_cache_active();
    if (active_cache && active_cache->ring_size == ctx_size) {
        return true;
    }

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!gpu_backend || !cuda_reg) {
        return false;
    }

    auto fn_write_d2d = (dflash_kv_cache_data::write_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_write_d2d");
    auto fn_write_d2d_no_check = (dflash_kv_cache_data::write_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_write_d2d_no_check");
    auto fn_append_d2d = (dflash_kv_cache_data::append_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_append_d2d");
    auto fn_append_d2d_no_check = (dflash_kv_cache_data::append_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_append_d2d_no_check");
    auto fn_copy_d2d = (dflash_kv_cache_data::copy_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_copy_d2d");
    auto fn_copy_d2d_no_check = (dflash_kv_cache_data::copy_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_copy_d2d_no_check");
    auto fn_prepare_ptr = (dflash_kv_cache_data::prepare_ptr_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_sync_ptr = (dflash_kv_cache_data::sync_ptr_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    auto fn_wait_backend_stream = (dflash_kv_cache_data::sync_backend_stream_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_stream");
    auto fn_wait_dflash_stream = (dflash_kv_cache_data::sync_backend_stream_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_dflash_stream");
    auto fn_interleave = (dflash_kv_cache_data::interleave_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_interleave");
    if (!fn_write_d2d || !fn_append_d2d || !fn_interleave) {
        return false;
    }

    const int n_layers = (int) model.hparams.n_layer();
    const int64_t n_embd_head = model.hparams.n_embd_head_v();
    const int64_t n_head_kv = model.hparams.n_head_kv();
    const int n_elem = (int) (n_embd_head * n_head_kv);
    if (n_layers <= 0 || n_elem <= 0) {
        return false;
    }

    const size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_layers * 2 + 8);
    struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
    ggml_context * cache_ctx = ggml_init(ctx_params);
    if (!cache_ctx) {
        return false;
    }

    auto cache = std::make_unique<dflash_kv_cache_data>();
    cache->ctx = cache_ctx;
    cache->ring_size = ctx_size;
    cache->n_elem = n_elem;
    cache->fn_write_d2d = fn_write_d2d;
    cache->fn_write_d2d_no_check = fn_write_d2d_no_check;
    cache->fn_append_d2d = fn_append_d2d;
    cache->fn_append_d2d_no_check = fn_append_d2d_no_check;
    cache->fn_copy_d2d = fn_copy_d2d;
    cache->fn_copy_d2d_no_check = fn_copy_d2d_no_check;
    cache->fn_prepare_ptr = fn_prepare_ptr;
    cache->fn_sync_ptr = fn_sync_ptr;
    cache->fn_wait_backend_stream = fn_wait_backend_stream;
    cache->fn_wait_dflash_stream = fn_wait_dflash_stream;
    cache->fn_interleave = fn_interleave;
    cache->view.n_layers = n_layers;
    cache->view.n_embd_head = n_embd_head;
    cache->view.n_head_kv = n_head_kv;
    cache->view.ctx_len = ctx_size;
    cache->view.ring_size = ctx_size;
    cache->k_ring.resize(n_layers);
    cache->v_ring.resize(n_layers);
    cache->view.k_ring.resize(n_layers);
    cache->view.v_ring.resize(n_layers);

    for (int il = 0; il < n_layers; ++il) {
        cache->k_ring[il] = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, ctx_size);
        cache->v_ring[il] = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, ctx_size);
        cache->view.k_ring[il] = cache->k_ring[il];
        cache->view.v_ring[il] = cache->v_ring[il];
    }

    cache->buf = ggml_backend_alloc_ctx_tensors(cache_ctx, gpu_backend);
    if (!cache->buf) {
        return false;
    }

    for (int il = 0; il < n_layers; ++il) {
        ggml_backend_tensor_memset(cache->k_ring[il], 0, 0, ggml_nbytes(cache->k_ring[il]));
        ggml_backend_tensor_memset(cache->v_ring[il], 0, 0, ggml_nbytes(cache->v_ring[il]));
    }

    const size_t total_size = ggml_backend_buffer_get_size(cache->buf);
    LLAMA_LOG_INFO("%s: allocated DFlash drafter K/V cache: %.1f MB (seq=%d, %d layers, %d tokens, %d elems/token)\n",
        __func__, total_size / (1024.0 * 1024.0), (int) dflash_kv_cache_active_seq, n_layers, ctx_size, n_elem);

    dflash_kv_cache_active_ref() = std::move(cache);
    cross.dflash_kv_cache = nullptr;
    return true;
}


bool llama_context::dflash_kv_cache_prepare_batch(const llama_seq_id * seq_ids, int n_seq, int ctx_window) {
    cross.dflash_kv_cache = nullptr;

    if (!seq_ids || n_seq < 2 || n_seq > (int) LLAMA_DFLASH_MAX_SLOTS ||
            ctx_window <= 0 || !llm_arch_is_dflash_drafter(model.arch)) {
        return false;
    }
    if (model.n_devices() > 1) {
        return false;
    }

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }

    struct batch_slot_cache {
        dflash_kv_cache_data * cache = nullptr;
        int n_copy = 0;
        int src_offset = 0;
    };

    std::vector<batch_slot_cache> slot_caches;
    slot_caches.reserve(n_seq);

    int n_layers = 0;
    int64_t n_embd_head = 0;
    int64_t n_head_kv = 0;
    int n_elem = 0;

    for (int s = 0; s < n_seq; ++s) {
        const llama_seq_id seq_id = seq_ids[s];
        auto cache_it = dflash_kv_caches.find(seq_id);
        if (cache_it == dflash_kv_caches.end() || !cache_it->second) {
            return false;
        }
        auto * cache = cache_it->second.get();

        auto cross_it = cross.v_embd_per_seq.find(seq_id);
        if (cross_it == cross.v_embd_per_seq.end()) {
            return false;
        }
        const int64_t cross_real = cross_it->second.v_embd_gpu
            ? cross_it->second.v_embd_gpu_n_enc_real
            : cross_it->second.n_enc_real;
        const int n_needed = (int) std::min<int64_t>(std::max<int64_t>(cross_real, 0), ctx_window);

        if (n_needed <= 0 ||
                cache->ring_size < ctx_window ||
                cache->n_filled < n_needed ||
                cache->write_pos != 0 ||
                cache->n_elem <= 0 ||
                !cache->fn_copy_d2d ||
                cache->view.n_layers <= 0 ||
                (int) cache->k_ring.size() < cache->view.n_layers ||
                (int) cache->v_ring.size() < cache->view.n_layers) {
            return false;
        }

        if (s == 0) {
            n_layers = cache->view.n_layers;
            n_embd_head = cache->view.n_embd_head;
            n_head_kv = cache->view.n_head_kv;
            n_elem = cache->n_elem;
        } else if (n_layers != cache->view.n_layers ||
                n_embd_head != cache->view.n_embd_head ||
                n_head_kv != cache->view.n_head_kv ||
                n_elem != cache->n_elem) {
            return false;
        }

        for (int il = 0; il < n_layers; ++il) {
            if (!cache->k_ring[il] || !cache->v_ring[il]) {
                return false;
            }
        }

        slot_caches.push_back({ cache, n_needed, cache->n_filled - n_needed });
    }

    if (n_layers <= 0 || n_elem <= 0 || n_embd_head <= 0 || n_head_kv <= 0) {
        return false;
    }

    const int total_ctx = n_seq * ctx_window;
    const bool reuse_batch =
        dflash_kv_cache_batch.ctx != nullptr &&
        dflash_kv_cache_batch.buf != nullptr &&
        dflash_kv_cache_batch.n_slots == n_seq &&
        dflash_kv_cache_batch.ctx_window == ctx_window &&
        dflash_kv_cache_batch.n_elem == n_elem &&
        dflash_kv_cache_batch.view.n_layers == n_layers &&
        dflash_kv_cache_batch.view.n_embd_head == n_embd_head &&
        dflash_kv_cache_batch.view.n_head_kv == n_head_kv &&
        dflash_kv_cache_batch.view.ctx_len == total_ctx;

    if (!reuse_batch) {
        dflash_kv_cache_batch.reset();

        const size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_layers * 2 + 8);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * batch_ctx = ggml_init(ctx_params);
        if (!batch_ctx) {
            return false;
        }

        dflash_kv_cache_batch.ctx = batch_ctx;
        dflash_kv_cache_batch.n_slots = n_seq;
        dflash_kv_cache_batch.ctx_window = ctx_window;
        dflash_kv_cache_batch.n_elem = n_elem;
        dflash_kv_cache_batch.view.n_layers = n_layers;
        dflash_kv_cache_batch.view.n_embd_head = n_embd_head;
        dflash_kv_cache_batch.view.n_head_kv = n_head_kv;
        dflash_kv_cache_batch.view.ctx_len = total_ctx;
        dflash_kv_cache_batch.view.ring_size = total_ctx;
        dflash_kv_cache_batch.k_ring.resize(n_layers);
        dflash_kv_cache_batch.v_ring.resize(n_layers);
        dflash_kv_cache_batch.view.k_ring.resize(n_layers);
        dflash_kv_cache_batch.view.v_ring.resize(n_layers);

        for (int il = 0; il < n_layers; ++il) {
            dflash_kv_cache_batch.k_ring[il] =
                ggml_new_tensor_3d(batch_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, total_ctx);
            dflash_kv_cache_batch.v_ring[il] =
                ggml_new_tensor_3d(batch_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, total_ctx);
            dflash_kv_cache_batch.view.k_ring[il] = dflash_kv_cache_batch.k_ring[il];
            dflash_kv_cache_batch.view.v_ring[il] = dflash_kv_cache_batch.v_ring[il];
        }

        dflash_kv_cache_batch.buf = ggml_backend_alloc_ctx_tensors(batch_ctx, gpu_backend);
        if (!dflash_kv_cache_batch.buf) {
            dflash_kv_cache_batch.reset();
            return false;
        }

        for (int il = 0; il < n_layers; ++il) {
            ggml_backend_tensor_memset(dflash_kv_cache_batch.k_ring[il], 0, 0, ggml_nbytes(dflash_kv_cache_batch.k_ring[il]));
            ggml_backend_tensor_memset(dflash_kv_cache_batch.v_ring[il], 0, 0, ggml_nbytes(dflash_kv_cache_batch.v_ring[il]));
        }
        ggml_backend_synchronize(gpu_backend);

        const size_t total_size = ggml_backend_buffer_get_size(dflash_kv_cache_batch.buf);
        LLAMA_LOG_INFO("%s: allocated DFlash batched K/V cache: %.1f MB (%d slots, %d tokens/slot, %d layers, %d elems/token)\n",
            __func__, total_size / (1024.0 * 1024.0), n_seq, ctx_window, n_layers, n_elem);
    }

    const size_t stride = (size_t) n_elem * sizeof(float);
    for (int s = 0; s < n_seq; ++s) {
        auto * cache = slot_caches[s].cache;
        const int n_copy = slot_caches[s].n_copy;
        const int src_offset = slot_caches[s].src_offset;
        const size_t bytes = (size_t) n_copy * stride;
        const size_t dst_offset = (size_t) s * (size_t) ctx_window * stride;

        for (int il = 0; il < n_layers; ++il) {
            auto copy_tensor = [&](ggml_tensor * dst_tensor, const ggml_tensor * src_tensor) -> bool {
                char * dst = (char *) dst_tensor->data + dst_offset;
                const char * src = (const char *) src_tensor->data + (size_t) src_offset * stride;
                const bool fast_copy =
                    cache->fn_copy_d2d_no_check &&
                    cache->fn_prepare_ptr &&
                    cache->fn_prepare_ptr(dst) &&
                    cache->fn_prepare_ptr(src);
                const auto fn_copy = fast_copy ? cache->fn_copy_d2d_no_check : cache->fn_copy_d2d;
                return fn_copy && fn_copy(dst, src, bytes);
            };

            if (!copy_tensor(dflash_kv_cache_batch.k_ring[il], cache->k_ring[il]) ||
                    !copy_tensor(dflash_kv_cache_batch.v_ring[il], cache->v_ring[il])) {
                cross.dflash_kv_cache = nullptr;
                return false;
            }
        }
    }

    auto * sync_cache = slot_caches.empty() ? nullptr : slot_caches[0].cache;
    if (sync_cache && sync_cache->fn_wait_dflash_stream) {
        if (!sync_cache->fn_wait_dflash_stream(gpu_backend)) {
            cross.dflash_kv_cache = nullptr;
            return false;
        }
    } else if (sync_cache && sync_cache->fn_sync_ptr && !dflash_kv_cache_batch.k_ring.empty()) {
        if (!sync_cache->fn_sync_ptr(dflash_kv_cache_batch.k_ring[0]->data)) {
            cross.dflash_kv_cache = nullptr;
            return false;
        }
    } else {
        ggml_backend_synchronize(gpu_backend);
    }

    dflash_kv_cache_batch.view.n_filled = total_ctx;
    dflash_kv_cache_batch.view.write_pos = 0;
    cross.dflash_kv_cache = &dflash_kv_cache_batch.view;
    return true;
}


bool llama_context::dflash_kv_cache_prepare(int ctx_window) {
    dflash_kv_cache_data * active_cache = dflash_kv_cache_active();
    if (!active_cache || ctx_window <= 0 || ctx_window > active_cache->ring_size) {
        return false;
    }
    if (cparams.dflash_n_slots > 1) {
        cross.dflash_kv_cache = nullptr;
        return false;
    }
    if (active_cache->n_filled <= 0) {
        return false;
    }

    active_cache->view.n_filled = std::min(active_cache->n_filled, ctx_window);
    active_cache->view.write_pos = active_cache->write_pos;
    return true;
}


bool llama_context::dflash_kv_cache_update_gpu(
        const void * d_hidden,
        int n_tokens,
        int n_layers,
        int n_embd_layer,
        set_tensor_d2d_fn_t fn_d2d,
        set_tensor_d2d_tensor_fn_t fn_d2d_tensor) {
    if (!d_hidden || n_tokens <= 0 || n_layers <= 0 || n_embd_layer <= 0 || (!fn_d2d && !fn_d2d_tensor)) {
        return false;
    }

    const int64_t n_target_features = (int64_t) n_layers * n_embd_layer;
    if (n_target_features != model.hparams.dflash_n_target_features) {
        return false;
    }

    cross.dflash_kv_update_gpu = d_hidden;
    cross.dflash_kv_update_n_embd = n_target_features;
    cross.dflash_kv_update_n_enc_real = n_tokens;
    cross.dflash_kv_update_fn_set_tensor_d2d = fn_d2d;
    cross.dflash_kv_update_fn_set_tensor_d2d_tensor = fn_d2d_tensor;

    const bool ok = dflash_kv_cache_update(n_tokens);

    cross.dflash_kv_update_gpu = nullptr;
    cross.dflash_kv_update_n_embd = 0;
    cross.dflash_kv_update_n_enc_real = 0;
    cross.dflash_kv_update_fn_set_tensor_d2d = nullptr;
    cross.dflash_kv_update_fn_set_tensor_d2d_tensor = nullptr;

    return ok;
}


bool llama_context::dflash_kv_cache_update(int n_tokens) {
    dflash_kv_cache_data * dflash_kv_cache = dflash_kv_cache_active();
    if (!dflash_kv_cache || n_tokens <= 0 || !llm_arch_is_dflash_drafter(model.arch)) {
        return false;
    }
    if (model.n_devices() > 1) {
        dflash_kv_caches.clear();
        cross.dflash_kv_cache = nullptr;
        return false;
    }
    if (!cross.dflash_kv_update_gpu && !cross.v_embd_gpu && cross.v_embd.empty()) {
        return false;
    }

    n_tokens = std::min(n_tokens, dflash_kv_cache->ring_size);

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }

    const size_t max_nodes = graph_max_nodes(n_tokens);
    auto res = std::make_unique<llm_graph_result>(max_nodes);

    llama_ubatch ubatch = {};
    ubatch.n_tokens = (uint32_t) n_tokens;
    ubatch.n_seq_tokens = (uint32_t) n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;

    const auto gparams = graph_params(res.get(), ubatch, nullptr, LLM_GRAPH_TYPE_DFLASH_KV_UPDATE);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (!gf) {
        return false;
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(res->get_ctx(), gpu_buft);

    if (needed > dflash_kv_cache->update_buf_size) {
        if (dflash_kv_cache->update_buf) {
            ggml_backend_buffer_free(dflash_kv_cache->update_buf);
        }
        dflash_kv_cache->update_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
        dflash_kv_cache->update_buf_size = dflash_kv_cache->update_buf
            ? ggml_backend_buffer_get_size(dflash_kv_cache->update_buf) : 0;
    }
    if (!dflash_kv_cache->update_buf) {
        return false;
    }

    {
        struct ggml_tallocr talloc = ggml_tallocr_new(dflash_kv_cache->update_buf);
        struct ggml_tensor * t = ggml_get_first_tensor(res->get_ctx());
        while (t) {
            if (t->data == nullptr && t->view_src == nullptr) {
                ggml_tallocr_alloc(&talloc, t);
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(res->get_ctx(), t);
        }
    }

    res->set_inputs(&ubatch);

    const ggml_status status = ggml_backend_graph_compute_async(gpu_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!dflash_kv_cache->fn_wait_backend_stream ||
            !dflash_kv_cache->fn_wait_backend_stream(gpu_backend)) {
        ggml_backend_synchronize(gpu_backend);
    }

    const int n_layers = dflash_kv_cache->view.n_layers;
    if ((int) res->dflash_k_update.size() < n_layers || (int) res->dflash_v_update.size() < n_layers) {
        return false;
    }

    const int filled_before = dflash_kv_cache->n_filled;
    const int total = filled_before + n_tokens;
    const bool needs_shift = total > dflash_kv_cache->ring_size;
    const bool fast_append =
        dflash_kv_cache->fn_append_d2d_no_check &&
        dflash_kv_cache->fn_prepare_ptr &&
        !dflash_kv_cache->k_ring.empty() &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->k_ring[0]->data);
    const auto fn_append = fast_append ? dflash_kv_cache->fn_append_d2d_no_check
                                       : dflash_kv_cache->fn_append_d2d;
    bool used_async_copy = false;

    const size_t stride = (size_t) dflash_kv_cache->n_elem * sizeof(float);
    const size_t shift_bytes = (size_t) dflash_kv_cache->ring_size * stride;
    if (needs_shift && shift_bytes > dflash_kv_cache->shift_buf_size) {
        if (dflash_kv_cache->shift_buf) {
            ggml_backend_buffer_free(dflash_kv_cache->shift_buf);
            dflash_kv_cache->shift_buf = nullptr;
            dflash_kv_cache->shift_buf_size = 0;
            dflash_kv_cache->shift_ptr = nullptr;
        }
        dflash_kv_cache->shift_buf = ggml_backend_buft_alloc_buffer(gpu_buft, shift_bytes);
        dflash_kv_cache->shift_buf_size = dflash_kv_cache->shift_buf
            ? ggml_backend_buffer_get_size(dflash_kv_cache->shift_buf) : 0;
        dflash_kv_cache->shift_ptr = dflash_kv_cache->shift_buf
            ? ggml_backend_buffer_get_base(dflash_kv_cache->shift_buf) : nullptr;
    }
    if (needs_shift && (!dflash_kv_cache->shift_buf || !dflash_kv_cache->shift_ptr)) {
        return false;
    }

    const bool fast_copy =
        needs_shift &&
        dflash_kv_cache->fn_copy_d2d_no_check &&
        dflash_kv_cache->fn_prepare_ptr &&
        dflash_kv_cache->shift_ptr &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->k_ring[0]->data) &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->shift_ptr);
    const auto fn_copy = fast_copy ? dflash_kv_cache->fn_copy_d2d_no_check
                                   : dflash_kv_cache->fn_copy_d2d;
    for (int il = 0; il < n_layers; ++il) {
        if (!needs_shift) {
            if (!fn_append(
                    dflash_kv_cache->k_ring[il]->data,
                    res->dflash_k_update[il]->data,
                    dflash_kv_cache->ring_size,
                    filled_before,
                    n_tokens,
                    dflash_kv_cache->n_elem)) {
                return false;
            }
            if (!fn_append(
                    dflash_kv_cache->v_ring[il]->data,
                    res->dflash_v_update[il]->data,
                    dflash_kv_cache->ring_size,
                    filled_before,
                    n_tokens,
                    dflash_kv_cache->n_elem)) {
                return false;
            }
            continue;
        }

        auto shift_and_append = [&](void * dst_cache, const void * src_update) -> bool {
            char * cache = (char *) dst_cache;
            const char * src = (const char *) src_update;

            if (n_tokens >= dflash_kv_cache->ring_size) {
                src += (size_t) (n_tokens - dflash_kv_cache->ring_size) * stride;
                used_async_copy = true;
                return fn_copy(cache, src, shift_bytes);
            }

            const int drop = total - dflash_kv_cache->ring_size;
            const int keep = filled_before - drop;
            if (keep > 0) {
                const size_t keep_bytes = (size_t) keep * stride;
                if (!fn_copy(dflash_kv_cache->shift_ptr, cache + (size_t) drop * stride, keep_bytes)) {
                    return false;
                }
                if (!fn_copy(cache, dflash_kv_cache->shift_ptr, keep_bytes)) {
                    return false;
                }
            }

            used_async_copy = true;
            return fn_copy(cache + (size_t) keep * stride, src, (size_t) n_tokens * stride);
        };

        if (!shift_and_append(dflash_kv_cache->k_ring[il]->data, res->dflash_k_update[il]->data)) {
            return false;
        }
        if (!shift_and_append(dflash_kv_cache->v_ring[il]->data, res->dflash_v_update[il]->data)) {
            return false;
        }
    }

    if (used_async_copy || (!needs_shift && fast_append)) {
        if (dflash_kv_cache->fn_wait_dflash_stream) {
            if (!dflash_kv_cache->fn_wait_dflash_stream(gpu_backend)) {
                return false;
            }
        } else if (!dflash_kv_cache->fn_sync_ptr || !dflash_kv_cache->fn_sync_ptr(dflash_kv_cache->k_ring[0]->data)) {
            return false;
        }
    }

    dflash_kv_cache->write_pos = 0;
    dflash_kv_cache->n_filled = std::min(filled_before + n_tokens, dflash_kv_cache->ring_size);
    dflash_kv_cache->view.n_filled = dflash_kv_cache->n_filled;
    dflash_kv_cache->view.write_pos = dflash_kv_cache->write_pos;
    return true;
}


bool llama_context::dflash_memory_seq_cp_recurrent_ordered(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    llama_memory_recurrent * mem_recr = get_recurrent_mem(get_memory());
    if (!mem_recr) {
        return false;
    }

    using sync_dflash_stream_to_backend_fn_t = bool (*)(ggml_backend_t);
    struct gpu_wait_backend {
        ggml_backend_t backend = nullptr;
        sync_dflash_stream_to_backend_fn_t fn_wait_backend = nullptr;
    };
    std::vector<gpu_wait_backend> gpu_wait_backends;

    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            ggml_backend_reg_t cuda_reg = ggml_backend_dev_backend_reg(dev);
            auto fn_wait_backend = cuda_reg
                ? (sync_dflash_stream_to_backend_fn_t)
                    ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_dflash_stream")
                : nullptr;
            if (!fn_wait_backend) {
                return false;
            }
            gpu_wait_backends.push_back({ backend.get(), fn_wait_backend });
        }
    }
    if (gpu_wait_backends.empty()) {
        return false;
    }

    if (dflash_crash_trace_enabled()) {
        LLAMA_LOG_INFO("%s: dflash crash breadcrumb: recurrent ordered copy enter src=%d dst=%d p0=%d p1=%d\n",
                __func__, (int) seq_id_src, (int) seq_id_dst, (int) p0, (int) p1);
    }
    mem_recr->seq_cp_recurrent_no_sync(seq_id_src, seq_id_dst, p0, p1);
    if (dflash_crash_trace_enabled()) {
        LLAMA_LOG_INFO("%s: dflash crash breadcrumb: recurrent ordered copy enqueued src=%d dst=%d\n",
                __func__, (int) seq_id_src, (int) seq_id_dst);
    }
    for (const auto & gpu_wait : gpu_wait_backends) {
        if (!gpu_wait.fn_wait_backend(gpu_wait.backend)) {
            LLAMA_LOG_ERROR("%s: failed to order DFlash recurrent backup stream before verifier compute\n", __func__);
            GGML_ABORT("failed to order DFlash recurrent backup stream before verifier compute");
        }
    }
    if (dflash_crash_trace_enabled()) {
        LLAMA_LOG_INFO("%s: dflash crash breadcrumb: recurrent ordered copy synced src=%d dst=%d\n",
                __func__, (int) seq_id_src, (int) seq_id_dst);
    }

    return true;
}


bool llama_context::dflash_prefill_capture_info(llama_seq_id seq_id, int32_t * n_tokens, int32_t * n_written) const {
    if (!dflash_capture) {
        return false;
    }

    const auto * plan = dflash_capture->prefill_plan_for_seq(seq_id);
    if (!plan) {
        return false;
    }

    if (!plan->active && plan->n_tokens <= 0) {
        return false;
    }

    if (plan->seq_id != seq_id) {
        return false;
    }

    if (n_tokens)  *n_tokens  = plan->n_tokens;
    if (n_written) *n_written = plan->n_written;
    return true;
}




bool llama_context::dflash_wait_for_gpu_capture_stream() {
    if (!dflash_capture) {
        return false;
    }

    const bool graph_gpu_capture_active =
        cparams.hidden_gpu_n_seqs > 0 ||
        cparams.prefill_gpu_n_seqs > 0 ||
        cparams.tape_gpu_n_seqs > 0;

    if (!graph_gpu_capture_active) {
        return false;
    }

    if (model.n_devices() > 1) {
        // Cross-device peer D2D ring writes are launched on the ring device
        // stream. The per-backend event fast path only orders same-device
        // streams, so use the caller's scheduler sync for split target
        // placement before hidden/tape data can feed the drafter.
        return false;
    }

    if (!dflash_capture->capture_wait_backends.empty()) {
        bool synced = true;
        for (const auto & wait_backend : dflash_capture->capture_wait_backends) {
            synced = wait_backend.fn && wait_backend.backend && wait_backend.fn(wait_backend.backend) && synced;
        }
        return synced;
    }

    auto fn = dflash_capture->fn_sync_backend_to_stream;
    auto backend = dflash_capture->sync_backend_to_stream_backend;

    return fn && backend && fn(backend);
}


bool llama_context::prefill_gpu_write_hidden(void * handle, int slot, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!handle || !dflash_capture) {
        return false;
    }
    if (slot < 0 || slot >= (int) dflash_capture->prefill_gpu.size()) {
        return false;
    }
    auto * pgpu = dflash_capture->prefill_gpu[slot].get();
    if (!pgpu || layer < 0 || layer >= (int) pgpu->layers.size()) {
        return false;
    }
    if (src_offset < 0 || n_tokens <= 0 || n_embd != pgpu->n_embd ||
            src_offset + n_tokens > pgpu->n_tokens) {
        return false;
    }

    auto * tensor = pgpu->layers[layer];
    if (!tensor || !tensor->data) {
        return false;
    }
    if (!dflash_gpu_hidden_span_in_bounds(tensor, src_offset, n_tokens, n_embd, __func__)) {
        return false;
    }

    auto * h = (dflash_cross_ring_handle *)handle;
    // Vulkan: tensor-variant D2D (see cross_ring_gpu_write_hidden). CUDA leaves it null.
    if (h->fn_write_d2d_tensor &&
        h->fn_write_d2d_tensor(h->gpu_ring, layer, ring_pos, tensor, src_offset, n_tokens, n_embd)) {
        return true;
    }
    const size_t src_offset_bytes = (size_t) src_offset * (size_t) n_embd * sizeof(float);
    const void * src = (const char *) tensor->data + src_offset_bytes;
    if (h->fn_write_d2d && h->fn_write_d2d(h->gpu_ring, layer, ring_pos, src, n_tokens, n_embd)) {
        return true;
    }

    static bool warned_prefill_d2h_fallback = false;
    if (!warned_prefill_d2h_fallback) {
        LLAMA_LOG_WARN("%s: prefill GPU D2D ring write unavailable; falling back to GPU readback + H2D ring upload\n",
            __func__);
        warned_prefill_d2h_fallback = true;
    }

    const size_t n_bytes = (size_t) n_tokens * (size_t) n_embd * sizeof(float);
    std::vector<float> staging((size_t) n_tokens * (size_t) n_embd);
    ggml_backend_tensor_get(tensor, staging.data(), src_offset_bytes, n_bytes);
    h->fn_write(h->gpu_ring, layer, ring_pos, staging.data(), n_tokens, n_embd);
    h->fn_synchronize(h->gpu_ring);
    return true;
}


bool llama_context::resize_recurrent_memory(uint32_t new_n_seq_max, bool expand) {
    if (!memory) {
        return false;
    }

    auto * recr = get_recurrent_mem(memory.get());
    if (!recr) {
        return true;
    }

    synchronize();

    const bool ok = expand ? recr->expand(new_n_seq_max) : recr->shrink(new_n_seq_max);
    if (ok) {
        sched_need_reserve = true;
        if (gf_res_prev) {
            gf_res_prev->reset();
        }
    }

    return ok;
}


bool llama_context::tape_replay_conv_gpu_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using prepare_ptr_fn_t = bool (*)(const void *);
    using rebuild_fn_t = bool (*)(void *, const void *, int, int, int);
    using sync_ptr_fn_t = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_rebuild = (rebuild_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    auto fn_sync = (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_rebuild || !fn_sync) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers = dflash_capture->tape_layers;
    if (rec_ids.empty() || tape_layers.empty()) {
        return false;
    }

    const uint32_t n_embd_r = model.hparams.n_embd_r();
    struct conv_upload {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        void * r_state = nullptr;
        ggml_tensor * qkv = nullptr;
        int conv_ch = 0;
        int conv_window = 0;
    };
    std::vector<conv_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & upload : uploads) {
            if (upload.buf) {
                ggml_backend_buffer_free(upload.buf);
                upload.buf = nullptr;
            }
            if (upload.ctx) {
                ggml_free(upload.ctx);
                upload.ctx = nullptr;
            }
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers.size()) {
            cleanup();
            return false;
        }

        const int il = rec_ids[li];
        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        if (!r_tensor) {
            continue;
        }
        if (!dflash_is_cuda_compatible_tensor(r_tensor)) {
            cleanup();
            return false;
        }

        const auto & tape = tape_layers[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens || tape.qkv_mixed.empty()) {
            cleanup();
            return false;
        }

        size_t qkv_seq_offset = 0;
        if (tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) {
                    found = true;
                    break;
                }
                qkv_seq_offset += (size_t) tape.n_tokens * (size_t) tape.conv_channels;
            }
            if (!found) {
                cleanup();
                return false;
            }
        }

        const int64_t conv_ch_i64 = tape.conv_channels;
        if (conv_ch_i64 <= 0 || n_embd_r % conv_ch_i64 != 0 ||
                conv_ch_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }

        const int64_t conv_window_i64 = n_embd_r / conv_ch_i64;
        if (conv_window_i64 <= 0 || conv_window_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }

        const size_t qkv_elems = (size_t) n_accepted * (size_t) conv_ch_i64;
        if (tape.qkv_mixed.size() < qkv_seq_offset + qkv_elems) {
            cleanup();
            return false;
        }

        const size_t ctx_mem = ggml_tensor_overhead();
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) {
            cleanup();
            return false;
        }

        conv_upload upload;
        upload.ctx = ctx;
        upload.qkv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) qkv_elems);
        if (!upload.qkv) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(r_tensor->buffer);
        upload.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!upload.buf) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_tensor_set(upload.qkv, tape.qkv_mixed.data() + qkv_seq_offset, 0, qkv_elems * sizeof(float));

        const size_t r_offset = (size_t) cell_idx * n_embd_r * ggml_element_size(r_tensor);
        upload.r_state = (char *) r_tensor->data + r_offset;
        upload.conv_ch = (int) conv_ch_i64;
        upload.conv_window = (int) conv_window_i64;
        uploads.push_back(upload);
    }

    if (uploads.empty()) {
        cleanup();
        return false;
    }

    for (const auto & upload : uploads) {
        if (!fn_prepare(upload.r_state)) {
            cleanup();
            return false;
        }
    }

    for (const auto & upload : uploads) {
        if (!fn_rebuild(upload.r_state, upload.qkv->data, n_accepted, upload.conv_ch, upload.conv_window)) {
            cleanup();
            GGML_ABORT("DFlash direct CPU-tape conv replay launch failed after validation\n");
        }
    }

    bool synced = true;
    for (const auto & upload : uploads) {
        synced = fn_sync(upload.r_state) && synced;
    }
    cleanup();
    if (!synced) {
        GGML_ABORT("DFlash direct CPU-tape conv replay sync failed after launch\n");
    }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
    return true;
}


bool llama_context::tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    return tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted, true);
}


bool llama_context::tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, bool advance_pos) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using ptr_device_fn_t = bool (*)(const void *, int *);
    using set_device_fn_t = bool (*)(int);
    using sync_device_fn_t = bool (*)(int);
    using rebuild_fn_t = bool (*)(void *, const void *, int, int, int);
    auto fn_ptr_device = (ptr_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_ptr_device");
    auto fn_set_device = (set_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_set_device");
    auto fn_sync_device = (sync_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_device");
    auto fn_rebuild = (rebuild_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    if (!fn_ptr_device || !fn_set_device || !fn_sync_device || !fn_rebuild) {
        return false;
    }

    dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const uint32_t n_embd_r = model.hparams.n_embd_r();
    struct conv_launch {
        void * r_state;
        const void * qkv;
        int conv_ch;
        int conv_window;
        int device;
    };
    std::vector<conv_launch> launches;
    launches.reserve(rec_ids.size());
    bool touched_devices[GGML_CUDA_MAX_DEVICES] = {};

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) {
            return false;
        }

        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        ggml_tensor * qkv_tensor = gpu_tape->layers[li].qkv;
        if (!r_tensor || !r_tensor->data || !qkv_tensor || !qkv_tensor->data) {
            return false;
        }
        if (!dflash_is_cuda_compatible_tensor(r_tensor) || !dflash_is_cuda_compatible_tensor(qkv_tensor)) {
            return false;
        }

        const int64_t conv_ch_i64 = qkv_tensor->ne[0];
        if (conv_ch_i64 <= 0 || n_embd_r % conv_ch_i64 != 0 ||
                conv_ch_i64 > std::numeric_limits<int>::max()) {
            return false;
        }

        const int64_t conv_window_i64 = n_embd_r / conv_ch_i64;
        if (conv_window_i64 <= 0 || conv_window_i64 > std::numeric_limits<int>::max()) {
            return false;
        }

        int r_dev = -1;
        int qkv_dev = -1;
        if (!fn_ptr_device(r_tensor->data, &r_dev) ||
                !fn_ptr_device(qkv_tensor->data, &qkv_dev)) {
            return false;
        }
        if (r_dev < 0 || r_dev >= GGML_CUDA_MAX_DEVICES || qkv_dev != r_dev) {
            return false;
        }

        const size_t r_offset = (size_t) cell_idx * n_embd_r * ggml_element_size(r_tensor);
        launches.push_back({
            (char *) r_tensor->data + r_offset,
            qkv_tensor->data,
            (int) conv_ch_i64,
            (int) conv_window_i64,
            r_dev,
        });
        touched_devices[r_dev] = true;
    }

    if (launches.empty()) {
        return false;
    }

    const int64_t t_gpu_start_us = dflash_capture->profile ? ggml_time_us() : 0;
    for (const auto & launch : launches) {
        if (!fn_set_device(launch.device)) {
            return false;
        }
        if (!fn_rebuild(launch.r_state, launch.qkv, n_accepted, launch.conv_ch, launch.conv_window)) {
            return false;
        }
    }
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; ++dev) {
        if (touched_devices[dev] && !fn_sync_device(dev)) {
            return false;
        }
    }
    if (dflash_capture->profile) {
        const uint64_t elapsed = ggml_time_us() - t_gpu_start_us;
        dflash_capture->profile_replay_conv_enqueue_us += elapsed;
        dflash_capture->profile_conv_gpu_us += elapsed;
    }

    if (advance_pos) {
        mem_recurrent->cells[cell_idx].pos += n_accepted;
    }
    return true;
}


bool llama_context::tape_replay_gdn_direct_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using prepare_ptr_fn_t = bool (*)(const void *);
    using replay_fn_t = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    using sync_ptr_fn_t = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_replay = (replay_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    auto fn_sync = (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_replay || !fn_sync) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers = dflash_capture->tape_layers;
    if (rec_ids.empty() || tape_layers.empty()) {
        return false;
    }

    const uint32_t n_embd_s = model.hparams.n_embd_s();
    struct replay_upload {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        void * state = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * beta = nullptr;
        int S = 0;
        int H_k = 0;
        int H_v = 0;
    };
    std::vector<replay_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & upload : uploads) {
            if (upload.buf) {
                ggml_backend_buffer_free(upload.buf);
                upload.buf = nullptr;
            }
            if (upload.ctx) {
                ggml_free(upload.ctx);
                upload.ctx = nullptr;
            }
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers.size()) {
            cleanup();
            return false;
        }

        const int il = rec_ids[li];
        const auto & tape = tape_layers[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) {
            cleanup();
            return false;
        }

        const int64_t S_i64 = tape.S_k;
        const int64_t H_k_i64 = tape.H_k;
        const int64_t H_v_i64 = tape.H_v;
        if (S_i64 <= 0 || H_k_i64 <= 0 || H_v_i64 <= 0 ||
                S_i64 > std::numeric_limits<int>::max() ||
                H_k_i64 > std::numeric_limits<int>::max() ||
                H_v_i64 > std::numeric_limits<int>::max()) {
            cleanup();
            return false;
        }
        if (!(S_i64 == 16 || S_i64 == 32 || S_i64 == 64 || S_i64 == 128)) {
            cleanup();
            return false;
        }
        if ((size_t) S_i64 * (size_t) S_i64 * (size_t) H_v_i64 != (size_t) n_embd_s) {
            cleanup();
            return false;
        }

        const size_t k_elems = (size_t) S_i64 * (size_t) H_k_i64 * (size_t) n_accepted;
        const size_t v_elems = (size_t) S_i64 * (size_t) H_v_i64 * (size_t) n_accepted;
        const size_t gb_elems = (size_t) H_v_i64 * (size_t) n_accepted;
        if (tape.k.size() < k_elems || tape.v.size() < v_elems ||
                tape.gate.size() < gb_elems || tape.beta.size() < gb_elems) {
            cleanup();
            return false;
        }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        if (!dflash_is_cuda_compatible_tensor(s_tensor)) {
            cleanup();
            return false;
        }

        const size_t ctx_mem = ggml_tensor_overhead() * 4;
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) {
            cleanup();
            return false;
        }

        replay_upload upload;
        upload.ctx = ctx;
        upload.k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) k_elems);
        upload.v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) v_elems);
        upload.gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) gb_elems);
        upload.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) gb_elems);
        if (!upload.k || !upload.v || !upload.gate || !upload.beta) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(s_tensor->buffer);
        upload.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!upload.buf) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }

        ggml_backend_tensor_set(upload.k, tape.k.data(), 0, k_elems * sizeof(float));
        ggml_backend_tensor_set(upload.v, tape.v.data(), 0, v_elems * sizeof(float));
        ggml_backend_tensor_set(upload.gate, tape.gate.data(), 0, gb_elems * sizeof(float));
        ggml_backend_tensor_set(upload.beta, tape.beta.data(), 0, gb_elems * sizeof(float));

        const size_t s_offset = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dflash_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            uploads.push_back(upload);
            cleanup();
            return false;
        }
        upload.state = (char *) s_tensor->data + s_offset;
        upload.S = (int) S_i64;
        upload.H_k = (int) H_k_i64;
        upload.H_v = (int) H_v_i64;
        uploads.push_back(upload);
    }

    if (uploads.empty()) {
        cleanup();
        return false;
    }

    for (const auto & upload : uploads) {
        if (!fn_prepare(upload.state)) {
            cleanup();
            return false;
        }
    }

    for (const auto & upload : uploads) {
        if (!fn_prepare(upload.state) ||
                !fn_replay(upload.state, upload.k->data, upload.v->data, upload.gate->data, upload.beta->data,
                    n_accepted, upload.S, upload.H_k, upload.H_v)) {
            cleanup();
            GGML_ABORT("DFlash direct CPU-tape GDN replay launch failed after validation\n");
        }
    }

    bool synced = true;
    for (const auto & upload : uploads) {
        synced = fn_sync(upload.state) && synced;
    }
    cleanup();
    if (!synced) {
        GGML_ABORT("DFlash direct CPU-tape GDN replay sync failed after launch\n");
    }

    return true;
}


bool llama_context::tape_replay_gdn_direct_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
    if (!cuda_reg) {
        return false;
    }
    using ptr_device_fn_t = bool (*)(const void *, int *);
    using prepare_ptr_fn_t = bool (*)(const void *);
    using replay_fn_t = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    auto fn_ptr_device = (ptr_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_ptr_device");
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_replay = (replay_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    if (!fn_ptr_device || !fn_prepare || !fn_replay) {
        return false;
    }

    dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const uint32_t n_embd_s = model.hparams.n_embd_s();
    struct replay_launch {
        void * state;
        const void * k;
        const void * v;
        const void * gate;
        const void * beta;
        int S;
        int H_k;
        int H_v;
        int device;
    };
    std::vector<replay_launch> launches;
    launches.reserve(rec_ids.size());

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) {
            return false;
        }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        auto & tl = gpu_tape->layers[li];
        if (!dflash_is_cuda_compatible_tensor(s_tensor) || !dflash_is_cuda_compatible_tensor(tl.k) ||
                !dflash_is_cuda_compatible_tensor(tl.v) || !dflash_is_cuda_compatible_tensor(tl.gate) ||
                !dflash_is_cuda_compatible_tensor(tl.beta)) {
            return false;
        }

        const int64_t S_i64 = tl.v->ne[0];
        const int64_t H_v_i64 = tl.v->ne[1];
        const int64_t H_k_i64 = tl.k->ne[1];
        if (tl.k->ne[0] != S_i64 || tl.gate->ne[0] != 1 || tl.beta->ne[0] != 1 ||
                tl.gate->ne[1] != H_v_i64 || tl.beta->ne[1] != H_v_i64 ||
                S_i64 <= 0 || H_k_i64 <= 0 || H_v_i64 <= 0 ||
                S_i64 > std::numeric_limits<int>::max() ||
                H_k_i64 > std::numeric_limits<int>::max() ||
                H_v_i64 > std::numeric_limits<int>::max()) {
            return false;
        }
        if (!(S_i64 == 16 || S_i64 == 32 || S_i64 == 64 || S_i64 == 128)) {
            return false;
        }
        if (!llama_dflash_replay_state_shape_valid_for_test(S_i64, H_v_i64, n_embd_s)) {
            return false;
        }

        const size_t s_offset = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dflash_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            return false;
        }
        launches.push_back({
            (char *) s_tensor->data + s_offset,
            tl.k->data,
            tl.v->data,
            tl.gate->data,
            tl.beta->data,
            (int) S_i64,
            (int) H_k_i64,
            (int) H_v_i64,
            -1,
        });
    }

    if (launches.empty()) {
        return false;
    }
    int replay_device = -1;
    bool replay_mixed_devices = false;
    for (auto & launch : launches) {
        int device = -1;
        if (!fn_ptr_device(launch.state, &device)) {
            return false;
        }
        int k_device = -1;
        int v_device = -1;
        int gate_device = -1;
        int beta_device = -1;
        if (!fn_ptr_device(launch.k, &k_device) ||
                !fn_ptr_device(launch.v, &v_device) ||
                !fn_ptr_device(launch.gate, &gate_device) ||
                !fn_ptr_device(launch.beta, &beta_device)) {
            return false;
        }
        if (k_device != device || v_device != device || gate_device != device || beta_device != device) {
            return false;
        }
        launch.device = device;
        if (!replay_mixed_devices && replay_device < 0) {
            replay_device = device;
        } else if (!replay_mixed_devices && device != replay_device) {
            replay_device = -1;
            replay_mixed_devices = true;
        }
    }

    const int64_t t_start_us = dflash_capture->profile ? ggml_time_us() : 0;
    dflash_capture->replay_sync_ptrs.clear();
    dflash_capture->replay_sync_device = replay_mixed_devices ? -1 : replay_device;
    for (const auto & launch : launches) {
        if (!fn_prepare(launch.state) ||
                !fn_replay(launch.state, launch.k, launch.v, launch.gate, launch.beta,
                    n_accepted, launch.S, launch.H_k, launch.H_v)) {
            GGML_ABORT("DFlash direct GPU GDN replay launch failed after validation\n");
        }
        if (replay_mixed_devices) {
            dflash_capture->replay_sync_ptrs.push_back(launch.state);
        }
    }
    if (dflash_capture->profile) {
        const uint64_t elapsed = ggml_time_us() - t_start_us;
        dflash_capture->profile_replay_gdn_enqueue_us += elapsed;
        dflash_capture->profile_conv_gpu_us += elapsed;
        dflash_capture->profile_replay_direct_gpu += 1;
        dflash_capture->profile_replay_layers += launches.size();
    }
    dflash_capture->replay_sync_ptr = replay_mixed_devices ? nullptr : launches.back().state;
    return true;
}


const char * llama_context::dflash_hidden_capture_unavailable_reason() const {
    return dflash_capture_invalid_reason.empty()
        ? ""
        : dflash_capture_invalid_reason.c_str();
}


// Readers return data from the active DFlash slot; multi-slot callers must
// call llama_dflash_set_active_slot() before reading.
float * llama_context::get_layer_hidden(int layer_idx) {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return nullptr;
    }
    auto & buf = (*sh)[layer_idx];
    if (buf.n_tokens <= 0 || buf.data.empty()) {
        return nullptr;
    }
    return buf.data.data();
}


float * llama_context::get_logits_argmax_probs() {
    synchronize();
    output_reorder();
    if (logits_argmax_prob_buf.empty()) {
        return nullptr;
    }
    return logits_argmax_prob_buf.data();
}


float * llama_context::get_logits_argmax_probs_ith(int32_t i) {
    synchronize();
    output_reorder();
    if (logits_argmax_prob_buf.empty()) {
        return nullptr;
    }
    try {
        const int64_t j = output_resolve_row(i);
        return logits_argmax_prob_buf.data() + j * logits_argmax_k;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid argmax prob id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}


// DFlash: get capture layer IDs
int llama_context::dflash_capture_layer_ids(int32_t * out, int out_size) const {
    auto * cap = dflash_capture.get();
    if (!cap)
        return 0;
    const auto & ids = cap->layer_ids;
    int n = (int) ids.size();
    if (out && out_size > 0) {
        int copy = std::min(n, out_size);
        for (int i = 0; i < copy; i++) {
            out[i] = ids[i];
        }
    }
    return n;
}

int llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_rollback requires hybrid memory\n", __func__);
        return 0;
    }

    const bool profile = dflash_capture && dflash_profile_has(dflash_capture->profile_flags, DFLASH_PROFILE_COPY);
    const int64_t t_start_us = profile ? ggml_time_us() : 0;
    int64_t t_phase_us = t_start_us;
    int64_t attn_us = 0;
    int64_t recurrent_restore_us = 0;
    int64_t tape_launch_us = 0;
    llama_memory_recurrent_copy_profile recurrent_restore_profile = {};
    auto profile_lap = [&](int64_t & dst) {
        if (!profile) {
            return;
        }
        const int64_t now = ggml_time_us();
        dst += now - t_phase_us;
        t_phase_us = now;
    };

    auto * mem_attn = mem_hybrid->get_mem_attn();
    auto * mem_recr = mem_hybrid->get_mem_recr();

    if (tree_bufs.n_tokens > 0) {
        // Tree mode: branch tokens may have polluted KV at accepted positions.
        // Remove ALL entries from n_past_before onwards and restore from backup.
        mem_attn->seq_rm(seq_id, n_past_before, -1);
        mem_attn->seq_cp(seq_backup, seq_id, n_past_before, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    } else {
        // Flat mode: no duplicate entries at same position, safe to keep accepted KV
        int kv_keep_pos = n_past_before + n_accepted;
        mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    }
    profile_lap(attn_us);

    // Recurrent state: restore from backup, then tape replay
    mem_recr->seq_rm(seq_id, -1, -1);
    if (profile) {
        mem_recr->recurrent_copy_profile_reset();
    }
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);
    if (profile) {
        recurrent_restore_profile = mem_recr->recurrent_copy_profile();
    }
    mem_recr->seq_rm(seq_backup, -1, -1);
    profile_lap(recurrent_restore_us);

    // Replay DeltaNet state updates for accepted tokens.
    // On Vulkan, tape_replay_cpu is used (GPU tape crashes due to missing GDN ops).
    // If tape_replay_cpu fails (NaN guard or empty tape), return n_accepted to signal
    // the server to re-decode the accepted positions instead.
    int n_positions_needing_reeval = 0;
    {
        n_positions_needing_reeval = tape_replay(seq_id, n_accepted);
        if (n_positions_needing_reeval > 0) {
            // Tape replay was skipped (e.g., Vulkan). The recurrent state is behind
            // by n_positions_needing_reeval positions. Trim the attention memory
            // to match the recurrent state, so seq_pos_max is consistent.
            mem_attn->seq_rm(seq_id, n_past_before, -1);
        }
    }
    profile_lap(tape_launch_us);

    if (profile) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: rollback accepted=%d attn=%.3f ms recurrent_restore=%.3f ms "
            "rollback_restore_enqueue=%.3f ms rollback_restore_sync=%.3f ms "
            "rollback_restore_layers=%" PRIu64 " rollback_restore_tensors=%" PRIu64
            " rollback_restore_cuda_d2d=%" PRIu64 " rollback_restore_fallback=%" PRIu64
            " tape_launch=%.3f ms total=%.3f ms\n",
            __func__, n_accepted,
            attn_us / 1e3,
            recurrent_restore_us / 1e3,
            recurrent_restore_profile.enqueue_us / 1e3,
            recurrent_restore_profile.sync_us / 1e3,
            recurrent_restore_profile.layers_scanned,
            recurrent_restore_profile.tensors_copied,
            recurrent_restore_profile.cuda_d2d_queued,
            recurrent_restore_profile.fallback_copies,
            tape_launch_us / 1e3,
            (ggml_time_us() - t_start_us) / 1e3);
    }

    return n_positions_needing_reeval;
}


// CPU fallback for tape replay (used when no GPU backend available)
// Returns 0 on success, n_accepted on failure (caller should re-decode instead).
int llama_context::tape_replay_cpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;
    const uint32_t n_embd_s = hparams.n_embd_s();
    if (dflash_capture->profile) {
        dflash_capture->profile_replay_cpu_fallback += 1;
        dflash_capture->profile_replay_layers += rec_ids.size();
    }

    // Helper to check for NaN/Inf in a float range
    auto tape_data_ok = [](const float * data, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return false;
            }
        }
        return true;
    };

    // Accumulate results for all layers before writing back.
    // This prevents partial write-back: if layer 2 fails, layers 0-1 are not
    // committed to GPU. On failure, no state is written and the caller
    // re-decodes from the backup position, advancing all layers consistently.
    struct layer_result {
        int layer_id;
        ggml_tensor * s_tensor;
        size_t s_offset;
        size_t state_bytes;
        std::vector<float> state;
    };
    std::vector<layer_result> results;
    results.reserve(rec_ids.size());

    int n_layers_skipped_dim = 0;
    int n_layers_processed   = 0;
    const bool dbg = dflash_diagnostic_debug_enabled();
    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];

        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { ++n_layers_skipped_dim; continue; }

        const int64_t S = tape.S_k;
        const int64_t H_k = tape.H_k;
        const int64_t H_v = tape.H_v;
        if (S <= 0 || H_k <= 0 || H_v <= 0) {
            if (dbg) fprintf(stderr, "[dflash-tr-cpu] layer=%d SKIP zero-dim S=%lld H_k=%lld H_v=%lld\n", il, (long long)S, (long long)H_k, (long long)H_v);
            ++n_layers_skipped_dim; continue;
        }
        if ((size_t) S * (size_t) S * (size_t) H_v != (size_t) n_embd_s) {
            if (dbg) fprintf(stderr, "[dflash-tr-cpu] layer=%d SKIP dim-check S*S*H_v=%zu != n_embd_s=%u (S=%lld H_v=%lld)\n",
                    il, (size_t)(S*S*H_v), n_embd_s, (long long)S, (long long)H_v);
            ++n_layers_skipped_dim; continue;
        }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        if (!s_tensor) {
            if (dbg) fprintf(stderr, "[dflash-tr-cpu] layer=%d SKIP no s_tensor\n", il);
            ++n_layers_skipped_dim; continue;
        }
        const size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
        const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
        if (!dflash_tensor_span_in_bounds(s_tensor, s_offset, state_bytes)) {
            LLAMA_LOG_WARN("%s: DFlash CPU recurrent replay state span out of bounds "
                    "(layer=%d tensor_bytes=%zu offset=%zu bytes=%zu cell=%d n_embd_s=%u)\n",
                    __func__, il, s_tensor ? ggml_nbytes(s_tensor) : (size_t) 0,
                    s_offset, state_bytes, cell_idx, n_embd_s);
            continue;
        }
        std::vector<float> state(n_embd_s);
        ggml_backend_tensor_get(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));

        // Guard: check tape data is available and valid before using it
        size_t tape_k_size = (size_t)n_accepted * S * H_k;
        size_t tape_v_size = (size_t)n_accepted * S * H_v;
        size_t tape_gate_size = (size_t)n_accepted * H_v;
        size_t tape_beta_size = (size_t)n_accepted * H_v;
        if (tape.k.empty() || tape.v.empty() || tape.gate.empty() || tape.beta.empty() ||
            tape.k.size() < tape_k_size || tape.v.size() < tape_v_size ||
            tape.gate.size() < tape_gate_size || tape.beta.size() < tape_beta_size ||
            !tape_data_ok(tape.k.data(), tape_k_size) ||
            !tape_data_ok(tape.v.data(), tape_v_size) ||
            !tape_data_ok(tape.gate.data(), tape_gate_size) ||
            !tape_data_ok(tape.beta.data(), tape_beta_size) ||
            !tape_data_ok(state.data(), n_embd_s)) {
            LLAMA_LOG_WARN("%s: DFlash CPU tape replay invalid tape data "
                    "(layer=%d n_accepted=%d k_size=%zu/%zu v_size=%zu/%zu) -> falling back to re-decode\n",
                    __func__, il, n_accepted,
                    tape_k_size, tape.k.size(),
                    tape_v_size, tape.v.size());
            return n_accepted;
        }

        for (int tok = 0; tok < n_accepted; ++tok) {
            for (int64_t hv = 0; hv < H_v; ++hv) {
                const int64_t hk = hv % H_k;
                float g_val = exp2f(tape.gate[tok * H_v + hv] * 1.442695041f);
                float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]));

                float * S_h = state.data() + hv * S * S;
                const float * k_t = tape.k.data() + tok * (S * H_k) + hk * S;
                const float * v_t = tape.v.data() + tok * (S * H_v) + hv * S;

                // kv = S^T @ k, delta = (v - g*kv) * beta, S = g*S + k⊗delta (fused)
                for (int64_t col = 0; col < S; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        kv += S_h[col * S + row] * k_t[row];
                    }
                    float delta_col = (v_t[col] - g_val * kv) * b_val;
                    for (int64_t row = 0; row < S; ++row) {
                        S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
                    }
                }
            }
        }

        // Guard: check result for NaN/Inf before accumulating
        if (!tape_data_ok(state.data(), n_embd_s)) {
            LLAMA_LOG_WARN("%s: DFlash CPU tape replay produced NaN/Inf in result "
                    "(layer=%d n_accepted=%d) -> falling back to re-decode\n",
                    __func__, il, n_accepted);
            return n_accepted;
        }

        // Accumulate result; write back only after all layers succeed
        if (dbg) {
            double sum = 0.0, abs = 0.0;
            for (size_t i = 0; i < n_embd_s; ++i) { sum += state[i]; abs += std::fabs(state[i]); }
            fprintf(stderr, "[dflash-tr-cpu] layer=%d OK n_accepted=%d S=%lld H_k=%lld H_v=%lld cell=%d "
                    "state[0..3]=%.6g,%.6g,%.6g,%.6g sum=%.4g abs_sum=%.4g gate0=%.6g beta0=%.6g k0=%.6g v0=%.6g\n",
                    il, n_accepted, (long long)S, (long long)H_k, (long long)H_v, cell_idx,
                    state[0], state[1], state[2], state[3], sum, abs,
                    tape.gate.empty()?0.0f:tape.gate[0], tape.beta.empty()?0.0f:tape.beta[0],
                    tape.k.empty()?0.0f:tape.k[0], tape.v.empty()?0.0f:tape.v[0]);
        }
        results.push_back({il, s_tensor, s_offset, state_bytes, std::move(state)});
        ++n_layers_processed;
    }

    if (dbg) {
        fprintf(stderr, "[dflash-tr-cpu] DONE n_accepted=%d cell=%d n_rec=%zu processed=%d skipped=%d results=%zu -> returning %d\n",
                n_accepted, cell_idx, rec_ids.size(), n_layers_processed, n_layers_skipped_dim, results.size(),
                results.empty() ? n_accepted : 0);
    }

    // All layers passed — write back results in one pass
    for (auto & r : results) {
        ggml_backend_tensor_set(r.s_tensor, r.state.data(), r.s_offset, r.state_bytes);
    }

    return results.empty() ? n_accepted : 0;
}


int llama_context::tape_replay(llama_seq_id seq_id, int n_accepted) {
    if (!dflash_capture || n_accepted <= 0) {
        return 0;
    }

    if (const char * env = std::getenv("GGML_DFLASH_FORCE_REDECODE"); env && std::atoi(env) != 0) {
        return n_accepted;
    }

    // ensure any previous async replay is complete before launching a new one
    tape_replay_sync();

    // GPU-resident tape path: data already on GPU from graph-embedded copies
    dflash_tape_gpu * const gpu_tape = dflash_capture->active_tape();
    const bool use_gpu_tape = (gpu_tape != nullptr &&
                               n_accepted <= gpu_tape->max_tokens &&
                               n_accepted <= gpu_tape->n_tokens);

    if (!use_gpu_tape && dflash_capture->tape_layers.empty()) {
        return 0;
    }

    auto * mem_recurrent = dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (!mem_recurrent) {
        auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
        if (mem_hybrid) {
            mem_recurrent = mem_hybrid->get_mem_recr();
        }
    }
    if (!mem_recurrent) {
        LLAMA_LOG_WARN("%s: tape replay requires recurrent memory\n", __func__);
        return 0;
    }

    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;

    // find the tail cell for this seq_id
    int32_t cell_idx = -1;
    if (seq_id >= 0 && (uint32_t) seq_id < mem_recurrent->size) {
        int32_t tail = mem_recurrent->cells[seq_id].tail;
        if (tail >= 0) {
            cell_idx = tail;
        }
    }
    if (cell_idx < 0) {
        LLAMA_LOG_WARN("%s: no active cell for seq %d\n", __func__, seq_id);
        return 0;
    }

    const uint32_t n_embd_s = hparams.n_embd_s();

    // find a GPU backend for graph computation
    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return n_fallback;
    }

    // partial offload: if any recurrent layer's state lives on CPU, fall back to CPU replay
    // (GPU graph uses DeviceToDevice copies that crash when the state buffer is host memory)
    for (int li = 0; li < (int) rec_ids.size(); ++li) {
        ggml_tensor * s_tensor = mem_recurrent->s_l[rec_ids[li]];
        if (s_tensor && s_tensor->buffer && ggml_backend_buffer_is_host(s_tensor->buffer)) {
            int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return n_fallback;
        }
    }

    // Vulkan GPU tape replay crashes (GDN ops not properly supported on Vulkan backends).
    // Use CPU fallback with NaN guard instead.
    // Note: Vulkan reports as GGML_BACKEND_DEVICE_TYPE_GPU, so we detect by driver name.
    {
        const char * dev_name = ggml_backend_dev_name(ggml_backend_get_device(gpu_backend));
        bool is_vulkan = dev_name && (strstr(dev_name, "vulkan") || strstr(dev_name, "Vulkan") ||
                                      strstr(dev_name, "RADV") || strstr(dev_name, "ANV") ||
                                      strstr(dev_name, "radeon"));
        if (is_vulkan) {
            int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return n_fallback;
        }
    }

    if (use_gpu_tape && tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)) {
        dflash_capture->replay_pending = true;
        dflash_capture->replay_gpu_backend = nullptr;
        dflash_capture->replay_graph_ctx = nullptr;
        dflash_capture->replay_direct_gpu = true;
        dflash_capture->replay_n_accepted = n_accepted;
        dflash_capture->replay_cell_idx = cell_idx;
        dflash_capture->replay_seq_id = seq_id;
        dflash_capture->replay_mem_recurrent = mem_recurrent;
        return 0;
    }

    const bool multi_gpu_target = model.n_devices() > 1;
    if (multi_gpu_target) {
        if (tape_replay_gdn_direct_from_cpu_tape(mem_recurrent, cell_idx, n_accepted)) {
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return 0;
        }
        if (!dflash_capture->multi_gpu_replay_fallback_logged) {
            LLAMA_LOG_WARN("%s: multi-GPU target detected (%zu devices); exact CUDA DFlash replay unavailable, using CPU recurrent replay fallback\n",
                    __func__, model.n_devices());
            dflash_capture->multi_gpu_replay_fallback_logged = true;
        }
        int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return n_fallback;
    }

    // GPU tape replay: build a ggml graph with GDN ops for all recurrent layers
    const int n_rec = (int) rec_ids.size();
    if (n_rec == 0) goto conv_rebuild;

    {
        // per layer: k_view + v_view + g_view + b_view + q + b_sigmoid + s_view + GDN + result_state + s_write + cpy = ~11 tensors
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t)n_rec * 14 + 4) + ggml_graph_overhead_custom(n_rec * 12, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 12, false);

        // track Q tensors that need zeroing (only when NOT using GPU tape for k/v/g/b)
        struct replay_input {
            ggml_tensor * q;
            ggml_tensor * k;
            ggml_tensor * v;
            ggml_tensor * g;
            ggml_tensor * b;
            size_t tape_li;
        };
        std::vector<replay_input> inputs;
        inputs.reserve(n_rec);
        bool replay_graph_unsafe = false;

        for (int li = 0; li < n_rec; ++li) {
            int il = rec_ids[li];

            int64_t S, H_k, H_v;
            if (use_gpu_tape) {
                auto & tl = gpu_tape->layers[li];
                S   = tl.k->ne[0];
                H_k = tl.k->ne[1];
                H_v = tl.v->ne[1];
            } else {
                auto & tape = tape_layers[li];
                if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;
                S   = tape.S_k;
                H_k = tape.H_k;
                H_v = tape.H_v;
            }

            if (S <= 0 || H_k <= 0 || H_v <= 0) {
                continue;
            }

            if (!llama_dflash_replay_gdn_supported_s_for_test(S) ||
                    !llama_dflash_replay_state_shape_valid_for_test(S, H_v, n_embd_s)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds or unsupported shape "
                        "(layer=%d S=%lld H_v=%lld n_embd_s=%u use_gpu_tape=%d); %s\n",
                        __func__, il, (long long) S, (long long) H_v, n_embd_s,
                        use_gpu_tape ? 1 : 0,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }

            ggml_tensor * k_in, * v_in, * g_in, * b_in;

            if (use_gpu_tape) {
                // create views into pre-filled GPU tape tensors (zero upload)
                auto & tl = gpu_tape->layers[li];
                k_in = ggml_view_4d(ctx, tl.k, S, H_k, (int64_t)n_accepted, (int64_t)1,
                    tl.k->nb[1], tl.k->nb[2], tl.k->nb[2] * n_accepted, 0);
                v_in = ggml_view_4d(ctx, tl.v, S, H_v, (int64_t)n_accepted, (int64_t)1,
                    tl.v->nb[1], tl.v->nb[2], tl.v->nb[2] * n_accepted, 0);
                g_in = ggml_view_4d(ctx, tl.gate, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1,
                    tl.gate->nb[1], tl.gate->nb[2], tl.gate->nb[2] * n_accepted, 0);
                b_in = ggml_view_4d(ctx, tl.beta, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1,
                    tl.beta->nb[1], tl.beta->nb[2], tl.beta->nb[2] * n_accepted, 0);
            } else {
                // allocate new tensors (will be filled from CPU tape data)
                k_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, (int64_t)1);
                v_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, (int64_t)n_accepted, (int64_t)1);
                g_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1);
                b_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1);
                ggml_set_input(k_in); ggml_set_input(v_in);
                ggml_set_input(g_in); ggml_set_input(b_in);
            }

            // Q: zeros (attention output discarded, only state update matters)
            ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, (int64_t)1);
            ggml_set_input(q_in);

            // apply sigmoid to beta on GPU
            ggml_tensor * b_sigmoid = ggml_sigmoid(ctx, b_in);

            // state view: reads directly from recurrent memory GPU buffer (zero-copy)
            ggml_tensor * s_tensor = mem_recurrent->s_l[il];
            if (!s_tensor) {
                LLAMA_LOG_WARN("%s: missing recurrent state tensor for DFlash replay layer=%d; %s\n",
                        __func__, il, use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            const size_t state_bytes = (size_t) n_embd_s * ggml_element_size(s_tensor);
            size_t s_byte_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
            if (!dflash_tensor_span_in_bounds(s_tensor, s_byte_offset, state_bytes)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds "
                        "(layer=%d tensor_bytes=%zu offset=%zu bytes=%zu cell=%d n_embd_s=%u); %s\n",
                        __func__, il, s_tensor ? ggml_nbytes(s_tensor) : (size_t) 0,
                        s_byte_offset, state_bytes, cell_idx, n_embd_s,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            ggml_tensor * s_view = ggml_view_4d(ctx, s_tensor, S, S, H_v, (int64_t)1,
                S * ggml_element_size(s_tensor),
                S * S * ggml_element_size(s_tensor),
                S * S * H_v * ggml_element_size(s_tensor),
                s_byte_offset);

            // GDN op: same kernel as forward pass, bit-identical state update
            ggml_tensor * result = ggml_gated_delta_net(ctx, q_in, k_in, v_in, g_in, b_sigmoid, s_view, 1);

            // extract state from result (layout: [attn_output | new_state])
            size_t attn_bytes = (size_t)(S * H_v * n_accepted) * ggml_element_size(result);
            const size_t result_state_bytes = (size_t) n_embd_s * ggml_element_size(result);
            if (!dflash_tensor_span_in_bounds(result, attn_bytes, result_state_bytes)) {
                LLAMA_LOG_WARN("%s: DFlash recurrent replay view out of bounds "
                        "(layer=%d result_bytes=%zu offset=%zu bytes=%zu S=%lld H_v=%lld n_accepted=%d); %s\n",
                        __func__, il, ggml_nbytes(result), attn_bytes, result_state_bytes,
                        (long long) S, (long long) H_v, n_accepted,
                        use_gpu_tape ? "skipping GPU replay" : "falling back to CPU replay");
                replay_graph_unsafe = true;
                break;
            }
            ggml_tensor * result_state = ggml_view_1d(ctx, result, n_embd_s, attn_bytes);

            // write-back view: points to same location in s_l[il]
            ggml_tensor * s_write = ggml_view_1d(ctx, s_tensor, n_embd_s, s_byte_offset);

            // copy result state back to recurrent memory (GPU→GPU)
            ggml_tensor * cpy = ggml_cpy(ctx, result_state, s_write);
            ggml_build_forward_expand(graph, cpy);

            inputs.push_back({ q_in, k_in, v_in, g_in, b_in, (size_t)li });
        }

        if (replay_graph_unsafe) {
            ggml_free(ctx);
            if (!use_gpu_tape) {
                int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
                tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
                return n_fallback;
            }
            goto conv_rebuild;
        }

        if (inputs.empty()) {
            ggml_free(ctx);
            goto conv_rebuild;
        }

        // allocate non-view tensors on GPU (reuse persistent buffer)
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
        size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, gpu_buft);


        if (needed > dflash_capture->replay_buf_size) {
            if (dflash_capture->replay_buf) {
                ggml_backend_buffer_free(dflash_capture->replay_buf);
            }
            dflash_capture->replay_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            dflash_capture->replay_buf_size = dflash_capture->replay_buf
                ? ggml_backend_buffer_get_size(dflash_capture->replay_buf) : 0;
        }

        if (!dflash_capture->replay_buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU buffer for tape replay, falling back to CPU\n", __func__);
            ggml_free(ctx);
            int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return n_fallback;
        }

        // assign tensors within the persistent buffer
        {
            struct ggml_tallocr talloc = ggml_tallocr_new(dflash_capture->replay_buf);
            struct ggml_tensor * t = ggml_get_first_tensor(ctx);
            while (t) {
                if (t->data == nullptr && t->view_src == nullptr) {
                    ggml_tallocr_alloc(&talloc, t);
                } else if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
                t = ggml_get_next_tensor(ctx, t);
            }
        }


        // upload data for tensors that need it

        for (size_t ii = 0; ii < inputs.size(); ++ii) {
            auto & inp = inputs[ii];

            // Q: always needs zeros
            {
                const int64_t S = inp.q->ne[0];
                const int64_t H = inp.q->ne[1];
                size_t q_size = (size_t)(S * H * n_accepted);
                if (dflash_capture->replay_zeros.size() < q_size) {
                    dflash_capture->replay_zeros.resize(q_size, 0.0f);
                }
                ggml_backend_tensor_set(inp.q, dflash_capture->replay_zeros.data(), 0, ggml_nbytes(inp.q));
            }

            // K, V, gate, beta: only upload from CPU if not using GPU tape
            if (!use_gpu_tape) {
                auto & tape = tape_layers[inp.tape_li];
                const int64_t S   = tape.S_k;
                const int64_t H_k = tape.H_k;
                const int64_t H_v = tape.H_v;

                ggml_backend_tensor_set(inp.k, tape.k.data(), 0, S * H_k * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.v, tape.v.data(), 0, S * H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.g, tape.gate.data(), 0, H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.b, tape.beta.data(), 0, H_v * n_accepted * sizeof(float));
            }
        }

        // compute: launch GDN ops + state copies on GPU (async — overlap with next draft)

        const int64_t t_replay_enqueue_us = dflash_capture->profile ? ggml_time_us() : 0;
        const ggml_status replay_status = ggml_backend_graph_compute_async(gpu_backend, graph);

        if (replay_status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_WARN("%s: GPU DFlash recurrent replay graph failed with status %d; %s\n",
                    __func__, (int) replay_status,
                    use_gpu_tape ? "CPU fallback unavailable for GPU tape" : "falling back to CPU");
            ggml_free(ctx);
            if (!use_gpu_tape) {
                int n_fallback = tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
                tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
                return n_fallback;
            }
            return 0;
        }
        if (dflash_capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_replay_enqueue_us;
            dflash_capture->profile_replay_gdn_enqueue_us += elapsed;
            dflash_capture->profile_conv_gpu_us += elapsed;
            dflash_capture->profile_replay_ggml_gpu += 1;
            dflash_capture->profile_replay_layers += inputs.size();
        }

        // save deferred state for async completion
        dflash_capture->replay_pending = true;
        dflash_capture->replay_gpu_backend = gpu_backend;
        dflash_capture->replay_graph_ctx = ctx; // freed in tape_replay_sync
        dflash_capture->replay_direct_gpu = false;
        dflash_capture->replay_sync_device = -1;
        dflash_capture->replay_n_accepted = n_accepted;
        dflash_capture->replay_cell_idx = cell_idx;
        dflash_capture->replay_seq_id = seq_id;
        dflash_capture->replay_mem_recurrent = mem_recurrent;
        return 0; // conv rebuild deferred to tape_replay_sync()
    }

conv_rebuild:
    tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
    return 0;
}


void llama_context::allocate_hidden_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture || dflash_capture->layer_ids.empty()) {
        return;
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (!dflash_capture->gpu_capture_enabled) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->fn_sync_backend_to_stream = nullptr;
        dflash_capture->sync_backend_to_stream_backend = nullptr;
        return;
    }
    if (model.n_devices() > 1 && !llama_dflash_allow_multi_gpu_tape()) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->fn_sync_backend_to_stream = nullptr;
        dflash_capture->sync_backend_to_stream_backend = nullptr;
        return;
    }
    if (!llama_dflash_gpu_hidden_supported_arch(model.arch)) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->fn_sync_backend_to_stream = nullptr;
        dflash_capture->sync_backend_to_stream_backend = nullptr;
        return;
    }

    const int n_layers = (int) dflash_capture->layer_ids.size();
    const int64_t n_embd = model.hparams.n_embd;

    dflash_capture->hidden_gpu.clear();
    dflash_capture->hidden_gpu.reserve(n_slots);

    size_t total_size = 0;
    for (int slot = 0; slot < n_slots; ++slot) {
        auto hidden = std::make_unique<dflash_hidden_gpu>();
        hidden->layers.resize(n_layers);
        hidden->layer_ids = dflash_capture->layer_ids;
        hidden->n_embd = n_embd;
        hidden->max_tokens = max_tokens;

        for (int i = 0; i < n_layers; ++i) {
            const int il = dflash_capture->layer_ids[i];
            ggml_backend_dev_t layer_dev = model.dev_layer(il);
            ggml_backend_t layer_backend = dflash_backend_for_dev(backends, layer_dev);
            if (!layer_backend) {
                LLAMA_LOG_WARN("%s: no GPU backend for hidden layer %d device %s; using callback hidden fallback\n",
                    __func__, il, layer_dev ? ggml_backend_dev_name(layer_dev) : "<null>");
                dflash_capture->hidden_gpu.clear();
                return;
            }

            const size_t ctx_mem = ggml_tensor_overhead() * 2;
            struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
            struct ggml_context * hidden_ctx = ggml_init(ctx_params);
            if (!hidden_ctx) {
                LLAMA_LOG_WARN("%s: failed to create GPU hidden context for slot %d layer %d; using callback hidden fallback\n",
                    __func__, slot, il);
                dflash_capture->hidden_gpu.clear();
                return;
            }
            hidden->ctxs.push_back(hidden_ctx);

            hidden->layers[i] = ggml_new_tensor_2d(hidden_ctx, GGML_TYPE_F32, n_embd, (int64_t) max_tokens);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(hidden_ctx, layer_backend);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate GPU hidden buffer for slot %d layer %d; using callback hidden fallback\n",
                    __func__, slot, il);
                dflash_capture->hidden_gpu.clear();
                return;
            }
            hidden->bufs.push_back(buf);
            dflash_capture_add_wait_backend(*dflash_capture, layer_backend, ggml_backend_dev_backend_reg(layer_dev));
            total_size += ggml_backend_buffer_get_size(buf);
        }

        dflash_capture->hidden_gpu.push_back(std::move(hidden));
    }

    LLAMA_LOG_INFO("%s: allocated GPU hidden buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_layers, max_tokens);
}


void llama_context::allocate_tape_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture) {
        return;
    }

    dflash_log_backend_layout(__func__, "target", backends);
    if (n_slots < 1) {
        n_slots = 1;
    }

    // Keep layer_hiddens outer dim in sync with the slot count regardless of
    // whether GPU tape gets allocated. Hidden-state capture is needed by every
    // DFlash-enabled context (target side); tape allocation only fires for
    // models with DeltaNet-style recurrent layers (drafter side).
    if (!layer_hiddens.empty() && (int) layer_hiddens.size() != n_slots) {
        const size_t n_capture_layers = layer_hiddens.front().size();
        layer_hiddens.resize(n_slots);
        for (auto & slot_bufs : layer_hiddens) {
            if (slot_bufs.size() != n_capture_layers) {
                slot_bufs.resize(n_capture_layers);
            }
        }
    }

    // populate recurrent-layer metadata if the caller beat set_tape_recording() to it
    dflash_ensure_recurrent_setup();

    if (!dflash_capture->gpu_capture_enabled) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->tapes.clear();
        return;
    }

    dflash_capture->capture_wait_backends.clear();
    dflash_capture->fn_sync_backend_to_stream = nullptr;
    dflash_capture->sync_backend_to_stream_backend = nullptr;

    if (model.n_devices() > 1 && !llama_dflash_allow_multi_gpu_tape()) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->tapes.clear();
        if (!dflash_capture->multi_gpu_capture_fallback_logged) {
            LLAMA_LOG_INFO("%s: multi-GPU target detected (%zu devices); using eval-callback DFlash capture/tape fallback by multi-GPU tape kill switch\n",
                __func__, model.n_devices());
            dflash_capture->multi_gpu_capture_fallback_logged = true;
        }
        return;
    }
    if (model.n_devices() > 1 && !dflash_capture->multi_gpu_capture_fallback_logged) {
        LLAMA_LOG_INFO("%s: multi-GPU target detected (%zu devices); enabling device-aware DFlash GPU capture/tape\n",
            __func__, model.n_devices());
        dflash_capture->multi_gpu_capture_fallback_logged = true;
    }

    allocate_hidden_gpu(n_slots, max_tokens);

    if (dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }
    if (!llama_dflash_gpu_tape_supported_arch(model.arch)) {
        dflash_capture->tapes.clear();
        LLAMA_LOG_INFO("%s: GPU tape disabled for model arch %s; using eval-callback tape fallback\n",
            __func__, llm_arch_name(model.arch));
        return;
    }

    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const int n_rec = (int) rec_ids.size();

    // DeltaNet dimensions
    // k shape at capture: [ssm_d_state, H_k, n_tokens] where H_k depends on fused GDN
    // v/gate/beta shape: [S, H_v, n_tokens] or [1, H_v, n_tokens]
    const int64_t S = hparams.ssm_d_state;     // 256 for Qwen3.5-27B
    const int64_t H_v = hparams.ssm_dt_rank;   // 8 (num_v_heads)
    // when fused GDN is active, k is NOT repeated (kernel handles GQA internally)
    const int64_t H_k = (cparams.fused_gdn_ar && cparams.fused_gdn_ch)
                       ? (int64_t) hparams.ssm_n_group   // 1 (not repeated)
                       : H_v;                             // 8 (repeated)
    if (!llama_dflash_replay_gdn_supported_s_for_test(S) ||
            !llama_dflash_replay_state_shape_valid_for_test(S, H_v, hparams.n_embd_s())) {
        dflash_capture->tapes.clear();
        LLAMA_LOG_INFO("%s: GPU tape disabled for DFlash recurrent replay shape "
                "S=%lld H_v=%lld n_embd_s=%u; using eval-callback tape fallback\n",
                __func__, (long long) S, (long long) H_v, hparams.n_embd_s());
        return;
    }

    dflash_capture->tapes.clear();
    dflash_capture->tapes.reserve(n_slots);

    size_t total_size = 0;
    struct dev_layer_count {
        ggml_backend_dev_t dev = nullptr;
        int count = 0;
    };
    std::vector<dev_layer_count> dev_counts;

    for (int slot = 0; slot < n_slots; ++slot) {
        auto tape = std::make_unique<dflash_tape_gpu>();
        tape->layers.resize(n_rec);
        tape->layer_ids = dflash_capture->recurrent_layer_ids;
        tape->max_tokens = max_tokens;

        for (int li = 0; li < n_rec; ++li) {
            const int il = rec_ids[li];
            ggml_backend_dev_t layer_dev = model.dev_layer(il);
            ggml_backend_t layer_backend = dflash_backend_for_dev(backends, layer_dev);
            if (!layer_backend) {
                LLAMA_LOG_WARN("%s: no GPU backend for recurrent layer %d device %s; falling back to CPU tape\n",
                    __func__, il, layer_dev ? ggml_backend_dev_name(layer_dev) : "<null>");
                dflash_capture->tapes.clear();
                return;
            }

            const size_t ctx_mem = ggml_tensor_overhead() * 7;
            struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
            struct ggml_context * tape_ctx = ggml_init(ctx_params);
            if (!tape_ctx) {
                LLAMA_LOG_WARN("%s: failed to create GPU tape context for slot %d layer %d, falling back to CPU tape\n",
                    __func__, slot, il);
                dflash_capture->tapes.clear();
                return;
            }

            const auto * conv_kernel = model.layers[il].ssm_conv1d;
            GGML_ASSERT(conv_kernel != nullptr);
            const int64_t conv_window = conv_kernel->ne[0] - 1;
            GGML_ASSERT(conv_window > 0 && hparams.n_embd_r() % conv_window == 0);
            const int64_t conv_ch = hparams.n_embd_r() / conv_window;

            auto & tl = tape->layers[li];
            tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, (int64_t)max_tokens);
            tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, (int64_t)max_tokens);
            tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            tl.qkv  = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, (int64_t)max_tokens);

            tl.ctx = tape_ctx;
            tl.dev = layer_dev;
            tl.buf = ggml_backend_alloc_ctx_tensors(tape_ctx, layer_backend);

            if (!tl.buf) {
                LLAMA_LOG_WARN("%s: failed to allocate GPU tape buffer for slot %d layer %d device %s, falling back to CPU tape\n",
                    __func__, slot, il, layer_dev ? ggml_backend_dev_name(layer_dev) : "<null>");
                dflash_capture->tapes.clear();
                return;
            }

            dflash_capture_add_wait_backend(*dflash_capture, layer_backend, ggml_backend_dev_backend_reg(layer_dev));
            total_size += ggml_backend_buffer_get_size(tl.buf);

            bool found = false;
            for (auto & dc : dev_counts) {
                if (dc.dev == layer_dev) {
                    ++dc.count;
                    found = true;
                    break;
                }
            }
            if (!found) {
                dev_counts.push_back({ layer_dev, 1 });
            }
        }

        dflash_capture->tapes.push_back(std::move(tape));
    }

    dflash_capture->active_tape_idx = 0;

    LLAMA_LOG_INFO("%s: allocated device-aware GPU tape buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_rec, max_tokens);
    for (const auto & dc : dev_counts) {
        LLAMA_LOG_INFO("%s: dflash tape placement: device=%s layers=%d\n",
            __func__, dc.dev ? ggml_backend_dev_name(dc.dev) : "<null>", dc.count);
    }
}


void llama_context::allocate_tree_buffers(int max_tree_tokens) {
    if (tree_bufs.disabled) {
        return;
    }
    if (tree_bufs.max_tree_tokens >= max_tree_tokens) {
        return; // already allocated enough
    }

    // Tree verify buffers live on GPU 0. When the model is split across multiple
    // GPUs, recurrent layers on other devices can't read parent_ids from GPU 0,
    // so the scheduler aborts. Disable tree mode and use the regular SSM_CONV +
    // GATED_DELTA_NET kernels instead. The verify batch is still processed in a
    // single llama_decode call — only the recurrent kernel changes, and for
    // linear chains the sequential kernel produces identical results.
    if (model.n_devices() > 1) {
        LLAMA_LOG_INFO(
            "%s: multi-GPU layer-split detected (%zu devices): disabling tree verify because parent_ids are not replicated per device; using flat chain verification\n",
            __func__, model.n_devices());
        tree_bufs.disabled = true;
        return;
    }

    if (getenv("GGML_NO_TREE_VERIFY")) {
        LLAMA_LOG_INFO("%s: GGML_NO_TREE_VERIFY set — disabling tree verify, using flat chain\n", __func__);
        tree_bufs.disabled = true;
        return;
    }

    // Free existing
    if (tree_bufs.buffer) {
        ggml_backend_buffer_free(tree_bufs.buffer);
        tree_bufs.buffer = nullptr;
    }
    if (tree_bufs.ggml_ctx) {
        ggml_free(tree_bufs.ggml_ctx);
        tree_bufs.ggml_ctx = nullptr;
    }

    tree_bufs.max_tree_tokens = max_tree_tokens;
    tree_bufs.ssm_intermediates.clear();

    const auto & hparams = model.hparams;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim = (num_v_heads > 0) ? d_inner / num_v_heads : 0;

    if (head_v_dim == 0 || num_v_heads == 0) {
        return; // not a hybrid model
    }

    // Count recurrent layers
    int n_recurrent = 0;
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        if (hparams.is_recurrent(i)) {
            n_recurrent++;
        }
    }
    if (n_recurrent == 0) return;

    // Calculate total buffer size
    // Per layer: [head_v_dim, head_v_dim, num_v_heads, max_tree_tokens] in f16
    const int64_t inter_elems_per_layer = head_v_dim * head_v_dim * num_v_heads * max_tree_tokens;
    const size_t inter_bytes_per_layer = inter_elems_per_layer * sizeof(ggml_fp16_t);
    const size_t parent_ids_bytes = max_tree_tokens * sizeof(int32_t);
    const size_t total_bytes = n_recurrent * inter_bytes_per_layer + parent_ids_bytes;

    // Create ggml context for tensor metadata
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_recurrent + 1) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    tree_bufs.ggml_ctx = ggml_init(params);

    // Create tensors
    tree_bufs.parent_ids_gpu = ggml_new_tensor_1d(tree_bufs.ggml_ctx, GGML_TYPE_I32, max_tree_tokens);
    ggml_set_name(tree_bufs.parent_ids_gpu, "tree_parent_ids");

    tree_bufs.ssm_intermediates.resize(n_recurrent);
    for (int i = 0; i < n_recurrent; i++) {
        // Flat 1D tensor for simplicity, reshape in graph building
        tree_bufs.ssm_intermediates[i] = ggml_new_tensor_1d(tree_bufs.ggml_ctx, GGML_TYPE_F16, inter_elems_per_layer);
        char name[64];
        snprintf(name, sizeof(name), "tree_ssm_inter_%d", i);
        ggml_set_name(tree_bufs.ssm_intermediates[i], name);
    }

    // Allocate GPU buffer
    auto * buft = ggml_backend_get_default_buffer_type(ggml_backend_sched_get_backend(sched.get(), 0));
    tree_bufs.buffer = ggml_backend_alloc_ctx_tensors_from_buft(tree_bufs.ggml_ctx, buft);

    if (!tree_bufs.buffer) {
        LLAMA_LOG_WARN("%s: failed to allocate tree verify buffers (%.1f MB) — using flat chain verify\n", __func__,
                        total_bytes / (1024.0 * 1024.0));
        tree_bufs.max_tree_tokens = 0;
        tree_bufs.disabled = true;
        ggml_free(tree_bufs.ggml_ctx);
        tree_bufs.ggml_ctx = nullptr;
        return;
    }

    LLAMA_LOG_INFO("%s: allocated tree verify buffers: %d layers × %d tokens = %.1f MB\n", __func__,
                   n_recurrent, max_tree_tokens, total_bytes / (1024.0 * 1024.0));

    tree_bufs.parent_ids_cpu.resize(max_tree_tokens);
}


void llama_context::clear_tree_mask() {
    tree_mask.active = false;
    // Preserve n_tree_tokens and visibility data for size comparison next cycle.
    // The graph builder checks tree_mask.active before using mask data.
}


void llama_context::clear_tree_parent_ids() {
    tree_bufs.active = false;
    tree_bufs.n_tokens = 0;
}


void llama_context::dflash_dump_recurrent_state_dbg(llama_seq_id seq_id, const char * tag) {
    auto * cap = dflash_capture.get();
    if (!cap || cap->recurrent_layer_ids.empty()) { fprintf(stderr, "[dflash-rs-dump] %s seq=%d no recurrent layers\n", tag ? tag : "?", (int)seq_id); return; }
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) return;
    auto * mem_recr = mem_hybrid->get_mem_recr();
    if (!mem_recr || (uint32_t)seq_id >= mem_recr->size) return;
    int32_t cell_idx = mem_recr->cells[seq_id].tail;
    const uint32_t n_embd_s = model.hparams.n_embd_s();
    int il0 = (int) cap->recurrent_layer_ids[0];
    ggml_tensor * s_tensor = (il0 >= 0 && il0 < (int)mem_recr->s_l.size()) ? mem_recr->s_l[il0] : nullptr;
    if (!s_tensor || cell_idx < 0 || n_embd_s == 0) {
        fprintf(stderr, "[dflash-rs-dump] %s seq=%d cell=%d il0=%d n_embd_s=%u (no state)\n", tag ? tag : "?", (int)seq_id, cell_idx, il0, n_embd_s);
        return;
    }
    const size_t off = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
    std::vector<float> st(n_embd_s);
    ggml_backend_tensor_get(s_tensor, st.data(), off, n_embd_s * sizeof(float));
    double sum = 0, ab = 0; for (size_t i=0;i<n_embd_s;++i){ sum += st[i]; ab += std::fabs(st[i]); }
    // also dump last recurrent layer s_l and first recurrent layer r_l (conv state)
    int il_last = (int) cap->recurrent_layer_ids.back();
    ggml_tensor * s_last = (il_last >= 0 && il_last < (int)mem_recr->s_l.size()) ? mem_recr->s_l[il_last] : nullptr;
    double sumL=0, abL=0, sL0=0;
    if (s_last) {
        std::vector<float> sl(n_embd_s);
        ggml_backend_tensor_get(s_last, sl.data(), off, n_embd_s * sizeof(float));
        for (size_t i=0;i<n_embd_s;++i){ sumL += sl[i]; abL += std::fabs(sl[i]); }
        sL0 = sl[0];
    }
    ggml_tensor * r_tensor = (il0 >= 0 && il0 < (int)mem_recr->r_l.size()) ? mem_recr->r_l[il0] : nullptr;
    const uint32_t n_embd_r = model.hparams.n_embd_r();
    double rsum=0, rab=0, r0=0; size_t rn=0;
    if (r_tensor && n_embd_r > 0) {
        rn = n_embd_r;
        const size_t roff = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
        const size_t rbytes = (size_t)n_embd_r * ggml_element_size(r_tensor);
        if (dflash_tensor_span_in_bounds(r_tensor, roff, rbytes)) {
            std::vector<float> rv(n_embd_r);
            ggml_backend_tensor_get(r_tensor, rv.data(), roff, n_embd_r * sizeof(float));
            for (size_t i=0;i<rn;++i){ rsum += rv[i]; rab += std::fabs(rv[i]); }
            r0 = rv[0];
        } else {
            rn = 0;
        }
    }
    fprintf(stderr, "[dflash-rs-dump] %s seq=%d cell=%d il0=%d ilL=%d n_embd_s=%u pos=%d "
            "s0=%.6g sum=%.4g abs=%.4g | sL0=%.6g sLsum=%.4g sLabs=%.4g | r0=%.6g rsum=%.4g rabs=%.4g rn=%zu\n",
            tag ? tag : "?", (int)seq_id, cell_idx, il0, il_last, n_embd_s, (int)mem_recr->cells[seq_id].pos,
            st[0], sum, ab, sL0, sumL, abL, r0, rsum, rab, rn);
}


// idempotent: populates recurrent-layer ids + tape name map the first time it's called.
// Both set_tape_recording(true) and allocate_tape_gpu() fall through here so the setup
// order between them is flexible.
void llama_context::dflash_ensure_recurrent_setup() {
    if (!dflash_capture || !dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }
    const auto & hparams = model.hparams;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.is_recurrent(il)) {
            int idx = (int) dflash_capture->recurrent_layer_ids.size();
            dflash_capture->recurrent_layer_ids.push_back(il);

            std::string il_str = std::to_string(il);
            dflash_capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
            dflash_capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
            dflash_capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
            dflash_capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
            // qwen3next (Qwen3-Coder-Next) names the beta tensor "b" instead of "beta"
            dflash_capture->tape_name_map["b-" + il_str]                       = {idx, DFLASH_TAPE_BETA};
            dflash_capture->tape_name_map["qkv_mixed_pretranspose-" + il_str] = {idx, DFLASH_TAPE_QKV};
            // qwen3next (Qwen3-Coder-Next) builds the pre-conv (pre-transpose) QKV via two
            // code paths in build_qkvz(), both projecting the layer INPUT (pre-conv1d):
            //   wqkv path:        "linear_attn_qkv_mixed-" (qwen3next.cpp:304) [used by Qwen3-Coder-Next]
            //   ssm_in legacy:    "qkv_mixed-"             (qwen3next.cpp:364) [concat of q/k/v flats]
            // Both are the input to build_conv_state() (qwen3next.cpp:440), i.e. PRE-conv.
            // Without recording one of these, tape_replay_conv skips every layer (empty
            // qkv_mixed) and the conv state (r_l) is never advanced, so the conv state stays
            // frozen at the backup value and the target output is garbled (0% acceptance).
            // Verified: 504/504 conv layers OK after this fix; conv state matches re-decode ref.
            dflash_capture->tape_name_map["linear_attn_qkv_mixed-" + il_str] = {idx, DFLASH_TAPE_QKV};
            dflash_capture->tape_name_map["qkv_mixed-" + il_str]             = {idx, DFLASH_TAPE_QKV};
        }
    }
    dflash_capture->tape_layers.resize(dflash_capture->recurrent_layer_ids.size());
}


void llama_context::dflash_kv_cache_reset() {
    dflash_kv_cache_data * active_cache = dflash_kv_cache_active();
    if (!active_cache) {
        return;
    }
    active_cache->write_pos = 0;
    active_cache->n_filled = 0;
    active_cache->view.n_filled = 0;
    active_cache->view.write_pos = 0;
    cross.dflash_kv_cache = nullptr;
}



dflash_kv_cache_data * llama_context::dflash_kv_cache_active() {
    auto it = dflash_kv_caches.find(dflash_kv_cache_active_seq);
    return it == dflash_kv_caches.end() ? nullptr : it->second.get();
}

std::unique_ptr<dflash_kv_cache_data> & llama_context::dflash_kv_cache_active_ref() {
    return dflash_kv_caches[dflash_kv_cache_active_seq];
}

void llama_context::dflash_kv_cache_set_active_seq(llama_seq_id seq_id) {
    dflash_kv_cache_active_seq = seq_id;
    cross.dflash_kv_cache = nullptr;
}


void llama_context::dflash_mark_capture_invalid(const char * reason) {
    dflash_capture_valid_last_decode = false;
    dflash_capture_invalid_reason = reason ? reason : "unknown";
}


void llama_context::dflash_mark_capture_valid() {
    dflash_capture_valid_last_decode = true;
    dflash_capture_invalid_reason.clear();
}


void llama_context::dflash_prefill_capture_begin(llama_seq_id seq_id, int32_t capture_begin, int32_t capture_end) {
    if (!dflash_capture) {
        return;
    }

    if (seq_id < 0 || seq_id >= (llama_seq_id) LLAMA_DFLASH_MAX_SLOTS) {
        LLAMA_LOG_WARN("%s: slot %d out of range [0, %d); ignoring prefill capture plan\n",
            __func__, (int) seq_id, (int) LLAMA_DFLASH_MAX_SLOTS);
        return;
    }

    if ((int) dflash_capture->prefill_plans.size() <= seq_id) {
        dflash_capture->prefill_plans.resize((size_t) seq_id + 1);
    }

    auto & plan = dflash_capture->prefill_plans[(size_t) seq_id];

    if (capture_end <= capture_begin) {
        plan = {};
        return;
    }

    plan.active = true;
    plan.seq_id = seq_id;
    plan.capture_begin = capture_begin;
    plan.capture_end = capture_end;
    plan.n_tokens = capture_end - capture_begin;
    plan.n_written = 0;

    for (auto & pf : dflash_capture->prefill_gpu) {
        if (pf) {
            pf->n_tokens = 0;
        }
    }

    if (dflash_profile_enabled(DFLASH_PROFILE_PREFILL)) {
        LLAMA_LOG_INFO("%s: dflash prefill plan: slot=%d capture_begin=%d capture_end=%d n_tokens=%d\n",
            __func__, (int) seq_id, (int) capture_begin, (int) capture_end, (int) plan.n_tokens);
    }
}


void llama_context::dflash_prefill_capture_end() {
    if (!dflash_capture) {
        return;
    }
    for (auto & plan : dflash_capture->prefill_plans) {
        plan.active = false;
    }
}


void llama_context::dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_prepare_branch requires hybrid memory\n", __func__);
        return;
    }

    auto * mem_recr = mem_hybrid->get_mem_recr();

    // restore recurrent state from backup (keep backup intact for subsequent branches)
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp_recurrent_no_sync(seq_backup, seq_id, -1, -1);

    // tape replay to get DeltaNet state after processing 'depth' tokens (root + main_path[1..depth-1])
    tape_replay(seq_id, depth);
}


void llama_context::dflash_reset_hidden_capture() {
    if (!dflash_capture) {
        return;
    }
    // reset every slot because a single decode() may hold ubatches for multiple slots
    for (auto & slot_bufs : layer_hiddens) {
        for (auto & buf : slot_bufs) {
            buf.n_tokens = 0;
            std::vector<float>().swap(buf.data);
            std::vector<int32_t>().swap(buf.token_ids);
        }
    }
    for (auto & hidden : dflash_capture->hidden_gpu) {
        if (hidden) {
            hidden->n_tokens = 0;
        }
    }
    for (auto & pf : dflash_capture->prefill_gpu) {
        if (pf) {
            pf->n_tokens = 0;
        }
    }
    for (auto & plan : dflash_capture->prefill_plans) {
        plan.n_written = 0;
    }
    for (auto & tl : dflash_capture->tape_layers) {
        tl.n_tokens = 0;
    }
    std::vector<float>().swap(dflash_capture->scatter_buf);
    if (dflash_capture->profile) {
        dflash_profile_reset(*dflash_capture);
    }
    // The decode loop sets ubatch per iteration; null it here so a callback
    // that fires outside the loop can't read a stale pointer.
    dflash_capture->ubatch = nullptr;
}


void * llama_context::init_cross_ring_gpu(int n_layers, int n_embd, int ring_size) {
    ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
    if (!cuda_reg) {
        if (dflash_multi_gpu_debug_enabled() || dflash_diagnostic_debug_enabled()) {
            LLAMA_LOG_WARN("%s: dflash GPU ring unavailable: CUDA/ROCm/Vulkan backend registry not found\n", __func__);
        }
        return nullptr;
    }

    if (dflash_multi_gpu_debug_enabled() || dflash_diagnostic_debug_enabled()) {
        LLAMA_LOG_INFO("%s: dflash GPU ring using backend registry %s\n",
            __func__, ggml_backend_reg_name(cuda_reg));
    }

    // resolve all function pointers
    using alloc_fn_t      = void * (*)(int, int, int);
    using alloc_device_fn_t = void * (*)(int, int, int, int);
    using free_fn_t       = void   (*)(void *);
    using write_fn_t      = void   (*)(void *, int, int, const float *, int, int);
    using write_d2d_fn_t  = bool   (*)(void *, int, int, const void *, int, int);
    using sync_fn_t       = void   (*)(void *);
    using snapshot_fn_t   = bool   (*)(void *, int, int, int, float *, int, int, int);
    using interleave_fn_t = const float * (*)(void *, int, int, int);
    using set_tensor_fn_t = void   (*)(void *, const void *, size_t, size_t);
    using set_tensor_tensor_fn_t = void (*)(ggml_tensor *, const void *, size_t, size_t);
    using write_d2d_tensor_fn_t  = bool (*)(void *, int, int, ggml_tensor *, int, int, int);

    auto fn_alloc_device = (alloc_device_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_alloc_device");
    auto fn_alloc      = (alloc_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_alloc");
    auto fn_free       = (free_fn_t)       ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_free");
    auto fn_write      = (write_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write");
    auto fn_write_d2d  = (write_d2d_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write_d2d");
    auto fn_sync       = (sync_fn_t)       ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_synchronize");
    auto fn_snapshot   = (snapshot_fn_t)   ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_snapshot");
    auto fn_interleave = (interleave_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_interleave");
    auto fn_set_tensor = (set_tensor_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_set_tensor");
    // Tensor variants (Vulkan-only). Null on CUDA, which uses the raw variants above.
    auto fn_set_tensor_tensor = (set_tensor_tensor_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_set_tensor_tensor");
    auto fn_write_d2d_tensor  = (write_d2d_tensor_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write_d2d_tensor");

    // Need the 7 shared fns, plus at least one of each {set_tensor, set_tensor_tensor} / {write_d2d, write_d2d_tensor} pair.
    if (!fn_alloc || !fn_free || !fn_write || !fn_sync || !fn_snapshot || !fn_interleave) {
        return nullptr;
    }
    if (!fn_write_d2d && !fn_write_d2d_tensor) return nullptr;
    if (!fn_set_tensor && !fn_set_tensor_tensor) return nullptr;

    const int ring_device = dflash_gpu_ring_device_override();

    void * gpu_ring = nullptr;
    if (fn_alloc_device && ring_device >= 0) {
        gpu_ring = fn_alloc_device(ring_device, n_layers, n_embd, ring_size);
    } else {
        gpu_ring = fn_alloc(n_layers, n_embd, ring_size);
    }
    if (!gpu_ring) return nullptr;

    auto * handle = new dflash_cross_ring_handle();
    handle->gpu_ring      = gpu_ring;
    handle->fn_free       = fn_free;
    handle->fn_write      = fn_write;
    handle->fn_write_d2d  = fn_write_d2d;
    handle->fn_synchronize = fn_sync;
    handle->fn_snapshot   = fn_snapshot;
    handle->fn_interleave = fn_interleave;
    handle->fn_set_tensor = fn_set_tensor;
    handle->fn_set_tensor_tensor = fn_set_tensor_tensor;
    handle->fn_write_d2d_tensor  = fn_write_d2d_tensor;
    return handle;
}


void llama_context::set_active_dflash_slot(int slot_idx) {
    if (!dflash_capture) {
        return;
    }
    int n_slots = (int) layer_hiddens.size();
    n_slots = std::max(n_slots, (int) dflash_capture->tapes.size());
    n_slots = std::max(n_slots, (int) dflash_capture->hidden_gpu.size());
    n_slots = std::max(n_slots, (int) dflash_capture->prefill_gpu.size());
    if (slot_idx < 0 || slot_idx >= n_slots) {
        LLAMA_LOG_WARN("%s: slot %d out of range [0, %d); ignoring\n",
            __func__, slot_idx, n_slots);
        return;
    }
    if (slot_idx == dflash_capture->active_tape_idx) {
        return;
    }
    dflash_capture->active_tape_idx = slot_idx;
    cparams.tape_gpu = dflash_capture->active_tape();
    // sync per-seq array for single-seq external callers; CPU fallback leaves this null
    cparams.tape_gpu_seqs[0] = cparams.tape_gpu;
    cparams.tape_gpu_n_seqs = cparams.tape_gpu ? 1 : 0;
    cparams.hidden_gpu_seqs[0] = dflash_capture->active_hidden_gpu();
    cparams.hidden_gpu_n_seqs = cparams.hidden_gpu_seqs[0] ? 1 : 0;
    cparams.prefill_gpu_seqs[0] =
        (slot_idx >= 0 && slot_idx < (int) dflash_capture->prefill_gpu.size())
            ? dflash_capture->prefill_gpu[slot_idx].get()
            : nullptr;
    cparams.prefill_gpu_n_seqs = cparams.prefill_gpu_seqs[0] ? 1 : 0;
    // graph nodes hold references to the previous slot's tape tensors; invalidate
    // so the next decode rebuilds with the new slot's tensors.
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
}




void llama_context::set_cross_data_gpu(
        llama_seq_id seq_id, const void * d_staging, int cross_len,
        int n_layers, int n_embd_layer, set_tensor_d2d_fn_t fn_d2d,
        set_tensor_d2d_tensor_fn_t fn_d2d_tensor) {
    int64_t n_target_features = (int64_t)n_layers * n_embd_layer;

    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && cross_len > max_ctx) ? max_ctx : cross_len;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd     = n_target_features;
    cross.n_enc      = bucket;
    cross.n_enc_real = cross_len;
    cross.v_embd_gpu = d_staging;
    cross.v_embd_gpu_n_enc_real = cross_len;
    cross.fn_set_tensor_d2d = fn_d2d;
    cross.fn_set_tensor_d2d_tensor = fn_d2d_tensor;
    cross.dflash_kv_cache = nullptr;
    if (seq_id >= 0) {
        dflash_kv_cache_active_seq = seq_id;
    }

    const bool use_gpu_only = d_staging != nullptr && (fn_d2d != nullptr || fn_d2d_tensor != nullptr) && cparams.dflash_n_slots <= 1;
    if (use_gpu_only) {
        std::vector<float>().swap(cross.v_embd);
    } else {
        // ensure v_embd is non-empty so graph builders (llama-graph.cpp) use cross.n_enc
        // for sizing instead of falling back to hparams defaults
        if (cross.v_embd.size() != (size_t)(n_target_features * cross_len)) {
            cross.v_embd.resize(n_target_features * cross_len);
        }
    }

    if (seq_id >= 0) {
        auto & entry = cross.v_embd_per_seq[seq_id];
        entry.n_enc      = bucket;
        entry.n_enc_real = cross_len;
        entry.v_embd_gpu = d_staging;
        entry.v_embd_gpu_n_enc_real = cross_len;
        if (use_gpu_only) {
            std::vector<float>().swap(entry.v_embd);
        } else {
            if (entry.v_embd.size() != (size_t)(n_target_features * cross_len)) {
                entry.v_embd.resize(n_target_features * cross_len);
            }
        }
    }

    if (dflash_kv_cache_active() && dflash_kv_cache_prepare((int) cross.n_enc)) {
        cross.dflash_kv_cache = &dflash_kv_cache_active()->view;
    }
}


// Per-seq cross data stash for multi-slot DFlash
void llama_context::set_cross_data_seq(llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens) {
    if (seq_id < 0) {
        set_cross_data(data, n_embd, n_tokens);
        return;
    }

    // Also update the single-slot v_embd — sequential (non-batched) draft() calls
    // read from v_embd directly, and the graph's set_input single-slot path uses it.
    set_cross_data(data, n_embd, n_tokens);

    auto & entry = cross.v_embd_per_seq[seq_id];
    entry.n_enc      = cross.n_enc;
    entry.n_enc_real = n_tokens;
    entry.v_embd_gpu = nullptr;
    entry.v_embd_gpu_n_enc_real = 0;
    entry.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(entry.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}


void llama_context::set_dflash_capture_active(bool active) {
    if (!dflash_capture) {
        return;
    }

    if (dflash_capture->capture_active == active) {
        return;
    }

    dflash_capture->capture_active = active;

    if (active) {
        // Restore capture callback if layer configuration exists.
        // The decode loop will further refine cb_eval based on GPU
        // hidden/tape readiness (see the dflash_capture block in decode).
        if (!dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        }
        if (memory) {
            memory->set_force_split_seq(true);
        }
    } else {
        // Remove the eval callback so graph builds and compute skip
        // hidden outputs entirely.  Do NOT destroy GPU buffers, tape
        // metadata, layer configuration, or profile counters.
        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;
        cparams.hidden_gpu_n_seqs = 0;
        dflash_clear_prefill_cparams(cparams);
        cparams.tape_gpu_n_seqs = 0;
        cparams.tape_gpu = nullptr;
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.hidden_gpu_seqs[s] = nullptr;
            cparams.tape_gpu_seqs[s] = nullptr;
        }
        if (memory) {
            memory->set_force_split_seq(false);
        }
    }

    cparams.dflash_capture_layers.clear();
    if (active && !dflash_capture->layer_ids.empty()) {
        for (auto lid : dflash_capture->layer_ids) {
            cparams.dflash_capture_layers.push_back(lid);
        }
    }
}


void llama_context::set_dflash_capture(const int32_t * layer_ids, int32_t n_layers) {
    if (layer_ids == nullptr || n_layers <= 0) {
        // Permanent deconfiguration: clear layer config and remove callback.
        // For temporary per-view capture toggling, use set_dflash_capture_active()
        // instead, which preserves GPU buffers and layer configuration.
        cparams.dflash_capture_layers.clear();

        if (dflash_capture) {
            dflash_capture->layer_ids.clear();
            dflash_capture->hidden_name_idx.clear();
            dflash_capture->tensor_names.clear();
            dflash_capture->capture_active = false;
        }

        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;

        return;
    }

    // Store layer IDs for the graph builder.
    cparams.dflash_capture_layers.clear();
    for (int32_t i = 0; i < n_layers; ++i) {
        cparams.dflash_capture_layers.push_back(layer_ids[i]);
    }

    // Initialise or reconfigure the capture data without destroying GPU
    // buffers, tape metadata, or profile state that may already exist.
    if (!dflash_capture) {
        dflash_capture = std::make_unique<dflash_capture_data>();
        dflash_capture->hiddens = &layer_hiddens;
        dflash_capture->profile_flags = dflash_profile_flags();
        dflash_capture->profile = dflash_capture->profile_flags != 0;
    }

    dflash_capture->capture_active = true;
    dflash_capture->layer_ids.clear();
    dflash_capture->hidden_name_idx.clear();
    dflash_capture->tensor_names.clear();

    layer_hiddens.assign(1, std::vector<dflash_layer_hidden_buf>(n_layers));

    for (int32_t i = 0; i < n_layers; ++i) {
        dflash_capture->layer_ids.push_back(layer_ids[i]);
        std::string name = "l_out-" + std::to_string(layer_ids[i]);
        dflash_capture->hidden_name_idx[name] = i;
        dflash_capture->tensor_names.push_back(std::move(name));
    }

    // Install our eval callback (replaces any existing one).
    // The decode loop may override this to nullptr when GPU graph-embedded
    // capture is ready (see dflash_capture block in decode).
    cparams.cb_eval = dflash_eval_callback;
    cparams.cb_eval_user_data = dflash_capture.get();

    // GPU tape, eval callback hidden scatter, and QKV per-seq metadata
    // all support multi-seq ubatches. However, the server's
    // batch can mix prompt + TG tokens from different slots; split_equal
    // on such mixed batches produces incorrect ubatches. Expose the flag
    // so callers can toggle it off for verify-only decodes.
    if (memory) {
        memory->set_force_split_seq(true);
    }
}


void llama_context::set_dflash_consume_reduced(bool enabled) {
    cparams.dflash_reduced_consumer_active = enabled;
}


void llama_context::set_dflash_gpu_capture(bool enabled) {
    if (!dflash_capture) {
        return;
    }

    dflash_capture->gpu_capture_enabled = enabled;

    // Always clear the graph-embedded capture cparams when changing mode;
    // the decode loop will repopulate them if GPU capture is active and
    // buffers are valid.  Do NOT destroy hidden_gpu/tapes vectors — those
    // are persistent GPU allocations that survive logical toggles.
    cparams.hidden_gpu_n_seqs = 0;
    dflash_clear_prefill_cparams(cparams);
    cparams.tape_gpu_n_seqs = 0;
    cparams.tape_gpu = nullptr;
    dflash_capture->capture_wait_backends.clear();
    dflash_capture->fn_sync_backend_to_stream = nullptr;
    dflash_capture->sync_backend_to_stream_backend = nullptr;
    for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        cparams.tape_gpu_seqs[s] = nullptr;
        cparams.hidden_gpu_seqs[s] = nullptr;
    }

    if (!enabled) {
        // When GPU capture is disabled but logical capture is still active,
        // the decode loop needs the eval callback for CPU fallback.
        // When logical capture is inactive, the callback should stay null.
        if (dflash_capture->capture_active && !dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        }
        // If GPU buffers were previously allocated and dimensions now differ,
        // reallocate. Otherwise keep existing buffers.
    }
    // When enabled is true, keep buffers as-is. The decode loop's
    // allocate_tape_gpu path handles slot-count changes.
}


void llama_context::set_dflash_n_slots(int n) {
    const int clamped = std::max(1, std::min(n, (int) LLAMA_DFLASH_MAX_SLOTS));
    if (cparams.dflash_n_slots == clamped) {
        return;
    }
    cparams.dflash_n_slots = clamped;
    // drafter graph ctx_len depends on n_slots → force a fresh reserve on next decode
    sched_need_reserve = true;
    gf_res_prev->reset();
}


void llama_context::set_dflash_sample_temp(float temp) {
    cparams.dflash_sample_temp = temp;
}


void llama_context::set_dflash_target_kv_available(bool avail) {
    if (cparams.dflash_target_kv_available == avail) {
        return;
    }
    cparams.dflash_target_kv_available = avail;
    // drafter graph attention branch depends on this → force a fresh reserve
    sched_need_reserve = true;
    gf_res_prev->reset();
}


void llama_context::set_dflash_topk(int k) {
    cparams.dflash_topk = (k >= 1) ? k : 1;
    // invalidate graph cache since output tensor shape changes with K
    gf_res_prev->reset();
}


void llama_context::set_dflash_verify_logits(bool enabled, int top_k) {
    const int clamped_top_k = std::max(1, std::min(top_k, 64));
    if (cparams.dflash_verify_logits == enabled && cparams.dflash_verify_topk == clamped_top_k) {
        return;
    }

    cparams.dflash_verify_logits = enabled;
    cparams.dflash_verify_topk = clamped_top_k;
    // Graph reuse already keys on dflash_verify_logits/topk. Do not proactively
    // reset here: the server keeps this stable across verifier cycles so CUDA
    // graph replay can warm up and remain reusable.
}


void llama_context::set_tape_recording(bool enable) {
    if (!dflash_capture) {
        return;
    }

    dflash_capture->tape_enabled = enable;

    if (enable) {
        dflash_ensure_recurrent_setup();
        if (dflash_capture->tapes.empty()) {
            allocate_tape_gpu(1, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
        }
        for (auto & tape : dflash_capture->tapes) {
            if (tape) {
                tape->n_tokens = 0;
            }
        }
        for (auto & hidden : dflash_capture->hidden_gpu) {
            if (hidden) {
                hidden->n_tokens = 0;
            }
        }
    }

    // expose to graph builder via cparams — populate all tape pointers so graph
    // reservation accounts for worst-case per-seq copy ops.
    if (enable && !dflash_capture->tapes.empty()) {
        const int n_tapes = (int) dflash_capture->tapes.size();
        cparams.tape_gpu = dflash_capture->tapes[0].get();
        cparams.tape_gpu_n_seqs = n_tapes;
        for (int s = 0; s < n_tapes && s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = dflash_capture->tapes[s].get();
        }
        for (int s = n_tapes; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
        }
    } else {
        cparams.tape_gpu = nullptr;
        cparams.tape_gpu_n_seqs = 0;
        cparams.hidden_gpu_n_seqs = 0;
        dflash_clear_prefill_cparams(cparams);
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
            cparams.hidden_gpu_seqs[s] = nullptr;
        }

        if (dflash_capture->capture_active && !dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        } else {
            cparams.cb_eval = nullptr;
            cparams.cb_eval_user_data = nullptr;
        }
    }
}


void llama_context::set_tree_mask(const uint8_t * visibility, int n_tree_tokens) {
    // Only trigger graph reserve if the tree size exceeds what we've previously allocated.
    // Same or smaller trees reuse the existing allocation without a reserve.
    bool need_reserve = (size_t)(n_tree_tokens * n_tree_tokens) > tree_mask.visibility.size();
    tree_mask.active = true;
    tree_mask.n_tree_tokens = n_tree_tokens;
    int n2 = n_tree_tokens * n_tree_tokens;
    tree_mask.visibility.assign(visibility, visibility + n2);
    if (need_reserve) {
        sched_need_reserve = true;
    }
}


void llama_context::set_tree_parent_ids(const int32_t * parents, int n_tokens) {
    if (tree_bufs.disabled) {
        return; // multi-GPU: silently use flat chain verify
    }
    if (parents == nullptr || n_tokens <= 0) {
        LLAMA_LOG_WARN("%s: invalid tree parent buffer\n", __func__);
        tree_bufs.active = false;
        tree_bufs.n_tokens = 0;
        return;
    }
    if (parents[0] != -1) {
        LLAMA_LOG_WARN("%s: invalid tree root parent id (%d)\n", __func__, parents[0]);
        tree_bufs.active = false;
        tree_bufs.n_tokens = 0;
        return;
    }
    for (int i = 1; i < n_tokens; ++i) {
        if (parents[i] < 0 || parents[i] >= i) {
            LLAMA_LOG_WARN("%s: invalid tree parent id at %d: %d\n", __func__, i, parents[i]);
            tree_bufs.active = false;
            tree_bufs.n_tokens = 0;
            return;
        }
    }
    if (tree_bufs.max_tree_tokens < n_tokens) {
        // Allocate or reallocate — use exact size + small margin
        int alloc_size = n_tokens + 4;
        allocate_tree_buffers(alloc_size);
    }
    if (tree_bufs.disabled) {
        return; // allocate_tree_buffers detected multi-GPU
    }
    if (n_tokens > tree_bufs.max_tree_tokens) {
        LLAMA_LOG_WARN("%s: tree buffers too small (%d > %d), falling back to flat verify\n",
            __func__, n_tokens, tree_bufs.max_tree_tokens);
        tree_bufs.active = false;
        return;
    }

    // Copy to CPU buffer
    tree_bufs.parent_ids_cpu.assign(parents, parents + n_tokens);

    // Upload to GPU
    ggml_backend_tensor_set(tree_bufs.parent_ids_gpu, parents, 0, n_tokens * sizeof(int32_t));

    tree_bufs.n_tokens = n_tokens;
    tree_bufs.active = true;
}


void llama_context::tape_replay_conv(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;
    const uint32_t n_embd_r = hparams.n_embd_r();

    // Attempt GPU conv replay path unconditionally. It handles its own gating checks internally
    // and returns false to fallback if split-device tape, GPU backend, or pointers are not eligible.
    if (tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted)) {
        return;
    }
    if (model.n_devices() > 1 && tape_replay_conv_gpu_from_cpu_tape(mem_recurrent, cell_idx, n_accepted, seq_id)) {
        return;
    }

    // Batch conv rebuild: issue all async reads first, sync once, compute, then async write all.
    // This reduces GPU sync points from 2*N to 2 (N = number of recurrent layers with conv).
    struct conv_layer_data {
        size_t tape_li;
        ggml_tensor * r_tensor;
        size_t r_offset;
        int64_t conv_ch;
        int64_t conv_window;
        std::vector<float> old_window;
        std::vector<float> qkv_mixed;
        std::vector<float> new_conv;
    };
    std::vector<conv_layer_data> layers;
    layers.reserve(rec_ids.size());

    // Phase 1: issue all async reads from GPU
    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    const bool use_async_backend = gpu_backend && model.n_devices() <= 1;
    auto get_tensor_data = [&](const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_get_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_get(tensor, data, offset, size);
        }
    };
    auto set_tensor_data = [&](ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_set_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_set(tensor, data, offset, size);
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];
        dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
        dflash_tape_gpu_layer * gpu_layer = nullptr;
        if (gpu_tape && li < gpu_tape->layers.size() &&
                n_accepted <= gpu_tape->max_tokens &&
                n_accepted <= gpu_tape->n_tokens) {
            gpu_layer = &gpu_tape->layers[li];
        }

        if (!mem_recurrent->r_l[il]) continue;
        const bool use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv;
        // tape.qkv_mixed layout is [conv_ch, n_seq_tokens, n_seqs] (ggml row-major); the
        // rebuild below indexes qkv_mixed[(src_pos-conv_window)*conv_ch + ch] which is only
        // valid for n_seqs==1 (seq 0). The CPU tape path returns false for n_seqs_unq>1
        // upstream, so this is safe today. If multi-seq CPU tape is ever enabled, the seq
        // stride (n_seq_tokens*conv_ch) must be added to the index.
        if (!use_gpu_qkv) {
            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { if (dflash_diagnostic_debug_enabled()) fprintf(stderr, "[dflash-tr-conv] layer=%d SKIP n_tokens=%d n_accepted=%d\n", il, tape.n_tokens, n_accepted); continue; }
            if (tape.qkv_mixed.empty()) { if (dflash_diagnostic_debug_enabled()) fprintf(stderr, "[dflash-tr-conv] layer=%d SKIP qkv_mixed EMPTY\n", il); continue; }
        }
        if (dflash_diagnostic_debug_enabled()) fprintf(stderr, "[dflash-tr-conv] layer=%d OK use_gpu_qkv=%d qkv_mixed.size=%zu n_tokens=%d n_accepted=%d\n", il, (int)use_gpu_qkv, tape.qkv_mixed.size(), tape.n_tokens, n_accepted);

        size_t qkv_seq_offset = 0;
        if (!use_gpu_qkv && tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t) tape.n_tokens * (size_t) tape.conv_channels;
            }
            GGML_ASSERT(found && "tape_replay_conv: seq_id not found in tape");
        }

        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
        const int64_t conv_ch = use_gpu_qkv ? gpu_layer->qkv->ne[0] : tape.conv_channels;
        GGML_ASSERT(conv_ch > 0 && n_embd_r % conv_ch == 0);
        const int64_t conv_window = (int64_t)(n_embd_r / conv_ch);

        conv_layer_data & d = layers.emplace_back();
        d.tape_li = li;
        d.r_tensor = r_tensor;
        d.r_offset = r_offset;
        d.conv_ch = conv_ch;
        d.conv_window = conv_window;
        d.old_window.resize(n_embd_r);
        d.qkv_mixed.resize((size_t) n_accepted * (size_t) conv_ch);
        d.new_conv.resize(n_embd_r);

        get_tensor_data(r_tensor, d.old_window.data(), r_offset, n_embd_r * sizeof(float));
        if (use_gpu_qkv) {
            get_tensor_data(gpu_layer->qkv, d.qkv_mixed.data(), 0, d.qkv_mixed.size() * sizeof(float));
        } else {
            std::memcpy(d.qkv_mixed.data(),
                        tape.qkv_mixed.data() + qkv_seq_offset,
                        d.qkv_mixed.size() * sizeof(float));
        }
    }

    // Phase 2: single sync point for all reads
    if (use_async_backend && !layers.empty()) {
        const int64_t t_start_us = dflash_capture->profile ? ggml_time_us() : 0;
        ggml_backend_synchronize(gpu_backend);
        if (dflash_capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            dflash_capture->profile_conv_read_wait_us += elapsed;
            dflash_capture->profile_replay_conv_wait_us += elapsed;
        }
    }

    // Phase 3: compute all conv rebuilds on CPU
    const int64_t t_cpu_start_us = dflash_capture->profile ? ggml_time_us() : 0;
    for (auto & d : layers) {
        for (int64_t w = 0; w < d.conv_window; ++w) {
            int src_pos = n_accepted + (int)w;
            for (int64_t ch = 0; ch < d.conv_ch; ++ch) {
                float val;
                if (src_pos < (int)d.conv_window) {
                    val = d.old_window[ch * d.conv_window + src_pos];
                } else {
                    val = d.qkv_mixed[(src_pos - d.conv_window) * d.conv_ch + ch];
                }
                d.new_conv[ch * d.conv_window + w] = val;
            }
        }
    }
    if (dflash_capture->profile && !layers.empty()) {
        dflash_capture->profile_conv_cpu_us += ggml_time_us() - t_cpu_start_us;
    }

    // Phase 4: issue all async writes to GPU, then single sync
    for (auto & d : layers) {
        set_tensor_data(d.r_tensor, d.new_conv.data(), d.r_offset, n_embd_r * sizeof(float));
    }
    if (use_async_backend && !layers.empty()) {
        const int64_t t_start_us = dflash_capture->profile ? ggml_time_us() : 0;
        ggml_backend_synchronize(gpu_backend);
        if (dflash_capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            dflash_capture->profile_conv_write_wait_us += elapsed;
            dflash_capture->profile_replay_conv_wait_us += elapsed;
        }
    }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
}


void llama_context::tape_replay_sync() {
    if (!dflash_capture || !dflash_capture->replay_pending) {
        return;
    }

    // P2-11: use CUDA event for fine-grained sync instead of full stream synchronize
    auto * backend = dflash_capture->replay_gpu_backend;
    if (backend) {
        const int64_t t_start_us = dflash_capture->profile ? ggml_time_us() : 0;
        if (!dflash_capture->replay_event) {
            auto * dev = ggml_backend_get_device(backend);
            if (dev) {
                dflash_capture->replay_event = ggml_backend_event_new(dev);
            }
        }
        if (dflash_capture->replay_event) {
            ggml_backend_event_record(dflash_capture->replay_event, backend);
            ggml_backend_event_synchronize(dflash_capture->replay_event);
        } else {
            ggml_backend_synchronize(backend);
        }
        if (dflash_capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            dflash_capture->profile_replay_wait_us += elapsed;
            dflash_capture->profile_replay_gdn_wait_us += elapsed;
            dflash_capture->profile_replay_sync_calls += 1;
        }
    } else if (dflash_capture->replay_direct_gpu &&
            (!dflash_capture->replay_sync_ptrs.empty() || dflash_capture->replay_sync_ptr)) {
        const int64_t t_start_us = dflash_capture->profile ? ggml_time_us() : 0;
        ggml_backend_reg_t cuda_reg = dflash_gpu_backend_reg();
        using sync_ptr_fn_t = bool (*)(const void *);
        using sync_device_fn_t = bool (*)(int);
        auto fn_sync_ptr = cuda_reg
            ? (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr")
            : nullptr;
        auto fn_sync_device = cuda_reg
            ? (sync_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_device")
            : nullptr;
        bool synced = false;
        uint64_t sync_calls = 0;
        if (fn_sync_device && dflash_capture->replay_sync_device >= 0) {
            synced = fn_sync_device(dflash_capture->replay_sync_device);
            sync_calls = 1;
        } else if (fn_sync_ptr) {
            if (dflash_capture->replay_sync_ptr) {
                synced = fn_sync_ptr(dflash_capture->replay_sync_ptr);
                sync_calls = 1;
            } else if (!dflash_capture->replay_sync_ptrs.empty()) {
                synced = true;
                for (const void * ptr : dflash_capture->replay_sync_ptrs) {
                    synced = fn_sync_ptr(ptr) && synced;
                    sync_calls++;
                }
            } else {
                synced = false;
            }
        }
        if (!synced) {
            LLAMA_LOG_WARN("%s: direct GPU tape replay sync failed\n", __func__);
        }
        if (dflash_capture->profile) {
            const uint64_t elapsed = ggml_time_us() - t_start_us;
            dflash_capture->profile_replay_wait_us += elapsed;
            dflash_capture->profile_replay_gdn_wait_us += elapsed;
            dflash_capture->profile_replay_sync_calls += sync_calls;
        }
    }

    // free the graph context
    if (dflash_capture->replay_graph_ctx) {
        ggml_free(dflash_capture->replay_graph_ctx);
        dflash_capture->replay_graph_ctx = nullptr;
    }

    // finish conv rebuild + position advance. Direct GDN replay only updates s_l;
    // the conv fast path remains responsible for r_l and can safely fall back.
    tape_replay_conv(dflash_capture->replay_mem_recurrent,
                     dflash_capture->replay_cell_idx,
                     dflash_capture->replay_n_accepted,
                     dflash_capture->replay_seq_id);

    if (dflash_profile_has(dflash_capture->profile_flags, DFLASH_PROFILE_REPLAY)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: replay_path=direct-gpu:%" PRIu64 " replay_path=ggml-gpu:%" PRIu64
            " replay_path=cpu-fallback:%" PRIu64 " replay_layers=%" PRIu64 " replay_sync_calls=%" PRIu64
            " gdn_enqueue=%.3f ms gdn_wait=%.3f ms conv_enqueue=%.3f ms conv_wait=%.3f ms "
            "legacy_replay_wait=%.3f ms legacy_conv_gpu_enqueue=%.3f ms "
            "legacy_conv_read_wait=%.3f ms legacy_conv_write_wait=%.3f ms conv_cpu=%.3f ms\n",
            __func__,
            dflash_capture->profile_replay_direct_gpu,
            dflash_capture->profile_replay_ggml_gpu,
            dflash_capture->profile_replay_cpu_fallback,
            dflash_capture->profile_replay_layers,
            dflash_capture->profile_replay_sync_calls,
            dflash_capture->profile_replay_gdn_enqueue_us / 1000.0,
            dflash_capture->profile_replay_gdn_wait_us / 1000.0,
            dflash_capture->profile_replay_conv_enqueue_us / 1000.0,
            dflash_capture->profile_replay_conv_wait_us / 1000.0,
            dflash_capture->profile_replay_wait_us / 1000.0,
            dflash_capture->profile_conv_gpu_us / 1000.0,
            dflash_capture->profile_conv_read_wait_us / 1000.0,
            dflash_capture->profile_conv_write_wait_us / 1000.0,
            dflash_capture->profile_conv_cpu_us / 1000.0);
    }

    dflash_capture->replay_pending = false;
    dflash_capture->replay_direct_gpu = false;
    dflash_capture->replay_sync_ptr = nullptr;
    dflash_capture->replay_sync_ptrs.clear();
    dflash_capture->replay_sync_device = -1;
}


void llama_context::tree_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int commit_n, const int32_t * parents) {
    if (!tree_bufs.active || commit_n < 0) return;

    const auto & hparams = model.hparams;

    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(get_memory());
    llama_memory_recurrent * mem_recr = nullptr;
    if (mem_hybrid) {
        mem_recr = mem_hybrid->get_mem_recr();
    } else {
        mem_recr = dynamic_cast<llama_memory_recurrent *>(get_memory());
    }
    if (!mem_recr) return;

    int32_t cell_idx = -1;
    if (seq_id >= 0 && (uint32_t) seq_id < mem_recr->size) {
        cell_idx = mem_recr->cells[seq_id].tail;
    }
    if (cell_idx < 0) return;

    const uint32_t n_embd_s = hparams.n_embd_s();
    const uint32_t n_embd_r = hparams.n_embd_r();

    (void)parents; // unused for now (linear parents in flat mode)

    // Find GPU backend (used by both SSM restore and conv batch sync)
    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dflash_backend_dev_is_gpu(dev)) {
            gpu_backend = backend.get();
            break;
        }
    }
    const bool use_async_backend = gpu_backend && model.n_devices() <= 1;
    auto get_tensor_data = [&](const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_get_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_get(tensor, data, offset, size);
        }
    };
    auto set_tensor_data = [&](ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        if (use_async_backend) {
            ggml_backend_tensor_set_async(gpu_backend, tensor, data, offset, size);
        } else {
            ggml_backend_tensor_set(tensor, data, offset, size);
        }
    };

    // Count recurrent layers
    int n_rec = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.is_recurrent(il)) n_rec++;
    }

    // Restore SSM state from f16 intermediates via GPU graph
    if (n_rec > 0) {
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t)n_rec * 4 + 2) +
                         ggml_graph_overhead_custom(n_rec * 4, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 4, false);

        int recurrent_idx = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.is_recurrent(il)) continue;

            ggml_tensor * inter = tree_bufs.ssm_intermediates[recurrent_idx];
            size_t src_offset = (size_t)commit_n * n_embd_s * sizeof(ggml_fp16_t);

            // Source: f16 view into intermediate buffer at commit_n
            ggml_tensor * src_view = ggml_view_1d(ctx, inter, n_embd_s, src_offset);

            // Destination: f32 view into recurrent state
            ggml_tensor * s_tensor = mem_recr->s_l[il];
            size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
            ggml_tensor * dst_view = ggml_view_1d(ctx, s_tensor, n_embd_s, s_offset);

            // Copy f16 → f32 (ggml_cpy handles type conversion)
            ggml_tensor * cpy = ggml_cpy(ctx, src_view, dst_view);
            ggml_build_forward_expand(graph, cpy);

            recurrent_idx++;
        }

        // Initialize view buffers (required for direct backend compute)
        struct ggml_tensor * t = ggml_get_first_tensor(ctx);
        while (t) {
            if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(ctx, t);
        }

        if (gpu_backend) {
            ggml_backend_graph_compute(gpu_backend, graph);
        } else {
            ggml_backend_sched_graph_compute(sched.get(), graph);
        }
        ggml_free(ctx);
    }

    // Reconstruct conv state: restore backup conv first, then shift by n_accepted
    // (Same approach as tape_replay_conv, but batched to reduce GPU syncs)
    if (dflash_capture && !dflash_capture->tape_layers.empty()) {
        const auto & rec_ids = dflash_capture->recurrent_layer_ids;
        auto & tape_layers = dflash_capture->tape_layers;
        const int n_accepted = commit_n + 1;

        // GPU tape dual path: when GPU tape captured tree-verify data, use it instead
        // of stale CPU tape_layers (which were not updated because GPU tape bypasses
        // the eval callback).
        dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();

        // Find backup cell to restore conv state from
        int32_t backup_cell = -1;
        if (seq_backup >= 0 && (uint32_t) seq_backup < mem_recr->size) {
            backup_cell = mem_recr->cells[seq_backup].tail;
        }

        // Batch conv rebuild: async read all layers, sync once, compute, async write all
        struct tree_conv_data {
            size_t tape_li;
            ggml_tensor * r_tensor;
            size_t r_offset;
            size_t read_offset;
            int64_t conv_ch;
            int64_t conv_window;
            bool use_gpu_qkv;
            std::vector<float> old_window;
            std::vector<float> qkv_mixed;
            std::vector<float> new_conv;
        };
        std::vector<tree_conv_data> conv_layers;
        conv_layers.reserve(rec_ids.size());

        for (size_t li = 0; li < rec_ids.size(); ++li) {
            int il = rec_ids[li];
            auto & tape = tape_layers[li];

            dflash_tape_gpu_layer * gpu_layer = nullptr;
            if (gpu_tape && li < gpu_tape->layers.size() &&
                    n_accepted <= gpu_tape->max_tokens &&
                    n_accepted <= gpu_tape->n_tokens) {
                gpu_layer = &gpu_tape->layers[li];
            }

            if (!mem_recr->r_l[il]) continue;
            const bool use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv;
            if (!use_gpu_qkv) {
                if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;
                if (tape.qkv_mixed.empty()) continue;
            }

            ggml_tensor * r_tensor = mem_recr->r_l[il];
            const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
            const int64_t conv_ch = use_gpu_qkv ? gpu_layer->qkv->ne[0] : tape.conv_channels;
            const int64_t conv_window = (int64_t)(n_embd_r / conv_ch);

            tree_conv_data & d = conv_layers.emplace_back();
            d.tape_li = li;
            d.r_tensor = r_tensor;
            d.r_offset = r_offset;
            d.conv_ch = conv_ch;
            d.conv_window = conv_window;
            d.use_gpu_qkv = use_gpu_qkv;
            d.old_window.resize(n_embd_r);
            d.qkv_mixed.resize((size_t) n_accepted * (size_t) conv_ch);
            d.new_conv.resize(n_embd_r);

            // Read from backup cell if available, otherwise current position
            size_t read_offset = r_offset;
            if (backup_cell >= 0) {
                read_offset = (size_t)backup_cell * n_embd_r * ggml_element_size(r_tensor);
            }
            d.read_offset = read_offset;

            get_tensor_data(r_tensor, d.old_window.data(), read_offset, n_embd_r * sizeof(float));
            if (use_gpu_qkv) {
                get_tensor_data(gpu_layer->qkv, d.qkv_mixed.data(), 0, d.qkv_mixed.size() * sizeof(float));
            } else {
                std::memcpy(d.qkv_mixed.data(),
                            tape.qkv_mixed.data(),
                            d.qkv_mixed.size() * sizeof(float));
            }
        }

        // Single sync for all reads
        if (use_async_backend && !conv_layers.empty()) {
            ggml_backend_synchronize(gpu_backend);
        }

        // Compute all conv rebuilds on CPU
        for (auto & d : conv_layers) {
            for (int64_t w = 0; w < d.conv_window; ++w) {
                int src_pos = n_accepted + (int)w;
                for (int64_t ch = 0; ch < d.conv_ch; ++ch) {
                    float val;
                    if (src_pos < (int)d.conv_window) {
                        val = d.old_window[ch * d.conv_window + src_pos];
                    } else {
                        val = d.qkv_mixed[(src_pos - d.conv_window) * d.conv_ch + ch];
                    }
                    d.new_conv[ch * d.conv_window + w] = val;
                }
            }
        }

        // Async write all results back
        for (auto & d : conv_layers) {
            set_tensor_data(d.r_tensor, d.new_conv.data(), d.r_offset, n_embd_r * sizeof(float));
        }
        if (use_async_backend && !conv_layers.empty()) {
            ggml_backend_synchronize(gpu_backend);
        }
    }

    // Set cell.pos to the target position (absolute, set by caller via set_tree_seq0_count).
    // In tree mode, prepare() sets cell.pos to last ubatch position which is unpredictable
    // (branches may be last). So we use the absolute target: n_past_before + commit_n.
    const int target_pos = tree_bufs.n_seq0_tokens; // repurposed: caller passes absolute target pos
    if (target_pos >= 0) {
        mem_recr->cells[cell_idx].pos = target_pos;
    }

    clear_tree_parent_ids();
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
