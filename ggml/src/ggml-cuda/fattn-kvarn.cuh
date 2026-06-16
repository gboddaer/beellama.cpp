#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdlib>
#include <cstdint>

static constexpr int FATTN_KVARN_DIM = 128;
static constexpr int FATTN_KVARN_MAX_D = 512;
static constexpr int FATTN_KVARN_STAGE_GROUPS = 3;
static constexpr int FATTN_KVARN_TILE_VALUES = FATTN_KVARN_DIM * FATTN_KVARN_DIM;

bool ggml_cuda_kvarn_fused_require_enabled() {
    static const bool require = [] {
        const char * env = std::getenv("GGML_KVARN_FUSED_REQUIRE");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
    }();
    return require;
}

static __device__ __forceinline__ void fattn_kvarn_wht_128(float * values, int tid) {
    for (int stride = 1; stride < FATTN_KVARN_DIM; stride *= 2) {
        if (tid < 64) {
            const int j = (tid / stride) * (2 * stride) + (tid % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    values[tid] *= 0.08838834764831845f;
    __syncthreads();
}

static __global__ void fattn_kvarn_live_groups_kernel(
        const int64_t * indices,
        int n_indices,
        int stream_start,
        int n_stream,
        int groups_per_stream,
        int * live_groups) {
    const int out_stream = blockIdx.x;
    const int stream = stream_start + out_stream;
    int local = 0;
    for (int i = threadIdx.x; i < n_indices; i += FATTN_KVARN_DIM) {
        const int group_global = (int) (indices[i] / FATTN_KVARN_DIM);
        const int idx_stream = group_global / groups_per_stream;
        if (idx_stream == stream) {
            const int group = group_global - idx_stream * groups_per_stream;
            local = max(local, group);
        }
    }

    __shared__ int partial[FATTN_KVARN_DIM];
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = FATTN_KVARN_DIM / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = max(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        live_groups[out_stream] = partial[0];
    }

    GGML_UNUSED(n_stream);
}

static __device__ __forceinline__ float fattn_kvarn_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

static __device__ __forceinline__ float fattn_kvarn_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

template<int BITS>
static __device__ __forceinline__ uint8_t fattn_kvarn_unpack_chunk(uint64_t lo, uint64_t hi, int index) {
    static_assert(BITS == 2 || BITS == 3 || BITS == 4 || BITS == 5 || BITS == 6 || BITS == 8, "unsupported KVarN bits");
    constexpr uint64_t mask = (uint64_t(1) << BITS) - 1;
    const int bit = index * BITS;
    uint64_t value;
    if (bit < 64) {
        value = lo >> bit;
        if (bit > 64 - BITS) {
            value |= hi << (64 - bit);
        }
    } else {
        value = hi >> (bit - 64);
    }
    return (uint8_t) (value & mask);
}

template<int BITS, bool VALUE>
static __device__ __forceinline__ void fattn_kvarn_load_tile(
        half * tile,
        const uint8_t * records,
        const half * stage,
        int n_record_heads,
        int record_head,
        int stream,
        int groups_per_stream,
        int record_bytes,
        int live_group,
        int k0) {
    const int tid = threadIdx.x;
    const int group = k0 / FATTN_KVARN_DIM;
    const int stage_base = stream * FATTN_KVARN_DIM * FATTN_KVARN_STAGE_GROUPS;

    if (group == 0 || (group > 0 && group <= live_group && group + 1 >= live_group)) {
        const int stage_group = group == 0 ? 0 : 1 + ((group - 1) & 1);
        const int stage_pos_base = stage_base + stage_group * FATTN_KVARN_DIM;
        for (int index = tid; index < FATTN_KVARN_TILE_VALUES; index += FATTN_KVARN_DIM) {
            const int pos = index / FATTN_KVARN_DIM;
            const int dim = index - pos * FATTN_KVARN_DIM;
            tile[index] = stage[((int64_t) (stage_pos_base + pos) * n_record_heads + record_head) * FATTN_KVARN_DIM + dim];
        }
        __syncthreads();
        return;
    }

    if (group >= live_group || group >= groups_per_stream) {
        for (int index = tid; index < FATTN_KVARN_TILE_VALUES; index += FATTN_KVARN_DIM) {
            tile[index] = __float2half(0.0f);
        }
        __syncthreads();
        return;
    }

    const int record_group = stream * groups_per_stream + group;
    const uint8_t * record = records + ((int64_t) record_group * n_record_heads + record_head) * record_bytes;
    constexpr int payload_bytes = FATTN_KVARN_TILE_VALUES * BITS / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis = scale_axis + FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + FATTN_KVARN_DIM;

    const int row = tid;
    const float scale = __half2float(scale_axis[row]);
    const float zp = __half2float(zp_axis[row]);
    for (int col_base = 0; col_base < FATTN_KVARN_DIM; col_base += 16) {
        const int bit_offset = (row * FATTN_KVARN_DIM + col_base) * BITS;
        const uint8_t * packed_ptr = record + (bit_offset >> 3);
        constexpr int segment_bytes = 16 * BITS / 8;
        uint64_t packed_lo = 0;
        uint64_t packed_hi = 0;
#pragma unroll
        for (int b = 0; b < segment_bytes; ++b) {
            if (b < 8) {
                packed_lo |= uint64_t(packed_ptr[b]) << (8 * b);
            } else {
                packed_hi |= uint64_t(packed_ptr[b]) << (8 * (b - 8));
            }
        }

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int col = col_base + i;
            const uint8_t q = fattn_kvarn_unpack_chunk<BITS>(packed_lo, packed_hi, i);
            const float dequant = (q * scale + zp) * __half2float(other_axis[col]);
            const int pos = VALUE ? row : col;
            const int dim = VALUE ? col : row;
            tile[pos * FATTN_KVARN_DIM + dim] = __float2half(dequant);
        }
    }
    __syncthreads();
}

template<int D, int K_BITS, int V_BITS>
static __global__ void flash_attn_ext_kvarn_tiled(
        const char * Q_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        float * partial,
        const uint8_t * k_records,
        const half * k_stage,
        const uint8_t * v_records,
        const half * v_stage,
        const int * live_groups,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap,
        int32_t ne01,
        int32_t ne02,
        int32_t ne03,
        int32_t nb01,
        int32_t nb02,
        int32_t nb03,
        int32_t n_kv,
        int32_t n_head_kv,
        int32_t record_heads,
        int32_t stream_start,
        int32_t groups_per_stream,
        int32_t k_record_bytes,
        int32_t v_record_bytes,
        int32_t n_splits,
        int32_t ne31,
        int32_t ne33,
        int32_t nb31,
        int64_t nb33) {
    static_assert(D >= FATTN_KVARN_DIM && D <= FATTN_KVARN_MAX_D && D % FATTN_KVARN_DIM == 0, "unsupported KVarN head dimension");
    constexpr int SLICES = D / FATTN_KVARN_DIM;

    extern __shared__ half tile[];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ic0 = blockIdx.x;
    const int split = blockIdx.y;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / n_head_kv;
    const int kv_head = head / gqa_ratio;
    const int stream = stream_start + sequence;
    const int live_group = live_groups[sequence];

    const char * Q = Q_ptr + (int64_t) nb03 * sequence + (int64_t) nb02 * head + (int64_t) nb01 * ic0;
    const half * maskh = mask_ptr ? (const half *) (mask_ptr + (int64_t) nb33 * (sequence % ne33) + (int64_t) nb31 * ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    __shared__ float q_rot[D];
    __shared__ float scores[FATTN_KVARN_DIM];
    __shared__ float weights[FATTN_KVARN_DIM];
    __shared__ float warp_reduce[4];
    __shared__ float shared_scalar;
    __shared__ float out_shared[D];

    for (int slice = 0; slice < SLICES; ++slice) {
        q_rot[slice * FATTN_KVARN_DIM + tid] = ((const float *) Q)[slice * FATTN_KVARN_DIM + tid];
        __syncthreads();
        fattn_kvarn_wht_128(q_rot + slice * FATTN_KVARN_DIM, tid);
        q_rot[slice * FATTN_KVARN_DIM + tid] *= scale;
        out_shared[slice * FATTN_KVARN_DIM + tid] = 0.0f;
        __syncthreads();
    }

    float kq_max = -FLT_MAX / 2.0f;
    float kq_sum = 0.0f;

    const int kv_begin = ((int64_t) n_kv * split) / n_splits;
    const int kv_end   = ((int64_t) n_kv * (split + 1)) / n_splits;
    const int k0_begin = (kv_begin / FATTN_KVARN_DIM) * FATTN_KVARN_DIM;
    for (int k0 = k0_begin; k0 < kv_end; k0 += FATTN_KVARN_DIM) {
        scores[tid] = 0.0f;
        __syncthreads();

#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int record_head = kv_head * SLICES + slice;
            fattn_kvarn_load_tile<K_BITS, false>(
                    tile, k_records, k_stage, record_heads, record_head, stream,
                    groups_per_stream, k_record_bytes, live_group, k0);

#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int pos = warp * 32 + i;
                const int token = k0 + pos;
                float sum = 0.0f;
                if (token >= kv_begin && token < kv_end) {
#pragma unroll
                    for (int d = 0; d < FATTN_KVARN_DIM; d += 32) {
                        const int dim = d + lane;
                        sum += __half2float(tile[pos * FATTN_KVARN_DIM + dim]) * q_rot[slice * FATTN_KVARN_DIM + dim];
                    }
                }
                sum = fattn_kvarn_warp_sum(sum);
                if (lane == i) {
                    scores[pos] += sum;
                }
            }
            __syncthreads();
        }

        const int token = k0 + tid;
        float score = scores[tid];
        if (logit_softcap != 0.0f) {
            score = logit_softcap * tanhf(score);
        }
        if (maskh && token >= kv_begin && token < kv_end) {
            score += slope * __half2float(maskh[token]);
        }
        if (token < kv_begin || token >= kv_end) {
            score = -FLT_MAX / 2.0f;
        }
        scores[tid] = score;
        __syncthreads();

        float warp_max = fattn_kvarn_warp_max(score + FATTN_KQ_MAX_OFFSET);
        if (lane == 0) {
            warp_reduce[warp] = warp_max;
        }
        __syncthreads();

        float block_max = tid < 4 ? warp_reduce[tid] : -FLT_MAX / 2.0f;
        block_max = fattn_kvarn_warp_max(block_max);
        block_max = fmaxf(block_max, kq_max);
        if (tid == 0) {
            shared_scalar = block_max;
        }
        __syncthreads();

        const float kq_max_new = shared_scalar;
        const float kq_max_scale = __expf(kq_max - kq_max_new);
        kq_max = kq_max_new;
        kq_sum *= kq_max_scale;
        for (int d = tid; d < D; d += FATTN_KVARN_DIM) {
            out_shared[d] *= kq_max_scale;
        }

        const float w = __expf(scores[tid] - kq_max);
        weights[tid] = w;
        float sum_w = fattn_kvarn_warp_sum(w);
        if (lane == 0) {
            warp_reduce[warp] = sum_w;
        }
        __syncthreads();

        float block_sum = tid < 4 ? warp_reduce[tid] : 0.0f;
        block_sum = fattn_kvarn_warp_sum(block_sum);
        if (tid == 0) {
            shared_scalar = block_sum;
        }
        __syncthreads();
        kq_sum += shared_scalar;

#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int record_head = kv_head * SLICES + slice;
            fattn_kvarn_load_tile<V_BITS, true>(
                    tile, v_records, v_stage, record_heads, record_head, stream,
                    groups_per_stream, v_record_bytes, live_group, k0);

            float out = 0.0f;
#pragma unroll
            for (int pos = 0; pos < FATTN_KVARN_DIM; ++pos) {
                const int v_token = k0 + pos;
                if (v_token >= kv_begin && v_token < kv_end) {
                    out += weights[pos] * __half2float(tile[pos * FATTN_KVARN_DIM + tid]);
                }
            }
            out_shared[slice * FATTN_KVARN_DIM + tid] += out;
            __syncthreads();
        }
    }

    if (sinks_ptr && split == 0) {
        const float sink = ((const float *) sinks_ptr)[head];
        const float kq_max_new = fmaxf(sink, kq_max);
        const float kq_max_scale = __expf(kq_max - kq_max_new);
        const float sink_w = __expf(sink - kq_max_new);
        kq_max = kq_max_new;
        kq_sum = kq_sum * kq_max_scale + sink_w;
        for (int d = tid; d < D; d += FATTN_KVARN_DIM) {
            out_shared[d] *= kq_max_scale;
        }
        __syncthreads();
    }

    const int partial_stride = D + 2;
    float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
    if (tid == 0) {
        partial_block[0] = kq_max;
        partial_block[1] = kq_sum;
    }
    for (int d = tid; d < D; d += FATTN_KVARN_DIM) {
        partial_block[2 + d] = out_shared[d];
    }

    GGML_UNUSED(ne03);
}

template<int D>
static __global__ void flash_attn_ext_kvarn_combine(
        const float * partial,
        float * dst,
        int32_t ne01,
        int32_t ne02,
        int32_t n_splits) {
    const int tid = threadIdx.x;
    const int ic0 = blockIdx.x;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int partial_stride = D + 2;

    __shared__ float out_shared[D];
    __shared__ float split_scale[32];
    __shared__ float shared_sum;

    if (tid == 0) {
        float global_max = -FLT_MAX / 2.0f;
        for (int split = 0; split < n_splits; ++split) {
            const float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
            global_max = fmaxf(global_max, partial_block[0]);
        }

        float global_sum = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            const float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
            const float scale = __expf(partial_block[0] - global_max);
            split_scale[split] = scale;
            global_sum += partial_block[1] * scale;
        }
        shared_sum = global_sum;
    }
    __syncthreads();

    for (int d = tid; d < D; d += FATTN_KVARN_DIM) {
        float out = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            const float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
            out += partial_block[2 + d] * split_scale[split];
        }
        out_shared[d] = out / shared_sum;
    }
    __syncthreads();

#pragma unroll
    for (int slice = 0; slice < D / FATTN_KVARN_DIM; ++slice) {
        fattn_kvarn_wht_128(out_shared + slice * FATTN_KVARN_DIM, tid);
        dst[((sequence * ne01 + ic0) * ne02 + head) * D + slice * FATTN_KVARN_DIM + tid] =
            out_shared[slice * FATTN_KVARN_DIM + tid];
        __syncthreads();
    }
}

static const ggml_tensor * ggml_cuda_kvarn_unwrap_fattn_src(const ggml_tensor * t) {
    while (t != nullptr &&
            (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE)) {
        t = t->src[0];
    }
    return t;
}

static bool fattn_kvarn_bits_supported(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool ggml_cuda_flash_attn_ext_kvarn_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_KVARN_FUSED");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
    }();
    if (!enabled || dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (Q == nullptr || K == nullptr || V == nullptr || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int head_dim = (int) Q->ne[0];
    if (head_dim < FATTN_KVARN_DIM || head_dim > FATTN_KVARN_MAX_D || head_dim % FATTN_KVARN_DIM != 0 ||
            K->ne[0] != head_dim || V->ne[0] != head_dim) {
        return false;
    }
    if (Q->ne[1] != 1 || K->ne[1] <= 0 || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || V->ne[3] != K->ne[3]) {
        return false;
    }
    if (mask != nullptr && mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (sinks != nullptr && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] < Q->ne[2])) {
        return false;
    }

    const ggml_tensor * k_mat = ggml_cuda_kvarn_unwrap_fattn_src(K);
    const ggml_tensor * v_mat = ggml_cuda_kvarn_unwrap_fattn_src(V);
    if (k_mat == nullptr || v_mat == nullptr || k_mat->op != GGML_OP_KVARN_MATERIALIZE || v_mat->op != GGML_OP_KVARN_MATERIALIZE) {
        return false;
    }
    const int k_bits = ggml_get_op_params_i32(k_mat, 0);
    const int v_bits = ggml_get_op_params_i32(v_mat, 0);
    if (!fattn_kvarn_bits_supported(k_bits) || !fattn_kvarn_bits_supported(v_bits)) {
        return false;
    }
    if (ggml_get_op_params_i32(k_mat, 1) != 0 || ggml_get_op_params_i32(v_mat, 1) != 1) {
        return false;
    }
    if (ggml_get_op_params_i32(k_mat, 2) != ggml_get_op_params_i32(v_mat, 2) ||
            ggml_get_op_params_i32(k_mat, 3) != ggml_get_op_params_i32(v_mat, 3)) {
        return false;
    }

    const ggml_tensor * k_records = k_mat->src[0];
    const ggml_tensor * k_stage = k_mat->src[1];
    const ggml_tensor * k_indices = k_mat->src[2];
    const ggml_tensor * v_records = v_mat->src[0];
    const ggml_tensor * v_stage = v_mat->src[1];
    const ggml_tensor * v_indices = v_mat->src[2];
    if (k_records == nullptr || k_stage == nullptr || k_indices == nullptr ||
            v_records == nullptr || v_stage == nullptr || v_indices == nullptr ||
            k_indices->type != GGML_TYPE_I64 || v_indices->type != GGML_TYPE_I64 ||
            k_indices->ne[0] != v_indices->ne[0] ||
            k_indices->ne[1] != v_indices->ne[1] ||
            k_indices->ne[2] != v_indices->ne[2] ||
            k_indices->ne[3] != v_indices->ne[3]) {
        return false;
    }

    const int slices = head_dim / FATTN_KVARN_DIM;
    if (k_records->ne[1] != K->ne[2] * slices || v_records->ne[1] != V->ne[2] * slices) {
        return false;
    }
    if (k_stage->ne[0] != FATTN_KVARN_DIM || v_stage->ne[0] != FATTN_KVARN_DIM ||
            k_stage->ne[1] != k_records->ne[1] || v_stage->ne[1] != v_records->ne[1] ||
            k_stage->ne[2] % (FATTN_KVARN_DIM * FATTN_KVARN_STAGE_GROUPS) != 0 ||
            v_stage->ne[2] % (FATTN_KVARN_DIM * FATTN_KVARN_STAGE_GROUPS) != 0) {
        return false;
    }
    if (k_stage->ne[2] != v_stage->ne[2]) {
        return false;
    }
    return true;
}

template<int D, int K_BITS, int V_BITS>
static void ggml_cuda_flash_attn_ext_kvarn_launch(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        int n_splits,
        int groups_per_stream,
        int stream_start,
        const int * live_groups,
        float * partial,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * k_mat = ggml_cuda_kvarn_unwrap_fattn_src(K);
    const ggml_tensor * v_mat = ggml_cuda_kvarn_unwrap_fattn_src(V);

    const ggml_tensor * k_records = k_mat->src[0];
    const ggml_tensor * k_stage = k_mat->src[1];
    const ggml_tensor * v_records = v_mat->src[0];
    const ggml_tensor * v_stage = v_mat->src[1];

    cudaStream_t stream = ctx.stream();
    const size_t tile_shared = FATTN_KVARN_TILE_VALUES * sizeof(half);
#if defined(GGML_USE_HIP)
    CUDA_CHECK(hipFuncSetAttribute(
        reinterpret_cast<const void *>(&flash_attn_ext_kvarn_tiled<D, K_BITS, V_BITS>),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        tile_shared));
#elif !defined(GGML_USE_MUSA)
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attn_ext_kvarn_tiled<D, K_BITS, V_BITS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        tile_shared));
#endif

    const dim3 blocks((uint32_t) Q->ne[1], (uint32_t) n_splits, (uint32_t) (Q->ne[2] * Q->ne[3]));
    const dim3 block_dim(FATTN_KVARN_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, block_dim, tile_shared, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_kvarn_tiled<D, K_BITS, V_BITS>, launch_params,
        (const char *) Q->data,
        mask ? (const char *) mask->data : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        partial,
        (const uint8_t *) k_records->data,
        (const half *) k_stage->data,
        (const uint8_t *) v_records->data,
        (const half *) v_stage->data,
        live_groups,
        scale,
        max_bias,
        m0,
        m1,
        n_head_log2,
        logit_softcap,
        (int32_t) Q->ne[1],
        (int32_t) Q->ne[2],
        (int32_t) Q->ne[3],
        (int32_t) Q->nb[1],
        (int32_t) Q->nb[2],
        (int32_t) Q->nb[3],
        (int32_t) K->ne[1],
        (int32_t) K->ne[2],
        (int32_t) k_records->ne[1],
        stream_start,
        groups_per_stream,
        (int32_t) k_records->ne[0],
        (int32_t) v_records->ne[0],
        (int32_t) n_splits,
        mask ? (int32_t) mask->ne[1] : 0,
        mask ? (int32_t) mask->ne[3] : 0,
        mask ? (int32_t) mask->nb[1] : 0,
        mask ? (int64_t) mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());

    const dim3 combine_blocks((uint32_t) Q->ne[1], 1, (uint32_t) (Q->ne[2] * Q->ne[3]));
    const ggml_cuda_kernel_launch_params combine_params(combine_blocks, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_kvarn_combine<D>, combine_params,
        partial,
        (float *) dst->data,
        (int32_t) Q->ne[1],
        (int32_t) Q->ne[2],
        (int32_t) n_splits);
    CUDA_CHECK(cudaGetLastError());
}

#define GGML_CUDA_KVARN_DISPATCH_D(KB, VB)                            \
    do {                                                               \
        switch (head_dim) {                                            \
            case 128: ggml_cuda_flash_attn_ext_kvarn_launch<128, KB, VB>(ctx, dst, n_splits, groups_per_stream, stream_start, live_groups.get(), partial.get(), scale, max_bias, m0, m1, n_head_log2, logit_softcap); break; \
            case 256: ggml_cuda_flash_attn_ext_kvarn_launch<256, KB, VB>(ctx, dst, n_splits, groups_per_stream, stream_start, live_groups.get(), partial.get(), scale, max_bias, m0, m1, n_head_log2, logit_softcap); break; \
            case 384: ggml_cuda_flash_attn_ext_kvarn_launch<384, KB, VB>(ctx, dst, n_splits, groups_per_stream, stream_start, live_groups.get(), partial.get(), scale, max_bias, m0, m1, n_head_log2, logit_softcap); break; \
            case 512: ggml_cuda_flash_attn_ext_kvarn_launch<512, KB, VB>(ctx, dst, n_splits, groups_per_stream, stream_start, live_groups.get(), partial.get(), scale, max_bias, m0, m1, n_head_log2, logit_softcap); break; \
            default: GGML_ABORT("unsupported KVarN fused attention head dimension"); \
        }                                                              \
    } while (0)

#define GGML_CUDA_KVARN_DISPATCH_V(KB)                                 \
    do {                                                               \
        switch (v_bits) {                                              \
            case 2: GGML_CUDA_KVARN_DISPATCH_D(KB, 2); break;          \
            case 3: GGML_CUDA_KVARN_DISPATCH_D(KB, 3); break;          \
            case 4: GGML_CUDA_KVARN_DISPATCH_D(KB, 4); break;          \
            case 5: GGML_CUDA_KVARN_DISPATCH_D(KB, 5); break;          \
            case 6: GGML_CUDA_KVARN_DISPATCH_D(KB, 6); break;          \
            case 8: GGML_CUDA_KVARN_DISPATCH_D(KB, 8); break;          \
            default: GGML_ABORT("unsupported KVarN fused attention V bits"); \
        }                                                              \
    } while (0)

void ggml_cuda_flash_attn_ext_kvarn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_kvarn_supported(dst));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * k_mat = ggml_cuda_kvarn_unwrap_fattn_src(K);
    const ggml_tensor * k_records = k_mat->src[0];
    const ggml_tensor * k_stage = k_mat->src[1];
    const ggml_tensor * indices = k_mat->src[2];
    const int head_dim = (int) Q->ne[0];
    const int k_bits = ggml_get_op_params_i32(k_mat, 0);
    const int v_bits = ggml_get_op_params_i32(ggml_cuda_kvarn_unwrap_fattn_src(dst->src[2]), 0);

    const int stream_start = ggml_get_op_params_i32(k_mat, 2);
    const int n_stream = ggml_get_op_params_i32(k_mat, 3);
    const int n_total_stream = (int) (k_stage->ne[2] / (FATTN_KVARN_DIM * FATTN_KVARN_STAGE_GROUPS));
    const int groups_per_stream = (int) (k_records->ne[2] / n_total_stream);

    ggml_cuda_pool_alloc<int> live_groups(ctx.pool(), n_stream);
    cudaStream_t stream = ctx.stream();
    fattn_kvarn_live_groups_kernel<<<n_stream, FATTN_KVARN_DIM, 0, stream>>>(
        (const int64_t *) indices->data,
        (int) indices->ne[0],
        stream_start,
        n_stream,
        groups_per_stream,
        live_groups.get());

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Split the KV/context dimension across enough CTAs to fill the GPU (FlashDecoding-style).
    // The fused tile kernel runs at ~2 blocks/SM with the f16 tile, so target ~2 waves; cap at
    // the number of context groups (avoids empty splits) and at 32 (combine split_scale[] size).
    const int n_sm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int n_cta_base = (int) (Q->ne[2] * Q->ne[3]);
    const int n_groups = (int) ((K->ne[1] + FATTN_KVARN_DIM - 1) / FATTN_KVARN_DIM);
    int n_splits = (4 * n_sm + n_cta_base - 1) / n_cta_base;
    n_splits = n_splits < 1 ? 1 : n_splits;
    n_splits = n_splits > n_groups ? n_groups : n_splits;
    n_splits = n_splits > 32 ? 32 : n_splits;

    const int partial_stride = head_dim + 2;
    const size_t partial_count =
        (size_t) Q->ne[1] * (size_t) Q->ne[2] * (size_t) Q->ne[3] *
        (size_t) n_splits * (size_t) partial_stride;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), partial_count);

    switch (k_bits) {
        case 2: GGML_CUDA_KVARN_DISPATCH_V(2); break;
        case 3: GGML_CUDA_KVARN_DISPATCH_V(3); break;
        case 4: GGML_CUDA_KVARN_DISPATCH_V(4); break;
        case 5: GGML_CUDA_KVARN_DISPATCH_V(5); break;
        case 6: GGML_CUDA_KVARN_DISPATCH_V(6); break;
        case 8: GGML_CUDA_KVARN_DISPATCH_V(8); break;
        default: GGML_ABORT("unsupported KVarN fused attention K bits");
    }
}

#undef GGML_CUDA_KVARN_DISPATCH_V
#undef GGML_CUDA_KVARN_DISPATCH_D
