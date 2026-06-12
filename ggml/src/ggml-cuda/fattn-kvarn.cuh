#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdlib>

static constexpr int FATTN_KVARN_DIM = 128;
static constexpr int FATTN_KVARN_D256 = 256;
static constexpr int FATTN_KVARN_STAGE_GROUPS = 3;

static __device__ __forceinline__ uint8_t fattn_kvarn_unpack_record(const uint8_t * record, int index, int bits) {
    if (bits == 4) {
        const uint8_t packed = record[index >> 1];
        return (packed >> ((index & 1) * 4)) & 0x0fu;
    }
    if (bits == 8) {
        return record[index];
    }
    if (bits == 2) {
        const uint8_t packed = record[index >> 2];
        return (packed >> ((index & 3) * 2)) & 0x03u;
    }

    uint8_t value = 0;
    const int bit_offset = index * bits;
    for (int bit = 0; bit < bits; ++bit) {
        const int absolute = bit_offset + bit;
        value |= ((record[absolute >> 3] >> (absolute & 7)) & 1u) << bit;
    }
    return value;
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

static __device__ __forceinline__ float fattn_kvarn_load(
        const uint8_t * records,
        const half * stage,
        int n_record_heads,
        int record_head,
        int stream,
        int groups_per_stream,
        int record_bytes,
        int live_group,
        int token,
        int dim,
        int bits,
        bool value) {
    const int group = token / FATTN_KVARN_DIM;
    const int pos = token - group * FATTN_KVARN_DIM;
    const int stage_base = stream * FATTN_KVARN_DIM * FATTN_KVARN_STAGE_GROUPS;

    if (group == 0 || (group > 0 && group <= live_group && group + 1 >= live_group)) {
        const int stage_pos = stage_base + (group == 0 ? pos : FATTN_KVARN_DIM + ((group - 1) & 1) * FATTN_KVARN_DIM + pos);
        return __half2float(stage[((int64_t) stage_pos * n_record_heads + record_head) * FATTN_KVARN_DIM + dim]);
    }

    if (group >= live_group || group >= groups_per_stream) {
        return 0.0f;
    }

    const int record_group = stream * groups_per_stream + group;
    const uint8_t * record = records + ((int64_t) record_group * n_record_heads + record_head) * record_bytes;
    const int row = value ? pos : dim;
    const int col = value ? dim : pos;
    const int payload_bytes = FATTN_KVARN_DIM * FATTN_KVARN_DIM * bits / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis = scale_axis + FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + FATTN_KVARN_DIM;
    const uint8_t q = fattn_kvarn_unpack_record(record, row * FATTN_KVARN_DIM + col, bits);
    return (q * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) * __half2float(other_axis[col]);
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

static __global__ void flash_attn_ext_kvarn_d256_k4v4(
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

    __shared__ float q_rot[FATTN_KVARN_D256];
    __shared__ float scores[FATTN_KVARN_DIM];
    __shared__ float weights[FATTN_KVARN_DIM];
    __shared__ float warp_reduce[4];
    __shared__ float shared_scalar;
    __shared__ float warp_partial[4][FATTN_KVARN_D256];
    __shared__ float out_shared[FATTN_KVARN_D256];

    q_rot[tid] = ((const float *) Q)[tid];
    __syncthreads();
    fattn_kvarn_wht_128(q_rot, tid);
    q_rot[FATTN_KVARN_DIM + tid] = ((const float *) Q)[FATTN_KVARN_DIM + tid];
    __syncthreads();
    fattn_kvarn_wht_128(q_rot + FATTN_KVARN_DIM, tid);
    q_rot[tid] *= scale;
    q_rot[FATTN_KVARN_DIM + tid] *= scale;
    __syncthreads();

    float kq_max = -FLT_MAX / 2.0f;
    float kq_sum = 0.0f;
    float out_local[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_local[i] = 0.0f;
    }

    const int kv_begin = ((int64_t) n_kv * split) / n_splits;
    const int kv_end   = ((int64_t) n_kv * (split + 1)) / n_splits;
    const int k0_begin = (kv_begin / FATTN_KVARN_DIM) * FATTN_KVARN_DIM;
    for (int k0 = k0_begin; k0 < kv_end; k0 += FATTN_KVARN_DIM) {
        float warp_max = kq_max;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int token = k0 + warp * 32 + i;
            float sum = 0.0f;
            if (token >= kv_begin && token < kv_end) {
#pragma unroll
                for (int d = 0; d < FATTN_KVARN_DIM; d += 32) {
                    const int dim = d + lane;
                    const int rh = kv_head * 2;
                    const float k = fattn_kvarn_load(k_records, k_stage, record_heads, rh, stream,
                            groups_per_stream, k_record_bytes, live_group, token, dim, 4, false);
                    sum += k * q_rot[dim];
                }
#pragma unroll
                for (int d = 0; d < FATTN_KVARN_DIM; d += 32) {
                    const int dim = d + lane;
                    const int rh = kv_head * 2 + 1;
                    const float k = fattn_kvarn_load(k_records, k_stage, record_heads, rh, stream,
                            groups_per_stream, k_record_bytes, live_group, token, dim, 4, false);
                    sum += k * q_rot[FATTN_KVARN_DIM + dim];
                }
            }
            sum = fattn_kvarn_warp_sum(sum);
            if (logit_softcap != 0.0f) {
                sum = logit_softcap * tanhf(sum);
            }
            if (maskh && token >= kv_begin && token < kv_end) {
                sum += slope * __half2float(maskh[token]);
            }
            if (token < kv_begin || token >= kv_end) {
                sum = -FLT_MAX / 2.0f;
            }
            warp_max = fmaxf(warp_max, sum + FATTN_KQ_MAX_OFFSET);
            if (lane == i) {
                scores[warp * 32 + i] = sum;
            }
        }

        if (lane == 0) {
            warp_reduce[warp] = warp_max;
        }
        __syncthreads();

        float block_max = tid < 4 ? warp_reduce[tid] : -FLT_MAX / 2.0f;
        block_max = fattn_kvarn_warp_max(block_max);
        if (tid == 0) {
            shared_scalar = block_max;
        }
        __syncthreads();

        const float kq_max_new = shared_scalar;
        const float kq_max_scale = __expf(kq_max - kq_max_new);
        kq_max = kq_max_new;
        kq_sum *= kq_max_scale;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_local[i] *= kq_max_scale;
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
        for (int i = 0; i < 32; ++i) {
            const int token = k0 + warp * 32 + i;
            if (token < kv_begin || token >= kv_end) {
                continue;
            }
            const float kq = weights[warp * 32 + i];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int out_dim = lane * 8 + j;
                const int slice = out_dim >= FATTN_KVARN_DIM;
                const int dim = out_dim - slice * FATTN_KVARN_DIM;
                const int rh = kv_head * 2 + slice;
                const float v = fattn_kvarn_load(v_records, v_stage, record_heads, rh, stream,
                        groups_per_stream, v_record_bytes, live_group, token, dim, 4, true);
                out_local[j] += kq * v;
            }
        }
        __syncthreads();
    }

    if (sinks_ptr && split == 0) {
        const float sink = ((const float *) sinks_ptr)[head];
        const float kq_max_new = fmaxf(sink, kq_max);
        const float kq_max_scale = __expf(kq_max - kq_max_new);
        const float sink_w = __expf(sink - kq_max_new);
        kq_max = kq_max_new;
        kq_sum = kq_sum * kq_max_scale + sink_w;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_local[i] *= kq_max_scale;
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        warp_partial[warp][lane * 8 + i] = out_local[i];
    }
    __syncthreads();

    out_shared[tid] = warp_partial[0][tid] + warp_partial[1][tid] + warp_partial[2][tid] + warp_partial[3][tid];
    out_shared[FATTN_KVARN_DIM + tid] =
        (warp_partial[0][FATTN_KVARN_DIM + tid] + warp_partial[1][FATTN_KVARN_DIM + tid] +
         warp_partial[2][FATTN_KVARN_DIM + tid] + warp_partial[3][FATTN_KVARN_DIM + tid]);
    __syncthreads();

    const int partial_stride = FATTN_KVARN_D256 + 2;
    float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
    if (tid == 0) {
        partial_block[0] = kq_max;
        partial_block[1] = kq_sum;
    }
    partial_block[2 + tid] = out_shared[tid];
    partial_block[2 + FATTN_KVARN_DIM + tid] = out_shared[FATTN_KVARN_DIM + tid];

    GGML_UNUSED(ne03);
}

static __global__ void flash_attn_ext_kvarn_d256_combine(
        const float * partial,
        float * dst,
        int32_t ne01,
        int32_t ne02,
        int32_t n_splits) {
    const int tid = threadIdx.x;
    const int ic0 = blockIdx.x;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int partial_stride = FATTN_KVARN_D256 + 2;

    __shared__ float out_shared[FATTN_KVARN_D256];
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

    float out0 = 0.0f;
    float out1 = 0.0f;
    for (int split = 0; split < n_splits; ++split) {
        const float * partial_block = partial + ((((sequence * ne01 + ic0) * ne02 + head) * n_splits + split) * partial_stride);
        const float scale = split_scale[split];
        out0 += partial_block[2 + tid] * scale;
        out1 += partial_block[2 + FATTN_KVARN_DIM + tid] * scale;
    }

    out_shared[tid] = out0 / shared_sum;
    out_shared[FATTN_KVARN_DIM + tid] = out1 / shared_sum;
    __syncthreads();

    fattn_kvarn_wht_128(out_shared, tid);
    dst[((sequence * ne01 + ic0) * ne02 + head) * FATTN_KVARN_D256 + tid] = out_shared[tid];
    __syncthreads();
    fattn_kvarn_wht_128(out_shared + FATTN_KVARN_DIM, tid);
    dst[((sequence * ne01 + ic0) * ne02 + head) * FATTN_KVARN_D256 + FATTN_KVARN_DIM + tid] =
        out_shared[FATTN_KVARN_DIM + tid];
}

static const ggml_tensor * ggml_cuda_kvarn_unwrap_fattn_src(const ggml_tensor * t) {
    while (t != nullptr &&
            (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE)) {
        t = t->src[0];
    }
    return t;
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
    if (Q == nullptr || K == nullptr || V == nullptr || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != FATTN_KVARN_D256 || K->ne[0] != FATTN_KVARN_D256 || V->ne[0] != FATTN_KVARN_D256) {
        return false;
    }
    if (Q->ne[1] != 1 || K->ne[1] <= 0 || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || V->ne[3] != K->ne[3]) {
        return false;
    }

    const ggml_tensor * k_mat = ggml_cuda_kvarn_unwrap_fattn_src(K);
    const ggml_tensor * v_mat = ggml_cuda_kvarn_unwrap_fattn_src(V);
    if (k_mat == nullptr || v_mat == nullptr || k_mat->op != GGML_OP_KVARN_MATERIALIZE || v_mat->op != GGML_OP_KVARN_MATERIALIZE) {
        return false;
    }
    if (ggml_get_op_params_i32(k_mat, 0) != 4 || ggml_get_op_params_i32(v_mat, 0) != 4) {
        return false;
    }
    if (ggml_get_op_params_i32(k_mat, 1) != 0 || ggml_get_op_params_i32(v_mat, 1) != 1) {
        return false;
    }
    if (ggml_get_op_params_i32(k_mat, 2) != ggml_get_op_params_i32(v_mat, 2) ||
            ggml_get_op_params_i32(k_mat, 3) != ggml_get_op_params_i32(v_mat, 3)) {
        return false;
    }
    const ggml_tensor * k_indices = k_mat->src[2];
    const ggml_tensor * v_indices = v_mat->src[2];
    if (k_indices == nullptr || v_indices == nullptr ||
            k_indices->type != GGML_TYPE_I64 || v_indices->type != GGML_TYPE_I64 ||
            k_indices->ne[0] != v_indices->ne[0] ||
            k_indices->ne[1] != v_indices->ne[1] ||
            k_indices->ne[2] != v_indices->ne[2] ||
            k_indices->ne[3] != v_indices->ne[3]) {
        return false;
    }
    if (k_mat->src[0]->ne[1] != K->ne[2] * 2 || v_mat->src[0]->ne[1] != V->ne[2] * 2) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_kvarn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_kvarn_supported(dst));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * k_mat = ggml_cuda_kvarn_unwrap_fattn_src(K);
    const ggml_tensor * v_mat = ggml_cuda_kvarn_unwrap_fattn_src(V);

    const ggml_tensor * k_records = k_mat->src[0];
    const ggml_tensor * k_stage = k_mat->src[1];
    const ggml_tensor * indices = k_mat->src[2];
    const ggml_tensor * v_records = v_mat->src[0];
    const ggml_tensor * v_stage = v_mat->src[1];

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

    int n_splits = (int) ((K->ne[1] + 2047) / 2048);
    n_splits = n_splits < 1 ? 1 : n_splits;
    n_splits = n_splits > 16 ? 16 : n_splits;

    const int partial_stride = FATTN_KVARN_D256 + 2;
    const size_t partial_count =
        (size_t) Q->ne[1] * (size_t) Q->ne[2] * (size_t) Q->ne[3] *
        (size_t) n_splits * (size_t) partial_stride;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), partial_count);

    const dim3 blocks((uint32_t) Q->ne[1], (uint32_t) n_splits, (uint32_t) (Q->ne[2] * Q->ne[3]));
    const dim3 block_dim(FATTN_KVARN_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_kvarn_d256_k4v4, launch_params,
        (const char *) Q->data,
        mask ? (const char *) mask->data : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        partial.get(),
        (const uint8_t *) k_records->data,
        (const half *) k_stage->data,
        (const uint8_t *) v_records->data,
        (const half *) v_stage->data,
        live_groups.get(),
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
    ggml_cuda_kernel_launch(flash_attn_ext_kvarn_d256_combine, combine_params,
        partial.get(),
        (float *) dst->data,
        (int32_t) Q->ne[1],
        (int32_t) Q->ne[2],
        (int32_t) n_splits);
    CUDA_CHECK(cudaGetLastError());
}
