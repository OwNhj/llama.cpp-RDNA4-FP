#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "radiance-gemm.cuh"

#include <cstdint>
#include <unordered_map>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
        case GGML_TYPE_Q4_0_ROCMI4:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMI4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_SYM4:
            mul_mat_q_case<GGML_TYPE_Q4_0_SYM4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP8:
            mul_mat_q_case<GGML_TYPE_MXFP8>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP6:
            mul_mat_q_case<GGML_TYPE_MXFP6>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4_E4M3:
            mul_mat_q_case<GGML_TYPE_MXFP4_E4M3>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct ggml_radiance_weight {
    void * W;
    void * Ws;
    void * Wref;
    int    device;
};

static std::unordered_map<const void *, ggml_radiance_weight> g_radiance_weights;

static void ggml_cuda_free_radiance_weights() {
    for (auto & kv : g_radiance_weights) {
        cudaFree(kv.second.W);
        cudaFree(kv.second.Ws);
        cudaFree(kv.second.Wref);
    }
    g_radiance_weights.clear();
}

// the repack cache is keyed by the weight tensor's device pointer. when a model's CUDA buffer
// is freed the keys become dangling: the next load may reuse the same address (stale repacked
// weights) or a different one (leaked repacks -> OOM across reloads). drop every cache entry
// whose key falls inside the freed range.
void ggml_cuda_invalidate_weight_caches(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    for (auto it = g_radiance_weights.begin(); it != g_radiance_weights.end();) {
        const char * k = (const char *) it->first;
        if (k >= lo && k < hi) {
            cudaFree(it->second.W);
            cudaFree(it->second.Ws);
            cudaFree(it->second.Wref);
            it = g_radiance_weights.erase(it);
        } else {
            ++it;
        }
    }
}

// radiance W8A8 fp8 WMMA prefill fast path for MXFP4 weights. returns true when handled.
static bool ggml_cuda_mul_mat_q_radiance(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                         const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (getenv("GGML_RAD_DISABLE")) {
        return false;
    }
    if (ids) {
        return false;
    }
    if (src0->ne[2] != 1 || src1->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }

    const int cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int device = ggml_cuda_get_device();

    const int64_t K    = src0->ne[0];             // also src1->ne[0]
    const int64_t N    = src0->ne[1];             // dst rows
    const int64_t M    = src1->ne[1];             // tokens

    const bool contig_w  = src0->nb[1] == ggml_row_size(src0->type, K);
    const bool contig_a  = src1->nb[1] == K * sizeof(float);
    const bool contig_dst = dst->nb[1] == N * sizeof(float) && dst->nb[2] == dst->nb[1] * M;
    const bool dbg = getenv("GGML_RAD_DEBUG") != nullptr;
    if (!contig_w || !contig_a || !contig_dst) {
        if (dbg) fprintf(stderr, "[rad] decline contig w=%d a=%d dst=%d K=%ld N=%ld M=%ld\n",
                          contig_w, contig_a, contig_dst, (long)K, (long)N, (long)M);
        return false;
    }
    if (!ggml_cuda_radiance_supported(cc, src0->type, K, N, M, src1->ne[0], contig_dst)) {
        if (dbg) fprintf(stderr, "[rad] decline policy K=%ld N=%ld M=%ld type=%d\n",
                         (long)K, (long)N, (long)M, (int)src0->type);
        return false;
    }
    if (dbg) fprintf(stderr, "[rad] ENGAGE K=%ld N=%ld M=%ld\n", (long)K, (long)N, (long)M);

    cudaStream_t stream = ctx.stream();

    // one-time repack of the weight tensors (weights are static across calls)
    const void * key = src0->data;
    auto it = g_radiance_weights.find(key);
    if (it == g_radiance_weights.end() || it->second.device != device) {
        // capture-safe contract: repack allocates; a miss while the stream is capturing must
        // fall through to MMQ instead of the illegal cudaMalloc inside the capture.
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            if (dbg) fprintf(stderr, "[rad] decline capture-miss K=%ld N=%ld M=%ld\n", (long)K, (long)N, (long)M);
            return false;
        }
        ggml_radiance_weight w;
        w.device = device;
        const size_t wq_bytes  = (size_t)N * K / 2;
        const size_t ws_bytes  = (size_t)N * (K / 32);
        const size_t wref_bytes = (size_t)N;
        CUDA_CHECK(cudaMalloc(&w.W,    wq_bytes));
        CUDA_CHECK(cudaMalloc(&w.Ws,   ws_bytes));
        CUDA_CHECK(cudaMalloc(&w.Wref, wref_bytes));
        ggml_cuda_radiance_repack(src0->data, K / 32, N, K, w.W, w.Ws, w.Wref, stream);
        CUDA_CHECK(cudaGetLastError());
        it = g_radiance_weights.emplace(key, w).first;
    }
    const ggml_radiance_weight & w = it->second;

    // activation fp8: prefer the q the fused GLU already produced; else quantize from f32
    unsigned char * q8 = nullptr;
    float * as = nullptr;
    if (!ggml_rad_fused_act_lookup(src1->data, K, (const unsigned char **) &q8, (const float **) &as)) {
        ggml_cuda_radiance_quantize_tokens((const float *) src1->data, src1->nb[1] / sizeof(float),
                                           K, M, &q8, &as, stream);
        if (q8 == nullptr) {
            return false;   // act scratch realloc declined under capture, fall back to MMQ
        }
    }

    ggml_cuda_radiance_gemm_f32(q8, w.W, w.Ws, w.Wref, as,
                                (float *) dst->data, (int) M, (int) N, (int) K, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    if (ggml_cuda_mul_mat_q_radiance(ctx, src0, src1, ids, dst)) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else if ((src0->type == GGML_TYPE_MXFP8 || src0->type == GGML_TYPE_MXFP6 || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_MXFP4_E4M3)
                       && GGML_CUDA_CC_IS_RDNA4(cc)) {
                // e4m3 y tiles for the W8A8 fp8 WMMA path (RDNA4 only). NVFP4 is deliberately
                // NOT here: its MMQ vec_dot is the mainline int8 one (q8_0_16), which consumes
                // q8_1 int8 y. Producing e4m3 y while running an int8 vec_dot feeds int8 code
                // e4m3 bytes and corrupts the result, so the two choices have to move together.
                quantize_mmq_mxfp8_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if ((src0->type == GGML_TYPE_MXFP8 || src0->type == GGML_TYPE_MXFP6 || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_MXFP4_E4M3)
                   && GGML_CUDA_CC_IS_RDNA4(cc)) {
            // e4m3 y tiles for the W8A8 fp8 WMMA path (RDNA4 only); other archs fall through
            // to the q8_1 int8 y consumed by the mainline int8 vec_dots. NVFP4 is excluded here
            // for the same reason as in ggml_cuda_mul_mat_q: its MMQ vec_dot is the int8 one
            // (q8_0_16), so e4m3 y would be read as int8 code and corrupt the result.
            if (dedup_bcast) {
                quantize_scatter_mmq_mxfp8_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_mxfp8_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                       ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_MXFP8:
        case GGML_TYPE_MXFP6:
        case GGML_TYPE_MXFP4_E4M3:
        case GGML_TYPE_Q4_0_ROCMI4:
        case GGML_TYPE_Q4_0_SYM4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    // The MXFP8/MXFP6 W8A8 fp8 WMMA path is implemented for RDNA4 only.
    if ((type == GGML_TYPE_MXFP8 || type == GGML_TYPE_MXFP6 || type == GGML_TYPE_MXFP4_E4M3) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

// free cached radiance weight buffers at process exit (fork-local simplification)
static struct ggml_radiance_weight_dtor {
    ~ggml_radiance_weight_dtor() { ggml_cuda_free_radiance_weights(); }
} ggml_radiance_weight_dtor_instance;
