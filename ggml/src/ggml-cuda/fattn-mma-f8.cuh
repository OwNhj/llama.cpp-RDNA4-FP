// P9 Stage 2: native fp8 WMMA flash attention for F8 (e4m3) KV cache on RDNA4 (gfx12).
//
// Design (see design.md §4.7.3):
//   - KQ matmul  S = Q·K^T  : native fp8 WMMA. Q (f32) is scaled + cast to e4m3, K is read raw
//     as e4m3 (+ per-32-head-group f16 scale). The 8-int fp8 `mma` overload covers one
//     32-elem scale group (K=32). kscale is applied per output element (per seq_k column).
//   - softmax    : f32, online (running max / rowsum / rescale), same structure as the f16
//     kernel (same T_C_KQ fragment layout: fp8 WMMA C == f16 WMMA C == 16x16 f32 I_MAJOR).
//   - VKQ matmul O = P·V    : f16 WMMA. P (f32 softmax) is cast to f16, V is dequantized from
//     e4m3 (+ per-32-head_out-group scale) to f16 in SRAM. V is dequant because its scale
//     depends on the contraction dim (seq_k) and cannot be factored out of a native fp8 mma
//     (P*vscale would risk e4m3 range). This is the one unavoidable dequant in the path.
//
// Simplifications vs the f16 kernel (correctness first):
//   - 1 warp per block, 16 query rows (ncols1=16), 1 Q head per block (ncols2=1, GQA via grid).
//   - nbatch_fa = 32 KV positions per iteration (multiple of 32, aligns with the fp8 scale group).
//   - No stream-K, no sparse, no sinks, no ALiBi, no swizzle (plain SRAM), no cp_async.
//   - Causal mask applied analytically (key_pos > query_pos -> -inf).
//
// Only instantiated for DKQ == DV in {96, 128} on RDNA4; everything else falls back to MMA_F16.
// The kernel signature is always declared (mirrors the f16 kernel); the body is RDNA4-only.

#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "dequantize.cuh"

using namespace ggml_cuda_mma;

// Pack 4 e4m3 bytes (little-endian) into one int.
static __device__ __forceinline__ int pack_e4m3_4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (int)(b0) | ((int)(b1) << 8) | ((int)(b2) << 16) | ((int)(b3) << 24);
}

// Decode a SIGNED e4m3 byte to f32, exactly like dequantize_f8() in dequantize.cuh:
//   value = (byte & 0x80 ? -1 : +1) * ue4m3_raw(byte & 0x7F)
//
// Do NOT call ggml_cuda_ue4m3_to_fp32_raw() on the raw byte: that helper decodes the unsigned
// "ue4m3" variant and ignores bit 7, so every negative KV value would come out positive.
// The F8 KV cache is written by quantize_f32_f8_block() with signed e4m3, and the fp8 WMMA
// hardware reads the bytes as signed e4m3 as well, so the dequant must match that convention.
static __device__ __forceinline__ float e4m3_to_fp32_signed(uint8_t x) {
    return (x & 0x80 ? -1.0f : 1.0f) * ggml_cuda_ue4m3_to_fp32_raw(x & 0x7F);
}

// Two packed e4m3 bytes -> f32x2. ggml_cuda_e4m3x2_to_fp32() (common.cuh) is the shared
// implementation: it uses the gfx12 hardware instruction and falls back to a portable software
// decode elsewhere, and it follows the signed e4m3 convention the KV cache is written with.

// Same as e4m3x2_to_fp32 but for 4 consecutive e4m3 bytes loaded as one 32-bit word, halving the
// number of global loads in the V dequant.
static __device__ __forceinline__ void e4m3x4_to_fp32(uint32_t w, float4 & out) {
    const float2 lo = ggml_cuda_e4m3x2_to_fp32((uint8_t)( w        & 0xFF), (uint8_t)((w >>  8) & 0xFF));
    const float2 hi = ggml_cuda_e4m3x2_to_fp32((uint8_t)((w >> 16) & 0xFF), (uint8_t)((w >> 24) & 0xFF));
    out = make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <int DKQ, int DV, bool use_logit_softcap>
static __global__ void flash_attn_ext_f8(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
            const int32_t nb01, const int32_t nb02, const int64_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
            const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)

    GGML_UNUSED_VARS(sinks_ptr, dst_meta_ptr,
        max_bias, m0, m1, n_head_log2, ne00, ne01, ne02, ne03,
        ne10, ne13, ne31, ne32, nb31, nb32);

    const int * KV_max = KV_max_ptr;   // per (sequence, jt) count of filled KV keys (null if unused)

    constexpr int warp_size   = 32;
    constexpr int ncols1      = 16;   // query rows per block
    constexpr int nbatch_fa   = 32;   // KV positions per iteration (multiple of 32 = the fp8 scale group)
    constexpr int n_kv_groups = DKQ/32;
    constexpr int n_out_tiles = DV/16;

    // fragment tile types (all I_MAJOR, plain load_ldmatrix)
    using T_A_KQ  = tile<16,  8, int,   DATA_LAYOUT_I_MAJOR>;  // Q e4m3 (16 seq_q x 32 head)
    using T_B_KQ  = tile<16,  8, int,   DATA_LAYOUT_I_MAJOR>;  // K e4m3 (16 seq   x 32 head)
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;  // S        (16 seq_q x 16 seq)
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;  // V f16    (16 head_out x 16 seq)
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;  // P f16    (16 seq_q x 16 seq)
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;  // O        (16 seq_q x 16 head_out)

    const float    * Q_f   = (const float    *) Q_ptr;
    const block_f8 * K_b   = (const block_f8 *) K_ptr;
    const block_f8 * V_b   = (const block_f8 *) V_ptr;
    float          * dst_f = (float          *) dst_ptr;
    const half     * mask_h = (const half    *) mask_ptr;   // F16 mask: mask[query=q][key=k] = mask_h[k + q*n_pos_kv]
    const bool      use_mask = (mask_ptr != nullptr);

    // strides (element / block units)
    const int stride_Q1 = nb01 / (int)sizeof(float);            // per query row
    const int stride_Q2 = nb02 / (int)sizeof(float);            // per Q head
    const int nb11_blk  = nb11 / (int)sizeof(block_f8);         // per key  (K)
    const int nb12_blk  = nb12 / (int)sizeof(block_f8);         // per kv head (K)
    const int nb13_blk  = nb13 / (int)sizeof(block_f8);         // per stream (K)
    const int nb21_blk  = nb21 / (int)sizeof(block_f8);         // per key  (V)
    const int nb22_blk  = nb22 / (int)sizeof(block_f8);         // per kv head (V)
    const int nb23_blk  = nb23 / (int)sizeof(block_f8);         // per stream (V)

    // ne01 = init_fastdiv_values(Q->ne[1]) -> uint3 <mp, L, divisor>; the divisor (z) is Q->ne[1]
    const int n_pos_q  = (int) ne01.z;   // number of query positions
    const int n_pos_kv = ne11;           // number of KV positions

    const int gqa_ratio = ne02 / ne12;

    // block mapping (no stream-K): x = query tile, z = (zt_gqa, z_KV, sequence)
    const int jt       = blockIdx.x;
    const int z_KV     = blockIdx.z % ne12;
    const int zt_gqa   = (blockIdx.z / ne12) % gqa_ratio;
    const int sequence = blockIdx.z / (ne12 * gqa_ratio);
    const int zt_Q     = z_KV * gqa_ratio + zt_gqa;

    const int q0 = jt*ncols1;   // first absolute query position of this block

    // ------------------------------------------------------------------
    // SRAM layout
    //   sQ  : 16 x DKQ/4   int    (Q e4m3, head_dim-major per query row)
    //   sK  : nbatch_fa x DKQ/4   int    (K e4m3, head_dim-major per seq)
    //   sKd : nbatch_fa x n_kv_groups float (K scale, per (seq, head group))
    //   sVf : DV x nbatch_fa/2 half2  (V dequant f16, head_out-major, 2 seq per half2)
    //   sScale : 16 float   (per-query-row softmax rescale factor, indexed by seq_q)
    //   sRowsum: 16 float   (per-query-row rowsum, indexed by seq_q)
    // ------------------------------------------------------------------
    // K/V are double buffered so that the loads of the next KV tile can be issued while the current
    // tile is still being consumed by the mma; without this the loads are fully serialized with the
    // compute and the kernel is latency bound (it runs a single warp per block).
    constexpr int n_kv_bufs = 2;

    extern __shared__ char sram[];
    int   * sQ     = (int   *) (sram + 0);
    int   * sK     = (int   *) (sQ     + 16*(DKQ/4));
    float * sKd    = (float *) (sK     + n_kv_bufs*nbatch_fa*(DKQ/4));
    half2 * sVf    = (half2 *) (sKd    + n_kv_bufs*nbatch_fa*n_kv_groups);
    float * sScale = (float *) (sVf    + (size_t)n_kv_bufs*DV*(nbatch_fa/2));
    float * sRowsum= (float *) (sScale + 16);

    // K/V base for this (z_KV, sequence)
    const block_f8 * K0 = K_b + z_KV*nb12_blk + sequence*nb13_blk;
    const block_f8 * V0 = V_b + z_KV*nb22_blk + sequence*nb23_blk;
    // Q / dst base for this batch (sequence) — Q and dst are 4-D with a batch stride (nb03)
    const float * Q_base  = Q_f   + sequence*(nb03/(int)sizeof(float));
    float       * dst_base = dst_f + sequence*(nb03/(int)sizeof(float));


    // ------------------------------------------------------------------
    // registers
    // ------------------------------------------------------------------
    T_A_KQ    Q_A[n_kv_groups];          // Q fragments (loaded once, loop invariant)
    T_C_VKQ   VKQ_C[n_out_tiles];        // O accumulator
    float     KQ_max    = -FLT_MAX/2.0f;
    float     KQ_rowsum = 0.0f;

    #pragma unroll
    for (int i = 0; i < n_out_tiles; ++i) {
        #pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) { VKQ_C[i].x[l] = 0.0f; }
    }

    // ------------------------------------------------------------------
    // 1. Load Q (f32) -> scale -> e4m3 -> sQ, then fragments
    // ------------------------------------------------------------------
    {
        for (int idx = threadIdx.x; idx < 16*(DKQ/4); idx += warp_size) {
            const int r  = idx / (DKQ/4);
            const int hg = idx % (DKQ/4);      // int group (4 e4m3)
            const int h0 = hg*4;
            const int seq_q = q0 + r;
            if (seq_q < n_pos_q) {
                const float * qr = Q_base + seq_q*stride_Q1 + zt_Q*stride_Q2 + h0;
                sQ[idx] = pack_e4m3_4(
                    ggml_cuda_fp32_to_e4m3(qr[0]*scale),
                    ggml_cuda_fp32_to_e4m3(qr[1]*scale),
                    ggml_cuda_fp32_to_e4m3(qr[2]*scale),
                    ggml_cuda_fp32_to_e4m3(qr[3]*scale));
            } else {
                sQ[idx] = 0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < n_kv_groups; ++g) {
            load_ldmatrix(Q_A[g], sQ + g*8, DKQ/4);
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // 2. KV iteration
    // ------------------------------------------------------------------
    // Limit the KV iterations to the number of filled keys (KV_max), mirroring the f16 kernel.
    // KV_max is per (sequence, jt): the count of filled KV keys for this (sequence, query tile).
    const int iter_j = (n_pos_q + ncols1 - 1) / ncols1;   // number of query tiles
    int n_iters = (n_pos_kv + nbatch_fa - 1) / nbatch_fa;
    if (KV_max) {
        n_iters = min(n_iters, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }
    if (n_iters < 0) n_iters = 0;

    // ---- KV tile loader, double buffered (buf 0/1) ----
    // A seq's head_dim row is DKQ/32 groups of 32 contiguous e4m3 bytes (the 2-byte scale sits
    // between groups, so the groups are not contiguous with each other); qs is only 2-byte aligned
    // inside block_f8, hence memcpy_1<4> instead of a wide vector copy.
    auto load_KV = [&](const int buf, const int k0) {
        int   * sK_buf  = sK  + (size_t)buf*nbatch_fa*(DKQ/4);
        float * sKd_buf = sKd + (size_t)buf*nbatch_fa*n_kv_groups;
        half2 * sVf_buf = sVf + (size_t)buf*DV*(nbatch_fa/2);

        {
            constexpr int ints_per_group = 32/4;         // 8 int columns per block_f8 group
            for (int s = threadIdx.x; s < nbatch_fa; s += warp_size) {
                const int seq_k = k0 + s;
                if (seq_k < n_pos_kv) {
                    const block_f8 * kb = K0 + seq_k*nb11_blk;
                    #pragma unroll
                    for (int g = 0; g < n_kv_groups; ++g) {
                        const uint8_t * src = kb[g].qs;
                        #pragma unroll
                        for (int e = 0; e < ints_per_group; ++e) {
                            ggml_cuda_memcpy_1<4>(&sK_buf[s*(DKQ/4) + g*ints_per_group + e], src + 4*e);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int ic = 0; ic < DKQ/4; ++ic) {
                        sK_buf[s*(DKQ/4) + ic] = 0;
                    }
                }
            }
        }
        for (int idx = threadIdx.x; idx < nbatch_fa*n_kv_groups; idx += warp_size) {
            const int s = idx / n_kv_groups;
            const int g = idx % n_kv_groups;
            const int seq_k = k0 + s;
            sKd_buf[idx] = (seq_k < n_pos_kv) ? __half2float(K0[seq_k*nb11_blk + g].d) : 0.0f;
        }
        // V: iterate over (vscale group, seq pair). One block_f8 group holds 32 consecutive
        // head_out bytes plus a single scale, so each work item decodes a whole group and reads its
        // scale once, instead of touching one byte per (head, seq pair) with the scale re-read 32x.
        {
            constexpr int n_grp  = DV/32;          // vscale groups per row
            constexpr int n_seq2 = nbatch_fa/2;    // half2 columns (2 seq each)
            for (int idx = threadIdx.x; idx < n_grp*n_seq2; idx += warp_size) {
                const int g   = idx / n_seq2;
                const int sh2 = idx - g*n_seq2;
                const int s0  = k0 + 2*sh2;
                const block_f8 * blk0 = V0 + s0*nb21_blk + g;
                const block_f8 * blk1 = blk0 + nb21_blk;      // seq s0+1
                const float d0 = (s0     < n_pos_kv) ? __half2float(blk0->d) : 0.0f;
                const float d1 = (s0 + 1 < n_pos_kv) ? __half2float(blk1->d) : 0.0f;
                // half2 = (seq s0, seq s0+1) for the same head_out, matching the ldmatrix row layout.
                #pragma unroll
                for (int hq4 = 0; hq4 < 8; ++hq4) {
                    uint32_t w0, w1;
                    ggml_cuda_memcpy_1<4>(&w0, blk0->qs + 4*hq4);
                    ggml_cuda_memcpy_1<4>(&w1, blk1->qs + 4*hq4);
                    float4 f0, f1;
                    e4m3x4_to_fp32(w0, f0);
                    e4m3x4_to_fp32(w1, f1);
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int hq = 4*hq4 + q;
                        const float a = (q == 0) ? f0.x : (q == 1) ? f0.y : (q == 2) ? f0.z : f0.w;
                        const float b = (q == 0) ? f1.x : (q == 1) ? f1.y : (q == 2) ? f1.z : f1.w;
                        sVf_buf[(g*32 + hq)*(nbatch_fa/2) + sh2] = make_half2(a*d0, b*d1);
                    }
                }
            }
        }
    };

    // Software pipeline: issue the loads for tile it+1 before consuming tile it, so the global
    // memory latency overlaps the mma / softmax work instead of serializing with it. One sync per
    // tile is enough: after it, every thread has finished reading the buffer being refilled.
    load_KV(0, 0);
    __syncthreads();

    for (int it = 0; it < n_iters; ++it) {
        const int k0  = it*nbatch_fa;   // first absolute key position of this chunk
        const int cur = it & 1;
        const int nxt = cur ^ 1;

        int   * sK_cur  = sK  + (size_t)cur*nbatch_fa*(DKQ/4);
        float * sKd_cur = sKd + (size_t)cur*nbatch_fa*n_kv_groups;
        half2 * sVf_cur = sVf + (size_t)cur*DV*(nbatch_fa/2);

        if (it + 1 < n_iters) {
            load_KV(nxt, (it + 1)*nbatch_fa);
        }

        // ---- KQ matmul: S (16 seq_q x 32 seq) = sum_g kscale[seq][g] * (Q_g . K_g^T) ----
        constexpr int n_sh = nbatch_fa/16;   // 16-seq halves
        T_C_KQ S[n_sh];
        #pragma unroll
        for (int sh = 0; sh < n_sh; ++sh) {
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) { S[sh].x[l] = 0.0f; }
            #pragma unroll
            for (int g = 0; g < n_kv_groups; ++g) {
                T_B_KQ K_B;
                load_ldmatrix(K_B, sK_cur + sh*16*(DKQ/4) + g*8, DKQ/4);
                T_C_KQ tmp;
                // mma convention: D = second x first^T  ->  D[m][n] = second[m]*first[n]
                // want S[m=query][n=key] = Q[m]*K[n]  =>  first=K, second=Q
                mma(tmp, K_B, Q_A[g]);   // 8-int fp8 mma: 32-head dot
                // apply kscale per C element (per seq_k column)
                #pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int seq_k_idx = T_C_KQ::get_j(l);   // 0..15 within the 16-seq half
                    const float d = sKd_cur[(sh*16 + seq_k_idx)*n_kv_groups + g];
                    S[sh].x[l] += d * tmp.x[l];
                }
            }
        }
        __syncthreads();


        // ---- attention mask + OOB key handling + logit softcap ----
        // The mask (F16) encodes the causal/sparse pattern: mask[query=q][key=k] = mask_h[k + q*n_pos_kv].
        // Without a mask, use the standard causal: query q sees keys 0..(n_pos_kv - n_pos_q + q).
        #pragma unroll
        for (int sh = 0; sh < n_sh; ++sh) {
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int seq_q = q0 + T_C_KQ::get_i(l);
                const int seq_k = k0 + sh*16 + T_C_KQ::get_j(l);
                if (seq_k >= n_pos_kv) {
                    S[sh].x[l] = -FLT_MAX;   // OOB key
                } else if (use_mask) {
                    if (seq_q < n_pos_q) {
                        S[sh].x[l] += __half2float(mask_h[seq_k + (uint64_t)seq_q*n_pos_kv]);
                    }
                } else {
                    const int seq_q_abs = n_pos_kv - n_pos_q + seq_q;
                    if (seq_k > seq_q_abs) S[sh].x[l] = -FLT_MAX;
                }
            }
        }
        if constexpr (use_logit_softcap) {
            #pragma unroll
            for (int sh = 0; sh < n_sh; ++sh) {
                #pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    S[sh].x[l] = logit_softcap * tanhf(S[sh].x[l]);
                }
            }
        }

        // ---- online softmax (mirror f16: 2 threads per query row, shfl offset 16) ----
        float KQ_max_new = KQ_max;
        float KQ_rowsum_add = 0.0f;
        #pragma unroll
        for (int sh = 0; sh < n_sh; ++sh) {
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_max_new = fmaxf(KQ_max_new, S[sh].x[l] + FATTN_KQ_MAX_OFFSET);
            }
        }
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, 16, warp_size));

        #pragma unroll
        for (int sh = 0; sh < n_sh; ++sh) {
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const float x = S[sh].x[l] - KQ_max_new;
                const float p = (x < -20.0f) ? 0.0f : expf(x);
                S[sh].x[l] = p;
                KQ_rowsum_add += p;
            }
        }
        // the 16 seq P of a query row are split across 2 threads (t, t^16): reduce
        KQ_rowsum_add += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum_add, 16, warp_size);

        // ---- rescale O + rowsum (per query-row; VKQ C get_j = seq_q) ----
        {
            const float KQ_max_diff = KQ_max - KQ_max_new;
            float KQ_max_scale = expf(KQ_max_diff);
            *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;
            KQ_max = KQ_max_new;
            KQ_rowsum = KQ_max_scale*KQ_rowsum + KQ_rowsum_add;


            // per-query-row scale into SRAM (thread t owns seq_q = t%16 for the KQ C)
            sScale[threadIdx.x % 16] = KQ_max_scale;
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < n_out_tiles; ++i) {
                #pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= sScale[T_C_VKQ::get_j(l)];   // seq_q = get_j
                }
            }
            __syncthreads();
        }

        // ---- P (f32) -> f16 fragment (T_B_VKQ), 2 seq halves ----
        T_B_VKQ P_B[n_sh];
        #pragma unroll
        for (int sh = 0; sh < n_sh; ++sh) {
            P_B[sh] = get_half2(S[sh]);   // tile<16,16,float> -> tile<16,8,half2>
        }

        // ---- VKQ matmul: O (16 seq_q x DV) += P . V (f16 WMMA) ----
        #pragma unroll
        for (int oh = 0; oh < n_out_tiles; ++oh) {
            T_C_VKQ O;
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) { O.x[l] = 0.0f; }
            #pragma unroll
            for (int sh = 0; sh < n_sh; ++sh) {
                T_A_VKQ V_A;
                // Each sh group covers 16 seq = 8 half2 columns of the sVf row; the offset must
                // NOT be derived from nbatch_fa (nbatch_fa/4 happens to equal 8 only when it is 32).
                load_ldmatrix(V_A, sVf_cur + oh*16*(nbatch_fa/2) + sh*8, nbatch_fa/2);
                // mma convention: D = B x A^T  ->  D[m][n] = B[m]*A[n]
                // want O[m=head_out][n=seq_q] = V[m]*P[n]  =>  A=P, B=V
                mma(O, P_B[sh], V_A);   // f16 mma (16-row, f32 C)
            }
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[oh].x[l] += O.x[l];
            }
        }
        __syncthreads();   // the buffer just consumed may now be refilled by the next prefetch
    }

    // ------------------------------------------------------------------
    // 3. Normalize + write O (f32) to dst
    //    VKQ C layout: get_i = head_out, get_j = seq_q (query row within block)
    // ------------------------------------------------------------------
    {
        // KQ_rowsum is already the full rowsum per seq_q (reduced in-loop); store per query row
        sRowsum[threadIdx.x % 16] = KQ_rowsum;
        __syncthreads();

        #pragma unroll
        for (int oh = 0; oh < n_out_tiles; ++oh) {
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int h = oh*16 + T_C_VKQ::get_i(l);   // head_out
                const int r = T_C_VKQ::get_j(l);           // seq_q (0..15)
                const int seq_q = q0 + r;
                if (seq_q < n_pos_q) {
                    const float inv = (sRowsum[r] == 0.0f) ? 0.0f : 1.0f / sRowsum[r];
                    dst_base[seq_q*stride_Q1 + zt_Q*stride_Q2 + h] = VKQ_C[oh].x[l] * inv;
                }
            }
        }


    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

// ---------------------------------------------------------------------------
// case function + launch (host; always compiled, mirrors the f16 case function)
// ---------------------------------------------------------------------------

template <int DKQ, int DV, bool use_logit_softcap>
static void ggml_cuda_flash_attn_ext_mma_f8_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int id  = ggml_cuda_get_device();

    // SRAM size
    const size_t nbytes_shared =
         16*(DKQ/4)*(size_t)sizeof(int)          // sQ
      + 2*32*(DKQ/4)*(size_t)sizeof(int)        // sK    x2 buffers (nbatch_fa = 32)
      + 2*32*(DKQ/32)*(size_t)sizeof(float)     // sKd   x2 buffers (nbatch_fa = 32)
      + (size_t)2*DV*16*sizeof(half2)           // sVf   x2 buffers (nbatch_fa/2 = 16)
      + 32*sizeof(float);                       // sScale[16] + sRowsum[16]

    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int gqa_ratio  = Q->ne[2] / K->ne[2];
    const int ntiles_x   = (Q->ne[1] + 15) / 16;
    const int ntiles_z   = gqa_ratio * K->ne[2] * Q->ne[3];


    // Compute KV_max from the mask (per (sequence, jt) count of valid keys), mirroring launch_fattn.
    // This lets the kernel skip unfilled/masked KV keys (the F16 path does the same).
    // The pool allocation must stay alive until the F8 kernel launch below (function scope, like launch_fattn).
    ggml_cuda_pool_alloc<int> KV_max(ctx.pool());
    int * KV_max_ptr = nullptr;
    if (mask && (K->ne[1] % FATTN_KQ_STRIDE) == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);
        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
        const int ne_KV_max = blocks_num_KV_max.x * blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;
        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params kvmax_launch(blocks_num_KV_max, block_dim_KV_max, 0, ctx.stream());
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<16>, kvmax_launch,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
        KV_max_ptr = KV_max.ptr;
    }

    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const dim3 block_dim(warp_size_host, 1, 1);
    const dim3 blocks_num(ntiles_x, 1, ntiles_z);

    ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, ctx.stream());

    ggml_cuda_kernel_launch(flash_attn_ext_f8<DKQ, DV, use_logit_softcap>, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? ((const char *) mask->data) : nullptr,
        nullptr,   // sinks
        KV_max_ptr,   // KV_max (per (sequence, jt) count of valid keys; nullptr if not computed)
        (float *) KQV->data,
        nullptr,   // dst_meta
        scale,
        max_bias,
        m0,
        m1,
        n_head_log2,
        logit_softcap,
        (int32_t) Q->ne[0], init_fastdiv_values(Q->ne[1]), (int32_t) Q->ne[2], (int32_t) Q->ne[3],
        (int32_t) Q->nb[1], (int32_t) Q->nb[2], Q->nb[3],
        (int32_t) K->ne[0], (int32_t) K->ne[1], (int32_t) K->ne[2], (int32_t) K->ne[3],
        (int32_t) K->nb[1], (int32_t) K->nb[2], K->nb[3],
        (int32_t) V->nb[1], (int32_t) V->nb[2], V->nb[3],
        (int32_t) 0, (int32_t) 0, mask ? (int32_t) mask->ne[3] : 0,
        (int32_t) 0, (int32_t) 0, mask ? (int64_t) mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());
    (void) id;
}
