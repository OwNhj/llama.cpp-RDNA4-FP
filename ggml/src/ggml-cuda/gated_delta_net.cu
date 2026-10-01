#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


// bf16 WMMA chunked prefill is the default on RDNA4; GGML_GDN_CHUNKED=0 selects the recurrence.
static bool gdn_chunked_enabled(
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t neq0, int64_t nek0, int64_t neq3, int64_t nek3, int64_t nev3,
        bool kda, bool keep_rs) {
    const char * env = getenv("GGML_GDN_CHUNKED");
    if (env != nullptr && env[0] == '0') {
        return false;
    }
    if (kda || keep_rs || S_v != 128 || neq0 != 128 || nek0 != 128 || n_tokens < 64 ||
        neq3 != nev3 || nek3 != nev3 || H <= 0 || n_seqs <= 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_RDNA4(cc);
}

// ===================== chunked prefill, bf16 WMMA (radiance-style) =====================
// The per-token recurrence becomes per-chunk matrix work, so the state chain runs n_tokens/C
// steps instead of n_tokens. With c_t = sum_{r<=t} g_r and A_t = exp(c_t):
//   L[t][s] = beta_t exp(c_t - c_s) (k_t . k_s),  s < t      (I + L is unit lower triangular)
//   RHS[t]  = beta_t (v_t - A_t (k_t . S0))
//   D       = (I + L)^-1 RHS
//   o_t     = scale (A_t (q_t . S0) + sum_{s<=t} exp(c_t - c_s) (q_t . k_s) D_s)
//   S0     <- A_{C-1} S0 + sum_s exp(c_{C-1} - c_s) k_s (x) D_s
// Every ratio is exp of a difference rather than a quotient of two A values: a strongly negative
// gate drives A_t to zero, and the quotient form turns the chunk into NaN.
//
// Every phase runs on 16x16x16 bf16 WMMA. The recurrent state stays in the fp32 accumulators for
// the whole chunk loop and is handed to the next matmul as a B operand by a plain bf16 convert:
// on gfx12 the C layout of a 16x16 tile equals the B layout of the same tile, so h needs neither
// LDS nor a shuffle. h, w, u and v_new never reach memory at all, which is where the win over the
// old f32 register-tiled kernel came from (it was slower than the plain recurrence).
//
// One workgroup per (seq, head), 8 warps. Warp w owns v-columns [16w, 16w+16) of the state, so
// the 8 fp32 accumulators of warp w are exactly the B fragments of that column block.
#if defined(GGML_USE_HIP)

typedef __attribute__((ext_vector_type(8))) __bf16 gdn_bf16x8;
typedef __attribute__((ext_vector_type(4))) __bf16 gdn_bf16x4;
typedef __attribute__((ext_vector_type(8))) float  gdn_f32x8;

__device__ __forceinline__ gdn_f32x8 gdn_wmma_bf16(gdn_bf16x8 a, gdn_bf16x8 b, gdn_f32x8 c) {
#if defined(RDNA4)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    // never reached: gdn_chunked_enabled only returns true on an RDNA4 device
    GGML_UNUSED_VARS(a, b);
    return c;
#endif
}

// lane L gets a 16x16 tile row (row0 + L%16) and the 8 columns starting at (col0 + 8*(L/16)).
// This is the gfx12 A layout. The same call loads a B fragment of K^T, i.e. the 16x16 tile of
// the transposed operand, which is why plain row-major data serves both A and B here.
__device__ __forceinline__ gdn_bf16x8 gdn_ld_frag(const __bf16 * __restrict__ p, int ld, int row0, int col0, int lane) {
    const __bf16 * q = p + (row0 + (lane & 15)) * ld + col0 + 8 * (lane >> 4);
    const gdn_bf16x4 lo = *(const gdn_bf16x4 *) q;
    const gdn_bf16x4 hi = *(const gdn_bf16x4 *) (q + 4);
    gdn_bf16x8 r;
    r[0] = lo[0]; r[1] = lo[1]; r[2] = lo[2]; r[3] = lo[3];
    r[4] = hi[0]; r[5] = hi[1]; r[6] = hi[2]; r[7] = hi[3];
    return r;
}

__device__ __forceinline__ gdn_bf16x8 gdn_to_bf16(gdn_f32x8 v) {
    gdn_bf16x8 r;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = (__bf16) v[i];
    }
    return r;
}

// C slot j of lane L is element (8*(L/16)+j, L%16) of the tile, so a f32x8 accumulated here is
// written back as element (tile_r + 8*(L/16)+j, tile_c + L%16).
template <int C>
__global__ void __launch_bounds__(256, 1) gdn_chunked_wmma_cuda(
        const float * __restrict__ q_d, const float * __restrict__ k_d,
        const float * __restrict__ v_d, const float * __restrict__ g_d,
        const float * __restrict__ b_d, const float * __restrict__ s_d,
        float * __restrict__ dst_d, float * __restrict__ state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, float scale) {
    constexpr int KD  = 128;
    constexpr int NW  = 8;
    constexpr int NTH = NW * 32;
    constexpr int KP  = KD + 4;   // k/q row pitch, bf16
    constexpr int KT  = C + 4;    // k-transpose row pitch, bf16
    constexpr int CP  = C + 4;    // fp32 scratch pitch
    constexpr int CB  = C + 4;    // bf16 scratch pitch
    static_assert(C == 32, "the warp job map below assumes C = 32 and nb = warp");

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb   = warp;
    const int h    = blockIdx.x;
    const int seq  = blockIdx.y;
    const int iq1  = (int) (h % neqk1);

    const float * kbase = k_d + seq * sq3 + iq1 * sq1;
    const float * qbase = q_d + seq * sq3 + iq1 * sq1;
    const float * vbase = v_d + seq * sv3 + h * sv1;
    const float * gbase = g_d + seq * sb3 + h * sb1;
    const float * bbase = b_d + seq * sb3 + h * sb1;
    const float * s_in  = s_d + seq * H * KD * KD + h * KD * KD;

    extern __shared__ char wsmem[];
    __bf16 * s_k  = (__bf16 *) wsmem;                    // C x KP
    __bf16 * s_q  = s_k  + C * KP;                       // C x KP
    __bf16 * s_kt = s_q  + C * KP;                       // KD x KT
    float  * s_gr = (float *) (s_kt + KD * KT);          // C x CP, gram then in-place inverse
    __bf16 * s_inv= (__bf16 *) (s_gr + C * CP);          // C x CB
    __bf16 * s_qk = s_inv + C * CB;                      // C x CB
    __bf16 * s_qw = s_qk  + C * CB;                      // C x CB
    float  * s_a  = (float *) (s_qw + C * CB);           // C, A_t = exp(c_t); may flush to 0
    float  * s_b  = s_a + C;                             // C
    float  * s_g  = s_b + C;                             // C
    float  * s_c  = s_g + C;                             // C, running sum of g, never underflows
    float  * s_x  = (float *) s_q;                       // C x CP, scratch: dead q tile after q.S0

    // ---- state -> accumulators. lane L of warp w holds h[16i + 8*(L/16) + j][16w + L%16].
    gdn_f32x8 st[KD / 16];
#pragma unroll
    for (int ib = 0; ib < KD / 16; ++ib) {
        const float * src = s_in + (int64_t) (16 * nb + (lane & 15)) * KD + 16 * ib + 8 * (lane >> 4);
        const float4 lo = *(const float4 *) src;
        const float4 hi = *(const float4 *) (src + 4);
        st[ib][0] = lo.x; st[ib][1] = lo.y; st[ib][2] = lo.z; st[ib][3] = lo.w;
        st[ib][4] = hi.x; st[ib][5] = hi.y; st[ib][6] = hi.z; st[ib][7] = hi.w;
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += C) {
        const int nt = min(C, (int) n_tokens - t0);

        // ---- stage k, q as bf16, k also transposed, plus the per-token gate and beta
        for (int v = tid; v < C * (KD / 4); v += NTH) {
            const int t  = v / (KD / 4);
            const int i4 = (v % (KD / 4)) * 4;
            float4 kk = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 qq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t < nt) {
                kk = *(const float4 *) (kbase + (int64_t) (t0 + t) * sq2 + i4);
                qq = *(const float4 *) (qbase + (int64_t) (t0 + t) * sq2 + i4);
            }
            __bf16 * krow = s_k + t * KP + i4;
            __bf16 * qrow = s_q + t * KP + i4;
            krow[0] = (__bf16) kk.x; krow[1] = (__bf16) kk.y; krow[2] = (__bf16) kk.z; krow[3] = (__bf16) kk.w;
            qrow[0] = (__bf16) qq.x; qrow[1] = (__bf16) qq.y; qrow[2] = (__bf16) qq.z; qrow[3] = (__bf16) qq.w;
            s_kt[(i4 + 0) * KT + t] = (__bf16) kk.x;
            s_kt[(i4 + 1) * KT + t] = (__bf16) kk.y;
            s_kt[(i4 + 2) * KT + t] = (__bf16) kk.z;
            s_kt[(i4 + 3) * KT + t] = (__bf16) kk.w;
        }
        if (tid < C) {
            s_g[tid] = tid < nt ? gbase[(int64_t) (t0 + tid) * sb2] : 0.0f;
            s_b[tid] = tid < nt ? bbase[(int64_t) (t0 + tid) * sb2] : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            float a = 0.0f;
            for (int t = 0; t < C; ++t) {
                a += s_g[t];
                s_c[t] = a;
                s_a[t] = expf(a);
            }
        }

        // ---- gram = K K^T and qk = Q K^T, one 16x16 tile per warp, no cross-warp reduction
        {
            const int  job   = warp;
            const bool is_qk = job >= 4;
            const int  tile  = job & 3;
            const int  tb    = tile >> 1;
            const int  sb    = tile & 1;
            const __bf16 * A = is_qk ? s_q : s_k;
            gdn_f32x8 acc0, acc1;
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
#pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                const gdn_bf16x8 a = gdn_ld_frag(A,    KP, 16 * tb, 16 * kb, lane);
                const gdn_bf16x8 b = gdn_ld_frag(s_k,  KP, 16 * sb, 16 * kb, lane);
                if (kb < 4) { acc0 = gdn_wmma_bf16(a, b, acc0); }
                else        { acc1 = gdn_wmma_bf16(a, b, acc1); }
            }
            const int r0 = 16 * tb + 8 * (lane >> 4);
            const int c0 = 16 * sb + (lane & 15);
            if (is_qk) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { s_qk[(r0 + j) * CB + c0] = (__bf16) (acc0[j] + acc1[j]); }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) { s_gr[(r0 + j) * CP + c0] = acc0[j] + acc1[j]; }
            }
        }
        __syncthreads();

        // ---- k.S0 and q.S0. The B operand is the state itself, converted in place.
        gdn_f32x8 ks0[2], qs0[2];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { ks0[mb][i] = 0.0f; qs0[mb][i] = 0.0f; }
        }
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            const gdn_bf16x8 b = gdn_to_bf16(st[kb]);
#pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                ks0[mb] = gdn_wmma_bf16(gdn_ld_frag(s_k, KP, 16 * mb, 16 * kb, lane), b, ks0[mb]);
                qs0[mb] = gdn_wmma_bf16(gdn_ld_frag(s_q, KP, 16 * mb, 16 * kb, lane), b, qs0[mb]);
            }
        }

        // ---- M = I + strict_lower(beta_t (A_t/A_s) K K^T) in s_gr, then X = M^-1 in s_x.
        // The sweep over column j needs M[i][m] while writing X[i][j], so M and X cannot share
        // storage: s_q is dead after q.S0 and serves as the second buffer.
        __syncthreads();
        if (warp == 0) {
            const int j = lane;
#pragma unroll 1
            for (int i = 0; i < C; ++i) {
                const float gv = s_gr[i * CP + j];
                s_gr[i * CP + j] = (i == j) ? 1.0f
                                 : (j < i) ? s_b[i] * expf(s_c[i] - s_c[j]) * gv
                                           : 0.0f;
            }
            __syncwarp();
#pragma unroll 1
            for (int i = 0; i < C; ++i) {
                if (i < j) {
                    s_x[i * CP + j] = 0.0f;
                } else if (i == j) {
                    s_x[i * CP + j] = 1.0f;
                } else {
                    float acc = 0.0f;
#pragma unroll 1
                    for (int m = j; m < i; ++m) {
                        acc += s_gr[i * CP + m] * s_x[m * CP + j];
                    }
                    s_x[i * CP + j] = -acc;
                }
            }
        }
        __syncthreads();

        // ---- RHS = beta (v - A_t k.S0) and the bf16 inverse, both layout-compatible with B
        gdn_f32x8 rhs[2];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int t = 16 * mb + 8 * (lane >> 4) + j;
                const float vv = t < nt
                    ? vbase[(int64_t) (t0 + t) * sv2 + 16 * nb + (lane & 15)]
                    : 0.0f;
                rhs[mb][j] = s_b[t] * (vv - s_a[t] * ks0[mb][j]);
            }
        }
        for (int idx = tid; idx < C * C; idx += NTH) {
            const int r = idx / C;
            const int c = idx % C;
            s_inv[r * CB + c] = (__bf16) s_x[r * CP + c];
            // qkw is the causal lower triangle of (A_t/A_s) Q K^T
            s_qw[r * CB + c] = (c <= r) ? (__bf16) (expf(s_c[r] - s_c[c]) * (float) s_qk[r * CB + c])
                                        : (__bf16) 0.0f;
        }
        __syncthreads();

        // ---- D = (I + L)^-1 RHS
        gdn_f32x8 d[2];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { d[mb][i] = 0.0f; }
        }
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const gdn_bf16x8 b = gdn_to_bf16(rhs[kb]);
#pragma unroll
            for (int mb = 0; mb < 2; ++mb) {
                d[mb] = gdn_wmma_bf16(gdn_ld_frag(s_inv, CB, 16 * mb, 16 * kb, lane), b, d[mb]);
            }
        }

        // ---- o_t = scale (A_t (q_t . S0) + sum_{s<=t} (A_t/A_s)(q_t . k_s) D_s)
        // ---- and S0 <- A_C S0 + sum_s (A_C/A_s) k_s (x) D_s, straight into the accumulators
        const float a_last = s_a[C - 1];
        const float c_last = s_c[C - 1];
#pragma unroll
        for (int ib = 0; ib < KD / 16; ++ib) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { st[ib][i] *= a_last; }
        }
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            gdn_bf16x8 dw;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int s = 16 * kb + 8 * (lane >> 4) + j;
                dw[j] = (__bf16) (d[kb][j] * expf(c_last - s_c[s]));
            }
#pragma unroll
            for (int ib = 0; ib < KD / 16; ++ib) {
                st[ib] = gdn_wmma_bf16(gdn_ld_frag(s_kt, KT, 16 * ib, 16 * kb, lane), dw, st[ib]);
            }
        }
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            gdn_f32x8 o;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int t = 16 * mb + 8 * (lane >> 4) + j;
                o[j] = s_a[t] * qs0[mb][j];
            }
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
                o = gdn_wmma_bf16(gdn_ld_frag(s_qw, CB, 16 * mb, 16 * kb, lane), gdn_to_bf16(d[kb]), o);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int t = 16 * mb + 8 * (lane >> 4) + j;
                if (t < nt) {
                    dst_d[(seq * n_tokens * H + h) * KD + (int64_t) (t0 + t) * KD * H +
                          16 * nb + (lane & 15)] = o[j] * scale;
                }
            }
        }
        __syncthreads();
    }

    // ---- accumulators -> state
    {
        const int j = 16 * nb + (lane & 15);
        float * out = state_d + seq * H * KD * KD + h * KD * KD + (int64_t) j * KD + 8 * (lane >> 4);
#pragma unroll
        for (int ib = 0; ib < KD / 16; ++ib) {
            float * p = out + 16 * ib;
            *(float4 *) p       = make_float4(st[ib][0], st[ib][1], st[ib][2], st[ib][3]);
            *(float4 *) (p + 4) = make_float4(st[ib][4], st[ib][5], st[ib][6], st[ib][7]);
        }
    }
}

static void launch_gated_delta_net_chunked_wmma(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, float scale, cudaStream_t stream) {
    constexpr int C  = 32;
    constexpr int KD = 128;
    constexpr int KP = KD + 4, KT = C + 4, CP = C + 4, CB = C + 4;
    const size_t smem = (size_t) (2 * C * KP + KD * KT) * sizeof(__bf16) +
                        (size_t) (C * CP + 4 * C) * sizeof(float) +
                        (size_t) (3 * C * CB) * sizeof(__bf16);
    const dim3 grid((unsigned) H, (unsigned) n_seqs, 1);
    const dim3 block(256);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, smem, stream);
    ggml_cuda_kernel_launch(gdn_chunked_wmma_cuda<C>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, scale);
}
#endif // defined(GGML_USE_HIP)

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else if (gdn_chunked_enabled(S_v, H, n_tokens, n_seqs, neq0, nek0, neq3, nek3, nev3, kda, keep_rs)) {
#if defined(GGML_USE_HIP)
            launch_gated_delta_net_chunked_wmma(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, scale, stream);
#else
            GGML_ABORT("fatal error");
#endif
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
