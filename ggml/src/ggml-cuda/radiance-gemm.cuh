#pragma once

#include "common.cuh"

// radiance MXFP4 x fp8 W8A8 WMMA GEMM fast path (RDNA4 prefill).
// all entries assume ggml_cuda_radiance_supported() was true and buffers are device-side.
void ggml_cuda_radiance_repack(const void * src_llama, int64_t s01, int N, int K,
                               void * W, void * Ws, void * Wref, cudaStream_t stream);

void ggml_cuda_radiance_quantize_tokens(const float * x, int64_t sx, int64_t K, int64_t M,
                                        unsigned char ** q, float ** scale, cudaStream_t stream);

void ggml_cuda_radiance_gemm_f32(const void * a_q, const void * w, const void * ws, const void * wref,
                                 const float * as, float * c, int M, int N, int K, cudaStream_t stream);

bool ggml_cuda_radiance_supported(int cc, ggml_type type, int64_t ne00, int64_t ne01,
                                  int64_t ne11, int64_t ne10, bool contiguous_dst);

// fused GLU fast path (B1): produce down-GEMM fp8 alongside y. lookup/registry span one graph.
bool ggml_cuda_try_swiglu_quant_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
bool ggml_rad_fused_act_lookup(const void * act, int64_t K,
                               const unsigned char ** q, const float ** scale);
void ggml_rad_fused_acts_reset(void);
