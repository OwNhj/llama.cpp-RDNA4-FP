# Pascal dp4a MMQ optimization (branch: llama.cpp-Pascal)

This branch carries a self-contained performance patch for pre-Volta NVIDIA GPUs
(Pascal, sm_60/sm_61) that use the `__dp4a`-based MMQ path. It is **not** targeted
at upstream; it exists to keep the experiment reproducible.

## The patch (1 commit, ggml/src/ggml-cuda/mmq-vec-dot.cuh)

`MMQ_DP4A_HOIST_VEC_LOADS` (guarded to `__CUDA_ARCH__` in [610, 700), no HIP/MUSA):

- Inside the dp4a `vec_dot` functions the x-tile shared-memory loads depend only on
  `(i, k0)` but were re-issued on every `j` iteration; the y-tile loads depend only on
  `(j, k0)` but were re-issued on every `i` iteration.
- The hoist moves each load out of the loop that repeats it. Accumulation order is
  unchanged, output is bit-identical.
- Rewritten functions: `q4_0`, `q4_1`, and the shared `q8_0` / `q8_1` functions that
  `q5_0` / `q5_1` reuse (q5 high bits are pre-unpacked in `load_tiles`, so `vec_dot`
  never sees them). All other quant types use the original loop.

## Why: where the time goes on a Tesla P4

Ablation (canary-normalized, bs 9..33):

| config                    | q4_0 time saved | q8_0 time saved |
|---------------------------|-----------------|-----------------|
| drop dot compute          | 56..72 %        | 26..36 %        |
| drop y global->smem       | 13..18 %        | 8..19 %         |
| drop x tile reload        | 5..16 %         | 13..23 %        |

Kernel is latency/issue bound: dp4a runs at 14..41 % of the measured 2.5/SM/cycle
peak while bandwidth sits at 32..46 % of the measured 175 GB/s. Smem load issue
rate inside `vec_dot` was the largest addressable component; dead ends ruled out
by A/B: occupancy hint, tile-I change, J=24/40/48/56 tiles, register prefetch of y.

## Measured gain (interleaved A/B, dp4a-canary normalized)

| type | bs 9 | bs 25 | bs 41 | bs 57 | mean |
|------|------|-------|-------|-------|------|
| q4_0 | 1.17x | 1.19x | 1.02x | 1.08x | 1.12x |
| q4_1 | 1.07x | 1.14x | 1.09x | 1.09x | 1.10x |
| q5_0 | 1.10x | 1.15x | 1.08x | 1.09x | 1.10x |
| q5_1 | 1.13x | 1.13x | 1.08x | 1.09x | 1.11x |
| q8_0 | 1.20x | 1.08x | 0.98x | 0.97x | 1.06x |

## When it helps

Dispatch is by `ne11` (tokens per mul_mat): <=8 MMVQ, on Pascal 9+ always MMQ dp4a
(no tensor cores, never falls to cuBLAS). Single-user decode without speculative
decoding runs at bs=1 and sees nothing; the gain window is bs in [9, 64]:
multi-slot servers, draft/speculative verification, batched rerank/embedding.

## Verification

- P4 (sm_61): MUL_MAT 1301/1301, MUL_MAT_ID 931/931 (test-backend-ops)
- A2000 (sm_86): MUL_MAT 1301/1301 (guard compiles out; byte-identical path)
- Output hash equality orig vs patch (q4_0/q8_0/q5_1, bs 17): bit-identical
