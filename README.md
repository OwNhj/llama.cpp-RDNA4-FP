# llama.cpp

> **This fork** is based on upstream `master` and adds RDNA4 (gfx12) FP8 support on top of it:
> an **F8 (E4M3) KV cache** with its own flash-attention kernels, **F8 WMMA** compute, and
> **MXFP8** weight quantization.
>
> **MXFP4** weights run through FP8 compute instead of the int8 WMMA path: their e2m1 values are
> expanded to e4m3 exactly (every e2m1 magnitude is representable in e4m3), and the prefill GEMM
> then uses the FP8 WMMA path with the E8M0 scales stored raw.
>
> **NVFP4 stays on the int8 WMMA path.** It keeps one UE4M3 scale per 16 elements rather than
> MXFP4's one per 32, so its FP8 k-tile has to be 4 ints wide where MXFP4's is 8. The WMMA count
> is the same either way (8 trips x 1 call vs 4 trips x 2), so the narrower tile only buys extra
> `ldmatrix` loads and loop overhead: a kernel profile measured 165.6 ms for the NVFP4 prefill
> kernel against 90.3 ms for MXFP4's, with every other kernel matching. The int8 path uses the
> same 4-int tile and the same 16-element scale, and it also clears four `test-backend-ops`
> MUL_MAT failures the FP8 path had. It is about 9% slower on prefill than the FP8 path was
> (20.3k vs 22.3k t/s at `-p 2048`), but it is correct on every case, so correctness wins here.
> Note that the two halves of this decision have to move together: NVFP4 is excluded from the
> RDNA4 e4m3-y branch in `mmq.cu` because an int8 vec_dot must consume q8_1 int8 y, not e4m3.
>
> It also adds **`Q4_0_ROCMI4`**: a signed-nibble 4-bit format that runs as **W4A4** on the RDNA4
> `v_wmma_i32_16x16x32_iu4` tensor core, ported from [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
> and re-adapted to this tree's MMQ. See [RDNA4 I4 W4A4](#rdna4-i4-w4a4) below.
>
> ### Building with the F8 KV cache
>
> The F8 flash-attention K-V combinations are **not** in the default `GGML_CUDA_FA_QUANTS` list and
> have to be enabled explicitly. Without them `-ctk f8` still works, but token generation falls back
> to converting K and V to f16 before running the f16 vector kernel (slower, with a warning); the
> prefill path is unaffected either way, since it dequantizes F8 to f16 and uses the f16 matrix
> kernel:
>
> ```bash
> cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
>       -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f8-f8;f8-f16;f16-f8"
> cmake --build build -j
> ```
>
> Then enable it at runtime with `-ctk f8 -ctv f8`:
>
> ```bash
> ./build/bin/llama-cli       -m model.gguf -ngl 99 -ctk f8 -ctv f8 -p "Hello"
> ./build/bin/llama-server    -m model.gguf -ngl 99 -ctk f8 -ctv f8
> ./build/bin/llama-perplexity -m model.gguf -ngl 99 -ctk f8 -ctv f8 -f wiki.test.raw
> ```
>
> MXFP8 weights are produced with `./build/bin/llama-quantize in.gguf out.gguf MXFP8`, or directly
> from a Hugging Face model with `python3 convert_hf_to_gguf.py --outtype mxfp8 --outfile out.gguf <model_dir>`.
>

## RDNA4 I4 W4A4 (weight 4-bit x activation 4-bit)

`Q4_0_ROCMI4` is a signed-nibble 4-bit format with a one-byte UE4M3 block scale
(32 elements per block, 4.25 bpw). On RDNA4 it has two compute paths:

* **W4A4 (prefill)** — weights stay packed as nibbles in LDS and are consumed by the native
  `v_wmma_i32_16x16x32_iu4` tensor-core instruction (K = 32 in a single op). Activations are
  quantized onto a signed 4-bit grid. Requires `-DGGML_HIP_ROCMI4_W4A4=ON`.
* **W8A8 (decode)** — the MMVQ path decodes nibbles to int8 and pairs them with q8_1
  activations, so decode keeps 8-bit activation precision. This is the default and is not
  affected by the build option above: at batch sizes up to 8 the activations stay q8_1, and
  only larger batches take the W4A4 path.

Enable W4A4 at configure time:

```bash
cmake -B build-gpu -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP_ROCMI4_W4A4=ON
cmake --build build-gpu -j
```

Quantize with the `Q4_0_ROCMI4` ftype; it reuses the Q4_0 per-tensor assignment, so the same
tensors are promoted to Q6_K and the resulting file size matches Q4_0's layout:

```bash
./build-gpu/bin/llama-quantize model-bf16.gguf model-rocmi4.gguf Q4_0_ROCMI4
```

**W4A4 is deliberately lossy on the activation side.** Prefill quantizes activations to a
4-bit grid (16 uniform levels) instead of the q8_1 used everywhere else, so a file built with
W4A4 enabled scores worse than the same file built with `-DGGML_HIP_ROCMI4_W4A4=OFF`. Measured
on Qwen3.8-27B, single R9700, `-c 512`:

| file | PPL |
|---|---|
| `Q4_0` (reference) | 3.7755 |
| `Q4_0_ROCMI4`, W4A4 off | 4.6937 |
| `Q4_0_ROCMI4`, W4A4 on | 5.4493 |

The same pattern shows up in decode without any build-flag change, because that is the same
trade-off reached from the other side: forcing a batch size of 1 puts prefill through the
q8_1 activation path and the same file then measures 4.7064.

For prefill throughput on a 1B dense model, single R9700, `-p 2048 -r 5`: `Q4_0` 25 049 t/s
against `Q4_0_ROCMI4` with W4A4 on at 29 588 t/s, i.e. about +18%, and roughly +14.5% at the
tile configuration shipped before the final one. These are dense-transformer numbers; the
27B figures measured later mix in SSM layers and are not directly comparable.

**Do not quantize `ssm_out` to `Q4_0_ROCMI4`.** (The same applies to `Q4_0_SYM4`; see the section below.) On hybrid SSM/attention architectures such as
Qwen3.8-27B, `ssm_out.weight` must be left at another type. Putting it in ROCMI4 drives
perplexity to ~4.3e5 while every other tensor in the same file stays correct, and it does so
identically on CPU and on GPU, so it is not a kernel defect. Pin those tensors to Q8_0 or
MXFP4 instead:

```bash
# one line per layer: blk.N.ssm_out.weight=q8_0
./build-gpu/bin/llama-quantize --tensor-type-file ssm_out.txt \
    model-bf16.gguf model-rocmi4.gguf Q4_0_ROCMI4
```

Both workarounds were measured on Qwen3.8-27B, single R9700, `-c 512`:

| `ssm_out` type | bpw | size | PPL |
|---|---|---|---|
| `Q4_0_ROCMI4` | 4.25 | 13.88 GiB | 429951 (broken) |
| `MXFP4` | 4.36 | 13.87 GiB | 4.3262 |
| `Q8_0` | 4.60 | 14.62 GiB | 4.4551 |

The damage scales with how many `ssm_out` tensors stay in ROCMI4 rather than coming from one
bad tensor, which fits the tensor feeding the SSM/gated-delta recurrence: a small per-layer
error enters a state carried across layers and tokens and compounds instead of washing out.
Pinning the first 24 of 48 layers to MXFP4 gives 274.2766, all 48 left in ROCMI4 gives
429951, and all 48 pinned gives 4.3262.

The root cause is not yet understood: the stored
`ssm_out` values are no less accurate than MXFP4's (NMSE 1.08e-2 against 1.31e-2 on the bf16
source), the standalone dot product is exact, and the failure reproduces bit-for-bit across
backends that use different activation precisions. Treat it as a known limitation of the
current ROCMI4 support rather than a property of the format.

## Q4_0_SYM4 — experimental symmetric 4-bit format

**`Q4_0_SYM4` is experimental. It is not a general-purpose replacement for `Q4_0_ROCMI4`, and
like ROCMI4 it must not be used for `ssm_out` tensors.** See the tuning rules below: used
indiscriminately it is *worse* than ROCMI4 on the metric that matters most, and only a
per-tensor assignment makes it a clear win.

SYM4 keeps ROCMI4's exact 17-byte block (32 elements, one UE4M3 scale, 4.25 bpw) and changes
only the code grid:

| format | decoded value | scale | reachable codes | step |
|---|---|---|---|---|
| `Q4_0_ROCMI4` | `n * s` | `nearest_ue4m3(amax/7)` | 15 of 16 | `amax/7` |
| `Q4_0_SYM4` | `(n + 0.5) * s` | `nearest_ue4m3(amax/7.5)` | **16 of 16** | **`amax/7.5`** |

Because `|x| <= amax = 7*s`, ROCMI4's code `n = -8` can never be selected — one of its 16 codes
is wasted. SYM4 shifts the grid by half a step so that all 16 codes are reachable and the grid
is exactly symmetric, which makes the step 6.67% finer and the rounding MSE about 12.9% lower.

The cost is that 16 is an even count, so a symmetric grid **cannot contain zero**: elements
below half a step are forced onto `±s/2` instead of being represented exactly. On most weight
tensors the finer step more than pays for this, but not on all of them — hence the per-tensor
rules.

### Compute paths

Both paths exist, mirroring ROCMI4:

* **W8A8 (decode / default)** — the grid offset is folded into the data rather than corrected
  afterwards. Since `(n + 0.5) * s == (2n + 1) * (s/2)` and `2n + 1` lies in `[-15, 15]`, it
  still fits an int8 operand, so the loader emits `2n + 1` and halves the scale and ROCMI4's
  int8 kernels apply unchanged. The per-byte transform is `((v & 0x7F7F7F7F) << 1) | 0x01010101`.
* **W4A4 (prefill, `-DGGML_HIP_ROCMI4_W4A4=ON`)** — the `iu4` tensor-core instruction reads the
  LDS row as packed 4-bit nibbles, so the `2n + 1` trick is impossible (`2n + 1` needs five
  bits). The nibble stays raw and the offset is corrected in the vec_dot epilogue:
  `sum_i (n_i + 0.5)*sx*(m_i*dB) = sx*dB*sum_i(n_i*m_i) + 0.5*sx*dB*sum_i m_i`.
  The activation packer emits `sum_i m_i` into y-row ints 20..23, which ROCMI4's W4A4 vec_dot
  never reads, so the two formats share one activation buffer layout.

Quantize with the `Q4_0_SYM4` ftype. The file size is **byte-identical** to the ROCMI4 file for
the same per-tensor assignment, because the block layout is the same.

### Measured on Qwen3.8-27B (single R9700, `-c 512`)

KL divergence against the bf16 model, with `ssm_out` pinned to `Q8_0` in every configuration:

| assignment | Mean KLD | 99% KLD | 99% Δp | RMS Δp | Same top-p |
|---|---|---|---|---|---|
| all ROCMI4 | 0.3200 | 7.604 | 37.26% | 14.66% | 86.42% |
| all SYM4 | 0.2932 | 7.683 | **80.17%** | 14.38% | 88.87% |
| **tuned SYM4 (below)** | **0.2173** | **4.605** | **30.61%** | **11.13%** | **89.71%** |

Read this table carefully: **all-SYM4 improves the mean but destroys the tail.** Its 99% Δp is
80.17% against ROCMI4's 37.26%, i.e. the worst 1% of tokens are pushed more than twice as far.
The tuned assignment fixes that and beats ROCMI4 on every column at once.

### Tuning rules

Enable SYM4 only for these tensor classes:

| tensor class | tensors | why |
|---|---|---|
| `token_embd` | 1 | by far the most important single tensor. Swapping just this one from ROCMI4 to SYM4 moves 99% Δp from 63.00% to 35.36% |
| `ffn_down` | 65 | the only FFN class that improves the tail: 99% Δp 35.36% → 30.63%, Max KLD 18.77 → 16.30 |
| `attn_output` | 17 | best single-class 99% KLD (5.742) and RMS Δp (12.245%) |
| `attn_gate` | 48 | best single-class Mean KLD (0.2522) and Same top-p (89.41%) |
| `attn_k` | 17 | good tail: 99% Δp 32.94% |
| `attn_q` | 17 | good tail: 99% Δp 31.59% |

Keep ROCMI4 for these:

| tensor class | tensors | evidence |
|---|---|---|
| `attn_v` | 17 | adding it moves 99% Δp from 30.61% to **73.20%** |
| `attn_qkv` | 48 | adding it moves 99% Δp from 73.20% to **76.65%** |
| `ffn_gate` | 65 | adding it moves 99% Δp from 30.63% to **42.01%** |
| `ffn_up` | 65 | adding it moves 99% Δp from 42.01% to 36.82%, still short of the 30.63% without it |

The four harmful classes are individually small (`attn_v` is 45 MiB, `ffn_gate`/`ffn_up` 2.9 GiB
each) yet each one degrades the tail, so the rule is not "use SYM4 where it is large" — it has
to be applied class by class. `attn_v` and `attn_qkv` are the sharpest examples: they are the
smallest classes in the set, and they are the ones that break the tail worst.

Reproduce the tuned assignment:

```bash
cat > sym4.txt <<'EOF'
ssm_out=q8_0
\.ffn_down\.weight=q4_0_sym4
\.attn_output\.weight=q4_0_sym4
\.attn_gate\.weight=q4_0_sym4
\.attn_k\.weight=q4_0_sym4
\.attn_q\.weight=q4_0_sym4
EOF

./build-gpu/bin/llama-quantize --tensor-type-file sym4.txt \
    --token-embedding-type q4_0_sym4 \
    model-bf16.gguf model-sym4.gguf Q4_0_ROCMI4
```

Note the two mechanisms at work: `--token-embedding-type` sets `token_embd`, while
`--tensor-type-file` uses regex search against the tensor name, so the patterns above are
anchored on `.weight` to avoid also matching `attn_q_norm`, `attn_k_norm`, and the like.
`ssm_out` must stay pinned to `Q8_0` (or MXFP4) exactly as for ROCMI4.

### imatrix support

`--imatrix` works for `Q4_0_SYM4`, matching `Q4_0_ROCMI4`. SYM4 has one free parameter per
block (the UE4M3 scale) and the codes follow by rounding, so the weighted path searches for the
scale that minimises the importance-weighted error instead of taking `amax/7.5`. Two constants
differ from the ROCMI4 variant because SYM4's grid tops out at `7.5*s`: the search starts at
`amax/7.5`, and the early-exit clipping bound uses 7.5.

**A full imatrix is not necessarily the best choice, so tune it against KL rather than assuming
it helps.** On Qwen3.8-27B the weighted path moved error from the typical token to the worst
one: the average-weighted metrics all improved (Mean KLD −4.7%, 95% KLD −11.3%, 95% Δp −12.8%)
while the 99th-percentile metrics got worse (99% Δp +10.8%, 99% KLD +3.8%). Which side of that
trade matters depends on the use. Under llama.cpp's speculative decoding the accept test is
whether the target's sampled token equals the draft's proposal, so what counts is agreement on
the top of the distribution — `Same top-p` and `99% Δp` — rather than `Mean KLD`, which averages
over the whole 248k-token vocabulary. For a draft model the 99% figures are the ones to watch.

No single tensor class causes the tail cost, and masking one does not recover it: removing the
imatrix entries for a class makes that tensor fall back to the unweighted quantiser (the loader
leaves the pointer NULL when a name is missing), and doing so for `attn_v` + `attn_qkv`,
`ffn_down`, `ffn_gate`, or `ffn_up` each made 99% Δp *worse* than the full imatrix. The
tail cost is a property of the weighted objective, not of any one class. Note also that
`ffn_up` scored the best `Same top-p` while having a poor 99% Δp, so those two metrics can
disagree — measure both, and do not trust differences below about 1pp, since these were single
runs whose spread has not been characterised.

### Cost

None measurable. On the tuned file, single R9700, `-p 512 -n 128 -r 3`:

| file | size | pp512 | tg128 |
|---|---|---|---|
| all ROCMI4 | 14.62 GiB | 1370 ± 401 | 28.52 ± 0.15 |
| tuned SYM4 | 14.62 GiB | 1338 ± 397 | 28.53 ± 0.18 |

Same size (byte-identical), same decode throughput; the prefill difference is well inside the
run-to-run spread. The gain is therefore pure KL improvement at zero cost.

### Status

The tuned assignment above was measured on Qwen3.8-27B only. The direction of each class was
consistent across repeated runs, but the harmful-class result (`attn_v`, `attn_qkv`) is not yet
explained mechanically — both feed the attention value path, yet `attn_k` and `attn_q` from the
same projection group are beneficial. Treat the class list as an empirical starting point to
re-derive per model, not as a universal rule, and re-measure the tail (99% Δp) rather than
trusting mean KL alone: as the all-SYM4 row shows, a format can win on the mean and still lose
badly on the tokens that matter.

## Attribution

The I4 W4A4 path and the `Q4_0_ROCMI4` format are ported from
[ROCmFPX](https://github.com/charlie12345/ROCmFPX), which targets RDNA3.5
(gfx1151). Porting it into this tree meant re-adapting it to the current upstream MMQ
generation: ROCmFPX is written against an older MMQ built on `mmq_type_traits` with a runtime
`ncols_dst`, while upstream now selects on compile-time `ncols_dst` templates and routes
through `calc_nwarps` / `calc_rows_per_block`. The loader, the byte-interleaved nibble
packing, the LDS stride, the `vec_dot` and the per-tile configuration were all re-derived for
that structure. RDNA4 also has the K = 32 form of the instruction, so a single
`v_wmma_i32_16x16x32_iu4` is issued where ROCmFPX calls the K = 16 form twice.

The ROCmFP4 / ROCmFPx family that the ROCmFPX port originally brought along (Q4_0_ROCMFP4,
Q4_0_ROCMFP4_FAST, Q3_0/Q2_0/Q6_0/Q8_0_ROCMFPX) has been removed: none of it is used by the
W4A4 work and its MMVQ paths were incomplete here.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
