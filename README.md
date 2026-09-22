# llama.cpp

> **This fork** is based on upstream `master` and adds RDNA4 (gfx12) FP8 support on top of it:
> an **F8 (E4M3) KV cache** with its own flash-attention kernels, **F8 WMMA** compute, and
> **MXFP8** weight quantization.
>
> MXFP4 / NVFP4 weights also run through FP8 compute instead of the int8 WMMA path: their e2m1
> values are expanded to e4m3 exactly (every e2m1 magnitude is representable in e4m3), and the
> prefill GEMM then uses the FP8 WMMA path with the E8M0 / UE4M3 scales stored raw.
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

**Do not quantize `ssm_out` to `Q4_0_ROCMI4`.** On hybrid SSM/attention architectures such as
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
