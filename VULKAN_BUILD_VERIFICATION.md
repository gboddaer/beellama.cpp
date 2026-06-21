# Vulkan Build Verification for PR #79

**Commit verified:** `b788b4af1` — `dflash: enable Qwen3-Coder-Next on Vulkan`

## Build System

| Component | Detail |
|---|---|
| **OS** | Debian 13 (trixie), Linux 6.12.90-amd64 |
| **CPU** | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (16C/32T, up to 5185 MHz) |
| **GPU** | AMD Strix Halo Radeon 8060S (RADV GFX1151, Mesa 25.0.7) |
| **RAM** | 30 GiB |
| **CMake** | 3.31.6 |
| **Compiler** | GCC 14.2.0 (`-march=native`) |
| **Make** | GNU Make 4.4.1, 32 parallel jobs |
| **Vulkan SDK** | 1.4.309, glslc (shaderc 2025.2, glslang 15.1.0) |
| **ccache** | Enabled (found automatically) |

## CMake Configuration

```
cmake -B build_vulkan -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
```

Key flags:
- `GGML_VULKAN=ON` — Vulkan backend enabled
- `GGML_NATIVE=ON` — CPU backend compiled with `-march=native`
- `GGML_CPU=ON` — CPU backend included (cooperative with Vulkan)
- `CMAKE_BUILD_TYPE=Release`

Vulkan shader features detected:
- `GL_KHR_cooperative_matrix` — supported
- `GL_NV_cooperative_matrix2` — supported
- `GL_EXT_integer_dot_product` — supported
- `GL_EXT_bfloat16` — not supported (driver/hardware limitation)

## Build Result

- **Configure:** Success (rc=0, 1.8s)
- **Build:** Success (100%, 0 warnings treated as errors)
- **Binaries produced:** `llama-server`, `llama-cli`, `llama-bench`, `llama`, `llama-perplexity`, `llama-imatrix`, `llama-quantize`, `llama-llama-bench`, all test binaries
- **Working tree:** Clean, nothing to commit

## Conclusion

PR #79 builds cleanly for Vulkan on native AMD hardware (Strix Halo / Radeon 8060S) with Mesa RADV driver. No compilation errors or warnings.
