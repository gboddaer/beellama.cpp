# Vulkan Build Verification for PR #79

Verifies that PR #79 (`dflash: enable Qwen3-Coder-Next on Vulkan`, commit `b788b4af1`) builds cleanly for the Vulkan backend on two independent host environments.

---

## Verification 1 — AMD Strix Halo (Debian 13 / Mesa RADV)

**Commit verified:** `b788b4af1` — `dflash: enable Qwen3-Coder-Next on Vulkan`

### Build System

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

### Conclusion

PR #79 builds cleanly for Vulkan on native AMD hardware (Strix Halo / Radeon 8060S) with Mesa RADV driver. No compilation errors or warnings.

---

## Verification 2 — NVIDIA RTX 3090 host (Debian 11 / NVIDIA Vulkan ICD)

**Commit verified:** `b788b4af1` — `dflash: enable Qwen3-Coder-Next on Vulkan` (on the updated PR head `4efe54303`)

### Build System

| Component | Detail |
|---|---|
| **OS** | Debian GNU/Linux 11 (bullseye), Linux 5.10.0-43-amd64 |
| **CPU** | AMD Ryzen 9 5950X 16-Core (32 threads, `-march=native`) |
| **GPU** | NVIDIA GeForce RTX 3090 (GA102) — host for the build |
| **RAM** | 62 GiB |
| **CMake** | 3.31.11 |
| **Compiler** | GCC 10.2.1 (`g++ (Debian 10.2.1-6)`) |
| **Make/Ninja** | Ninja 1.10.1, 32 parallel jobs |
| **glslc** | shaderc v2023.2 (built from source, installed to `~/.local/bin/glslc`; not packaged in Debian 11) |
| **Vulkan headers** | 1.4.309 (KhronosGroup `Vulkan-Headers` v1.4.309, header-only, installed to `~/.local`; Debian 11 ships only 1.2.162 which is too old for current `ggml-vulkan.cpp`) |
| **Vulkan loader** | libvulkan 1.2.162 (system `libvulkan-dev`) — sufficient to link; newer symbols are resolved at runtime via the loader/ICD |

### Setup notes for Debian 11

Debian 11 does not ship `glslc`/`shaderc` and its Vulkan headers (1.2.162) predate symbols the current Vulkan backend requires (`vk::PhysicalDeviceMaintenance4Properties`, `vk::DriverId::eMesaTurnip`/`eMesaDozen`, `layer_setting_info`). Two host-side additions were needed, neither touching the PR source:

1. Build and install `glslc` from Shaderc `v2023.2` source → `~/.local/bin/glslc`.
2. Install header-only `Vulkan-Headers` v1.4.309 → `~/.local/include/vulkan`.

Then configure with the local prefix so CMake's `FindVulkan` picks up the new headers:

```bash
export CMAKE_PREFIX_PATH="$HOME/.local:$CMAKE_PREFIX_PATH"
cmake -B build_vulkan -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_INCLUDE_DIR="$HOME/.local/include"
cmake --build build_vulkan -j32
```

### CMake Configuration

```
-- Found Vulkan: /usr/lib/x86_64-linux-gnu/libvulkan.so (found version "1.4.309") found components: glslc missing components: glslangValidator
-- Vulkan found
-- GL_KHR_cooperative_matrix not supported by glslc
-- GL_NV_cooperative_matrix2 not supported by glslc
-- GL_EXT_integer_dot_product not supported by glslc
-- GL_EXT_bfloat16 not supported by glslc
-- Including Vulkan backend
```

(Shader feature probes are `not supported` due to the older `glslc` v2023.2 build; this only disables optional cooperative-matrix fast paths and does not affect compilation.)

### Build Result

- **Configure:** Success (rc=0)
- **ggml-vulkan target:** Success (rc=0, 100%)
- **Full build:** Success (rc=0, 100%)
- **Binaries produced:** `llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, `test-dflash-plumbing` (plus shared libs `libggml-vulkan.so`, `libllama-server-impl.so`, etc.)
- **DFlash plumbing test:** `test-dflash-plumbing` → rc=0
- **Errors:** 0 (zero `error:` lines in the full build log)
- **Warnings:** only benign — 35× `-Wdouble-promotion`, 1× `-Wmissing-field-initializers` (no `-Werror`)

### Conclusion

PR #79 builds cleanly for Vulkan on a Debian 11 / NVIDIA RTX 3090 host once `glslc` and current Vulkan headers are supplied locally (no PR source changes required). Combined with Verification 1, the Vulkan backend compiles end-to-end on both AMD/Mesa RADV (Debian 13) and NVIDIA (Debian 11) hosts.
