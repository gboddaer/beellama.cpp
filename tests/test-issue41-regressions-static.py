#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def slice_between(text: str, begin: str, end: str) -> str:
    b = text.find(begin)
    require(b != -1, f"missing start marker: {begin}")
    e = text.find(end, b)
    require(e != -1, f"missing end marker: {end}")
    return text[b:e]


def main() -> None:
    fattn = read("ggml/src/ggml-cuda/fattn.cu")
    llama_context = read("src/llama-context.cpp")
    set_rows = read("ggml/src/ggml-cuda/set-rows.cu")
    build_docs = read("docs/build.md")
    readme = read("README.md")
    convert = read("convert_hf_to_gguf.py")
    conversion_init = read("conversion/__init__.py")

    exec_body = slice_between(
        fattn,
        "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst)",
        "bool ggml_cuda_flash_attn_ext_supported",
    )
    require(
        "ggml_cuda_fattn_pair_compiled(K->type, V->type)" not in exec_body,
        "CUDA FA execution must not reject raw Turbo K/V pairs before route planning",
    )
    require(
        "ggml_cuda_fattn_pair_compiled(plan.effective_type_K, plan.effective_type_V)" in exec_body,
        "CUDA FA execution must validate the route plan's effective K/V pair",
    )
    require(
        exec_body.find("ggml_cuda_fattn_make_route_plan") < exec_body.find("ggml_cuda_fattn_pair_compiled(plan.effective_type_K, plan.effective_type_V)"),
        "CUDA FA execution must build the route plan before checking the effective pair",
    )
    require(
        "GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS" in fattn and
        "GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS" in llama_context,
        "CUDA FA must expose one env gate for ignoring uncompiled pair checks",
    )
    require(
        "llama_cuda_fa_find_uncompiled_pair" in llama_context and
        "llama_cuda_fa_find_uncompiled_pair(gf, model)" in llama_context,
        "CUDA FA startup check must scan requested graph pairs directly before scheduler fallback can hide them",
    )
    require(
        "allow_vec = allow_vec && pair_compiled" in fattn,
        "ignored uncompiled CUDA FA pairs must not select missing vec dispatch cases",
    )
    require(
        "WARNING: GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1" in fattn and
        "WARNING: GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1" in llama_context,
        "ignored uncompiled CUDA FA pairs must emit warnings before continuing",
    )
    require(
        "GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1" in build_docs and
        "GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1" in readme,
        "CUDA FA uncompiled-pair ignore env must be documented",
    )
    require(
        "is not compiled in this build. " in llama_context and
        "Default builds compile standard q/KVarN-fallback pairs only (no TurboQuant/TCQ), " in llama_context and
        "K must be the same or higher precision than V, and V may be no more than two tier groups below K. " in llama_context and
        "Use a default compiled pair, rebuild with GGML_CUDA_FA_HALF_QUANTS=ON for TurboQuant/TCQ or wider K>=V, " in llama_context,
        "default-build missing-pair diagnostic must match the concise policy text",
    )
    require(
        "HALF builds compile same-or-higher ranked K than V, with TurboQuant/TCQ is treated as pseudo-equal to qX_1. " in llama_context and
        "Use a HALF compiled pair or rebuild with GGML_CUDA_FA_ALL_QUANTS=ON for the full matrix." in llama_context,
        "HALF-build missing-pair diagnostic must match the concise policy text",
    )

    require(
        "uint8_t * new_buf = nullptr;" in set_rows and
        "CUDA_CHECK(cudaMalloc(&new_buf, bytes_needed));" in set_rows and
        "tcq_bt_buf[device] = new_buf;" in set_rows,
        "TCQ backtrace allocation must publish a new buffer only after cudaMalloc succeeds",
    )
    require(
        "CUDA_CHECK(cudaMalloc(&tcq_dump_x_dev" in set_rows and
        "CUDA_CHECK(cudaMalloc(&tcq_dump_out_dev" in set_rows,
        "TCQ diagnostic dump allocations must be checked",
    )
    require(
        set_rows.count("ggml_cuda_kernel_launch(k_set_rows_turbo") == 5,
        "Turbo/TCQ set_rows kernels must launch through the checked CUDA wrapper",
    )
    require(
        "k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size" not in set_rows,
        "quantized set_rows helper must launch through the checked CUDA wrapper",
    )

    require(
        "supports_mmproj_model" in conversion_init,
        "conversion package must expose a way to detect supported mmproj companions",
    )
    require(
        "supports_mmproj_model(model_architecture)" in convert and
        "Use --mmproj to export the multimodal projector" in convert,
        "converter must warn when text-converting a multimodal config with a supported mmproj companion",
    )


if __name__ == "__main__":
    main()
