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
    set_rows = read("ggml/src/ggml-cuda/set-rows.cu")
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
