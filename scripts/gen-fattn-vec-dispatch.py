#!/usr/bin/env python3
# Generates FATTN_VEC_CASES_ALL_D_512 dispatch lines for ggml_cuda_flash_attn_ext_vec().
# Usage: python scripts/gen-fattn-vec-dispatch.py

TYPES = [
    ("GGML_TYPE_F16",        "f16"),
    ("GGML_TYPE_BF16",       "bf16"),
    ("GGML_TYPE_Q8_0",       "q8_0"),
    ("GGML_TYPE_Q6_0",       "q6_0"),
    ("GGML_TYPE_Q5_1",       "q5_1"),
    ("GGML_TYPE_Q5_0",       "q5_0"),
    ("GGML_TYPE_TURBO4_0",   "turbo4_0"),
    ("GGML_TYPE_Q4_1",       "q4_1"),
    ("GGML_TYPE_Q4_0",       "q4_0"),
    ("GGML_TYPE_TURBO3_TCQ", "turbo3_tcq"),
    ("GGML_TYPE_TURBO3_0",   "turbo3_0"),
    ("GGML_TYPE_TURBO2_TCQ", "turbo2_tcq"),
    ("GGML_TYPE_TURBO2_0",   "turbo2_0"),
]

DEFAULT_PAIRS = [
    ("GGML_TYPE_F16",  "GGML_TYPE_F16"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_BF16", "GGML_TYPE_BF16"),
    ("GGML_TYPE_TURBO2_0",   "GGML_TYPE_TURBO2_0"),
    ("GGML_TYPE_TURBO3_0",   "GGML_TYPE_TURBO3_0"),
    ("GGML_TYPE_TURBO4_0",   "GGML_TYPE_TURBO4_0"),
    ("GGML_TYPE_TURBO2_0",   "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_TURBO3_0",   "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_TURBO4_0",   "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_Q8_0",       "GGML_TYPE_TURBO2_0"),
    ("GGML_TYPE_Q8_0",       "GGML_TYPE_TURBO3_0"),
    ("GGML_TYPE_Q8_0",       "GGML_TYPE_TURBO4_0"),
    ("GGML_TYPE_TURBO4_0",   "GGML_TYPE_TURBO3_0"),
    ("GGML_TYPE_TURBO3_0",   "GGML_TYPE_TURBO4_0"),
    ("GGML_TYPE_TURBO2_0",   "GGML_TYPE_TURBO3_0"),
    ("GGML_TYPE_TURBO3_0",   "GGML_TYPE_TURBO2_0"),
    ("GGML_TYPE_TURBO3_TCQ", "GGML_TYPE_TURBO3_TCQ"),
    ("GGML_TYPE_TURBO2_TCQ", "GGML_TYPE_TURBO2_TCQ"),
    ("GGML_TYPE_TURBO3_TCQ", "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_TURBO2_TCQ", "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_Q8_0",       "GGML_TYPE_TURBO3_TCQ"),
    ("GGML_TYPE_Q8_0",       "GGML_TYPE_TURBO2_TCQ"),
    ("GGML_TYPE_TURBO3_TCQ", "GGML_TYPE_TURBO2_TCQ"),
    ("GGML_TYPE_TURBO2_TCQ", "GGML_TYPE_TURBO3_TCQ"),
    ("GGML_TYPE_TURBO4_0",   "GGML_TYPE_TURBO3_TCQ"),
    ("GGML_TYPE_TURBO3_0",   "GGML_TYPE_TURBO3_TCQ"),
]

def emit_pair(k, v):
    print(f"    FATTN_VEC_CASES_ALL_D_512({k}, {v})")

print("#if defined(GGML_CUDA_FA_ALL_QUANTS)")
for k, _ in TYPES:
    for v, _ in TYPES:
        emit_pair(k, v)

print("#elif defined(GGML_CUDA_FA_HALF_QUANTS)")
for ki, (k, _) in enumerate(TYPES):
    for vi, (v, _) in enumerate(TYPES):
        if ki <= vi:
            emit_pair(k, v)

print("#else")
for k, v in DEFAULT_PAIRS:
    emit_pair(k, v)

print("#endif")