#!/usr/bin/env python3
# Generates FATTN_VEC_CASES_ALL_D_512 dispatch lines for ggml_cuda_flash_attn_ext_vec().
# Usage: python scripts/gen-fattn-vec-dispatch.py

TYPES = [
    ("GGML_TYPE_F16",        "f16"),
    ("GGML_TYPE_BF16",       "bf16"),
    ("GGML_TYPE_Q8_0",       "q8_0"),
    ("GGML_TYPE_Q6_1",       "q6_1"),
    ("GGML_TYPE_Q6_0",       "q6_0"),
    ("GGML_TYPE_Q5_1",       "q5_1"),
    ("GGML_TYPE_Q5_0",       "q5_0"),
    ("GGML_TYPE_Q4_1",       "q4_1"),
    ("GGML_TYPE_TURBO4_TCQ", "turbo4_tcq"),
    ("GGML_TYPE_TURBO4_0",   "turbo4_0"),
    ("GGML_TYPE_Q4_0",       "q4_0"),
    ("GGML_TYPE_Q3_1",       "q3_1"),
    ("GGML_TYPE_TURBO3_TCQ", "turbo3_tcq"),
    ("GGML_TYPE_TURBO3_0",   "turbo3_0"),
    ("GGML_TYPE_Q3_0",       "q3_0"),
    ("GGML_TYPE_Q2_1",       "q2_1"),
    ("GGML_TYPE_TURBO2_TCQ", "turbo2_tcq"),
    ("GGML_TYPE_TURBO2_0",   "turbo2_0"),
    ("GGML_TYPE_Q2_0",       "q2_0"),
]

HALF_RANKS = {
    "GGML_TYPE_F16": 0,
    "GGML_TYPE_BF16": 1,
    "GGML_TYPE_Q8_0": 2,
    "GGML_TYPE_Q6_1": 3,
    "GGML_TYPE_Q6_0": 4,
    "GGML_TYPE_Q5_1": 5,
    "GGML_TYPE_Q5_0": 6,
    "GGML_TYPE_Q4_1": 7,
    "GGML_TYPE_TURBO4_TCQ": 7,
    "GGML_TYPE_TURBO4_0": 7,
    "GGML_TYPE_Q4_0": 8,
    "GGML_TYPE_Q3_1": 9,
    "GGML_TYPE_TURBO3_TCQ": 9,
    "GGML_TYPE_TURBO3_0": 9,
    "GGML_TYPE_Q3_0": 10,
    "GGML_TYPE_Q2_1": 11,
    "GGML_TYPE_TURBO2_TCQ": 11,
    "GGML_TYPE_TURBO2_0": 11,
    "GGML_TYPE_Q2_0": 12,
}

DEFAULT_PAIRS = [
    ("GGML_TYPE_F16",  "GGML_TYPE_F16"),
    ("GGML_TYPE_F16",  "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_F16",  "GGML_TYPE_Q6_1"),
    ("GGML_TYPE_F16",  "GGML_TYPE_Q6_0"),
    ("GGML_TYPE_F16",  "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_F16",  "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_BF16", "GGML_TYPE_BF16"),
    ("GGML_TYPE_BF16", "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_BF16", "GGML_TYPE_Q6_1"),
    ("GGML_TYPE_BF16", "GGML_TYPE_Q6_0"),
    ("GGML_TYPE_BF16", "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_BF16", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q8_0"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q6_1"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q6_0"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q8_0", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q6_1"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q6_0"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q6_1", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q6_0", "GGML_TYPE_Q6_0"),
    ("GGML_TYPE_Q6_0", "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_Q6_0", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q6_0", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q6_0", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q5_1"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q3_1"),
    ("GGML_TYPE_Q5_1", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q5_0", "GGML_TYPE_Q5_0"),
    ("GGML_TYPE_Q5_0", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q5_0", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q5_0", "GGML_TYPE_Q3_1"),
    ("GGML_TYPE_Q5_0", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q4_1"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q3_1"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q2_1"),
    ("GGML_TYPE_Q4_1", "GGML_TYPE_Q2_0"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q4_0"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q3_1"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q2_1"),
    ("GGML_TYPE_Q4_0", "GGML_TYPE_Q2_0"),
    ("GGML_TYPE_Q3_1", "GGML_TYPE_Q3_1"),
    ("GGML_TYPE_Q3_1", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q3_1", "GGML_TYPE_Q2_1"),
    ("GGML_TYPE_Q3_1", "GGML_TYPE_Q2_0"),
    ("GGML_TYPE_Q3_0", "GGML_TYPE_Q3_0"),
    ("GGML_TYPE_Q3_0", "GGML_TYPE_Q2_1"),
    ("GGML_TYPE_Q3_0", "GGML_TYPE_Q2_0"),
    ("GGML_TYPE_Q2_1", "GGML_TYPE_Q2_1"),
    ("GGML_TYPE_Q2_1", "GGML_TYPE_Q2_0"),
    ("GGML_TYPE_Q2_0", "GGML_TYPE_Q2_0"),
]

def emit_pair(k, v):
    print(f"    FATTN_VEC_CASES_ALL_D_512({k}, {v})")

print("#if defined(GGML_CUDA_FA_ALL_QUANTS)")
all_count = 0
for k, _ in TYPES:
    for v, _ in TYPES:
        emit_pair(k, v)
        all_count += 1

print("#elif defined(GGML_CUDA_FA_HALF_QUANTS)")
half_count = 0
for k, _ in TYPES:
    for v, _ in TYPES:
        if HALF_RANKS[k] <= HALF_RANKS[v] or k == "GGML_TYPE_F16" or v == "GGML_TYPE_F16":
            emit_pair(k, v)
            half_count += 1

print("#else")
for k, v in DEFAULT_PAIRS:
    emit_pair(k, v)

print("#endif")

assert len(TYPES) == 19
assert all_count == 361, all_count
assert half_count == 217, half_count
assert len(DEFAULT_PAIRS) == 62, len(DEFAULT_PAIRS)
