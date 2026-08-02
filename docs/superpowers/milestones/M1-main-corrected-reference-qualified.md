# M1: Main Corrected Reference Qualified

Date: 2026-08-02

## Classification

`qualified`

The corrected reference passes all defined correctness and performance gates. This qualifies the reference only. It does not claim that the work branch has reached final parity.

## Branch and Source

```text
historical reference: 130ea2480bee8268907a102bd814e276f4e88bdf
corrected branch: main-corrected-reference
corrected source commit: 78db8bd06fd3937f0397d447ad051edeea0bbeb2
work control: 986ce7aed9b30a00145422c1cbb75da9a6557d05
source patch SHA-256: 450b5bc4ea495839ac60f159cbeb024afb1add236efde939676b9e76b7961714
```

The production source change is limited to:

```text
common/speculative.cpp
common/speculative.h
tools/server/server-context.cpp
```

The patch resets per-request MTP/DFlash implementation state, clears target and draft request state, bypasses stale speculative prompt-cache restoration, and serializes speculative launches before clearing shared target memory. It does not modify DFlash graph, verifier, replay, copy, Vulkan queue, scheduler, draft horizon, or cache-performance paths.

## Binary and Model Provenance

```text
historical server SHA-256: 45ce479714de2e1528fba6c83b118ea99698d5f14cdde71642c673bfe43ffa69
corrected server SHA-256: 37e74e6b70a2fc2f6ec7bbe68cdd1afe64b4e98ca62a8086fb5ffee265d10b27
corrected libllama SHA-256: bab10c41b02c4ff384b239e7c5ad4124e03c6042d998ad466cda676d3e48f4ab
target model SHA-256: a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f
draft model SHA-256: af2d6a6fa0fcd1953214143720b8e7d653bc09b3490ef45c5f668badb7a19c0d
compiler: GNU 14.2.0
device: Vulkan0: AMD Radeon Graphics (RADV GFX1151)
```

Corrected binary version:

```text
version: 10119 (78db8bd06)
```

## Failure Reproduction

Unmodified `130ea2480` reproduced the expected post-warmup defect:

```text
MTP: invalid prompt_echo, 256 tokens, draft_n=227
DFlash: invalid prompt_echo, 256 tokens, draft_n=281
```

The corrected build changed both reproductions to valid output.

## Model-free Verification

```text
strict harness and repeated-request unit tests: 34/34 passed
reference DFlash CTests present at 130ea2480: 2/2 passed
```

The reference contains `test-dflash-ring` and `test-dflash-plumbing`. `test-dflash-decode` was introduced after the historical reference and is not present in this source tree.

## Throughput Medians

All values are coding-prompt predicted tokens per second.

| Mode | Historical first request | Corrected first request | Corrected persistent |
|---|---:|---:|---:|
| BASE | 12.379542 | 12.377061 | 12.353944 |
| MTP | 21.651411 | 21.810663 | 21.186074 |
| DFlash | 30.078299 | 29.913420 | 36.765106 |

Historical and corrected first-request medians each use five independent server processes. Corrected persistent medians use five measured rows after warmup on one server.

## Performance Ratios

```text
corrected-first/historical BASE: 0.999800
corrected-first/historical MTP: 1.007355
corrected-first/historical DFlash: 0.994518
corrected-persistent DFlash/BASE: 2.975981
```

Performance gates:

```text
BASE retention 0.95 to 1.05: pass
MTP retention 0.95 to 1.05: pass
DFlash retention 0.95 to 1.05: pass
persistent DFlash >= 28.457 t/s: pass
persistent DFlash >= 1.5x BASE: pass
```

## Persistent Correctness

```text
BASE: 5/5 valid
MTP: 5/5 valid, all rows have nonzero draft counts
DFlash: 5/5 valid, all rows have nonzero draft counts
```

Corrected independent first-request hashes are stable within every mode. Persistent rows 2-5 are hash-identical within every mode. The post-warmup first row is recorded separately and is not filtered.

Historical/corrected first-request hashes match for BASE and MTP. DFlash changes from the historical valid 98-token variant to the corrected valid 106-token variant. This difference is retained in `summary.json`; it is not hidden or relabeled as equivalence.

## Concurrent Correctness

Correctness generation ceiling: 512 tokens.

```text
MTP: 6/6 valid, nonzero drafts, contamination-free
DFlash: 6/6 valid, nonzero drafts, contamination-free
```

Each mode ran three concurrent coding-plus-math rounds with `np=2`. The analyzer found no coding text in math responses and no math text in coding responses.

## Profile Evidence

The profile server used:

```text
GGML_DFLASH_PROFILE=summary,verify
GGML_DFLASH_PROFILE_SYNC_SPLIT=1
LLAMA_ARG_LOG_VERBOSITY=4
```

Results:

```text
profile row valid: true
dflash profile decode markers: 22
verify_sync_split markers: 22
```

## Independent Review

GLM found no critical issue. Its three requested checks were resolved:

```text
MTP per-sequence containers are sized to n_seq during construction
DFlash cross-ring state is implementation-global
speculative target clearing is guarded by the serial task-queue idle check
```

A concise source comment records the shared-target clearing invariant. The final source commit and all corrected measurements were rebuilt and rerun after that review change.

## Evidence

```text
root: /crypt/tmp/dflash-main-corrected-reference-20260802T085318Z
preflight: preflight.txt
corrected provenance: corrected-provenance.txt
source patch: source.patch
source patch hash: source-patch.sha256
historical rows: historical-{base,mtp,dflash}-coding.jsonl
corrected first rows: corrected-first-{base,mtp,dflash}-coding.jsonl
corrected persistent rows: corrected-persistent-{base,mtp,dflash}-coding.jsonl
concurrent MTP: corrected-mtp-concurrent.json
concurrent DFlash: corrected-dflash-concurrent.json
profile: corrected-dflash-profile.jsonl
analyzer: analyze.py
summary: summary.json
analysis: analysis.txt
```

## Final Objectives Preserved

```text
work DFlash median >= 28.457 t/s
work DFlash median >= 1.5x work BASE
work/corrected-reference BASE ratio between 0.95 and 1.05
BASE and MTP correctness and performance preserved
all measured BASE, MTP, and DFlash outputs valid
final BASE, MTP, and DFlash persistent gates pass on one server without restart
final MTP and DFlash concurrent gates pass without cross-request text or token contamination
```

The corrected reference satisfies its side of these objectives. Work-branch acceptance remains a separate future gate.
