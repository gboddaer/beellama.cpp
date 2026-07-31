# Anbeeld's BeeLlama.cpp

![BeeLlama.cpp logo](beellama.jpg)

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

BeeLlama.cpp (or just Bee) is a performance-focused llama.cpp fork for squeezing more speed and context out of local GGUF inference. It keeps the familiar llama.cpp tools and server flow, then adds DFlash speculative decoding, adaptive draft control, TurboQuant/TCQ KV-cache compression, and reasoning-loop protection, with full multimodal support.

> Not quite a pegasus, but close enough.

### Plug-and-Play Setups

- [Qwen 3.6 27B Q5_K_S + DFlash + vision + 160k context in 24 GB VRAM](docs/quickstart-qwen36-dflash.md)
- [Gemma 4 31B Q4_K_S + DFlash + vision + 140k context in 24 GB VRAM](docs/quickstart-gemma-4-31b-dflash.md)

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)

## Fork Features

- **DFlash speculative decoding**: `--spec-type dflash` drives a DFlash draft GGUF alongside the target model. The target captures hidden states into a ring buffer, the drafter cross-attends to the most recent `--spec-dflash-cross-ctx` hidden-state tokens and proposes drafts for target verification.
- **Adaptive draft-max control**: The server adjusts the active draft horizon at runtime instead of using a fixed `--spec-draft-n-max`. The default `profit` controller compares speculative throughput against a no-spec baseline, the `fringe` alternative maps acceptance-rate bands to draft depth.
- **Full multimodal support**: When `--mmproj` is active, the server keeps DFlash available for text generation. The model can be fully offloaded to CPU with no problems to reduce VRAM pressure.
- **Up to date with upstream llama.cpp**: MTP speculative decoding, parallel drafting, the unified llama app, updated server/API behavior, backend improvements across CUDA, HIP, Metal, Vulkan, and more.
- **TurboQuant / TCQ KV-cache compression**: Five cache types (`turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq`) spanning from 4x to 7.5x compression, with TCQ types offering good precision for their size. Set independently with `--cache-type-k` and `--cache-type-v`.
- **Tom TQ3/TQ4 model weight formats**: `TQ3_1S` and `TQ4_1S` are available through `llama-quantize` with non-conflicting GGML type IDs `47` and `48`. These are GGUF weight formats, not KV-cache types; backend acceleration should be validated before claiming it for a deployment.
- **Reasoning-loop protection**: The server detects repeated hidden reasoning output and intervenes. Default mode is `force-close`, with `--reasoning-loop-window` and `--reasoning-loop-max-period` tuning available.
- **Sampled DFlash verification**: `--spec-draft-temp` enables rejection-sampling drafter behavior. Activates when both draft and target temperature exceed zero. Draft log probabilities must be available for rejection sampling to produce correct output.
- **DDTree branch verification**: optional `--spec-branch-budget` adds branch nodes beyond the main draft path with GPU `parent_ids`, tree masks, and recurrent tree kernels. Disabled automatically when the target model spans more than one GPU. This one is very much work in progress!
- **Request-level speculative overrides**: Draft-max and branch budget can be overridden per-request through JSON fields without restarting the server.
- **CopySpec model-free speculation**: `--spec-type copyspec` provides rolling-hash suffix matching over previous tokens without a draft model.

For the full feature and public-repo comparison, read [docs/beellama-features.md](docs/beellama-features.md). For the complete argument reference, read [docs/beellama-args.md](docs/beellama-args.md).

TurboQuant (WHT-based scalar quantization) and the `TQ3_1S` / `TQ4_1S` weight formats originate from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). TCQ (Trellis-Coded Quantization) and basic DFlash implementation originate from [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (paper: [Closing the Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits](https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache)).

## DFlash Speedup

DFlash is strongest on structured, repetitive generation: code, tests, boilerplate, JSON-like formats, and other low-entropy continuations. Open-ended prose is much less predictable, so gains are smaller.

- Install `llama.cpp` using [brew, nix, winget, or conda-forge](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

* Setup: Windows 11, AMD Ryzen 7 5700X3D, 32 GB DDR4 RAM, RTX 3090 24 GB
* Config: same as in quick start docs ([Qwen 3.6 27B](docs/quickstart-qwen36-dflash.md), [Gemma 4 31B](docs/quickstart-gemma-4-31b-dflash.md)), but with reasoning off for non-chat prompts
* Baseline and MTP server in comparison: llama.cpp [b9275](https://github.com/ggml-org/llama.cpp/releases/tag/b9275) CUDA 13.1 Windows prebuilt

<details>
<summary>Benchmark prompts</summary>

**Task store module**

```text
Write one complete Python 3 file using only the standard library.

Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text.

Implement a deterministic Task store module with a compact, repetitive structure that is easy to predict.

Required shape:
- imports: dataclasses, datetime, typing
- dataclass Task with fields id: int, title: str, status: str, created_at: str
- class TaskStore with an internal dict[int, Task]
- methods: add, get, rename, mark_done, reopen, delete, clear, list_all, list_open, list_done, count_open, count_done, titles, to_dicts, __len__, __contains__
- add assigns increasing integer ids starting at 1
- valid statuses are "open" and "done"
- all list methods return tasks sorted by id
- count_open and count_done use explicit loops
- titles returns task titles sorted by task id
- to_dicts returns deterministic dictionaries sorted by id
- to_dicts includes id, title, status, and created_at keys for every task
- raise ValueError for empty title or missing task id
- use straightforward if statements and explicit loops
- keep method bodies short and similar in style
- no argparse, no JSON, no file IO, no unittest, no pytest
- target about 110 to 132 lines of code
- define __all__ = ["Task", "TaskStore"]
- stop immediately after defining __all__
```

**KV report module**

```text
Write one complete Python 3 file using only the standard library.

Return only Python code. Do not use markdown, comments, tests, examples, or explanatory text.

Implement a deterministic KV report module with a compact, repetitive structure that is easy to predict.

Required shape:
- imports: dataclasses, typing
- dataclass Row with fields key: str, value: str
- class Report with an internal list[Row]
- methods: add, set, get, delete, clear, keys, values, items, sorted_rows, render_lines, render_text, render_csv, filter_prefix, update_many, to_dict, copy, count_prefix, first_key, __len__, __contains__
- add appends a new row and rejects duplicate keys
- set updates an existing row or appends a new row
- get returns the value for a key
- delete removes a row by key
- keys, values, and items preserve insertion order
- sorted_rows returns rows sorted by key
- render_lines returns strings formatted as "key: value"
- render_text joins render_lines with newline characters
- render_csv returns deterministic "key,value" lines with a header
- filter_prefix returns a new Report containing keys that start with the prefix
- update_many applies set for each key and value in a dictionary sorted by key
- to_dict returns a deterministic dictionary sorted by key
- copy returns a new Report with the same rows in the same order
- count_prefix returns the number of keys that start with the prefix using an explicit loop
- first_key returns the first key and raises ValueError when there are no rows
- raise ValueError for empty keys, duplicate keys, or missing keys
- use straightforward if statements and explicit loops
- keep method bodies short and similar in style
- no enum, no alignment modes, no markdown table, no textwrap, no itertools, no unittest, no pytest
- target about 130 to 155 lines of code
- define __all__ = ["Row", "Report"]
- stop immediately after defining __all__

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [Liquid LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2)
- [x] [Liquid LFM2.5 models](https://huggingface.co/collections/LiquidAI/lfm25)
- [x] [Liquid Nanos](https://huggingface.co/collections/LiquidAI/liquid-nanos)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)
- [x] [Mellum models](https://huggingface.co/JetBrains/models?search=mellum)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Ruby: [docusealco/rllama](https://github.com/docusealco/rllama)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

**Doubly-linked list**

```text
Write a complete Python 3 module implementing a doubly-linked list with the following methods: append, prepend, insert_at, remove_at, find, reverse, to_list, length, is_empty, iter. Include comprehensive docstrings, type hints, and pytest unit tests for every method. Return only the code, no commentary.
```

**Multi-turn coding**

**Turn 1**

```text
Build an async WebSocket gateway for telemetry devices. Use Python 3.12 style typing.

Return exactly one labeled code block per requested file. Label each code block with a file header like `# ws_gateway/router.py`; outside code blocks, use at most one short sentence per file explaining why it exists. Assume this is a chat response only: do not mention saving files or accessing a filesystem.

Return code blocks for:
- `ws_gateway/models.py`: dataclasses/enums for `Topic`, `OutboundMessage`, `AckState`, `ClientId`, and `MetricSnapshot`
- `ws_gateway/metrics.py`: small standard-library metrics collector with counters/gauges and a Prometheus text export method
- `ws_gateway/errors.py`: error types for invalid topics, duplicate subscriptions, queue overflow policy errors, and closed sessions
- `ws_gateway/config.py`: immutable config dataclass for queue depth, ping interval, ack timeout, and rate limits

Keep prose brief and adjacent to the code. Do not write a separate architecture essay. Do not add extra files, alternate designs, or optional extensions.
```

**Turn 2**

```text
Now implement the topic router and subscriber queues. Return code-first output only: labeled code blocks plus brief inline notes if needed.

Return code blocks for:
- `ws_gateway/router.py`
- `tests/test_router.py`

Requirements:
- MQTT-like topics with `+` for one segment and `#` for the remaining suffix
- `subscribe`, `unsubscribe`, `publish`, `drain`, and `close_subscriber`
- stable per-subscriber ordering
- bounded per-subscriber queues with drop-oldest backpressure using a per-subscriber `deque(maxlen=...)`
- injectable clock for tests
- metrics updates for publishes, deliveries, drops, and active subscriptions
- pytest tests for wildcard matching, ordering, unsubscribe cleanup, and queue overflow

Use only the standard library in runtime code. Stay within the names and structure already established unless this prompt explicitly changes them.
```

**Turn 3**

```text
Add the connection/session layer over the router. Keep the answer mostly code: labeled code blocks and tests, no standalone design section.

Return code blocks for:
- `ws_gateway/session.py`
- `tests/test_session.py`

Implement:
- `ClientSession` with user id, connection id, subscribed topics, last pong time, pending ack ids, and lifecycle state
- `WebSocketLike` protocol with `send_json`, `close`, and `ping`
- `ConnectionManager` that accepts an authenticated session, starts/stops tasks, maps users to sessions, and unsubscribes on disconnect
- outbound delivery loop from subscriber queue to websocket
- ping/pong keepalive with zombie detection at 3x interval
- incoming token-bucket rate limiter at 10 messages/sec with burst 20
- tests for clean disconnect, zombie close, slow consumer drops, and rate limit rejection

Reuse the exact names and APIs already established unless this prompt explicitly changes them.
```

**Turn 4**

```text
Wire the service boundary. Keep prose to one-line notes beside code blocks.

Return code blocks for:
- `ws_gateway/app.py`: aiohttp app factory, websocket handler skeleton, health/readiness handlers, JWT verifier protocol, and graceful shutdown hooks
- `ws_gateway/persistence.py`: minimal ack persistence protocol plus an in-memory implementation for tests
- `tests/test_app_lifecycle.py`: tests for shutdown order, readiness false when persistence is unhealthy, and ack drain on disconnect
- `docs/load_test_plan.md`: concise checklist only, no paragraphs

Keep handlers thin and concrete. Do not introduce extra modules or abstractions beyond the listed files. The shutdown order and dependency boundaries must be concrete in code. Reuse the exact names and APIs already established unless this prompt explicitly changes them.
```

**Turn 5**

```text
Patch the code from this conversation. Output only chat-safe patch content: one short comment line per issue, then a unified diff or replacement snippet, then a pytest test. No separate review prose and no filesystem instructions.

Fix the top 5 concrete correctness problems likely in this gateway:
- leaked delivery or keepalive tasks on disconnect
- duplicate subscription state after reconnect
- unfair drop-oldest behavior across hot and cold topics
- ack persistence race during shutdown
- readiness or metrics reporting success while internal tasks are failing

For each fix, include:
1. a one-line bug label as a code comment
2. the smallest Python diff or replacement snippet
3. one focused pytest test that fails before the fix

Stay within the existing file labels and APIs.
```

</details>

### Qwen 3.6 27B

Target model: [Qwen 3.6 27B Q5_K_S](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) or [Qwen 3.6 27B MTP Q5_K_S](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF). DFlash model: [Q4_K_M](https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF).

| Prompt | Server | Output | Median | Best | Speedup | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Task store module | Baseline | ~1K tok | 37.2 tok/s | 37.2 tok/s | 1.00x | N/A |
| Task store module | DFlash | ~1K tok | **163.9 tok/s** | 181.9 tok/s | **4.40x** | 67.7% / 89.2% |
| Task store module | MTP | ~1K tok | 69.3 tok/s | 69.6 tok/s | 1.86x | 92.0% / 73.3% |
| KV report module | Baseline | ~1K tok | 34.6 tok/s | 36.5 tok/s | 1.00x | N/A |
| KV report module | DFlash | ~1K tok | **157.7 tok/s** | 162.5 tok/s | **4.56x** | 58.8% / 88.9% |
| KV report module | MTP | ~1K tok | 67.3 tok/s | 68.1 tok/s | 1.94x | 89.3% / 73.0% |
| Doubly-linked list | Baseline | ~4K tok | 36.8 tok/s | 36.9 tok/s | 1.00x | N/A |
| Doubly-linked list | DFlash | ~4K tok | **130.8 tok/s** | 154.1 tok/s | **3.56x** | 50.4% / 86.8% |
| Doubly-linked list | MTP | ~4K tok | 66.3 tok/s | 68.0 tok/s | 1.80x | 87.8% / 72.5% |
| Prompt processing | Baseline | ~20K tok | 1229.5 tok/s | 1229.5 tok/s | 1.00x | N/A |
| Prompt processing | DFlash | ~20K tok | **1214.4 tok/s** | 1221.7 tok/s | **0.99x** | N/A |
| Prompt processing | MTP | ~20K tok | 1162.6 tok/s | 1164.7 tok/s | 0.95x | N/A |
| Multi-turn coding | Baseline | ~28K tok | 33.3 tok/s | 33.3 tok/s | 1.00x | N/A |
| Multi-turn coding | DFlash | ~30K tok | **64.6 tok/s** | 65.4 tok/s | **1.94x** | 24.9% / 72.9% |
| Multi-turn coding | MTP | ~34K tok | 56.5 tok/s | 56.5 tok/s | 1.70x | 71.9% / 68.3% |

*Acceptance: accepted to proposed draft tokens / accepted draft tokens to final generated tokens*

### Gemma 4 31B

Target model: [Gemma 4 31B Q4_K_S](https://huggingface.co/unsloth/gemma-4-31b-it-GGUF). DFlash model: [Q5_K_M](https://huggingface.co/Anbeeld/gemma-4-31B-it-DFlash-GGUF).

| Prompt | Server | Output | Median | Best | Speedup | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Task store module | Baseline | ~1K tok | 36.1 tok/s | 36.1 tok/s | 1.00x | N/A |
| Task store module | DFlash | ~1K tok | **177.8 tok/s** | 182.0 tok/s | **4.93x** | 65.7% / 90.0% |
| KV report module | Baseline | ~1K tok | 35.9 tok/s | 36.0 tok/s | 1.00x | N/A |
| KV report module | DFlash | ~1K tok | **154.3 tok/s** | 162.8 tok/s | **4.29x** | 55.7% / 88.6% |
| Doubly-linked list | Baseline | ~1.9K tok | 36.0 tok/s | 36.0 tok/s | 1.00x | N/A |
| Doubly-linked list | DFlash | ~1.9K tok | **116.6 tok/s** | 127.3 tok/s | **3.24x** | 44.5% / 84.9% |
| Prompt processing | Baseline | ~24K tok | 1021.3 tok/s | 1021.3 tok/s | 1.00x | N/A |
| Prompt processing | DFlash | ~24K tok | **954.5 tok/s** | 954.9 tok/s | **0.93x** | N/A |
| Multi-turn coding | Baseline | ~12K tok | 34.8 tok/s | 34.8 tok/s | 1.00x | N/A |
| Multi-turn coding | DFlash | ~12K tok | **60.6 tok/s** | 64.1 tok/s | **1.74x** | 24.4% / 72.3% |

*Acceptance: accepted to proposed draft tokens / accepted draft tokens to final generated tokens*

## KV Cache Quantization

K and V cache types are set independently with `--cache-type-k` and `--cache-type-v`. For the preset rationale and benchmark details, see [KV Cache Quantization Benchmarks for Long Context](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context).

### Preset Ladder

| K / V | % of bf16 size | 99.9% precision | What it is for |
| --- | ---: | ---: | --- |
| bf16 / bf16 | 100.0 | 100.00% | Preserving full quality |
| q8_0 / q8_0 | 53.1 | 94.62% | Validation and blame-isolation mode |
| **q8_0 / q6_0** | **46.9** | **94.33%** | **Recommended high-end preset** |
| q8_0 / q5_1 | 45.3 | 94.21% | Fallback if q6_0 V is unavailable |
| q8_0 / q5_0 | 43.8 | 93.69% | If the high-end rows miss the fit by a narrow margin |
| q6_0 / q5_0 | 37.5 | 93.29% | Optional headroom tier between q5 and q8 K |
| q5_0 / q5_0 | 34.4 | 93.16% | Normal quality preset |
| **q5_0 / q4_1** | **32.8** | **92.65%** | **Best default if VRAM-constrained** |
| q5_0 / q4_0 | 31.3 | 91.39% | If q5_0 / q4_1 misses the fit by a narrow margin |
| q4_0 / q4_0 | 28.1 | 88.87% | Memory saving with visible precision loss |
| q4_0 / turbo3_tcq | 24.2 | 84.93% | Smaller than q4, cleaner than symmetric turbo3_tcq |
| **turbo3_tcq / turbo3_tcq** | **20.3** | **81.56%** | **Viable extreme-compression mode** |
| turbo2_tcq / turbo2_tcq | 14.1 | 54.38% | Last resort: not for code, JSON, math, or tool calls |

*99.9% precision = `100 · exp(−(quantKLD − bf16KLD))` at the 99.9% KL-divergence tail.*

### Type Reference

| Type | Origin | bpv | Diff vs bf16 | Notes |
| --- | --- | ---: | ---: | --- |
| q8_0 | upstream | 8.5 | 1.88× | High-fidelity K or V |
| q6_0 | upstream | 6.5 | 2.46× | Robust type for high-end presets |
| q5_1 | upstream | 6 | 2.67× | Conservative, might be better for V than q5_0 |
| q5_0 | upstream | 5.5 | 2.91× | Strong K type for VRAM constrained configs |
| q4_1 | upstream | 5 | 3.2× | Smaller than q5_0, but weaker in the tail. Prefer q5_0 for K |
| q4_0 | upstream | 4.5 | 3.56× | Default high compression type, decent at its size |
| turbo4 | fork | 4.125 | 3.88× | Barely smaller than q4_0, slower, worse tail |
| turbo3_tcq | fork | 3.25 | 4.92× | Viable compact mode, 82% precision at KLD 99.9%. CUDA-only |
| turbo3 | fork | 3.125 | 5.12× | Weaker than turbo3_tcq. Use only when TCQ is unavailable |
| turbo2_tcq | fork | 2.25 | 7.11× | Last resort, 54% precision at KLD 99.9%. CUDA-only |
| turbo2 | fork | 2.125 | 7.53× | Extreme quality risk. Use only when TCQ is unavailable |

## Installation

### Plug-and-Play Setups

- [Qwen 3.6 27B Q5_K_S + DFlash + vision + 160k context in 24 GB VRAM](docs/quickstart-qwen36-dflash.md)
- [Gemma 4 31B Q4_K_S + DFlash + vision + 140k context in 24 GB VRAM](docs/quickstart-gemma-4-31b-dflash.md)

### Prebuilt

Current release binaries are on the [releases page](https://github.com/Anbeeld/beellama.cpp/releases):

| Platform | Backend | Archive |
| --- | --- | --- |
| macOS arm64 | Metal | `bin-macos-arm64.tar.gz` |
| Ubuntu x64 | CPU | `bin-ubuntu-x64.tar.gz` |
| Ubuntu arm64 | CPU | `bin-ubuntu-arm64.tar.gz` |
| Ubuntu x64 | CUDA 12.4 | `bin-ubuntu-cuda-12.4-x64.tar.gz` |
| Ubuntu x64 | CUDA 13.1 | `bin-ubuntu-cuda-13.1-x64.tar.gz` |
| Ubuntu x64 | Vulkan | `bin-ubuntu-vulkan-x64.tar.gz` |
| Ubuntu x64 | ROCm 7.2 | `bin-ubuntu-rocm-7.2-x64.tar.gz` |
| Ubuntu x64 | SYCL | `bin-ubuntu-sycl-x64.tar.gz` |
| Windows x64 | CPU | `bin-win-cpu-x64.zip` |
| Windows x64 | SYCL | `bin-win-sycl-x64.zip` |
| Windows x64 | CUDA 12.4 | `bin-win-cuda-12.4-x64.zip` |
| Windows x64 | CUDA 13.1 | `bin-win-cuda-13.1-x64.zip` |
| Windows x64 | HIP/Radeon | `bin-win-hip-radeon-x64.zip` |

Windows CUDA archives contain a `ggml-cuda.dll` backend; download the matching `cudart-win-cuda-*-x64.zip` runtime archive and extract it into the same folder. Windows SYCL and HIP archives ship as standalone packages with all required runtime DLLs bundled.

Docker images are published to `ghcr.io/anbeeld/beellama.cpp`:

| Image | Acceleration | Platforms |
| --- | --- | --- |
| `server`, `server-cpu` | CPU | linux/amd64, linux/arm64 |
| `server-cuda`, `server-cuda12` | CUDA 12.4 | linux/amd64 |
| `server-cuda13` | CUDA 13.1 | linux/amd64 |
| `server-rocm` | ROCm | linux/amd64 |
| `server-vulkan` | Vulkan | linux/amd64 |
| `server-sycl` | SYCL | linux/amd64 |

Building from source with `-DGGML_NATIVE=ON` *may* result in a *tiny* bit better performance, so it might still be a good idea to do that if/when you decide to use this fork long-term.

### CUDA Build

```bash
# Linux (GCC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`GGML_CUDA_FA_ALL_QUANTS=ON` is required for TurboQuant and TCQ cache types. Add `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090, or `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090, if cross-compiling or building in CI without a GPU.

### Other Backends

Bee inherits llama.cpp backend support, including Metal, HIP, Vulkan, SYCL, BLAS, CANN, MUSA, OpenVINO, OpenCL, and RPC. Use the upstream-style build docs in [docs/build.md](docs/build.md) and backend-specific pages under [docs/backend](docs/backend).

## Common Commands

### Local CLI

```sh
llama-cli -m model.gguf
llama-cli -m model.gguf -cnv --chat-template chatml
llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p "Request: schedule a call at 8pm; Command:"
```

### OpenAI-Compatible Server

```sh
llama-server -m model.gguf --port 8080
llama-server -m model.gguf -c 16384 -np 4
llama-server -m model.gguf -md draft.gguf
```

### DFlash And TurboQuant Together

```sh
llama-server -m target.gguf --spec-type dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-ngl all \
  --flash-attn on --cache-type-k turbo4 --cache-type-v turbo3_tcq
```

## Documentation

- [BeeLlama features and public repo diff](docs/beellama-features.md)
- [BeeLlama args reference](docs/beellama-args.md)
- [Build docs](docs/build.md)
- [Server docs](tools/server/README.md)
- [Docker docs](docs/docker.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)

## Contributing

Keep PRs small and scoped. Run the narrowest relevant tests or benchmarks before opening a PR, and include the exact commands. For fork-specific speculative decoding, DFlash, TurboQuant, or reasoning-loop changes, update the corresponding docs when behavior or args change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for inherited llama.cpp contribution conventions and this fork's AI usage policy.

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - single-header HTTP server used by `llama-server` - MIT
- [stb-image](https://github.com/nothings/stb) - single-header image decoder used by multimodal code - public domain
- [nlohmann/json](https://github.com/nlohmann/json) - single-header JSON library - MIT
- [miniaudio.h](https://github.com/mackron/miniaudio) - single-header audio decoder - public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - process launching helper - public domain
- [Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference) - suffix tree and int32 map used in speculative decoding (`common/suffix-tree.*`, `common/int32-map.h`) - Apache-2.0
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - frontend header used in OpenVINO backend (`ggml/src/ggml-openvino/openvino/frontend.h`) - Apache-2.0
- Intel SYCL/oneAPI - SYCL backend (`ggml/src/ggml-sycl/`) - Apache-2.0 WITH LLVM-exception

See the `licenses/` directory for full license texts.

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)
