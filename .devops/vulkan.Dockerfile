# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=26.04

FROM ubuntu:$UBUNTU_VERSION AS build

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ccache wget xz-utils

# Install SSL and Vulkan SDK dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev libvulkan-dev glslc spirv-headers

ENV CCACHE_SLOPPINESS=time_macros CCACHE_MAXSIZE=5G

# Build it
WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build \
      -DGGML_NATIVE=OFF \
      -DGGML_VULKAN=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    cmake --build build --config Release -j$(nproc) && \
    ccache --show-stats

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ubuntu:$UBUNTU_VERSION AS base

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl libvulkan1 mesa-vulkan-drivers \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    && rm -rf /tmp/* /var/tmp/*

COPY --from=build /app/lib/ /app

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/Anbeeld/beellama.cpp
ARG IMAGE_SOURCE=https://github.com/Anbeeld/beellama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="BeeLlama.cpp" \
      org.opencontainers.image.description="BeeLlama.cpp GGUF inference with DFlash, TurboQuant, and TCQ cache types" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

ENV PATH="/root/.venv/bin:/root/.local/bin:${PATH}"

# Flag for compatibility with pip
ARG UV_INDEX_STRATEGY="unsafe-best-match"
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.13 \
    && uv venv --python 3.13 /root/.venv \
    && uv pip install --python /root/.venv/bin/python -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

WORKDIR /app

RUN /app/llama-server --version

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
