# Shared builder: compiles the profile binary once for both targets.
FROM ubuntu:22.04 AS profile-builder

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

COPY Cargo.toml Cargo.lock ./

RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --locked

COPY src ./src

RUN touch src/main.rs && cargo build --release --locked

# NVIDIA runtime: CUDA devel image + vLLM installed via pip at container start.
# Build: docker build --target nvidia -t profile:nvidia .
# 12.8 is the Blackwell floor and matches driver 570. 12.9 made nvidia-container-cli
# refuse to start on those hosts (cuda>=12.9). Newer drivers still run this image.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS nvidia

ENV APP_DIR=/home/appuser/app
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/home/appuser/vllm-env
ENV PATH="${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# noninteractive only for this layer so apt/debconf never prompts during build; omit from runtime ENV.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    bash \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    jq \
    gawk \
    tmux \
    ffmpeg \
    vim \
    sudo \
    ca-certificates \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    openssh-client \
    rsync \
    && /usr/sbin/useradd -m -u 1000 -s /bin/bash appuser \
    && echo "appuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/appuser \
    && rm -rf /var/lib/apt/lists/*

# ttyd: terminal-to-browser bridge required by VHS
RUN curl -fsSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    -o /usr/local/bin/ttyd \
    && chmod 0755 /usr/local/bin/ttyd

# VHS: terminal session recorder (v0.11.0)
RUN curl -fsSL https://github.com/charmbracelet/vhs/releases/download/v0.11.0/vhs_0.11.0_Linux_x86_64.tar.gz \
    | tar -xz -C /tmp \
    && find /tmp -name vhs -type f -exec mv {} /usr/local/bin/vhs \; \
    && chmod 0755 /usr/local/bin/vhs

# Do not mkdir VENV_DIR: an empty dir breaks start.sh's "create venv if missing" check
RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" /workspace && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

COPY --chown=appuser:appuser scripts/load.sh ./load.sh
COPY --chown=appuser:appuser scripts/start.sh ./start.sh
COPY --chown=appuser:appuser scripts/start-gemma.sh ./start-gemma.sh
COPY --chown=appuser:appuser scripts/start-muse.sh ./start-muse.sh
COPY --chown=appuser:appuser scripts/tool_chat_template_gemma4.jinja ./tool_chat_template_gemma4.jinja
COPY --chown=appuser:appuser scripts/agent-swarm.sh ./agent-swarm.sh
# Swarm task list: agent-swarm.sh reads swarm-tasks.json next to itself.
# fetch-swarm-tasks.py regenerates it if needed (requires internet).
COPY --chown=appuser:appuser scripts/swarm-tasks.json ./swarm-tasks.json
COPY --chown=appuser:appuser scripts/fetch-swarm-tasks.py ./fetch-swarm-tasks.py
COPY --chown=appuser:appuser scripts/support-load.py ./support-load.py
COPY --from=profile-builder --chown=appuser:appuser /build/target/release/profile ./profile

RUN chmod 0755 ./load.sh ./start.sh ./start-gemma.sh ./start-muse.sh ./agent-swarm.sh ./support-load.py ./profile

USER appuser

# Default NVIDIA stack is Muse Glimmer NVFP4 (CMD / start-muse.sh) for RTX 5090.
# Same pattern as Qwen/Gemma: pip vLLM in start-*.sh, download weights, serve.
# Do not bake SERVED_NAME: each launcher exports PROFILE_MODEL + SERVED_NAME.
# Qwen: ./start.sh   Gemma: ./start-gemma.sh
ENV PROFILE_MODEL=muse

CMD ["bash", "-lc", "/home/appuser/app/start-muse.sh"]

# AMD runtime: official vLLM ROCm image (includes ROCm + Python 3.12 + vLLM + PyTorch).
# Build: docker build --target amd -t profile:amd .
# Run (RunPod may set some of these automatically; verify on your pod):
#   docker run --device=/dev/kfd --device=/dev/dri --group-add video \
#     --shm-size 16G --security-opt seccomp=unconfined \
#     -p 8000:8000 -it profile:amd
# start.sh downloads Qwen3.6-27B only (at runtime). Swarm: ./agent-swarm.sh
# Optional: -e HF_TOKEN=... for Hugging Face rate limits (model is ungated).
FROM vllm/vllm-openai-rocm:v0.25.1 AS amd

ENV APP_DIR=/home/appuser/app
ENV MODELS_DIR=/workspace/models

# The vLLM image runs as root by default. Create appuser like the NVIDIA image.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    wget \
    git \
    jq \
    gawk \
    tmux \
    vim \
    sudo \
    ca-certificates \
    openssh-client \
    rsync \
    && /usr/sbin/useradd -m -u 1000 -s /bin/bash appuser \
    && echo "appuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/appuser \
    && rm -rf /var/lib/apt/lists/*

# libdrm 2.4.113 (Ubuntu 22.04) is missing drmSyncobjEventfd (added 2.4.116).
# libdrm_amdgpu_sys v0.8.16 requires all bound symbols at dlopen time.
# Build libdrm 2.4.123 from source with only the amdgpu backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    meson ninja-build pkg-config \
    && cd /tmp \
    && wget -q https://dri.freedesktop.org/libdrm/libdrm-2.4.123.tar.xz \
    && tar xf libdrm-2.4.123.tar.xz && cd libdrm-2.4.123 \
    && meson setup build \
       -Dprefix=/usr \
       -Dlibdir=lib/x86_64-linux-gnu \
       -Damdgpu=enabled \
       -Dintel=disabled \
       -Dnouveau=disabled \
       -Dvmwgfx=disabled \
       -Dradeon=disabled \
    && ninja -C build install \
    && ldconfig \
    && cd / && rm -rf /tmp/libdrm* \
    && apt-get purge -y meson ninja-build pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Intentionally omitted vs NVIDIA stage:
#   - python3, python3-venv, python3-pip, python3-dev, build-essential
#     (already in vLLM base image)
#   - ffmpeg, libnss3, libatk*, libcups2, libxkbcommon0, etc.
#     (X11/browser deps for VHS/ttyd, not needed for e2e testing)
#   - ttyd and VHS binaries (demo tooling, not needed for e2e testing)
# libdrm built from source above (2.4.123) because base image ships 2.4.113
# which is missing drmSyncobjEventfd required by libdrm_amdgpu_sys v0.8.16.

RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" /workspace && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

COPY --chown=appuser:appuser scripts/load.sh ./load.sh
COPY --chown=appuser:appuser scripts/start-amd.sh ./start.sh
COPY --chown=appuser:appuser scripts/start-gemma-amd.sh ./start-gemma.sh
COPY --chown=appuser:appuser scripts/tool_chat_template_gemma4.jinja ./tool_chat_template_gemma4.jinja
COPY --chown=appuser:appuser scripts/agent-swarm.sh ./agent-swarm.sh
# Swarm task list: agent-swarm.sh reads swarm-tasks.json next to itself.
# fetch-swarm-tasks.py regenerates it if needed (requires internet). Do not
# run the fetcher at image build — no model or task downloads in the Dockerfile.
COPY --chown=appuser:appuser scripts/swarm-tasks.json ./swarm-tasks.json
COPY --chown=appuser:appuser scripts/fetch-swarm-tasks.py ./fetch-swarm-tasks.py
COPY --chown=appuser:appuser scripts/support-load.py ./support-load.py
# AMD demo load: enterprise RAG over pinned real vLLM docs (rag-tasks.json).
# fetch-vllm-docs-rag.py regenerates the pack if needed (git + network).
COPY --chown=appuser:appuser scripts/rag-load.sh ./rag-load.sh
COPY --chown=appuser:appuser scripts/rag-load.py ./rag-load.py
COPY --chown=appuser:appuser scripts/fetch-vllm-docs-rag.py ./fetch-vllm-docs-rag.py
COPY --chown=appuser:appuser scripts/rag-tasks.json ./rag-tasks.json

COPY --from=profile-builder --chown=appuser:appuser /build/target/release/profile ./profile

RUN chmod 0755 ./load.sh ./start.sh ./start-gemma.sh ./agent-swarm.sh \
    ./support-load.py ./rag-load.sh ./rag-load.py ./fetch-vllm-docs-rag.py ./profile

USER appuser

# Default AMD stack is Qwen (CMD / start.sh) + vLLM-docs RAG load (rag-load.sh).
# Do not bake SERVED_NAME: each launcher exports PROFILE_MODEL + SERVED_NAME.
# Gemma on AMD: ./start-gemma.sh. No model weights are downloaded at build time.
ENV PROFILE_MODEL=qwen

ENTRYPOINT []
CMD ["bash", "-lc", "/home/appuser/app/start.sh"]
