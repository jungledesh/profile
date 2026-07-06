# Build profile binary on Ubuntu 22.04 — matches runtime GLIBC and can load libnvidia-ml.so
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
RUN cargo build --release
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Re-pin with: docker buildx imagetools inspect nvidia/cuda:12.4.1-devel-ubuntu22.04 --format '{{json .Manifest.Digest}}'
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

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

# ttyd — terminal-to-browser bridge required by VHS
RUN curl -fsSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    -o /usr/local/bin/ttyd \
    && chmod 0755 /usr/local/bin/ttyd

# VHS — terminal session recorder (v0.11.0)
RUN curl -fsSL https://github.com/charmbracelet/vhs/releases/download/v0.11.0/vhs_0.11.0_Linux_x86_64.tar.gz \
    | tar -xz -C /tmp \
    && find /tmp -name vhs -type f -exec mv {} /usr/local/bin/vhs \; \
    && chmod 0755 /usr/local/bin/vhs

# Do not mkdir VENV_DIR — an empty dir breaks start.sh's "create venv if missing" check
RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" /workspace && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

COPY --chown=appuser:appuser scripts/load.sh ./load.sh
COPY --chown=appuser:appuser scripts/start.sh ./start.sh
COPY --chown=appuser:appuser scripts/demo.sh ./demo.sh

COPY --from=profile-builder --chown=appuser:appuser /build/target/release/profile ./profile

RUN chmod 0755 ./load.sh ./start.sh ./demo.sh ./profile

USER appuser

CMD ["bash", "-lc", "/home/appuser/app/start.sh"]