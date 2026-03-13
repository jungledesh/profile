# Stage 1 — Builder: Rust + Python + vLLM
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/tmp/venv

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    tmux \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build Rust binary
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Python environment + vLLM
RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124 \
    && rm -rf /root/.cache/pip

# Stage 2 — Final runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/workspace/venv
ENV PATH="$VENV_DIR/bin:$PATH"

WORKDIR /workspace

# Copy Rust binary
COPY --from=builder /workspace/target/release/profile /workspace/profile

# Copy vLLM venv
COPY --from=builder /tmp/venv /workspace/venv

# Copy start script
COPY scripts/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Create models directory
RUN mkdir -p $MODELS_DIR

CMD ["/workspace/start.sh"]