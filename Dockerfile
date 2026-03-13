FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/workspace/vllm-env

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip build-essential tmux curl wget \
    && rm -rf /var/lib/apt/lists/*

# Rust + binary
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
COPY target/release/profile /workspace/profile

# Scripts
COPY ./scripts/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Models directory
RUN mkdir -p $MODELS_DIR

CMD ["/workspace/start.sh"]