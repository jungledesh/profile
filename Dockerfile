FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/workspace/vllm-env

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    tmux \
    python3 \
    python3-venv \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# python venv
RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

RUN pip install --upgrade pip

# vLLM
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# HF CLI
RUN pip install huggingface_hub

RUN mkdir -p $MODELS_DIR

COPY target/release/profile /workspace/profile
COPY ./scripts/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]