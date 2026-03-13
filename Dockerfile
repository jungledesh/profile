FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Environment
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/workspace/vllm-env
ENV APP_DIR=/root/app

WORKDIR $APP_DIR

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip build-essential tmux curl wget jq vim \
    && rm -rf /var/lib/apt/lists/*

# Rust + binary
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy Rust binary and scripts to safe location
COPY target/release/profile ./profile
COPY scripts/start.sh ./start.sh
COPY scripts/test.sh ./test.sh
RUN chmod +x ./start.sh ./test.sh

# CMD to start your app
CMD ["/root/app/start.sh"]