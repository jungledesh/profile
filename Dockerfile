FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Stable paths
ENV APP_DIR=/home/appuser/app
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/home/appuser/vllm-env
ENV PATH="${VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin"

# System packages + non-root user (minimal CUDA bases omit useradd until passwd is installed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    passwd \
    python3 \
    python3-venv \
    python3-pip \
    tmux \
    curl \
    wget \
    jq \
    ca-certificates \
    git \
    && useradd -m -u 1000 -s /bin/bash appuser \
    && rm -rf /var/lib/apt/lists/*

# Create runtime dirs
RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" "/home/appuser" && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

# Copy only what is needed at runtime
COPY --chown=appuser:appuser scripts/start.sh ./start.sh
COPY --chown=appuser:appuser scripts/test.sh ./test.sh
COPY --chown=appuser:appuser target/release/profile ./profile

RUN chmod 0755 ./start.sh ./test.sh ./profile

USER appuser

CMD ["/home/appuser/app/start.sh"]