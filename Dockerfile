# Immutable digest for nvidia/cuda:12.4.1-devel-ubuntu22.04 (CUDA 12.4.1, Ubuntu 22.04).
# Re-resolve with: docker buildx imagetools inspect nvidia/cuda:12.4.1-devel-ubuntu22.04 --format '{{json .Manifest.Digest}}'
FROM nvidia/cuda@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ENV DEBIAN_FRONTEND=noninteractive

ENV APP_DIR=/home/appuser/app
ENV MODELS_DIR=/workspace/models
ENV VENV_DIR=/home/appuser/vllm-env
ENV PATH="${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    passwd \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    tmux \
    curl \
    wget \
    jq \
    ca-certificates \
    git \
    && /usr/sbin/useradd -m -u 1000 -s /bin/bash appuser \
    && rm -rf /var/lib/apt/lists/*

# Do not mkdir VENV_DIR — an empty dir breaks start.sh's "create venv if missing" check
RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" /workspace && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

COPY --chown=appuser:appuser scripts/start.sh ./start.sh
COPY --chown=appuser:appuser scripts/test.sh ./test.sh
COPY --chown=appuser:appuser target/release/profile ./profile

RUN chmod 0755 ./start.sh ./test.sh ./profile

USER appuser

CMD ["bash", "-lc", "/home/appuser/app/start.sh"]