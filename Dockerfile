# Re-pin with: docker buildx imagetools inspect nvidia/cuda:12.4.1-runtime-ubuntu22.04 --format '{{json .Manifest.Digest}}'
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
    ca-certificates \
    && /usr/sbin/useradd -m -u 1000 -s /bin/bash appuser \
    && rm -rf /var/lib/apt/lists/*

# VHS — terminal session recorder (pinned release)
RUN curl -fsSL https://github.com/charmbracelet/vhs/releases/download/v0.9.0/vhs_0.9.0_Linux_x86_64.tar.gz \
    | tar -xz -C /usr/local/bin vhs \
    && chmod 0755 /usr/local/bin/vhs

# Do not mkdir VENV_DIR — an empty dir breaks start.sh's "create venv if missing" check
RUN mkdir -p "${APP_DIR}" "${MODELS_DIR}" /workspace && \
    chown -R appuser:appuser /home/appuser /workspace

WORKDIR ${APP_DIR}

COPY --chown=appuser:appuser scripts/start.sh ./start.sh
COPY --chown=appuser:appuser scripts/load.sh ./load.sh
COPY --chown=appuser:appuser scripts/test.sh ./test.sh
COPY --chown=appuser:appuser scripts/test2.sh ./test2.sh
COPY --chown=appuser:appuser scripts/demo.sh ./demo.sh
COPY --chown=appuser:appuser scripts/demo.tape ./demo.tape
COPY --chown=appuser:appuser target/release/profile ./profile

RUN chmod 0755 ./start.sh ./load.sh ./test.sh ./test2.sh ./demo.sh ./profile

USER appuser

CMD ["bash", "-lc", "/home/appuser/app/start.sh"]