# Slim CUDA runtime base. Heavy deps (torch, NeMo, model) are installed at RUNTIME by
# entrypoint.sh, so this image stays small and builds fast on a CPU-only CI runner.
# Base CUDA (12.9) does NOT force torch's CUDA build: torch wheels bundle their own runtime
# and CUDA-12 has minor-version forward-compat; only the RunPod host driver must be >= the wheel's.
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04
# Fallback if a NeMo ASR dep lacks an Ubuntu 24.04 wheel:
# FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/sruckh/nemotron35-runpod"
LABEL org.opencontainers.image.title="nemotron35-runpod"
LABEL org.opencontainers.image.description="NVIDIA Nemotron 3.5 ASR streaming (Gradio) pod for RunPod"
LABEL org.opencontainers.image.licenses="OpenMDW-1.1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/workspace/hf_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# Minimal OS + Python. No ML libs baked in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libsndfile1 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# Light server deps only (gradio + numpy + soundfile). torch/nemo install at runtime.
# NOTE: do NOT `--upgrade pip` here — the debian-managed pip has no RECORD file and refuses
# self-uninstall. The shipped pip (24.0) is recent enough for these wheels.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY entrypoint.sh ./
COPY app ./app
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

# entrypoint.sh: idempotent runtime bootstrap (apt -> torch -> NeMo -> prefetch) then exec CMD.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "app.gradio_app"]
