#!/usr/bin/env bash
# Runtime bootstrap — idempotent. Heavy deps install here (NOT in the image) so the image stays slim.
# First pod boot pays the install + model download; subsequent restarts skip everything via the checks.
# Mount a RunPod network volume at /workspace to persist HF + pip caches across restarts.
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# Caches: prefer network volume, fall back to ephemeral container storage.
export MODEL_CACHE="${MODEL_CACHE:-/workspace/hf_cache}"
export PIP_CACHE="${PIP_CACHE:-/workspace/pip_cache}"
mkdir -p "$MODEL_CACHE" "$PIP_CACHE" 2>/dev/null || true
export HF_HOME="$MODEL_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# 1. apt: audio libs + git (git is required by `pip install git+...` for NeMo) (idempotent)
if ! command -v git >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || ! ldconfig -p 2>/dev/null | grep -q libsndfile; then
  log "installing git + ffmpeg + libsndfile1"
  apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg
  rm -rf /var/lib/apt/lists/*
else
  log "git + ffmpeg + libsndfile already present"
fi

# 2. build deps (idempotent). Do NOT `--upgrade pip`: debian's pip has no RECORD file and
#    refuses self-uninstall ("Cannot uninstall pip ... installed by debian").
log "ensuring Cython + packaging"
python -m pip install -q "Cython>=3.0" "packaging"

# 3. torch (pinned, cu128) — idempotent
if ! python -c "import torch" 2>/dev/null; then
  log "installing torch 2.9.1 / torchvision 0.24.1 / torchaudio 2.9.1 (cu128)"
  python -m pip install -q torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
else
  log "torch already present: $(python -c 'import torch;print(torch.__version__)')"
fi

# 4. NeMo from git (idempotent)
if ! python -c "import nemo.collections.asr" 2>/dev/null; then
  NEMO_REPO="${NEMO_REPO:-https://github.com/NVIDIA/NeMo.git}"
  NEMO_REF="${NEMO_REF:-main}"
  log "installing nemo_toolkit[asr] from ${NEMO_REPO}@${NEMO_REF}"
  # PEP 508 direct reference (no #egg= fragment, which pip 25.0 deprecates).
  python -m pip install -q "nemo_toolkit[asr] @ git+${NEMO_REPO}@${NEMO_REF}"
else
  log "nemo already present"
fi

# 5. prefetch model (idempotent) — avoids first-request latency; non-fatal if it fails
if [ "${PREFETCH_MODEL:-1}" = "1" ]; then
  log "prefetching model (first run may take several minutes)"
  python - <<'PY' || log "model prefetch failed; will lazy-load on first request"
import os
import nemo.collections.asr as nemo_asr
name = os.environ.get("MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b")
m = nemo_asr.models.ASRModel.from_pretrained(name)
print(f"[entrypoint] prefetched {name} -> {os.environ.get('HF_HOME')}")
del m
PY
fi

log "bootstrap complete; starting: $*"
exec "$@"
