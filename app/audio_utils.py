"""Audio normalization + resampling to the model's expected 16 kHz mono float32 PCM.

torchaudio is a RUNTIME dependency (installed by entrypoint.sh), so these helpers are
only used after the bootstrap has completed. A dependency-free linear-resample fallback
keeps the module importable even if torchaudio is absent.
"""

import numpy as np

try:
    import torchaudio  # runtime dep
    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover - exercised only without runtime deps
    _HAS_TORCHAUDIO = False


def to_float32_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix to mono and scale integer PCM to float32 in roughly [-1, 1]."""
    audio = np.asarray(audio)
    if audio.ndim > 1:
        # (frames, channels) -> mono
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(int(info.min)), abs(int(info.max)))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if _HAS_TORCHAUDIO:
        import torch
        t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_freq=orig_sr, new_freq=target_sr)
        return np.ascontiguousarray(t.squeeze(0).numpy(), dtype=np.float32)
    # fallback: linear interpolation (no extra deps)
    n = int(round(len(audio) * target_sr / orig_sr))
    idx = np.arange(n, dtype=np.float64) * (orig_sr / target_sr)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def prepare(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Full pipeline: -> mono float32 @ target_sr. Returns 1D contiguous float32."""
    return resample(to_float32_mono(audio), sr, target_sr)
