"""Offline verification — transcribe audio through the cache-aware engine (no browser needed).

Run ON THE POD after entrypoint.sh has installed deps (GPU + NeMo present):

    python -m tests.test_transcribe /path/to/sample.wav   # real speech -> asserts non-empty
    python -m tests.test_transcribe                        # generates a sine tone (pipeline smoke test)

A sine tone usually yields empty/garbled text; pass a real speech wav for a meaningful assertion.
"""

import sys
from pathlib import Path

import numpy as np

from app import config as cfg_mod
from app.audio_utils import prepare
from app.logging_setup import setup as setup_logging
from app.nemo_stream import NemoStreamEngine


def _tone(seconds: float = 3.0, sr: int = 16000, freq: float = 220.0) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main(argv: list[str]) -> int:
    log = setup_logging("nemotron35.test")
    cfg = cfg_mod.load()
    engine = NemoStreamEngine(cfg)

    wav = Path(argv[1]) if len(argv) > 1 else None
    if wav is not None:
        import soundfile as sf
        audio, sr = sf.read(str(wav), dtype="float32")
        pcm = prepare(audio, sr, engine.sample_rate)
        log.info("loaded %s (%.1fs @ %dHz)", wav, len(pcm) / engine.sample_rate, sr)
    else:
        pcm = _tone()
        log.info("no wav given; using %.1fs sine tone (smoke test)", len(pcm) / engine.sample_rate)

    text = engine.transcribe_pcm(pcm).strip()
    print("TRANSCRIPT:", repr(text))

    if wav is not None:
        assert text, "expected non-empty transcript for a speech file"
        print("OK: non-empty transcript")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
