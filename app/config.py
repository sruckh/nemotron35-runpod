"""Environment-driven configuration (single source of truth).

Chunk size and language are PROCESS-GLOBAL: they are applied once at model load
(set_default_att_context_size / set_inference_prompt mutate global encoder state), so a
per-connection override would corrupt concurrent sessions. Change them via env and restart.
"""

import os
from dataclasses import dataclass

# att_context_size right-context N -> effective chunk length in ms
# att_context_size = [56, N], N in this set
N_TO_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}
_VALID_N = set(N_TO_MS)

_BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in _BOOL_TRUE


@dataclass(frozen=True)
class Config:
    model_name: str
    device: str
    att_context_n: int            # right context N -> chunk ms (see N_TO_MS)
    target_lang: str              # locale e.g. "en-US", or "auto"
    strip_lang_tags: bool
    online_normalization: bool
    amp: bool                     # autocast (float32 weights) speedup
    amp_dtype: str                # "bfloat16" | "float16" (only when amp=True)
    gradio_port: int
    share_link: bool              # create public .gradio.live link
    gradio_auth: str              # "" or "user:pass"
    max_concurrent_streams: int
    nemo_ref: str

    @property
    def chunk_ms(self) -> int:
        return N_TO_MS[self.att_context_n]


def load() -> Config:
    n = int(os.environ.get("ATT_CONTEXT_N", "3"))
    if n not in _VALID_N:
        raise ValueError(
            f"ATT_CONTEXT_N={n} invalid; must be one of {sorted(_VALID_N)} "
            f"(ms: {N_TO_MS})"
        )

    amp_dtype_raw = os.environ.get("AMP_DTYPE", "bfloat16").strip().lower()
    if amp_dtype_raw in ("bfloat16", "bf16"):
        amp_dtype = "bfloat16"
    elif amp_dtype_raw in ("float16", "fp16", "half"):
        amp_dtype = "float16"
    else:
        raise ValueError("AMP_DTYPE must be 'bfloat16' or 'float16'")

    return Config(
        model_name=os.environ.get("MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
        device=os.environ.get("DEVICE", "cuda"),
        att_context_n=n,
        target_lang=os.environ.get("TARGET_LANG", "auto"),
        strip_lang_tags=_bool(os.environ.get("STRIP_LANG_TAGS"), True),
        online_normalization=_bool(os.environ.get("ONLINE_NORMALIZATION"), True),
        amp=_bool(os.environ.get("AMP"), False),
        amp_dtype=amp_dtype,
        gradio_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share_link=_bool(os.environ.get("SHARE_LINK"), True),
        gradio_auth=os.environ.get("GRADIO_AUTH", ""),
        max_concurrent_streams=int(os.environ.get("MAX_CONCURRENT_STREAMS", "8")),
        nemo_ref=os.environ.get("NEMO_REF", "main"),
    )
