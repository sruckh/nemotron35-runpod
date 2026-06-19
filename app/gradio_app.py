"""Gradio web UI for Nemotron 3.5 ASR.

Two tabs:
  * Upload   — upload an audio file (wav/mp3/...), pick language + strip-tag option, transcribe.
               This is the robust path on a headless server (no local mic needed).
  * Microphone — live streaming from the browser's mic (works via the public link on a
               device that has a microphone).

Note: Nemotron 3.5 ASR is ASR-only (speech -> text in the SAME language). It does NOT translate
between languages. `auto` = auto-detect the spoken language and label it.
"""

from __future__ import annotations

import logging
import os
import uuid

import gradio as gr
import numpy as np

from app import config as cfg_mod
from app.audio_utils import prepare
from app.logging_setup import setup as setup_logging
from app.nemo_stream import NemoStreamEngine

log = setup_logging("nemotron35")

# Model is loaded once at process start. Chunk size is process-global (env); language is
# re-applied per transcription via ENGINE.configure() below.
CFG = cfg_mod.load()
ENGINE = NemoStreamEngine(CFG)

# auto + the 40 model-card locales (transcription-ready / broad-coverage / adaptation-ready).
LANGUAGES = [
    "auto",
    "en-US", "en-GB", "es-US", "es-ES", "fr-FR", "fr-CA", "it-IT", "pt-BR", "pt-PT",
    "nl-NL", "de-DE", "tr-TR", "ru-RU", "ar-AR", "hi-IN", "ja-JP", "ko-KR", "vi-VN", "uk-UA",
    "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK", "bg-BG", "fi-FI", "hr-HR", "sk-SK",
    "zh-CN", "hu-HU", "ro-RO", "et-EE",
    "el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH", "nn-NO",
]

# Server-side mic sessions keyed by an id stored in gr.State (nothing GPU-heavy serialized by the queue).
_SESSIONS: dict[str, "StreamSession"] = {}  # noqa: F821
STREAM_EVERY_S = float(os.environ.get("STREAM_EVERY_S", "0.5"))
STREAM_TIME_LIMIT_S = int(os.environ.get("STREAM_TIME_LIMIT_S", "600"))


# --------------------------------------------------------------------------- upload (full-file)
def upload_transcribe(audio, language, strip_tags):
    if audio is None:
        return "", "No audio uploaded."
    try:
        sr, y = audio
        if y is None or getattr(y, "size", 0) == 0:
            return "", "Uploaded audio is empty."
        ENGINE.configure(target_lang=language, strip_lang_tags=bool(strip_tags))
        pcm = prepare(y, sr, ENGINE.sample_rate)
        seconds = len(pcm) / ENGINE.sample_rate
        log.info("upload transcribe: %.1fs @ %dHz -> 16k mono, lang=%s", seconds, sr, language)
        text = ENGINE.transcribe_pcm(pcm).strip()
        info = f"transcribed {seconds:.1f}s · language={language} · strip_lang_tags={strip_tags}"
        return text or "(no speech detected)", info
    except Exception as e:  # pragma: no cover
        log.exception("upload_transcribe failed")
        return "", f"error: {e}"


# --------------------------------------------------------------------------- microphone (live)
def _get_or_create(session_id):
    if session_id and session_id in _SESSIONS:
        return session_id, _SESSIONS[session_id]
    sid = uuid.uuid4().hex
    sess = ENGINE.new_session()
    _SESSIONS[sid] = sess
    log.info("new mic session %s (active=%d)", sid, len(_SESSIONS))
    return sid, sess


def on_chunk(session_id, chunk):
    if not session_id and (chunk is None):
        return session_id, ""
    sid, sess = _get_or_create(session_id)
    try:
        sr, y = chunk
        if y is None or getattr(y, "size", 0) == 0:
            return sid, sess.text
        pcm = prepare(y, sr, ENGINE.sample_rate)
        return sid, sess.feed(pcm)
    except Exception as e:
        log.exception("on_chunk failed")
        return sid, (sess.text + f"\n[error: {e}]")


def on_stop(session_id):
    sess = _SESSIONS.pop(session_id, None)
    text = sess.finish() if sess is not None else ""
    log.info("mic session %s finalized (active=%d)", session_id, len(_SESSIONS))
    return text, None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Nemotron 3.5 ASR") as demo:
        gr.Markdown(
            f"# Nemotron 3.5 ASR\n"
            f"Multilingual speech-to-text (`{CFG.model_name}`). **Transcription only** — it does not "
            f"translate between languages. `auto` detects the spoken language.\n\n"
            f"Active chunk: {CFG.chunk_ms} ms."
        )
        with gr.Tab("Upload audio file"):
            gr.Markdown("Upload a wav/mp3/etc. Choose a language (or `auto` to detect) and transcribe.")
            with gr.Row():
                up_audio = gr.Audio(sources=["upload"], type="numpy", label="Audio file")
            with gr.Row():
                up_lang = gr.Dropdown(LANGUAGES, value="auto", label="Language", scale=2)
                up_strip = gr.Checkbox(value=True, label="Strip language tag from output", scale=2)
                up_btn = gr.Button("Transcribe", variant="primary", scale=1)
            up_text = gr.Textbox(label="Transcript", lines=10)
            up_info = gr.Markdown("")
            up_btn.click(upload_transcribe, [up_audio, up_lang, up_strip], [up_text, up_info])

        with gr.Tab("Microphone (live)"):
            gr.Markdown(
                "Streams from your browser's microphone (works via the public link on a device with a "
                "mic). Uses the startup language (`TARGET_LANG`) for live mode."
            )
            mic_audio = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                 label="Speak — transcript streams as you talk")
            mic_text = gr.Textbox(label="Transcript", lines=8, placeholder="…")
            session_id = gr.State(value=None)
            clear = gr.Button("Clear")
            mic_audio.stream(
                on_chunk, [session_id, mic_audio], [session_id, mic_text],
                stream_every=STREAM_EVERY_S, time_limit=STREAM_TIME_LIMIT_S,
            )
            mic_audio.stop_recording(on_stop, [session_id], [mic_text, session_id])
            clear.click(lambda: ("", None), outputs=[mic_text, session_id])
    return demo


def main():
    demo = build_demo()
    auth = None
    if CFG.gradio_auth and ":" in CFG.gradio_auth:
        u, p = CFG.gradio_auth.split(":", 1)
        auth = (u, p)
    log.info("launching Gradio on 0.0.0.0:%d share=%s", CFG.gradio_port, CFG.share_link)
    demo.queue(default_concurrency_limit=CFG.max_concurrent_streams).launch(
        server_name="0.0.0.0",
        server_port=CFG.gradio_port,
        share=CFG.share_link,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
