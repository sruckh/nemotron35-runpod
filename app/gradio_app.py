"""Gradio web UI: real-time microphone streaming -> partial/final transcript.

Design:
  * Each recording owns a server-side StreamSession (kept in _SESSIONS, keyed by an id stored in
    gr.State). We store only the id in Gradio state so nothing GPU-heavy is serialized by the queue.
  * audio.stream() fires on each mic chunk -> feed ONLY the new chunk to the cache-aware session
    (not the whole history) -> streaming partial transcript.
  * audio.stop_recording() flushes the session (silence-pad drain) -> final transcript, then frees it.

Public link: SHARE_LINK=1 -> Gradio prints a temporary https://<id>.gradio.live URL to the logs
(capture it from pod logs; it bypasses the unreliable RunPod proxy).
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

# --- model is loaded once at process start (chunk size + lang are process-global) ---
CFG = cfg_mod.load()
ENGINE = NemoStreamEngine(CFG)

# server-side sessions; id-only in Gradio state. Cleaned up on stop. (v1: in-memory, per-pod)
_SESSIONS: dict[str, "StreamSession"] = {}  # noqa: F821
STREAM_EVERY_S = float(os.environ.get("STREAM_EVERY_S", "0.5"))
STREAM_TIME_LIMIT_S = int(os.environ.get("STREAM_TIME_LIMIT_S", "600"))


def _get_or_create(session_id):
    if session_id and session_id in _SESSIONS:
        return session_id, _SESSIONS[session_id]
    sid = uuid.uuid4().hex
    sess = ENGINE.new_session()
    _SESSIONS[sid] = sess
    log.info("new session %s (active=%d)", sid, len(_SESSIONS))
    return sid, sess


def on_chunk(session_id, chunk):
    """chunk == (sample_rate, np.ndarray) from the mic; None/empty before first audio."""
    if not session_id and (chunk is None):
        return session_id, ""
    sid, sess = _get_or_create(session_id)
    try:
        sr, y = chunk
        if y is None or getattr(y, "size", 0) == 0:
            return sid, sess.text
        pcm = prepare(y, sr, ENGINE.sample_rate)
        text = sess.feed(pcm)
        return sid, text
    except Exception as e:  # keep the stream alive; surface in transcript
        log.exception("on_chunk failed")
        return sid, (sess.text + f"\n[error: {e}]")


def on_stop(session_id):
    sess = _SESSIONS.pop(session_id, None)
    text = sess.finish() if sess is not None else ""
    log.info("session %s finalized (active=%d)", session_id, len(_SESSIONS))
    # clear the Gradio-side id so the next recording starts fresh
    return text, None


def build_demo() -> gr.Blocks:
    chunk_ms = CFG.chunk_ms
    with gr.Blocks(title="Nemotron 3.5 ASR") as demo:
        gr.Markdown(
            f"# Nemotron 3.5 ASR — streaming\n"
            f"Model `{CFG.model_name}` · chunk `{chunk_ms} ms` · target_lang `{CFG.target_lang}` · "
            f"strip_lang_tags `{CFG.strip_lang_tags}`"
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Speak — transcript streams as you talk",
            )
        transcript = gr.Textbox(label="Transcript", lines=8, placeholder="…")
        session_id = gr.State(value=None)
        clear = gr.Button("Clear")

        audio.stream(
            on_chunk,
            inputs=[session_id, audio],
            outputs=[session_id, transcript],
            stream_every=STREAM_EVERY_S,
            time_limit=STREAM_TIME_LIMIT_S,
        )
        audio.stop_recording(on_stop, inputs=[session_id], outputs=[transcript, session_id])
        clear.click(lambda: ("", None), outputs=[transcript, session_id])
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
