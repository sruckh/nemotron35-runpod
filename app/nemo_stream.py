"""NeMo cache-aware streaming wrapper for Nemotron 3.5 ASR.

Verified against NeMo main source:
  examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
  nemo/collections/asr/parts/utils/streaming_utils.py  (CacheAwareStreamingAudioBuffer)

Key facts that shape this code:
  * Cache-aware models REQUIRE float32 compute (NeMo raises NotImplementedError otherwise).
    Speedup comes from `amp` autocast (float32 weights), not from a different compute dtype.
  * Nemotron is a streaming-trained model: `encoder.streaming_cfg` already exists, so we do NOT
    call setup_streaming_params(). We only pick the latency point via set_default_att_context_size.
  * CacheAwareStreamingAudioBuffer.__iter__ is a generator: it drains complete chunks and stops
    when the remaining tail is too short to yield. buffer_idx persists across iter() calls, so we
    can append_audio() then drain the newly-complete chunks incrementally — ideal for live mic input.
  * is_buffer_empty() == (buffer_idx >= buffer.size(-1)) -> drive keep_all_outputs on the last chunk.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

from app.config import N_TO_MS, Config

log = logging.getLogger("nemotron35.engine")


def _extract_transcriptions(hyps) -> list[str]:
    """conformer_stream_step may return Hypothesis objects or plain strings."""
    hyps = hyps or []
    if len(hyps) and isinstance(hyps[0], Hypothesis):
        return [h.text for h in hyps]
    return list(hyps)


class NemoStreamEngine:
    """Loads the model once per process and configures it for cache-aware streaming."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Cache-aware models force float32; autocast (amp) is the only speedup path.
        self.compute_dtype = torch.float32
        self.amp = cfg.amp
        self.amp_dtype = getattr(torch, cfg.amp_dtype)

        log.info("Loading model %s ...", cfg.model_name)
        self.model = nemo_asr.models.ASRModel.from_pretrained(cfg.model_name)

        # Pick the latency point on the multi-lookahead encoder: att_context_size = [56, N].
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(att_context_size=[56, cfg.att_context_n])
            log.info("att_context_size=[56, %d] -> %d ms chunk", cfg.att_context_n, N_TO_MS[cfg.att_context_n])
        else:
            log.warning("encoder has no set_default_att_context_size; using model default lookahead")

        # Language-ID prompt + lang-tag handling for the prompt-conditioned model.
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(cfg.target_lang)
            log.info("target_lang=%s", cfg.target_lang)
        if hasattr(self.model, "decoding") and hasattr(self.model.decoding, "set_strip_lang_tags"):
            self.model.decoding.set_strip_lang_tags(cfg.strip_lang_tags, lang_tag_pattern=None)

        self.model = self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()

        # streaming_cfg is set for streaming-trained models (no setup_streaming_params needed).
        self.streaming_cfg = self.model.encoder.streaming_cfg
        self.sample_rate = int(self.model.cfg.sample_rate)  # 16000
        log.info("Model ready on %s, sample_rate=%d, amp=%s(%s)",
                 self.device, self.sample_rate, self.amp, cfg.amp_dtype)

    def configure(self, target_lang: str | None = None, strip_lang_tags: bool | None = None) -> None:
        """Re-apply the inference language prompt and/or lang-tag stripping at runtime.

        Cheap and idempotent, but mutates GLOBAL model state — fine for sequential single-user
        use; NOT safe to change mid-stream under concurrency (one stream's chunks would shift language).
        """
        if target_lang is not None and hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(target_lang)
            log.info("target_lang=%s", target_lang)
        if (
            strip_lang_tags is not None
            and hasattr(self.model, "decoding")
            and hasattr(self.model.decoding, "set_strip_lang_tags")
        ):
            self.model.decoding.set_strip_lang_tags(strip_lang_tags, lang_tag_pattern=None)
            log.info("strip_lang_tags=%s", strip_lang_tags)

    @property
    def chunk_samples(self) -> int:
        ms = N_TO_MS.get(self.cfg.att_context_n, 320)
        return int(ms / 1000.0 * self.sample_rate)

    def silence_pad_samples(self) -> int:
        """>= one chunk so finish() fully drains the tail; floor at 1.0 s."""
        return max(int(1.0 * self.sample_rate), 2 * self.chunk_samples)

    def new_session(self) -> "StreamSession":
        return StreamSession(self)

    def transcribe_pcm(self, pcm: np.ndarray) -> str:
        """Convenience: one-shot transcribe of a full 16 kHz mono float32 array (used by tests)."""
        sess = self.new_session()
        sess.feed(pcm)
        return sess.finish()


class StreamSession:
    """One cache-aware streaming session (per mic recording / connection).

    Holds the incremental buffer + the encoder caches and RNNT hypotheses that thread across chunks.
    """

    def __init__(self, engine: NemoStreamEngine):
        self.engine = engine
        self.model = engine.model
        self.buf = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=engine.cfg.online_normalization,
            pad_and_drop_preencoded=False,
        )
        (self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len) = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step = 0
        self.text = ""
        self._stream_id = None  # resolved on first append; reused so we never spawn extra streams

    def _append(self, pcm: np.ndarray) -> None:
        """Append PCM to ONE stream.

        CacheAwareStreamingAudioBuffer treats stream_id < 0 (the -1 default) as 'create a NEW stream'
        whenever the buffer already exists — it pads the batch dim +1 each call. Calling
        append_audio(stream_id=-1) on every mic chunk therefore grows the batch (1,2,…,N) while the
        encoder cache stays batch=1 -> attention `torch.cat([cache,key])` crashes
        ("Expected size 1 but got size N").

        So: the FIRST append (buffer is None) MUST use stream_id<0, which creates stream 0 — but note
        append_processed_signal does NOT resolve -1 to the new index, it returns -1 unchanged. Derive
        the real index from streams_length (= 0) and reuse it on every later append.
        """
        if self._stream_id is None:
            self.buf.append_audio(pcm, stream_id=-1)
            self._stream_id = int(len(self.buf.streams_length)) - 1  # first stream is always index 0
        else:
            self.buf.append_audio(pcm, stream_id=self._stream_id)

    def _drop_extra(self) -> int:
        # Step 0 drops nothing; subsequent steps drop streaming_cfg.drop_extra_pre_encoded.
        if self.step == 0:
            return 0
        return self.model.encoder.streaming_cfg.drop_extra_pre_encoded

    def _drain(self) -> list[str]:
        """Iterate the buffer, running conformer_stream_step per complete chunk."""
        out: list[str] = []
        with torch.inference_mode(), torch.no_grad():
            with torch.amp.autocast(
                device_type=self.engine.device.type,
                dtype=self.engine.amp_dtype,
                enabled=self.engine.amp,
            ):
                for chunk_audio, chunk_lengths in self.buf:
                    chunk_audio = chunk_audio.to(self.engine.compute_dtype)
                    (
                        self.pred_out_stream,
                        transcribed_texts,
                        self.cache_last_channel,
                        self.cache_last_time,
                        self.cache_last_channel_len,
                        self.previous_hypotheses,
                    ) = self.model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=self.cache_last_channel,
                        cache_last_time=self.cache_last_time,
                        cache_last_channel_len=self.cache_last_channel_len,
                        keep_all_outputs=self.buf.is_buffer_empty(),
                        previous_hypotheses=self.previous_hypotheses,
                        previous_pred_out=self.pred_out_stream,
                        drop_extra_pre_encoded=self._drop_extra(),
                        return_transcription=True,
                    )
                    self.step += 1
                    texts = _extract_transcriptions(transcribed_texts)
                    out.append(texts[0] if texts else "")
        return out

    def feed(self, pcm: np.ndarray) -> str:
        """Feed raw 16 kHz mono float32 PCM; return the full running transcript so far.

        conformer_stream_step returns the CUMULATIVE hypothesis text each step (the RNNT decoder
        carries previous_hypotheses forward), so each drained chunk already holds the full running
        transcript. We therefore REPLACE self.text with the latest non-empty chunk's text — never
        append (appending would duplicate the growing transcript at every step).
        """
        if pcm is None:
            return self.text
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            return self.text
        self._append(np.ascontiguousarray(pcm))
        drained = [t for t in self._drain() if t]
        if drained:
            self.text = drained[-1]
        return self.text

    def finish(self) -> str:
        """End of stream: pad silence so the tail fully drains (keep_all_outputs on the last chunk)."""
        pad = np.zeros(self.engine.silence_pad_samples(), dtype=np.float32)
        self._append(pad)
        drained = [t for t in self._drain() if t]
        if drained:
            self.text = drained[-1]
        return self.text
