# nemotron35-runpod

NVIDIA **Nemotron 3.5 ASR streaming 0.6B** (`nvidia/nemotron-3.5-asr-streaming-0.6b`) served as a
persistent **RunPod GPU pod** with a **Gradio web UI** for real-time microphone→transcript streaming,
built on NeMo's cache-aware streaming. The slim image is built in CI and pushed to Docker Hub
(`gemneye/nemotron35-runpod`).

## How it works

- **Slim image, runtime install.** The Docker image contains only Python + light server deps
  (gradio, numpy, soundfile). `entrypoint.sh` installs the heavy stack at container start:
  `ffmpeg/libsndfile` → `Cython` → **torch 2.9.1 (cu128)** → `nemo_toolkit[asr]` from git → model
  prefetch. Everything is idempotent, so pod restarts are cheap after the first run.
- **Cache-aware streaming.** Audio is fed to NeMo's `CacheAwareStreamingAudioBuffer` and decoded
  chunk-by-chunk via `model.conformer_stream_step(...)`, threading encoder caches + RNNT hypotheses.
  Only new mic audio is processed per step — no redundant recompute.
- **Gradio UI + public link.** `gr.Audio(streaming=True)` mic input → streaming `Textbox` output.
  `SHARE_LINK=true` prints a temporary `https://<id>.gradio.live` URL to the pod logs, bypassing
  the (unreliable) RunPod proxy.

```
mic chunk -> prepare (->16k mono f32) -> StreamSession.feed() -> CacheAwareStreamingAudioBuffer
   -> conformer_stream_step (caches threaded) -> incremental text -> Gradio Textbox
stop rec -> StreamSession.finish() (silence-pad drain, keep_all_outputs) -> final text
```

## Files

| Path | Role |
| --- | --- |
| `app/nemo_stream.py` | `NemoStreamEngine` (model load + setup) + `StreamSession` (verified `conformer_stream_step` loop). The core. |
| `app/gradio_app.py` | Gradio UI: mic streaming → partial/final transcript; `share=True`; port 7860. |
| `app/config.py` | Env-var config + validation. |
| `app/audio_utils.py` | Downmix/resample to 16 kHz mono float32 (torchaudio at runtime). |
| `entrypoint.sh` | Idempotent runtime bootstrap (apt→torch→NeMo→prefetch) then `exec` server. |
| `Dockerfile` | Slim `nvidia/cuda:12.9.0-runtime-ubuntu24.04` base; light deps only; `EXPOSE 7860`. |
| `.github/workflows/docker-publish.yml` | CI: buildx build + push `gemneye/nemotron35-runpod` to Docker Hub. |
| `tests/test_transcribe.py` | Offline verification (feed a wav through the engine). |

## Build & publish (CI)

Push to `main` (or a `v*` tag) on `github.com/sruckh/nemotron35-runpod` triggers the GitHub Action,
which builds `linux/amd64` on `ubuntu-latest` (no GPU needed) and pushes to `gemneye/nemotron35-runpod`.
Required repo secrets: `DOCKER_USERNAME`, `DOCKER_PASSWORD` (push access to the `gemneye` namespace).

Local build (optional): `docker build -t nemotron35-runpod .`

## Deploy on RunPod

1. Create a pod from Docker Hub image `gemneye/nemotron35-runpod:latest`, GPU on
   (Ampere→Blackwell; H100 reference).
2. **Mount a network volume at `/workspace`** (strongly recommended) so the torch/NeMo/model caches
   persist — first boot downloads ~6 GB; restarts skip it.
3. Expose port **7860**. Set env vars as needed (see table below).
4. Start. Tail logs: watch `[entrypoint]` lines (apt → torch → NeMo → prefetch), then the
   `Running on local URL: http://0.0.0.0:7860` and `Running on public URL: https://<id>.gradio.live`
   lines. **Open the `.gradio.live` URL** (it bypasses the RunPod proxy).

## Configuration (env)

Chunk size and language are **process-global** (applied once at model load); restart the pod to change.

| Var | Default | Meaning |
| --- | --- | --- |
| `MODEL_NAME` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | HF model id |
| `DEVICE` | `cuda` | torch device |
| `ATT_CONTEXT_N` | `3` | right context N ∈ {0,1,3,6,13} → 80/160/320/560/1120 ms chunk |
| `TARGET_LANG` | `auto` | locale (`en-US`) or `auto` (detect + tag) |
| `STRIP_LANG_TAGS` | `true` | remove the trailing `<xx-XX>` tag |
| `ONLINE_NORMALIZATION` | `true` | per-chunk normalize |
| `AMP` | `false` | autocast speedup (cache-aware models need float32 weights) |
| `AMP_DTYPE` | `bfloat16` | `bfloat16` \| `float16` (only when `AMP=true`) |
| `GRADIO_PORT` | `7860` | Gradio port |
| `SHARE_LINK` | `true` | create public `.gradio.live` link |
| `GRADIO_AUTH` | (unset) | optional `user:pass` gate |
| `MAX_CONCURRENT_STREAMS` | `8` | Gradio queue concurrency |
| `STREAM_EVERY_S` | `0.5` | seconds between mic chunks |
| `STREAM_TIME_LIMIT_S` | `600` | max seconds per recording |
| `PREFETCH_MODEL` | `1` | prefetch model at boot |
| `MODEL_CACHE` / `PIP_CACHE` | `/workspace/hf_cache` / `/workspace/pip_cache` | caches (network volume) |
| `NEMO_REF` | `main` | NeMo ref (pin to a tag once 26.06 ships) |
| `HF_TOKEN` | (unset) | only if the model is gated |

## Verify

- `python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"`
  → `2.9.1+cu128 12.8 True`.
- Offline: `python -m tests.test_transcribe /path/to/sample.wav` → prints a non-empty transcript.
- Online: open the `.gradio.live` URL, allow mic, speak → transcript streams live; stops on release.

## Notes & risks

- **Cold start:** first boot pays the runtime install + model download (~8–10 min). Mount the
  network volume to make restarts <60 s.
- **Public link is temporary:** the `.gradio.live` URL changes each launch — capture it from logs.
  For a stable URL, add a fixed tunnel (cloudflared/ngrok) later.
- **NeMo `@main`:** the only branch with this 2026-06 model until 26.06 tags; pin `NEMO_REF` then.
- **Ubuntu 24.04:** if a NeMo ASR dep lacks a 24.04 wheel, switch the Dockerfile `FROM` to
  `nvidia/cuda:12.9.0-runtime-ubuntu22.04`.
- **License:** model is OpenMDW-1.1 (commercial use OK).

## License

Code: see repository. Model weights: OpenMDW-1.1 (NVIDIA).
