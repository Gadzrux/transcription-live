"""
Real-time Hinglish transcription server using faster-whisper + FastAPI WebSocket.

Run with:
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env before reading any env vars


# ---------------------------------------------------------------------------
# Preload NVIDIA CUDA libs from pip packages before ctranslate2 needs them.
# On Linux, dlopen() only reads LD_LIBRARY_PATH at process startup, so setting
# os.environ after that has no effect.  We use ctypes to force-load the .so
# files from the nvidia pip packages installed in the venv.
# ---------------------------------------------------------------------------
def _preload_nvidia_libs() -> None:
    """Find and preload NVIDIA shared libraries from pip-installed nvidia-* packages."""
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return

    # Collect all lib dirs under nvidia/*/lib/
    lib_dirs = sorted(nvidia_dir.glob("*/lib"))
    if not lib_dirs:
        return

    # Also add them to LD_LIBRARY_PATH for any child processes (e.g. ffmpeg won't need them,
    # but just in case any subprocess does).
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [str(d) for d in lib_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + ([ld_path] if ld_path else []))

    # Preload key libraries in dependency order
    lib_names = [
        "libnvJitLink.so*",  # nvjitlink (dependency of cublas)
        "libcublas.so*",  # cuBLAS
        "libcublasLt.so*",  # cuBLAS Lt
        "libcudnn*.so*",  # cuDNN
        "libcufft.so*",  # cuFFT
    ]
    for lib_dir in lib_dirs:
        for pattern in lib_names:
            for lib_path in sorted(lib_dir.glob(pattern)):
                if lib_path.is_file() and ".so" in lib_path.name:
                    try:
                        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass  # skip if can't load (wrong arch, etc.)


_preload_nvidia_libs()

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration  (all configurable via .env)
# ---------------------------------------------------------------------------
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # "cuda", "cpu", "auto"
COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE", "int8"
)  # "int8", "float16", "int8_float16", "auto"

# Restrict language detection to these languages only.
# Comma-separated ISO codes. Empty or "auto" = no restriction.
ALLOWED_LANGUAGES = os.getenv("WHISPER_LANGUAGES", "hi,en")

# Minimum audio duration (seconds) before we attempt transcription.
MIN_AUDIO_SECONDS = 1.5

# Throttle: minimum gap between successive transcriptions (seconds).
TRANSCRIBE_INTERVAL = float(os.getenv("TRANSCRIBE_INTERVAL", "1.5"))

# Sliding window: only transcribe the last N seconds of audio.
# Keeps latency constant regardless of how long the recording runs.
SLIDING_WINDOW_SECONDS = float(os.getenv("SLIDING_WINDOW", "15"))

# Hinglish-optimised initial prompt – biases the model towards code-switching.
# NOTE: Keep this short and natural-sounding. Long/distinctive phrases like
# "Hinglish code-mixing" WILL leak into transcription output during silence.
# INITIAL_PROMPT = "हाँ, तो मैं बता रहा था कि, actually, हमें ये काम करना है."
INITIAL_PROMPT = ""


def _parse_allowed_languages() -> set[str] | None:
    """Parse ALLOWED_LANGUAGES into a set, or None if unrestricted."""
    raw = ALLOWED_LANGUAGES.strip().lower()
    if not raw or raw == "auto":
        return None
    langs = {l.strip() for l in raw.split(",") if l.strip()}
    return langs if langs else None


LANG_WHITELIST = _parse_allowed_languages()

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the faster-whisper model once at startup."""
    global model
    logger.info(
        "Loading faster-whisper model '%s' (device=%s, compute=%s) …",
        MODEL_SIZE,
        DEVICE,
        COMPUTE_TYPE,
    )
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Transcription Live", lifespan=lifespan)

# Serve static files (HTML client)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def webm_bytes_to_pcm(webm_bytes: bytes) -> np.ndarray | None:
    """Convert accumulated webm/opus bytes to 16 kHz mono float32 PCM via ffmpeg.

    Returns a numpy float32 array, or None if conversion fails.
    """
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",  # read from stdin
                "-ar",
                "16000",  # 16 kHz
                "-ac",
                "1",  # mono
                "-f",
                "s16le",  # signed 16-bit little-endian PCM
                "-acodec",
                "pcm_s16le",
                "pipe:1",  # write to stdout
            ],
            input=webm_bytes,
            capture_output=True,
            timeout=30,
        )
        if process.returncode != 0:
            logger.warning("ffmpeg error: %s", process.stderr.decode(errors="replace"))
            return None

        pcm_s16 = np.frombuffer(process.stdout, dtype=np.int16)
        if pcm_s16.size == 0:
            return None

        # Convert to float32 in [-1, 1] range (what faster-whisper expects)
        return pcm_s16.astype(np.float32) / 32768.0

    except Exception:
        logger.exception("Failed to convert webm to PCM")
        return None


def detect_language_restricted(audio: np.ndarray) -> tuple[str, float]:
    """Detect language but only consider LANG_WHITELIST languages.

    Returns (language_code, probability).  Falls back to "hi" if nothing matches.
    """
    if LANG_WHITELIST is None:
        return (None, 0.0)

    _lang, _prob, all_probs = model.detect_language(audio)
    filtered = [(lang, prob) for lang, prob in all_probs if lang in LANG_WHITELIST]

    if not filtered:
        return ("hi", 0.0)

    best_lang, best_prob = max(filtered, key=lambda x: x[1])
    logger.info(
        "Lang detect (restricted %s): %s (%.1f%%)",
        LANG_WHITELIST,
        best_lang,
        best_prob * 100,
    )
    return (best_lang, best_prob)


def transcribe_audio(
    audio: np.ndarray,
    confirmed_text: str,
) -> dict:
    """Run faster-whisper transcription on a sliding window of audio.

    Args:
        audio: The sliding-window PCM audio (last N seconds only).
        confirmed_text: Already-confirmed transcription text from earlier audio
                        that has scrolled out of the window.
    """
    t0 = time.monotonic()

    # Detect language restricted to allowed set (hi, en)
    detected_lang, _detected_prob = detect_language_restricted(audio)

    t1 = time.monotonic()

    segments_iter, info = model.transcribe(
        audio,
        language=detected_lang,
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    segments = []
    window_text_parts = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
        )
        window_text_parts.append(seg.text.strip())

    t2 = time.monotonic()
    window_text = " ".join(window_text_parts)
    audio_dur = len(audio) / 16000.0

    logger.info(
        "⏱  detect=%.2fs  transcribe=%.2fs  total=%.2fs  audio=%.1fs  RTF=%.2f",
        t1 - t0,
        t2 - t1,
        t2 - t0,
        audio_dur,
        (t2 - t0) / audio_dur if audio_dur > 0 else 0,
    )

    # Combine confirmed (old) text with fresh window transcription
    full_text = (
        (confirmed_text + " " + window_text).strip() if confirmed_text else window_text
    )

    return {
        "text": full_text,
        "window_text": window_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the HTML client."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """WebSocket endpoint for real-time audio transcription.

    Protocol (binary frames from client -> JSON frames to client):
    - Client sends binary webm/opus audio chunks.
    - Server accumulates, converts to PCM, applies a sliding window,
      transcribes only the recent audio, and responds with JSON.
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    audio_buffer = bytearray()  # raw webm bytes from browser
    last_transcribe_time: float = 0.0
    confirmed_text: str = ""  # text from audio that scrolled past the window
    prev_window_text: str = ""  # last window transcription (for confirming)

    window_samples = int(SLIDING_WINDOW_SECONDS * 16000)

    try:
        while True:
            data = await ws.receive_bytes()
            audio_buffer.extend(data)

            now = time.monotonic()
            if now - last_transcribe_time < TRANSCRIBE_INTERVAL:
                continue

            # --- Convert accumulated webm to PCM ---
            t_ffmpeg_start = time.monotonic()
            full_pcm = webm_bytes_to_pcm(bytes(audio_buffer))
            t_ffmpeg_end = time.monotonic()

            if full_pcm is None or len(full_pcm) == 0:
                continue

            full_duration = len(full_pcm) / 16000.0
            if full_duration < MIN_AUDIO_SECONDS:
                continue

            last_transcribe_time = now

            # --- Sliding window: only transcribe the last N seconds ---
            if len(full_pcm) > window_samples:
                # Audio exceeds window — confirm text from previous window
                # and only process the tail.
                if prev_window_text:
                    confirmed_text = (confirmed_text + " " + prev_window_text).strip()
                window_pcm = full_pcm[-window_samples:]
            else:
                window_pcm = full_pcm

            logger.info(
                "⏱  ffmpeg=%.2fs  full_audio=%.1fs  window=%.1fs",
                t_ffmpeg_end - t_ffmpeg_start,
                full_duration,
                len(window_pcm) / 16000.0,
            )

            # --- Transcribe in thread pool ---
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, transcribe_audio, window_pcm, confirmed_text
            )

            prev_window_text = result.get("window_text", "")
            await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        logger.info("Cleaning up WebSocket connection.")
