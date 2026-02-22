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

# Chunk size: minimum new audio (seconds) to transcribe per cycle.
# Each chunk is transcribed exactly once and appended — no overlapping or repetition.
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "3.0"))

# Hinglish-optimised initial prompt – biases the model towards code-switching.
# Use natural Hinglish examples so the model learns the style without echoing
# meta-descriptions like "Hinglish code-mixing" into the output.
INITIAL_PROMPT = (
    "Namaste, aaj hum discuss karenge. "
    "Toh basically yeh cheez hai ki hum log Hindi aur English dono use karte hain. "
    "Achha, so let me explain. Yeh bahut important hai."
)

# Phrases that Whisper may echo from the prompt; strip them from output.
# Includes both old and new prompt fragments plus common hallucinations.
PROMPT_LEAKAGE_PHRASES = [
    # Old prompt leakage
    "Hindi English Hinglish code-mixing",
    "Hinglish code-mixing",
    "Hinglish code mixing",
    "code-mixing",
    "code mixing",
    "This is a Hinglish conversation with both Hindi and English",
    "This is a Hinglish conversation with both Hindi and English words",
    "Yeh ek Hindi aur English mixed conversation hai",
    "Main abhi transcription test kar raha hoon",
    # New prompt leakage
    "Namaste, aaj hum discuss karenge",
    "Toh basically yeh cheez hai ki hum log Hindi aur English dono use karte hain",
    "Achha, so let me explain",
    "Yeh bahut important hai",
    # Common Whisper hallucinations on silence/noise
    "Thank you for watching",
    "Thanks for watching",
    "Please subscribe",
    "Subtitles by",
    "ご視聴ありがとうございました",
    "MBC 뉴스",
]


def _parse_allowed_languages() -> set[str] | None:
    """Parse ALLOWED_LANGUAGES into a set, or None if unrestricted."""
    raw = ALLOWED_LANGUAGES.strip().lower()
    if not raw or raw == "auto":
        return None
    langs = {l.strip() for l in raw.split(",") if l.strip()}
    return langs if langs else None


LANG_WHITELIST = _parse_allowed_languages()


def _strip_prompt_leakage(text: str) -> str:
    """Remove known prompt phrases that Whisper may echo in transcription."""
    result = text.strip()
    for phrase in PROMPT_LEAKAGE_PHRASES:
        # Case-insensitive, remove phrase and surrounding punctuation/spaces
        lower = result.lower()
        idx = lower.find(phrase.lower())
        while idx >= 0:
            start = idx
            end = idx + len(phrase)
            # Extend to trim leading/trailing punctuation and whitespace
            while start > 0 and result[start - 1] in " .,;:!?":
                start -= 1
            while end < len(result) and result[end] in " .,;:!?":
                end += 1
            result = (result[:start] + " " + result[end:]).strip()
            lower = result.lower()
            idx = lower.find(phrase.lower())
    return result.strip()


def _is_hallucination(text: str) -> bool:
    """Detect likely hallucinated output (repetitive or nonsensical)."""
    cleaned = text.strip()
    if not cleaned:
        return True
    # If the same short phrase repeats 3+ times, it's likely hallucination
    words = cleaned.split()
    if len(words) >= 6:
        # Check for repeating bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        from collections import Counter
        counts = Counter(bigrams)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 3 and most_common_count / len(bigrams) > 0.4:
            return True
    return False


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

    For Hinglish (mixed Hindi+English), we use "hi" as default because:
    - Whisper's Hindi mode handles Devanagari and romanised Hindi well
    - English words embedded in Hindi speech are still transcribed correctly
    - Using "en" would miss Hindi-script words entirely

    Returns (language_code, probability).  Falls back to "hi" if nothing matches.
    """
    if LANG_WHITELIST is None:
        return (None, 0.0)

    _lang, _prob, all_probs = model.detect_language(audio)
    filtered = {lang: prob for lang, prob in all_probs if lang in LANG_WHITELIST}

    if not filtered:
        return ("hi", 0.0)

    best_lang = max(filtered, key=filtered.get)
    best_prob = filtered[best_lang]

    # For Hinglish: if Hindi and English are close (within 20pp), prefer Hindi
    # because Whisper Hindi mode handles code-mixed text better.
    hi_prob = filtered.get("hi", 0.0)
    en_prob = filtered.get("en", 0.0)
    if best_lang == "en" and hi_prob > 0.15 and (en_prob - hi_prob) < 0.20:
        logger.info(
            "Lang detect: en=%.1f%% hi=%.1f%% → overriding to 'hi' for Hinglish",
            en_prob * 100,
            hi_prob * 100,
        )
        return ("hi", hi_prob)

    logger.info(
        "Lang detect (restricted %s): %s (%.1f%%)",
        LANG_WHITELIST,
        best_lang,
        best_prob * 100,
    )
    return (best_lang, best_prob)


def transcribe_chunk(audio: np.ndarray) -> dict:
    """Run faster-whisper transcription on a single chunk of audio.

    Args:
        audio: PCM audio chunk (no overlap with previously transcribed audio).
    """
    t0 = time.monotonic()

    # Detect language restricted to allowed set (hi, en)
    detected_lang, _detected_prob = detect_language_restricted(audio)

    t1 = time.monotonic()

    segments_iter, info = model.transcribe(
        audio,
        language=detected_lang,
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,  # No previous context — avoids repetition
        beam_size=5,
        best_of=3,
        patience=1.5,
        temperature=[0.0, 0.2, 0.4],  # fallback temperatures for robustness
        compression_ratio_threshold=2.4,  # default; reject overly compressed (repetitive) text
        log_prob_threshold=-1.0,  # reject low-confidence outputs
        no_speech_threshold=0.6,  # skip segments likely to be silence
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,
            speech_pad_ms=200,  # pad detected speech regions
            threshold=0.35,  # slightly lower VAD threshold to catch softer speech
        ),
    )

    segments = []
    chunk_text_parts = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
        )
        chunk_text_parts.append(seg.text.strip())

    t2 = time.monotonic()
    chunk_text = " ".join(chunk_text_parts)
    chunk_text = _strip_prompt_leakage(chunk_text)

    # Drop hallucinated or empty output
    if _is_hallucination(chunk_text):
        logger.warning("Dropped likely hallucinated chunk: %r", chunk_text[:120])
        chunk_text = ""

    audio_dur = len(audio) / 16000.0

    logger.info(
        "⏱  detect=%.2fs  transcribe=%.2fs  total=%.2fs  audio=%.1fs  RTF=%.2f",
        t1 - t0,
        t2 - t1,
        t2 - t0,
        audio_dur,
        (t2 - t0) / audio_dur if audio_dur > 0 else 0,
    )

    return {
        "chunk_text": chunk_text,
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
    - Server accumulates, converts to PCM, and transcribes only NEW audio
      (chunk-based, no overlap). Each chunk is transcribed exactly once.
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    audio_buffer = bytearray()  # raw webm bytes from browser
    last_transcribe_time: float = 0.0
    full_text: str = ""  # accumulated transcript (no repetition)
    transcribed_until_sample: int = 0  # samples we've already transcribed

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

            # --- Chunk-based: only transcribe NEW audio not yet transcribed ---
            new_start = transcribed_until_sample
            new_end = len(full_pcm)
            new_samples = new_end - new_start
            new_duration = new_samples / 16000.0

            if new_duration < CHUNK_SECONDS:
                # Not enough new audio yet; skip (avoid tiny partial transcriptions)
                continue

            last_transcribe_time = now

            chunk_pcm = full_pcm[new_start:new_end]
            transcribed_until_sample = new_end

            logger.info(
                "⏱  ffmpeg=%.2fs  full_audio=%.1fs  new_chunk=%.1fs",
                t_ffmpeg_end - t_ffmpeg_start,
                full_duration,
                new_duration,
            )

            # --- Transcribe in thread pool ---
            loop = asyncio.get_running_loop()
            chunk_result = await loop.run_in_executor(None, transcribe_chunk, chunk_pcm)

            chunk_text = chunk_result.get("chunk_text", "")
            if chunk_text:
                full_text = (full_text + " " + chunk_text).strip()

            # Response format matches frontend expectations (text, window_text, segments)
            result = {
                "text": full_text,
                "window_text": chunk_text,  # current chunk for live display
                "language": chunk_result.get("language"),
                "language_probability": chunk_result.get("language_probability"),
                "segments": chunk_result.get("segments", []),
            }
            await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        logger.info("Cleaning up WebSocket connection.")
