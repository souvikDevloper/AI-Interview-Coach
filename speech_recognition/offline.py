# speech_recognition/offline.py
# --------------------------------------------------------------------
from faster_whisper import WhisperModel
import tempfile, soundfile as sf, os

# ── load the 140-MB “small” model once (quantised 8-bit kernels) ─────
model = WhisperModel(
    "small",
    device="cpu",                # use "cpu" if you have no GPU
    compute_type="int8"    # fits in <2 GB VRAM on RTX 3060
)

# ── helper 1: write microphone bytes to a .wav file ──────────────────
def save_wav_file(path: str, raw_bytes: bytes) -> None:
    """Store raw PCM/MP3 bytes (from audio_recorder) as a WAV container."""
    with open(path, "wb") as f:
        f.write(raw_bytes)

# ── helper 2: transcribe an existing .wav file on disk ───────────────
def transcribe(wav_path: str, lang: str = "en") -> str:
    """Return plain text transcription of wav_path."""
    # faster-whisper can read a filepath directly
    segments, _ = model.transcribe(wav_path, language=lang)
    return " ".join(seg.text for seg in segments)

# ── helper 3: transcribe raw bytes without keeping a file ------------
def transcribe_audio(raw_bytes: bytes, lang: str = "en") -> str:
    """
    Convenience: feed recorder bytes, get text, no file left behind.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
        fp.write(raw_bytes)
        fp.flush()
        return transcribe(fp.name, lang=lang)
# --------------------------------------------------------------------
