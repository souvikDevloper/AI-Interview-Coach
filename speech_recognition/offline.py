# speech_recognition/offline.py
# ---------------------------------------------------------------
# Whisper (faster-whisper) with configurable CUDA/CPU selection.
# Model loading is lazy so importing the application does not allocate a GPU
# or download a model before speech recognition is used.
from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

_model: Optional[Any] = None
_model_lock = threading.Lock()


def _candidate_configs() -> List[Dict[str, Any]]:
    """Return deterministic device configurations from environment settings."""
    device = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    compute = os.getenv("WHISPER_COMPUTE", "auto").strip().lower()
    allow_fallback = os.getenv("WHISPER_ALLOW_FALLBACK", "1").strip() not in {
        "0",
        "false",
        "no",
    }
    cpu_threads = max(1, (os.cpu_count() or 2) // 2)

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("WHISPER_DEVICE must be one of: auto, cuda, cpu")

    if device == "cpu":
        return [{"device": "cpu", "compute_type": "int8" if compute == "auto" else compute,
                 "cpu_threads": cpu_threads}]

    cuda_cfg: Dict[str, Any] = {
        "device": "cuda",
        "device_index": int(os.getenv("WHISPER_DEVICE_INDEX", "0")),
        "compute_type": "float16" if compute == "auto" else compute,
    }
    if device == "cuda" and not allow_fallback:
        return [cuda_cfg]

    return [
        cuda_cfg,
        {"device": "cpu", "compute_type": "int8", "cpu_threads": cpu_threads},
    ]


def _make_model() -> Any:
    """Build the configured model, with an optional CPU fallback."""
    from faster_whisper import WhisperModel

    last_err: Optional[Exception] = None
    model_name = os.getenv("WHISPER_MODEL", "small.en").strip() or "small.en"
    for cfg in _candidate_configs():
        try:
            return WhisperModel(model_name, **cfg)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to initialize Whisper: {last_err!r}")


def get_model() -> Any:
    """Create the Whisper model once, on first transcription."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _make_model()
    return _model


def save_wav_file(path: str, raw_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(raw_bytes)


def transcribe(wav_path: str, lang: str = "en") -> str:
    segments, _info = get_model().transcribe(
        wav_path,
        language=lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 450},
        condition_on_previous_text=False,   # avoids repetition/drift
        beam_size=1,                        # latency > tiny accuracy bump
        temperature=0.0,
        word_timestamps=False,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text)


def transcribe_audio(raw_bytes: bytes, lang: str = "en") -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
        fp.write(raw_bytes)
        fp.flush()
        return transcribe(fp.name, lang=lang)
# ---------------------------------------------------------------
