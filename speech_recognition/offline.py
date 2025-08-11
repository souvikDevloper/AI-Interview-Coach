# speech_recognition/offline.py
# ---------------------------------------------------------------
# Whisper (faster-whisper) with CUDA on RTX 3060 → CPU fallback.
# Low-latency settings for interview UX.
from __future__ import annotations

import os
import tempfile
from typing import Optional

from faster_whisper import WhisperModel


def _make_model() -> WhisperModel:
    """
    Prefer GPU (float16) on RTX 3060.
    If that fails (no CUDA/cuDNN), fall back to CPU int8.
    """
    last_err: Optional[Exception] = None
    attempts = [
        dict(device="cuda", device_index=0, compute_type="float16"),
        dict(device="cpu",  compute_type="int8", cpu_threads=max(1, os.cpu_count() // 2)),
    ]
    for cfg in attempts:
        try:
            # English-only model → a bit faster & smaller than multilingual
            return WhisperModel("small.en", **cfg)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to initialize Whisper: {last_err!r}")


_model = _make_model()


def save_wav_file(path: str, raw_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(raw_bytes)


def transcribe(wav_path: str, lang: str = "en") -> str:
    segments, _info = _model.transcribe(
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
