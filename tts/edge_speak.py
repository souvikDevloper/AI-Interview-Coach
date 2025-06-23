# tts/edge_speak.py
# ─────────────────────────────────────────────────────────────
import asyncio, edge_tts, tempfile, os, contextlib
import soundfile as sf
import numpy as np

VOICE = os.getenv("EDGE_VOICE", "en-US-AriaNeural")   # list: edge-tts --list-voices


async def _gen(text: str, voice: str = VOICE) -> np.ndarray:
    """
    Generate speech for *text* with Edge-TTS, return waveform (float32).
    Works on Windows because the temp file is not kept open.
    """
    fd, path = tempfile.mkstemp(suffix=".mp3")   # only returns a filename
    os.close(fd)                                 # close immediately

    try:
        await edge_tts.Communicate(text, voice).save(path)
        wav, _ = sf.read(path, dtype="float32")  # load as numpy array
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)                      # silent cleanup

    return wav


def speak(text: str) -> np.ndarray:
    """Synchronous wrapper used by the Streamlit pages."""
    return asyncio.run(_gen(text))
# ─────────────────────────────────────────────────────────────
