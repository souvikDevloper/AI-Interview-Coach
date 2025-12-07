"""
app_utils.py – tiny helpers for the AI-Interviewer app
• ensure_punkt()  → guarantee NLTK’s Punkt tokenizer is available
• switch_page()   → wrapper around Streamlit’s public st.switch_page()
"""
from __future__ import annotations

import contextlib
import nltk
import streamlit as st

# ────────────────────── NLTK bootstrap ────────────────────────
def ensure_punkt() -> None:
    """Download Punkt resources once if missing."""
    for pkg in ("punkt", "punkt_tab"):
        with contextlib.suppress(LookupError):
            nltk.data.find(f"tokenizers/{pkg}")
            continue
        nltk.download(pkg, quiet=True)

# Run on import so all pages are safe
ensure_punkt()

# ───────────────────── local Ollama guard ─────────────────────
def require_ollama() -> bool:
    """Ping the local Ollama daemon and guide the user if it's offline."""
    import http.client

    try:
        conn = http.client.HTTPConnection("localhost", 11434, timeout=2)
        conn.request("GET", "/api/tags")
        res = conn.getresponse()
        return 200 <= res.status < 500
    except Exception:
        st.error(
            "Start the **Ollama** daemon before launching the app.\n\n"
            "- Install from https://ollama.com/download\n"
            "- Run `ollama serve` in a terminal\n"
            "- Pull a chat model, e.g. `ollama pull llama3.1`"
        )
        return False

# ───────────────────── page switch wrapper ─────────────────────
def switch_page(page_name_or_path: str) -> None:
    """
    Prefer Streamlit’s public API. You can pass either:
      • a path like 'pages/Professional Screen.py'
      • (not recommended anymore) a page title
    """
    # If it looks like a file path in /pages, call the public API directly.
    if ("/" in page_name_or_path) or (page_name_or_path.lower().endswith(".py")):
        if hasattr(st, "switch_page"):
            st.switch_page(page_name_or_path)
        else:
            st.warning("Your Streamlit build doesn’t expose `st.switch_page`. Click below:")
            st.page_link(page_name_or_path, label=f"➡️ {page_name_or_path}")
            st.stop()
        return

    # Legacy title-based usage: map common titles to the file path.
    title_map = {
        "professional screen": "pages/Professional Screen.py",
        "resume screen":       "pages/Resume Screen.py",
        "behavioral screen":   "pages/Behavioral Screen.py",
        "homepage":            "Homepage.py",
    }
    key = page_name_or_path.strip().lower()
    if key in title_map:
        return switch_page(title_map[key])

    raise ValueError(
        f"Unknown page '{page_name_or_path}'. "
        "Pass a path under /pages or one of: "
        f"{', '.join(sorted(title_map))}"
    )
