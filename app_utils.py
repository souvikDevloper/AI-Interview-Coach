# app_utils.py ────────────────────────────────────────────────────────────
"""
Utility helpers for the Streamlit AI‑Interviewer project
─────────────────────────────────────────────────────────
• ensure_punkt() → make sure the NLTK sentence‑tokenizer data is available  
• switch_page()  → programmatically jump between multipage Streamlit files
"""

from __future__ import annotations

import contextlib
import nltk


# ────────────────────────── NLTK setup ────────────────────────────────
def ensure_punkt() -> None:
    """Guarantee that NLTK’s ‘punkt’ resources are downloaded once."""
    for pkg in ("punkt", "punkt_tab"):
        with contextlib.suppress(LookupError):
            nltk.data.find(f"tokenizers/{pkg}")  # already present → exit early
            continue
        nltk.download(pkg, quiet=True)


# run at import so every Streamlit page has the tokenizer
ensure_punkt()
# ───────────────────────────────────────────────────────────────────────


# ────────────────── Streamlit multipage helper ────────────────────────
def switch_page(page_name: str) -> None:
    """
    Hard‑rerun the app so that it lands on another multipage screen.

    Example
    -------
    ```python
    if st.button("Professional interview"):
        switch_page("Professional Screen")
    ```
    """
    from streamlit.runtime.scriptrunner import RerunData, RerunException

    # 🔧  Streamlit ≥ 1.31 moved `get_pages`
    try:  #  new location first
        from streamlit.runtime.scriptrunner.script_runner import get_pages
    except ImportError:  #  fallback for Streamlit ≤ 1.30
        from streamlit.source_util import get_pages  # type: ignore

    def normalise(name: str) -> str:
        """Lower‑case & strip underscores so titles are matched loosely."""
        return name.strip().lower().replace("_", " ")

    # collect the registered pages for the *current* app
    pages = get_pages("")  # empty path = current script folder
    wanted = normalise(page_name)

    for page_hash, cfg in pages.items():
        if normalise(cfg["page_name"]) == wanted:
            # trigger the rerun to the target page
            raise RerunException(RerunData(page_script_hash=page_hash))

    # nothing matched → show helpful error
    available = ", ".join(sorted(normalise(p["page_name"]) for p in pages.values()))
    raise ValueError(f"❌ No page titled “{page_name}”. Available: {available}")
# ───────────────────────────────────────────────────────────────────────
