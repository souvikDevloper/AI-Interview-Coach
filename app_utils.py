# app_utils.py  ───────────────────────────────────────────────────────
"""
Utility helpers for the Streamlit AI-Interviewer app
─────────────────────────────────────────────────────
* ensure_punkt()  – guarantees NLTK sentence-tokenizer data is present
* switch_page()   – programmatically jump between multipage scripts
"""

from __future__ import annotations
import contextlib
import nltk


# ────────────────── NLTK setup ──────────────────────────────────────
def ensure_punkt() -> None:
    """Download 'punkt' and 'punkt_tab' once, if missing."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


# run once on import so every page has the data
ensure_punkt()
# ────────────────────────────────────────────────────────────────────


# ────────────────── Streamlit page switcher ─────────────────────────
def switch_page(page_name: str) -> None:
    """
    Jump to another Streamlit multipage script by its visible title.

    Example:
        switch_page("Professional Screen")
    """
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages

    def standardize(name: str) -> str:
        return name.lower().replace("_", " ")

    target = standardize(page_name)
    pages  = get_pages("Homepage.py")        # main entry script name

    for page_hash, cfg in pages.items():
        if standardize(cfg["page_name"]) == target:
            raise RerunException(
                RerunData(page_script_hash=page_hash, page_name=target)
            )

    valid = [standardize(cfg["page_name"]) for cfg in pages.values()]
    raise ValueError(f"Could not find page '{page_name}'. Must be one of {valid}")
# ────────────────────────────────────────────────────────────────────
