"""
app_utils.py – tiny helpers for the AI-Interviewer app
• ensure_punkt()  → guarantee NLTK’s Punkt tokenizer is available
• switch_page()   → wrapper around Streamlit’s public st.switch_page()
"""
from __future__ import annotations

import contextlib
import nltk
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

# ───────────────────── Model builders ─────────────────────────
def build_embeddings():
    """Return a HuggingFace embedding model configured via env vars."""
    model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)


def build_chat_model(temperature: float, *, context_window: int | None = None):
    """Create an open-source chat model.

    Prefers an Ollama endpoint if available, otherwise falls back to a local
    transformers pipeline. Configure via env vars:
        LLM_BACKEND   (ollama | hf)
        OLLAMA_MODEL  (defaults to "llama3")
        OLLAMA_BASE_URL
        HF_MODEL      (defaults to "sshleifer/tiny-gpt2" for CPU-only)
    """

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "ollama":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=temperature,
        )

    model_name = os.getenv("HF_MODEL", "sshleifer/tiny-gpt2")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=context_window or 256,
        do_sample=True,
        temperature=temperature,
    )
    return HuggingFacePipeline(pipeline=gen)

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
