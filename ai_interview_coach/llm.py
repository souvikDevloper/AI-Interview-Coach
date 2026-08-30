"""Explicit LLM backend selection; local Ollama is the paper-aligned default."""

from __future__ import annotations

import os


def build_chat_model(temperature: float, max_tokens: int):
    backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    if backend == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=temperature,
            num_predict=max_tokens,
        )
    if backend == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        try:
            return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
        except TypeError:
            return ChatGroq(model_name=model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError("LLM_BACKEND must be 'ollama' or 'groq'")
