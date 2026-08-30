"""Mode-adaptive BM25 + dense retrieval with explicit provenance.

The production path uses Sentence-Transformers embeddings and a FAISS inner-
product index. Tests can inject a deterministic embedder, so the ranking logic
is verifiable without downloading a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
import os
import re
from typing import Protocol, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+#/-][A-Za-z0-9]+)*|[^\s]", re.UNICODE)


class RetrievalMode(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    RESUME = "resume"


MODE_ALPHA = {
    RetrievalMode.TECHNICAL: 0.8,
    RetrievalMode.BEHAVIORAL: 0.2,
    RetrievalMode.RESUME: 0.4,
}


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per input text."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start_token: int
    end_token: int
    source_id: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    rank: int
    hybrid_score: float
    bm25_score: float
    dense_score: float
    alpha: float


def _lexical_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    source_id: str = "document",
) -> list[Chunk]:
    """Split text into overlapping token-like windows while preserving text."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    matches = list(TOKEN_RE.finditer(text or ""))
    if not matches:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    while start < len(matches):
        end = min(len(matches), start + chunk_size)
        char_start = matches[start].start()
        char_end = matches[end - 1].end()
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}:chunk-{len(chunks):04d}",
                text=(text or "")[char_start:char_end].strip(),
                start_token=start,
                end_token=end,
                source_id=source_id,
            )
        )
        if end == len(matches):
            break
        start += step
    return chunks


class BM25Index:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.k1 = float(k1)
        self.b = float(b)
        self.tokens = [_lexical_tokens(doc) for doc in documents]
        self.lengths = np.asarray([len(tokens) for tokens in self.tokens], dtype=float)
        self.avg_length = float(self.lengths.mean()) or 1.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        n = len(documents)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = _lexical_tokens(query)
        scores = np.zeros(len(self.tokens), dtype=float)
        for index, frequencies in enumerate(self.term_frequencies):
            length_norm = 1.0 - self.b + self.b * self.lengths[index] / self.avg_length
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                scores[index] += self.idf.get(term, 0.0) * numerator / denominator
        return scores


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


class DenseIndex:
    def __init__(self, documents: Sequence[str], embedder: Embedder):
        self.embedder = embedder
        self.vectors = _normalise_rows(embedder.encode(documents))
        self.backend = "numpy"
        self.index = None
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.backend = "faiss"
        except (ImportError, AttributeError):
            self.index = None

    def scores(self, query: str) -> np.ndarray:
        vector = _normalise_rows(self.embedder.encode([query]))
        if vector.shape[1] != self.vectors.shape[1]:
            raise ValueError("query and document embedding dimensions do not match")
        if self.index is None:
            return (self.vectors @ vector[0]).astype(float)

        distances, indices = self.index.search(vector, len(self.vectors))
        scores = np.zeros(len(self.vectors), dtype=float)
        for distance, index in zip(distances[0], indices[0]):
            if index >= 0:
                scores[int(index)] = float(distance)
        return scores


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        mode: RetrievalMode | str,
        embedder: Embedder | None = None,
    ):
        if not chunks:
            raise ValueError("retriever requires at least one non-empty chunk")
        self.chunks = list(chunks)
        self.mode = RetrievalMode(mode)
        texts = [chunk.text for chunk in self.chunks]
        self.bm25 = BM25Index(texts)
        self.dense = DenseIndex(texts, embedder or SentenceTransformerEmbedder())

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        mode: RetrievalMode | str,
        source_id: str = "document",
        chunk_size: int = 500,
        overlap: int = 50,
        embedder: Embedder | None = None,
    ) -> "HybridRetriever":
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            source_id=source_id,
        )
        if not chunks:
            chunks = [Chunk(f"{source_id}:chunk-0000", "(empty)", 0, 1, source_id)]
        return cls(chunks, mode=mode, embedder=embedder)

    def search(
        self,
        query: str,
        *,
        k: int = 4,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        if not (query or "").strip():
            raise ValueError("query must not be blank")
        if k <= 0:
            raise ValueError("k must be positive")
        weight = MODE_ALPHA[self.mode] if alpha is None else float(alpha)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        raw_bm25 = self.bm25.scores(query)
        raw_dense = self.dense.scores(query)
        bm25 = _minmax(raw_bm25)
        dense = _minmax(raw_dense)
        hybrid = weight * bm25 + (1.0 - weight) * dense
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(hybrid[i]), self.chunks[i].chunk_id),
        )[: min(k, len(self.chunks))]
        return [
            SearchResult(
                chunk=self.chunks[index],
                rank=rank,
                hybrid_score=float(hybrid[index]),
                bm25_score=float(raw_bm25[index]),
                dense_score=float(raw_dense[index]),
                alpha=weight,
            )
            for rank, index in enumerate(order, start=1)
        ]

    def context(self, query: str, *, k: int = 4, max_chars: int = 12000) -> str:
        sections: list[str] = []
        length = 0
        for result in self.search(query, k=k):
            section = f"[{result.chunk.chunk_id}]\n{result.chunk.text}"
            if length + len(section) > max_chars:
                remaining = max_chars - length
                if remaining > 0:
                    sections.append(section[:remaining])
                break
            sections.append(section)
            length += len(section) + 2
        return "\n\n".join(sections)
