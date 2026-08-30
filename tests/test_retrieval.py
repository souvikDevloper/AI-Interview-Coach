import unittest

import numpy as np

from ai_interview_coach.retrieval import HybridRetriever, RetrievalMode, chunk_text


class KeywordEmbedder:
    vocabulary = ("python", "leadership", "database", "customer")

    def encode(self, texts):
        return np.asarray(
            [[text.lower().count(token) for token in self.vocabulary] for text in texts],
            dtype=np.float32,
        )


class RetrievalTests(unittest.TestCase):
    def test_chunking_uses_500_50_windows(self):
        text = " ".join(f"token{i}" for i in range(1001))
        chunks = chunk_text(text, chunk_size=500, overlap=50, source_id="jd-1")
        self.assertEqual(
            [(chunk.start_token, chunk.end_token) for chunk in chunks],
            [(0, 500), (450, 950), (900, 1001)],
        )
        self.assertEqual(chunks[1].chunk_id, "jd-1:chunk-0001")

    def test_mode_adaptive_weights_are_applied(self):
        text = "Python database indexing. Leadership conflict resolution. Customer discovery."
        technical = HybridRetriever.from_text(
            text, mode=RetrievalMode.TECHNICAL, chunk_size=3, overlap=0, embedder=KeywordEmbedder()
        )
        behavioral = HybridRetriever.from_text(
            text, mode=RetrievalMode.BEHAVIORAL, chunk_size=3, overlap=0, embedder=KeywordEmbedder()
        )
        self.assertEqual(technical.search("Python database", k=1)[0].alpha, 0.8)
        self.assertEqual(behavioral.search("leadership", k=1)[0].alpha, 0.2)

    def test_exact_match_ranks_first_and_has_provenance(self):
        text = "Python database indexing improves query latency. Leadership resolves conflict."
        retriever = HybridRetriever.from_text(
            text,
            mode="technical",
            chunk_size=5,
            overlap=0,
            source_id="resume-42",
            embedder=KeywordEmbedder(),
        )
        result = retriever.search("database indexing", k=1)[0]
        self.assertIn("database indexing", result.chunk.text.lower())
        self.assertTrue(result.chunk.chunk_id.startswith("resume-42:chunk-"))
        self.assertIn("[resume-42:chunk-", retriever.context("database indexing", k=1))

    def test_invalid_alpha_is_rejected(self):
        retriever = HybridRetriever.from_text(
            "one two three", mode="resume", embedder=KeywordEmbedder()
        )
        with self.assertRaises(ValueError):
            retriever.search("one", alpha=1.1)


if __name__ == "__main__":
    unittest.main()
