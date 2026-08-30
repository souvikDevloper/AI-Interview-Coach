import tempfile
import unittest
from pathlib import Path

from ai_interview_coach.evidence import build_manifest, validate_rows


class EvidenceTests(unittest.TestCase):
    def test_valid_retrieval_rows(self):
        result = validate_rows(
            [{
                "cluster_id": "jd-01",
                "item_id": "q-01",
                "configuration": "D",
                "mode": "technical",
                "relevance": "4.5",
                "star_score": "75",
            }],
            "retrieval",
        )
        self.assertEqual(result["rows"], 1)

    def test_direct_identifiers_are_rejected(self):
        row = {
            "participant_id": "p1",
            "institution_id": "i1",
            "item_id": "q1",
            "response": "4",
            "email": "person@example.org",
        }
        with self.assertRaisesRegex(ValueError, "identifier"):
            validate_rows([row], "survey")

    def test_duplicate_keys_are_rejected(self):
        row = {"participant_id": "p1", "institution_id": "i1", "item_id": "q1", "response": "4"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([row, dict(row)], "survey")

    def test_manifest_is_stable_and_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evidence.txt"
            path.write_text("verified\n", encoding="utf-8")
            manifest = build_manifest([path], root=root)
            self.assertEqual(manifest["files"][0]["path"], "evidence.txt")
            self.assertEqual(len(manifest["files"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
