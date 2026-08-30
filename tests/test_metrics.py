import unittest

from ai_interview_coach.metrics import cronbach_alpha, word_error_counts


class MetricsTests(unittest.TestCase):
    def test_word_error_counts(self):
        errors, words = word_error_counts("the quick brown fox", "the quick fox")
        self.assertEqual(errors, 1)
        self.assertEqual(words, 4)

    def test_cronbach_alpha_perfect_consistency(self):
        rows = []
        for participant, score in (("p1", 1), ("p2", 2), ("p3", 4)):
            for item in ("q1", "q2", "q3"):
                rows.append({"participant_id": participant, "item_id": item, "response": score})
        self.assertAlmostEqual(cronbach_alpha(rows), 1.0)

    def test_cronbach_alpha_rejects_incomplete_rows(self):
        rows = [
            {"participant_id": "p1", "item_id": "q1", "response": 1},
            {"participant_id": "p1", "item_id": "q2", "response": 2},
            {"participant_id": "p2", "item_id": "q1", "response": 3},
        ]
        with self.assertRaises(ValueError):
            cronbach_alpha(rows)


if __name__ == "__main__":
    unittest.main()
