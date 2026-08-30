import unittest

from ai_interview_coach.analysis import cluster_independent_analysis, cluster_paired_analysis


class ClusterAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for cluster, control, treatment in (
            ("r1", 2.0, 4.0),
            ("r2", 3.0, 4.0),
            ("r3", 4.0, 5.0),
        ):
            for item in range(2):
                self.rows.append({
                    "cluster_id": cluster,
                    "configuration": "A",
                    "relevance": control,
                    "item_id": f"{cluster}-a-{item}",
                })
                self.rows.append({
                    "cluster_id": cluster,
                    "configuration": "D",
                    "relevance": treatment,
                    "item_id": f"{cluster}-d-{item}",
                })

    def test_analysis_aggregates_before_inference(self):
        result = cluster_paired_analysis(
            self.rows,
            metric="relevance",
            treatment="D",
            control="A",
            bootstrap_replicates=500,
            seed=7,
        )
        self.assertEqual(result.clusters_used, 3)
        self.assertAlmostEqual(result.treatment_mean, 13 / 3)
        self.assertAlmostEqual(result.control_mean, 3.0)
        self.assertAlmostEqual(result.mean_difference, 4 / 3)
        self.assertLessEqual(result.ci_low, result.mean_difference)
        self.assertGreaterEqual(result.ci_high, result.mean_difference)

    def test_incomplete_cluster_is_reported_as_dropped(self):
        rows = self.rows + [{
            "cluster_id": "r4", "configuration": "D", "relevance": 5, "item_id": "r4-d"
        }]
        result = cluster_paired_analysis(
            rows,
            metric="relevance",
            treatment="D",
            control="A",
            bootstrap_replicates=200,
        )
        self.assertEqual(result.clusters_dropped, 1)

    def test_independent_analysis_aggregates_sessions(self):
        rows = [
            {"session_id": "t1", "system": "new", "score": 1},
            {"session_id": "t1", "system": "new", "score": 1},
            {"session_id": "t2", "system": "new", "score": 0.8},
            {"session_id": "c1", "system": "base", "score": 0},
            {"session_id": "c1", "system": "base", "score": 0},
            {"session_id": "c2", "system": "base", "score": 0.2},
        ]
        result = cluster_independent_analysis(
            rows,
            metric="score",
            treatment="new",
            control="base",
            bootstrap_replicates=200,
            permutation_replicates=200,
            seed=7,
        )
        self.assertEqual(result.treatment_clusters, 2)
        self.assertEqual(result.control_clusters, 2)
        self.assertAlmostEqual(result.mean_difference, 0.8)

    def test_independent_analysis_rejects_overlapping_clusters(self):
        rows = [
            {"session_id": "s1", "system": "new", "score": 1},
            {"session_id": "s1", "system": "base", "score": 0},
            {"session_id": "s2", "system": "new", "score": 1},
            {"session_id": "s3", "system": "base", "score": 0},
        ]
        with self.assertRaises(ValueError):
            cluster_independent_analysis(
                rows,
                metric="score",
                treatment="new",
                control="base",
                bootstrap_replicates=100,
                permutation_replicates=100,
            )


if __name__ == "__main__":
    unittest.main()
