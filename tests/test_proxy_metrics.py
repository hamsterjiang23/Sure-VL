import math
import unittest

from sure_vl.proxy_metrics import evaluate_proxy_records


class ProxyMetricsTests(unittest.TestCase):
    def test_proxy_std_excludes_fallback(self):
        rows = [
            self._record("a", True, 0.4, 0.9, 0.4),
            self._record("b", False, 0.8, 0.1, 1.0),
            self._record("c", False, None, None, 0.0, fallback=True, vision_tokens=0),
        ]
        report = evaluate_proxy_records(rows)
        self.assertEqual(report["visual_proxy_stats"]["sample_count"], 2)
        self.assertAlmostEqual(report["visual_proxy_stats"]["std"], 0.3)


    @staticmethod
    def _record(
        record_id, correct, visual_confidence, answer_confidence, proxy, *,
        fallback=False, vision_tokens=5, errors=None, proxy_components=None,
        reward=None, opsd=None,
    ):
        return {
            "id": record_id,
            "answer_correct": correct,
            "visual_confidence": visual_confidence,
            "answer_confidence": answer_confidence,
            "visual_proxy": proxy,
            "proxy_fallback": fallback,
            "vision_tokens": vision_tokens,
            "format_errors": [] if errors is None else errors,
            "proxy_components": {} if proxy_components is None else proxy_components,
            "reward": reward,
            "opsd": opsd,
        }

    def test_answer_and_proxy_metrics_use_distinct_denominators(self):
        rows = [
            self._record("a", True, 0.8, 0.9, 0.75, vision_tokens=10,
                         proxy_components={"js": 0.1},
                         reward={"total": 1.0, "visual_score": -0.0025},
                         opsd={"content_tokens": 10, "raw_forward_kl_mean": 0.2}),
            self._record("b", False, 0.2, 0.1, 0.25, vision_tokens=5,
                         proxy_components={"js": 0.3},
                         reward={"total": -0.5}, opsd={"content_tokens": 5}),
            self._record("c", True, None, None, 0.0, fallback=True,
                         vision_tokens=0, errors=["missing_vision", "missing_confidence"],
                         reward={"total": -2.0, "format_penalty": -1.0},
                         opsd={"content_tokens": 3, "raw_forward_kl_mean": None}),
            self._record("d", False, 0.6, 0.8, 0.5, vision_tokens=8),
        ]
        report = evaluate_proxy_records(rows)
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["answer_correct_count"], 2)
        self.assertEqual(report["answer_accuracy"], 0.5)
        self.assertEqual(report["answer_confidence_count"], 3)
        self.assertAlmostEqual(report["answer_brier"], 0.22)
        self.assertAlmostEqual(report["answer_ece"], 1.0 / 3.0)
        self.assertAlmostEqual(report["answer_ece10"], 1.0 / 3.0)
        self.assertEqual(report["high_confidence_answer_errors"], {
            "threshold": 0.8, "error_count": 1, "sample_count": 2, "error_rate": 0.5,
        })
        self.assertEqual(
            [point["risk"] for point in report["answer_risk_coverage"]],
            [0.0, 0.5, 2.0 / 3.0],
        )
        self.assertEqual(
            [point["coverage_of_all"] for point in report["answer_risk_coverage"]],
            [0.25, 0.5, 0.75],
        )
        self.assertEqual(report["visual_proxy_eligible_count"], 3)
        self.assertEqual(report["visual_proxy_fallback_count"], 1)
        self.assertEqual(report["visual_proxy_pair_count"], 3)
        self.assertAlmostEqual(report["visual_proxy_mse"], 0.005)
        self.assertAlmostEqual(report["visual_proxy_binned_mean_error"], 1.0 / 15.0)
        self.assertAlmostEqual(report["visual_proxy_binned_mean_error10"], 1.0 / 15.0)
        self.assertAlmostEqual(report["visual_proxy_stats"]["mean"], 0.5)
        self.assertAlmostEqual(report["visual_proxy_stats"]["variance"], 1.0 / 24.0)
        self.assertEqual(report["visual_proxy_stats"]["p50"], 0.5)
        relation = report["visual_proxy_answer_relation"]
        self.assertEqual(relation["sample_count"], 3)
        self.assertEqual(relation["answer_correct"], {"sample_count": 1, "proxy_mean": 0.75})
        self.assertEqual(relation["answer_incorrect"], {"sample_count": 2, "proxy_mean": 0.375})
        self.assertAlmostEqual(relation["mean_difference_correct_minus_incorrect"], 0.375)
        self.assertGreater(relation["correlation"], 0)
        self.assertEqual(report["output_coverage"]["format_error_count"], 1)
        self.assertEqual(report["output_coverage"]["visual_report_count"], 3)
        self.assertEqual(report["output_coverage"]["answer_report_count"], 3)
        self.assertEqual(report["output_coverage"]["answer_report_rate"], 0.75)
        self.assertEqual(report["proxy_components"]["js"], {"sample_count": 2, "mean": 0.2})
        self.assertEqual(report["reward_components"]["total"], {
            "sample_count": 3, "mean": -0.5,
        })
        self.assertEqual(report["opsd_components"]["content_tokens"], {
            "sample_count": 3, "mean": 6.0,
        })
        self.assertEqual(report["opsd_components"]["raw_forward_kl_mean"], {
            "sample_count": 1, "mean": 0.2,
        })
        self.assertNotIn("visual_brier", report)
        self.assertNotIn("visual_accuracy", report)

    def test_missing_reports_and_fallback_never_become_numeric_zero(self):
        rows = [
            self._record("a", True, None, None, 0.0, fallback=True, vision_tokens=0,
                         errors=["empty_vision"]),
            self._record("b", False, None, None, 0.0, fallback=True, vision_tokens=2,
                         errors=["short_vision"]),
        ]
        report = evaluate_proxy_records(rows)
        self.assertEqual(report["answer_accuracy"], 0.5)
        self.assertEqual(report["answer_confidence_count"], 0)
        self.assertIsNone(report["answer_brier"])
        self.assertIsNone(report["answer_ece"])
        self.assertIsNone(report["answer_ece10"])
        self.assertEqual(report["answer_risk_coverage"], [])
        self.assertEqual(report["visual_proxy_fallback_count"], 2)
        self.assertEqual(report["visual_proxy_pair_count"], 0)
        self.assertIsNone(report["visual_proxy_mse"])
        self.assertIsNone(report["visual_proxy_binned_mean_error"])
        self.assertIsNone(report["visual_proxy_binned_mean_error10"])
        self.assertIsNone(report["visual_proxy_stats"]["mean"])
        self.assertIsNone(report["visual_proxy_answer_relation"]["correlation"])
        self.assertEqual(report["output_coverage"]["both_reports_count"], 0)

    def test_constant_values_have_undefined_correlation_and_one_is_last_bin(self):
        rows = [
            self._record("a", True, 1.0, 1.0, 0.5),
            self._record("b", False, 1.0, 1.0, 0.5),
        ]
        report = evaluate_proxy_records(rows)
        self.assertEqual(report["visual_proxy_bins"][-1]["sample_count"], 2)
        self.assertEqual(report["answer_ece_bins"][-1]["sample_count"], 2)
        self.assertIsNone(report["visual_proxy_report_correlation"])
        self.assertIsNone(report["visual_proxy_answer_relation"]["correlation"])
        self.assertAlmostEqual(report["answer_brier"], 0.5)
        self.assertAlmostEqual(report["visual_proxy_mse"], 0.25)

    def test_missing_answer_label_is_accuracy_failure_but_not_brier_target(self):
        labeled = self._record("labeled", True, 0.6, 0.9, 0.7)
        unavailable = {
            **self._record("unavailable", False, 0.4, 0.95, 0.3),
            "answer_label_available": False,
        }
        report = evaluate_proxy_records([labeled, unavailable])
        self.assertEqual(report["answer_accuracy"], 0.5)
        self.assertEqual(report["answer_label_available_count"], 1)
        self.assertEqual(report["answer_label_unavailable_count"], 1)
        self.assertEqual(report["output_coverage"]["answer_report_count"], 2)
        self.assertEqual(report["answer_confidence_count"], 1)
        self.assertAlmostEqual(report["answer_brier"], 0.01)
        self.assertEqual(report["high_confidence_answer_errors"]["sample_count"], 1)
        self.assertEqual(report["visual_proxy_answer_relation"]["sample_count"], 1)
        self.assertEqual(len(report["answer_risk_coverage"]), 1)

    def test_invalid_or_unlabeled_rows_are_rejected(self):
        valid = self._record("a", True, 0.4, 0.5, 0.6)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            evaluate_proxy_records([])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_proxy_records([valid, valid])
        for modification, pattern in [
            ({"answer_correct": None}, "answer_correct"),
            ({"answer_label_available": None}, "answer_label_available"),
            ({"answer_confidence": float("nan")}, "answer_confidence"),
            ({"visual_confidence": 1.1}, "visual_confidence"),
            ({"visual_proxy": -0.1}, "visual_proxy"),
            ({"proxy_fallback": True}, "visual_proxy=0"),
            ({"vision_tokens": 0}, "no vision tokens"),
            ({"format_errors": "bad"}, "format_errors"),
            ({"proxy_components": {"js": math.inf}}, "proxy_components.js"),
            ({"reward": {"total": math.nan}}, "reward.total"),
            ({"opsd": {"content_tokens": True}}, "opsd.content_tokens"),
        ]:
            with self.subTest(modification=modification), self.assertRaisesRegex(ValueError, pattern):
                evaluate_proxy_records([{**valid, **modification}])


if __name__ == "__main__":
    unittest.main()
