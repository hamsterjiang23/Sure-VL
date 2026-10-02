"""Numerical checks for the scalar objective accounting."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_vl.objective import (  # noqa: E402
    Reward,
    sampled_content_log_ratio,
    sampled_weights,
    score,
)


class ScoreTests(unittest.TestCase):
    def test_perfect_reports_preserve_correctness_utility(self) -> None:
        for visual in (0, 1):
            for answer in (0, 1):
                with self.subTest(visual=visual, answer=answer):
                    reward = score(visual, answer, float(visual), float(answer))
                    self.assertEqual(reward.utility, 2 * visual + answer)
                    self.assertEqual(reward.calibration, 0.0)
                    self.assertEqual(reward.total, 2 * visual + answer)

    def test_answer_report_is_gated_by_visual_correctness(self) -> None:
        low = score(0, 1, 0.8, 0.0)
        high = score(0, 1, 0.8, 1.0)
        self.assertAlmostEqual(low.calibration, -0.64)
        self.assertEqual(low, high)
        self.assertAlmostEqual(low.total, 0.36)

    def test_both_calibration_penalties_when_visual_is_correct(self) -> None:
        reward = score(1, 0, 0.6, 0.9)
        self.assertAlmostEqual(reward.calibration, -0.97)
        self.assertAlmostEqual(reward.total, 1.03)

    def test_score_rejects_invalid_labels_and_probabilities(self) -> None:
        for bad in (-1, 2, 0.5, "1"):
            with self.subTest(label=bad), self.assertRaises((TypeError, ValueError)):
                score(bad, 1, 0.5, 0.5)
        for bad in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5"):
            with self.subTest(probability=bad), self.assertRaises((TypeError, ValueError)):
                score(1, 1, bad, 0.5)
            with self.subTest(answer_probability=bad), self.assertRaises(
                (TypeError, ValueError)
            ):
                score(1, 1, 0.5, bad)


class SampledRatioTests(unittest.TestCase):
    def test_ratio_uses_only_aligned_content_log_probs(self) -> None:
        ratio = sampled_content_log_ratio([-0.2, -1.3], [-0.5, -0.9])
        self.assertAlmostEqual(ratio, -0.1)

    def test_ratio_rejects_missing_or_invalid_content(self) -> None:
        cases = (
            ([], []),
            ([-0.2], []),
            ([-0.2], [-0.2, -0.3]),
            ([math.nan], [-0.2]),
            ([-0.2], [math.inf]),
            ([0.01], [-0.2]),
            ([-0.2], [0.01]),
        )
        for student, teacher in cases:
            with self.subTest(student=student, teacher=teacher):
                with self.assertRaises((TypeError, ValueError)):
                    sampled_content_log_ratio(student, teacher)

    def test_ratio_rejects_non_sequences(self) -> None:
        with self.assertRaises(TypeError):
            sampled_content_log_ratio((x for x in [-0.2]), [-0.2])
        with self.assertRaises(TypeError):
            sampled_content_log_ratio([-0.2], "-0.2")


class WeightTests(unittest.TestCase):
    def test_segment_weights_and_loss_coefficients(self) -> None:
        reward = score(1, 0, 0.6, 0.9)
        weights = sampled_weights(
            reward,
            student_content_log_probs=[-0.2, -1.3],
            teacher_content_log_probs=[-0.5, -0.9],
            beta=0.4,
            content_baseline=0.2,
            confidence_baseline=-0.1,
        )
        self.assertAlmostEqual(weights.sampled_log_ratio, -0.1)
        self.assertAlmostEqual(weights.content, 1.03 - 0.4 * (-0.1) - 0.2)
        self.assertAlmostEqual(weights.confidence, -0.97 - (-0.1))
        self.assertAlmostEqual(weights.content_loss_coefficient, -weights.content)
        self.assertAlmostEqual(
            weights.confidence_loss_coefficient, -weights.confidence
        )

    def test_teacher_ratio_has_no_effect_on_confidence_weight(self) -> None:
        reward = score(1, 1, 0.8, 0.7)
        first = sampled_weights(reward, [-0.2], [-0.3], beta=4.0)
        second = sampled_weights(reward, [-0.2], [-2.3], beta=4.0)
        self.assertNotEqual(first.content, second.content)
        self.assertEqual(first.confidence, second.confidence)
        self.assertEqual(first.confidence, reward.calibration)

    def test_beta_and_baselines_must_be_finite_and_beta_nonnegative(self) -> None:
        reward = score(1, 1, 1.0, 1.0)
        for bad in (-0.1, math.nan, math.inf, True):
            with self.subTest(beta=bad), self.assertRaises((TypeError, ValueError)):
                sampled_weights(reward, [-0.2], [-0.3], beta=bad)
        for bad in (math.nan, math.inf, True):
            with self.subTest(baseline=bad), self.assertRaises((TypeError, ValueError)):
                sampled_weights(reward, [-0.2], [-0.3], content_baseline=bad)
            with self.subTest(confidence_baseline=bad), self.assertRaises(
                (TypeError, ValueError)
            ):
                sampled_weights(reward, [-0.2], [-0.3], confidence_baseline=bad)

    def test_nonfinite_result_is_rejected(self) -> None:
        reward = score(1, 1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            sampled_weights(reward, [-1.0], [-1e308], beta=1e308)
        with self.assertRaises(ValueError):
            sampled_weights(Reward(math.inf, 0.0, 0.0), [-0.2], [-0.3])
        with self.assertRaises(ValueError):
            sampled_weights(Reward(1.0, 0.0, math.nan), [-0.2], [-0.3])


if __name__ == "__main__":
    unittest.main()
