"""Math and boundary checks for the detached internal visual proxy."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_vl.proxy_method import empty_visual_proxy, score_proxy, visual_certainty_proxy  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class ProxyRewardTests(unittest.TestCase):
    def test_reward_has_unconditional_answer_and_separate_report_terms(self) -> None:
        reward = score_proxy(1, visual_certainty=0.25, visual_confidence=0.75, answer_confidence=0.8)
        self.assertEqual(reward.utility, 1.0)
        self.assertAlmostEqual(reward.answer_calibration, -0.04)
        self.assertAlmostEqual(reward.visual_proxy_alignment, -0.25)
        self.assertAlmostEqual(reward.report, -0.29)
        self.assertAlmostEqual(reward.total, 0.71)
        wrong = score_proxy(0, visual_certainty=0.6, visual_confidence=0.6, answer_confidence=0.9)
        self.assertAlmostEqual(wrong.answer_calibration, -0.81)
        self.assertEqual(wrong.visual_proxy_alignment, 0.0)
        self.assertAlmostEqual(wrong.total, -0.81)

        weighted = score_proxy(
            1, visual_certainty=0.25, visual_confidence=0.75, answer_confidence=0.8,
            answer_utility=3.0, rho_answer=2.0, rho_visual=0.5,
        )
        self.assertEqual(weighted.utility, 3.0)
        self.assertAlmostEqual(weighted.answer_calibration, -0.08)
        self.assertAlmostEqual(weighted.visual_proxy_alignment, -0.125)
        self.assertAlmostEqual(weighted.total, 2.795)

    def test_reward_rejects_missing_or_invalid_targets(self) -> None:
        for invalid in (True, -1, 2, 0.5):
            with self.subTest(answer_correct=invalid), self.assertRaises(ValueError):
                score_proxy(invalid, 0.5, 0.5, 0.5)
        for field, values in {
            "visual_certainty": (-0.1, 1.1, float("nan")),
            "visual_confidence": (-0.1, 1.1, float("inf")),
            "answer_confidence": (-0.1, 1.1, float("nan")),
        }.items():
            for value in values:
                arguments = {"visual_certainty": 0.5, "visual_confidence": 0.5, "answer_confidence": 0.5}
                arguments[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    score_proxy(1, **arguments)
        for weights in (
            {"answer_utility": 0}, {"rho_answer": 0}, {"rho_visual": 0},
            {"answer_utility": 0.9, "rho_answer": 1.0},
            {"rho_answer": float("nan")}, {"rho_visual": float("inf")},
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                score_proxy(1, 0.5, 0.5, 0.5, **weights)

    def test_empty_factory_fails_closed_without_fake_measurements(self) -> None:
        result = empty_visual_proxy()
        self.assertEqual(result.certainty, 0.0)
        self.assertEqual(result.vision_token_count, 0)
        self.assertTrue(result.fallback)
        self.assertIsNone(result.mean_raw_js)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class VisualProxyTests(unittest.TestCase):
    def test_full_vocab_js_and_entropy_have_expected_endpoints(self) -> None:
        sharp = torch.tensor([[25.0, -25.0], [25.0, -25.0]])
        opposite = torch.tensor([[-25.0, 25.0], [-25.0, 25.0]])
        mask = torch.tensor([1, 1])
        result = visual_certainty_proxy(
            sharp, opposite, mask, alpha=1.0, lambda_b=0.0,
            tau_s=0.5, min_vision_tokens=1, chunk_size=1,
        )
        self.assertAlmostEqual(result.mean_raw_js, 1.0, places=5)
        self.assertAlmostEqual(result.mean_corrected_gap, 1.0, places=5)
        self.assertAlmostEqual(result.mean_teacher_entropy, 0.0, places=5)
        self.assertAlmostEqual(result.certainty, math.exp(-2.0), places=5)
        self.assertIsNone(result.mean_baseline_js)
        self.assertFalse(result.fallback)

        uniform = torch.zeros_like(sharp)
        uncertain = visual_certainty_proxy(
            uniform, uniform, mask, alpha=0.5, lambda_b=0.0,
            tau_s=0.5, min_vision_tokens=1,
        )
        self.assertAlmostEqual(uncertain.mean_raw_js, 0.0, places=6)
        self.assertAlmostEqual(uncertain.mean_teacher_entropy, 1.0, places=6)
        self.assertAlmostEqual(uncertain.certainty, math.exp(-1.0), places=6)

    def test_baseline_correction_subtracts_only_the_same_view_gap(self) -> None:
        student = torch.tensor([[3.0, 0.0, -1.0]])
        teacher = torch.tensor([[0.0, 3.0, -1.0]])
        raw = visual_certainty_proxy(
            student, teacher, [1], lambda_b=0.0, alpha=1.0, min_vision_tokens=1,
        )
        corrected = visual_certainty_proxy(
            student, teacher, [1], teacher.clone(),
            lambda_b=1.0, alpha=1.0, min_vision_tokens=1,
        )
        self.assertGreater(raw.mean_raw_js, 0)
        self.assertAlmostEqual(corrected.mean_raw_js, corrected.mean_baseline_js, places=6)
        self.assertAlmostEqual(corrected.mean_corrected_gap, 0.0, places=6)
        self.assertAlmostEqual(corrected.certainty, 1.0, places=6)
        self.assertLess(raw.certainty, corrected.certainty)

    def test_mask_chunking_and_detach(self) -> None:
        student = torch.tensor([
            [3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [9.0, -9.0, 0.0],
            [3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [3.0, 0.0, -1.0],
            [0.0, 3.0, -1.0], [3.0, 0.0, -1.0], [0.0, 3.0, -1.0],
        ], requires_grad=True)
        clear = torch.flip(student.detach(), dims=(-1,)).requires_grad_()
        restricted = torch.zeros_like(student, requires_grad=True)
        mask = [1, 1, 0, 1, 1, 1, 1, 1, 1]
        first = visual_certainty_proxy(
            student, clear, mask, restricted, chunk_size=1,
            min_vision_tokens=8,
        )
        second = visual_certainty_proxy(
            student, clear, mask, restricted, chunk_size=32,
            min_vision_tokens=8,
        )
        self.assertEqual(first.vision_token_count, 8)
        self.assertFalse(first.fallback)
        for key in (
            "certainty", "mean_raw_js", "mean_baseline_js", "mean_corrected_gap",
            "mean_teacher_entropy", "mean_uncertainty",
        ):
            self.assertAlmostEqual(getattr(first, key), getattr(second, key), places=6)
            self.assertIsInstance(getattr(first, key), float)
        self.assertIsNone(student.grad)
        self.assertIsNone(clear.grad)
        self.assertIsNone(restricted.grad)

        altered = student.detach().clone()
        altered[2] = torch.tensor([-100.0, 100.0, 0.0])
        same_selected = visual_certainty_proxy(
            altered, clear, mask, restricted, chunk_size=2,
            min_vision_tokens=8,
        )
        self.assertAlmostEqual(first.certainty, same_selected.certainty, places=6)

    def test_empty_or_short_visual_span_fails_closed(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        empty = visual_certainty_proxy(logits, logits, [0, 0])
        self.assertEqual(empty, empty_visual_proxy())
        short = visual_certainty_proxy(
            logits, logits, [1, 0], lambda_b=0.0, min_vision_tokens=8,
        )
        self.assertTrue(short.fallback)
        self.assertEqual(short.certainty, 0.0)
        self.assertEqual(short.vision_token_count, 1)
        self.assertAlmostEqual(short.mean_raw_js, 0.0, places=6)

    def test_settings_alignment_and_selected_finiteness_are_checked(self) -> None:
        logits = torch.zeros((2, 3))
        for kwargs in (
            {"alpha": -0.1}, {"alpha": 1.1}, {"lambda_b": -0.1},
            {"lambda_b": 1.1}, {"tau_s": 0}, {"temperature": float("nan")},
            {"min_vision_tokens": 0}, {"chunk_size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                visual_certainty_proxy(logits, logits, [1, 1], logits, **kwargs)
        with self.assertRaisesRegex(ValueError, "restricted teacher"):
            visual_certainty_proxy(logits, logits, [1, 1])
        with self.assertRaisesRegex(ValueError, "share"):
            visual_certainty_proxy(logits, logits[:1], [1, 1])
        with self.assertRaisesRegex(ValueError, "vision_mask"):
            visual_certainty_proxy(logits, logits, [1])
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            visual_certainty_proxy(logits, logits, [1, 0.5])
        bad = logits.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            visual_certainty_proxy(bad, logits, [1, 0], logits)
        # Unselected invalid entries do not affect the proxy or force a full-vocab scan.
        selected = visual_certainty_proxy(bad, logits, [0, 1], logits, min_vision_tokens=1)
        self.assertTrue(math.isfinite(selected.certainty))


if __name__ == "__main__":
    unittest.main()
