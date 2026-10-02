"""Focused checks for OPSD's distribution loss and paired-view input helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_vl.trl_distillation import (  # noqa: E402
    build_teacher_rescore_ids,
    build_visual_teacher_messages,
    opsd_content_loss,
    opsd_signal_diagnostics,
)

try:
    import torch
except ImportError:
    torch = None


class TeacherInputTests(unittest.TestCase):
    def test_swap_preserves_question_and_does_not_mutate_student(self) -> None:
        student = [
            {"role": "system", "content": "Answer carefully."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/data/blur.png"},
                    {"type": "text", "text": "What is on the sign?"},
                ],
            },
        ]
        teacher = build_visual_teacher_messages(student, "/data/clear.png")
        self.assertEqual(teacher[1]["content"][0]["image"], "/data/clear.png")
        self.assertEqual(teacher[1]["content"][1], student[1]["content"][1])
        self.assertEqual(student[1]["content"][0]["image"], "/data/blur.png")

    def test_teacher_requires_one_explicit_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_visual_teacher_messages([{"role": "user", "content": "<image>"}], "clear.png")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_visual_teacher_messages(
                [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}],
                "clear.png",
            )

    def test_rescore_appends_exact_sampled_ids(self) -> None:
        response = [7, 8, 7]
        result = build_teacher_rescore_ids([11, 12], response)
        self.assertEqual(result.input_ids, (11, 12, 7, 8, 7))
        self.assertEqual(result.response_start, 2)
        self.assertEqual(result.input_ids[result.response_start :], tuple(response))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_teacher_rescore_ids([11], [7, -1])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OPSDLossTests(unittest.TestCase):
    def test_diagnostics_show_predictive_shift_and_clipping(self) -> None:
        student = torch.tensor([[0.0, 4.0], [2.0, 0.0], [0.0, 4.0]])
        teacher = torch.tensor([[4.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        report = opsd_signal_diagnostics(
            student, teacher, temperature=1.0, pointwise_clip=0.05,
            weight=1.0, max_positions=2,
        )
        self.assertEqual(report.sampled_positions, 2)
        self.assertGreater(report.raw_forward_kl_mean, 0)
        self.assertEqual(report.top1_disagreement_rate, 1.0)
        self.assertGreater(report.clipped_vocabulary_fraction, 0)
        self.assertGreater(report.weighted_logit_grad_l2, 0)
        self.assertTrue(math.isfinite(report.clipped_loss_mean))

    def test_diagnostics_reject_invalid_settings(self) -> None:
        logits = torch.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "temperature"):
            opsd_signal_diagnostics(logits, logits, temperature=0)
        with self.assertRaisesRegex(ValueError, "pointwise_clip"):
            opsd_signal_diagnostics(logits, logits, pointwise_clip=float("nan"))
        with self.assertRaisesRegex(ValueError, "max_positions"):
            opsd_signal_diagnostics(logits, logits, max_positions=0)

    def test_forward_kl_content_mask_and_teacher_stop_gradient(self) -> None:
        student = torch.tensor([[[0.3, -0.4, 0.2], [8.0, -8.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.8, -0.6, 0.1], [-9.0, 9.0, 0.0]]], requires_grad=True)
        mask = torch.tensor([[1, 0]])
        loss = opsd_content_loss(
            student, teacher, mask, temperature=1.0, pointwise_clip=None
        )
        q = torch.softmax(teacher.detach()[0, 0], dim=-1)
        p = torch.softmax(student.detach()[0, 0], dim=-1)
        expected = (q * (q.log() - p.log())).sum()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertEqual(student.grad[0, 1].tolist(), [0.0, 0.0, 0.0])
        self.assertGreater(student.grad[0, 0].abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)

    def test_beta_endpoints_and_jsd(self) -> None:
        student = torch.tensor([[[0.0, 2.0]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0]]])
        mask = torch.tensor([[True]])
        p = torch.softmax(student.detach()[0, 0], -1)
        q = torch.softmax(teacher[0, 0], -1)
        mixture = 0.5 * (p + q)
        expected_forward = (q * (q.log() - p.log())).sum().item()
        expected_reverse = (p * (p.log() - q.log())).sum().item()
        expected_jsd = (
            0.5 * (q * (q.log() - mixture.log())).sum()
            + 0.5 * (p * (p.log() - mixture.log())).sum()
        ).item()
        kwargs = {"temperature": 1.0, "pointwise_clip": None}
        self.assertAlmostEqual(opsd_content_loss(student, teacher, mask, beta=0, **kwargs).item(), expected_forward, places=6)
        self.assertAlmostEqual(opsd_content_loss(student, teacher, mask, beta=1, **kwargs).item(), expected_reverse, places=6)
        self.assertAlmostEqual(opsd_content_loss(student, teacher, mask, beta=0.5, **kwargs).item(), expected_jsd, places=6)

    def test_clips_each_vocabulary_contribution_before_summing(self) -> None:
        student = torch.tensor([[[math.log(0.1), math.log(0.9)]]], requires_grad=True)
        teacher = torch.tensor([[[math.log(0.9), math.log(0.1)]]])
        loss = opsd_content_loss(
            student, teacher, torch.tensor([[1]]), temperature=1.0, pointwise_clip=0.05
        )
        expected = 0.05 + 0.1 * math.log(0.1 / 0.9)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertLess(loss.item(), 0.0)  # Documented pointwise-clipping behavior.

    def test_optional_top_k_uses_teacher_ids_and_renormalizes(self) -> None:
        student = torch.tensor([[[-2.0, 1.0, 10.0]]], requires_grad=True)
        teacher = torch.tensor([[[3.0, 2.0, 1.0]]])
        loss = opsd_content_loss(
            student,
            teacher,
            torch.tensor([[1]]),
            temperature=1.0,
            pointwise_clip=None,
            top_k=2,
        )
        p = torch.softmax(student.detach()[0, 0, :2], -1)
        q = torch.softmax(teacher[0, 0, :2], -1)
        expected = (q * (q.log() - p.log())).sum()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_rejects_bad_alignment_and_empty_content(self) -> None:
        logits = torch.zeros((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "share shape"):
            opsd_content_loss(logits, torch.zeros((1, 1, 3)), torch.ones((1, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            opsd_content_loss(logits, logits, torch.ones((2,)))
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            opsd_content_loss(logits, logits, torch.tensor([[1.0, 0.5]]))
        with self.assertRaisesRegex(ValueError, "at least one"):
            opsd_content_loss(logits, logits, torch.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = logits.clone()
            bad[0, 0, 0] = float("nan")
            opsd_content_loss(logits, bad, torch.tensor([[1, 0]]))


if __name__ == "__main__":
    unittest.main()
