"""Gradient-boundary checks for the optional one-step tensor adapter."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_vl.train_step import SampledRollout, train_step  # noqa: E402

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # Optional dependency: the fake autograd checks still run.
    torch = None
    functional = None


class FakeParameter:
    def __init__(self, value: float) -> None:
        self.value = value
        self.grad = 0.0


class DetachedScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeScalar:
    backward_calls = 0

    def __init__(
        self, value: float, derivatives: dict[FakeParameter, float]
    ) -> None:
        self.value = value
        self.derivatives = derivatives
        self.requires_grad = bool(derivatives)

    def detach(self) -> DetachedScalar:
        return DetachedScalar(self.value)

    def __add__(self, other: FakeScalar) -> FakeScalar:
        derivatives = dict(self.derivatives)
        for parameter, derivative in other.derivatives.items():
            derivatives[parameter] = derivatives.get(parameter, 0.0) + derivative
        return FakeScalar(self.value + other.value, derivatives)

    def __mul__(self, scalar: float) -> FakeScalar:
        return FakeScalar(
            self.value * scalar,
            {parameter: derivative * scalar for parameter, derivative in self.derivatives.items()},
        )

    __rmul__ = __mul__

    def backward(self) -> None:
        FakeScalar.backward_calls += 1
        for parameter, derivative in self.derivatives.items():
            parameter.grad += derivative


class FakeOptimizer:
    def __init__(self, parameters: list[FakeParameter], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        for parameter in self.parameters:
            parameter.grad = 0.0

    def step(self) -> None:
        self.step_calls += 1
        for parameter in self.parameters:
            parameter.value -= self.learning_rate * parameter.grad


def fake_rollout(
    content_parameter: FakeParameter,
    report_parameter: FakeParameter,
    *,
    content_log_prob: float = -0.2,
    teacher_log_prob: float = -0.5,
    report_log_prob: float = -0.4,
    visual_correct: int = 1,
    answer_correct: int = 0,
    visual_confidence: float = 0.6,
    conditional_answer_confidence: float = 0.9,
) -> SampledRollout:
    return SampledRollout(
        student_content_log_probs=[FakeScalar(content_log_prob, {content_parameter: 1.0})],
        student_confidence_log_probs=[FakeScalar(report_log_prob, {report_parameter: 1.0})],
        teacher_content_log_probs=[teacher_log_prob],
        visual_correct=visual_correct,
        answer_correct=answer_correct,
        visual_confidence=visual_confidence,
        conditional_answer_confidence=conditional_answer_confidence,
    )


class DuckTypedTrainStepTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeScalar.backward_calls = 0

    def test_batch_mean_update_has_one_backward_and_one_step(self) -> None:
        content_parameter = FakeParameter(0.0)
        report_parameter = FakeParameter(0.0)
        content_parameter.grad = 123.0  # The adapter must clear stale gradients.
        optimizer = FakeOptimizer([content_parameter, report_parameter], 0.1)
        rollouts = [
            fake_rollout(content_parameter, report_parameter),
            fake_rollout(
                content_parameter,
                report_parameter,
                content_log_prob=-1.2,
                teacher_log_prob=-1.0,
                report_log_prob=-0.6,
                visual_correct=0,
                answer_correct=1,
                visual_confidence=0.8,
                conditional_answer_confidence=0.1,
            ),
        ]

        result = train_step(rollouts, optimizer, beta=0.4)

        # First: R=1.03, S=-0.97, k=0.3, content weight=0.91.
        # Second: R=0.36, S=-0.64, k=-0.2, content weight=0.44.
        self.assertAlmostEqual(content_parameter.grad, (-0.91 - 0.44) / 2)
        self.assertAlmostEqual(report_parameter.grad, (0.97 + 0.64) / 2)
        self.assertAlmostEqual(content_parameter.value, 0.0675)
        self.assertAlmostEqual(report_parameter.value, -0.0805)
        self.assertAlmostEqual(result.loss, -0.031)
        self.assertAlmostEqual(result.mean_reward, (1.03 + 0.36) / 2)
        self.assertAlmostEqual(result.mean_calibration, (-0.97 - 0.64) / 2)
        self.assertAlmostEqual(result.mean_sampled_log_ratio, (0.3 - 0.2) / 2)
        self.assertEqual(result.rollout_count, 2)
        self.assertEqual(FakeScalar.backward_calls, 1)
        self.assertEqual(optimizer.zero_grad_calls, 1)
        self.assertEqual(optimizer.step_calls, 1)

    def test_teacher_changes_content_gradient_only(self) -> None:
        report_gradients = []
        content_gradients = []
        for teacher_log_prob in (-0.5, -2.0):
            content_parameter = FakeParameter(0.0)
            report_parameter = FakeParameter(0.0)
            optimizer = FakeOptimizer([content_parameter, report_parameter], 0.1)
            train_step(
                [
                    fake_rollout(
                        content_parameter,
                        report_parameter,
                        teacher_log_prob=teacher_log_prob,
                    )
                ],
                optimizer,
                beta=0.4,
            )
            content_gradients.append(content_parameter.grad)
            report_gradients.append(report_parameter.grad)
        self.assertNotEqual(content_gradients[0], content_gradients[1])
        self.assertEqual(report_gradients[0], report_gradients[1])

    def test_invalid_rollout_does_not_mutate_optimizer(self) -> None:
        content_parameter = FakeParameter(0.0)
        report_parameter = FakeParameter(0.0)
        optimizer = FakeOptimizer([content_parameter, report_parameter], 0.1)
        valid = fake_rollout(content_parameter, report_parameter)
        invalid = SampledRollout(
            student_content_log_probs=valid.student_content_log_probs,
            student_confidence_log_probs=[],
            teacher_content_log_probs=valid.teacher_content_log_probs,
            visual_correct=1,
            answer_correct=0,
            visual_confidence=0.6,
            conditional_answer_confidence=0.9,
        )
        with self.assertRaises(ValueError):
            train_step([valid, invalid], optimizer, beta=0.4)
        self.assertEqual(optimizer.zero_grad_calls, 0)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(FakeScalar.backward_calls, 0)

    def test_teacher_must_be_detached_and_student_must_have_gradient(self) -> None:
        content_parameter = FakeParameter(0.0)
        report_parameter = FakeParameter(0.0)
        optimizer = FakeOptimizer([content_parameter, report_parameter], 0.1)
        valid = fake_rollout(content_parameter, report_parameter)
        attached_teacher = SampledRollout(
            student_content_log_probs=valid.student_content_log_probs,
            student_confidence_log_probs=valid.student_confidence_log_probs,
            teacher_content_log_probs=[FakeScalar(-0.5, {content_parameter: 1.0})],
            visual_correct=1,
            answer_correct=0,
            visual_confidence=0.6,
            conditional_answer_confidence=0.9,
        )
        with self.assertRaisesRegex(ValueError, "detached"):
            train_step([attached_teacher], optimizer)
        detached_student = SampledRollout(
            student_content_log_probs=[FakeScalar(-0.2, {})],
            student_confidence_log_probs=valid.student_confidence_log_probs,
            teacher_content_log_probs=valid.teacher_content_log_probs,
            visual_correct=1,
            answer_correct=0,
            visual_confidence=0.6,
            conditional_answer_confidence=0.9,
        )
        with self.assertRaisesRegex(ValueError, "student gradient"):
            train_step([detached_student], optimizer)
        self.assertEqual(optimizer.step_calls, 0)

    def test_nonfinite_and_negative_beta_fail_before_update(self) -> None:
        content_parameter = FakeParameter(0.0)
        report_parameter = FakeParameter(0.0)
        optimizer = FakeOptimizer([content_parameter, report_parameter], 0.1)
        rollout = fake_rollout(content_parameter, report_parameter)
        for beta in (-0.1, math.inf, math.nan):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                train_step([rollout], optimizer, beta=beta)
        self.assertEqual(optimizer.step_calls, 0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PyTorchTrainStepTests(unittest.TestCase):
    def test_real_autograd_receives_segment_specific_weights(self) -> None:
        content_parameter = torch.nn.Parameter(torch.tensor(0.0))
        report_parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([content_parameter, report_parameter], lr=0.1)
        rollout = SampledRollout(
            student_content_log_probs=[functional.logsigmoid(content_parameter)],
            student_confidence_log_probs=[functional.logsigmoid(report_parameter)],
            teacher_content_log_probs=[torch.tensor(-0.5)],
            visual_correct=1,
            answer_correct=0,
            visual_confidence=0.6,
            conditional_answer_confidence=0.9,
        )
        ratio = -math.log(2.0) - (-0.5)
        content_weight = 1.03 - 0.4 * ratio

        train_step([rollout], optimizer, beta=0.4)

        self.assertAlmostEqual(content_parameter.grad.item(), -content_weight * 0.5)
        self.assertAlmostEqual(report_parameter.grad.item(), 0.97 * 0.5)
        self.assertAlmostEqual(content_parameter.item(), 0.1 * content_weight * 0.5)
        self.assertAlmostEqual(report_parameter.item(), -0.1 * 0.97 * 0.5)


if __name__ == "__main__":
    unittest.main()
