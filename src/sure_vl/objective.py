"""Scalar accounting for the dual-confidence, teacher-constrained objective.

This module only computes detached numbers. A training backend must sample from
the current student, keep its own differentiable token log-probabilities, and
apply the returned score-function weights as constants (stop gradient). The
teacher scores the sampled *content* tokens on the same student prefixes.

The sampled log-ratio is the reverse-KL Monte Carlo term. It is already in the
content weight; adding a separate loss for the same KL would count it twice.
No teacher term belongs on confidence-report tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class Reward:
    """One rollout's task utility, calibration score, and their sum."""

    total: float
    calibration: float
    utility: float


@dataclass(frozen=True, slots=True)
class ScoreFunctionWeights:
    """Detached multipliers for content and confidence log-probability sums.

    For a minimization loss, multiply each segment's student log-probability
    sum by its corresponding ``*_loss_coefficient``. These are scalar
    coefficients, not differentiable losses or a KL gradient implementation.
    """

    content: float
    confidence: float
    sampled_log_ratio: float

    @property
    def content_loss_coefficient(self) -> float:
        """Coefficient of the content log-probability sum in a loss."""

        return -self.content

    @property
    def confidence_loss_coefficient(self) -> float:
        """Coefficient of the report log-probability sum in a loss."""

        return -self.confidence


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _binary_label(name: str, value: int) -> int:
    if not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f"{name} must be a binary integer (0 or 1)")
    return int(value)


def _probability(name: str, value: Real) -> float:
    result = _finite_real(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def score(
    visual_correct: int,
    answer_correct: int,
    visual_confidence: float,
    conditional_answer_confidence: float,
) -> Reward:
    """Compute ``R = 2V + Y - (v-V)^2 - V(r-Y)^2``.

    ``V`` and ``Y`` are observed binary labels. ``v`` and ``r`` are reported
    probabilities in [0, 1]. The answer-confidence penalty is gated by ``V``.
    This routine neither parses generated text nor differentiates through the
    reported confidence values.
    """

    visual = _binary_label("visual_correct", visual_correct)
    answer = _binary_label("answer_correct", answer_correct)
    visual_report = _probability("visual_confidence", visual_confidence)
    answer_report = _probability(
        "conditional_answer_confidence", conditional_answer_confidence
    )
    utility = float(2 * visual + answer)
    calibration = -((visual_report - visual) ** 2) - visual * (
        (answer_report - answer) ** 2
    )
    return Reward(
        total=utility + calibration,
        calibration=calibration,
        utility=utility,
    )


def sampled_content_log_ratio(
    student_content_log_probs: Sequence[Real],
    teacher_content_log_probs: Sequence[Real],
) -> float:
    """Sum sampled ``log pi - log mu`` over aligned content tokens only.

    Inputs must score the *same sampled tokens at the same student prefixes*.
    Sequence alignment cannot be checked from log-probabilities alone. Each
    sequence must be nonempty, have the same length, and contain finite log
    probabilities (at most zero). A single sample's ratio may be negative;
    only its expectation under the student is the reverse KL.
    """

    if not isinstance(student_content_log_probs, Sequence) or isinstance(
        student_content_log_probs, (str, bytes)
    ):
        raise TypeError("student_content_log_probs must be a sequence")
    if not isinstance(teacher_content_log_probs, Sequence) or isinstance(
        teacher_content_log_probs, (str, bytes)
    ):
        raise TypeError("teacher_content_log_probs must be a sequence")
    count = len(student_content_log_probs)
    if count == 0:
        raise ValueError("content log-probability sequences must be nonempty")
    if count != len(teacher_content_log_probs):
        raise ValueError("student and teacher content lengths must match")

    differences = []
    for index, (student, teacher) in enumerate(
        zip(student_content_log_probs, teacher_content_log_probs)
    ):
        student_log_prob = _finite_real(f"student log-probability[{index}]", student)
        teacher_log_prob = _finite_real(f"teacher log-probability[{index}]", teacher)
        if student_log_prob > 0.0 or teacher_log_prob > 0.0:
            raise ValueError(f"content log-probability[{index}] must be <= 0")
        differences.append(student_log_prob - teacher_log_prob)

    try:
        ratio = fsum(differences)
    except OverflowError as exc:
        raise ValueError("sampled content log-ratio must be finite") from exc
    if not isfinite(ratio):
        raise ValueError("sampled content log-ratio must be finite")
    return ratio


def sampled_weights(
    reward: Reward,
    student_content_log_probs: Sequence[Real],
    teacher_content_log_probs: Sequence[Real],
    *,
    beta: float = 0.0,
    content_baseline: float = 0.0,
    confidence_baseline: float = 0.0,
) -> ScoreFunctionWeights:
    """Return score-function weights for one current-policy rollout.

    Content weight is ``R - beta * sampled_log_ratio - content_baseline``.
    Confidence-report weight is ``S - confidence_baseline``, where ``S`` is the
    calibration part of ``R``. Baselines must be independent of the action
    whose score they multiply. This function cannot verify that independence.
    Do not add another loss for this same sampled reverse-KL term.
    """

    if not isinstance(reward, Reward):
        raise TypeError("reward must be a Reward returned by score")
    reward_total = _finite_real("reward.total", reward.total)
    calibration = _finite_real("reward.calibration", reward.calibration)
    _finite_real("reward.utility", reward.utility)
    multiplier = _finite_real("beta", beta)
    if multiplier < 0.0:
        raise ValueError("beta must be nonnegative")
    content_control = _finite_real("content_baseline", content_baseline)
    confidence_control = _finite_real("confidence_baseline", confidence_baseline)
    ratio = sampled_content_log_ratio(
        student_content_log_probs, teacher_content_log_probs
    )
    content = reward_total - multiplier * ratio - content_control
    confidence = calibration - confidence_control
    if not isfinite(content) or not isfinite(confidence):
        raise ValueError("score-function weights must be finite")
    return ScoreFunctionWeights(
        content=content,
        confidence=confidence,
        sampled_log_ratio=ratio,
    )
