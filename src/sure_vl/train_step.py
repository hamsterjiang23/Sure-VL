"""One score-function update for already sampled dual-confidence rollouts.

The adapter uses scalar tensor and optimizer duck typing; PyTorch tensors and
optimizers satisfy it, but the core package does not import or require torch.
This is an on-policy REINFORCE-style *update step*, not rollout generation.

Callers must supply current-policy student token log-probabilities with live
gradients, frozen-teacher log-probabilities of the same sampled content tokens
at the same student prefixes, and an exact content/report token boundary.
Reported confidences and V/Y labels are parsed/frozen before this call. The
teacher never scores report tokens. Baselines must be independent of the
corresponding sampled action. These conditions cannot be inferred from token
log-probabilities, so the caller must enforce them.

The sampled reverse-KL log-ratio appears only inside the content score-function
weight. Do not add a second direct loss for the same KL. This function makes
one backward call and one optimizer step for the whole batch; it does not
implement PPO/GRPO, repeated updates, a dual beta update, or model evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from typing import Any

from .objective import sampled_weights, score


@dataclass(frozen=True, slots=True)
class SampledRollout:
    """Inputs for one current-policy sampled sequence.

    Each student field is an iterable of differentiable scalar token
    log-probabilities (a one-dimensional PyTorch tensor is also accepted).
    Teacher content log-probabilities may be detached scalar tensors or Python
    numbers. Content and report spans must each be nonempty.
    """

    student_content_log_probs: Any
    student_confidence_log_probs: Any
    teacher_content_log_probs: Any
    visual_correct: int
    answer_correct: int
    visual_confidence: float
    conditional_answer_confidence: float
    content_baseline: float = 0.0
    confidence_baseline: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """Detached batch means and the scalar loss used for the update."""

    loss: float
    mean_reward: float
    mean_calibration: float
    mean_sampled_log_ratio: float
    rollout_count: int


def _tokens(name: str, values: Any) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain token log-probabilities")
    try:
        tokens = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain token log-probabilities") from exc
    if not tokens:
        raise ValueError(f"{name} must be nonempty")
    return tokens


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _detached_scalar(name: str, value: Any) -> float:
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise TypeError(f"{name} must be a scalar tensor with detach()")
    detached = detach()
    item = getattr(detached, "item", None)
    if not callable(item):
        raise TypeError(f"{name}.detach() must expose item()")
    try:
        raw = item()
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly one scalar") from exc
    return _finite_number(name, raw)


def _student_tokens(name: str, values: Any) -> tuple[tuple[Any, ...], list[float]]:
    tokens = _tokens(name, values)
    numbers = []
    for index, token in enumerate(tokens):
        token_name = f"{name}[{index}]"
        if getattr(token, "requires_grad", False) is not True:
            raise ValueError(f"{token_name} must require a student gradient")
        log_prob = _detached_scalar(token_name, token)
        if log_prob > 0.0:
            raise ValueError(f"{token_name} must be a log-probability <= 0")
        numbers.append(log_prob)
    return tokens, numbers


def _teacher_values(name: str, values: Any) -> list[float]:
    tokens = _tokens(name, values)
    numbers = []
    for index, token in enumerate(tokens):
        token_name = f"{name}[{index}]"
        if getattr(token, "requires_grad", False):
            raise ValueError(f"{token_name} must be detached from autograd")
        if callable(getattr(token, "detach", None)):
            number = _detached_scalar(token_name, token)
        else:
            number = _finite_number(token_name, token)
        if number > 0.0:
            raise ValueError(f"{token_name} must be a log-probability <= 0")
        numbers.append(number)
    return numbers


def _sum_tokens(tokens: tuple[Any, ...]) -> Any:
    total = tokens[0]
    for token in tokens[1:]:
        total = total + token
    return total


def train_step(
    rollouts: Sequence[SampledRollout],
    optimizer: Any,
    *,
    beta: float = 0.0,
) -> TrainStepResult:
    """Apply one batch-mean score-function update and return detached metrics.

    The scalar minimization loss is the mean of
    ``-stopgrad(R - beta*k - b_content) * sum(log_pi_content)`` plus
    ``-stopgrad(S - b_confidence) * sum(log_pi_report)``. The weights are
    computed by :func:`sure_vl.objective.sampled_weights`; ``k`` uses content
    tokens only. Passing a stale-policy rollout or updating a batch more than
    once invalidates the claimed on-policy estimator.

    All inputs are validated and the complete loss is built before calling
    ``optimizer.zero_grad()``, ``loss.backward()``, and ``optimizer.step()``
    exactly once each. The optimizer must own the parameters that produced the
    supplied student token log-probabilities.
    """

    if not isinstance(rollouts, Sequence) or isinstance(rollouts, (str, bytes)):
        raise TypeError("rollouts must be a sequence of SampledRollout values")
    if not rollouts:
        raise ValueError("rollouts must be nonempty")
    zero_grad = getattr(optimizer, "zero_grad", None)
    step = getattr(optimizer, "step", None)
    if not callable(zero_grad) or not callable(step):
        raise TypeError("optimizer must expose zero_grad() and step()")

    losses = []
    rewards = []
    calibrations = []
    ratios = []
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, SampledRollout):
            raise TypeError(f"rollouts[{index}] must be SampledRollout")
        content_tokens, student_content_values = _student_tokens(
            f"rollouts[{index}].student_content_log_probs",
            rollout.student_content_log_probs,
        )
        report_tokens, _ = _student_tokens(
            f"rollouts[{index}].student_confidence_log_probs",
            rollout.student_confidence_log_probs,
        )
        teacher_values = _teacher_values(
            f"rollouts[{index}].teacher_content_log_probs",
            rollout.teacher_content_log_probs,
        )
        reward = score(
            rollout.visual_correct,
            rollout.answer_correct,
            rollout.visual_confidence,
            rollout.conditional_answer_confidence,
        )
        weights = sampled_weights(
            reward,
            student_content_values,
            teacher_values,
            beta=beta,
            content_baseline=rollout.content_baseline,
            confidence_baseline=rollout.confidence_baseline,
        )
        loss = (
            weights.content_loss_coefficient * _sum_tokens(content_tokens)
            + weights.confidence_loss_coefficient * _sum_tokens(report_tokens)
        )
        losses.append(loss)
        rewards.append(reward.total)
        calibrations.append(reward.calibration)
        ratios.append(weights.sampled_log_ratio)

    total_loss = _sum_tokens(tuple(losses)) * (1.0 / len(losses))
    loss_value = _detached_scalar("batch loss", total_loss)
    backward = getattr(total_loss, "backward", None)
    if not callable(backward):
        raise TypeError("combined loss must expose backward()")
    count = len(losses)
    try:
        mean_reward = fsum(rewards) / count
        mean_calibration = fsum(calibrations) / count
        mean_sampled_log_ratio = fsum(ratios) / count
    except OverflowError as exc:
        raise ValueError("batch metrics must be finite") from exc
    if not all(isfinite(value) for value in (mean_reward, mean_calibration, mean_sampled_log_ratio)):
        raise ValueError("batch metrics must be finite")

    zero_grad()
    backward()
    step()
    return TrainStepResult(
        loss=loss_value,
        mean_reward=mean_reward,
        mean_calibration=mean_calibration,
        mean_sampled_log_ratio=mean_sampled_log_ratio,
        rollout_count=count,
    )
