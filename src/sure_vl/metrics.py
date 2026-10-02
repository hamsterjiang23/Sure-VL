"""Offline, split-level audit of a frozen set of structured rollouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Any

from .objective import score
from .protocol import Example, OutputAttempt, ProtocolError, StudentOutput, verify


def _ece(pairs: Sequence[tuple[float, int]], bins: int = 10) -> float | None:
    """Equal-width expected calibration error; the final bin includes 1.0."""
    if not pairs:
        return None
    if bins < 1:
        raise ValueError("bins must be positive")
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    accuracy_sums = [0] * bins
    for confidence, correct in pairs:
        index = min(int(confidence * bins), bins - 1)
        counts[index] += 1
        confidence_sums[index] += confidence
        accuracy_sums[index] += correct
    total = len(pairs)
    return sum(
        count / total * abs(accuracy_sums[index] / count - confidence_sums[index] / count)
        for index, count in enumerate(counts)
        if count
    )


def evaluate(
    examples: Sequence[Example], outputs: Sequence[StudentOutput]
) -> dict[str, Any]:
    """Strict audit for callers that already have complete parsed outputs."""
    if not examples:
        raise ProtocolError("examples must be nonempty")
    if len({example.split for example in examples}) != 1:
        raise ProtocolError("evaluate requires examples from exactly one split")
    missing = sorted({example.id for example in examples} - {output.id for output in outputs})
    unknown = sorted({output.id for output in outputs} - {example.id for example in examples})
    if missing or unknown:
        raise ProtocolError(f"output ids do not match examples; missing={missing}, unknown={unknown}")
    attempts = [OutputAttempt(id=output.id, parsed=output, format_error=None) for output in outputs]
    return audit_attempts(examples, attempts)


def audit_attempts(
    examples: Sequence[Example], attempts: Sequence[OutputAttempt]
) -> dict[str, Any]:
    """Audit labeled examples, counting missing and malformed attempts as failures.

    An invalid attempt has ``V=Y=0`` and every required fact false for
    correctness denominators. It has no parsed confidence or defined reward,
    so calibration and reward means state their smaller effective counts.
    Every input example must carry verified visual facts and accepted answers;
    raw unlabeled rows cannot be scored as incorrect examples.
    The training pipeline must define its own invalid-output reward before
    learning; this offline audit does not invent one.
    """
    if not examples:
        raise ProtocolError("examples must be nonempty")
    if any(not isinstance(example, Example) for example in examples):
        raise ProtocolError("audit requires labeled Example records")
    splits = {example.split for example in examples}
    if len(splits) != 1:
        raise ProtocolError("evaluate requires examples from exactly one split")
    example_ids = [example.id for example in examples]
    attempt_ids = [attempt.id for attempt in attempts]
    if len(set(example_ids)) != len(example_ids):
        raise ProtocolError("examples contain duplicate ids")
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ProtocolError("attempts contain duplicate ids")
    unknown = sorted(set(attempt_ids) - set(example_ids))
    if unknown:
        raise ProtocolError(f"attempts contain unknown example ids: {unknown}")

    attempt_by_id = {attempt.id: attempt for attempt in attempts}
    visual_pairs: list[tuple[float, int]] = []
    conditional_pairs: list[tuple[float, int]] = []
    rewards: list[float] = []
    utilities: list[float] = []
    calibrations: list[float] = []
    joint_counts: dict[tuple[int, int], int] = {
        (0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0,
    }
    fact_correct: dict[str, int] = defaultdict(int)
    fact_total: dict[str, int] = defaultdict(int)
    answer_correct_count = 0
    high_visual_errors = 0
    high_conditional_errors = 0
    format_failures = 0
    missing_attempts = 0

    for example in examples:
        attempt = attempt_by_id.get(example.id)
        output = attempt.parsed if attempt is not None else None
        if output is None:
            if attempt is None:
                missing_attempts += 1
            else:
                format_failures += 1
            visual = answer = 0
            per_fact = {slot: False for slot in example.required_visual_facts}
        else:
            verification = verify(example, output)
            visual = int(verification.visual_correct)
            answer = int(verification.answer_correct)
            per_fact = verification.per_fact
            v = output.visual_confidence / 100
            r = output.conditional_answer_confidence / 100
            reward = score(visual, answer, v, r)
            rewards.append(reward.total)
            utilities.append(reward.utility)
            calibrations.append(reward.calibration)
            visual_pairs.append((v, visual))
            if visual:
                conditional_pairs.append((r, answer))
            if v >= 0.8 and not visual:
                high_visual_errors += 1
            if visual and r >= 0.8 and not answer:
                high_conditional_errors += 1
        answer_correct_count += answer
        joint_counts[(visual, answer)] += 1
        for slot, correct in per_fact.items():
            fact_total[slot] += 1
            fact_correct[slot] += int(correct)

    n = len(examples)
    visual_correct_count = len(conditional_pairs)
    visual_incorrect_count = joint_counts[(0, 0)] + joint_counts[(0, 1)]
    return {
        "split": next(iter(splits)),
        "sample_count": n,
        "labeled_sample_count": n,
        "required_facts_per_example": {
            "min": min(len(example.required_visual_facts) for example in examples),
            "max": max(len(example.required_visual_facts) for example in examples),
            "mean": fmean(len(example.required_visual_facts) for example in examples),
        },
        "output_coverage": {
            "received": len(attempts), "expected": n, "rate": len(attempts) / n,
            "parsed": len(rewards), "parsed_rate": len(rewards) / n,
        },
        "missing_attempt_count": missing_attempts,
        "format_failure_count": format_failures,
        "visual_correct": {"count": visual_correct_count, "rate": visual_correct_count / n},
        "answer_correct": {"count": answer_correct_count, "rate": answer_correct_count / n},
        "joint_outcomes": {
            f"v{visual}_y{answer}": {
                "count": joint_counts[(visual, answer)],
                "rate": joint_counts[(visual, answer)] / n,
            }
            for visual, answer in ((0, 0), (0, 1), (1, 0), (1, 1))
        },
        "answer_correct_given_visual_incorrect": {
            "count": joint_counts[(0, 1)],
            "denominator": visual_incorrect_count,
            "rate": (
                joint_counts[(0, 1)] / visual_incorrect_count
                if visual_incorrect_count else None
            ),
        },
        "required_fact_accuracy": {
            slot: {"correct": fact_correct[slot], "total": count, "rate": fact_correct[slot] / count}
            for slot, count in sorted(fact_total.items())
        },
        "reward_sample_count": len(rewards),
        "reward_mean": fmean(rewards) if rewards else None,
        "reward_components": {
            "sample_count": len(rewards),
            "utility_mean": fmean(utilities) if utilities else None,
            "calibration_mean": fmean(calibrations) if calibrations else None,
            "total_mean": fmean(rewards) if rewards else None,
        },
        "visual_confidence_sample_count": len(visual_pairs),
        "visual_confidence_mean": (
            fmean(confidence for confidence, _ in visual_pairs) if visual_pairs else None
        ),
        "visual_confidence_observed_rate": (
            fmean(correct for _, correct in visual_pairs) if visual_pairs else None
        ),
        "visual_brier": (
            fmean((confidence - correct) ** 2 for confidence, correct in visual_pairs)
            if visual_pairs else None
        ),
        "visual_ece10": _ece(visual_pairs),
        "conditional_answer_sample_count": len(conditional_pairs),
        "conditional_answer_confidence_mean": (
            fmean(confidence for confidence, _ in conditional_pairs)
            if conditional_pairs else None
        ),
        "conditional_answer_observed_rate": (
            fmean(correct for _, correct in conditional_pairs)
            if conditional_pairs else None
        ),
        "conditional_answer_brier": (
            fmean((confidence - correct) ** 2 for confidence, correct in conditional_pairs)
            if conditional_pairs else None
        ),
        "conditional_answer_ece10": _ece(conditional_pairs),
        "high_confidence_threshold": 0.8,
        "high_confidence_errors": {
            "visual": high_visual_errors,
            "conditional_answer_given_visual_correct": high_conditional_errors,
        },
    }
