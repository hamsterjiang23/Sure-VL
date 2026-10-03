"""Offline metrics for the teacher-grounded *internal* visual proxy.

``visual_proxy`` is a continuous teacher/student agreement target. It is not a
label for factual visual correctness, so this module deliberately does not
report a visual accuracy or a factual visual Brier score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


_INVALID_REASON_ERRORS = frozenset({
    "missing_or_invalid_reason", "empty_reason", "legacy_reasoning_tag",
})


def _probability(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1]" if not optional else
                         f"{name} must be a number in [0, 1] or None")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("correlation vectors must have equal length")
    if len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0 or right_var == 0:
        return None
    result = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    ) / math.sqrt(left_var * right_var)
    return max(-1.0, min(1.0, result))


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (position - low) * (
        sorted_values[high] - sorted_values[low]
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0, "mean": None, "variance": None, "std": None,
            "p05": None, "p25": None, "p50": None, "p75": None, "p95": None,
        }
    ordered = sorted(values)
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    return {
        "sample_count": len(values),
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
        **{
            f"p{int(probability * 100):02d}": _quantile(ordered, probability)
            for probability in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
    }


def _binned_pairs(
    pairs: Sequence[tuple[float, float]], bins: int,
) -> tuple[list[dict[str, Any]], float | None]:
    groups: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for report, target in pairs:
        groups[min(int(report * bins), bins - 1)].append((report, target))
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    for index, group in enumerate(groups):
        report_mean = _mean([report for report, _ in group])
        target_mean = _mean([target for _, target in group])
        absolute_mean_error = (
            abs(report_mean - target_mean)
            if report_mean is not None and target_mean is not None else None
        )
        if absolute_mean_error is not None:
            weighted_error += len(group) * absolute_mean_error
        rows.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "sample_count": len(group),
            "report_mean": report_mean,
            "target_mean": target_mean,
            "absolute_mean_error": absolute_mean_error,
        })
    return rows, weighted_error / len(pairs) if pairs else None


def _normalized_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("records must be nonempty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record {index} needs a nonempty string id")
        if record_id in seen:
            raise ValueError(f"duplicate record id: {record_id}")
        seen.add(record_id)
        answer_correct = record.get("answer_correct")
        if not isinstance(answer_correct, bool):
            raise ValueError(f"{record_id}.answer_correct must be bool")
        answer_label_available = record.get("answer_label_available", True)
        if not isinstance(answer_label_available, bool):
            raise ValueError(f"{record_id}.answer_label_available must be bool")
        if not answer_label_available and answer_correct:
            raise ValueError(f"{record_id} cannot be answer-correct without an answer label")
        fallback = record.get("proxy_fallback")
        if not isinstance(fallback, bool):
            raise ValueError(f"{record_id}.proxy_fallback must be bool")
        vision_tokens = record.get("vision_tokens")
        if isinstance(vision_tokens, bool) or not isinstance(vision_tokens, int) or vision_tokens < 0:
            raise ValueError(f"{record_id}.vision_tokens must be a nonnegative int")
        if not fallback and vision_tokens == 0:
            raise ValueError(f"{record_id} has no vision tokens but no proxy fallback")
        visual_proxy = _probability(record.get("visual_proxy"), f"{record_id}.visual_proxy")
        if fallback and visual_proxy != 0:
            raise ValueError(f"{record_id} proxy fallback must have visual_proxy=0")
        errors = record.get("format_errors")
        if not isinstance(errors, (list, tuple)) or any(
            not isinstance(error, str) for error in errors
        ):
            raise ValueError(f"{record_id}.format_errors must be a list of strings")
        reason_text = record.get("reason_text")
        if reason_text is not None and not isinstance(reason_text, str):
            raise ValueError(f"{record_id}.reason_text must be text or None")
        reason_present = bool(
            isinstance(reason_text, str) and reason_text.strip()
            and not _INVALID_REASON_ERRORS.intersection(errors)
        )
        validated_components: dict[str, dict[str, float]] = {}
        for group_name in ("proxy_components", "reward", "opsd"):
            components = record.get(group_name, {})
            if components is None and group_name != "proxy_components":
                components = {}
            if not isinstance(components, Mapping):
                raise ValueError(f"{record_id}.{group_name} must be a mapping")
            validated_group: dict[str, float] = {}
            for name, value in components.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{record_id} has an invalid {group_name} name")
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{record_id}.{group_name}.{name} must be finite")
                validated_group[name] = float(value)
            validated_components[group_name] = validated_group
        normalized.append({
            "id": record_id,
            "answer_correct": answer_correct,
            "answer_label_available": answer_label_available,
            "answer_confidence": _probability(
                record.get("answer_confidence"), f"{record_id}.answer_confidence", optional=True,
            ),
            "visual_confidence": _probability(
                record.get("visual_confidence"), f"{record_id}.visual_confidence", optional=True,
            ),
            "visual_proxy": visual_proxy,
            "proxy_fallback": fallback,
            "vision_tokens": vision_tokens,
            "format_errors": list(errors),
            "reason_present": reason_present,
            **validated_components,
        })
    return normalized


def evaluate_proxy_records(
    records: Sequence[Mapping[str, Any]], *, bins: int = 10,
    high_confidence_threshold: float = 0.8,
) -> dict[str, Any]:
    """Summarize a frozen set of generated outputs and detached proxy targets.

    Answer accuracy uses every record with a known boolean answer label. Answer
    calibration uses only valid answer reports. Visual-proxy matching uses only
    non-fallback targets with valid visual reports. Missing reports are counted
    as missing, never treated as numeric zero. Scores are descriptive; the
    trainer must separately define rewards for malformed outputs.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    threshold = _probability(high_confidence_threshold, "high_confidence_threshold")
    normalized = _normalized_records(records)
    total = len(normalized)
    reason_present_count = sum(record["reason_present"] for record in normalized)
    correct_count = sum(record["answer_correct"] for record in normalized)
    label_available_count = sum(record["answer_label_available"] for record in normalized)
    answer_pairs = [
        (record["answer_confidence"], int(record["answer_correct"]))
        for record in normalized
        if record["answer_label_available"] and record["answer_confidence"] is not None
    ]
    answer_bins, answer_ece = _binned_pairs(answer_pairs, bins)
    _, answer_ece10 = _binned_pairs(answer_pairs, 10)
    confident = [(report, label) for report, label in answer_pairs if report >= threshold]
    high_confidence_errors = sum(1 - label for _, label in confident)
    ranked = sorted(
        (
            (record["answer_confidence"], int(record["answer_correct"]), record["id"])
            for record in normalized
            if record["answer_label_available"] and record["answer_confidence"] is not None
        ),
        key=lambda item: (-item[0], item[2]),
    )
    risk_coverage = []
    errors = 0
    for rank, (_, label, _) in enumerate(ranked, start=1):
        errors += 1 - label
        risk_coverage.append({
            "retained_count": rank,
            "reported_count": len(ranked),
            "total_count": total,
            "coverage_of_reported": rank / len(ranked),
            "coverage_of_labeled": rank / label_available_count,
            "coverage_of_all": rank / total,
            "error_count": errors,
            "risk": errors / rank,
        })

    eligible = [record for record in normalized if not record["proxy_fallback"]]
    visual_pairs = [
        (record["visual_confidence"], record["visual_proxy"])
        for record in eligible if record["visual_confidence"] is not None
    ]
    visual_bins, visual_binned_error = _binned_pairs(visual_pairs, bins)
    _, visual_binned_error10 = _binned_pairs(visual_pairs, 10)
    proxy_values = [record["visual_proxy"] for record in eligible]
    labeled_eligible = [record for record in eligible if record["answer_label_available"]]
    by_answer = {
        label: [record["visual_proxy"] for record in labeled_eligible if record["answer_correct"] == label]
        for label in (False, True)
    }
    def component_means(group_name: str, source: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
        values_by_name: dict[str, list[float]] = defaultdict(list)
        for source_record in source:
            for name, value in source_record[group_name].items():
                values_by_name[name].append(value)
        return {
            name: {"sample_count": len(values), "mean": fmean(values)}
            for name, values in sorted(values_by_name.items())
        }
    answer_relation = {
        "sample_count": len(labeled_eligible),
        "correlation": _pearson(
            [record["visual_proxy"] for record in labeled_eligible],
            [float(record["answer_correct"]) for record in labeled_eligible],
        ),
        "answer_incorrect": {
            "sample_count": len(by_answer[False]), "proxy_mean": _mean(by_answer[False]),
        },
        "answer_correct": {
            "sample_count": len(by_answer[True]), "proxy_mean": _mean(by_answer[True]),
        },
        "mean_difference_correct_minus_incorrect": (
            fmean(by_answer[True]) - fmean(by_answer[False])
            if by_answer[True] and by_answer[False] else None
        ),
    }
    return {
        "sample_count": total,
        "answer_correct_count": correct_count,
        "answer_accuracy": correct_count / total,
        "answer_label_available_count": label_available_count,
        "answer_label_unavailable_count": total - label_available_count,
        "answer_confidence_count": len(answer_pairs),
        "answer_confidence_coverage": (
            len(answer_pairs) / label_available_count if label_available_count else None
        ),
        "answer_brier": _mean([(report - label) ** 2 for report, label in answer_pairs]),
        "answer_ece": answer_ece,
        "answer_ece10": answer_ece10,
        "answer_ece_bins": answer_bins,
        "high_confidence_answer_errors": {
            "threshold": threshold,
            "error_count": high_confidence_errors,
            "sample_count": len(confident),
            "error_rate": high_confidence_errors / len(confident) if confident else None,
        },
        "answer_risk_coverage": risk_coverage,
        "visual_proxy_eligible_count": len(eligible),
        "visual_proxy_fallback_count": total - len(eligible),
        "visual_proxy_pair_count": len(visual_pairs),
        "visual_proxy_report_coverage": len(visual_pairs) / len(eligible) if eligible else None,
        "visual_proxy_mse": _mean([(report - target) ** 2 for report, target in visual_pairs]),
        "visual_proxy_binned_mean_error": visual_binned_error,
        "visual_proxy_binned_mean_error10": visual_binned_error10,
        "visual_proxy_bins": visual_bins,
        "visual_proxy_report_correlation": _pearson(
            [report for report, _ in visual_pairs],
            [target for _, target in visual_pairs],
        ),
        "visual_proxy_stats": _distribution(proxy_values),
        "visual_proxy_answer_relation": answer_relation,
        "proxy_components": component_means("proxy_components", eligible),
        "reward_components": component_means("reward", normalized),
        "opsd_components": component_means("opsd", normalized),
        "output_coverage": {
            "total_count": total,
            "reason_present_count": reason_present_count,
            "reason_present_fraction": reason_present_count / total,
            "answer_label_available_count": label_available_count,
            "answer_label_available_rate": label_available_count / total,
            "format_clean_count": sum(not record["format_errors"] for record in normalized),
            "format_clean_rate": sum(not record["format_errors"] for record in normalized) / total,
            "format_error_count": sum(bool(record["format_errors"]) for record in normalized),
            "visual_report_count": sum(record["visual_confidence"] is not None for record in normalized),
            "visual_report_rate": sum(record["visual_confidence"] is not None for record in normalized) / total,
            "answer_report_count": sum(record["answer_confidence"] is not None for record in normalized),
            "answer_report_rate": sum(record["answer_confidence"] is not None for record in normalized) / total,
            "answer_scored_report_count": len(answer_pairs),
            "both_reports_count": sum(
                record["visual_confidence"] is not None and record["answer_confidence"] is not None
                for record in normalized
            ),
            "both_reports_rate": sum(
                record["visual_confidence"] is not None and record["answer_confidence"] is not None
                for record in normalized
            ) / total,
            "proxy_nonfallback_count": len(eligible),
            "proxy_nonfallback_rate": len(eligible) / total,
        },
        "vision_tokens": _distribution([float(record["vision_tokens"]) for record in normalized]),
    }
