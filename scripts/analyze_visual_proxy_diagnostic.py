"""Analyze a frozen visual-proxy intervention run with stdlib only.

Each paired example ID, never each token, is one bootstrap observation.
Token-pooled distributions are descriptive and may overweight long spans.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "raw_js", "baseline_js", "corrected_gap", "teacher_entropy",
    "student_entropy", "uncertainty",
)
CORE_METRICS = tuple(item for item in METRICS if item != "student_entropy")
DISPLAY_METRICS = (*METRICS, "visual_proxy")
KNOWN_ORDER = (
    "normal", "blur_student", "occluded_student", "noise_student",
    "blank_student", "wrong_teacher_crop", "same_template_crop",
    "template_only_teacher", "identity_teacher",
)


def _number(value: Any, context: str, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            rows.append(item)
    if not rows and not allow_empty:
        raise ValueError(f"{path} has no records")
    return rows, digest.hexdigest()


def _key(row: dict[str, Any], context: str) -> tuple[str, str]:
    example_id, condition = row.get("example_id"), row.get("condition")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"{context}.example_id must be nonempty text")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError(f"{context}.condition must be nonempty text")
    return example_id, condition


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _histogram(values: list[float], bins: int) -> dict[str, Any]:
    counts = [0] * bins
    for value in values:
        if not -1e-4 <= value <= 1 + 1e-4:
            raise ValueError("normalized metric value outside [0,1]")
        index = min(bins - 1, int(min(1.0, max(0.0, value)) * bins))
        counts[index] += 1
    return {"range": [0.0, 1.0], "bins": bins, "counts": counts}


def _describe(values: list[float], bins: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "population_std": statistics.pstdev(values) if values else None,
        "min": min(values) if values else None,
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
        "histogram": _histogram(values, bins),
    }


def _bootstrap_mean_ci(deltas: list[float], reps: int, seed: int) -> list[float] | None:
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _metric_seed(base_seed: int, condition: str, metric: str, scope: str) -> int:
    key = f"{base_seed}|{condition}|{metric}|{scope}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _paired(values: list[tuple[str, float]], *, reps: int, seed: int) -> dict[str, Any]:
    deltas = [value for _, value in values]
    tolerance = 1e-12
    return {
        "n_example_ids": len(values),
        "mean_delta_condition_minus_normal": statistics.mean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
        "population_std_delta": statistics.pstdev(deltas) if deltas else None,
        "bootstrap_95_ci_mean_delta": _bootstrap_mean_ci(deltas, reps, seed),
        "positive_count": sum(value > tolerance for value in deltas),
        "negative_count": sum(value < -tolerance for value in deltas),
        "zero_count": sum(abs(value) <= tolerance for value in deltas),
        "unit": "paired example_id",
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _validate_and_index(
    samples: list[dict[str, Any]], tokens: list[dict[str, Any]],
    *, alpha: float, lambda_b: float, tau_s: float, tolerance: float,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    sample_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        context = f"samples row {index + 1}"
        key = _key(sample, context)
        if key in sample_by_key:
            raise ValueError(f"duplicate sample {key}")
        count = sample.get("vision_tokens")
        if type(count) is not int or count < 0:
            raise ValueError(f"{context}.vision_tokens must be a nonnegative integer")
        fallback = sample.get("proxy_fallback")
        if type(fallback) is not bool:
            raise ValueError(f"{context}.proxy_fallback must be boolean")
        proxy = _number(sample.get("visual_proxy"), f"{context}.visual_proxy")
        if not 0 <= proxy <= 1:
            raise ValueError(f"{context}.visual_proxy outside [0,1]")
        if fallback and proxy != 0:
            raise ValueError(f"{context}: fallback proxy must be exactly zero")
        for metric in CORE_METRICS:
            if count and sample.get(metric) is None:
                raise ValueError(f"{context}: {metric} mean missing despite visual tokens")
            if sample.get(metric) is not None:
                _number(sample[metric], f"{context}.{metric}")
        if sample.get("student_entropy") is not None:
            _number(sample["student_entropy"], f"{context}.student_entropy")
        if "answer_correct" in sample and sample["answer_correct"] is not None and type(sample["answer_correct"]) is not bool:
            raise ValueError(f"{context}.answer_correct must be boolean or null")
        sample_by_key[key] = sample

    token_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    token_positions: set[tuple[str, str, int]] = set()
    optional_student_count = 0
    for index, token in enumerate(tokens):
        context = f"token_records row {index + 1}"
        key = _key(token, context)
        if key not in sample_by_key:
            raise ValueError(f"{context} has no matching sample {key}")
        position, token_id = token.get("token_index"), token.get("token_id")
        if type(position) is not int or position < 0 or type(token_id) is not int or token_id < 0:
            raise ValueError(f"{context} requires nonnegative token_index/token_id integers")
        if (key[0], key[1], position) in token_positions:
            raise ValueError(f"duplicate token index for {key}, position {position}")
        token_positions.add((key[0], key[1], position))
        if not isinstance(token.get("token_text"), str):
            raise ValueError(f"{context}.token_text must be text")
        for metric in CORE_METRICS:
            value = _number(token.get(metric), f"{context}.{metric}")
            if not -tolerance <= value <= 1 + tolerance:
                raise ValueError(f"{context}.{metric} outside normalized [0,1]")
        optional = _number(token.get("student_entropy"), f"{context}.student_entropy", optional=True)
        optional_student_count += optional is not None
        expected_gap = min(1.0, max(0.0, token["raw_js"] - lambda_b * token["baseline_js"]))
        expected_u = alpha * token["corrected_gap"] + (1 - alpha) * token["teacher_entropy"]
        if abs(token["corrected_gap"] - expected_gap) > tolerance:
            raise ValueError(f"{context}: corrected_gap formula mismatch")
        if abs(token["uncertainty"] - expected_u) > tolerance:
            raise ValueError(f"{context}: uncertainty formula mismatch")
        if key[1] == "identity_teacher" and (
            abs(token["raw_js"] - token["baseline_js"]) > 1e-6
            or abs(token["corrected_gap"]) > 1e-6
        ):
            raise ValueError(f"{context}: identity_teacher must have raw_js=baseline_js and corrected_gap=0")
        token_by_key[key].append(token)

    if 0 < optional_student_count < len(tokens):
        raise ValueError("student_entropy must be present on all token rows or none")
    for key, sample in sample_by_key.items():
        group = token_by_key.get(key, [])
        if len(group) != sample["vision_tokens"]:
            raise ValueError(f"{key}: vision_tokens={sample['vision_tokens']} but {len(group)} token rows")
        if group:
            for metric in CORE_METRICS + (("student_entropy",) if optional_student_count else ()):
                mean = statistics.mean(float(row[metric]) for row in group)
                sample_mean = _number(sample.get(metric), f"sample {key}.{metric}")
                if abs(mean - sample_mean) > tolerance:
                    raise ValueError(f"sample {key}.{metric} differs from token mean by {abs(mean - sample_mean):.6g}")
            if not sample["proxy_fallback"]:
                expected_proxy = math.exp(-float(sample["uncertainty"]) / tau_s)
                if abs(sample["visual_proxy"] - expected_proxy) > tolerance:
                    raise ValueError(f"sample {key}.visual_proxy differs from exp(-mean U/tau_s)")
        elif not sample["proxy_fallback"]:
            raise ValueError(f"{key}: zero visual tokens cannot have a valid proxy")
        group.sort(key=lambda row: row["token_index"])

    # Each condition must use exactly the same sampled visual-token IDs for
    # each repeated example. Different generated text would confound the probe.
    by_example: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for key in sample_by_key:
        by_example[key[0]].append(key)
    for example_id, keys in by_example.items():
        reference = [(row["token_index"], row["token_id"])
                     for row in token_by_key.get(keys[0], [])]
        reference_answer = sample_by_key[keys[0]].get("answer_correct")
        for key in keys[1:]:
            sequence = [(row["token_index"], row["token_id"])
                        for row in token_by_key.get(key, [])]
            if sequence != reference:
                raise ValueError(f"{example_id}: visual-token index/ID sequence changed across conditions")
            if (reference_answer is not None and sample_by_key[key].get("answer_correct") is not None
                    and sample_by_key[key]["answer_correct"] != reference_answer):
                raise ValueError(f"{example_id}: answer_correct changed across fixed-ID conditions")
    checks = {
        "sample_rows": len(samples), "token_rows": len(tokens),
        "distinct_example_ids": len(by_example),
        "distinct_conditions": len({key[1] for key in sample_by_key}),
        "student_entropy_available": bool(optional_student_count),
        "sample_token_counts_match": True,
        "sample_token_means_match": True,
        "per_token_formula_matches": True,
        "fixed_visual_token_ids_across_conditions": True,
        "fallback_zero_and_raw_proxy_formula_match": True,
        "identity_teacher_exact_cancellation": "identity_teacher" in {key[1] for key in sample_by_key},
    }
    return sample_by_key, dict(token_by_key), checks


def _condition_summary(
    condition: str, sample_by_key: dict[tuple[str, str], dict[str, Any]],
    token_by_key: dict[tuple[str, str], list[dict[str, Any]]], bins: int,
) -> dict[str, Any]:
    keys = sorted(key for key in sample_by_key if key[1] == condition)
    sample_rows = [sample_by_key[key] for key in keys]
    token_rows = [token for key in keys for token in token_by_key.get(key, [])]
    metrics: dict[str, Any] = {}
    for metric in DISPLAY_METRICS:
        sample_values = [float(row[metric]) for row in sample_rows if row.get(metric) is not None]
        metric_result: dict[str, Any] = {"equal_weight_example_means": _describe(sample_values, bins)}
        if metric != "visual_proxy":
            token_values = [float(row[metric]) for row in token_rows if row.get(metric) is not None]
            metric_result["pooled_tokens_descriptive_only"] = _describe(token_values, bins)
        metrics[metric] = metric_result
    labels = [row.get("answer_correct") for row in sample_rows if row.get("answer_correct") is not None]
    return {
        "sample_count": len(sample_rows), "vision_token_count": len(token_rows),
        "fallback_count": sum(row["proxy_fallback"] for row in sample_rows),
        "fallback_rate": sum(row["proxy_fallback"] for row in sample_rows) / len(sample_rows),
        "labeled_sample_count": len(labels), "answer_correct_count": sum(labels),
        "token_corrected_gap_zero_fraction": (
            sum(abs(float(row["corrected_gap"])) <= 1e-6 for row in token_rows) / len(token_rows)
            if token_rows else None
        ),
        "token_raw_js_le_baseline_js_fraction": (
            sum(float(row["raw_js"]) <= float(row["baseline_js"]) + 1e-6 for row in token_rows)
            / len(token_rows) if token_rows else None
        ),
        "metrics": metrics,
    }


def _normal_term_relationships(
    baseline: str, sample_by_key: dict[tuple[str, str], dict[str, Any]], alpha: float,
) -> dict[str, Any]:
    valid = [row for (example_id, condition), row in sorted(sample_by_key.items())
             if condition == baseline and not row["proxy_fallback"]
             and all(row.get(metric) is not None for metric in ("corrected_gap", "teacher_entropy", "visual_proxy"))]
    gap = [float(row["corrected_gap"]) for row in valid]
    entropy = [float(row["teacher_entropy"]) for row in valid]
    proxy = [float(row["visual_proxy"]) for row in valid]
    return {
        "n_valid_examples": len(valid),
        "pearson_corrected_gap_teacher_entropy": _pearson(gap, entropy),
        "pearson_corrected_gap_visual_proxy": _pearson(gap, proxy),
        "pearson_teacher_entropy_visual_proxy": _pearson(entropy, proxy),
        "mean_gap_contribution_to_uncertainty": alpha * statistics.mean(gap) if gap else None,
        "mean_entropy_contribution_to_uncertainty": (1 - alpha) * statistics.mean(entropy) if entropy else None,
        "scope": "equal-weight nonfallback baseline examples; descriptive correlation",
    }


def _paired_summary(
    condition: str, baseline: str,
    sample_by_key: dict[tuple[str, str], dict[str, Any]],
    *, reps: int, seed: int,
) -> dict[str, Any]:
    baseline_ids = {example_id for example_id, name in sample_by_key if name == baseline}
    condition_ids = {example_id for example_id, name in sample_by_key if name == condition}
    shared = sorted(baseline_ids & condition_ids)
    transitions = collections.Counter(
        ("fallback" if sample_by_key[example_id, baseline]["proxy_fallback"] else "valid")
        + "_to_"
        + ("fallback" if sample_by_key[example_id, condition]["proxy_fallback"] else "valid")
        for example_id in shared
    )
    metrics = {}
    for metric in DISPLAY_METRICS:
        by_scope = {}
        for scope in ("all_available", "valid_to_valid"):
            values = []
            for example_id in shared:
                old = sample_by_key[example_id, baseline]
                new = sample_by_key[example_id, condition]
                if scope == "valid_to_valid" and (old["proxy_fallback"] or new["proxy_fallback"]):
                    continue
                if old.get(metric) is None or new.get(metric) is None:
                    continue
                values.append((example_id, float(new[metric]) - float(old[metric])))
            by_scope[scope] = _paired(
                values, reps=reps, seed=_metric_seed(seed, condition, metric, scope)
            )
        metrics[metric] = by_scope
    return {
        "normal_sample_count": len(baseline_ids),
        "condition_sample_count": len(condition_ids),
        "shared_example_ids": len(shared),
        "normal_only_count": len(baseline_ids - condition_ids),
        "condition_only_count": len(condition_ids - baseline_ids),
        "fallback_transitions": dict(sorted(transitions.items())),
        "metrics": metrics,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    conditions = result["condition_order"]
    lines = [
        "# Visual proxy controlled diagnostic",
        "",
        "This is a zero-update, fixed-completion descriptive analysis. It does not measure factual visual correctness or calibration.",
        "",
        f"Token SHA-256: `{result['input_sha256']['token_records.jsonl']}`  ",
        f"Sample SHA-256: `{result['input_sha256']['samples.jsonl']}`",
        "",
        f"Independent paired unit: example ID; bootstrap: {result['bootstrap']['replicates']} resamples, seed {result['bootstrap']['seed']}, percentile 95% interval.",
        "Pooled token statistics are descriptive only and are never used as independent bootstrap observations.",
        "",
        "## Condition coverage and equal-weight sample means",
        "",
        "| Condition | Examples | Tokens | Fallback | mean raw JS | mean baseline JS | mean d | mean H(q+) | mean H(p) | mean U | mean S |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        row = result["conditions"][condition]
        metrics = row["metrics"]
        metric_values = [_fmt(metrics[name]["equal_weight_example_means"]["mean"])
                         for name in DISPLAY_METRICS]
        lines.append(
            f"| {condition} | {row['sample_count']} | {row['vision_token_count']} | "
            f"{row['fallback_count']} | " + " | ".join(metric_values) + " |"
        )
    lines += [
        "", "The zero-gap and raw≤baseline fractions below pool tokens descriptively; they are not independent observations.",
        "", "| Condition | Corrected gap exactly zero, token fraction | Raw JS ≤ baseline JS, token fraction |",
        "| --- | ---: | ---: |",
    ]
    for condition in conditions:
        row = result["conditions"][condition]
        lines.append(
            f"| {condition} | {_fmt(row['token_corrected_gap_zero_fraction'])} | "
            f"{_fmt(row['token_raw_js_le_baseline_js_fraction'])} |"
        )
    relationships = result["normal_term_relationships"]
    lines += [
        "", f"Valid `{result['baseline_condition']}` examples for term correlations: "
        f"{relationships['n_valid_examples']}.",
        f"Pearson(d, H(q+)) = {_fmt(relationships['pearson_corrected_gap_teacher_entropy'])}; "
        f"Pearson(d, S) = {_fmt(relationships['pearson_corrected_gap_visual_proxy'])}; "
        f"Pearson(H(q+), S) = {_fmt(relationships['pearson_teacher_entropy_visual_proxy'])}.",
        f"Mean contributions to U: gap {_fmt(relationships['mean_gap_contribution_to_uncertainty'])}, "
        f"Teacher entropy {_fmt(relationships['mean_entropy_contribution_to_uncertainty'])}.",
        "", "## Paired changes against normal", "",
        "Each delta is condition minus `normal` on the same IDs. Token-term rows require present sample means; valid-to-valid results exclude fallback pairs.",
        "",
        "| Condition | Shared IDs | Fallback transitions | Δ raw JS [95% CI] | Δ baseline JS [95% CI] | Δ corrected gap [95% CI] | Δ Teacher entropy [95% CI] | Δ Student entropy [95% CI] | Δ S [95% CI] |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in conditions:
        if condition == result["baseline_condition"]:
            continue
        row = result["paired_vs_normal"][condition]
        parts = []
        for name in ("raw_js", "baseline_js", "corrected_gap", "teacher_entropy", "student_entropy", "visual_proxy"):
            metric = row["metrics"][name]["valid_to_valid"]
            ci = metric["bootstrap_95_ci_mean_delta"]
            parts.append(
                f"{_fmt(metric['mean_delta_condition_minus_normal'])} "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "NA"
            )
        transitions = ", ".join(f"{name}:{count}" for name, count in row["fallback_transitions"].items())
        lines.append(f"| {condition} | {row['shared_example_ids']} | {transitions} | "
                     + " | ".join(parts) + " |")
    lines += [
        "", "Full metrics, pooled-token histograms, all-pair S deltas, direction counts, and sample coverage are in `summary.json`.",
        "`distribution.svg` shows equal-weight per-example mean distributions. `token_distributions.svg` shows pooled-token frequency curves. Both are descriptive.",
        "", "## Interpretation limits", "",
        "- A JS or entropy change is a next-token distribution change, not proof that an image claim is true or false.",
        "- The same-view baseline correction can cancel some Student-image perturbation. Inspect raw and baseline JS before interpreting corrected gap.",
        "- Perturbation comparisons are paired by example ID; the script does not treat many tokens from one image as independent evidence.",
        "- Frozen token IDs and zero optimizer updates must also be confirmed from the runner provenance; this script checks the token-ID equality visible in its inputs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg(path: Path, result: dict[str, Any]) -> None:
    conditions = result["condition_order"]
    metrics = DISPLAY_METRICS
    label_width, col_width, row_height = 200, 180, 28
    left, top = 30, 78
    width = left + label_width + len(metrics) * col_width + 30
    height = top + (len(conditions) + 2) * row_height + 36
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202936} .head{font-size:14px;font-weight:700} .label{font-size:12px} .axis{font-size:10px;fill:#667085}</style>',
        '<text x="30" y="27" class="head">Visual proxy diagnostic: per-example distributions</text>',
        '<text x="30" y="47" class="axis">Bars: 5th–95th percentile of equal-weight sample means. Dot: median. Scales differ by metric.</text>',
    ]
    for col, metric in enumerate(metrics):
        x0 = left + label_width + col * col_width
        summaries = [result["conditions"][condition]["metrics"][metric]["equal_weight_example_means"]
                     for condition in conditions]
        observed = [summary["max"] for summary in summaries if summary["max"] is not None]
        scale = max(0.05, min(1.0, max(observed, default=0.05) * 1.08))
        pieces.append(f'<text x="{x0 + 4}" y="69" class="label">{html.escape(metric)}</text>')
        pieces.append(f'<text x="{x0 + col_width - 24}" y="69" class="axis">{scale:.2f}</text>')
        pieces.append(f'<line x1="{x0 + 6}" y1="{top - 2}" x2="{x0 + 6}" y2="{height - 44}" stroke="#d0d5dd"/>')
        pieces.append(f'<line x1="{x0 + col_width - 10}" y1="{top - 2}" x2="{x0 + col_width - 10}" y2="{height - 44}" stroke="#d0d5dd"/>')
        for row_index, summary in enumerate(summaries):
            if summary["n"] == 0:
                continue
            y = top + row_index * row_height + row_height / 2
            pixel = lambda value: x0 + 6 + (col_width - 16) * min(scale, max(0.0, value)) / scale
            color = "#0b6bcb" if conditions[row_index] == result["baseline_condition"] else "#db6d26"
            pieces.append(f'<line x1="{pixel(summary["p05"]):.2f}" y1="{y:.2f}" x2="{pixel(summary["p95"]):.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="5" stroke-linecap="round" opacity="0.65"/>')
            pieces.append(f'<circle cx="{pixel(summary["p50"]):.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    for row_index, condition in enumerate(conditions):
        y = top + row_index * row_height + row_height / 2
        if row_index % 2:
            pieces.insert(4, f'<rect x="{left}" y="{top + row_index * row_height}" width="{width - left - 25}" height="{row_height}" fill="#f7f9fc"/>')
        pieces.append(f'<text x="{left + 4}" y="{y + 4:.2f}" class="label">{html.escape(condition)}</text>')
    pieces.append(f'<text x="{left + 4}" y="{height - 17}" class="axis">Descriptive only · sample ID is the inferential unit · no visual-correctness claim</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _write_token_svg(path: Path, result: dict[str, Any]) -> None:
    """Draw one normalized pooled-token frequency panel per measured term."""
    metrics = ("raw_js", "baseline_js", "corrected_gap", "teacher_entropy", "student_entropy")
    conditions = result["condition_order"]
    palette = (
        "#2061a3", "#d64a3b", "#328a52", "#8b52a8", "#e09422",
        "#8a5b43", "#008f9c", "#d75f9a", "#646d7a",
    )
    colors = {name: palette[index % len(palette)] for index, name in enumerate(conditions)}
    width, panel_height = 1240, 226
    header_height = 150 + 24 * max(0, math.ceil(len(conditions) / 3) - 3)
    height = header_height + panel_height * len(metrics) + 34
    plot_left, plot_right = 95, 1135
    plot_width = plot_right - plot_left
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263044}'
        '.title{font-weight:700;font-size:19px}.metric{font-weight:700;font-size:15px}'
        '.note{font-size:11px;fill:#58677c}.legend{font-size:12px}.tick{font-size:10px;fill:#6f7d8e}</style>',
        '<text x="40" y="31" class="title">Visual proxy: pooled visual-token distributions</text>',
        '<text x="40" y="53" class="note">Descriptive only · tokens from one example are correlated · x is normalized [0,1] · y is fraction of tokens per histogram bin</text>',
    ]
    for index, condition in enumerate(conditions):
        col, row = index % 3, index // 3
        x, y = 42 + col * 394, 82 + row * 23
        count = result["conditions"][condition]["vision_token_count"]
        pieces.append(f'<line x1="{x}" y1="{y-4}" x2="{x+23}" y2="{y-4}" '
                      f'stroke="{colors[condition]}" stroke-width="3"/>')
        pieces.append(f'<text x="{x+31}" y="{y}" class="legend">'
                      f'{html.escape(condition)} (n={count})</text>')

    for metric_index, metric in enumerate(metrics):
        top = header_height + metric_index * panel_height
        plot_top, plot_bottom = top + 37, top + 178
        plot_height = plot_bottom - plot_top
        series = []
        for condition in conditions:
            histogram = result["conditions"][condition]["metrics"][metric]["pooled_tokens_descriptive_only"]["histogram"]
            counts = histogram["counts"]
            total = sum(counts)
            fractions = [count / total for count in counts] if total else []
            series.append((condition, fractions))
        peak = max((max(fractions, default=0) for _, fractions in series), default=0)
        y_max = max(0.1, math.ceil(peak * 10) / 10)
        pieces.append(f'<text x="{plot_left}" y="{top+22}" class="metric">{html.escape(metric)}</text>')
        pieces.append(f'<text x="{plot_right}" y="{top+22}" text-anchor="end" class="note">'
                      f'bin width = {1 / result["conditions"][conditions[0]]["metrics"][metric]["pooled_tokens_descriptive_only"]["histogram"]["bins"]:.3f}</text>')
        for tick in range(5):
            x = plot_left + plot_width * tick / 4
            pieces.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" '
                          'stroke="#edf0f4"/>')
            pieces.append(f'<text x="{x:.2f}" y="{plot_bottom+17}" text-anchor="middle" class="tick">'
                          f'{tick/4:.2f}</text>')
        for tick in range(3):
            value = y_max * tick / 2
            y = plot_bottom - plot_height * tick / 2
            pieces.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
                          'stroke="#e2e7ee"/>')
            pieces.append(f'<text x="{plot_left-9}" y="{y+3:.2f}" text-anchor="end" class="tick">'
                          f'{value:.2f}</text>')
        for condition, fractions in series:
            if not fractions:
                continue
            points = [(plot_left, plot_bottom)]
            count = len(fractions)
            points.extend((plot_left + (index + 0.5) * plot_width / count,
                           plot_bottom - fraction * plot_height / y_max)
                          for index, fraction in enumerate(fractions))
            points.append((plot_right, plot_bottom))
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            pieces.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[condition]}" '
                          f'stroke-width="2" stroke-linejoin="round" opacity="0.85">'
                          f'<title>{html.escape(condition)}; {metric}; n={sum(result["conditions"][condition]["metrics"][metric]["pooled_tokens_descriptive_only"]["histogram"]["counts"])}</title>'
                          '</polyline>')
        if peak == 0:
            pieces.append(f'<text x="{(plot_left+plot_right)/2:.2f}" y="{top+111}" '
                          'text-anchor="middle" class="note">No token measurements</text>')
    pieces.append(f'<text x="40" y="{height-12}" class="note">'
                  'Frequency curves are descriptive; paired uncertainty intervals use example IDs in summary.json.</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-condition", default="normal")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tau-s", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.bootstrap_reps < 100 or args.hist_bins < 2 or args.seed < 0:
        parser.error("bootstrap-reps>=100, hist-bins>=2 and seed>=0 are required")
    if not (0 <= args.alpha <= 1 and 0 <= args.lambda_b <= 1 and args.tau_s > 0
            and args.tolerance > 0):
        parser.error("alpha/lambda-b must be in [0,1], tau-s/tolerance must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty to preserve an earlier diagnostic")
    sample_rows, sample_sha = _read_jsonl(args.input_dir / "samples.jsonl")
    token_rows, token_sha = _read_jsonl(args.input_dir / "token_records.jsonl", allow_empty=True)
    sample_by_key, token_by_key, checks = _validate_and_index(
        sample_rows, token_rows, alpha=args.alpha, lambda_b=args.lambda_b,
        tau_s=args.tau_s, tolerance=args.tolerance,
    )
    available = {name for _, name in sample_by_key}
    if args.baseline_condition not in available:
        raise ValueError(f"baseline condition {args.baseline_condition!r} is missing")
    order = [name for name in KNOWN_ORDER if name in available]
    order += sorted(available - set(order))
    if args.baseline_condition in order:
        order.remove(args.baseline_condition)
    order.insert(0, args.baseline_condition)
    result = {
        "status": "validated_descriptive_diagnostic",
        "input_sha256": {"samples.jsonl": sample_sha, "token_records.jsonl": token_sha},
        "input_dir": str(args.input_dir.resolve()),
        "baseline_condition": args.baseline_condition,
        "condition_order": order,
        "formula_parameters": {"alpha": args.alpha, "lambda_b": args.lambda_b,
                               "tau_s": args.tau_s, "numeric_tolerance": args.tolerance},
        "bootstrap": {"unit": "paired example_id", "replicates": args.bootstrap_reps,
                      "seed": args.seed, "method": "paired percentile bootstrap of mean condition-minus-normal delta"},
        "checks": checks,
        "conditions": {name: _condition_summary(name, sample_by_key, token_by_key, args.hist_bins)
                       for name in order},
        "normal_term_relationships": _normal_term_relationships(
            args.baseline_condition, sample_by_key, args.alpha
        ),
        "paired_vs_normal": {
            name: _paired_summary(name, args.baseline_condition, sample_by_key,
                                  reps=args.bootstrap_reps, seed=args.seed)
            for name in order if name != args.baseline_condition
        },
        "limits": [
            "Token-pooled distributions are descriptive only; paired bootstrap resamples example IDs, never tokens.",
            "A shifted JS/entropy/S distribution is not evidence of factual visual correctness or calibration.",
            "The runner must separately prove frozen model parameters, exact sampled completion IDs, and zero optimizer updates.",
            "No confidence-spread mapping or training effect is evaluated here.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(args.output_dir / "report.md", result)
    _write_svg(args.output_dir / "distribution.svg", result)
    _write_token_svg(args.output_dir / "token_distributions.svg", result)
    print(json.dumps({"output_dir": str(args.output_dir), "sample_rows": len(sample_rows),
                      "token_rows": len(token_rows), "conditions": order,
                      "paired_unit": "example_id"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
