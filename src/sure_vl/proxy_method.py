"""Detached teacher-grounded visual certainty and answer/report reward.

``certainty`` measures agreement between the restricted-view student and the
clear-view teacher, discounted by the clear teacher's own entropy. It is an
internal proxy, never a label for factual visual correctness.

The divergence uses the *full* vocabulary at natural softmax temperature 1.
OPSD's separate distillation temperature does not enter this calculation.
Only selected ``<vision>`` positions are materialized, in bounded chunks, and
all results are detached Python numbers for score-function reward accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True, slots=True)
class VisualProxyResult:
    """Per-rollout proxy and mean diagnostics over selected vision tokens."""

    certainty: float
    mean_raw_js: float | None
    mean_baseline_js: float | None
    mean_corrected_gap: float | None
    mean_teacher_entropy: float | None
    mean_uncertainty: float | None
    vision_token_count: int
    fallback: bool


@dataclass(frozen=True, slots=True)
class ProxyReward:
    """Answer utility and two separate verbal-report contributions."""

    utility: float
    answer_calibration: float
    visual_proxy_alignment: float
    report: float
    total: float


def _real(name: str, value: Real, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and in {bound}")
    return result


def _positive_real(name: str, value: Real) -> float:
    result = _real(name, value, minimum=0.0)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def empty_visual_proxy() -> VisualProxyResult:
    """Fail closed when no usable visual span was generated."""
    return VisualProxyResult(
        certainty=0.0,
        mean_raw_js=None,
        mean_baseline_js=None,
        mean_corrected_gap=None,
        mean_teacher_entropy=None,
        mean_uncertainty=None,
        vision_token_count=0,
        fallback=True,
    )


def visual_certainty_proxy(
    student_logits: Any,
    clear_teacher_logits: Any,
    vision_mask: Any,
    restricted_teacher_logits: Any | None = None,
    *,
    alpha: float = 0.5,
    tau_s: float = 0.5,
    lambda_b: float = 1.0,
    temperature: float = 1.0,
    min_vision_tokens: int = 8,
    chunk_size: int = 16,
) -> VisualProxyResult:
    """Compute the detached proxy on aligned next-token distributions.

    All logit arrays have shape ``[completion positions, full vocabulary]`` and
    refer to the same sampled student prefix at each position. ``vision_mask``
    selects exactly the generated ``<vision>`` tokens. The optional restricted
    teacher sees the student's view; it is required when ``lambda_b > 0`` and
    the visual span is nonempty. The baseline correction is a heuristic:
    ``clamp(JS(p,q+) / log(2) - lambda_b * JS(p,q-) / log(2), 0, 1)``.

    A missing or shorter-than-minimum visual span receives certainty zero.
    Diagnostics for a short nonempty span are still returned for audit. All
    tensor work happens in ``no_grad`` and one bounded token chunk at a time.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError("visual_certainty_proxy requires the optional train dependency torch") from error

    alpha = _real("alpha", alpha, minimum=0.0, maximum=1.0)
    lambda_b = _real("lambda_b", lambda_b, minimum=0.0, maximum=1.0)
    tau_s = _positive_real("tau_s", tau_s)
    temperature = _positive_real("temperature", temperature)
    min_vision_tokens = _positive_int("min_vision_tokens", min_vision_tokens)
    chunk_size = _positive_int("chunk_size", chunk_size)

    if not isinstance(student_logits, torch.Tensor) or not isinstance(clear_teacher_logits, torch.Tensor):
        raise TypeError("student and clear teacher logits must be torch tensors")
    if student_logits.ndim != 2 or student_logits.shape != clear_teacher_logits.shape:
        raise ValueError("student and clear teacher logits must share [positions, vocabulary] shape")
    if student_logits.shape[1] < 2:
        raise ValueError("full vocabulary must have at least two tokens")
    if not student_logits.is_floating_point() or not clear_teacher_logits.is_floating_point():
        raise TypeError("logits must be floating-point tensors")
    if student_logits.device != clear_teacher_logits.device:
        raise ValueError("student and clear teacher logits must be on the same device")
    if restricted_teacher_logits is not None:
        if not isinstance(restricted_teacher_logits, torch.Tensor):
            raise TypeError("restricted teacher logits must be a torch tensor")
        if restricted_teacher_logits.shape != student_logits.shape:
            raise ValueError("restricted teacher logits must share [positions, vocabulary] shape")
        if not restricted_teacher_logits.is_floating_point():
            raise TypeError("restricted teacher logits must be floating-point")
        if restricted_teacher_logits.device != student_logits.device:
            raise ValueError("restricted teacher logits must be on the student device")

    mask = torch.as_tensor(vision_mask, device=student_logits.device)
    if mask.ndim != 1 or mask.shape[0] != student_logits.shape[0]:
        raise ValueError("vision_mask must have one entry per logit position")
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("vision_mask may contain only 0 and 1")
    positions = mask.bool().nonzero(as_tuple=True)[0]
    count = int(positions.numel())
    if count == 0:
        return empty_visual_proxy()
    if lambda_b > 0 and restricted_teacher_logits is None:
        raise ValueError("restricted teacher logits are required when lambda_b > 0")

    log_two = math.log(2.0)
    log_vocab = math.log(student_logits.shape[1])
    totals = {"raw": 0.0, "base": 0.0, "gap": 0.0, "entropy": 0.0, "uncertainty": 0.0}

    def _log_distribution(logits: Any, selected: Any) -> Any:
        chunk = logits.index_select(0, selected).detach().float()
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("selected logits must be finite")
        return torch.log_softmax(chunk / temperature, dim=-1)

    def _normalized_js(log_p: Any, log_q: Any) -> Any:
        log_mix = torch.logaddexp(log_p, log_q) - log_two
        p = log_p.exp()
        q = log_q.exp()
        # A finite but extreme logit gap can make log_softmax underflow to
        # -inf. Treat 0 * log(0 / m) as its limiting value zero.
        p_term = torch.where(p > 0, p * (log_p - log_mix), 0.0)
        q_term = torch.where(q > 0, q * (log_q - log_mix), 0.0)
        return ((p_term + q_term).sum(dim=-1) / (2.0 * log_two)).clamp(0.0, 1.0)

    with torch.no_grad():
        for start in range(0, count, chunk_size):
            selected = positions[start : start + chunk_size]
            log_p = _log_distribution(student_logits, selected)
            log_q_plus = _log_distribution(clear_teacher_logits, selected)
            raw = _normalized_js(log_p, log_q_plus)
            q_plus = log_q_plus.exp()
            entropy_terms = torch.where(q_plus > 0, q_plus * log_q_plus, 0.0)
            entropy = (-entropy_terms.sum(dim=-1) / log_vocab).clamp(0.0, 1.0)
            if restricted_teacher_logits is not None:
                log_q_minus = _log_distribution(restricted_teacher_logits, selected)
                baseline = _normalized_js(log_p, log_q_minus)
            else:
                baseline = torch.zeros_like(raw)
            gap = (raw - lambda_b * baseline).clamp(0.0, 1.0)
            uncertainty = alpha * gap + (1.0 - alpha) * entropy
            totals["raw"] += float(raw.sum().item())
            totals["base"] += float(baseline.sum().item())
            totals["gap"] += float(gap.sum().item())
            totals["entropy"] += float(entropy.sum().item())
            totals["uncertainty"] += float(uncertainty.sum().item())

    mean_uncertainty = totals["uncertainty"] / count
    fallback = count < min_vision_tokens
    certainty = 0.0 if fallback else math.exp(-mean_uncertainty / tau_s)
    return VisualProxyResult(
        certainty=certainty,
        mean_raw_js=totals["raw"] / count,
        mean_baseline_js=totals["base"] / count if restricted_teacher_logits is not None else None,
        mean_corrected_gap=totals["gap"] / count,
        mean_teacher_entropy=totals["entropy"] / count,
        mean_uncertainty=mean_uncertainty,
        vision_token_count=count,
        fallback=fallback,
    )


def score_proxy(
    answer_correct: int,
    visual_certainty: float,
    visual_confidence: float,
    answer_confidence: float,
    *,
    answer_utility: float = 1.0,
    rho_answer: float = 1.0,
    rho_visual: float = 1.0,
) -> ProxyReward:
    """Return ``aY - rho_answer(r-Y)^2 - rho_visual(v-S)^2``.

    ``S`` is a detached internal proxy, not an observed factual visual label.
    The answer-confidence event is unconditional answer correctness. The first
    version defaults all coefficients to one and requires
    ``answer_utility >= rho_answer > 0`` and ``rho_visual > 0``.
    """
    if isinstance(answer_correct, bool) or not isinstance(answer_correct, Integral) or answer_correct not in (0, 1):
        raise ValueError("answer_correct must be a binary integer")
    answer = int(answer_correct)
    certainty = _real("visual_certainty", visual_certainty, minimum=0.0, maximum=1.0)
    visual = _real("visual_confidence", visual_confidence, minimum=0.0, maximum=1.0)
    reported_answer = _real("answer_confidence", answer_confidence, minimum=0.0, maximum=1.0)
    answer_utility = _positive_real("answer_utility", answer_utility)
    rho_answer = _positive_real("rho_answer", rho_answer)
    rho_visual = _positive_real("rho_visual", rho_visual)
    if answer_utility < rho_answer:
        raise ValueError("answer_utility must be at least rho_answer")
    utility = answer_utility * answer
    answer_calibration = -rho_answer * ((reported_answer - answer) ** 2)
    visual_proxy_alignment = -rho_visual * ((visual - certainty) ** 2)
    report = answer_calibration + visual_proxy_alignment
    return ProxyReward(
        utility=utility,
        answer_calibration=answer_calibration,
        visual_proxy_alignment=visual_proxy_alignment,
        report=report,
        total=utility + report,
    )
