"""Original OPSD's token-distribution loss for a future TRL training adapter.

The caller samples a continuation from the current student, then evaluates the
*same response token IDs* under student and privileged teacher contexts. Pass
the next-token logits from those two forwards, aligned to response positions.
The teacher's logits are detached here; only the student receives gradients.

The original OPSD main runs use full-vocabulary forward KL, temperature 1.1,
and clipping each vocabulary entry's KL contribution above 0.05 before summing.
``beta`` also exposes the original implementation's generalized JSD family:
0 is KL(teacher || student), 1 is KL(student || teacher), and an interior value
uses their weighted divergence to the mixture. Clipping can make the returned
value negative because negative pointwise contributions are not clipped.

This is a direct distribution-matching auxiliary loss at fixed sampled prefixes.
It is distinct from the repository's sampled reverse-KL score-function term in
``objective.py`` and must not be counted as that same constraint twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, log, log1p
from numbers import Integral, Real
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("opsd_content_loss requires PyTorch") from exc
    return torch


def _real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def opsd_content_loss(
    student_logits: Any,
    teacher_logits: Any,
    content_mask: Any,
    *,
    beta: float = 0.0,
    temperature: float = 1.1,
    pointwise_clip: float | None = 0.05,
    top_k: int | None = None,
    reduction: str = "token_mean",
) -> Any:
    """Distill aligned response logits on content tokens only.

    ``student_logits`` and ``teacher_logits`` have shape ``[batch, time, vocab]``
    and must describe the same student-sampled response positions. Teacher
    prompt lengths may differ; the caller must slice each forward at its own
    response start. ``content_mask`` is a binary ``[batch, time]`` tensor; it
    excludes padding and confidence-report tokens. At least one content token
    must be active. The default ``token_mean`` reduction matches the original
    OPSD trainer's masked sum divided by the number of active positions.

    ``top_k`` is optional and follows original OPSD: choose the *teacher's*
    top-k vocabulary IDs and renormalize both distributions within that set.
    The main OPSD setting leaves it ``None`` and uses the full vocabulary.
    This is not Vision-OPD's student-top-k-plus-tail approximation.
    """

    torch = _torch()
    if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
        raise TypeError("student_logits and teacher_logits must be PyTorch tensors")
    if not isinstance(content_mask, torch.Tensor):
        raise TypeError("content_mask must be a PyTorch tensor")
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student_logits and teacher_logits must share shape [batch, time, vocab]")
    batch_size, sequence_length, vocab_size = student_logits.shape
    if batch_size < 1 or sequence_length < 1 or vocab_size < 2:
        raise ValueError("logits must have nonempty batch/time and vocab size >= 2")
    if content_mask.shape != (batch_size, sequence_length):
        raise ValueError("content_mask must have shape [batch, time]")
    if student_logits.device != teacher_logits.device or content_mask.device != student_logits.device:
        raise ValueError("student, teacher, and mask must be on the same device")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("logits must be floating-point tensors")
    if not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("logits must be finite")
    if not bool(((content_mask == 0) | (content_mask == 1)).all()):
        raise ValueError("content_mask must contain only 0 and 1")
    active = content_mask.bool()
    count = active.sum()
    if not bool(count > 0):
        raise ValueError("content_mask must include at least one content token")

    mixture_weight = _real("beta", beta)
    if not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    scale = _real("temperature", temperature)
    if scale <= 0.0:
        raise ValueError("temperature must be positive")
    if pointwise_clip is not None:
        clip = _real("pointwise_clip", pointwise_clip)
        if clip <= 0.0:
            raise ValueError("pointwise_clip must be positive or None")
    else:
        clip = None
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise TypeError("top_k must be an integer or None")
        if not 1 <= top_k <= vocab_size:
            raise ValueError("top_k must be between 1 and vocab size")
    if reduction not in {"token_mean", "sum", "none"}:
        raise ValueError("reduction must be 'token_mean', 'sum', or 'none'")

    # Restrict before the full-vocabulary softmax: report tokens carry no
    # distillation gradient and need no large distribution tensors.
    # Float32 log-softmax avoids avoidable loss of precision under bf16/fp16.
    student_scaled = student_logits[active].float() / scale
    teacher_scaled = teacher_logits.detach()[active].float() / scale
    if top_k is not None:
        teacher_indices = teacher_scaled.topk(top_k, dim=-1).indices
        student_scaled = student_scaled.gather(-1, teacher_indices)
        teacher_scaled = teacher_scaled.gather(-1, teacher_indices)

    student_logp = torch.log_softmax(student_scaled, dim=-1)
    teacher_logp = torch.log_softmax(teacher_scaled, dim=-1)
    if mixture_weight == 0.0:
        pointwise = teacher_logp.exp() * (teacher_logp - student_logp)
    elif mixture_weight == 1.0:
        pointwise = student_logp.exp() * (student_logp - teacher_logp)
    else:
        mixture_logp = torch.logaddexp(
            student_logp + log1p(-mixture_weight),
            teacher_logp + log(mixture_weight),
        )
        teacher_part = teacher_logp.exp() * (teacher_logp - mixture_logp)
        student_part = student_logp.exp() * (student_logp - mixture_logp)
        pointwise = mixture_weight * teacher_part + (1.0 - mixture_weight) * student_part
    if clip is not None:
        pointwise = pointwise.clamp(max=clip)

    per_position = pointwise.sum(dim=-1)
    if not bool(torch.isfinite(per_position).all()):
        raise ValueError("active OPSD token losses must be finite")
    if reduction == "none":
        result = torch.zeros((batch_size, sequence_length), dtype=per_position.dtype, device=per_position.device)
        return result.masked_scatter(active, per_position)
    total = per_position.sum()
    if reduction == "sum":
        return total
    return total / count


def build_visual_teacher_messages(
    student_messages: Sequence[Mapping[str, Any]], teacher_image: Any
) -> list[dict[str, Any]]:
    """Replace one structured student image while preserving all prompt text.

    This is a Vision-OPD-style paired-view input helper, separate from the
    original OPSD loss. It accepts Hugging Face-style chat messages whose
    ``content`` is a list containing exactly one ``{"type": "image", ...}``
    item. Model-specific image encoding remains the caller's responsibility.
    """

    if not isinstance(student_messages, Sequence) or isinstance(student_messages, (str, bytes)):
        raise TypeError("student_messages must be a sequence of message mappings")
    if not student_messages:
        raise ValueError("student_messages must be nonempty")
    if teacher_image is None or (isinstance(teacher_image, str) and not teacher_image):
        raise ValueError("teacher_image must be provided")

    copied: list[dict[str, Any]] = []
    image_count = 0
    for message in student_messages:
        if not isinstance(message, Mapping):
            raise TypeError("each student message must be a mapping")
        item = deepcopy(dict(message))
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image_count += 1
                    part.pop("path", None)
                    part.pop("image_url", None)
                    part["image"] = teacher_image
        copied.append(item)
    if image_count != 1:
        raise ValueError("student_messages must contain exactly one structured image item")
    return copied


@dataclass(frozen=True, slots=True)
class TeacherRescoreIds:
    """Tokenized teacher prompt followed by the exact student continuation."""

    input_ids: tuple[int, ...]
    response_start: int


@dataclass(frozen=True, slots=True)
class OPSDSignalDiagnostics:
    """Bounded logit-space diagnostics, not an optimizer update measurement."""

    sampled_positions: int
    raw_forward_kl_mean: float
    clipped_loss_mean: float
    clipped_vocabulary_fraction: float
    top1_disagreement_rate: float
    weighted_logit_grad_l2: float


def opsd_signal_diagnostics(
    student_logits: Any,
    teacher_logits: Any,
    *,
    temperature: float = 1.1,
    pointwise_clip: float = 0.05,
    weight: float = 1.0,
    max_positions: int = 16,
) -> OPSDSignalDiagnostics:
    """Measure teacher/student predictive shift on evenly spaced content tokens.

    The sampled logits are detached from the training graph. The final number
    is the L2 norm of the weighted OPSD gradient *with respect to these sampled
    logits*, not a parameter-gradient norm or an observed optimizer update.
    This bounded diagnostic keeps full-vocabulary scans from doubling memory
    for all generated content positions.
    """
    torch = _torch()
    if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
        raise TypeError("diagnostic logits must be PyTorch tensors")
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape or student_logits.shape[0] < 1:
        raise ValueError("diagnostic logits must share nonempty [tokens, vocabulary] shape")
    if type(max_positions) is not int or max_positions < 1:
        raise ValueError("max_positions must be a positive integer")
    scale = _real("temperature", temperature)
    if scale <= 0:
        raise ValueError("temperature must be positive")
    clip = _real("pointwise_clip", pointwise_clip)
    if clip <= 0:
        raise ValueError("pointwise_clip must be positive")
    multiplier = _real("weight", weight)
    if multiplier < 0:
        raise ValueError("weight must be nonnegative")
    count = min(student_logits.shape[0], max_positions)
    indices = torch.linspace(0, student_logits.shape[0] - 1, steps=count, device=student_logits.device).long()
    student = student_logits.detach().index_select(0, indices).float().requires_grad_(True)
    teacher = teacher_logits.detach().index_select(0, indices).float()
    student_logp = torch.log_softmax(student / scale, dim=-1)
    teacher_logp = torch.log_softmax(teacher / scale, dim=-1)
    pointwise = teacher_logp.exp() * (teacher_logp - student_logp)
    raw = pointwise.sum(dim=-1).mean()
    clipped = pointwise.clamp(max=clip).sum(dim=-1).mean()
    gradient = torch.autograd.grad(clipped * multiplier, student)[0]
    return OPSDSignalDiagnostics(
        sampled_positions=count,
        raw_forward_kl_mean=float(raw.detach()),
        clipped_loss_mean=float(clipped.detach()),
        clipped_vocabulary_fraction=float((pointwise.detach() > clip).float().mean()),
        top1_disagreement_rate=float((student.detach().argmax(-1) != teacher.argmax(-1)).float().mean()),
        weighted_logit_grad_l2=float(gradient.detach().norm()),
    )


def build_teacher_rescore_ids(
    teacher_prompt_ids: Sequence[int], sampled_response_ids: Sequence[int]
) -> TeacherRescoreIds:
    """Append sampled IDs without decoding or re-tokenizing them."""

    def checked_ids(name: str, values: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of token IDs")
        if not values:
            raise ValueError(f"{name} must be nonempty")
        if any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in values):
            raise ValueError(f"{name} must contain nonnegative integer token IDs")
        return tuple(int(token) for token in values)

    prompt = checked_ids("teacher_prompt_ids", teacher_prompt_ids)
    response = checked_ids("sampled_response_ids", sampled_response_ids)
    return TeacherRescoreIds(input_ids=prompt + response, response_start=len(prompt))
