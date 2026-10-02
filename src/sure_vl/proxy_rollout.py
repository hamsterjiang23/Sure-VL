"""Framework-independent Sure-VL proxy scoring and joint on-policy loss.

The caller owns the VLM forwards.  All three distributions must be evaluated
at the same sampled response token IDs, with each view's own prompt offset.
The visual proxy and both rewards are detached before the policy update.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .proxy_method import empty_visual_proxy, visual_certainty_proxy
from .proxy_prompt import (
    ParsedProxyCompletion,
    build_proxy_masks,
    parse_proxy_completion,
    split_proxy_generated_eos,
)
from .proxy_protocol import ProxyExample, grade_proxy_answer, normalize_proxy_answer
from .trl_distillation import opsd_content_loss


@dataclass(frozen=True)
class PreparedProxyRollout:
    completion_ids: tuple[int, ...]
    content_ids: tuple[int, ...]
    text: str
    parsed: ParsedProxyCompletion
    content_mask: tuple[bool, ...]
    report_mask: tuple[bool, ...]
    vision_mask: tuple[bool, ...]
    format_errors: tuple[str, ...]

    @property
    def content_count(self) -> int:
        return len(self.content_ids)

    @property
    def vision_positions(self) -> tuple[int, ...]:
        return tuple(index for index, active in enumerate(self.vision_mask[: self.content_count]) if active)


@dataclass(frozen=True)
class ScoredProxyRollout:
    prepared: PreparedProxyRollout
    record: dict[str, Any]
    content_reward: float
    report_reward: float


@dataclass(frozen=True)
class JointProxyLoss:
    loss: Any
    policy_loss: Any
    opsd_loss: Any
    policy_loss_per_generated_token: Any
    content_mean_nll: Any | None
    report_mean_nll: Any | None
    content_tokens: int
    report_tokens: int


def prepare_proxy_rollout(
    example: ProxyExample,
    tokenizer: Any,
    completion_ids: Sequence[int],
    *,
    eos_token_ids: int | Sequence[int] | None = None,
) -> PreparedProxyRollout:
    """Parse the exact sampled IDs and conservatively partition their tokens."""
    if not isinstance(example, ProxyExample):
        raise TypeError("example must be a ProxyExample")
    ids = tuple(int(value) for value in completion_ids)
    if not ids:
        raise ValueError("sampled completion must contain at least one token")
    body, terminal = split_proxy_generated_eos(
        tokenizer, ids, generation_eos_token_id=eos_token_ids,
    )
    text = tokenizer.decode(
        list(body), skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    parsed = parse_proxy_completion(example, text)
    masks = build_proxy_masks(tokenizer, body, text, parsed)
    content = tuple(bool(value) for value in masks.content_mask) + (False,) * len(terminal)
    report = tuple(bool(value) for value in masks.report_mask) + (True,) * len(terminal)
    vision = tuple(bool(value) for value in masks.vision_mask) + (False,) * len(terminal)
    if not (len(content) == len(report) == len(vision) == len(ids)):
        raise RuntimeError("generated token masks differ from sampled completion length")
    if any((c == r) or (v and not c) for c, r, v in zip(content, report, vision, strict=True)):
        raise RuntimeError("content/report masks must partition, and vision must be content")
    content_count = sum(content)
    if content != (True,) * content_count + (False,) * (len(ids) - content_count):
        raise RuntimeError("teacher content must be a sampled completion prefix")
    errors = tuple(parsed.format_errors) + ((masks.failure_reason,) if masks.failure_reason else ())
    return PreparedProxyRollout(
        completion_ids=ids,
        content_ids=ids[:content_count],
        text=text,
        parsed=parsed,
        content_mask=content,
        report_mask=report,
        vision_mask=vision,
        format_errors=errors,
    )


def score_proxy_rollout(
    example: ProxyExample,
    prepared: PreparedProxyRollout,
    student_logits: Any,
    clear_teacher_logits: Any | None,
    restricted_teacher_logits: Any | None,
    *,
    proxy_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    teacher_conditioning: Mapping[str, Any] | None = None,
) -> ScoredProxyRollout:
    """Score a frozen p/q+/q- snapshot; never put visual proxy in autograd."""
    import torch

    count = len(prepared.completion_ids)
    if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 2:
        raise ValueError("student logits must be [sampled tokens, vocabulary]")
    if student_logits.shape[0] != count:
        raise ValueError("student logits do not align with sampled completion")
    content_count = prepared.content_count
    if clear_teacher_logits is not None and (
        clear_teacher_logits.ndim != 2
        or clear_teacher_logits.shape != (content_count, student_logits.shape[1])
    ):
        raise ValueError("q+ logits must align with sampled content prefix")
    vision_positions = prepared.vision_positions
    if vision_positions and clear_teacher_logits is None:
        raise ValueError("q+ logits are required for a nonempty vision span")
    proxy = empty_visual_proxy()
    if vision_positions:
        selected = torch.as_tensor(vision_positions, dtype=torch.long, device=student_logits.device)
        baseline = None
        if float(proxy_config["lambda_b"]) > 0:
            last = vision_positions[-1] + 1
            if restricted_teacher_logits is None or restricted_teacher_logits.shape != (
                last, student_logits.shape[1]
            ):
                raise ValueError("q- logits must reach the last vision token")
            baseline = restricted_teacher_logits.index_select(0, selected)
        with torch.no_grad():
            proxy = visual_certainty_proxy(
                student_logits.index_select(0, selected),
                clear_teacher_logits.index_select(0, selected),
                torch.ones(len(vision_positions), dtype=torch.bool, device=student_logits.device),
                baseline,
                alpha=proxy_config["alpha"],
                tau_s=proxy_config["tau_s"],
                lambda_b=proxy_config["lambda_b"],
                temperature=1.0,
                min_vision_tokens=proxy_config["min_vision_tokens"],
                chunk_size=proxy_config["chunk_size"],
            )

    errors = list(prepared.format_errors)
    if proxy.fallback:
        errors.append("vision_proxy_fallback")
    correct, canonical = grade_proxy_answer(example, prepared.parsed.answer)
    parsed_answer_available = bool(
        prepared.parsed.answer is not None
        and normalize_proxy_answer(prepared.parsed.answer)
    )
    multiple_choice = all(re.fullmatch(r"[a-d]", normalize_proxy_answer(item))
                          for item in example.accepted_answers)
    if parsed_answer_available and multiple_choice and not canonical:
        errors.append("noncanonical_option_answer")
    v = None if prepared.parsed.visual_confidence is None else prepared.parsed.visual_confidence / 10.0
    r = None if prepared.parsed.answer_confidence is None else prepared.parsed.answer_confidence / 10.0
    utility = float(reward_config["answer_utility"]) * int(correct)
    answer_score = -float(reward_config["rho_answer"]) * (
        (r - int(correct)) ** 2 if r is not None else 1.0
    )
    visual_score = -float(reward_config["rho_visual"]) * (
        (v - proxy.certainty) ** 2 if v is not None else 1.0
    )
    format_penalty = float(reward_config["format_penalty"]) if errors else 0.0
    report_reward = answer_score + visual_score - format_penalty
    content_reward = utility + report_reward
    components = {key: value for key, value in {
        "raw_js": proxy.mean_raw_js,
        "baseline_js": proxy.mean_baseline_js,
        "corrected_gap": proxy.mean_corrected_gap,
        "teacher_entropy": proxy.mean_teacher_entropy,
        "uncertainty": proxy.mean_uncertainty,
    }.items() if value is not None}
    record = {
        "id": example.id,
        "raw_completion": prepared.text,
        "vision_text": prepared.parsed.vision_text,
        "answer": prepared.parsed.answer,
        "answer_correct": correct,
        "answer_label_available": True,
        "ground_truth_available": True,
        "parsed_answer_available": parsed_answer_available,
        "answer_format_canonical": canonical,
        "visual_confidence": v,
        "answer_confidence": r,
        "visual_confidence_score": prepared.parsed.visual_confidence,
        "answer_confidence_score": prepared.parsed.answer_confidence,
        "confidence_score_max": 10,
        "teacher_conditioning": dict(teacher_conditioning or {}),
        "visual_proxy": proxy.certainty,
        "proxy_fallback": proxy.fallback,
        "vision_tokens": proxy.vision_token_count,
        "content_tokens": content_count,
        "generated_tokens": count,
        "format_errors": errors,
        "proxy_components": components,
        "reward": {
            "utility": utility,
            "answer_score": answer_score,
            "visual_score": visual_score,
            "report": report_reward,
            "total": content_reward,
            "format_penalty": format_penalty,
        },
        "opsd": {"content_tokens": float(content_count), "sampled_positions": 0.0},
    }
    return ScoredProxyRollout(prepared, record, content_reward, report_reward)


def joint_proxy_loss(
    scored: ScoredProxyRollout,
    student_logits: Any,
    clear_teacher_logits: Any | None,
    *,
    policy_weight: float,
    opsd_weight: float,
    opsd_temperature: float = 1.1,
    opsd_token_clip: float = 0.05,
) -> JointProxyLoss:
    """One rollout's score-function sum plus content-only OPSD forward KL.

    The policy term is deliberately not normalized by generated token count;
    the content KL is averaged over its supervised tokens.  There is no PPO
    ratio, group advantage, or sampled reverse KL in this objective.
    """
    import torch
    import torch.nn.functional as F

    ids = scored.prepared.completion_ids
    if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 2:
        raise ValueError("student logits must be [sampled tokens, vocabulary]")
    if student_logits.shape[0] != len(ids):
        raise ValueError("student logits do not align with sampled completion")
    if not student_logits.requires_grad:
        raise ValueError("student logits must retain gradients for the joint update")
    device = student_logits.device
    sampled = torch.as_tensor(ids, dtype=torch.long, device=device)
    content = torch.as_tensor(scored.prepared.content_mask, dtype=torch.bool, device=device)
    report = torch.as_tensor(scored.prepared.report_mask, dtype=torch.bool, device=device)
    logps = F.log_softmax(student_logits.float(), dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    policy = -float(scored.content_reward) * logps[content].sum()
    policy = policy - float(scored.report_reward) * logps[report].sum()
    content_count = scored.prepared.content_count
    if content_count and opsd_weight > 0:
        if clear_teacher_logits is None or clear_teacher_logits.shape != (
            content_count, student_logits.shape[1]
        ):
            raise ValueError("q+ cache must align with supervised content tokens")
        teacher = clear_teacher_logits.detach().to(device)
        opsd = opsd_content_loss(
            student_logits[:content_count].unsqueeze(0),
            teacher.unsqueeze(0),
            torch.ones((1, content_count), dtype=torch.bool, device=device),
            beta=0.0,
            temperature=opsd_temperature,
            pointwise_clip=opsd_token_clip,
            top_k=None,
            reduction="token_mean",
        )
    else:
        opsd = student_logits.sum() * 0.0
    loss = float(policy_weight) * policy + float(opsd_weight) * opsd
    return JointProxyLoss(
        loss=loss,
        policy_loss=policy,
        opsd_loss=opsd,
        policy_loss_per_generated_token=policy / len(ids),
        content_mean_nll=-logps[content].mean() if content_count else None,
        report_mean_nll=-logps[report].mean() if bool(report.any()) else None,
        content_tokens=content_count,
        report_tokens=sum(scored.prepared.report_mask),
    )
