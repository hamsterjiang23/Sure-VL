"""Free-text visual prompt, partial parser, and conservative token segments.

Unlike the legacy fact-slot prompt, this protocol asks for a free-text visual
description. The report's visual value targets a teacher-grounded *internal*
certainty proxy; the answer value targets ordinary answer correctness.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .proxy_protocol import ProxyExample, ProxyProtocolError


_SCORE = r"(?:10|[0-9])"
_DIRECT_FULL = re.compile(
    r"\A\s*<vision>(?P<vision>.*?)</vision>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*"
    r"(?P<report><confidence>\s*"
    rf"<visual_confidence>(?P<visual>{_SCORE})</visual_confidence>\s*"
    rf"<answer_confidence>(?P<answer_conf>{_SCORE})</answer_confidence>\s*"
    r"</confidence>)\s*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedProxyCompletion:
    """Recoverable fields even when the full generated format is invalid."""

    vision_text: str | None
    reasoning_text: str | None
    answer: str | None
    visual_confidence: int | None
    answer_confidence: int | None
    format_errors: tuple[str, ...]
    vision_span_chars: tuple[int, int] | None
    report_start_char: int | None

    @property
    def format_valid(self) -> bool:
        return not self.format_errors


@dataclass(frozen=True)
class ProxyMasks:
    """Disjoint content/report masks plus a vision-body subset of content."""

    vision_mask: tuple[int, ...]
    content_mask: tuple[int, ...]
    report_mask: tuple[int, ...]
    report_start_token: int | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not (len(self.vision_mask) == len(self.content_mask) == len(self.report_mask)):
            raise ProxyProtocolError("proxy masks must share a token length")
        if any(v not in (0, 1) or c not in (0, 1) or r not in (0, 1)
               or c + r != 1 or v > c
               for v, c, r in zip(self.vision_mask, self.content_mask, self.report_mask, strict=True)):
            raise ProxyProtocolError("vision must be a content subset; content/report must partition tokens")


def build_proxy_prompt(example: ProxyExample) -> str:
    """Describe the output contract without leaking any accepted answer."""
    if not isinstance(example, ProxyExample):
        raise ProxyProtocolError("build_proxy_prompt requires a ProxyExample")
    return (
        "Answer directly without reasoning, analysis, or a thinking block. "
        "First describe the relevant visual evidence, then give the final answer, "
        "and finally report two confidence scores. Write only the tagged fields shown below.\n"
        "Keep the visual description to at most 40 words. "
        "Give the actual final answer; if the question asks for a number, output only the number.\n"
        "Each confidence score must be one integer from 0 to 10. "
        "Visual confidence estimates your internal certainty about the visual description "
        "given this image, not externally verified visual truth. "
        "Answer confidence estimates the unconditional chance that your final answer is correct, "
        "not a probability conditioned on the visual description being correct.\n"
        "Unrelated format example; do not copy its objects, answer, or scores:\n"
        "<vision>A blue umbrella is beside a table.</vision>\n"
        "<answer>umbrella</answer>\n"
        "<confidence><visual_confidence>8</visual_confidence>"
        "<answer_confidence>7</answer_confidence></confidence>\n"
        "Now answer using the actual image and question, with tags only.\n"
        f"Question: {example.question.strip()}"
    )


def _unique_tag_inner(text: str, tag: str) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        return None
    start = text.find(opening)
    inner_start = start + len(opening)
    end = text.find(closing)
    if end < inner_start:
        return None
    return text[inner_start:end], (inner_start, end), (start, end + len(closing))


def _optional_score(text: str, tag: str) -> int | None:
    found = _unique_tag_inner(text, tag)
    if found is None:
        return None
    value = found[0]
    return int(value) if re.fullmatch(_SCORE, value) else None


def parse_proxy_completion(example: ProxyExample, completion: str) -> ParsedProxyCompletion:
    """Parse what is reliable, even if other sections are missing or malformed.

    A unique nonempty ``<answer>`` is recoverable without a valid vision or
    confidence block. Missing confidence leaves ``report_start_char=None``.
    The active format has a direct ``<vision>`` block. Older completions with
    a complete ``<think>`` wrapper remain readable, but are noncanonical and
    incur a format error. Neither format fabricates a visual correctness label.
    """
    if not isinstance(example, ProxyExample):
        raise ProxyProtocolError("parse_proxy_completion requires a ProxyExample")
    if not isinstance(completion, str):
        raise ProxyProtocolError("completion must be a string")

    full = _DIRECT_FULL.fullmatch(completion)
    vision_tag = _unique_tag_inner(completion, "vision")
    reasoning_tag = _unique_tag_inner(completion, "reasoning")
    answer_tag = _unique_tag_inner(completion, "answer")
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")
    direct_vision = (
        vision_tag is not None and think_start < 0 and think_end < 0
        and reasoning_tag is None
        and (answer_tag is None or vision_tag[2][1] <= answer_tag[2][0])
    )
    legacy_vision = (
        vision_tag is not None and think_start >= 0 and think_end >= 0
        and completion.count("<think>") == completion.count("</think>") == 1
        and think_start < vision_tag[2][0] < vision_tag[2][1] < think_end
        and (reasoning_tag is None or vision_tag[2][1] <= reasoning_tag[2][0])
    )
    legal_vision = direct_vision or legacy_vision
    vision_text = vision_tag[0].strip() if legal_vision else None
    vision_span = vision_tag[1] if legal_vision and vision_text else None
    reasoning_text = reasoning_tag[0].strip() if reasoning_tag is not None else None
    answer = answer_tag[0].strip() if answer_tag is not None else None
    if answer == "":
        answer = None

    # A misplaced report marker still starts a report suffix: Teacher OPSD
    # must not teach confidence tokens merely because their order is wrong.
    # Literal markers inside free-text fields are excluded from this scan.
    protected_inner_spans = [
        tag[1] for tag in (vision_tag, reasoning_tag, answer_tag) if tag is not None
    ]
    report_start = -1
    for marker in re.finditer(re.escape("<confidence>"), completion):
        if not any(start <= marker.start() < end for start, end in protected_inner_spans):
            report_start = marker.start()
            break
    report_start_char = report_start if report_start >= 0 else None
    report_text = completion[report_start:] if report_start >= 0 else ""
    visual_confidence = _optional_score(report_text, "visual_confidence") if report_text else None
    answer_confidence = _optional_score(report_text, "answer_confidence") if report_text else None

    errors: list[str] = []
    if not legal_vision:
        errors.append("missing_or_invalid_vision")
    elif not vision_text:
        errors.append("empty_vision")
    if answer is None:
        errors.append("missing_or_empty_answer")
    if report_start_char is None:
        errors.append("missing_confidence")
    else:
        if visual_confidence is None:
            errors.append("invalid_visual_confidence")
        if answer_confidence is None:
            errors.append("invalid_answer_confidence")
    if full is None:
        errors.append("noncanonical_structure")
    return ParsedProxyCompletion(
        vision_text=vision_text,
        reasoning_text=reasoning_text,
        answer=answer,
        visual_confidence=visual_confidence,
        answer_confidence=answer_confidence,
        format_errors=tuple(errors),
        vision_span_chars=vision_span,
        report_start_char=report_start_char,
    )


def split_proxy_generated_eos(
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    *,
    generation_eos_token_id: int | Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Remove at most one actual terminal token before strict text parsing."""
    token_ids = tuple(int(value) for value in completion_token_ids)
    eos_ids: set[int] = set()
    for candidate in (getattr(tokenizer, "eos_token_id", None), generation_eos_token_id):
        if isinstance(candidate, (list, tuple, set)):
            eos_ids.update(int(value) for value in candidate)
        elif candidate is not None:
            eos_ids.add(int(candidate))
    if token_ids and token_ids[-1] in eos_ids:
        return token_ids[:-1], token_ids[-1:]
    return token_ids, ()


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids), skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _token_spans(
    tokenizer: Any, token_ids: tuple[int, ...], text: str
) -> tuple[tuple[int, int], ...] | None:
    """Derive spans only after checking token IDs and full decoded text."""
    try:
        if _decode(tokenizer, token_ids) != text:
            return None
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = tuple(int(value) for value in encoding["input_ids"])
        offsets = tuple((int(start), int(end)) for start, end in encoding["offset_mapping"])
        if encoded_ids == token_ids and len(offsets) == len(token_ids) and all(
            0 <= start <= end <= len(text) for start, end in offsets
        ) and all(
            offsets[index][0] >= offsets[index - 1][0]
            and offsets[index][1] >= offsets[index - 1][1]
            for index in range(1, len(offsets))
        ):
            return offsets
    except (TypeError, ValueError, KeyError, AttributeError):
        pass
    try:
        ends = [0]
        for count in range(1, len(token_ids) + 1):
            prefix = _decode(tokenizer, token_ids[:count])
            if not text.startswith(prefix) or len(prefix) < ends[-1]:
                return None
            ends.append(len(prefix))
        if ends[-1] != len(text):
            return None
        return tuple(zip(ends[:-1], ends[1:], strict=True))
    except (TypeError, ValueError, AttributeError):
        return None


def build_proxy_masks(
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    decoded_text: str,
    parsed: ParsedProxyCompletion,
) -> ProxyMasks:
    """Map free-text vision, all content, and confidence report to token IDs.

    With no ``<confidence>`` marker, every completion token remains content,
    so teacher distillation cannot be evaded by omitting the report. If a
    boundary falls inside a token, that token is excluded from vision; at the
    report boundary it belongs to report. An unverifiable vision span is zero
    while otherwise safe content tokens remain supervised.
    """
    if not isinstance(parsed, ParsedProxyCompletion):
        raise ProxyProtocolError("build_proxy_masks requires ParsedProxyCompletion")
    if not isinstance(decoded_text, str):
        raise ProxyProtocolError("decoded_text must be a string")
    token_ids = tuple(int(value) for value in completion_token_ids)
    size = len(token_ids)
    spans = _token_spans(tokenizer, token_ids, decoded_text)
    boundary = parsed.report_start_char
    failure: list[str] = []
    if boundary is not None and not (0 <= boundary < len(decoded_text) and decoded_text.startswith("<confidence>", boundary)):
        failure.append("invalid_report_boundary")
        boundary = 0
    if boundary is None:
        content_end = size
    elif spans is not None:
        # The first token that touches any report character is a report token.
        content_end = next((index for index, (start, end) in enumerate(spans) if end > boundary), size)
        if content_end == size:
            failure.append("report_boundary_not_mapped")
            content_end = 0
    else:
        # A verified decoded prefix can still retain teacher signal before
        # the report marker; route every unverified suffix token to report.
        content_end = 0
        try:
            if _decode(tokenizer, token_ids) != decoded_text:
                failure.append("completion_decode_mismatch")
            else:
                for index in range(1, size + 1):
                    prefix = _decode(tokenizer, token_ids[:index])
                    if decoded_text.startswith(prefix) and len(prefix) <= boundary:
                        content_end = index
                failure.append("token_spans_unavailable")
        except (TypeError, ValueError, AttributeError):
            failure.append("completion_decode_failed")

    content = (1,) * content_end + (0,) * (size - content_end)
    report = (0,) * content_end + (1,) * (size - content_end)
    vision = [0] * size
    if parsed.vision_span_chars is not None:
        if spans is None:
            failure.append("vision_span_unverifiable")
        else:
            visual_start, visual_end = parsed.vision_span_chars
            for index, (start, end) in enumerate(spans):
                if index < content_end and visual_start <= start < end <= visual_end:
                    vision[index] = 1
    return ProxyMasks(
        vision_mask=tuple(vision), content_mask=content,
        report_mask=report,
        report_start_token=content_end if boundary is not None else None,
        failure_reason=";".join(failure) if failure else None,
    )
