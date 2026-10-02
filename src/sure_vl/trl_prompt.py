"""VL-Calibration-style prompt and strict completion adapter for TRL.

This module keeps the two Sure-VL events: visual confidence concerns *all*
predeclared visual facts, and conditional answer confidence concerns the answer
given correct visual facts. These are not VL-Calibration's harmonic-mean scores.
The caller supplies the student image as multimodal message content; the prompt
contains only the question and output instructions, never the fact answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .protocol import Example, ProtocolError, StudentOutput


_CONTENT_PREFIX = re.compile(
    r"\A\s*<think>\s*<vision>(?P<vision>.*?)</vision>\s*"
    r"<reasoning>(?P<reasoning>.*?)</reasoning>\s*</think>\s*",
    re.DOTALL,
)
_REPORT_SUFFIX = re.compile(
    r"\s*(?P<report><confidence>\s*"
    r"<vision_confidence>(?P<visual>0|[1-9][0-9]?|100)</vision_confidence>\s*"
    r"<conditional_answer_confidence>(?P<answer>0|[1-9][0-9]?|100)"
    r"</conditional_answer_confidence>\s*</confidence>)\s*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedCompletion:
    output: StudentOutput
    report_start_char: int


@dataclass(frozen=True)
class SegmentMasks:
    """Disjoint completion-token masks. ``None`` means fail-closed routing.

    On malformed text or a decode mismatch, all completion tokens are assigned
    to the report mask. If a verified token straddles the report boundary, that
    token enters the report mask; preceding tokens can still receive content
    teacher distillation. The caller still scores and retains malformed
    rollouts in RL accounting.
    """

    content_mask: tuple[int, ...]
    report_mask: tuple[int, ...]
    report_start_token: int | None
    failure_reason: str | None = None
    boundary_exact: bool = False


def build_user_prompt(example: Example) -> str:
    """Build the text part of a student user message from frozen slot names."""
    if not isinstance(example, Example):
        raise ProtocolError("build_user_prompt requires an Example")
    visual_shape = json.dumps(
        {slot: "visible value" for slot in example.required_visual_facts},
        ensure_ascii=False,
    )
    return (
        f"{example.question.strip()}\n\n"
        "First inspect the image and record only the requested visual facts. "
        "Then reason from those facts and give a final answer. Use exactly this order and format:\n"
        "<think>\n"
        f"<vision>{visual_shape}</vision>\n"
        "<reasoning>Your reasoning</reasoning>\n"
        "</think>\n"
        "\\boxed{Your final answer}\n"
        "<confidence>\n"
        "<vision_confidence>0-100 integer</vision_confidence>\n"
        "<conditional_answer_confidence>0-100 integer</conditional_answer_confidence>\n"
        "</confidence>\n"
        "Replace each visual JSON value with your observation and keep the slot keys exactly as shown. "
        "Do not use a confidence analysis block. Report confidence only after the answer. "
        "Visual confidence is the chance that every requested visual fact is correct. "
        "Conditional answer confidence is the chance that the answer is correct assuming those facts are correct."
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate visual fact key: {key!r}")
        result[key] = value
    return result


def _boxed_answer(text: str, start: int) -> tuple[str, int]:
    marker = r"\boxed{"
    if not text.startswith(marker, start):
        raise ProtocolError("expected boxed answer immediately after </think>")
    depth = 1
    answer_start = start + len(marker)
    for index in range(answer_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[answer_start:index].strip(), index + 1
    raise ProtocolError("boxed answer has unbalanced braces")


def parse_student_completion(example: Example, completion: str) -> ParsedCompletion:
    """Parse one complete output; reject extra text and wrong segment order.

    Missing declared fact keys remain valid parsed content so the frozen
    verifier can score those facts as incorrect. Unknown or duplicate keys are
    format errors. Confidence values must be canonical integers from 0 to 100.
    """
    if not isinstance(example, Example):
        raise ProtocolError("parse_student_completion requires an Example")
    if not isinstance(completion, str):
        raise ProtocolError("completion must be a string")
    prefix = _CONTENT_PREFIX.match(completion)
    if prefix is None:
        raise ProtocolError("expected <think><vision>...</vision><reasoning>...</reasoning></think>")
    try:
        facts = json.loads(prefix.group("vision").strip(), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"visual facts must be a JSON object: {error.msg}") from error
    if not isinstance(facts, dict):
        raise ProtocolError("visual facts must be a JSON object")
    unknown = facts.keys() - example.required_visual_facts.keys()
    if unknown:
        raise ProtocolError(f"unknown visual fact keys: {', '.join(sorted(unknown))}")
    if any(not isinstance(value, str) for value in facts.values()):
        raise ProtocolError("visual fact values must be strings")

    answer, answer_end = _boxed_answer(completion, prefix.end())
    suffix = _REPORT_SUFFIX.fullmatch(completion[answer_end:])
    if suffix is None:
        raise ProtocolError("expected final <confidence> block with two integer reports")
    report_start = answer_end + suffix.start("report")
    output = StudentOutput(
        id=example.id,
        visual_facts=facts,
        reasoning=prefix.group("reasoning").strip(),
        answer=answer,
        visual_confidence=int(suffix.group("visual")),
        conditional_answer_confidence=int(suffix.group("answer")),
    )
    return ParsedCompletion(output=output, report_start_char=report_start)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def split_generated_eos(
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    *,
    generation_eos_token_id: int | Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Separate one trailing EOS ID before strict text parsing.

    The returned suffix is empty or a one-token tuple. After building masks
    for the first tuple, append one report-mask token for that EOS. EOS is an
    end marker, never a content token for teacher distillation.
    """
    token_ids = tuple(int(value) for value in completion_token_ids)
    eos_ids: set[int] = set()
    for eos in (getattr(tokenizer, "eos_token_id", None), generation_eos_token_id):
        if isinstance(eos, (list, tuple, set)):
            eos_ids.update(int(value) for value in eos)
        elif eos is not None:
            eos_ids.add(int(eos))
    if token_ids and token_ids[-1] in eos_ids:
        return token_ids[:-1], token_ids[-1:]
    return token_ids, ()


def _offset_boundary(tokenizer: Any, token_ids: tuple[int, ...], text: str, boundary: int) -> tuple[int, bool] | None:
    """Use verified fast-tokenizer offsets and conservatively route straddles."""
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = tuple(int(value) for value in encoding["input_ids"])
        offsets = encoding["offset_mapping"]
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if encoded_ids != token_ids or len(offsets) != len(token_ids):
        return None
    for index, (start, end) in enumerate(offsets):
        if end <= boundary:
            continue
        if start > boundary:
            return None
        # The first token touching the marker goes to the report segment.
        # Verify decoded halves because offset maps can be misleading around
        # added tokens and Unicode byte fragments.
        try:
            if _decode(tokenizer, token_ids[:index]) == text[:start] and _decode(tokenizer, token_ids[index:]) == text[start:]:
                return index, start == boundary
        except (TypeError, ValueError, AttributeError):
            return None
        return None
    return None


def _decode_boundary(tokenizer: Any, token_ids: tuple[int, ...], text: str, boundary: int) -> int | None:
    """Fallback for tokenizers without offsets, using exact decoded halves."""
    prefix = text[:boundary]
    candidates: set[int] = set()
    try:
        candidates.add(len(tokenizer.encode(prefix, add_special_tokens=False)))
    except (TypeError, ValueError, AttributeError):
        pass
    # A byte token may change the length of a decoded prefix near the split.
    # Probe a small neighborhood around the independently encoded prefix.
    nearby = {index + shift for index in candidates for shift in range(-4, 5)}
    for index in sorted(candidates | nearby):
        if 0 <= index <= len(token_ids):
            try:
                if _decode(tokenizer, token_ids[:index]) == prefix and _decode(tokenizer, token_ids[index:]) == text[boundary:]:
                    return index
            except (TypeError, ValueError, AttributeError):
                return None
    return None


def build_content_report_masks(
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    decoded_text: str,
    report_start_char: int | None,
) -> SegmentMasks:
    """Map a successfully parsed ``<confidence>`` start to token masks.

    Pass ``None`` for malformed completions. The function never guesses a
    boundary from malformed text. If the parsed character boundary cannot be
    mapped exactly to a token edge, it routes all tokens to the report mask.
    """
    token_ids = tuple(int(value) for value in completion_token_ids)
    size = len(token_ids)

    def fail(reason: str) -> SegmentMasks:
        return SegmentMasks((0,) * size, (1,) * size, None, reason)

    if report_start_char is None:
        return fail("completion was not parsed")
    if not isinstance(decoded_text, str):
        return fail("decoded_text is not a string")
    if type(report_start_char) is not int or not 0 <= report_start_char < len(decoded_text):
        return fail("invalid report character boundary")
    if not decoded_text.startswith("<confidence>", report_start_char):
        return fail("boundary does not start at <confidence>")
    try:
        if _decode(tokenizer, token_ids) != decoded_text:
            return fail("completion tokens do not decode to supplied text")
    except (TypeError, ValueError, AttributeError):
        return fail("completion tokens cannot be decoded")

    offset_result = _offset_boundary(tokenizer, token_ids, decoded_text, report_start_char)
    if offset_result is None:
        index = _decode_boundary(tokenizer, token_ids, decoded_text, report_start_char)
        if index is None:
            return fail("report token boundary cannot be verified")
        exact = True
    else:
        index, exact = offset_result
    return SegmentMasks(
        (1,) * index + (0,) * (size - index),
        (0,) * index + (1,) * (size - index),
        index,
        boundary_exact=exact,
    )
