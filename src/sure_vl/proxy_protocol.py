"""Input and answer-only labels for the internal visual-certainty protocol.

The visual description has no factual ground-truth slots. ``accepted_answers``
supervises only the final answer event Y; the visual certainty target is built
later from frozen teacher/student token distributions.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProxyProtocolError(ValueError):
    """A proxy-protocol input violates its frozen data contract."""


def normalize_proxy_answer(value: str) -> str:
    """NFKC, casefold, and collapse whitespace; retain punctuation."""
    if not isinstance(value, str):
        raise ProxyProtocolError("answer must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProxyProtocolError(f"{name} must be a nonempty string")
    return value


def _copy_json_value(value: Any, path: str, active: set[int]) -> Any:
    """Copy a JSON value without coercing keys or accepting nonfinite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProxyProtocolError(f"{path} must not contain NaN or Infinity")
        return value
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in active:
            raise ProxyProtocolError(f"{path} must not contain a cycle")
        active.add(identity)
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ProxyProtocolError(f"{path} object keys must be strings")
                return {key: _copy_json_value(item, f"{path}.{key}", active)
                        for key, item in value.items()}
            return [_copy_json_value(item, f"{path}[{index}]", active)
                    for index, item in enumerate(value)]
        finally:
            active.remove(identity)
    raise ProxyProtocolError(f"{path} must contain only JSON values")


def validate_teacher_evidence(value: Any) -> str | dict[str, Any] | list[Any] | None:
    """Accept optional nonempty text or a finite JSON object/array."""
    if value is None:
        return None
    if isinstance(value, str):
        return _nonempty_text(value, "teacher_evidence")
    if not isinstance(value, (dict, list)):
        raise ProxyProtocolError("teacher_evidence must be nonempty text or a JSON object/array")
    return _copy_json_value(value, "teacher_evidence", set())


@dataclass(frozen=True)
class ProxyExample:
    id: str
    split: str
    student_image: str
    teacher_image: str
    question: str
    accepted_answers: tuple[str, ...]
    student_image_hint: str | None = None
    teacher_evidence: str | dict[str, Any] | list[Any] | None = None

    def __post_init__(self) -> None:
        for name in ("id", "split", "student_image", "teacher_image", "question"):
            _nonempty_text(getattr(self, name), name)
        answers = self.accepted_answers
        if not isinstance(answers, (list, tuple)) or not answers:
            raise ProxyProtocolError("accepted_answers must be a nonempty array")
        checked = tuple(_nonempty_text(value, f"accepted_answers[{index}]") for index, value in enumerate(answers))
        normalized = [normalize_proxy_answer(value) for value in checked]
        if len(set(normalized)) != len(normalized):
            raise ProxyProtocolError("accepted_answers contains duplicate normalized answers")
        object.__setattr__(self, "accepted_answers", checked)
        if self.student_image_hint is not None:
            object.__setattr__(self, "student_image_hint",
                           _nonempty_text(self.student_image_hint, "student_image_hint"))
        object.__setattr__(self, "teacher_evidence", validate_teacher_evidence(self.teacher_evidence))

    @classmethod
    def from_dict(cls, raw: Any) -> ProxyExample:
        if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
            raise ProxyProtocolError("proxy example must be an object with string keys")
        required = {"id", "split", "student_image", "teacher_image", "question", "accepted_answers"}
        allowed = required | {"student_image_hint", "teacher_evidence"}
        missing = required - raw.keys()
        extra = raw.keys() - allowed
        if missing or extra:
            raise ProxyProtocolError(f"proxy example keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(
            id=raw["id"], split=raw["split"], student_image=raw["student_image"],
            teacher_image=raw["teacher_image"], question=raw["question"],
            accepted_answers=raw["accepted_answers"],
            student_image_hint=raw.get("student_image_hint"),
            teacher_evidence=raw.get("teacher_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "split": self.split,
            "student_image": self.student_image,
            "teacher_image": self.teacher_image,
            "question": self.question,
            "accepted_answers": list(self.accepted_answers),
        }
        if self.student_image_hint is not None:
            result["student_image_hint"] = self.student_image_hint
        if self.teacher_evidence is not None:
            result["teacher_evidence"] = validate_teacher_evidence(self.teacher_evidence)
        return result


def verify_proxy_answer(example: ProxyExample, answer: str | None) -> bool | None:
    """Return None for an unextractable answer, rather than fabricating Y=0."""
    if not isinstance(example, ProxyExample):
        raise ProxyProtocolError("verify_proxy_answer requires a ProxyExample")
    if answer is None:
        return None
    candidate = normalize_proxy_answer(answer)
    if not candidate:
        return None
    return candidate in {normalize_proxy_answer(item) for item in example.accepted_answers}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProxyProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ProxyProtocolError(f"nonfinite JSON number is forbidden: {value}")


def load_proxy_examples_jsonl(path: str | Path) -> tuple[ProxyExample, ...]:
    """Load one frozen JSONL manifest, rejecting blanks and duplicate IDs."""
    source_path = Path(path)
    examples: list[ProxyExample] = []
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                if not line.strip():
                    raise ProxyProtocolError("blank line")
                example = ProxyExample.from_dict(json.loads(
                    line, object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=_reject_nonfinite_json_constant,
                ))
                if example.id in seen:
                    raise ProxyProtocolError(f"duplicate example ID: {example.id!r}")
                seen.add(example.id)
                examples.append(example)
            except (json.JSONDecodeError, ProxyProtocolError) as error:
                raise ProxyProtocolError(f"{source_path}:{line_number}: {error}") from error
    if not examples:
        raise ProxyProtocolError(f"{source_path}: JSONL file is empty")
    return tuple(examples)
