"""Frozen inputs, structured outputs, and deterministic V/Y verification.

The JSONL contract intentionally makes the required visual slots part of the
example, before a student output exists. A fact can be written as a canonical
string or as ``{"canonical": "...", "aliases": ["..."]}``.

Verification is exact after Unicode NFKC normalization, case folding, and
whitespace collapsing. It does not infer synonyms or use a model as a judge.
Any omitted required visual slot is false and remains in the V denominator.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class ProtocolError(ValueError):
    """An input or output violates the frozen evaluation contract."""


def normalize_text(value: str) -> str:
    """Normalize for exact matching without deleting punctuation or words."""
    if not isinstance(value, str):
        raise ProtocolError("value to normalize must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field} must be a string")
    if nonempty and not value.strip():
        raise ProtocolError(f"{field} must be nonempty")
    return value


def _keys(value: Any, field: str, required: set[str], optional: set[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{field} keys must be strings")
    allowed = required | (optional or set())
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ProtocolError(f"{field} is missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise ProtocolError(f"{field} has unknown keys: {', '.join(sorted(extra))}")
    return value


def _strings(value: Any, field: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, str):
        raise ProtocolError(f"{field} must be an array of strings")
    if nonempty and not value:
        raise ProtocolError(f"{field} must be nonempty")
    result = tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))
    normalized = [normalize_text(item) for item in result]
    if len(set(normalized)) != len(normalized):
        raise ProtocolError(f"{field} contains duplicate normalized values")
    return result


@dataclass(frozen=True)
class RequiredVisualFact:
    canonical: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _string(self.canonical, "fact.canonical")
        aliases = _strings(self.aliases, "fact.aliases", nonempty=False)
        if normalize_text(self.canonical) in {normalize_text(item) for item in aliases}:
            raise ProtocolError("fact.aliases duplicates the canonical value")
        object.__setattr__(self, "aliases", aliases)

    @classmethod
    def from_raw(cls, value: Any) -> RequiredVisualFact:
        if isinstance(value, str):
            return cls(canonical=value)
        data = _keys(value, "required visual fact", {"canonical"}, {"aliases"})
        return cls(canonical=data["canonical"], aliases=data.get("aliases", ()))

    def accepts(self, value: str) -> bool:
        candidate = normalize_text(value)
        return candidate in {normalize_text(item) for item in (self.canonical, *self.aliases)}


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    student_image: str
    teacher_image: str
    question: str
    required_visual_facts: Mapping[str, RequiredVisualFact]
    accepted_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("id", "split", "student_image", "teacher_image", "question"):
            _string(getattr(self, field), field)
        facts = self.required_visual_facts
        if not isinstance(facts, Mapping) or not facts:
            raise ProtocolError("required_visual_facts must be a nonempty object")
        frozen_facts: dict[str, RequiredVisualFact] = {}
        for slot, fact in facts.items():
            _string(slot, "required_visual_facts slot")
            if not isinstance(fact, RequiredVisualFact):
                raise ProtocolError(f"required_visual_facts[{slot!r}] must be a RequiredVisualFact")
            frozen_facts[slot] = fact
        object.__setattr__(self, "required_visual_facts", MappingProxyType(frozen_facts))
        object.__setattr__(self, "accepted_answers", _strings(self.accepted_answers, "accepted_answers", nonempty=True))

    @classmethod
    def from_dict(cls, value: Any) -> Example:
        data = _keys(
            value,
            "example",
            {"id", "split", "student_image", "teacher_image", "question", "required_visual_facts", "accepted_answers"},
        )
        raw_facts = data["required_visual_facts"]
        if not isinstance(raw_facts, Mapping):
            raise ProtocolError("required_visual_facts must be an object")
        facts = {slot: RequiredVisualFact.from_raw(fact) for slot, fact in raw_facts.items()}
        return cls(
            id=data["id"],
            split=data["split"],
            student_image=data["student_image"],
            teacher_image=data["teacher_image"],
            question=data["question"],
            required_visual_facts=facts,
            accepted_answers=data["accepted_answers"],
        )


def _percent(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 100:
        raise ProtocolError(f"{field} must be an integer from 0 to 100")
    return value


@dataclass(frozen=True)
class StudentOutput:
    id: str
    visual_facts: Mapping[str, str]
    reasoning: str
    answer: str
    visual_confidence: int
    conditional_answer_confidence: int

    def __post_init__(self) -> None:
        _string(self.id, "id")
        _string(self.reasoning, "reasoning")
        _string(self.answer, "answer")
        if not isinstance(self.visual_facts, Mapping):
            raise ProtocolError("visual_facts must be an object")
        frozen_facts: dict[str, str] = {}
        for slot, value in self.visual_facts.items():
            _string(slot, "visual_facts slot")
            frozen_facts[slot] = _string(value, f"visual_facts[{slot!r}]", nonempty=False)
        object.__setattr__(self, "visual_facts", MappingProxyType(frozen_facts))
        _percent(self.visual_confidence, "visual_confidence")
        _percent(self.conditional_answer_confidence, "conditional_answer_confidence")

    @classmethod
    def from_dict(cls, value: Any) -> StudentOutput:
        data = _keys(
            value,
            "student output",
            {"id", "visual_facts", "reasoning", "answer", "visual_confidence", "conditional_answer_confidence"},
        )
        return cls(
            id=data["id"],
            visual_facts=data["visual_facts"],
            reasoning=data["reasoning"],
            answer=data["answer"],
            visual_confidence=data["visual_confidence"],
            conditional_answer_confidence=data["conditional_answer_confidence"],
        )


@dataclass(frozen=True)
class OutputAttempt:
    """One output line, including assignable but invalid generated content."""

    id: str
    parsed: StudentOutput | None
    format_error: str | None

    def __post_init__(self) -> None:
        _string(self.id, "id")
        if self.parsed is None:
            _string(self.format_error, "format_error")
        elif not isinstance(self.parsed, StudentOutput) or self.parsed.id != self.id or self.format_error is not None:
            raise ProtocolError("valid output attempt requires matching parsed id and no format_error")


@dataclass(frozen=True)
class Verification:
    visual_correct: bool
    answer_correct: bool
    per_fact: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_fact", MappingProxyType(dict(self.per_fact)))


def verify(example: Example, output: StudentOutput) -> Verification:
    """Evaluate fixed slots and accepted answers; missing slots always fail V."""
    if not isinstance(example, Example) or not isinstance(output, StudentOutput):
        raise ProtocolError("verify requires an Example and StudentOutput")
    if example.id != output.id:
        raise ProtocolError(f"output id {output.id!r} does not match example id {example.id!r}")
    per_fact = {
        slot: slot in output.visual_facts and fact.accepts(output.visual_facts[slot])
        for slot, fact in example.required_visual_facts.items()
    }
    answer = normalize_text(output.answer)
    return Verification(
        visual_correct=all(per_fact.values()),
        answer_correct=answer in {normalize_text(item) for item in example.accepted_answers},
        per_fact=per_fact,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_jsonl(path: str | Path, record_type: type[Example] | type[StudentOutput]) -> tuple[Example, ...] | tuple[StudentOutput, ...]:
    records: list[Example | StudentOutput] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                if not line.strip():
                    raise ProtocolError("blank line")
                data = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                record = record_type.from_dict(data)
                if record.id in seen_ids:
                    raise ProtocolError(f"duplicate id: {record.id!r}")
                seen_ids.add(record.id)
                records.append(record)
            except (json.JSONDecodeError, ProtocolError) as error:
                raise ProtocolError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ProtocolError(f"{path}: JSONL file is empty")
    return tuple(records)


def load_examples_jsonl(path: str | Path) -> tuple[Example, ...]:
    """Load validated examples, rejecting duplicate IDs and blank lines."""
    return _load_jsonl(path, Example)


def load_outputs_jsonl(path: str | Path) -> tuple[StudentOutput, ...]:
    """Load validated student outputs, rejecting duplicate IDs and blank lines."""
    return _load_jsonl(path, StudentOutput)


def load_attempts_jsonl(path: str | Path) -> tuple[OutputAttempt, ...]:
    """Preserve malformed generated outputs when their metadata ID is valid.

    An empty file means no attempts. Invalid JSON, duplicate JSON keys, and
    missing/invalid IDs cannot be assigned to an example and raise instead.
    """
    attempts: list[OutputAttempt] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                if not line.strip():
                    raise ProtocolError("blank line has no assignable id")
                data = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                if not isinstance(data, Mapping) or "id" not in data:
                    raise ProtocolError("output attempt must be an object with id")
                attempt_id = _string(data["id"], "id")
                if attempt_id in seen_ids:
                    raise ProtocolError(f"duplicate id: {attempt_id!r}")
                seen_ids.add(attempt_id)
                try:
                    parsed = StudentOutput.from_dict(data)
                except ProtocolError as error:
                    attempts.append(OutputAttempt(id=attempt_id, parsed=None, format_error=str(error)))
                else:
                    attempts.append(OutputAttempt(id=attempt_id, parsed=parsed, format_error=None))
            except (json.JSONDecodeError, ProtocolError) as error:
                raise ProtocolError(f"{path}:{line_number}: {error}") from error
    return tuple(attempts)
