"""Paired-image GOLD rows for answer-labeled internal visual certainty.

The train and validation manifests contain neither visual fact labels nor a
static visual-confidence target. The latter is measured on each rollout from
the privileged-view teacher and the current student distributions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .proxy_prompt import build_proxy_student_messages
from .proxy_protocol import ProxyExample, ProxyProtocolError, load_proxy_examples_jsonl


def _resolve_image(path_text: str, manifest_dir: Path, *, check_images: bool) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if check_images and not path.is_file():
        raise ProxyProtocolError(f"image file does not exist: {path}")
    return str(path)


def manifest_to_proxy_rows(
    manifest: str | Path, *, check_images: bool = True
) -> list[dict[str, Any]]:
    """Convert one single-split manifest into raw GOLD VLM rows."""
    manifest_path = Path(manifest).expanduser().resolve()
    examples = load_proxy_examples_jsonl(manifest_path)
    if len({example.split for example in examples}) != 1:
        raise ProxyProtocolError(f"{manifest_path}: manifest must contain exactly one split")
    rows: list[dict[str, Any]] = []
    for example in examples:
        student_path = _resolve_image(example.student_image, manifest_path.parent, check_images=check_images)
        teacher_path = _resolve_image(example.teacher_image, manifest_path.parent, check_images=check_images)
        if student_path == teacher_path:
            raise ProxyProtocolError(f"{example.id}: student and teacher image paths must differ")
        payload = ProxyExample(
            id=example.id, split=example.split,
            student_image=student_path, teacher_image=teacher_path,
            question=example.question, accepted_answers=example.accepted_answers,
            student_image_hint=example.student_image_hint,
            teacher_question=example.teacher_question,
            teacher_evidence=example.teacher_evidence,
        ).to_dict()
        rows.append({
            "prompt": build_proxy_student_messages(example),
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
            "image": student_path,
            "teacher_image": teacher_path,
            "example_id": example.id,
            "split": example.split,
            "student_image_hint": example.student_image_hint,
            "example_payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "accepted_answers": list(example.accepted_answers),
        })
    return rows


def assert_disjoint_proxy_manifests(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> None:
    """Reject ID or image reuse across the fixed train/validation split."""
    if not train_rows or not validation_rows:
        raise ProxyProtocolError("train and validation manifests must be nonempty")
    train_splits = {row["split"] for row in train_rows}
    validation_splits = {row["split"] for row in validation_rows}
    if len(train_splits) != 1 or len(validation_splits) != 1 or train_splits == validation_splits:
        raise ProxyProtocolError("train and validation manifests must name different single splits")
    for field in ("example_id", "image", "teacher_image"):
        shared = {row[field] for row in train_rows} & {row[field] for row in validation_rows}
        if shared:
            raise ProxyProtocolError(f"train/validation overlap in {field}: {sorted(shared)[:5]}")
    train_images = {row[field] for row in train_rows for field in ("image", "teacher_image")}
    validation_images = {row[field] for row in validation_rows for field in ("image", "teacher_image")}
    if shared := train_images & validation_images:
        raise ProxyProtocolError(f"train/validation overlap in image roles: {sorted(shared)[:5]}")


def build_proxy_dataset(rows: list[dict[str, Any]]) -> Any:
    """Decode both image columns lazily through the datasets Image feature."""
    if not rows:
        raise ProxyProtocolError("cannot build a proxy dataset from no rows")
    try:
        from datasets import Dataset, Image
    except ImportError as error:
        raise RuntimeError("build_proxy_dataset requires the optional train dependencies") from error
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("image", Image(mode="RGB"))
    dataset = dataset.cast_column("teacher_image", Image(mode="RGB"))
    return dataset
