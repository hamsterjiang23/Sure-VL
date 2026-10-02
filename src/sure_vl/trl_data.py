"""Frozen paired-image manifests for TRL GOLD's vision-language path.

The source VL-Calibration-12K dataset has questions, answers, and original
images, but no prescribed fact slots or paired restricted/clear images. This
loader therefore accepts only a completed Sure-VL Example manifest. It does
not infer facts or silently treat the answer as the visual label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import Example, ProtocolError, load_examples_jsonl
from .trl_prompt import build_user_prompt


def _resolved_image(path_text: str, manifest_dir: Path, *, check_exists: bool) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if check_exists and not path.is_file():
        raise ProtocolError(f"image file does not exist: {path}")
    return str(path)


def _example_payload(example: Example, student_path: str, teacher_path: str) -> dict[str, Any]:
    return {
        "id": example.id,
        "split": example.split,
        "student_image": student_path,
        "teacher_image": teacher_path,
        "question": example.question,
        "required_visual_facts": {
            slot: {"canonical": fact.canonical, "aliases": list(fact.aliases)}
            for slot, fact in example.required_visual_facts.items()
        },
        "accepted_answers": list(example.accepted_answers),
    }


def manifest_to_gold_rows(
    manifest: str | Path, *, check_images: bool = True
) -> list[dict[str, Any]]:
    """Build raw GOLD VLM rows from one frozen, single-split Example JSONL.

    Paths are made absolute relative to the manifest file. ``image`` is the
    student view; ``teacher_image`` is the privileged clear view. Both remain
    paths in the returned rows so the datasets Image feature can decode them
    lazily into RGB PIL images. No images are copied or degraded here.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    examples = load_examples_jsonl(manifest_path)
    if len({example.split for example in examples}) != 1:
        raise ProtocolError(f"{manifest_path}: one manifest must contain exactly one split")
    rows: list[dict[str, Any]] = []
    for example in examples:
        student_path = _resolved_image(example.student_image, manifest_path.parent, check_exists=check_images)
        teacher_path = _resolved_image(example.teacher_image, manifest_path.parent, check_exists=check_images)
        if student_path == teacher_path:
            raise ProtocolError(f"{example.id}: student and teacher image paths must differ")
        payload = _example_payload(example, student_path, teacher_path)
        rows.append({
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_user_prompt(example)},
                ],
            }],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
            "image": student_path,
            "teacher_image": teacher_path,
            "example_id": example.id,
            "split": example.split,
            "example_payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "required_visual_facts_json": json.dumps(payload["required_visual_facts"], ensure_ascii=False, sort_keys=True),
            "accepted_answers": list(example.accepted_answers),
        })
    return rows


def assert_disjoint_manifests(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    """Reject ID or image overlap between training and held-out manifests."""
    if not train_rows or not eval_rows:
        raise ProtocolError("train and eval manifests must both be nonempty")
    train_splits = {row["split"] for row in train_rows}
    eval_splits = {row["split"] for row in eval_rows}
    if len(train_splits) != 1 or len(eval_splits) != 1 or train_splits == eval_splits:
        raise ProtocolError("train and eval manifests must name different single splits")
    for key in ("example_id", "image", "teacher_image"):
        shared = {row[key] for row in train_rows} & {row[key] for row in eval_rows}
        if shared:
            raise ProtocolError(f"train/eval overlap in {key}: {sorted(shared)[:5]}")


def build_gold_dataset(rows: list[dict[str, Any]]) -> Any:
    """Create a Hugging Face Dataset with lazily decoded student/teacher PIL images."""
    if not rows:
        raise ProtocolError("cannot build a GOLD dataset from no rows")
    try:
        from datasets import Dataset, Image
    except ImportError as error:
        raise RuntimeError("build_gold_dataset requires the optional train dependencies: datasets and Pillow") from error
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("image", Image(mode="RGB"))
    dataset = dataset.cast_column("teacher_image", Image(mode="RGB"))
    return dataset
