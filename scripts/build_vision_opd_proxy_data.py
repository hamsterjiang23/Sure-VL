#!/usr/bin/env python3
"""Build Sure-VL manifests from locally extracted official Vision-OPD-6K pairs.

Download the pinned ``train.jsonl`` and extract only the official ``images``
and ``teacher_images`` archives into ``source_root``. This builder never
reconstructs crops or treats ``extra_info.question`` as teacher evidence.
``--inspect-only`` reports the number of complete local pairs before a split
is frozen. The source metadata has a single training split, so the new
train/validation split is deterministic and groups by ``original_images[0]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sure_vl.proxy_protocol import ProxyExample


HF_REPO = "yuanqianhao/Vision-OPD-6K"
HF_REVISION = "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4"
METADATA_SHA256 = "8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df"
METADATA_ROW_COUNT = 6241
REDBOX_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)
OFFICIAL_FIELDS = {
    "images", "teacher_images", "original_images", "bbox", "problem", "answer", "extra_info",
}


@dataclass(frozen=True)
class OfficialRow:
    index: int
    student_rel: str
    teacher_rel: str
    original_rel: str
    student_path: Path
    teacher_path: Path
    original_path: Path
    bbox: tuple[int, int, int, int]
    student_question: str
    teacher_question: str
    answer: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"nonfinite JSON value: {value}")


def _one_official_path(value: Any, prefix: str, source_root: Path, index: int) -> tuple[str, Path]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        raise ValueError(f"row {index}: {prefix} must be one relative path")
    rel = value[0]
    parts = PurePosixPath(rel)
    if (parts.is_absolute() or len(parts.parts) != 2 or parts.parts[0] != prefix
            or parts.parts[1] in {"", ".", ".."} or "\\" in rel):
        raise ValueError(f"row {index}: invalid official {prefix} path: {rel!r}")
    candidate = (source_root / rel).resolve()
    if not candidate.is_relative_to(source_root):
        raise ValueError(f"row {index}: {prefix} path escapes source_root")
    return rel, candidate


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split())


def _validate_questions(problem: Any, extra_question: Any, index: int) -> tuple[str, str]:
    if not isinstance(problem, str) or not problem.startswith("<image>") or problem.count("<image>") != 1:
        raise ValueError(f"row {index}: problem must start with exactly one <image> token")
    if not isinstance(extra_question, str) or not extra_question.strip():
        raise ValueError(f"row {index}: extra_info.question is required")
    student_question = problem[len("<image>"):].strip()
    teacher_question = extra_question.strip()
    if student_question.count(REDBOX_HINT) != 1 or REDBOX_HINT in teacher_question:
        raise ValueError(f"row {index}: student needs one red-box hint and teacher none")
    clean_student = student_question.replace(REDBOX_HINT, "", 1)
    if _normalize_spaces(clean_student) != _normalize_spaces(teacher_question):
        raise ValueError(f"row {index}: clean student question differs from extra_info.question")
    return student_question, teacher_question


def _validate_answer(answer: Any, extra_answer: Any, teacher_question: str, index: int) -> str:
    if not isinstance(answer, str) or answer not in {"A", "B", "C", "D"} or extra_answer != answer:
        raise ValueError(f"row {index}: answer must be matching official A-D letters")
    choices = re.findall(r"(?m)^([A-D])\.\s+\S", teacher_question)
    if sorted(choices) != ["A", "B", "C", "D"]:
        raise ValueError(f"row {index}: teacher question must contain each A-D choice once")
    return answer


def load_official_rows(
    source_root: Path,
    *,
    expected_metadata_sha256: str = METADATA_SHA256,
    expected_row_count: int = METADATA_ROW_COUNT,
) -> tuple[list[OfficialRow], str]:
    """Verify the pinned metadata and all row contracts without requiring images."""
    source_root = source_root.expanduser().resolve()
    metadata_path = source_root / "train.jsonl"
    digest = _sha256(metadata_path)
    if digest != expected_metadata_sha256:
        raise ValueError(f"official train.jsonl SHA256 differs: expected {expected_metadata_sha256}, got {digest}")
    rows: list[OfficialRow] = []
    seen_student: set[str] = set()
    seen_teacher: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                raise ValueError(f"row {index}: blank metadata line")
            raw = json.loads(line, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_nonfinite)
            if not isinstance(raw, dict) or set(raw) != OFFICIAL_FIELDS:
                raise ValueError(f"row {index}: official metadata fields differ")
            student_rel, student_path = _one_official_path(raw["images"], "images", source_root, index)
            teacher_rel, teacher_path = _one_official_path(
                raw["teacher_images"], "teacher_images", source_root, index,
            )
            original_rel, original_path = _one_official_path(
                raw["original_images"], "original_images", source_root, index,
            )
            if student_rel in seen_student or teacher_rel in seen_teacher:
                raise ValueError(f"row {index}: duplicate student/teacher path")
            seen_student.add(student_rel)
            seen_teacher.add(teacher_rel)
            bbox = raw["bbox"]
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(type(v) is not int for v in bbox)
                    or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3])):
                raise ValueError(f"row {index}: bbox must be a positive integer xyxy region")
            extra = raw["extra_info"]
            if not isinstance(extra, dict) or set(extra) != {"answer", "question"}:
                raise ValueError(f"row {index}: extra_info must contain answer and question")
            student_question, teacher_question = _validate_questions(
                raw["problem"], extra["question"], index,
            )
            answer = _validate_answer(raw["answer"], extra["answer"], teacher_question, index)
            rows.append(OfficialRow(
                index=index, student_rel=student_rel, teacher_rel=teacher_rel,
                original_rel=original_rel, student_path=student_path,
                teacher_path=teacher_path, original_path=original_path,
                bbox=tuple(bbox), student_question=student_question,
                teacher_question=teacher_question, answer=answer,
            ))
    if len(rows) != expected_row_count:
        raise ValueError(f"official row count differs: expected {expected_row_count}, got {len(rows)}")
    return rows, digest


def inspect_local_pairs(rows: list[OfficialRow]) -> tuple[list[OfficialRow], dict[str, int]]:
    available: list[OfficialRow] = []
    student_only = teacher_only = neither = 0
    for row in rows:
        student, teacher = row.student_path.is_file(), row.teacher_path.is_file()
        if student and teacher:
            available.append(row)
        elif student:
            student_only += 1
        elif teacher:
            teacher_only += 1
        else:
            neither += 1
    return available, {
        "metadata_rows": len(rows), "paired_images": len(available),
        "paired_original_groups": len({row.original_rel for row in available}),
        "student_only": student_only, "teacher_only": teacher_only, "neither": neither,
    }


def _rank(row: OfficialRow, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{row.original_rel}\0{row.index}".encode()).hexdigest()


def select_rows(
    available: list[OfficialRow], *, train_count: int, validation_count: int, seed: str
) -> tuple[list[OfficialRow], list[OfficialRow]]:
    if type(train_count) is not int or type(validation_count) is not int or train_count < 1 or validation_count < 1:
        raise ValueError("train_count and validation_count must be positive integers")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty string")
    # One item per original image means correlated questions cannot cross splits.
    representatives: dict[str, OfficialRow] = {}
    for row in sorted(available, key=lambda item: (_rank(item, seed), item.index)):
        representatives.setdefault(row.original_rel, row)
    ranked = sorted(representatives.values(), key=lambda item: (_rank(item, seed), item.index))
    need = train_count + validation_count
    if len(ranked) < need:
        raise ValueError(f"need {need} paired original groups, found {len(ranked)}")
    return ranked[:train_count], ranked[train_count:need]


def _verified_image_sha256(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("image verification requires Pillow; use `uv run --extra train`") from error
    with Image.open(path) as image:
        image.verify()
    return _sha256(path)


def build_dataset(
    source_root: Path,
    output_dir: Path,
    *,
    train_count: int = 32,
    validation_count: int = 8,
    seed: str = "sure-vl-vision-opd-6k-v1",
    expected_metadata_sha256: str = METADATA_SHA256,
    expected_row_count: int = METADATA_ROW_COUNT,
) -> dict[str, Any]:
    """Freeze a fresh, path-referencing split of official paired images."""
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows, metadata_sha256 = load_official_rows(
        source_root, expected_metadata_sha256=expected_metadata_sha256,
        expected_row_count=expected_row_count,
    )
    available, inventory = inspect_local_pairs(rows)
    train, validation = select_rows(
        available, train_count=train_count, validation_count=validation_count, seed=seed,
    )
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    audit: list[dict[str, Any]] = []
    seen_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, split_rows in (("train", train), ("validation", validation)):
        for row in split_rows:
            student_hash = _verified_image_sha256(row.student_path)
            teacher_hash = _verified_image_sha256(row.teacher_path)
            original_hash = _verified_image_sha256(row.original_path) if row.original_path.is_file() else None
            seen_hashes[split].update((student_hash, teacher_hash))
            if original_hash is not None:
                seen_hashes[split].add(original_hash)
            sample_id = f"vision-opd-6k-{row.index:05d}"
            manifest = ProxyExample(
                id=sample_id, split=split,
                student_image=str(row.student_path), teacher_image=str(row.teacher_path),
                question=row.student_question, teacher_question=row.teacher_question,
                accepted_answers=(row.answer,),
            ).to_dict()
            manifests[split].append(manifest)
            audit.append({
                "id": sample_id, "split": split, "official_row_index": row.index,
                "student_source_path": row.student_rel,
                "teacher_source_path": row.teacher_rel,
                "original_scene_group": row.original_rel,
                "official_bbox_xyxy": list(row.bbox),
                "student_image_sha256": student_hash,
                "teacher_image_sha256": teacher_hash,
                "original_image_sha256": original_hash,
                "accepted_answer": row.answer,
                "teacher_evidence_present": False,
            })
    if seen_hashes["train"] & seen_hashes["validation"]:
        raise ValueError("train/validation overlap in actual image SHA256")
    if {row.original_rel for row in train} & {row.original_rel for row in validation}:
        raise ValueError("train/validation overlap in original scene group")
    if _sha256(source_root / "train.jsonl") != metadata_sha256:
        raise ValueError("source metadata changed during the build")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    for split, manifest_rows in manifests.items():
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in manifest_rows),
            encoding="utf-8",
        )
    provenance = {
        "builder": "sure_vl_official_vision_opd_6k_v1",
        "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
        "hf_metadata_url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/train.jsonl",
        "metadata_sha256": metadata_sha256, "metadata_rows": len(rows),
        "source_root": str(source_root), "selection_seed": seed,
        "selection_policy": "SHA256(seed, original_images[0], row_index); one row per original image",
        "selection_counts": {"train": len(train), "validation": len(validation)},
        "inventory": inventory,
        "view_policy": "use the official preconstructed red-box full image and teacher crop unchanged",
        "image_materialization": "none; manifests reference extracted official files by absolute path",
        "source_split": "train only; Sure-VL validation is a deterministic held-out subset",
        "original_image_policy": (
            "Original images are not required for this paired-view adaptation. "
            "Original-image paths group scenes; content SHA256 is recorded when a local original exists."
        ),
        "teacher_evidence_policy": "absent; extra_info.question is a question, not evidence",
        "manifest_sha256": {split: _sha256(output_dir / f"{split}.jsonl") for split in manifests},
        "examples": audit,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--seed", default="sure-vl-vision-opd-6k-v1")
    args = parser.parse_args(argv)
    if args.inspect_only:
        rows, digest = load_official_rows(args.source_root)
        _, inventory = inspect_local_pairs(rows)
        print(json.dumps({"metadata_sha256": digest, "inventory": inventory}, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --inspect-only is used")
    provenance = build_dataset(
        args.source_root, args.output_dir, train_count=args.train_count,
        validation_count=args.validation_count, seed=args.seed,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "selection_counts": provenance["selection_counts"],
        "inventory": provenance["inventory"],
        "metadata_sha256": provenance["metadata_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
