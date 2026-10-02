#!/usr/bin/env python3
"""Build new paired-view proxy manifests from explicit source-image ROI data.

Input JSONL fields per row: id, split (train/validation), source_image,
question, accepted_answers, optional evidence_bbox_xyxy, optional
source_image_sha256, and optional teacher_evidence. A bbox is required unless
--allow-no-roi is explicit. Bboxes are half-open xyxy pixel coordinates after
EXIF orientation. No existing output directory is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sure_vl.proxy_protocol import ProxyExample, validate_teacher_evidence
from sure_vl.teacher_view import (
    STUDENT_FOCUS_HINT, VISION_OPD_REFERENCE_COMMIT, build_teacher_view,
    validate_source_bbox,
)


_SOURCE_REQUIRED = {"id", "split", "source_image", "question", "accepted_answers"}
_SOURCE_ALLOWED = _SOURCE_REQUIRED | {"evidence_bbox_xyxy", "source_image_sha256", "teacher_evidence"}


@dataclass(frozen=True)
class SourceRow:
    line_number: int
    raw: dict[str, Any]
    image_path: Path
    image_sha256: str
    bbox_xyxy: tuple[int, int, int, int] | None
    teacher_evidence: str | dict[str, Any] | list[Any] | None


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


def load_source_rows(path: Path, *, allow_no_roi: bool) -> list[SourceRow]:
    """Preflight every row before creating an output directory."""
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError("privileged view builder requires Pillow; use `uv run --extra train`") from error
    path = path.expanduser().resolve()
    rows: list[SourceRow] = []
    ids: set[str] = set()
    split_image_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"input line {line_number} is blank")
        raw = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(raw, dict):
            raise ValueError(f"input line {line_number} must be a JSON object")
        missing, extra = _SOURCE_REQUIRED - raw.keys(), raw.keys() - _SOURCE_ALLOWED
        if missing or extra:
            raise ValueError(f"input line {line_number} keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
        sample_id, split = raw["id"], raw["split"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in ids:
            raise ValueError(f"input line {line_number} needs a unique nonempty id")
        if not isinstance(split, str) or split not in split_image_hashes:
            raise ValueError(f"input line {line_number} split must be train or validation")
        ids.add(sample_id)
        image_name = raw["source_image"]
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"input line {line_number} source_image must be a path")
        candidate = Path(image_name).expanduser()
        image_path = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        if not image_path.is_file():
            raise ValueError(f"input line {line_number} source image does not exist: {image_path}")
        image_sha256 = _sha256(image_path)
        asserted_hash = raw.get("source_image_sha256")
        if asserted_hash is not None:
            if not isinstance(asserted_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", asserted_hash):
                raise ValueError(f"input line {line_number} source_image_sha256 must be lowercase SHA256")
            if asserted_hash != image_sha256:
                raise ValueError(f"input line {line_number} source_image_sha256 differs from the file")
        with Image.open(image_path) as original:
            oriented_size = ImageOps.exif_transpose(original).size
        bbox_raw = raw.get("evidence_bbox_xyxy")
        if bbox_raw is None:
            if not allow_no_roi:
                raise ValueError(f"input line {line_number} has no evidence_bbox_xyxy; pass --allow-no-roi explicitly")
            bbox = None
        else:
            bbox = validate_source_bbox(bbox_raw, oriented_size)
        evidence = validate_teacher_evidence(raw.get("teacher_evidence"))
        # Reuse the manifest's answer and question checks before any image write.
        ProxyExample(
            id=sample_id, split=split, student_image="pending.student.png",
            teacher_image="pending.teacher.png", question=raw["question"],
            accepted_answers=raw["accepted_answers"], teacher_evidence=evidence,
        )
        split_image_hashes[split].add(image_sha256)
        rows.append(SourceRow(line_number, raw, image_path, image_sha256, bbox, evidence))
    if not all(split_image_hashes.values()):
        raise ValueError("source JSONL must contain nonempty train and validation splits")
    if split_image_hashes["train"] & split_image_hashes["validation"]:
        raise ValueError("train and validation source images overlap by SHA256")
    return rows


def build_dataset(
    input_jsonl: Path, output_dir: Path, *, allow_no_roi: bool = False, box_width: int = 2
) -> dict[str, Any]:
    """Render teacher/student PNGs and write manifests with an audit record."""
    if type(box_width) is not int or box_width < 1:
        raise ValueError("box_width must be a positive integer")
    input_jsonl = input_jsonl.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows = load_source_rows(input_jsonl, allow_no_roi=allow_no_roi)
    from PIL import Image, __version__ as pillow_version

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    image_dir = output_dir / "images"
    image_dir.mkdir()
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        split = row.raw["split"]
        stem = f"{split}-{index:06d}"
        student_path = image_dir / f"{stem}.student.png"
        teacher_path = image_dir / f"{stem}.teacher.png"
        with Image.open(row.image_path) as source:
            view = build_teacher_view(
                source, row.bbox_xyxy, allow_no_roi=allow_no_roi, box_width=box_width,
            )
        view.student_image.save(student_path)
        view.teacher_image.save(teacher_path)
        manifest_row: dict[str, Any] = {
            "id": row.raw["id"], "split": split,
            "student_image": f"images/{student_path.name}",
            "teacher_image": f"images/{teacher_path.name}",
            "question": row.raw["question"],
            "accepted_answers": list(row.raw["accepted_answers"]),
        }
        if row.bbox_xyxy is not None:
            manifest_row["student_image_hint"] = STUDENT_FOCUS_HINT
        if row.teacher_evidence is not None:
            manifest_row["teacher_evidence"] = row.teacher_evidence
        ProxyExample.from_dict(manifest_row)
        manifests[split].append(manifest_row)
        evidence_hash = None
        if row.teacher_evidence is not None:
            serialized = json.dumps(row.teacher_evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            evidence_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        examples.append({
            "id": row.raw["id"], "source_line_number": row.line_number,
            "source_image": str(row.image_path), "source_image_sha256": row.image_sha256,
            "student_image_sha256": _sha256(student_path),
            "teacher_image_sha256": _sha256(teacher_path),
            "teacher_view": view.metadata,
            "teacher_evidence_sha256": evidence_hash,
        })
    for split, split_rows in manifests.items():
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in split_rows),
            encoding="utf-8",
        )
    provenance = {
        "builder": "sure_vl_privileged_proxy_data_v1",
        "input_jsonl": str(input_jsonl), "input_sha256": _sha256(input_jsonl),
        "vision_opd_paper": "https://arxiv.org/html/2605.18740v1",
        "vision_opd_repository": "https://github.com/VisionOPD/Vision-OPD",
        "vision_opd_reference_commit": VISION_OPD_REFERENCE_COMMIT,
        "implementation_note": (
            "Vision-OPD releases prebuilt teacher crops; Sure-VL uses an explicit source bbox, "
            "Pillow LANCZOS 2x resize, and a configurable red outline. Interpolation and outline "
            "width are Sure-VL choices, not parameters reported by the reference repository."
        ),
        "pillow_version": pillow_version,
        "no_roi_policy": "retain source image for both views" if allow_no_roi else "reject missing bbox",
        "box_width": box_width,
        "sample_count": {split: len(split_rows) for split, split_rows in manifests.items()},
        "roi_count": sum(row.bbox_xyxy is not None for row in rows),
        "no_roi_count": sum(row.bbox_xyxy is None for row in rows),
        "manifest_sha256": {split: _sha256(output_dir / f"{split}.jsonl") for split in manifests},
        "examples": examples,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-no-roi", action="store_true")
    parser.add_argument("--box-width", type=int, default=2)
    args = parser.parse_args(argv)
    provenance = build_dataset(
        args.input_jsonl, args.output_dir, allow_no_roi=args.allow_no_roi, box_width=args.box_width,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "sample_count": provenance["sample_count"],
        "roi_count": provenance["roi_count"],
        "no_roi_count": provenance["no_roi_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
