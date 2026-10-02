#!/usr/bin/env python3
"""Freeze a real official Vision-OPD subset from an append-only Student prefix.

Run from the Sure-VL root as
``python -m scripts.freeze_vision_opd_prefix_subset --source-root ... --output-root ...``.
Like the archive fetcher, this module uses package imports and is not a
standalone ``python scripts/...py`` entry point.

The six-part full download may continue in parallel. This command captures a
fixed byte count from part 00, records its SHA-256, then uses only complete PNG
members within that snapshot. The Teacher archive must already be complete and
match its pinned LFS SHA-256. The Student *full* LFS SHA is not asserted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.build_vision_opd_proxy_data import (
    HF_REVISION, METADATA_SHA256, build_dataset, inspect_local_pairs, load_official_rows,
)
from scripts.fetch_vision_opd_pairs import (
    MIB, MIRROR_ROOT, STUDENT_ARCHIVE, STUDENT_ARCHIVE_SIZE,
    STUDENT_ARCHIVE_SHA256, TEACHER_ARCHIVE, TEACHER_ARCHIVE_SHA256,
    TEACHER_ARCHIVE_SIZE, _full_cache_path, _log, _sha256, _write_json_atomic,
    extract_complete_pngs,
)


def _snapshot_prefix(source_root: Path, snapshot_bytes: int | None) -> tuple[Path, dict[str, Any]]:
    source = _full_cache_path(source_root, STUDENT_ARCHIVE)
    source_record = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
    expected_identity = {
        "url": f"{MIRROR_ROOT}/{STUDENT_ARCHIVE}",
        "hf_revision": HF_REVISION,
        "archive_size": STUDENT_ARCHIVE_SIZE,
        "metadata_sha256": METADATA_SHA256,
    }
    if any(source_record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Student append-only cache has a different source identity")
    source_before = source.stat()
    if snapshot_bytes is None:
        snapshot_bytes = min(source_before.st_size, STUDENT_ARCHIVE_SIZE) // (64 * MIB) * (64 * MIB)
    if type(snapshot_bytes) is not int or not 512 * MIB <= snapshot_bytes <= min(source_before.st_size, STUDENT_ARCHIVE_SIZE):
        raise ValueError("snapshot byte count must be at least 512 MiB and present in Student part 00")
    directory = source.parent / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"images.tar.gz00.{snapshot_bytes}.prefix"
    sidecar = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or sidecar.exists():
        if not destination.is_file() or not sidecar.is_file():
            raise ValueError("incomplete existing Student snapshot")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if (record.get("snapshot_bytes") != snapshot_bytes
                or record.get("source_identity") != expected_identity
                or destination.stat().st_size != snapshot_bytes
                or _sha256(destination) != record.get("snapshot_sha256")):
            raise ValueError("existing Student snapshot identity or SHA256 differs")
        return destination, record
    digest = hashlib.sha256()
    with source.open("rb") as reader, tempfile.NamedTemporaryFile(
        dir=directory, prefix=".student-prefix-", delete=False,
    ) as writer:
        temporary = Path(writer.name)
        remaining = snapshot_bytes
        copied = 0
        while remaining:
            chunk = reader.read(min(4 * MIB, remaining))
            if not chunk:
                temporary.unlink(missing_ok=True)
                raise ValueError("Student cache shrank during fixed-byte snapshot")
            writer.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
            remaining -= len(chunk)
            if copied % (512 * MIB) == 0:
                _log("snapshot", f"copied {copied / MIB:.0f}/{snapshot_bytes / MIB:.0f} MiB")
    source_after = source.stat()
    if source_after.st_ino != source_before.st_ino or source_after.st_size < snapshot_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError("Student cache inode changed or shrank during snapshot")
    os.replace(temporary, destination)
    record = {
        "source_identity": expected_identity,
        "source_cache_path": str(source),
        "snapshot_path": str(destination),
        "snapshot_bytes": snapshot_bytes,
        "snapshot_sha256": digest.hexdigest(),
        "full_student_lfs_sha256_expected": STUDENT_ARCHIVE_SHA256,
        "full_student_lfs_sha256_verified": snapshot_bytes == STUDENT_ARCHIVE_SIZE
        and digest.hexdigest() == STUDENT_ARCHIVE_SHA256,
    }
    _write_json_atomic(sidecar, record)
    return destination, record


def freeze_prefix_subset(source_root: Path, output_root: Path, *, snapshot_bytes: int | None = None) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    rows, metadata_sha256 = load_official_rows(source_root)
    snapshot_path, snapshot = _snapshot_prefix(source_root, snapshot_bytes)
    teacher = _full_cache_path(source_root, TEACHER_ARCHIVE)
    teacher_record = json.loads(teacher.with_suffix(teacher.suffix + ".json").read_text(encoding="utf-8"))
    if (teacher_record.get("url") != f"{MIRROR_ROOT}/{TEACHER_ARCHIVE}"
            or teacher_record.get("hf_revision") != HF_REVISION
            or teacher_record.get("archive_size") != TEACHER_ARCHIVE_SIZE
            or teacher_record.get("metadata_sha256") != METADATA_SHA256
            or teacher.stat().st_size != TEACHER_ARCHIVE_SIZE
            or _sha256(teacher) != TEACHER_ARCHIVE_SHA256):
        raise ValueError("Teacher archive does not match the complete pinned LFS object")
    student_names = {PurePosixPath(row.student_rel).name for row in rows}
    teacher_names = {PurePosixPath(row.teacher_rel).name for row in rows}
    seen_student: set[str] = set()
    student_result = extract_complete_pngs(
        label="student-snapshot", prefix_path=snapshot_path,
        output_dir=source_root / "images", allowed_filenames=student_names,
        seen_filenames=seen_student,
    )
    teacher_result = extract_complete_pngs(
        label="teacher-full", prefix_path=teacher,
        output_dir=source_root / "teacher_images", allowed_filenames=teacher_names,
        complete_archive=True,
    )
    available, inventory = inspect_local_pairs(rows)
    candidate = [row for row in available if PurePosixPath(row.student_rel).name in seen_student]
    distinct_scenes = len({row.original_rel for row in candidate})
    _log("snapshot", f"complete pairs in fixed Student prefix: {len(candidate)}, scenes: {distinct_scenes}")
    if distinct_scenes >= 640:
        train_count, validation_count = 512, 128
    elif distinct_scenes >= 320:
        train_count, validation_count = 256, 64
    else:
        raise ValueError(f"need at least 320 real paired scenes in snapshot, found {distinct_scenes}")
    total = train_count + validation_count
    output_dir = output_root / f"vision_opd_proxy_{total}_v1"
    seed = f"sure-vl-vision-opd-prefix-{snapshot['snapshot_bytes']}-{snapshot['snapshot_sha256'][:16]}-v1"
    provenance = build_dataset(
        source_root, output_dir, train_count=train_count,
        validation_count=validation_count, seed=seed,
        allowed_student_names=seen_student,
    )
    if not all(PurePosixPath(item["student_source_path"]).name in seen_student for item in provenance["examples"]):
        raise ValueError("selected a Student image outside the fixed archive snapshot")
    provenance["fetch"] = {
        "mode": "fixed_student_part00_prefix_and_complete_teacher",
        "mirror_root": MIRROR_ROOT,
        "metadata_sha256": metadata_sha256,
        "student_snapshot": snapshot,
        "teacher_lfs_sha256": TEACHER_ARCHIVE_SHA256,
        "teacher_lfs_sha256_verified": True,
        "student_extraction": student_result,
        "teacher_extraction": teacher_result,
        "snapshot_student_png_count": len(seen_student),
        "candidate_paired_images": len(candidate),
        "candidate_original_scenes": distinct_scenes,
        "source_inventory_after_extraction": inventory,
        "original_images_downloaded": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    _log("snapshot", f"frozen official subset {train_count}/{validation_count}: {output_dir}")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--snapshot-bytes", type=int)
    args = parser.parse_args()
    result = freeze_prefix_subset(args.source_root, args.output_root, snapshot_bytes=args.snapshot_bytes)
    print(json.dumps({
        "selection_counts": result["selection_counts"],
        "manifest_sha256": result["manifest_sha256"],
        "student_snapshot": result["fetch"]["student_snapshot"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
