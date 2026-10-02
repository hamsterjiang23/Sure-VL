#!/usr/bin/env python3
"""Resume pinned Vision-OPD archive prefixes and extract safe official pairs.

Run as ``python -m scripts.fetch_vision_opd_pairs`` from the Sure-VL root.
Only the official full red-box Student PNG and cropped Teacher PNG archives
are fetched. Tar members are never passed to ``extractall``. The incomplete
final member of a compressed prefix is discarded, and every retained PNG is
verified before it is atomically installed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.build_vision_opd_proxy_data import (
    HF_REPO, HF_REVISION, METADATA_SHA256, build_dataset, inspect_local_pairs, load_official_rows,
)


MIB = 1024 * 1024
STUDENT_ARCHIVE_SIZE = 5_368_709_120  # First of six split gzip parts.
TEACHER_ARCHIVE_SIZE = 2_942_713_517
STUDENT_ARCHIVE_SHA256 = "e72239bb03d393886e84aa2758eabcad387b03e86a7d7ce7238178f5832f52d0"
TEACHER_ARCHIVE_SHA256 = "f2fb6541e8e1ea4e33114aff9d511c5e8ce764c0972798151c8d5b1b5b91883e"
STUDENT_ARCHIVE = "images/images.tar.gz00"
TEACHER_ARCHIVE = "teacher_images/teacher_images.tar.gz"
MIRROR_ROOT = f"https://hf-mirror.com/datasets/{HF_REPO}/resolve/{HF_REVISION}"
PROGRESS_BYTES = 64 * MIB
MAX_PNG_BYTES = 128 * MIB
RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _log(label: str, message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}: {message}", flush=True)


def download_prefix(
    *,
    label: str,
    url: str,
    cache_path: Path,
    target_bytes: int,
    archive_size: int,
    session: Any = None,
    max_attempts: int = 12,
) -> dict[str, Any]:
    """Append HTTP Range bytes to a pinned cache; never fetch archive tails."""
    if type(target_bytes) is not int or not (1 <= target_bytes <= archive_size):
        raise ValueError("target_bytes must be inside the official archive")
    if not url.startswith(MIRROR_ROOT + "/"):
        raise ValueError("archive URL must use the pinned HF mirror revision")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = cache_path.with_suffix(cache_path.suffix + ".json")
    identity = {
        "url": url, "hf_revision": HF_REVISION, "archive_size": archive_size,
        "metadata_sha256": METADATA_SHA256,
    }
    if sidecar.exists():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{label}: existing range cache has a different source identity")
    elif cache_path.exists():
        raise ValueError(f"{label}: range cache lacks its source identity sidecar")
    else:
        record = {**identity, "downloaded_bytes": 0}
        _write_json_atomic(sidecar, record)
    current = cache_path.stat().st_size if cache_path.exists() else 0
    if current > archive_size:
        raise ValueError(f"{label}: range cache exceeds official archive size")
    if current and record.get("prefix_sha256") and record.get("downloaded_bytes") == current:
        if _sha256(cache_path) != record["prefix_sha256"]:
            raise ValueError(f"{label}: range cache SHA256 changed")
    try:
        import requests
    except ImportError as error:
        raise RuntimeError("HTTP prefix fetch requires requests; use `uv run --extra train`") from error
    if session is None:
        session = requests.Session()
    failures = 0
    next_progress = ((current // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
    started = time.monotonic()
    last_progress_bytes = current
    _log(label, f"resume {current:,}/{target_bytes:,} bytes from {url}")
    while current < target_bytes:
        before = current
        headers = {"Range": f"bytes={current}-{target_bytes - 1}", "Accept-Encoding": "identity"}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(25, 90)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206 Range response, got {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                match = RANGE_PATTERN.fullmatch(content_range)
                if (match is None or int(match.group(1)) != current
                        or int(match.group(3)) != archive_size
                        or int(match.group(2)) < current):
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                etag = response.headers.get("ETag")
                old_etag = record.get("etag") if sidecar.exists() else None
                if old_etag and etag and old_etag != etag:
                    raise RuntimeError(f"ETag changed: {old_etag!r} -> {etag!r}")
                if etag and not old_etag:
                    record = {**identity, "downloaded_bytes": current, "etag": etag}
                    _write_json_atomic(sidecar, record)
                with cache_path.open("ab") as sink:
                    for chunk in response.iter_content(chunk_size=MIB):
                        if not chunk:
                            continue
                        write = chunk[:target_bytes - current]
                        sink.write(write)
                        current += len(write)
                        if current >= next_progress or time.monotonic() - started >= 60:
                            sink.flush()
                            speed = (current - last_progress_bytes) / max(time.monotonic() - started, 1e-9) / MIB
                            _log(label, f"{current / MIB:.1f}/{target_bytes / MIB:.1f} MiB, {speed:.1f} MiB/s")
                            next_progress = ((current // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
                            started = time.monotonic()
                            last_progress_bytes = current
                        if current == target_bytes:
                            break
            if current == before:
                raise RuntimeError("Range response delivered no bytes")
            failures = 0
        except (OSError, RuntimeError, requests.RequestException) as error:
            failures += 1
            _write_json_atomic(sidecar, {**identity, "downloaded_bytes": current,
                                         **({"etag": record["etag"]} if record.get("etag") else {})})
            _log(label, f"retry {failures}/{max_attempts} at byte {current:,}: {error}")
            if failures >= max_attempts:
                raise
            time.sleep(min(2 * failures, 10))
    digest = _sha256(cache_path)
    result = {**identity, "downloaded_bytes": current, "prefix_sha256": digest}
    if record.get("etag"):
        result["etag"] = record["etag"]
    _write_json_atomic(sidecar, result)
    _log(label, f"prefix ready: {current:,} bytes, SHA256 {digest}")
    return result


def extract_complete_pngs(
    *, label: str, prefix_path: Path, output_dir: Path, allowed_filenames: set[str],
    complete_archive: bool = False,
) -> dict[str, int]:
    """Stream gzip/tar prefix and install only complete whitelisted regular PNGs."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("image verification requires Pillow; use `uv run --extra train`") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = skipped_existing = ignored = 0
    truncated = False
    with prefix_path.open("rb") as source:
        try:
            with tarfile.open(fileobj=source, mode="r|gz") as archive:
                for member in archive:
                    name_path = PurePosixPath(member.name)
                    if (name_path.is_absolute() or len(name_path.parts) != 1
                            or not member.isfile() or name_path.suffix.lower() != ".png"
                            or name_path.name not in allowed_filenames):
                        ignored += 1
                        continue
                    if member.size < 1 or member.size > MAX_PNG_BYTES:
                        raise ValueError(f"{label}: unreasonable official PNG size: {member.name} {member.size}")
                    fileobj = archive.extractfile(member)
                    if fileobj is None:
                        raise ValueError(f"{label}: cannot read regular PNG: {member.name}")
                    try:
                        image_bytes = fileobj.read(member.size)
                    except (EOFError, OSError, tarfile.TarError):
                        truncated = True
                        break
                    if len(image_bytes) != member.size:
                        truncated = True
                        break
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            if image.format != "PNG":
                                raise ValueError(f"{label}: archive member is not PNG: {member.name}")
                            image.verify()
                    except (OSError, ValueError) as error:
                        raise ValueError(f"{label}: invalid complete PNG {member.name}: {error}") from error
                    destination = output_dir / name_path.name
                    if (not destination.is_symlink() and destination.is_file()
                            and destination.stat().st_size == len(image_bytes)
                            and _sha256(destination) == hashlib.sha256(image_bytes).hexdigest()):
                        skipped_existing += 1
                        continue
                    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".vision-opd-", delete=False) as temporary:
                        temporary.write(image_bytes)
                        temporary_name = temporary.name
                    os.replace(temporary_name, destination)
                    extracted += 1
        except (EOFError, OSError, tarfile.ReadError):
            truncated = True
    if truncated and complete_archive:
        raise ValueError(f"{label}: complete archive ended with truncated or invalid gzip/tar data")
    result = {
        "extracted": extracted, "skipped_existing": skipped_existing,
        "ignored": ignored, "truncated_prefix": int(truncated),
    }
    _log(label, f"safe extraction {result}")
    return result


def fetch_and_freeze(
    *, source_root: Path, output_dir: Path, train_count: int = 32,
    validation_count: int = 8,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows, metadata_sha = load_official_rows(source_root)
    student_names = {PurePosixPath(row.student_rel).name for row in rows}
    teacher_names = {PurePosixPath(row.teacher_rel).name for row in rows}
    cache_dir = source_root / ".range-cache"
    student_cache = cache_dir / "images.tar.gz00.prefix"
    teacher_cache = cache_dir / "teacher_images.tar.gz.prefix"
    stage_targets = [
        (512 * MIB, 1536 * MIB),
        (512 * MIB, TEACHER_ARCHIVE_SIZE),
        (1024 * MIB, TEACHER_ARCHIVE_SIZE),
    ]
    for stage_index, (student_target, teacher_target) in enumerate(stage_targets, 1):
        _log("stage", f"{stage_index}/{len(stage_targets)} targets: student={student_target:,}, teacher={teacher_target:,}")
        for label, rel, cache, target, total, expected_full_sha, names, directory in (
            ("student", STUDENT_ARCHIVE, student_cache, student_target, STUDENT_ARCHIVE_SIZE,
             STUDENT_ARCHIVE_SHA256, student_names, source_root / "images"),
            ("teacher", TEACHER_ARCHIVE, teacher_cache, teacher_target, TEACHER_ARCHIVE_SIZE,
             TEACHER_ARCHIVE_SHA256, teacher_names, source_root / "teacher_images"),
        ):
            cache_record = download_prefix(label=label, url=f"{MIRROR_ROOT}/{rel}",
                                           cache_path=cache, target_bytes=target, archive_size=total)
            if target == total and cache_record["prefix_sha256"] != expected_full_sha:
                raise ValueError(f"{label}: complete archive differs from pinned HF LFS SHA256")
            extract_complete_pngs(label=label, prefix_path=cache, output_dir=directory,
                                  allowed_filenames=names, complete_archive=(target == total))
        available, inventory = inspect_local_pairs(rows)
        _log("pairs", f"inventory after stage {stage_index}: {inventory}")
        if inventory["paired_original_groups"] >= train_count + validation_count:
            provenance = build_dataset(
                source_root, output_dir, train_count=train_count,
                validation_count=validation_count,
            )
            provenance["fetch"] = {
                "mirror_root": MIRROR_ROOT, "metadata_sha256": metadata_sha,
                "stage_index": stage_index,
                "student_prefix": json.loads(student_cache.with_suffix(student_cache.suffix + ".json").read_text()),
                "teacher_prefix": json.loads(teacher_cache.with_suffix(teacher_cache.suffix + ".json").read_text()),
            }
            (output_dir / "provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            _log("freeze", f"selected train={train_count}, validation={validation_count}; {output_dir}")
            return provenance
    raise ValueError(f"fewer than {train_count + validation_count} pairs after all bounded prefixes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    args = parser.parse_args(argv)
    result = fetch_and_freeze(
        source_root=args.source_root, output_dir=args.output_dir,
        train_count=args.train_count, validation_count=args.validation_count,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "selection_counts": result["selection_counts"],
        "manifest_sha256": result["manifest_sha256"],
        "fetch_stage": result["fetch"]["stage_index"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
