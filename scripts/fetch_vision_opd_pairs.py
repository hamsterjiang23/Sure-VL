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
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

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
# Pinned LFS object sizes and SHA-256 values for the complete official archive.
# The Student gzip/tar stream is split into six consecutive byte parts; it must
# be reassembled as one stream before interpreting tar members.
STUDENT_PARTS = (
    ("images/images.tar.gz00", 5_368_709_120, "e72239bb03d393886e84aa2758eabcad387b03e86a7d7ce7238178f5832f52d0"),
    ("images/images.tar.gz01", 5_368_709_120, "3e0872e7ae37cc4b94019ec0f23ae274ec5c1f7dbebfa087e6219e0da9301979"),
    ("images/images.tar.gz02", 5_368_709_120, "b0bd79677a7439e87745f39716da57b85796dc241974032831ef67772f96be1a"),
    ("images/images.tar.gz03", 5_368_709_120, "4b230e8f814b22117126064e8eb6d4d7ba6f639860c6982a387ce8bb814e99ee"),
    ("images/images.tar.gz04", 5_368_709_120, "d7872e12c1ab49ca4fd0773f1e6aa621a7c3ef8355ca7e72fc57e8ce6292cfc6"),
    ("images/images.tar.gz05", 1_496_961_022, "44844e97bc43ee0bf8546ae50ca5b709decb22c144bfda410f3ddfe8c546060a"),
)
FULL_ARCHIVES = (*STUDENT_PARTS, (TEACHER_ARCHIVE, TEACHER_ARCHIVE_SIZE, TEACHER_ARCHIVE_SHA256))
FULL_VALIDATION_COUNT = 256
FULL_TRAIN_COUNT = 5_985
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
    """Resume one pinned Range cache under a per-archive single-writer lock."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{label}: archive cache already has a writer") from error
        return _download_prefix_unlocked(
            label=label, url=url, cache_path=cache_path,
            target_bytes=target_bytes, archive_size=archive_size,
            session=session, max_attempts=max_attempts,
        )


def _download_prefix_unlocked(
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
        if current == target_bytes:
            _log(label, f"verified cached prefix: {current:,} bytes, SHA256 {record['prefix_sha256']}")
            return record
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


class _ConcatenatedFiles:
    """Expose split archive bytes as one forward-only stream without a 28 GB copy."""

    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths:
            raise ValueError("at least one archive path is required")
        self.paths = tuple(paths)
        self.position = 0
        self.next_index = 0
        self.current: Any | None = None

    def __enter__(self) -> "_ConcatenatedFiles":
        return self

    def __exit__(self, *_: Any) -> None:
        if self.current is not None:
            self.current.close()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise ValueError("split archive reader requires bounded reads")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            if self.current is None:
                if self.next_index == len(self.paths):
                    break
                self.current = self.paths[self.next_index].open("rb")
                self.next_index += 1
            data = self.current.read(remaining)
            if data:
                chunks.append(data)
                count = len(data)
                self.position += count
                remaining -= count
            else:
                self.current.close()
                self.current = None
        return b"".join(chunks)

    def tell(self) -> int:
        return self.position


def extract_complete_pngs(
    *, label: str, prefix_path: Path | Sequence[Path], output_dir: Path, allowed_filenames: set[str],
    complete_archive: bool = False, seen_filenames: set[str] | None = None,
) -> dict[str, int]:
    """Stream gzip/tar prefix and install only complete whitelisted regular PNGs."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("image verification requires Pillow; use `uv run --extra train`") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = skipped_existing = ignored = 0
    truncated = False
    paths = (prefix_path,) if isinstance(prefix_path, Path) else tuple(prefix_path)
    with _ConcatenatedFiles(paths) as source:
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
                    if seen_filenames is not None:
                        seen_filenames.add(name_path.name)
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


def _full_cache_path(source_root: Path, relative_archive: str) -> Path:
    return source_root / ".range-cache" / (PurePosixPath(relative_archive).name + ".prefix")


def download_full_archives(source_root: Path, *, max_workers: int = 3) -> dict[str, dict[str, Any]]:
    """Resume and verify every pinned Student/Teacher LFS object, without extraction."""
    source_root = source_root.expanduser().resolve()
    load_official_rows(source_root)
    if type(max_workers) is not int or not 1 <= max_workers <= len(FULL_ARCHIVES):
        raise ValueError("max_workers must be between one and the number of archives")

    def one_archive(item: tuple[str, int, str]) -> tuple[str, dict[str, Any]]:
        relative, size, expected_sha256 = item
        result = download_prefix(
            label=relative, url=f"{MIRROR_ROOT}/{relative}",
            cache_path=_full_cache_path(source_root, relative),
            target_bytes=size, archive_size=size,
        )
        if result["prefix_sha256"] != expected_sha256:
            raise ValueError(f"{relative}: full LFS SHA256 differs from pinned revision")
        return relative, result

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(one_archive, item) for item in FULL_ARCHIVES]
        for future in as_completed(futures):
            relative, record = future.result()
            results[relative] = record
            _log("full", f"verified {len(results)}/{len(FULL_ARCHIVES)} archives: {relative}")
    return {relative: results[relative] for relative, _, _ in FULL_ARCHIVES}


def fetch_full_and_freeze(source_root: Path, output_dir: Path) -> dict[str, Any]:
    """Verify all 6,241 official pairs and freeze a 5,985/256 scene-disjoint split."""
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows, metadata_sha = load_official_rows(source_root)
    if len(rows) != FULL_TRAIN_COUNT + FULL_VALIDATION_COUNT:
        raise ValueError("the pinned official metadata does not contain 6,241 rows")
    archive_records = download_full_archives(source_root)
    student_names = {PurePosixPath(row.student_rel).name for row in rows}
    teacher_names = {PurePosixPath(row.teacher_rel).name for row in rows}
    student_extraction = extract_complete_pngs(
        label="student-full",
        prefix_path=tuple(_full_cache_path(source_root, part) for part, _, _ in STUDENT_PARTS),
        output_dir=source_root / "images", allowed_filenames=student_names,
        complete_archive=True,
    )
    teacher_extraction = extract_complete_pngs(
        label="teacher-full", prefix_path=_full_cache_path(source_root, TEACHER_ARCHIVE),
        output_dir=source_root / "teacher_images", allowed_filenames=teacher_names,
        complete_archive=True,
    )
    available, inventory = inspect_local_pairs(rows)
    _log("full", f"verified extracted pair inventory: {inventory}")
    if (len(available) != len(rows) or inventory["paired_original_groups"] != len(rows)
            or any(inventory[key] for key in ("student_only", "teacher_only", "neither"))):
        raise ValueError("full official paired-image inventory is incomplete or scene groups collide")
    provenance = build_dataset(
        source_root, output_dir, train_count=FULL_TRAIN_COUNT,
        validation_count=FULL_VALIDATION_COUNT,
        seed="sure-vl-vision-opd-6k-full-v1",
    )
    selected_indices = [item["official_row_index"] for item in provenance["examples"]]
    if len(selected_indices) != len(rows) or set(selected_indices) != set(range(len(rows))):
        raise ValueError("full split did not cover each official metadata row exactly once")
    train_groups = {item["original_scene_group"] for item in provenance["examples"] if item["split"] == "train"}
    validation_groups = {item["original_scene_group"] for item in provenance["examples"] if item["split"] == "validation"}
    train_hashes = {item[key] for item in provenance["examples"] if item["split"] == "train"
                    for key in ("student_image_sha256", "teacher_image_sha256")}
    validation_hashes = {item[key] for item in provenance["examples"] if item["split"] == "validation"
                         for key in ("student_image_sha256", "teacher_image_sha256")}
    if train_groups & validation_groups or train_hashes & validation_hashes:
        raise ValueError("full split has source-scene or actual-image SHA256 leakage")
    provenance["full_selection_audit"] = {
        "official_rows_covered_once": len(selected_indices),
        "train_original_groups": len(train_groups),
        "validation_original_groups": len(validation_groups),
        "cross_split_original_group_overlap": 0,
        "cross_split_actual_image_sha256_overlap": 0,
    }
    provenance["fetch"] = {
        "mode": "complete_pinned_lfs_archives",
        "mirror_root": MIRROR_ROOT,
        "metadata_sha256": metadata_sha,
        "total_archive_bytes": sum(size for _, size, _ in FULL_ARCHIVES),
        "archives": archive_records,
        "student_extraction": student_extraction,
        "teacher_extraction": teacher_extraction,
        "original_images_downloaded": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    _log("full", f"frozen official train={FULL_TRAIN_COUNT} validation={FULL_VALIDATION_COUNT}: {output_dir}")
    return provenance


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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--download-full-only", action="store_true",
                        help="resume and SHA-verify all seven pinned official archive parts")
    parser.add_argument("--download-teacher-only", action="store_true",
                        help="finish the pinned Teacher archive while Student parts download")
    parser.add_argument("--freeze-full", action="store_true",
                        help="extract verified full archives and freeze 5985/256 paired-image manifests")
    parser.add_argument("--download-workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.download_teacher_only:
        if args.download_full_only or args.freeze_full:
            parser.error("--download-teacher-only cannot be combined with other full-data modes")
        result = download_prefix(
            label="teacher-full", url=f"{MIRROR_ROOT}/{TEACHER_ARCHIVE}",
            cache_path=_full_cache_path(args.source_root.expanduser().resolve(), TEACHER_ARCHIVE),
            target_bytes=TEACHER_ARCHIVE_SIZE, archive_size=TEACHER_ARCHIVE_SIZE,
        )
        if result["prefix_sha256"] != TEACHER_ARCHIVE_SHA256:
            raise ValueError("complete Teacher LFS SHA256 differs from pinned revision")
        print(json.dumps({"archive": TEACHER_ARCHIVE, "record": result}, indent=2))
        return 0
    if args.download_full_only:
        if args.freeze_full:
            parser.error("--download-full-only and --freeze-full are mutually exclusive")
        records = download_full_archives(args.source_root, max_workers=args.download_workers)
        print(json.dumps({"verified_archive_count": len(records), "archives": records}, indent=2))
        return 0
    if args.freeze_full:
        if args.output_dir is None:
            parser.error("--output-dir is required for --freeze-full")
        result = fetch_full_and_freeze(args.source_root, args.output_dir)
        print(json.dumps({
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "selection_counts": result["selection_counts"],
            "manifest_sha256": result["manifest_sha256"],
            "paired_images": result["inventory"]["paired_images"],
            "total_archive_bytes": result["fetch"]["total_archive_bytes"],
        }, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --download-full-only is used")
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
