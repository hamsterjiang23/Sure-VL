"""Build a tiny, audited Sure-VL pilot from official VL-Calibration splits.

Run with ``uv run --no-project --with pyarrow --with pillow python
scripts/build_vlcalib_pilot.py --output-dir /data/LHJ/Sure-VL/data/vlcalib_pilot_v1``.
The official Parquet files stay outside Git. Each selected answer and a separate
human-checked visual fact are frozen in configs/vlcalib_pilot_v1.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from io import BytesIO
from pathlib import Path


DEFAULT_ANNOTATIONS = Path(__file__).resolve().parents[1] / "configs" / "vlcalib_pilot_v1.json"
DATASET_URL = "https://modelscope.cn/datasets/xiaowenyi/VL-Calibration-12K"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(split: str, expected: dict, local_path: Path | None, source_dir: Path) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    path = local_path or source_dir / expected["filename"]
    if not path.is_file():
        if local_path is not None:
            raise FileNotFoundError(path)
        url = f"{DATASET_URL}/resolve/master/{expected['filename']}"
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        partial.replace(path)
    actual = sha256_file(path)
    if actual != expected["sha256"]:
        raise ValueError(f"{split} Parquet SHA256 mismatch: {actual}")
    return path


def validate_and_select(table, split: str, annotations: list[dict]) -> list[dict]:
    selected = []
    for annotation in annotations:
        if annotation["official_split"] != split:
            continue
        index = annotation["row_index"]
        if type(index) is not int or not 0 <= index < len(table):
            raise ValueError(f"invalid {split} row index: {index}")
        row = table.slice(index, 1).to_pylist()[0]
        images = row["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"{split}:{index} must have exactly one image")
        image = images[0]
        checks = {
            "raw_problem": row["problem"],
            "raw_answer": row["answer"],
            "source_image_path": image["path"],
            "source_image_sha256": sha256_bytes(image["bytes"]),
        }
        for key, actual in checks.items():
            if annotation[key] != actual:
                raise ValueError(f"{split}:{index} frozen {key} mismatch: {actual!r}")
        facts = annotation["required_visual_facts"]
        if not isinstance(facts, dict) or len(facts) != 1:
            raise ValueError(f"{split}:{index} needs one independently checked fact")
        fact = next(iter(facts.values()))
        canonical = fact["canonical"] if isinstance(fact, dict) else fact
        if canonical.strip().casefold() == row["answer"].strip().casefold():
            raise ValueError(f"{split}:{index} fact duplicates QA answer")
        selected.append({"annotation": annotation, "image_bytes": image["bytes"]})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--validation-parquet", type=Path)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as parquet
        from PIL import Image, ImageOps, __version__ as pillow_version
    except ImportError as error:
        raise SystemExit("Run with uv: uv run --no-project --with pyarrow --with pillow python scripts/build_vlcalib_pilot.py ...") from error

    config = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations = config["annotations"]
    if len({(item["official_split"], item["row_index"]) for item in annotations}) != len(annotations):
        raise ValueError("duplicate official split/row index")

    output_dir = args.output_dir.resolve()
    sources = {}
    selected = []
    for official_split, local_path in (("train", args.train_parquet), ("val", args.validation_parquet)):
        expected = config["source_files"][official_split]
        path = source_file(official_split, expected, local_path, output_dir / "_source")
        sources[official_split] = {"path": str(path), "sha256": expected["sha256"]}
        table = parquet.ParquetFile(path).read(columns=["problem", "answer", "images"])
        if len(table) != expected["rows"]:
            raise ValueError(f"{official_split} source row count mismatch: {len(table)}")
        selected.extend(validate_and_select(table, official_split, annotations))

    image_hashes = [sha256_bytes(item["image_bytes"]) for item in selected]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError("selected source images overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    manifests = {"train": [], "validation": []}
    provenance_rows = []
    for item in selected:
        annotation = item["annotation"]
        official_split = annotation["official_split"]
        split = "train" if official_split == "train" else "validation"
        index = annotation["row_index"]
        name = f"{split}-{index:05d}"
        with Image.open(BytesIO(item["image_bytes"])) as source_image:
            teacher = ImageOps.exif_transpose(source_image).convert("RGB")
        width, height = teacher.size
        reduced = teacher.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.LANCZOS)
        student = reduced.resize((width, height), Image.Resampling.BICUBIC)
        if student.tobytes() == teacher.tobytes():
            raise ValueError(f"{split}:{index} restriction had no pixel effect")
        teacher_name = f"{name}.teacher.png"
        student_name = f"{name}.student.png"
        teacher_path = image_dir / teacher_name
        student_path = image_dir / student_name
        teacher.save(teacher_path, format="PNG")
        student.save(student_path, format="PNG")
        manifest_row = {
            "id": f"vlcalib-{name}",
            "split": split,
            "student_image": f"images/{student_name}",
            "teacher_image": f"images/{teacher_name}",
            "question": annotation["raw_problem"].replace("<image>", "").strip(),
            "required_visual_facts": annotation["required_visual_facts"],
            "accepted_answers": [annotation["raw_answer"]],
        }
        manifests[split].append(manifest_row)
        provenance_rows.append({
            "id": manifest_row["id"],
            "official_split": official_split,
            "row_index": index,
            "raw_problem": annotation["raw_problem"],
            "raw_answer": annotation["raw_answer"],
            "source_image_path": annotation["source_image_path"],
            "source_image_sha256": annotation["source_image_sha256"],
            "teacher_image_sha256": sha256_file(teacher_path),
            "student_image_sha256": sha256_file(student_path),
            "fact_audit_note": annotation["fact_audit_note"],
        })

    for split, rows in manifests.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    provenance = {
        "source_dataset": DATASET_URL,
        "source_revision": config["source_revision"],
        "source_files": sources,
        "annotation_file": str(args.annotations.resolve()),
        "annotation_sha256": sha256_file(args.annotations),
        "restriction": "RGB source, half width and height with Lanczos downsampling, then Bicubic upsampling to original size; teacher is full-resolution RGB PNG",
        "pillow_version": pillow_version,
        "sample_count": {key: len(value) for key, value in manifests.items()},
        "examples": provenance_rows,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "sample_count": provenance["sample_count"], "annotation_sha256": provenance["annotation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
