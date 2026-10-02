#!/usr/bin/env python3
"""Build paired QA manifests for internal visual proxy training.

Uses the already audited 16/8 official QA selection. Visual fact annotations
are not read or emitted, and no binary V label is constructed. Source files,
answers, selected rows, and image degradation remain frozen and hashed.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from build_vlcalib_pilot import DATASET_URL, DEFAULT_ANNOTATIONS, sha256_bytes, sha256_file, source_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--validation-parquet", type=Path)
    args = parser.parse_args()

    import pyarrow.parquet as parquet
    from PIL import Image, ImageOps, __version__ as pillow_version

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    manifests = {"train": [], "validation": []}
    sources = {}
    records = []
    source_image_hashes = set()
    selected_keys = set()
    for official_split, local_path in (("train", args.train_parquet), ("val", args.validation_parquet)):
        expected = selection["source_files"][official_split]
        path = source_file(official_split, expected, local_path, root / "_source")
        table = parquet.ParquetFile(path).read(columns=["problem", "answer", "images"])
        if len(table) != expected["rows"]:
            raise ValueError(f"{official_split} source row count mismatch")
        sources[official_split] = {"path": str(path.resolve()), "sha256": expected["sha256"]}
        split = "train" if official_split == "train" else "validation"
        for selected in selection["annotations"]:
            if selected["official_split"] != official_split:
                continue
            index = selected["row_index"]
            if type(index) is not int or not 0 <= index < len(table):
                raise ValueError("invalid selected row index")
            key = (official_split, index)
            if key in selected_keys:
                raise ValueError("duplicate selected row")
            selected_keys.add(key)
            row = table.slice(index, 1).to_pylist()[0]
            if not isinstance(row["images"], list) or len(row["images"]) != 1:
                raise ValueError(f"{key} must have one image")
            image = row["images"][0]
            image_hash = sha256_bytes(image["bytes"])
            for field, actual in (
                ("raw_problem", row["problem"]), ("raw_answer", row["answer"]),
                ("source_image_path", image["path"]), ("source_image_sha256", image_hash),
            ):
                if selected[field] != actual:
                    raise ValueError(f"{key} frozen {field} differs")
            if image_hash in source_image_hashes:
                raise ValueError("selected source images overlap")
            source_image_hashes.add(image_hash)
            with Image.open(BytesIO(image["bytes"])) as original:
                teacher = ImageOps.exif_transpose(original).convert("RGB")
            width, height = teacher.size
            reduced = teacher.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.LANCZOS)
            student = reduced.resize((width, height), Image.Resampling.BICUBIC)
            if student.tobytes() == teacher.tobytes():
                raise ValueError(f"{key} degradation has no effect")
            stem = f"{split}-{index:05d}"
            student_path = root / "images" / f"{stem}.student.png"
            teacher_path = root / "images" / f"{stem}.teacher.png"
            student.save(student_path)
            teacher.save(teacher_path)
            item = {
                "id": f"vlcalib-proxy-{stem}", "split": split,
                "student_image": f"images/{student_path.name}",
                "teacher_image": f"images/{teacher_path.name}",
                "question": row["problem"].replace("<image>", "").strip(),
                "accepted_answers": [row["answer"]],
            }
            manifests[split].append(item)
            records.append({
                "id": item["id"], "official_split": official_split, "row_index": index,
                "source_image_sha256": image_hash,
                "student_image_sha256": sha256_file(student_path),
                "teacher_image_sha256": sha256_file(teacher_path),
                "answer_label": row["answer"],
            })
    for split, rows in manifests.items():
        if not rows:
            raise ValueError(f"empty {split} selection")
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    record = {
        "method": "internal_visual_proxy", "source_dataset": DATASET_URL,
        "source_revision": selection["source_revision"], "source_files": sources,
        "selection_sha256": sha256_file(args.selection),
        "supervision": "Official QA answers only; no binary V or visual-fact target is emitted.",
        "restriction": "Half-resolution Lanczos downsample then Bicubic upsample; full-resolution RGB teacher.",
        "pillow_version": pillow_version,
        "sample_count": {split: len(rows) for split, rows in manifests.items()},
        "manifests": {split: sha256_file(root / f"{split}.jsonl") for split in manifests},
        "examples": records,
    }
    (root / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(root), "sample_count": record["sample_count"]}, indent=2))


if __name__ == "__main__":
    main()
