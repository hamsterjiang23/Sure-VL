"""Official Vision-OPD metadata and paired-image adapter checks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_vision_opd_proxy_data import (
    REDBOX_HINT, build_dataset, inspect_local_pairs, load_official_rows,
)

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow is an optional training dependency")
class OfficialVisionOPDBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "official"
        (self.source / "images").mkdir(parents=True)
        (self.source / "teacher_images").mkdir()
        self.rows = [self._row(i) for i in range(4)]
        self._write_rows()

    @staticmethod
    def _row(index: int) -> dict:
        question = (
            f"What color is object {index}?\n\n"
            "A. blue\nB. red\nC. green\nD. yellow\n\n"
            "Answer with the option's letter from the given choices."
        )
        return {
            "images": [f"images/scene-{index}.png"],
            "teacher_images": [f"teacher_images/{index:06d}_scene-{index}.png"],
            "original_images": [f"original_images/scene-{index}.jpg"],
            "bbox": [1, 2, 4, 5],
            "problem": f"<image>\nWhat color is object {index}?\n\n{REDBOX_HINT}\n\n"
                       "A. blue\nB. red\nC. green\nD. yellow\n\n"
                       "Answer with the option's letter from the given choices.",
            "answer": "B", "extra_info": {"answer": "B", "question": question},
        }

    def _write_rows(self) -> str:
        metadata = self.source / "train.jsonl"
        metadata.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        return hashlib.sha256(metadata.read_bytes()).hexdigest()

    def _make_pair(self, index: int, *, student_color: tuple[int, int, int] | None = None) -> None:
        row = self.rows[index]
        color = student_color or (20 + index * 30, 50, 100)
        Image.new("RGB", (8, 8), color).save(self.source / row["images"][0])
        Image.new("RGB", (4, 4), (100, 20 + index * 30, 50)).save(
            self.source / row["teacher_images"][0],
        )

    def _load(self):
        return load_official_rows(
            self.source, expected_metadata_sha256=self._write_rows(),
            expected_row_count=len(self.rows),
        )

    def _build(self, output: Path, **kwargs):
        return build_dataset(
            self.source, output, expected_metadata_sha256=self._write_rows(),
            expected_row_count=len(self.rows), **kwargs,
        )

    def test_manifest_uses_official_pairs_and_separate_teacher_question(self) -> None:
        for index in range(4):
            self._make_pair(index)
        output = self.root / "selected"
        record = self._build(output, train_count=2, validation_count=1)
        self.assertEqual(record["selection_counts"], {"train": 2, "validation": 1})
        self.assertEqual(record["inventory"]["paired_images"], 4)
        manifest = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        self.assertEqual(len(manifest), 2)
        self.assertTrue(all(REDBOX_HINT in row["question"] for row in manifest))
        self.assertTrue(all("<image>" not in row["question"] for row in manifest))
        self.assertTrue(all(REDBOX_HINT not in row["teacher_question"] for row in manifest))
        self.assertTrue(all(row["accepted_answers"] == ["B"] for row in manifest))
        self.assertTrue(all("teacher_evidence" not in row and "student_image_hint" not in row
                            for row in manifest))
        self.assertTrue(all(Path(row["student_image"]).is_file()
                            and Path(row["teacher_image"]).is_file() for row in manifest))
        groups = {item["original_scene_group"] for item in record["examples"]}
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(item["student_image_sha256"]) == 64 for item in record["examples"]))
        self.assertTrue(all(item["original_image_sha256"] is None for item in record["examples"]))
        self.assertTrue(all(item["teacher_evidence_present"] is False for item in record["examples"]))
        repeat = self.root / "repeat-selection"
        self._build(repeat, train_count=2, validation_count=1)
        for split in ("train", "validation"):
            self.assertEqual(
                (output / f"{split}.jsonl").read_bytes(),
                (repeat / f"{split}.jsonl").read_bytes(),
            )
        sentinel = output / "keep.txt"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            self._build(output, train_count=2, validation_count=1)
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_missing_pair_inventory_and_group_count_fail_before_output(self) -> None:
        self._make_pair(0)
        Image.new("RGB", (8, 8), (3, 4, 5)).save(self.source / self.rows[1]["images"][0])
        rows, _ = self._load()
        available, inventory = inspect_local_pairs(rows)
        self.assertEqual(len(available), 1)
        self.assertEqual(inventory["student_only"], 1)
        output = self.root / "too-few"
        with self.assertRaisesRegex(ValueError, "need 2 paired original groups"):
            self._build(output, train_count=1, validation_count=1)
        self.assertFalse(output.exists())

    def test_cross_split_identical_image_bytes_are_rejected(self) -> None:
        self._make_pair(0, student_color=(12, 34, 56))
        self._make_pair(1, student_color=(12, 34, 56))
        output = self.root / "collision"
        with self.assertRaisesRegex(ValueError, "overlap in actual image SHA256"):
            self._build(output, train_count=1, validation_count=1)
        self.assertFalse(output.exists())

    def test_answer_question_and_revision_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            load_official_rows(self.source)
        self.rows[0]["extra_info"]["answer"] = "A"
        with self.assertRaisesRegex(ValueError, "matching official A-D"):
            self._load()
        self.rows[0]["extra_info"]["answer"] = "B"
        self.rows[0]["problem"] = self.rows[0]["problem"].replace("<image>", "<image><image>")
        with self.assertRaisesRegex(ValueError, "exactly one <image>"):
            self._load()


if __name__ == "__main__":
    unittest.main()
