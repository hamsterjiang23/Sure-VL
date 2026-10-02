"""Explicit ROI construction and generic paired-view dataset audits."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_privileged_proxy_data import build_dataset
from sure_vl.teacher_view import STUDENT_FOCUS_HINT, build_teacher_view, validate_source_bbox

try:
    from PIL import Image
except ImportError:
    Image = None


class TeacherViewValidationTests(unittest.TestCase):
    def test_bbox_requires_integer_positive_inbounds_pixels(self) -> None:
        self.assertEqual(validate_source_bbox([2, 1, 6, 5], (10, 8)), (2, 1, 6, 5))
        for bbox in ([1, 1, 1, 5], [-1, 1, 5, 5], [1, 1, 11, 5], [0.1, 1, 5, 5], [True, 1, 5, 5]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                validate_source_bbox(bbox, (10, 8))


@unittest.skipIf(Image is None, "Pillow is an optional training dependency")
class TeacherViewBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.jsonl"
        for split, color in (("train", (20, 40, 60)), ("validation", (60, 40, 20))):
            image = Image.new("RGB", (12, 10), color)
            image.putpixel((5, 4), (0, 255, 0))
            image.save(self.root / f"{split}.png")

    def _row(self, split: str, *, bbox: list[int] | None = None, evidence: object = None) -> dict:
        row = {
            "id": f"sample-{split}", "split": split,
            "source_image": f"{split}.png", "question": "What color is the object?",
            "accepted_answers": ["green"],
        }
        if bbox is not None:
            row["evidence_bbox_xyxy"] = bbox
        if evidence is not None:
            row["teacher_evidence"] = evidence
        return row

    def _write(self, train: dict, validation: dict) -> None:
        self.source.write_text(
            json.dumps(train) + "\n" + json.dumps(validation) + "\n", encoding="utf-8",
        )

    def test_crop_is_only_bbox_2x_and_student_gets_full_red_box(self) -> None:
        image = Image.open(self.root / "train.png")
        view = build_teacher_view(image, [3, 2, 8, 7])
        self.assertEqual(view.student_image.size, (12, 10))
        self.assertEqual(view.teacher_image.size, (10, 10))
        self.assertEqual(view.student_image.getpixel((3, 2)), (255, 0, 0))
        self.assertEqual(view.student_image.getpixel((0, 0)), (20, 40, 60))
        red, green, blue = view.teacher_image.getpixel((5, 5))
        self.assertGreater(green, 200)
        self.assertLess(max(red, blue), 20)
        self.assertEqual(view.metadata["source_bbox_xyxy"], [3, 2, 8, 7])
        self.assertEqual(view.metadata["crop_resize_scale"], 2)

    def test_builder_writes_separate_views_manifest_and_provenance(self) -> None:
        self._write(
            self._row("train", bbox=[3, 2, 8, 7], evidence={"objects": [{"shape": "cube"}]}),
            self._row("validation", bbox=[3, 2, 8, 7]),
        )
        output = self.root / "new-paired-data"
        record = build_dataset(self.source, output)
        self.assertEqual(record["roi_count"], 2)
        self.assertEqual(record["no_roi_count"], 0)
        manifest = json.loads((output / "train.jsonl").read_text().strip())
        self.assertEqual(manifest["student_image_hint"], STUDENT_FOCUS_HINT)
        self.assertEqual(manifest["teacher_evidence"], {"objects": [{"shape": "cube"}]})
        self.assertNotIn("teacher_view", manifest)
        self.assertEqual(manifest["question"], "What color is the object?")
        with Image.open(output / manifest["student_image"]) as student, Image.open(output / manifest["teacher_image"]) as teacher:
            self.assertEqual(student.size, (12, 10))
            self.assertEqual(teacher.size, (10, 10))
        example = record["examples"][0]
        self.assertEqual(example["teacher_view"]["kind"], "vision_opd_evidence_crop_2x")
        self.assertEqual(example["source_image_sha256"], hashlib.sha256((self.root / "train.png").read_bytes()).hexdigest())
        self.assertEqual(len(example["teacher_evidence_sha256"]), 64)
        sentinel = output / "keep.txt"
        sentinel.write_text("must remain")
        with self.assertRaises(FileExistsError):
            build_dataset(self.source, output)
        self.assertEqual(sentinel.read_text(), "must remain")

    def test_no_roi_requires_explicit_flag_and_is_recorded_without_augmentation(self) -> None:
        self._write(self._row("train"), self._row("validation"))
        output = self.root / "no-roi"
        with self.assertRaisesRegex(ValueError, "allow-no-roi"):
            build_dataset(self.source, output)
        self.assertFalse(output.exists())
        record = build_dataset(self.source, output, allow_no_roi=True)
        self.assertEqual(record["no_roi_count"], 2)
        self.assertEqual(record["roi_count"], 0)
        self.assertTrue(all(item["teacher_view"]["kind"] == "no_evidence_roi" for item in record["examples"]))
        manifest = json.loads((output / "train.jsonl").read_text().strip())
        self.assertNotIn("student_image_hint", manifest)
        self.assertEqual(
            (output / manifest["student_image"]).read_bytes(),
            (output / manifest["teacher_image"]).read_bytes(),
        )

    def test_source_hash_and_cross_split_reuse_are_rejected_before_output(self) -> None:
        train = self._row("train", bbox=[3, 2, 8, 7])
        validation = self._row("validation", bbox=[3, 2, 8, 7])
        train["source_image_sha256"] = "0" * 64
        self._write(train, validation)
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "source_image_sha256 differs"):
            build_dataset(self.source, output)
        self.assertFalse(output.exists())
        train.pop("source_image_sha256")
        validation["source_image"] = "train.png"
        self._write(train, validation)
        with self.assertRaisesRegex(ValueError, "overlap by SHA256"):
            build_dataset(self.source, output)
        self.assertFalse(output.exists())

    def test_malformed_split_is_rejected_before_output(self) -> None:
        train = self._row("train", bbox=[3, 2, 8, 7])
        train["split"] = ["train"]
        self._write(train, self._row("validation", bbox=[3, 2, 8, 7]))
        output = self.root / "bad-split"
        with self.assertRaisesRegex(ValueError, "split must be train or validation"):
            build_dataset(self.source, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
