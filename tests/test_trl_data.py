import json
import tempfile
import unittest
from pathlib import Path

from sure_vl.protocol import Example, ProtocolError
from sure_vl.trl_data import assert_disjoint_manifests, manifest_to_gold_rows


def _row(example_id: str, split: str, student: str, teacher: str) -> dict:
    return {
        "id": example_id,
        "split": split,
        "student_image": student,
        "teacher_image": teacher,
        "question": "What color is the square?",
        "required_visual_facts": {"color": "blue", "shape": "square"},
        "accepted_answers": ["blue"],
    }


class TRLDataTests(unittest.TestCase):
    def test_manifest_builds_paired_gold_rows_without_answer_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restricted.png").touch()
            (root / "clear.png").touch()
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(_row("sample-1", "train", "restricted.png", "clear.png")) + "\n")
            rows = manifest_to_gold_rows(manifest)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["example_id"], "sample-1")
            self.assertEqual(row["image"], str((root / "restricted.png").resolve()))
            self.assertEqual(row["teacher_image"], str((root / "clear.png").resolve()))
            self.assertEqual(row["completion"][0]["role"], "assistant")
            self.assertEqual(row["completion"][0]["content"][0]["text"], "")
            self.assertNotIn('"blue"', row["prompt"][0]["content"][1]["text"])
            restored = Example.from_dict(json.loads(row["example_payload"]))
            self.assertEqual(restored.id, "sample-1")
            self.assertEqual(restored.required_visual_facts["color"].canonical, "blue")

    def test_manifest_requires_existing_images_and_single_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(_row("a", "train", "absent.png", "also-absent.png")) + "\n")
            with self.assertRaisesRegex(ProtocolError, "image file does not exist"):
                manifest_to_gold_rows(manifest)
            manifest.write_text(
                json.dumps(_row("a", "train", "a.png", "b.png")) + "\n"
                + json.dumps(_row("b", "dev", "c.png", "d.png")) + "\n"
            )
            with self.assertRaisesRegex(ProtocolError, "one manifest"):
                manifest_to_gold_rows(manifest, check_images=False)

    def test_train_eval_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text(json.dumps(_row("a", "train", "one.png", "one-clear.png")) + "\n")
            dev.write_text(json.dumps(_row("b", "dev", "two.png", "one-clear.png")) + "\n")
            train_rows = manifest_to_gold_rows(train, check_images=False)
            dev_rows = manifest_to_gold_rows(dev, check_images=False)
            with self.assertRaisesRegex(ProtocolError, "teacher_image"):
                assert_disjoint_manifests(train_rows, dev_rows)

    def test_pair_must_have_distinct_image_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text(json.dumps(_row("a", "train", "same.png", "same.png")) + "\n")
            with self.assertRaisesRegex(ProtocolError, "paths must differ"):
                manifest_to_gold_rows(manifest, check_images=False)


if __name__ == "__main__":
    unittest.main()
