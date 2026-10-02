import json
import tempfile
import unittest
from pathlib import Path

from sure_vl.proxy_data import (
    assert_disjoint_proxy_manifests, manifest_to_proxy_rows,
)
from sure_vl.proxy_protocol import ProxyProtocolError


def _example(example_id: str, split: str, student: str, teacher: str) -> dict:
    return {
        "id": example_id, "split": split,
        "student_image": student, "teacher_image": teacher,
        "question": "What shape?", "accepted_answers": ["circle"],
    }


class ProxyDataTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("train-blur.png", "train-clear.png", "dev-blur.png", "dev-clear.png"):
            (self.root / name).write_bytes(b"image placeholder")
        self.train = self.root / "train.jsonl"
        self.dev = self.root / "dev.jsonl"
        self.train.write_text(json.dumps(_example("train-1", "train", "train-blur.png", "train-clear.png")) + "\n")
        self.dev.write_text(json.dumps(_example("dev-1", "dev", "dev-blur.png", "dev-clear.png")) + "\n")

    def test_rows_have_gold_schema_without_visual_fact_labels(self) -> None:
        rows = manifest_to_proxy_rows(self.train)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["example_id"], "train-1")
        self.assertEqual(row["image"], str((self.root / "train-blur.png").resolve()))
        self.assertEqual(row["teacher_image"], str((self.root / "train-clear.png").resolve()))
        self.assertNotIn("required_visual_facts", row)
        self.assertNotIn("required_visual_facts", row["example_payload"])
        self.assertIsNone(row["student_image_hint"])
        self.assertNotIn("circle", row["prompt"][0]["content"][1]["text"].lower())
        assert_disjoint_proxy_manifests(rows, manifest_to_proxy_rows(self.dev))

    def test_optional_evidence_reaches_payload_but_not_student_prompt(self) -> None:
        raw = _example("train-1", "train", "train-blur.png", "train-clear.png")
        raw["student_image_hint"] = "Only focus on the region inside the red bounding box."
        raw["teacher_evidence"] = {"scene_graph": [{"id": "secret-evidence-marker", "shape": "square"}]}
        self.train.write_text(json.dumps(raw) + "\n")
        row = manifest_to_proxy_rows(self.train)[0]
        payload = json.loads(row["example_payload"])
        student_text = row["prompt"][0]["content"][1]["text"]
        self.assertEqual(payload["teacher_evidence"], raw["teacher_evidence"])
        self.assertEqual(payload["student_image_hint"], raw["student_image_hint"])
        self.assertEqual(row["student_image_hint"], raw["student_image_hint"])
        self.assertIn(raw["student_image_hint"], student_text)
        self.assertNotIn("secret-evidence-marker", student_text)
        self.assertNotIn("scene_graph", student_text)
        self.assertNotIn("circle", student_text.lower())

    def test_missing_image_and_same_image_are_rejected(self) -> None:
        self.train.write_text(json.dumps(_example("train-1", "train", "missing.png", "train-clear.png")) + "\n")
        with self.assertRaisesRegex(ProxyProtocolError, "does not exist"):
            manifest_to_proxy_rows(self.train)
        self.train.write_text(json.dumps(_example("train-1", "train", "train-blur.png", "train-blur.png")) + "\n")
        with self.assertRaisesRegex(ProxyProtocolError, "must differ"):
            manifest_to_proxy_rows(self.train)

    def test_cross_split_image_overlap_is_rejected(self) -> None:
        self.dev.write_text(json.dumps(_example("dev-1", "dev", "train-blur.png", "dev-clear.png")) + "\n")
        with self.assertRaisesRegex(ProxyProtocolError, "overlap in image"):
            assert_disjoint_proxy_manifests(manifest_to_proxy_rows(self.train), manifest_to_proxy_rows(self.dev))

    def test_one_manifest_must_have_one_split(self) -> None:
        self.train.write_text(
            json.dumps(_example("train-1", "train", "train-blur.png", "train-clear.png")) + "\n"
            + json.dumps(_example("dev-1", "dev", "dev-blur.png", "dev-clear.png")) + "\n"
        )
        with self.assertRaisesRegex(ProxyProtocolError, "exactly one split"):
            manifest_to_proxy_rows(self.train)

    def test_cross_split_cross_role_image_overlap_is_rejected(self) -> None:
        self.dev.write_text(json.dumps(_example("dev-1", "dev", "dev-blur.png", "train-blur.png")) + "\n")
        with self.assertRaisesRegex(ProxyProtocolError, "overlap in image roles"):
            assert_disjoint_proxy_manifests(manifest_to_proxy_rows(self.train), manifest_to_proxy_rows(self.dev))


if __name__ == "__main__":
    unittest.main()
