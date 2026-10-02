import json
import math
import tempfile
import unittest
from pathlib import Path

from sure_vl.proxy_protocol import (
    ProxyExample, ProxyProtocolError, load_proxy_examples_jsonl, verify_proxy_answer,
)


def _example() -> dict:
    return {
        "id": "sample-1", "split": "dev", "student_image": "blur.png",
        "teacher_image": "clear.png", "question": "What is in the image?",
        "accepted_answers": ["blue circle", "circle in blue"],
    }


class ProxyProtocolTests(unittest.TestCase):
    def test_answer_only_schema_and_normalization(self) -> None:
        example = ProxyExample.from_dict(_example())
        self.assertTrue(verify_proxy_answer(example, "  BLUE   CIRCLE "))
        self.assertFalse(verify_proxy_answer(example, "red circle"))
        self.assertIsNone(verify_proxy_answer(example, None))
        self.assertIsNone(verify_proxy_answer(example, "  "))
        self.assertNotIn("required_visual_facts", example.to_dict())
        self.assertNotIn("teacher_evidence", example.to_dict())
        self.assertNotIn("student_image_hint", example.to_dict())
        self.assertNotIn("teacher_question", example.to_dict())

    def test_optional_hint_and_teacher_evidence_round_trip(self) -> None:
        raw = _example()
        raw["student_image_hint"] = "Only focus on the region inside the red bounding box."
        raw["teacher_question"] = "What is shown?\n\nA. circle\nB. square\nC. triangle\nD. star\n\nAnswer with the option's letter."
        raw["teacher_evidence"] = {
            "scene_graph": {"objects": [{"color": "blue", "x": 1.5}], "relations": []},
            "available": True,
        }
        example = ProxyExample.from_dict(raw)
        raw["teacher_evidence"]["scene_graph"]["objects"][0]["color"] = "red"
        self.assertEqual(example.teacher_evidence["scene_graph"]["objects"][0]["color"], "blue")
        self.assertEqual(example.to_dict()["student_image_hint"],
                         "Only focus on the region inside the red bounding box.")
        self.assertEqual(example.to_dict()["teacher_question"], raw["teacher_question"])
        self.assertEqual(ProxyExample.from_dict(example.to_dict()).to_dict(), example.to_dict())
        for evidence in ("A clearly visible blue object.", ["object", {"count": 2}]):
            variant = _example()
            variant["teacher_evidence"] = evidence
            self.assertEqual(ProxyExample.from_dict(variant).to_dict()["teacher_evidence"], evidence)

    def test_rejects_invalid_optional_evidence_and_hint(self) -> None:
        for evidence in (" ", 4, True, {"score": math.nan}, [float("inf")],
                         {1: "bad key"}, {"nested": ("tuple",)}):
            raw = _example()
            raw["teacher_evidence"] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ProxyProtocolError):
                ProxyExample.from_dict(raw)
        raw = _example()
        raw["student_image_hint"] = "  "
        with self.assertRaisesRegex(ProxyProtocolError, "student_image_hint"):
            ProxyExample.from_dict(raw)
        for bad_question in ("  ", 42, []):
            raw = _example()
            raw["teacher_question"] = bad_question
            with self.subTest(teacher_question=bad_question), self.assertRaisesRegex(ProxyProtocolError, "teacher_question"):
                ProxyExample.from_dict(raw)

    def test_rejects_legacy_fact_slots_and_duplicate_answer_aliases(self) -> None:
        legacy = _example()
        legacy["required_visual_facts"] = {"color": "blue"}
        with self.assertRaisesRegex(ProxyProtocolError, "extra"):
            ProxyExample.from_dict(legacy)
        repeated = _example()
        repeated["accepted_answers"] = ["Blue Circle", " blue  circle "]
        with self.assertRaisesRegex(ProxyProtocolError, "duplicate"):
            ProxyExample.from_dict(repeated)

    def test_jsonl_rejects_duplicate_ids_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            path.write_text(json.dumps(_example()) + "\n" + json.dumps(_example()) + "\n")
            with self.assertRaisesRegex(ProxyProtocolError, "duplicate example ID"):
                load_proxy_examples_jsonl(path)
            path.write_text('{"id":"a","id":"b"}\n')
            with self.assertRaisesRegex(ProxyProtocolError, "duplicate JSON key"):
                load_proxy_examples_jsonl(path)
            raw = _example()
            raw["teacher_evidence"] = {"score": float("nan")}
            path.write_text(json.dumps(raw) + "\n")
            with self.assertRaisesRegex(ProxyProtocolError, "nonfinite JSON number"):
                load_proxy_examples_jsonl(path)


if __name__ == "__main__":
    unittest.main()
