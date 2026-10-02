import json
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


if __name__ == "__main__":
    unittest.main()
