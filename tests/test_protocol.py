"""Tests for the fixed event definition and strict JSONL contract."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from sure_vl.protocol import (
    Example,
    OutputAttempt,
    ProtocolError,
    StudentOutput,
    load_attempts_jsonl,
    load_examples_jsonl,
    load_outputs_jsonl,
    normalize_text,
    verify,
)


def example_record():
    return {
        "id": "item-1",
        "split": "dev",
        "student_image": "images/limited.png",
        "teacher_image": "images/clear.png",
        "question": "What color is the sign, and what number is on it?",
        "required_visual_facts": {
            "sign_color": {"canonical": "red", "aliases": ["crimson"]},
            "sign_number": "3",
        },
        "accepted_answers": ["red 3", "crimson three"],
    }


def output_record():
    return {
        "id": "item-1",
        "visual_facts": {"sign_color": "  CRIMSON  ", "sign_number": "３"},
        "reasoning": "The sign is crimson and shows 3.",
        "answer": " RED   3 ",
        "visual_confidence": 75,
        "conditional_answer_confidence": 60,
    }


class ProtocolTests(unittest.TestCase):
    def test_exact_normalized_match_and_alias(self):
        example = Example.from_dict(example_record())
        output = StudentOutput.from_dict(output_record())
        result = verify(example, output)
        self.assertEqual(dict(result.per_fact), {"sign_color": True, "sign_number": True})
        self.assertTrue(result.visual_correct)
        self.assertTrue(result.answer_correct)
        self.assertEqual(normalize_text("  ＲＥＤ \t3  "), "red 3")

    def test_missing_required_fact_is_false_even_when_answer_is_correct(self):
        data = output_record()
        del data["visual_facts"]["sign_number"]
        result = verify(Example.from_dict(example_record()), StudentOutput.from_dict(data))
        self.assertEqual(dict(result.per_fact), {"sign_color": True, "sign_number": False})
        self.assertFalse(result.visual_correct)
        self.assertTrue(result.answer_correct)

    def test_empty_fact_value_is_false_and_extra_fact_cannot_replace_required_slot(self):
        data = output_record()
        data["visual_facts"] = {"sign_color": "", "different_number": "3"}
        result = verify(Example.from_dict(example_record()), StudentOutput.from_dict(data))
        self.assertEqual(dict(result.per_fact), {"sign_color": False, "sign_number": False})
        self.assertFalse(result.visual_correct)

    def test_verifier_does_not_use_substrings_or_loose_punctuation(self):
        data = output_record()
        data["visual_facts"]["sign_color"] = "dark red"
        data["answer"] = "red 3."
        result = verify(Example.from_dict(example_record()), StudentOutput.from_dict(data))
        self.assertFalse(result.per_fact["sign_color"])
        self.assertFalse(result.visual_correct)
        self.assertFalse(result.answer_correct)

    def test_id_mismatch_is_rejected(self):
        data = output_record()
        data["id"] = "item-2"
        with self.assertRaisesRegex(ProtocolError, "does not match"):
            verify(Example.from_dict(example_record()), StudentOutput.from_dict(data))

    def test_confidences_must_be_integer_percent(self):
        for field in ("visual_confidence", "conditional_answer_confidence"):
            for invalid in (-1, 101, 0.5, True, "75"):
                data = output_record()
                data[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ProtocolError, "integer from 0 to 100"):
                        StudentOutput.from_dict(data)
        for bound in (0, 100):
            data = output_record()
            data["visual_confidence"] = bound
            data["conditional_answer_confidence"] = bound
            StudentOutput.from_dict(data)

    def test_invalid_schema_is_rejected(self):
        bad_example = example_record()
        bad_example["unknown"] = "ignored?"
        with self.assertRaisesRegex(ProtocolError, "unknown keys"):
            Example.from_dict(bad_example)

        bad_example = example_record()
        bad_example["required_visual_facts"] = {}
        with self.assertRaisesRegex(ProtocolError, "nonempty object"):
            Example.from_dict(bad_example)

        bad_example = example_record()
        bad_example["required_visual_facts"]["sign_color"]["aliases"] = ["RED"]
        with self.assertRaisesRegex(ProtocolError, "duplicates the canonical"):
            Example.from_dict(bad_example)

        bad_example = example_record()
        bad_example["accepted_answers"] = []
        with self.assertRaisesRegex(ProtocolError, "must be nonempty"):
            Example.from_dict(bad_example)

        bad_output = output_record()
        del bad_output["answer"]
        with self.assertRaisesRegex(ProtocolError, "missing keys"):
            StudentOutput.from_dict(bad_output)

    def test_example_and_output_are_frozen(self):
        example = Example.from_dict(example_record())
        output = StudentOutput.from_dict(output_record())
        with self.assertRaises(FrozenInstanceError):
            example.question = "Changed"
        with self.assertRaises(TypeError):
            example.required_visual_facts["sign_color"] = "blue"
        with self.assertRaises(TypeError):
            output.visual_facts["sign_color"] = "blue"

    def test_jsonl_loaders_reject_silent_record_loss_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(json.dumps(example_record(), ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(len(load_examples_jsonl(path)), 1)

            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, ":1: blank line"):
                load_examples_jsonl(path)

            record = json.dumps(example_record(), ensure_ascii=False)
            path.write_text(record + "\n" + record + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, ":2: duplicate id"):
                load_examples_jsonl(path)

            path.write_text('{"id":"a","id":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, ":1: duplicate JSON key"):
                load_examples_jsonl(path)

            output_path = Path(directory) / "outputs.jsonl"
            output_path.write_text(json.dumps(output_record()) + "\n", encoding="utf-8")
            self.assertEqual(load_outputs_jsonl(output_path)[0].id, "item-1")

    def test_attempt_loader_preserves_assignable_format_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            malformed = {"id": "item-2", "visual_facts": {}, "reasoning": "I cannot tell."}
            path.write_text(
                json.dumps(output_record()) + "\n" + json.dumps(malformed) + "\n",
                encoding="utf-8",
            )
            valid, failed = load_attempts_jsonl(path)
            self.assertIsInstance(valid, OutputAttempt)
            self.assertEqual(valid.id, valid.parsed.id)
            self.assertIsNone(valid.format_error)
            self.assertEqual(failed.id, "item-2")
            self.assertIsNone(failed.parsed)
            self.assertIn("missing keys", failed.format_error)
            with self.assertRaisesRegex(ProtocolError, ":2: .*missing keys"):
                load_outputs_jsonl(path)

            path.write_text("", encoding="utf-8")
            self.assertEqual(load_attempts_jsonl(path), ())
            with self.assertRaisesRegex(ProtocolError, "JSONL file is empty"):
                load_outputs_jsonl(path)

    def test_attempt_loader_rejects_unassignable_and_duplicate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            bad_records = (
                ('{"id":"item-1",', "Expecting"),
                ('{"reasoning":"no id"}', "with id"),
                ('{"id":" "}', "id must be nonempty"),
                ('{"id":9}', "id must be a string"),
                ('{"id":"a","id":"b"}', "duplicate JSON key"),
                ('[]', "with id"),
                ('', "blank line"),
            )
            for raw, message in bad_records:
                with self.subTest(raw=raw):
                    path.write_text(raw + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ProtocolError, ":1: .*" + message):
                        load_attempts_jsonl(path)

            path.write_text('{"id":"item-1"}\n{"id":"item-1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, ":2: duplicate id"):
                load_attempts_jsonl(path)


if __name__ == "__main__":
    unittest.main()
