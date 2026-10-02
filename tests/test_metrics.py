import unittest
from pathlib import Path

from sure_vl.metrics import audit_attempts, evaluate
from sure_vl.protocol import Example, OutputAttempt, ProtocolError, StudentOutput, load_examples_jsonl, load_outputs_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "examples"


class MetricsTests(unittest.TestCase):
    def test_synthetic_split_counts_and_gating(self):
        examples = load_examples_jsonl(FIXTURES / "synthetic_examples.jsonl")
        outputs = load_outputs_jsonl(FIXTURES / "synthetic_outputs.jsonl")
        report = evaluate(examples, outputs)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["visual_correct"], {"count": 1, "rate": 0.5})
        self.assertEqual(report["answer_correct"], {"count": 1, "rate": 0.5})
        self.assertEqual(report["conditional_answer_sample_count"], 1)
        self.assertEqual(report["required_facts_per_example"], {"min": 2, "max": 2, "mean": 2})
        self.assertAlmostEqual(report["visual_brier"], 0.2)
        self.assertAlmostEqual(report["conditional_answer_brier"], 0.01)
        self.assertAlmostEqual(report["reward_mean"], 1.295)
        self.assertEqual(report["required_fact_accuracy"]["color"]["total"], 2)

    def test_missing_required_fact_stays_in_denominator(self):
        example = Example.from_dict({
            "id": "a", "split": "dev", "student_image": "restricted",
            "teacher_image": "clear", "question": "Q?",
            "required_visual_facts": {"shape": "circle", "color": "blue"},
            "accepted_answers": ["blue"],
        })
        output = StudentOutput.from_dict({
            "id": "a", "visual_facts": {"shape": "circle"},
            "reasoning": "Guess.", "answer": "blue",
            "visual_confidence": 0, "conditional_answer_confidence": 100,
        })
        report = evaluate([example], [output])
        self.assertEqual(report["visual_correct"]["count"], 0)
        self.assertEqual(report["answer_correct"]["count"], 1)
        self.assertEqual(report["required_fact_accuracy"]["color"]["total"], 1)
        self.assertEqual(report["required_fact_accuracy"]["color"]["correct"], 0)
        self.assertIsNone(report["conditional_answer_brier"])

    def test_missing_output_is_rejected(self):
        examples = load_examples_jsonl(FIXTURES / "synthetic_examples.jsonl")
        outputs = load_outputs_jsonl(FIXTURES / "synthetic_outputs.jsonl")
        with self.assertRaisesRegex(ProtocolError, "missing="):
            evaluate(examples, outputs[:1])

    def test_attempt_audit_counts_format_and_missing_failures(self):
        examples = load_examples_jsonl(FIXTURES / "synthetic_examples.jsonl")
        malformed = OutputAttempt(id="toy-1", parsed=None, format_error="missing confidence")
        report = audit_attempts(examples, [malformed])
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["format_failure_count"], 1)
        self.assertEqual(report["missing_attempt_count"], 1)
        self.assertEqual(report["output_coverage"]["rate"], 0.5)
        self.assertEqual(report["required_fact_accuracy"]["color"]["total"], 2)
        self.assertEqual(report["visual_correct"]["count"], 0)
        self.assertEqual(report["answer_correct"]["count"], 0)
        self.assertEqual(report["reward_sample_count"], 0)
        self.assertIsNone(report["reward_mean"])
        self.assertIsNone(report["visual_brier"])

    def test_mixed_splits_are_rejected(self):
        examples = load_examples_jsonl(FIXTURES / "synthetic_examples.jsonl")
        other = Example.from_dict({
            "id": "other", "split": "test", "student_image": "restricted",
            "teacher_image": "clear", "question": "Q?",
            "required_visual_facts": {"shape": "circle"},
            "accepted_answers": ["circle"],
        })
        with self.assertRaisesRegex(ProtocolError, "one split"):
            evaluate([*examples, other], [])


if __name__ == "__main__":
    unittest.main()
