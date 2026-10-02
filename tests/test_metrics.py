import unittest
from pathlib import Path

from sure_vl.metrics import audit_attempts, evaluate
from sure_vl.protocol import Example, OutputAttempt, ProtocolError, StudentOutput, load_examples_jsonl, load_outputs_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "examples"


class MetricsTests(unittest.TestCase):
    @staticmethod
    def _example(example_id: str) -> Example:
        return Example.from_dict({
            "id": example_id, "split": "dev", "student_image": "restricted",
            "teacher_image": "clear", "question": "Q?",
            "required_visual_facts": {"shape": "circle"},
            "accepted_answers": ["blue"],
        })

    @staticmethod
    def _output(
        example_id: str, *, visual_correct: bool, answer_correct: bool,
        visual_confidence: int, conditional_confidence: int,
    ) -> StudentOutput:
        return StudentOutput.from_dict({
            "id": example_id,
            "visual_facts": {"shape": "circle" if visual_correct else "square"},
            "reasoning": "A test answer.",
            "answer": "blue" if answer_correct else "red",
            "visual_confidence": visual_confidence,
            "conditional_answer_confidence": conditional_confidence,
        })

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
        self.assertEqual(report["joint_outcomes"]["v0_y0"], {"count": 2, "rate": 1.0})
        self.assertEqual(report["answer_correct_given_visual_incorrect"], {
            "count": 0, "denominator": 2, "rate": 0.0,
        })
        self.assertEqual(report["reward_components"], {
            "sample_count": 0, "utility_mean": None,
            "calibration_mean": None, "total_mean": None,
        })
        self.assertIsNone(report["visual_confidence_mean"])
        self.assertIsNone(report["visual_confidence_observed_rate"])
        self.assertIsNone(report["conditional_answer_confidence_mean"])
        self.assertIsNone(report["conditional_answer_observed_rate"])

    def test_joint_outcomes_confidence_observation_and_reward_components(self):
        examples = [self._example(f"case-{index}") for index in range(4)]
        outputs = [
            self._output("case-0", visual_correct=True, answer_correct=True,
                         visual_confidence=90, conditional_confidence=80),
            self._output("case-1", visual_correct=True, answer_correct=False,
                         visual_confidence=70, conditional_confidence=30),
            self._output("case-2", visual_correct=False, answer_correct=True,
                         visual_confidence=20, conditional_confidence=95),
            self._output("case-3", visual_correct=False, answer_correct=False,
                         visual_confidence=10, conditional_confidence=90),
        ]
        report = evaluate(examples, outputs)
        self.assertEqual(report["labeled_sample_count"], 4)
        self.assertEqual(set(report["joint_outcomes"]), {
            "v0_y0", "v0_y1", "v1_y0", "v1_y1",
        })
        for outcome in report["joint_outcomes"].values():
            self.assertEqual(outcome, {"count": 1, "rate": 0.25})
        self.assertEqual(report["answer_correct_given_visual_incorrect"], {
            "count": 1, "denominator": 2, "rate": 0.5,
        })
        self.assertEqual(report["visual_confidence_sample_count"], 4)
        self.assertAlmostEqual(report["visual_confidence_mean"], 0.475)
        self.assertAlmostEqual(report["visual_confidence_observed_rate"], 0.5)
        self.assertEqual(report["conditional_answer_sample_count"], 2)
        self.assertAlmostEqual(report["conditional_answer_confidence_mean"], 0.55)
        self.assertAlmostEqual(report["conditional_answer_observed_rate"], 0.5)
        self.assertEqual(report["reward_components"]["sample_count"], 4)
        self.assertAlmostEqual(report["reward_components"]["utility_mean"], 1.5)
        self.assertAlmostEqual(report["reward_components"]["calibration_mean"], -0.07)
        self.assertAlmostEqual(report["reward_components"]["total_mean"], 1.43)
        self.assertAlmostEqual(report["reward_mean"], 1.43)

    def test_visual_correct_cohort_has_no_visual_failure_rate(self):
        example = self._example("correct")
        output = self._output(
            "correct", visual_correct=True, answer_correct=True,
            visual_confidence=90, conditional_confidence=80,
        )
        report = evaluate([example], [output])
        self.assertEqual(report["answer_correct_given_visual_incorrect"], {
            "count": 0, "denominator": 0, "rate": None,
        })

    def test_unlabeled_rows_are_rejected_instead_of_scored_as_zero(self):
        with self.assertRaisesRegex(ProtocolError, "labeled Example"):
            audit_attempts([{"id": "raw-unlabeled-row"}], [])

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
