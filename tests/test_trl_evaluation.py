"""Frozen validation selection, strict parsing, and callback scheduling."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sure_vl.protocol import Example, ProtocolError
from sure_vl.trl_evaluation import (
    FrozenEvaluation,
    FrozenValidationCallback,
    attempt_from_text,
    evaluate_frozen_subset,
    select_frozen_subset,
)


def _row(example_id: str) -> dict:
    payload = {
        "id": example_id,
        "split": "dev",
        "student_image": f"/test/{example_id}-blur.png",
        "teacher_image": f"/test/{example_id}-clear.png",
        "question": "What shape is visible?",
        "required_visual_facts": {"shape": "circle"},
        "accepted_answers": ["circle"],
    }
    return {
        "example_id": example_id,
        "split": "dev",
        "image": payload["student_image"],
        "example_payload": json.dumps(payload),
    }


def _completion(*, fact: str = "circle", answer: str = "circle") -> str:
    return (
        '<think><vision>{"shape":"' + fact + '"}</vision>'
        '<reasoning>I see a shape.</reasoning></think>'
        '\\boxed{' + answer + '}'
        '<confidence><vision_confidence>80</vision_confidence>'
        '<conditional_answer_confidence>70</conditional_answer_confidence></confidence>'
    )


class FrozenEvaluationTests(unittest.TestCase):
    def test_subset_selection_is_order_independent_and_labeled(self) -> None:
        rows = [_row(f"dev-{index}") for index in range(8)]
        first = select_frozen_subset(rows, size=3, seed=42)
        second = select_frozen_subset(list(reversed(rows)), size=3, seed=42)
        self.assertEqual([row["example_id"] for row in first], [row["example_id"] for row in second])
        self.assertEqual(len(first), 3)
        self.assertIsNot(first[0], rows[0])

        unlabeled = _row("bad")
        payload = json.loads(unlabeled["example_payload"])
        del payload["required_visual_facts"]
        unlabeled["example_payload"] = json.dumps(payload)
        with self.assertRaises(ProtocolError):
            select_frozen_subset([unlabeled], size=1, seed=42)

    def test_attempt_preserves_raw_failure_and_verified_valid_output(self) -> None:
        example = Example.from_dict(json.loads(_row("dev-1")["example_payload"]))
        valid, record = attempt_from_text(example, _completion(fact="square", answer="circle"))
        self.assertIsNotNone(valid.parsed)
        self.assertEqual(record["visual_correct"], False)
        self.assertEqual(record["answer_correct"], True)
        self.assertEqual(record["per_fact_correct"], {"shape": False})
        self.assertEqual(record["visual_confidence"], 80)
        self.assertNotIn("missing", record["reward"])

        invalid, record = attempt_from_text(example, "cannot parse")
        self.assertIsNone(invalid.parsed)
        self.assertEqual(record["raw_completion"], "cannot parse")
        self.assertFalse(record["parsed"])
        self.assertIsNone(record["visual_correct"])
        self.assertIsNone(record["answer_correct"])
        self.assertIn("expected", record["format_error"])

    def test_callback_emits_step_zero_interval_and_final_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            callback = FrozenValidationCallback(
                validation_rows=[_row("dev-1"), _row("dev-2")],
                processor=object(),
                output_dir=temporary,
                subset_size=2,
                every_n_steps=20,
                final_step=100,
            )
            fake = FrozenEvaluation(
                {"sample_count": 2, "visual_brier": None},
                ({"id": "dev-1", "raw_completion": "bad"},),
            )
            state = SimpleNamespace(global_step=0, is_world_process_zero=True)
            control = object()
            with patch("sure_vl.trl_evaluation.evaluate_frozen_subset", return_value=fake) as evaluate:
                self.assertIs(callback.on_train_begin(None, state, control, model=object()), control)
                state.global_step = 19
                callback.on_step_end(None, state, control, model=object())
                state.global_step = 20
                callback.on_step_end(None, state, control, model=object())
                state.global_step = 100
                callback.on_step_end(None, state, control, model=object())
                callback.on_train_end(None, state, control, model=object())
                self.assertEqual(evaluate.call_count, 3)

            metrics_path = Path(temporary) / "validation_metrics.jsonl"
            reports = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            self.assertEqual([report["optimizer_step"] for report in reports], [0, 20, 100])
            self.assertEqual(len({tuple(report["subset_ids"]) for report in reports}), 1)
            self.assertTrue((Path(temporary) / "validation_attempts_step_000100.jsonl").exists())

    def test_callback_reads_early_end_as_its_actual_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            callback = FrozenValidationCallback(
                validation_rows=[_row("dev-1")], processor=object(), output_dir=temporary,
                subset_size=1, every_n_steps=20, final_step=100,
            )
            state = SimpleNamespace(global_step=0, is_world_process_zero=True)
            fake = FrozenEvaluation({"sample_count": 1}, ())
            with patch("sure_vl.trl_evaluation.evaluate_frozen_subset", return_value=fake):
                callback.on_train_begin(None, state, None, model=object())
                state.global_step = 17
                callback.on_train_end(None, state, None, model=object())
            reports = [json.loads(line) for line in (Path(temporary) / "validation_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([report["optimizer_step"] for report in reports], [0, 17])

    def test_one_shot_generation_chain_with_fake_model_and_processor(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            @property
            def shape(self):
                return (len(self.values), len(self.values[0])) if isinstance(self.values[0], list) else (len(self.values),)

            @property
            def ndim(self):
                return len(self.shape)

            def to(self, _device):
                return self

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row, selection = key
                    return FakeTensor(self.values[row][selection])
                return FakeTensor(self.values[key])

            def tolist(self):
                return self.values

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def decode(self, ids, **_):
                return "".join(chr(value) for value in ids)

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, prompt, **_):
                self.prompt = prompt
                return "PROMPT"

            def __call__(self, *, images, text, padding, return_tensors):
                self.images = images
                self.text = text
                return {"input_ids": FakeTensor([[1, 2]]), "attention_mask": FakeTensor([[1, 1]])}

        class Model:
            training = True

            def parameters(self):
                yield SimpleNamespace(device=SimpleNamespace(type="cpu", index=None))

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeTensor([[1, 2, *(ord(char) for char in _completion()), 0]])

        fake_torch = SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            random=SimpleNamespace(fork_rng=lambda **_: contextlib.nullcontext()),
            manual_seed=lambda _: None,
        )
        processor = Processor()
        model = Model()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("sure_vl.trl_evaluation._student_image", return_value="student-image"):
                result = evaluate_frozen_subset(model, processor, [_row("dev-1")], max_new_tokens=512)
        self.assertTrue(model.training)
        self.assertEqual(result.report["sample_count"], 1)
        self.assertEqual(result.report["visual_correct"]["count"], 1)
        self.assertEqual(result.report["answer_correct"]["count"], 1)
        self.assertEqual(result.report["conditional_answer_sample_count"], 1)
        self.assertTrue(result.attempts[0]["parsed"])
        self.assertEqual(processor.prompt[0]["content"][0]["image"], "student-image")
        self.assertEqual(processor.images, [["student-image"]])
        self.assertEqual(model.kwargs["pad_token_id"], 0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"),
        "optional train dependencies are not installed",
    )
    def test_one_shot_uses_student_image_and_shared_parser(self) -> None:
        import torch
        from PIL import Image

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def decode(self, ids, **_):
                return "".join(chr(value) for value in ids)

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, prompt, **_):
                self.prompt = prompt
                return "PROMPT"

            def __call__(self, *, images, text, padding, return_tensors):
                self.images = images
                self.text = text
                return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        class Model(torch.nn.Module):
            def __init__(self, completion):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.completion = completion

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, *(ord(char) for char in self.completion), 0]])

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "student.png"
            Image.new("RGB", (2, 2), "blue").save(image_path)
            row = _row("dev-1")
            row["image"] = str(image_path)
            payload = json.loads(row["example_payload"])
            payload["student_image"] = str(image_path)
            row["example_payload"] = json.dumps(payload)
            processor = Processor()
            model = Model(_completion())
            model.train()
            result = evaluate_frozen_subset(
                model, processor, [row], max_new_tokens=512, seed=42,
            )
            self.assertTrue(model.training)
            self.assertEqual(result.report["sample_count"], 1)
            self.assertEqual(result.report["visual_correct"]["count"], 1)
            self.assertEqual(result.report["answer_correct"]["count"], 1)
            self.assertEqual(result.report["conditional_answer_sample_count"], 1)
            self.assertEqual(result.attempts[0]["parsed"], True)
            self.assertEqual(processor.prompt[0]["content"][0]["type"], "image")
            self.assertEqual(processor.images[0][0].size, (2, 2))


if __name__ == "__main__":
    unittest.main()
