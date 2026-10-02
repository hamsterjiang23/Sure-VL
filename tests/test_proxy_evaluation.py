import importlib.util
import json
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sure_vl.proxy_evaluation import (
    ProxyEvaluation,
    ProxyValidationCallback,
    evaluate_proxy_subset,
    select_proxy_subset,
)


def _row(example_id):
    return {
        "example_id": example_id,
        "split": "dev",
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}],
        "image": "student-image",
        "teacher_image": "teacher-image",
        "example_payload": json.dumps({"id": example_id, "answer": "A"}),
    }


class ProxyEvaluationTests(unittest.TestCase):
    def test_subset_selection_is_stable_and_rejects_duplicate_ids(self):
        rows = [_row(f"dev-{index}") for index in range(8)]
        forward = select_proxy_subset(rows, size=3, seed=17)
        reverse = select_proxy_subset(list(reversed(rows)), size=3, seed=17)
        self.assertEqual([row["example_id"] for row in forward], [row["example_id"] for row in reverse])
        self.assertEqual(len(forward), 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_proxy_subset([rows[0], rows[0]], size=2, seed=17)
        with self.assertRaisesRegex(ValueError, "teacher_image"):
            select_proxy_subset([{**rows[0], "teacher_image": None}], size=1, seed=17)
        with self.assertRaisesRegex(ValueError, "differs from example_payload"):
            select_proxy_subset([{**rows[0], "example_payload": json.dumps({"id": "other"})}], size=1, seed=17)

    def test_callback_writes_step_zero_periodic_and_final_once_with_update_evidence(self):
        class Evidence:
            successful_updates = 0

            def summary(self):
                return {
                    "optimizer_attempted_steps": self.successful_updates + 1 if self.successful_updates else 0,
                    "optimizer_successful_updates": self.successful_updates,
                    "optimizer_skipped_updates": 1 if self.successful_updates else 0,
                    "teacher_ema_updates": self.successful_updates,
                }

        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                optimizer_evidence=Evidence(),
                optimizer=SimpleNamespace(state={}),
                state=SimpleNamespace(global_step=0),
            )
            callback = ProxyValidationCallback(
                trainer=trainer, validation_rows=[_row("a"), _row("b")],
                processor=object(), output_dir=directory,
                subset_size=2, subset_seed=3, every_n_steps=20, final_step=100,
            )
            result = ProxyEvaluation(
                report={"sample_count": 2, "answer_accuracy": 0.5},
                attempts=({"id": "a", "raw_completion": "first"},
                          {"id": "b", "raw_completion": "second"}),
            )
            state = SimpleNamespace(global_step=0, is_world_process_zero=True)
            with patch("sure_vl.proxy_evaluation.evaluate_proxy_subset", return_value=result) as evaluate:
                callback.on_train_begin(None, state, None)
                state.global_step = trainer.state.global_step = 19
                callback.on_step_end(None, state, None)
                trainer.optimizer_evidence.successful_updates = 19
                state.global_step = trainer.state.global_step = 20
                callback.on_step_end(None, state, None)
                trainer.optimizer_evidence.successful_updates = 99
                state.global_step = trainer.state.global_step = 100
                callback.on_step_end(None, state, None)
                callback.on_train_end(None, state, None)
                self.assertEqual(evaluate.call_count, 3)
            reports = [
                json.loads(line)
                for line in (Path(directory) / "proxy_validation_metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([report["trainer_state_global_step"] for report in reports], [0, 20, 100])
            self.assertEqual([report["optimizer_step"] for report in reports], [0, 20, 100])
            self.assertEqual([report["actual_successful_optimizer_updates"] for report in reports], [0, 19, 99])
            self.assertEqual([report["optimizer_state_max_step"] for report in reports], [0, None, None])
            self.assertEqual(len({tuple(report["subset_ids"]) for report in reports}), 1)
            self.assertTrue(reports[-1]["reached_requested_final_step"])
            attempts = Path(directory) / "proxy_validation_attempts_step_000100.jsonl"
            self.assertEqual(len(attempts.read_text().splitlines()), 2)

    def test_callback_reads_early_end_at_actual_step(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(optimizer_evidence=None, optimizer=None)
            callback = ProxyValidationCallback(
                trainer=trainer, validation_rows=[_row("a")], processor=object(),
                output_dir=directory, subset_size=1, every_n_steps=20, final_step=100,
            )
            state = SimpleNamespace(global_step=0, is_world_process_zero=True)
            result = ProxyEvaluation(report={"sample_count": 1}, attempts=({"id": "a"},))
            with patch("sure_vl.proxy_evaluation.evaluate_proxy_subset", return_value=result):
                callback.on_train_begin(None, state, None)
                state.global_step = 17
                callback.on_train_end(None, state, None)
            reports = [
                json.loads(line)
                for line in (Path(directory) / "proxy_validation_metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([report["trainer_state_global_step"] for report in reports], [0, 17])
            self.assertIsNone(reports[-1]["actual_successful_optimizer_updates"])
            self.assertFalse(reports[-1]["reached_requested_final_step"])

    def test_logit_alignment_without_optional_torch_install(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            @property
            def shape(self):
                result = []
                value = self.values
                while isinstance(value, list):
                    result.append(len(value))
                    value = value[0] if value else None
                return tuple(result)

            @property
            def ndim(self):
                return len(self.shape)

            def __getitem__(self, keys):
                if not isinstance(keys, tuple):
                    keys = (keys,)

                def select(value, remaining):
                    if not remaining:
                        return value
                    key, *rest = remaining
                    if isinstance(key, slice):
                        return [select(item, rest) for item in value[key]]
                    return select(value[key], rest)

                return Tensor(select(self.values, keys))

            def to(self, _device):
                return self

            def tolist(self):
                return self.values

            def numel(self):
                size = 1
                for dimension in self.shape:
                    size *= dimension
                return size

            def new_ones(self, shape):
                return Tensor([[1] * shape[-1]])

            def new_zeros(self, shape):
                return Tensor([[0] * shape[-1]])

        def cat(tensors, dim):
            self.assertIn(dim, (1, -1))
            return Tensor([sum((tensor.values[row] for tensor in tensors), []) for row in range(len(tensors[0].values))])

        fake_torch = SimpleNamespace(
            no_grad=contextlib.nullcontext,
            random=SimpleNamespace(fork_rng=lambda **_: contextlib.nullcontext()),
            manual_seed=lambda _: None,
            equal=lambda a, b: a.values == b.values,
            cat=cat,
        )

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def decode(self, ids, **_):
                return "".join(chr(value) for value in ids)

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, prompts, **_):
                return ["PROMPT"]

            def __call__(self, **_):
                return {
                    "input_ids": Tensor([[5, 6]]),
                    "attention_mask": Tensor([[1, 1]]),
                    "mm_token_type_ids": Tensor([[4, 4]]),
                    "token_type_ids": Tensor([[3, 3]]),
                    "pixel_values": Tensor([[0.25, 0.5]]),
                    "image_grid_thw": Tensor([[1, 2, 3]]),
                }

        class Model:
            training = True

            def parameters(self):
                yield SimpleNamespace(device=SimpleNamespace(type="cpu", index=None))

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

            def generate(self, **kwargs):
                return Tensor([[5, 6, 65, 66, 0]])

            def __call__(self, **kwargs):
                self.forward_kwargs = kwargs
                return SimpleNamespace(logits=Tensor([[[index] * 3 for index in range(5)]]))

        class Trainer:
            def __init__(self):
                self.model = Model()
                self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)

            def _extract_images_and_prompts(self, rows):
                return [[rows[0]["image"]]], [rows[0]["prompt"]]

            def measure_rollout(self, row, completion_ids, selected_student_logits, *, diagnostics):
                self.measured = (row, completion_ids.tolist(), selected_student_logits.tolist(), diagnostics)
                return SimpleNamespace(record={
                    "id": row["example_id"], "answer_correct": True,
                    "visual_confidence": 0.8, "answer_confidence": 0.9,
                    "visual_proxy": 0.7, "proxy_fallback": False,
                    "vision_tokens": 2, "format_errors": [],
                })

        trainer = Trainer()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("sure_vl.proxy_evaluation._image", side_effect=lambda value: value):
                result = evaluate_proxy_subset(trainer, Processor(), [_row("dev-1")], max_new_tokens=8)
        self.assertTrue(trainer.model.training)
        self.assertEqual(result.attempts[0]["raw_completion"], "AB")
        self.assertEqual(result.attempts[0]["generated_ids"], [65, 66, 0])
        measured_row, ids, logits, diagnostics = trainer.measured
        self.assertEqual(ids, [65, 66, 0])
        self.assertEqual(logits, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        self.assertFalse(diagnostics)
        self.assertEqual(measured_row["student_image"], "student-image")
        self.assertEqual(measured_row["teacher_image"], "teacher-image")
        forward = trainer.model.forward_kwargs
        self.assertEqual(forward["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(forward["mm_token_type_ids"].tolist(), [[4, 4, 0, 0, 0]])
        self.assertEqual(forward["token_type_ids"].tolist(), [[3, 3, 0, 0, 0]])
        self.assertEqual(forward["pixel_values"].tolist(), [[0.25, 0.5]])
        self.assertEqual(forward["image_grid_thw"].tolist(), [[1, 2, 3]])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"),
        "torch and Pillow are needed for the multimodal forward test",
    )
    def test_generated_token_positions_and_image_kwargs_are_preserved(self):
        import torch
        from PIL import Image

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def decode(self, ids, **_):
                return "".join(chr(value) for value in ids)

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, prompts, **_):
                self.prompts = prompts
                return ["PROMPT"]

            def __call__(self, **kwargs):
                self.processor_kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[5, 6]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "mm_token_type_ids": torch.tensor([[4, 4]]),
                    "token_type_ids": torch.tensor([[3, 3]]),
                    "pixel_values": torch.tensor([[0.25, 0.5]]),
                    "image_grid_thw": torch.tensor([[1, 2, 3]]),
                }

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.forward_kwargs = None

            def generate(self, **kwargs):
                self.generation_kwargs = kwargs
                return torch.tensor([[5, 6, 65, 66, 0]])

            def forward(self, **kwargs):
                self.forward_kwargs = kwargs
                length = kwargs["input_ids"].shape[1]
                logits = torch.arange(length, dtype=torch.float32).view(1, length, 1).expand(1, length, 3)
                return SimpleNamespace(logits=logits)

        class Trainer:
            def __init__(self):
                self.model = Model()
                self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)

            def _extract_images_and_prompts(self, rows):
                return [[rows[0]["image"]]], [rows[0]["prompt"]]

            def measure_rollout(self, row, completion_ids, selected_student_logits, *, diagnostics):
                self.measurement = (row, completion_ids.clone(), selected_student_logits.clone(), diagnostics)
                return SimpleNamespace(record={
                    "id": row["example_id"],
                    "answer_correct": True,
                    "answer_label_available": True,
                    "visual_confidence": 0.8,
                    "answer_confidence": 0.9,
                    "visual_proxy": 0.7,
                    "proxy_fallback": False,
                    "vision_tokens": 2,
                    "format_errors": [],
                })

        trainer = Trainer()
        processor = Processor()
        row = _row("dev-1")
        student_image = Image.new("RGB", (2, 2), "blue")
        teacher_image = Image.new("RGB", (3, 3), "blue")
        row["image"] = student_image
        row["teacher_image"] = teacher_image
        trainer.model.train()
        result = evaluate_proxy_subset(
            trainer, processor, [row], max_new_tokens=8, seed=7,
        )
        self.assertTrue(trainer.model.training)
        self.assertEqual(result.report["answer_accuracy"], 1.0)
        self.assertEqual(result.report["visual_proxy_pair_count"], 1)
        self.assertEqual(result.attempts[0]["generated_ids"], [65, 66, 0])
        self.assertEqual(result.attempts[0]["raw_completion"], "AB")
        measured_row, ids, logits, diagnostics = trainer.measurement
        self.assertIsInstance(measured_row["student_image"], Image.Image)
        self.assertIsInstance(measured_row["teacher_image"], Image.Image)
        self.assertEqual(ids.tolist(), [65, 66, 0])
        self.assertEqual(logits[:, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertFalse(diagnostics)
        forward = trainer.model.forward_kwargs
        self.assertEqual(forward["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(forward["mm_token_type_ids"].tolist(), [[4, 4, 0, 0, 0]])
        self.assertEqual(forward["token_type_ids"].tolist(), [[3, 3, 0, 0, 0]])
        self.assertEqual(forward["pixel_values"].tolist(), [[0.25, 0.5]])
        self.assertEqual(forward["image_grid_thw"].tolist(), [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
