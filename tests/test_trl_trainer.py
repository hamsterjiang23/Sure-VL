"""Small tensor checks for the TRL adapter without model downloads."""

import json
import unittest
from collections import defaultdict
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from sure_vl.trl_trainer import SureVLGOLDTrainer, _TRAIN_IMPORT_ERROR


class CharacterTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token)) for token in token_ids if token != 0)

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainerTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _TRAIN_IMPORT_ERROR is not None:
            raise _TRAIN_IMPORT_ERROR

    def _inputs(self, completion):
        payload = {
            "id": "x", "split": "train", "student_image": "student.png",
            "teacher_image": "teacher.png", "question": "What color?",
            "required_visual_facts": {"color": "blue"}, "accepted_answers": ["blue"],
        }
        completion_ids = [ord(character) for character in completion] + [0]
        ids = torch.tensor([[1] + completion_ids], dtype=torch.long)
        labels = ids.clone()
        labels[0, 0] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones_like(ids),
            "_sure_vl_rows": [{"example_payload": json.dumps(payload), "prompt": [], "teacher_image": "teacher.png"}],
        }

    def _trainer_stub(self):
        stub = SimpleNamespace()
        stub._tokenizer = CharacterTokenizer()
        stub.policy_weight = 1.0
        stub.opsd_weight = 1.0
        stub.opsd_temperature = 1.1
        stub.opsd_token_clip = 0.05
        stub.diagnostic_tokens = 4
        stub._metrics = {"train": defaultdict(list)}
        stub._get_model_forward_kwargs = lambda inputs: {}
        stub.teacher_calls = []

        def teacher_logits(row, content_ids):
            stub.teacher_calls.append((row, content_ids.detach().clone()))
            logits = torch.zeros((len(content_ids), 128))
            logits[:, ord("b")] = 3.0
            return logits

        stub._teacher_logits_for_content = teacher_logits
        return stub

    def _model(self, inputs):
        class Model(torch.nn.Module):
            def __init__(self, length):
                super().__init__()
                self.raw = torch.nn.Parameter(torch.zeros((1, length, 128)))

            def forward(self, **_kwargs):
                return SimpleNamespace(logits=self.raw)

        return Model(inputs["input_ids"].shape[1])

    def test_kl_gradient_only_reaches_content_positions(self):
        completion = (
            '<think><vision>{"color":"blue"}</vision>'
            '<reasoning>It is blue.</reasoning></think>\\boxed{blue}'
            '<confidence><vision_confidence>80</vision_confidence>'
            '<conditional_answer_confidence>90</conditional_answer_confidence></confidence>'
        )
        inputs = self._inputs(completion)
        stub = self._trainer_stub()
        model = self._model(inputs)
        stub.opsd_weight = 0.0
        baseline = SureVLGOLDTrainer.compute_loss(stub, model, inputs)
        baseline.backward()
        base_grad = model.raw.grad.detach().clone()
        self.assertEqual(stub.teacher_calls, [])
        model.zero_grad()
        stub.opsd_weight = 1.0
        joint = SureVLGOLDTrainer.compute_loss(stub, model, inputs)
        joint.backward()
        joint_grad = model.raw.grad.detach().clone()
        self.assertEqual(len(stub.teacher_calls), 1)
        self.assertEqual(stub._metrics["train"]["sure_vl/rank_local/opsd_rows"], [0.0, 1.0])
        self.assertGreater(stub._metrics["train"]["sure_vl/rank_local/opsd_raw_forward_kl_mean"][-1], 0)
        content_ids = stub.teacher_calls[0][1].tolist()
        self.assertEqual(content_ids, [ord(character) for character in completion.split("<confidence>")[0]])
        report_start = completion.index("<confidence>")
        self.assertGreater((joint_grad[0, :report_start] - base_grad[0, :report_start]).abs().sum().item(), 0)
        self.assertTrue(torch.allclose(joint_grad[0, report_start:], base_grad[0, report_start:]))

    def test_malformed_completion_keeps_policy_gradient_without_teacher(self):
        inputs = self._inputs("malformed")
        stub = self._trainer_stub()
        model = self._model(inputs)
        loss = SureVLGOLDTrainer.compute_loss(stub, model, inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.raw.grad.abs().sum().item(), 0)
        self.assertEqual(stub.teacher_calls, [])
        self.assertEqual(stub._metrics["train"]["sure_vl/rank_local/format_failure_count"], [1.0])
        self.assertEqual(stub._metrics["train"]["sure_vl/rank_local/effective_policy_reward"], [-3.0])

    def test_teacher_uses_clear_image_and_exact_sampled_ids(self):
        stub = SimpleNamespace()
        stub._SEQUENCE_KEYS = ()
        stub.accelerator = SimpleNamespace(device=torch.device("cpu"))
        seen = {}

        def extract(rows):
            seen["image"] = rows[0]["image"]
            return [[rows[0]["image"]]], [rows[0]["prompt"]]

        class Processor:
            def apply_chat_template(self, prompts, **_kwargs):
                return ["rendered prompt"]

            def __call__(self, *, images, text, padding, padding_side, add_special_tokens, return_tensors):
                seen["images"] = images
                seen["add_special_tokens"] = add_special_tokens
                return {
                    "input_ids": torch.tensor([[7, 8, 9]]),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                    "pixel_values": torch.tensor([[2.0]]),
                }

        class Teacher(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache, pixel_values):
                seen["input_ids"] = input_ids.tolist()
                seen["pixel_values"] = pixel_values.tolist()
                return SimpleNamespace(logits=torch.arange(input_ids.numel() * 4).reshape(1, -1, 4))

        stub._extract_images_and_prompts = extract
        stub._get_model_forward_kwargs = lambda values, exclude=(): {"pixel_values": values["pixel_values"]}
        stub.processing_class = Processor()
        stub.teacher_model = Teacher()
        result = SureVLGOLDTrainer._teacher_logits_for_content(
            stub, {"prompt": [{"role": "user", "content": "Q"}], "teacher_image": "clear.png"},
            torch.tensor([11, 12]),
        )
        self.assertEqual(seen["image"], "clear.png")
        self.assertEqual(seen["images"], [["clear.png"]])
        self.assertIs(seen["add_special_tokens"], False)
        self.assertEqual(seen["input_ids"], [[7, 8, 9, 11, 12]])
        self.assertEqual(seen["pixel_values"], [[2.0]])
        self.assertEqual(result.tolist(), [[8, 9, 10, 11], [12, 13, 14, 15]])


if __name__ == "__main__":
    unittest.main()
