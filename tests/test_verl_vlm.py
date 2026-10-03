"""Focused fake-VLM checks for the independent veRL HF runtime."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sure_vl.proxy_protocol import ProxyExample
from sure_vl.training.verl.vlm import VLMRuntime, _checked_config


class RuntimeConfigurationTests(unittest.TestCase):
    def test_requires_frozen_fp32_nonthinking_cuda_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {"model_id": directory, "model_revision": None,
                    "setting": {"max_pixels": 65536, "bf16": False, "fp16": False,
                                "enable_thinking": False, "gradient_checkpointing": True}}
            self.assertEqual(_checked_config(base), (directory, None, 65536, True))
            for bad, message in (
                ({"bf16": True}, "FP32"),
                ({"fp16": True}, "FP32"),
                ({"enable_thinking": True}, "enable_thinking"),
                ({"max_pixels": 0}, "max_pixels"),
            ):
                altered = {**base, "setting": {**base["setting"], **bad}}
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                    _checked_config(altered)
            with self.assertRaisesRegex(ValueError, "pinned model_revision"):
                _checked_config({**base, "model_id": "remote/model"})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"),
                     "fake VLM tensor checks require torch and Pillow")
class FakeVLMRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image

        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.student_path = str(Path(self.temporary.name) / "student.png")
        self.teacher_path = str(Path(self.temporary.name) / "teacher.png")
        Image.new("RGB", (8, 8), "red").save(self.student_path)
        Image.new("RGB", (8, 8), "blue").save(self.teacher_path)

        class Processor:
            def __init__(self):
                self.messages = []
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.messages.append(messages)
                return repr(messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

            def __call__(self, *, images, text, **kwargs):
                self.calls.append({"images": images, "text": text, **kwargs})
                assert len(images) == len(text) == 1
                assert len(images[0]) == 1
                pixel = images[0][0].getpixel((0, 0))
                ids = [1] + [ord(char) % 61 + 2 for char in text[0][::13]]
                return {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[1] + [0] * (len(ids) - 1)]),
                    "position_ids": torch.arange(len(ids), dtype=torch.long).view(1, -1),
                    "image_grid_thw": torch.tensor([[1, 2, 3]]),
                    "pixel_values": torch.tensor([[float(pixel[0]), float(pixel[2])]]),
                }

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.forward_kwargs = None
                self.generate_kwargs = None
                self.generated_under_grad = None

            def forward(self, **kwargs):
                self.forward_kwargs = kwargs
                ids = kwargs["input_ids"]
                positions = torch.arange(ids.shape[1], dtype=torch.float32).view(1, -1)
                pixel = kwargs["pixel_values"][0, 0]
                logits = torch.stack((positions + pixel, ids.float(),
                                      torch.ones_like(positions) * self.weight), dim=-1)
                return SimpleNamespace(logits=logits)

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                self.generated_under_grad = torch.is_grad_enabled()
                suffix = torch.tensor([[31, 32]], dtype=torch.long, device=kwargs["input_ids"].device)
                return torch.cat((kwargs["input_ids"], suffix), dim=1)

        self.processor = Processor()
        self.student = Model()
        self.teacher = Model()
        self.runtime = object.__new__(VLMRuntime)
        self.runtime.device = torch.device("cpu")
        self.runtime.processor = self.processor
        self.runtime.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)
        self.runtime.student = self.student
        self.runtime.teacher = self.teacher
        self.runtime.eos_token_ids = (99, 100)
        self.example = ProxyExample(
            id="paired", split="train", student_image=self.student_path,
            teacher_image=self.teacher_path,
            question="Which letter is marked in the red box? A. first B. second",
            teacher_question="Which letter appears? A. first B. second",
            teacher_evidence={"objects": [{"shape": "square"}]},
            accepted_answers=("SECRET_LABEL",),
        )

    def test_separate_teacher_prompt_same_student_encoding_and_image_kwargs(self):
        student = self.runtime.encode_student(self.example)
        teacher = self.runtime.encode_teacher(self.example)
        baseline = self.runtime.encode_student(self.example)
        self.assertTrue(self.torch.equal(student.input_ids, baseline.input_ids))
        self.assertEqual(student.sha256, baseline.sha256)
        self.assertNotEqual(student.sha256, teacher.sha256)
        self.assertNotEqual(len(student.input_ids), len(teacher.input_ids))
        student_text = self.processor.calls[0]["text"][0]
        teacher_text = self.processor.calls[1]["text"][0]
        self.assertEqual([message["role"] for message in self.processor.messages[0]],
                         ["system", "user"])
        self.assertEqual(self.processor.messages[0][1]["content"][0]["type"], "image")
        self.assertIn("Start with <vision>", student_text)
        self.assertNotIn("SECRET_LABEL", student_text)
        self.assertNotIn("SECRET_LABEL", teacher_text)
        self.assertNotIn("objects", student_text)
        self.assertIn("objects", teacher_text)
        self.assertIn("Which letter appears?", teacher_text)
        self.assertNotIn("red box?", teacher_text)
        self.assertEqual(len(self.processor.calls[0]["images"][0]), 1)
        self.assertEqual(student.inputs["pixel_values"].tolist(), [[255.0, 0.0]])
        self.assertEqual(teacher.inputs["pixel_values"].tolist(), [[0.0, 255.0]])

    def test_generate_keeps_exact_ids_and_forward_aligns_different_prefix_lengths(self):
        student = self.runtime.encode_student(self.example)
        teacher = self.runtime.encode_teacher(self.example)
        completion = self.runtime.generate(student, {
            "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0,
            "top_k": 0, "seed": 7,
        })
        self.assertEqual(completion.tolist(), [31, 32])
        self.assertEqual(completion.device.type, "cpu")
        self.assertFalse(self.student.generated_under_grad)
        self.assertTrue(self.student.training)
        self.assertEqual(self.student.generate_kwargs["image_grid_thw"].tolist(), [[1, 2, 3]])
        self.assertEqual(self.student.generate_kwargs["mm_token_type_ids"].shape[1], len(student.input_ids))
        selected_student = self.runtime.forward_response(self.student, student, completion)
        with self.torch.no_grad():
            selected_teacher = self.runtime.forward_response(self.teacher, teacher, completion)
        self.assertEqual(selected_student.shape, (2, 3))
        self.assertEqual(selected_teacher.shape, (2, 3))
        self.assertEqual(selected_student[:, 0].tolist(),
                         [float(len(student.input_ids) - 1 + 255), float(len(student.input_ids) + 255)])
        self.assertEqual(selected_teacher[:, 0].tolist(),
                         [float(len(teacher.input_ids) - 1), float(len(teacher.input_ids))])
        self.assertEqual(self.student.forward_kwargs["mm_token_type_ids"][0, -2:].tolist(), [0, 0])
        self.assertEqual(self.student.forward_kwargs["position_ids"][0, -2:].tolist(),
                         [len(student.input_ids), len(student.input_ids) + 1])
        self.assertEqual(self.student.forward_kwargs["image_grid_thw"].tolist(), [[1, 2, 3]])
        self.assertFalse(self.student.forward_kwargs["use_cache"])
        selected_student.sum().backward()
        self.assertIsNotNone(self.student.weight.grad)
        self.assertGreater(float(self.student.weight.grad), 0)
        self.assertFalse(selected_teacher.requires_grad)

    def test_forward_rejects_changed_prompt_ids(self):
        encoded = self.runtime.encode_student(self.example)
        encoded.inputs["input_ids"][0, 0] = 42
        with self.assertRaisesRegex(RuntimeError, "changed since generation"):
            self.runtime.forward_response(self.student, encoded,
                                          self.torch.tensor([31], dtype=self.torch.long))


if __name__ == "__main__":
    unittest.main()
