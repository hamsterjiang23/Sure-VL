"""Small-model-free tensor checks of the internal-proxy GOLD adapter."""

import json
import tempfile
import unittest
from collections import defaultdict
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from sure_vl.proxy_trainer import ProxyGOLDTrainer, _TRAIN_IMPORT_ERROR


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


def _payload():
    return json.dumps({
        "id": "x", "split": "train", "student_image": "student.png",
        "teacher_image": "clear.png", "question": "What color?",
        "accepted_answers": ["yes"],
    })


def _completion():
    return (
        "<vision>A</vision>"
        "<answer>yes</answer><confidence>"
        "<visual_confidence>8</visual_confidence>"
        "<answer_confidence>9</answer_confidence></confidence>"
    )


@unittest.skipIf(torch is None or _TRAIN_IMPORT_ERROR is not None, "torch/TRL train dependencies are absent")
class ProxyTrainerTensorTests(unittest.TestCase):
    def _stub(self, *, min_vision_tokens=1, opsd_weight=1.0):
        stub = SimpleNamespace()
        stub._tokenizer = CharacterTokenizer()
        stub.generation_config = SimpleNamespace(eos_token_id=0)
        stub.proxy_config = {
            "alpha": 0.5, "tau_s": 0.5, "lambda_b": 0.0,
            "min_vision_tokens": min_vision_tokens, "chunk_size": 8,
        }
        stub.reward_config = {
            "answer_utility": 1.0, "rho_answer": 1.0,
            "rho_visual": 1.0, "format_penalty": 1.0,
        }
        stub.policy_weight = 1.0
        stub.opsd_weight = opsd_weight
        stub.opsd_temperature = 1.1
        stub.opsd_token_clip = 0.05
        stub.diagnostic_tokens = 2
        stub._metrics = {"train": defaultdict(list)}
        stub._get_model_forward_kwargs = lambda _: {}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        stub.args = SimpleNamespace(output_dir=temporary.name)
        stub.state = SimpleNamespace(global_step=0)
        stub.accelerator = SimpleNamespace(process_index=0)
        stub.teacher_calls = []

        def teacher_logits(row, content_ids):
            stub.teacher_calls.append((row["teacher_image"], content_ids.detach().clone()))
            logits = torch.zeros((len(content_ids), 128))
            logits[:, ord("A")] = 2.0
            return logits

        stub._teacher_logits_for_content = teacher_logits
        stub.measure_rollout = lambda row, completion, logits, diagnostics=False: (
            ProxyGOLDTrainer.measure_rollout(stub, row, completion, logits, diagnostics=diagnostics)
        )
        return stub

    @staticmethod
    def _row():
        return {"example_payload": _payload(), "prompt": [],
                "student_image": "student.png", "teacher_image": "clear.png"}

    @staticmethod
    def _ids(text):
        return torch.tensor([ord(character) for character in text] + [0], dtype=torch.long)

    def test_same_prefix_vision_only_proxy_and_content_only_teacher(self):
        completion = _completion()
        ids = self._ids(completion)
        logits = torch.zeros((len(ids), 128), requires_grad=True)
        stub = self._stub()
        measured = stub.measure_rollout(self._row(), ids, logits)
        report_start = completion.index("<confidence>")
        vision_start = completion.index("<vision>") + len("<vision>")
        self.assertEqual(measured.record["vision_tokens"], 1)
        self.assertEqual(measured.content_mask.tolist()[:report_start], [True] * report_start)
        self.assertEqual(measured.content_mask.tolist()[report_start:], [False] * (len(ids) - report_start))
        self.assertEqual(measured.report_mask.tolist()[report_start:], [True] * (len(ids) - report_start))
        self.assertEqual(stub.teacher_calls[0][0], "clear.png")
        self.assertEqual(stub.teacher_calls[0][1].tolist(), [ord(char) for char in completion[:report_start]])
        self.assertEqual(len(stub.teacher_calls), 1)
        self.assertFalse(measured.record["proxy_fallback"])
        self.assertGreater(measured.record["proxy_components"]["raw_js"], 0)
        self.assertEqual(measured.record["answer_correct"], True)
        self.assertEqual(measured.record["answer_label_available"], True)
        self.assertAlmostEqual(measured.record["reward"]["answer_score"], -0.01)
        self.assertAlmostEqual(
            measured.content_reward,
            measured.record["reward"]["utility"] + measured.report_reward,
        )
        self.assertIsInstance(measured.record["visual_proxy"], float)
        measured.opsd_loss_sum.backward()
        self.assertGreater(logits.grad[:report_start].abs().sum().item(), 0)
        self.assertEqual(logits.grad[report_start:].abs().sum().item(), 0)
        self.assertEqual(vision_start, completion.index("A</vision>"))

    def test_baseline_teacher_uses_restricted_view_and_same_vision_prefix(self):
        completion = _completion()
        ids = self._ids(completion)
        student_logits = torch.zeros((len(ids), 128), requires_grad=True)
        stub = self._stub(opsd_weight=0.0)
        stub.proxy_config["lambda_b"] = 1.0

        def teacher_logits(row, content_ids):
            image = row["teacher_image"]
            stub.teacher_calls.append((image, content_ids.detach().clone()))
            logits = torch.zeros((len(content_ids), 128))
            logits[:, ord("A")] = 2.0 if image == "clear.png" else 1.0
            return logits

        stub._teacher_logits_for_content = teacher_logits
        measured = stub.measure_rollout(self._row(), ids, student_logits)
        self.assertEqual(len(stub.teacher_calls), 2)
        clear_view, clear_ids = stub.teacher_calls[0]
        restricted_view, restricted_ids = stub.teacher_calls[1]
        self.assertEqual(clear_view, "clear.png")
        self.assertEqual(restricted_view, "student.png")
        self.assertEqual(
            clear_ids.tolist(),
            [ord(char) for char in completion.split("<confidence>")[0]],
        )
        vision_last = completion.index("<vision>") + len("<vision>") + len("A")
        self.assertEqual(
            restricted_ids.tolist(),
            [ord(char) for char in completion[:vision_last]],
        )
        proxy = measured.record["proxy_components"]
        self.assertGreater(proxy["raw_js"], proxy["baseline_js"])
        self.assertGreater(proxy["baseline_js"], 0.0)
        self.assertAlmostEqual(
            proxy["corrected_gap"],
            max(0.0, min(1.0, proxy["raw_js"] - proxy["baseline_js"])),
            places=6,
        )
        self.assertTrue(0.0 <= measured.record["visual_proxy"] <= 1.0)
        self.assertIsInstance(measured.record["visual_proxy"], float)
        measured.opsd_loss_sum.backward()
        self.assertEqual(student_logits.grad.abs().sum().item(), 0.0)

    def test_missing_confidence_keeps_content_opsd_and_penalizes_reports(self):
        completion = _completion().split("<confidence>")[0]
        ids = self._ids(completion)
        stub = self._stub()
        measured = stub.measure_rollout(self._row(), ids, torch.zeros((len(ids), 128), requires_grad=True))
        self.assertIn("missing_confidence", measured.record["format_errors"])
        self.assertEqual(measured.record["answer_correct"], True)
        self.assertEqual(measured.record["answer_confidence"], None)
        self.assertEqual(measured.record["visual_confidence"], None)
        self.assertEqual(measured.content_token_count, len(ids) - 1)
        self.assertEqual(len(stub.teacher_calls), 1)
        self.assertEqual(stub.teacher_calls[0][1].tolist(), [ord(char) for char in completion])
        self.assertAlmostEqual(measured.record["reward"]["answer_score"], -1.0)
        self.assertAlmostEqual(measured.record["reward"]["visual_score"], -1.0)
        self.assertAlmostEqual(measured.record["reward"]["format_penalty"], 1.0)
        self.assertAlmostEqual(measured.report_reward, -3.0)
        self.assertAlmostEqual(measured.content_reward, -2.0)

    def test_short_vision_fallback_and_missing_answer_label_are_explicit(self):
        completion = _completion().replace("<answer>yes</answer>", "<answer></answer>")
        ids = self._ids(completion)
        stub = self._stub(min_vision_tokens=8)
        measured = stub.measure_rollout(self._row(), ids, torch.zeros((len(ids), 128)))
        self.assertEqual(measured.record["vision_tokens"], 1)
        self.assertTrue(measured.record["proxy_fallback"])
        self.assertEqual(measured.record["visual_proxy"], 0.0)
        self.assertIn("vision_proxy_fallback", measured.record["format_errors"])
        self.assertEqual(measured.record["answer_correct"], False)
        self.assertEqual(measured.record["answer_label_available"], False)
        self.assertAlmostEqual(measured.record["reward"]["answer_score"], -1.0)
        self.assertGreater(measured.content_token_count, 0)
        self.assertEqual(len(stub.teacher_calls), 1)

    def test_missing_vision_does_not_disable_content_teacher(self):
        completion = _completion().replace("<vision>A</vision>", "")
        ids = self._ids(completion)
        stub = self._stub()
        measured = stub.measure_rollout(self._row(), ids, torch.zeros((len(ids), 128)))
        self.assertTrue(measured.record["proxy_fallback"])
        self.assertEqual(measured.record["visual_proxy"], 0.0)
        self.assertEqual(measured.record["vision_tokens"], 0)
        self.assertIn("missing_or_invalid_vision", measured.record["format_errors"])
        self.assertGreater(measured.content_token_count, 0)
        self.assertEqual(len(stub.teacher_calls), 1)
        self.assertEqual(
            stub.teacher_calls[0][1].tolist(),
            [ord(char) for char in completion.split("<confidence>")[0]],
        )
        self.assertTrue(torch.isfinite(measured.opsd_loss_sum))

    def test_opsd_changes_content_gradient_without_touching_report_gradient(self):
        completion = _completion()
        ids = self._ids(completion)
        inputs = {
            "input_ids": torch.cat((torch.tensor([[1]]), ids.unsqueeze(0)), dim=1),
        }
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][0, 0] = -100
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["_sure_vl_proxy_rows"] = [self._row()]

        class Model(torch.nn.Module):
            def __init__(self, length):
                super().__init__()
                self.raw = torch.nn.Parameter(torch.zeros((1, length, 128)))

            def forward(self, **_):
                return SimpleNamespace(logits=self.raw)

        model = Model(inputs["input_ids"].shape[1])
        stub = self._stub(opsd_weight=0.0)
        without_opsd = ProxyGOLDTrainer.compute_loss(stub, model, inputs)
        without_opsd.backward()
        baseline_grad = model.raw.grad.detach().clone()
        model.zero_grad()
        stub.opsd_weight = 1.0
        with_opsd = ProxyGOLDTrainer.compute_loss(stub, model, inputs)
        with_opsd.backward()
        joint_grad = model.raw.grad.detach().clone()
        report_start = completion.index("<confidence>")
        self.assertGreater((joint_grad[0, :report_start] - baseline_grad[0, :report_start]).abs().sum().item(), 0)
        self.assertTrue(torch.allclose(joint_grad[0, report_start:], baseline_grad[0, report_start:]))
        self.assertEqual(stub._metrics["train"]["sure_vl/rank_local/proxy_usable_rows"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
