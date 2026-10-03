"""Detached scoring and content KL checks for the veRL group Worker."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from sure_vl.proxy_protocol import ProxyExample
from sure_vl.trl_distillation import opsd_content_loss
from sure_vl.training.verl.objective import (
    prepare_proxy_rollout,
    score_proxy_rollout,
)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(value)) for value in ids if int(value))

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def _example():
    return ProxyExample(
        id="x", split="train", student_image="student.png", teacher_image="crop.png",
        question="Is it blue?", accepted_answers=("yes",),
    )


def _ids(text=None):
    if text is None:
        text = ("<vision>A</vision><reason>R</reason><answer>yes</answer><confidence>"
                "<visual_confidence>8</visual_confidence>"
                "<answer_confidence>9</answer_confidence></confidence>")
    return [ord(character) for character in text] + [0]


def _configs(lambda_b=1.0, min_tokens=1):
    return (
        {"alpha": 0.5, "tau_s": 0.5, "lambda_b": lambda_b,
         "min_vision_tokens": min_tokens, "chunk_size": 8},
        {"answer_utility": 1.0, "rho_answer": 1.0,
         "rho_visual": 1.0, "format_penalty": 1.0},
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class VerlObjectiveTests(unittest.TestCase):
    def test_q_minus_uses_same_prefix_and_content_only_kl_gradient(self):
        ids = _ids()
        prepared = prepare_proxy_rollout(_example(), CharacterTokenizer(), ids, eos_token_ids=[0])
        vision_position = prepared.vision_positions[0]
        reason_position = prepared.text.index("<reason>") + len("<reason>")
        self.assertTrue(prepared.parsed.format_valid)
        self.assertEqual(prepared.parsed.reasoning_text, "R")
        self.assertTrue(prepared.content_mask[reason_position])
        self.assertFalse(prepared.vision_mask[reason_position])
        self.assertEqual(prepared.vision_positions, (prepared.text.index("<vision>") + len("<vision>"),))
        self.assertEqual(prepared.content_count, "".join(map(chr, ids[:-1])).index("<confidence>"))
        self.assertTrue(prepared.report_mask[-1])  # generated EOS is a report action
        student = torch.zeros(len(ids), 128, requires_grad=True)
        clear = torch.zeros(prepared.content_count, 128, requires_grad=True)
        clear.data[vision_position, ord("A")] = 3.0
        clear.data[reason_position, ord("R")] = 3.0
        last = vision_position + 1
        restricted = student.detach()[:last].clone()
        proxy, reward = _configs()
        scored = score_proxy_rollout(
            _example(), prepared, student, clear, restricted,
            proxy_config=proxy, reward_config=reward,
        )
        self.assertAlmostEqual(scored.record["proxy_components"]["baseline_js"], 0.0, places=7)
        self.assertGreater(scored.record["proxy_components"]["raw_js"], 0.0)
        self.assertEqual(scored.record["answer_confidence_score"], 9)
        self.assertAlmostEqual(scored.record["reward"]["answer_score"], -0.01)
        self.assertTrue(math.isfinite(scored.record["visual_proxy"]))
        self.assertFalse(scored.record["proxy_fallback"])
        clear_without_reason = clear.detach().clone()
        clear_without_reason[reason_position].zero_()
        scored_without_reason = score_proxy_rollout(
            _example(), prepared, student.detach(), clear_without_reason, restricted,
            proxy_config=proxy, reward_config=reward,
        )
        self.assertEqual(scored.record["visual_proxy"], scored_without_reason.record["visual_proxy"])

        # Isolate teacher KL: its full-vocabulary gradient must reach content
        # but neither report tokens nor the frozen teacher logits.
        opsd = opsd_content_loss(
            student[:prepared.content_count].unsqueeze(0), clear.detach().unsqueeze(0),
            torch.ones((1, prepared.content_count), dtype=torch.bool),
            beta=0.0, temperature=1.1, pointwise_clip=0.05,
            reduction="token_mean",
        )
        opsd.backward()
        self.assertGreater(float(student.grad[:prepared.content_count].abs().sum()), 0.0)
        self.assertGreater(float(student.grad[reason_position].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[prepared.content_count:].abs().sum()), 0.0)
        self.assertIsNone(clear.grad)

    def test_malformed_report_keeps_teacher_content_and_explicit_fallback(self):
        ids = _ids("<vision>A</vision><answer>yes</answer>")
        prepared = prepare_proxy_rollout(_example(), CharacterTokenizer(), ids)
        self.assertEqual(prepared.content_count, len(ids) - 1)
        self.assertIn("missing_confidence", prepared.format_errors)
        student = torch.zeros(len(ids), 128, requires_grad=True)
        clear = torch.zeros(prepared.content_count, 128)
        proxy, reward = _configs(lambda_b=0.0, min_tokens=8)
        scored = score_proxy_rollout(
            _example(), prepared, student, clear, None,
            proxy_config=proxy, reward_config=reward,
        )
        self.assertTrue(scored.record["proxy_fallback"])
        self.assertEqual(scored.record["visual_proxy"], 0.0)
        self.assertEqual(scored.record["visual_confidence"], None)
        self.assertEqual(scored.record["reward"]["visual_score"], -1.0)
        self.assertEqual(scored.record["reward"]["answer_score"], -1.0)
        self.assertEqual(scored.record["reward"]["format_penalty"], 1.0)
        self.assertEqual(scored.content_reward, -2.0)
        self.assertEqual(scored.report_reward, -3.0)
        opsd = opsd_content_loss(
            student[:prepared.content_count].unsqueeze(0), clear.detach().unsqueeze(0),
            torch.ones((1, prepared.content_count), dtype=torch.bool),
            beta=0.0, temperature=1.1, pointwise_clip=0.05,
            reduction="token_mean",
        )
        opsd.backward()
        self.assertGreater(float(student.grad[:prepared.content_count].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[prepared.content_count:].abs().sum()), 0.0)

    def test_known_gt_missing_answer_uses_y_zero_and_option_text_is_noncanonical(self):
        example = ProxyExample(
            id="choice", split="train", student_image="student.png", teacher_image="crop.png",
            question="A. cat\nB. harness\nC. shoe\nD. boat", accepted_answers=("B",),
        )
        proxy, reward = _configs(lambda_b=0.0)
        missing = _ids(
            "<vision>A harness is visible</vision><reason>The shape matches a harness.</reason><confidence>"
            "<visual_confidence>8</visual_confidence>"
            "<answer_confidence>9</answer_confidence></confidence>"
        )
        prepared = prepare_proxy_rollout(example, CharacterTokenizer(), missing)
        scored = score_proxy_rollout(
            example, prepared, torch.zeros(len(missing), 128),
            torch.zeros(prepared.content_count, 128), None,
            proxy_config=proxy, reward_config=reward,
        )
        self.assertTrue(scored.record["ground_truth_available"])
        self.assertTrue(scored.record["answer_label_available"])
        self.assertFalse(scored.record["parsed_answer_available"])
        self.assertFalse(scored.record["answer_correct"])
        self.assertAlmostEqual(scored.record["reward"]["answer_score"], -0.81)

        expanded = _ids(
            "<vision>A harness is visible</vision><reason>The shape matches a harness.</reason>"
            "<answer>B. harness</answer><confidence>"
            "<visual_confidence>8</visual_confidence>"
            "<answer_confidence>9</answer_confidence></confidence>"
        )
        prepared = prepare_proxy_rollout(example, CharacterTokenizer(), expanded)
        scored = score_proxy_rollout(
            example, prepared, torch.zeros(len(expanded), 128),
            torch.zeros(prepared.content_count, 128), None,
            proxy_config=proxy, reward_config=reward,
        )
        self.assertTrue(scored.record["answer_correct"])
        self.assertFalse(scored.record["answer_format_canonical"])
        self.assertIn("noncanonical_option_answer", scored.record["format_errors"])


if __name__ == "__main__":
    unittest.main()
