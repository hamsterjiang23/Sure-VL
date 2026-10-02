"""Small real-Torch checks for the native GRPO segment and OPSD extension."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from trl import GRPOTrainer
    from sure_vl.proxy_protocol import ProxyExample
    from sure_vl.proxy_rollout import PreparedProxyRollout, ScoredProxyRollout
    from sure_vl.training.trl.trainer import (
        ProxyGRPOTrainer, _SampleScore, _global_group_center, _group_center,
    )
except ImportError:
    torch = GRPOTrainer = ProxyExample = PreparedProxyRollout = ScoredProxyRollout = None
    ProxyGRPOTrainer = _SampleScore = _global_group_center = _group_center = None


if torch is not None:
    class CharacterTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(character) for character in text]

        def decode(self, ids, **_kwargs):
            return "".join(chr(int(value)) for value in ids if int(value))

        def __call__(self, text, **_kwargs):
            return {"input_ids": self.encode(text),
                    "offset_mapping": [(index, index + 1) for index in range(len(text))]}


    class TinyCausalModel(torch.nn.Module):
        def __init__(self, offset: float = 0.0):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.head = torch.nn.Linear(8, 128)
            with torch.no_grad():
                self.head.bias[3] += offset

        def forward(self, input_ids, attention_mask=None, logits_to_keep=None, **_kwargs):
            logits = self.head(self.embedding(input_ids))
            if logits_to_keep is not None:
                logits = logits[:, -logits_to_keep:, :]
            return SimpleNamespace(logits=logits)


def _payload() -> str:
    return json.dumps({
        "id": "tiny", "split": "train", "student_image": "student.png",
        "teacher_image": "teacher.png", "question": "Pick B", "accepted_answers": ["B"],
    })


def _fake_scored(ids=(10, 11), content_reward=-3.0, report_reward=-3.0):
    prepared = PreparedProxyRollout(
        completion_ids=ids, content_ids=ids[:1], text="test", parsed=None,
        content_mask=(True, False), report_mask=(False, True),
        vision_mask=(False, False), format_errors=(),
    )
    return ScoredProxyRollout(
        prepared=prepared,
        record={"id": "tiny", "proxy_fallback": True},
        content_reward=content_reward, report_reward=report_reward,
    )


@unittest.skipIf(torch is None, "Torch and TRL are unavailable")
class ProxyGRPOTests(unittest.TestCase):
    def test_group_center_removes_constant_failure_reward(self):
        values = torch.tensor([-3.0, -3.0, -3.0, -3.0])
        self.assertTrue(torch.equal(_group_center(values, 4, "none"), torch.zeros(4)))
        changing = torch.tensor([-3.0, -2.0, -3.0, 0.0])
        advantage = _group_center(changing, 4, "none")
        self.assertAlmostEqual(float(advantage.sum()), 0.0, places=6)
        self.assertGreater(float(advantage[-1]), 0.0)

    def test_ddp_two_plus_two_rewards_center_one_global_group(self):
        local = torch.tensor([-3.0, -2.0])
        remote = torch.tensor([-1.0, -3.0])
        full = torch.cat((local, remote))
        expected = full - full.mean()
        for rank, rank_rewards in ((0, local), (1, remote)):
            accelerator = SimpleNamespace(
                num_processes=2, process_index=rank,
                gather=lambda values, combined=full: combined,
            )
            subset, global_advantages, global_rewards = _global_group_center(
                rank_rewards, accelerator, 4, "none", training=True,
            )
            self.assertTrue(torch.equal(global_rewards, full))
            self.assertTrue(torch.equal(global_advantages, expected))
            self.assertTrue(torch.equal(subset, expected[rank * 2:(rank + 1) * 2]))
        self.assertAlmostEqual(float(expected.sum()), 0.0, places=6)

    def test_native_batch_keys_remain_and_two_segment_advantages_split(self):
        trainer = object.__new__(ProxyGRPOTrainer)
        trainer.model = TinyCausalModel().train()
        trainer.num_generations = 4
        trainer.num_generations_eval = 1
        trainer.accelerator = SimpleNamespace(num_processes=1, process_index=0)
        trainer.scale_rewards = "none"
        trainer.loss_type = "grpo"
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer._latest_sample_scores = None
        trainer.last_rollout_records = []
        content_rewards = [-3.0, -3.0, -2.0, -2.0]
        report_rewards = [-3.0, -2.0, -3.0, -2.0]
        scores = [
            _SampleScore(_payload(), (10, 11), (1, 2), 0,
                         _fake_scored(content_reward=content_rewards[i], report_reward=report_rewards[i]))
            for i in range(4)
        ]
        batch = {
            "prompt_ids": torch.tensor([[1, 2]] * 4),
            "prompt_mask": torch.ones((4, 2), dtype=torch.long),
            "completion_ids": torch.tensor([[10, 11]] * 4),
            "completion_mask": torch.ones((4, 2), dtype=torch.long),
            "mm_token_type_ids": torch.zeros((4, 4), dtype=torch.long),
            "advantages": torch.zeros(4),
        }

        def upstream(_self, _inputs):
            trainer._latest_sample_scores = scores
            return batch

        with patch.object(GRPOTrainer, "_generate_and_score_completions", upstream):
            result = ProxyGRPOTrainer._generate_and_score_completions(trainer, [{}] * 4)
        self.assertIs(result, batch)
        self.assertIn("mm_token_type_ids", result)
        self.assertEqual(result["advantages"].shape, (4, 2))
        self.assertTrue(torch.equal(result["advantages"][:, 0], torch.tensor([-0.5, -0.5, 0.5, 0.5])))
        self.assertTrue(torch.equal(result["advantages"][:, 1], torch.tensor([-0.5, 0.5, -0.5, 0.5])))
        self.assertEqual(len(result["_proxy_example_payloads"]), 4)

    def test_same_student_forward_supplies_content_only_kl_gradient(self):
        student = TinyCausalModel().train()
        teacher = TinyCausalModel(offset=3.0).eval()
        trainer = object.__new__(ProxyGRPOTrainer)
        trainer.model = student
        trainer.teacher_model = teacher
        trainer.opsd_weight = 0.7
        trainer.loss_config = {"opsd_temperature": 1.1, "opsd_token_clip": 0.05}
        trainer.model_kwarg_keys = {"logits_to_keep"}
        trainer.current_gradient_accumulation_steps = 4
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer._teacher_version = lambda: 0
        trainer._teacher_messages = lambda _example: []
        trainer._encode_messages = lambda _messages: {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        trainer._record_training_attempt = lambda _inputs, **_kwargs: None
        logits_seen = []

        def upstream(_self, model, inputs, return_outputs=False, num_items_in_batch=None):
            output = model(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                           logits_to_keep=3)
            output.logits.retain_grad()
            logits_seen.append(output.logits)
            return output.logits.sum() * 0.0  # isolate the added KL gradient

        inputs = {
            "_proxy_example_payloads": [_payload()],
            "_proxy_teacher_version": torch.tensor([0]),
            "_proxy_content_mask": torch.tensor([[True, False]]),
            "completion_ids": torch.tensor([[3, 4]]),
            "completion_mask": torch.tensor([[True, True]]),
            "advantages": torch.tensor([[1.0, 0.0]]),
        }
        with patch.object(GRPOTrainer, "compute_loss", upstream):
            loss = ProxyGRPOTrainer.compute_loss(trainer, student, inputs)
        self.assertTrue(math.isfinite(float(loss)))
        loss.backward()
        self.assertEqual(len(logits_seen), 1)
        self.assertGreater(float(logits_seen[0].grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits_seen[0].grad[0, 1:].abs().sum()), 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_scoring_uses_actual_ids_and_same_student_view_for_q_minus(self):
        student = TinyCausalModel().train()
        teacher = TinyCausalModel(offset=2.0).eval()
        student.generation_config = SimpleNamespace(eos_token_id=0)
        trainer = object.__new__(ProxyGRPOTrainer)
        trainer.model = student
        trainer.teacher_model = teacher
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
        trainer.processing_class = SimpleNamespace(apply_chat_template=lambda **kwargs: {
            "input_ids": torch.tensor([[1, 3 if kwargs["conversation"][0][0]["role"] == "teacher" else 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        })
        trainer.tools = []
        trainer.chat_template = None
        trainer.chat_template_kwargs = {}
        trainer._tokenizer = CharacterTokenizer()
        trainer.generation_config = SimpleNamespace(eos_token_id=0)
        trainer.model_kwarg_keys = {"logits_to_keep"}
        trainer.opsd_weight = 1.0
        trainer.loss_config = {"opsd_temperature": 1.1, "opsd_token_clip": 0.05,
                               "diagnostic_tokens": 4}
        trainer.proxy_config = {"alpha": 0.5, "tau_s": 0.5, "lambda_b": 1.0,
                                "min_vision_tokens": 1, "chunk_size": 8}
        trainer.reward_config = {"answer_utility": 1.0, "rho_answer": 1.0,
                                 "rho_visual": 1.0, "format_penalty": 1.0}
        trainer._teacher_messages = lambda _example: [{"role": "teacher", "content": "clear"}]
        example = ProxyExample(
            id="tiny", split="train", student_image="student.png", teacher_image="teacher.png",
            question="Pick B", accepted_answers=("B",),
        )
        text = ("<vision>blue</vision><answer>B</answer><confidence>"
                "<visual_confidence>8</visual_confidence>"
                "<answer_confidence>9</answer_confidence></confidence>")
        ids = tuple(ord(character) for character in text) + (0,)
        scored, student_prompt_ids = trainer._score_one(
            example, [{"role": "user", "content": "student"}], ids,
        )
        self.assertEqual(student_prompt_ids, (1, 2))
        self.assertEqual(scored.prepared.completion_ids, ids)
        self.assertFalse(scored.record["proxy_fallback"])
        self.assertGreater(scored.record["vision_tokens"], 0)
        self.assertTrue(scored.record["answer_correct"])
        self.assertEqual(scored.record["answer_confidence_score"], 9)
        self.assertEqual(scored.record["teacher_conditioning"]["baseline_input"]["prompt_tokens"], 2)
        self.assertGreater(scored.record["opsd"]["sampled_positions"], 0)
        self.assertTrue(math.isfinite(scored.record["opsd"]["weighted_logit_grad_l2"]))
        self.assertTrue(student.training)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
