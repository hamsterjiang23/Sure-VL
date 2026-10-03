"""Teacher privilege routing and exact-prefix causal alignment checks."""

import json
import unittest
from types import SimpleNamespace

from sure_vl.proxy_trainer import ProxyGOLDTrainer, proxy_teacher_row, _TRAIN_IMPORT_ERROR

try:
    import torch
except ImportError:
    torch = None


def _row():
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": "Student guidance"}]},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "Student question"}]},
        ],
        "student_image": "full.png", "teacher_image": "crop.png",
        "example_payload": json.dumps({
            "id": "one", "split": "train", "student_image": "full.png",
            "teacher_image": "crop.png", "question": "Which color?",
            "accepted_answers": ["secret-answer-do-not-leak"],
            "teacher_evidence": {"scene_graph": {"objects": [{"color": "blue"}]}},
        }),
    }


class TeacherRoutingTests(unittest.TestCase):
    def test_privilege_changes_teacher_prompt_only_and_does_not_leak_label(self):
        row = _row()
        original = json.dumps(row, sort_keys=True)
        teacher = proxy_teacher_row(row)
        self.assertEqual(teacher["teacher_image"], "crop.png")
        self.assertNotEqual(teacher["prompt"], row["prompt"])
        self.assertEqual([message["role"] for message in teacher["prompt"]], ["system", "user"])
        self.assertEqual([part["type"] for part in teacher["prompt"][-1]["content"]],
                         ["image", "text"])
        self.assertTrue(all(part["type"] == "text" for part in teacher["prompt"][0]["content"]))
        teacher_system = json.dumps(teacher["prompt"][0])
        teacher_user = json.dumps(teacher["prompt"][-1])
        self.assertIn("regional view", teacher_system)
        self.assertIn("scene_graph", teacher_user)
        self.assertNotIn("scene_graph", teacher_system)
        self.assertNotIn("regional view", teacher_user)
        prompt = json.dumps(teacher["prompt"])
        self.assertIn("scene_graph", prompt)
        self.assertIn("blue", prompt)
        self.assertNotIn("secret-answer-do-not-leak", prompt)
        self.assertNotIn("scene_graph", json.dumps(row["prompt"]))
        self.assertEqual(original, json.dumps(row, sort_keys=True))

    def test_same_view_baseline_preserves_exact_student_prompt_and_no_evidence(self):
        row = _row()
        row["_proxy_teacher_privileged"] = False
        teacher = proxy_teacher_row(row)
        self.assertIs(teacher["prompt"], row["prompt"])
        self.assertEqual([message["role"] for message in teacher["prompt"]], ["system", "user"])
        self.assertEqual(teacher["teacher_image"], "full.png")
        self.assertNotIn("scene_graph", json.dumps(teacher["prompt"]))


@unittest.skipIf(torch is None or _TRAIN_IMPORT_ERROR is not None, "torch/TRL dependencies absent")
class TeacherCausalTests(unittest.TestCase):
    def _trainer(self):
        trainer = object.__new__(ProxyGOLDTrainer)
        trainer._SEQUENCE_KEYS = ("mm_token_type_ids",)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        seen = {}

        def extract(rows):
            seen["image"] = rows[0]["image"]
            return [[rows[0]["image"]]], [rows[0]["prompt"]]

        class Processor:
            def apply_chat_template(self, prompts, **kwargs):
                seen["prompt"] = prompts[0]
                return [json.dumps(prompts[0])]

            def __call__(self, **kwargs):
                # Distinct teacher template changes prompt length. Completion
                # IDs must still be appended verbatim, never tokenized again.
                prefix = [7, 8, 9, 10] if "scene_graph" in kwargs["text"][0] else [7, 8]
                return {"input_ids": torch.tensor([prefix]),
                        "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long),
                        "mm_token_type_ids": torch.ones((1, len(prefix)), dtype=torch.long),
                        "pixel_values": torch.tensor([[2.0]])}

        class Teacher(torch.nn.Module):
            def forward(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(logits=torch.arange(kwargs["input_ids"].numel() * 4)
                                       .reshape(1, -1, 4).float())

        trainer._extract_images_and_prompts = extract
        trainer._get_model_forward_kwargs = lambda values, exclude=(): {"pixel_values": values["pixel_values"]}
        trainer.processing_class = Processor()
        trainer.teacher_model = Teacher()
        return trainer, seen

    def test_different_teacher_prompt_keeps_actual_ids_and_causal_positions(self):
        trainer, seen = self._trainer()
        logits = trainer._teacher_logits_for_content(_row(), torch.tensor([11, 12]))
        self.assertEqual(seen["image"], "crop.png")
        self.assertEqual(seen["input_ids"].tolist(), [[7, 8, 9, 10, 11, 12]])
        self.assertEqual(seen["mm_token_type_ids"].tolist(), [[1, 1, 1, 1, 0, 0]])
        self.assertEqual(logits.tolist(), [[12, 13, 14, 15], [16, 17, 18, 19]])
        self.assertFalse(logits.requires_grad)
        self.assertEqual(trainer._proxy_last_teacher_input, {"prompt_tokens": 4, "sampled_prefix_tokens": 2})

    def test_reason_tokens_are_appended_verbatim_to_teacher_content_prefix(self):
        trainer, seen = self._trainer()
        content = "<vision>A</vision><reason>Because A is visible.</reason><answer>B</answer>"
        content_ids = torch.tensor([ord(character) for character in content])
        logits = trainer._teacher_logits_for_content(_row(), content_ids)
        self.assertEqual(seen["input_ids"].tolist(), [[7, 8, 9, 10, *content_ids.tolist()]])
        self.assertEqual(seen["mm_token_type_ids"].tolist(), [[1, 1, 1, 1] + [0] * len(content_ids)])
        self.assertEqual(logits.shape, (len(content_ids), 4))
        self.assertEqual(logits[0].tolist(), [12, 13, 14, 15])
        reason_position = content.index("Because A is visible.")
        self.assertEqual(logits[reason_position].tolist(),
                         [4 * (3 + reason_position) + value for value in range(4)])
        self.assertEqual(trainer._proxy_last_teacher_input,
                         {"prompt_tokens": 4, "sampled_prefix_tokens": len(content_ids)})

    def test_baseline_rejects_changed_student_prompt_ids(self):
        trainer, seen = self._trainer()
        row = dict(_row(), _proxy_teacher_privileged=False,
                   _student_prompt_ids=torch.tensor([7, 8]))
        trainer._teacher_logits_for_content(row, torch.tensor([11, 12]))
        self.assertEqual(seen["image"], "full.png")
        self.assertEqual(seen["input_ids"].tolist(), [[7, 8, 11, 12]])
        row["_student_prompt_ids"] = torch.tensor([7, 9])
        with self.assertRaisesRegex(RuntimeError, "exact encoded student prompt"):
            trainer._teacher_logits_for_content(row, torch.tensor([11, 12]))


if __name__ == "__main__":
    unittest.main()
