"""Real Torch updates and optional native Ray RPC for the custom veRL Worker."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "verl"
if VENDOR_ROOT.is_dir():
    sys.path.insert(0, str(VENDOR_ROOT))

try:
    import numpy as np
    import ray
    import torch
    from verl.protocol import DataProto
    from sure_vl.proxy_protocol import ProxyExample
    from sure_vl.trl_distillation import opsd_content_loss
    from sure_vl.training.verl.worker import SureVLProxyWorker
except ImportError:
    np = ray = torch = DataProto = SureVLProxyWorker = None


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(value)) for value in ids if int(value))

    def __call__(self, text, **_kwargs):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


if torch is not None:
    class TinyCausalModel(torch.nn.Module):
        """Predict each sampled ID from the preceding ID through real parameters."""

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.head = torch.nn.Linear(8, 128)

        def response_logits(self, prompt_ids, response_ids):
            context_ids = torch.cat((prompt_ids, response_ids))[:-1]
            hidden = self.embedding(context_ids)[-len(response_ids):]
            return self.head(hidden)


    class TinyRuntime:
        def __init__(self, _config):
            import copy

            torch.manual_seed(4)
            self.student = TinyCausalModel()
            self.teacher = copy.deepcopy(self.student)
            self.processor = SimpleNamespace()
            self.tokenizer = CharacterTokenizer()
            self.eos_token_ids = (0,)
            self.device = torch.device("cpu")
            self.template_sha256 = "tiny-template"
            self.corrupt_next_student_forward = False
            self.student_prompt_hash = "student-view"
            self.last_generate_parameters = None

        def encode_student(self, _example):
            return SimpleNamespace(inputs={}, input_ids=torch.tensor([1, 2]),
                                   sha256=self.student_prompt_hash)

        def encode_teacher(self, _example):
            return SimpleNamespace(inputs={}, input_ids=torch.tensor([1, 3]), sha256="teacher-view")

        def generate(self, _encoded, parameters):
            self.last_generate_parameters = dict(parameters)
            answer = "B" if parameters["seed"] % 2 == 0 else "C"
            detail = "A clear detail appears here" if answer == "B" else "A detail"
            text = (f"<vision>{detail}</vision><answer>{answer}</answer><confidence>"
                    "<visual_confidence>8</visual_confidence>"
                    "<answer_confidence>9</answer_confidence></confidence>")
            return torch.tensor([ord(character) for character in text] + [0], dtype=torch.long)

        def forward_response(self, model, encoded, completion_ids):
            output = model.response_logits(encoded.input_ids, completion_ids)
            if model is self.student and self.corrupt_next_student_forward:
                self.corrupt_next_student_forward = False
                return output * float("nan")
            return output


    class TinyWorker(SureVLProxyWorker):
        def _make_runtime(self):
            return TinyRuntime(self.config)


def _config():
    return {
        "model_id": "tiny-causal-vlm-test",
        "setting": {
            "target_world_size": 1, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4, "generation_batch_size": 4,
            "num_generations": 4, "bf16": False, "fp16": False,
            "rollout_temperature": 1.0, "rollout_top_p": 1.0,
            "max_completion_length": 256, "learning_rate": 1e-3,
            "max_steps": 100,
            "max_grad_norm": 1.0, "seed": 42,
        },
        "proxy": {"alpha": 0.5, "tau_s": 0.5, "lambda_b": 1.0,
                  "min_vision_tokens": 1, "chunk_size": 8},
        "reward": {"answer_utility": 1.0, "rho_answer": 1.0,
                   "rho_visual": 1.0, "format_penalty": 1.0},
        "teacher": {"mode": "ema", "ema_decay": 0.9},
        "loss": {"policy_weight": 1.0, "opsd_weight": 1.0,
                 "opsd_beta": 0.0, "opsd_temperature": 1.1,
                 "opsd_token_clip": 0.05},
        "grpo": {"loss_type": "grpo", "scale_rewards": "none", "num_iterations": 1},
        "validation": {"max_new_tokens": 256, "temperature": 0.6, "top_p": 0.95},
    }


def _payload():
    return {
        "id": "tiny-1", "split": "train", "student_image": "student.png",
        "teacher_image": "crop.png", "question": "Choose the answer",
        "accepted_answers": ["B"],
    }


def _request(seed=42):
    return DataProto.from_dict(
        non_tensors={"example_payload": np.array([_payload()], dtype=object)},
        meta_info={"max_new_tokens": 256, "seed": seed},
    )


@unittest.skipIf(torch is None, "veRL/Torch dependencies are unavailable")
class WorkerLifecycleTests(unittest.TestCase):
    def _worker(self):
        env = {"WORLD_SIZE": "1", "RANK": "0", "MASTER_ADDR": "127.0.0.1",
               "MASTER_PORT": "12345", "WG_BACKEND": "ray"}
        with patch.dict(os.environ, env):
            worker = TinyWorker(_config())
        self.assertEqual(worker.initialize()["world_size"], 1)
        return worker

    def test_real_parameter_update_finite_ema_and_replay_gate(self):
        worker = self._worker()
        initial_student = [parameter.detach().clone() for parameter in worker.runtime.student.parameters()]
        initial_teacher = [parameter.detach().clone() for parameter in worker.runtime.teacher.parameters()]
        generated = worker.generate_sequences(_request())
        self.assertEqual(len(generated), 4)
        self.assertEqual(generated.meta_info["sample_seeds"], [42, 43, 44, 45])
        with self.assertRaisesRegex(RuntimeError, "pending sampled rollout"):
            worker.generate_sequences(_request())
        scored = worker.score_sequences(generated)
        record = json.loads(scored.non_tensor_batch["record_json"][0])
        self.assertEqual(len(scored), 4)
        self.assertGreater(float(scored.batch["content_advantage"].var()), 0.0)
        self.assertAlmostEqual(float(scored.batch["content_advantage"].mean()), 0.0, places=5)
        self.assertTrue(torch.allclose(
            scored.batch["content_advantage"],
            scored.batch["content_reward"] - scored.batch["content_reward"].mean(),
            atol=1e-6,
        ))
        self.assertEqual(record["answer_correct"], True)
        self.assertEqual(record["sampled_completion_ids"],
                         generated.batch["completion_ids"][0, generated.batch["completion_mask"][0]].tolist())
        self.assertEqual(len(record["student_prompt_ids_sha256"]), 64)
        self.assertFalse(record["proxy_fallback"])
        self.assertGreater(record["vision_tokens"], 0)
        self.assertEqual(record["teacher_conditioning"]["baseline_prompt_sha256"], "student-view")
        self.assertEqual(record["teacher_conditioning"]["teacher_prompt_sha256"], "teacher-view")
        pending = worker._scored
        prompt = worker.runtime.encode_student(ProxyExample.from_dict(_payload()))
        expected_policy = expected_opsd = expected_score_function_sum = 0.0
        content_lengths = []
        with torch.no_grad():
            for index, ids in enumerate(pending.completion_ids):
                logits = worker.runtime.forward_response(
                    worker.runtime.student, prompt, torch.tensor(ids, dtype=torch.long)
                )
                logp = torch.log_softmax(logits.float(), -1).gather(
                    -1, torch.tensor(ids).unsqueeze(-1)
                ).squeeze(-1)
                mask = torch.tensor(pending.scored[index].prepared.content_mask)
                advantage = torch.where(
                    mask, torch.tensor(pending.content_advantages[index]),
                    torch.tensor(pending.report_advantages[index]),
                )
                expected_policy += float((-advantage).mean()) / 4
                expected_score_function_sum += float((-logp * advantage).sum())
                count = pending.scored[index].prepared.content_count
                content_lengths.append(count)
                expected_opsd += float(opsd_content_loss(
                    logits[:count].unsqueeze(0),
                    pending.clear_teacher_logits[index].unsqueeze(0),
                    torch.ones((1, count), dtype=torch.bool),
                    beta=0.0, temperature=1.1, pointwise_clip=0.05,
                    reduction="token_mean",
                )) / 4
        self.assertGreater(len(set(content_lengths)), 1)
        result = worker.update_actor(scored)
        records = json.loads(result.non_tensor_batch["group_records_json"][0])
        self.assertEqual(len(records), 4)
        self.assertEqual({item["group_index"] for item in records}, {0, 1, 2, 3})
        self.assertTrue(bool(result.batch["update_succeeded"][0]))
        self.assertTrue(torch.isfinite(result.batch["loss"]).all())
        self.assertAlmostEqual(float(result.batch["policy_loss"][0]), expected_policy, places=5)
        self.assertAlmostEqual(float(result.batch["opsd_loss"][0]), expected_opsd, places=5)
        generated_count = sum(len(ids) for ids in pending.completion_ids)
        self.assertAlmostEqual(
            float(result.batch["score_function_token_mean_diagnostic"][0]),
            expected_score_function_sum / generated_count, places=5,
        )
        self.assertEqual(result.meta_info["evidence"]["optimizer_successful_updates"], 1)
        self.assertEqual(result.meta_info["evidence"]["teacher_ema_updates"], 1)
        self.assertEqual(result.meta_info["evidence"]["adam_max_step"], 1)
        self.assertEqual(result.meta_info["evidence"]["optimizer_state_max_step"], 1)
        self.assertEqual(result.meta_info["evidence"]["scheduler_last_epoch"], 1)
        self.assertEqual(result.meta_info["metrics"]["optimizer_state_max_step"], 1)
        self.assertEqual(result.meta_info["metrics"]["scheduler_last_epoch"], 1)
        self.assertAlmostEqual(result.meta_info["metrics"]["learning_rate_used"], 1e-3)
        self.assertAlmostEqual(result.meta_info["metrics"]["learning_rate_next"], 0.99e-3)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(initial_student, worker.runtime.student.parameters())))
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(initial_teacher, worker.runtime.teacher.parameters())))
        self.assertEqual(result.meta_info["loss_diagnostics"]["content_tokens"],
                         sum(item["content_tokens"] for item in records))
        self.assertIsNotNone(result.meta_info["loss_diagnostics"]["content_mean_nll"])
        self.assertLessEqual(result.meta_info["metrics"]["grad_norm_after_clip"], 1.0 + 1e-6)
        self.assertLessEqual(result.meta_info["metrics"]["grad_norm_after_clip"],
                             result.meta_info["metrics"]["grad_norm"] + 1e-6)
        with self.assertRaisesRegex(RuntimeError, "previously scored"):
            worker.update_actor(scored)
        validation = worker.evaluate(_request(seed=127))
        self.assertEqual(worker.runtime.last_generate_parameters["seed"], 127)
        self.assertEqual(len(validation.non_tensor_batch["record_json"]), 1)
        self.assertEqual(validation.meta_info["evidence"]["optimizer_successful_updates"], 1)
        with tempfile.TemporaryDirectory() as directory:
            saved = worker.save_checkpoint(str(Path(directory) / "step-1"))
            self.assertEqual(saved["evidence"]["adam_max_step"], 1)
            self.assertEqual(saved["optimizer_state_max_step"], 1)
            self.assertTrue((Path(directory) / "step-1" / "manifest.json").is_file())
            self.assertTrue((Path(directory) / "step-1" / "scheduler.pt").is_file())

    def test_nonfinite_student_forward_skips_adam_and_ema(self):
        worker = self._worker()
        scored = worker.score_sequences(worker.generate_sequences(_request()))
        initial_student = [parameter.detach().clone() for parameter in worker.runtime.student.parameters()]
        initial_teacher = [parameter.detach().clone() for parameter in worker.runtime.teacher.parameters()]
        worker.runtime.corrupt_next_student_forward = True
        result = worker.update_actor(scored)
        self.assertFalse(bool(result.batch["update_succeeded"][0]))
        evidence = result.meta_info["evidence"]
        self.assertEqual(evidence["optimizer_attempted_steps"], 1)
        self.assertEqual(evidence["optimizer_successful_updates"], 0)
        self.assertEqual(evidence["optimizer_skipped_updates"], 1)
        self.assertEqual(evidence["teacher_ema_updates"], 0)
        self.assertEqual(evidence["adam_max_step"], 0)
        self.assertEqual(evidence["optimizer_state_max_step"], 0)
        self.assertEqual(evidence["scheduler_last_epoch"], 0)
        self.assertEqual(result.meta_info["metrics"]["optimizer_state_max_step"], 0)
        self.assertEqual(result.meta_info["metrics"]["scheduler_last_epoch"], 0)
        self.assertIsNone(result.meta_info["metrics"]["learning_rate_used"])
        self.assertAlmostEqual(result.meta_info["metrics"]["learning_rate_next"], 1e-3)
        for old, new in zip(initial_student, worker.runtime.student.parameters()):
            self.assertTrue(torch.equal(old, new))
        for old, new in zip(initial_teacher, worker.runtime.teacher.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_image_tensor_digest_must_match_generation_and_update(self):
        worker = self._worker()
        generated = worker.generate_sequences(_request())
        worker.runtime.student_prompt_hash = "mutated-image-tensors"
        with self.assertRaisesRegex(RuntimeError, "prompt or image tensors differ from generation"):
            worker.score_sequences(generated)
        worker.runtime.student_prompt_hash = "student-view"
        scored = worker.score_sequences(generated)
        worker.runtime.student_prompt_hash = "mutated-image-tensors"
        with self.assertRaisesRegex(RuntimeError, "prompt or image differs from sampled prompt"):
            worker.update_actor(scored)


@unittest.skipUnless(os.environ.get("SURE_VL_RUN_RAY_WORKER") == "1", "opt-in Ray Worker RPC")
@unittest.skipIf(torch is None, "veRL/Torch dependencies are unavailable")
class NativeRayWorkerTests(unittest.TestCase):
    def test_actual_ray_dataproto_generation_score_update(self):
        from verl.single_controller.ray.base import (
            RayClassWithInitArgs, RayResourcePool, RayWorkerGroup,
        )

        python_path = os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parent),
            str(Path(__file__).resolve().parents[1] / "src"),
            str(VENDOR_ROOT),
            os.environ.get("PYTHONPATH", ""),
        )))
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": python_path}})
        try:
            pool = RayResourcePool(process_on_nodes=[1], use_gpu=False, max_colocate_count=1)
            group = RayWorkerGroup(
                resource_pool=pool,
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(TinyWorker), config=_config()),
                device_name="cpu",
                worker_env={"PYTHONPATH": python_path},
            )
            self.assertEqual(group.initialize()[0]["world_size"], 1)
            result = group.update_actor(group.score_sequences(group.generate_sequences(_request())))
            self.assertTrue(bool(result.batch["update_succeeded"][0]))
            self.assertEqual(result.meta_info["evidence"]["adam_max_step"], 1)
            self.assertEqual(group.close()[0]["teacher_ema_updates"], 1)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
