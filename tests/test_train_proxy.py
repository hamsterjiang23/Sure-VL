"""Model-free preflight and mocked launcher checks for the proxy method."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sure_vl.train_proxy import (
    DEFAULT_CONFIG,
    _QWEN35_NONTHINKING_SUFFIX,
    _QWEN35_THINKING_SUFFIX,
    _ensure_processor_chat_template,
    _source_provenance,
    configure_generation_terminators,
    configure_nonthinking_template,
    load_plan,
    main,
    run_training,
)


def _example(example_id: str, split: str, student: str, teacher: str) -> dict:
    return {
        "id": example_id,
        "split": split,
        "student_image": student,
        "teacher_image": teacher,
        "question": "What color is the square?",
        "accepted_answers": ["blue"],
    }


class ProxyTrainingPreflightTests(unittest.TestCase):
    def test_copied_source_cannot_inherit_enclosing_clean_git_identity(self):
        with patch("sure_vl.train_proxy.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="/unrelated/enclosing/repository\n",
        )) as git:
            self.assertEqual(_source_provenance(), (None, None))
            self.assertEqual(git.call_count, 1)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("train-blur.png", "train-clear.png", "dev-blur.png", "dev-clear.png"):
            (self.root / name).write_bytes(b"image placeholder")
        self.train = self.root / "train.jsonl"
        self.dev = self.root / "dev.jsonl"
        self.train.write_text(
            json.dumps(_example("train-1", "train", "train-blur.png", "train-clear.png")) + "\n"
        )
        self.dev.write_text(
            json.dumps(_example("dev-1", "dev", "dev-blur.png", "dev-clear.png")) + "\n"
        )
        self.config = self.root / "proxy.json"
        self.settings = json.loads(DEFAULT_CONFIG.read_text())
        self.settings["train_manifest"] = "train.jsonl"
        self.settings["validation_manifest"] = "dev.jsonl"
        self.settings["output_dir"] = "run"
        self._save()

    def _save(self) -> None:
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")

    def test_preflight_freezes_proxy_data_and_does_not_require_visual_labels(self) -> None:
        plan = load_plan(self.config)
        report = plan.summary()
        self.assertEqual(report["method"], "internal_visual_proxy")
        self.assertEqual(report["train_count"], 1)
        self.assertEqual(report["validation_count"], 1)
        self.assertEqual(report["global_batch_size"], 256)
        self.assertEqual(len(report["config_sha256"]), 64)
        self.assertEqual(len(report["train_sha256"]), 64)
        self.assertNotIn("required_visual_facts", plan.train_rows[0]["example_payload"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--config", str(self.config), "--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["train_sha256"], report["train_sha256"])

    def test_proxy_reward_teacher_and_rollout_settings_are_guarded(self) -> None:
        cases = (
            ("proxy", "alpha", 1.1, "proxy.alpha"),
            ("proxy", "lambda_b", -0.1, "proxy.lambda_b"),
            ("proxy", "min_vision_tokens", 0, "proxy.min_vision_tokens"),
            ("proxy", "tau_s", 0, "proxy.tau_s"),
            ("reward", "rho_visual", 0, "reward.rho_visual"),
            ("reward", "answer_utility", 0.5, "answer_utility"),
            ("teacher", "ema_decay", 1.0, "teacher.ema_decay"),
            ("setting", "rollout_temperature", 0.6, "temperature=1"),
            ("setting", "rollout_top_p", 0.9, "top_p=1"),
            ("setting", "enable_thinking", True, "enable_thinking=false"),
            ("setting", "num_generations", 4, "generation_batch_size"),
            ("setting", "fp16_initial_scale", 0, "fp16_initial_scale"),
        )
        for section, key, value, error in cases:
            with self.subTest(section=section, key=key, value=value):
                self.settings[section][key] = value
                self._save()
                with self.assertRaisesRegex(ValueError, error):
                    load_plan(self.config)
                self.settings = json.loads(DEFAULT_CONFIG.read_text())
                self.settings["train_manifest"] = "train.jsonl"
                self.settings["validation_manifest"] = "dev.jsonl"
                self.settings["output_dir"] = "run"

    def test_world_size_guard_precedes_torch_or_model_loading(self) -> None:
        plan = load_plan(self.config)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "WORLD_SIZE=1"):
                run_training(plan)

    def test_output_dir_with_prior_run_evidence_cannot_be_reused(self) -> None:
        self.settings["setting"]["target_world_size"] = 1
        self._save()
        plan = load_plan(self.config)
        plan.output_dir.mkdir()
        for filename in ("run_manifest.json", "training_completed.json", "proxy_validation_metrics.jsonl"):
            with self.subTest(filename=filename):
                marker = plan.output_dir / filename
                marker.write_text("existing run evidence", encoding="utf-8")
                with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                    with self.assertRaisesRegex(RuntimeError, "cannot resume or reuse a run"):
                        run_training(plan)
                self.assertEqual(marker.read_text(encoding="utf-8"), "existing run evidence")
                marker.unlink()

    def test_nonzero_rank_does_not_reject_current_run_manifest(self) -> None:
        self.settings["setting"]["target_world_size"] = 2
        self._save()
        plan = load_plan(self.config)
        plan.output_dir.mkdir()
        (plan.output_dir / "run_manifest.json").write_text("written by rank zero", encoding="utf-8")
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}), patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "optional training dependencies"):
                run_training(plan)

    def test_chat_template_fallback_and_generation_terminators(self) -> None:
        tokenizer = SimpleNamespace(
            chat_template="tokenizer template", eos_token_id=7,
            all_special_tokens=["<|im_end|>"], unk_token_id=0,
            convert_tokens_to_ids=lambda _token: 11,
        )
        processor = SimpleNamespace(chat_template=None, tokenizer=tokenizer)
        _ensure_processor_chat_template(processor)
        self.assertEqual(processor.chat_template, "tokenizer template")
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[5, 7]))
        self.assertEqual(configure_generation_terminators(model, tokenizer), (5, 7, 11))
        self.assertEqual(model.generation_config.eos_token_id, [5, 7, 11])
        empty = SimpleNamespace(eos_token_id=None, all_special_tokens=[], unk_token_id=None)
        with self.assertRaisesRegex(ValueError, "no generation terminator"):
            configure_generation_terminators(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None)), empty)

    def test_qwen_template_keeps_closed_empty_think_prefill(self) -> None:
        class Processor:
            chat_template = None

            def __init__(self):
                self.tokenizer = SimpleNamespace(chat_template="prefix" + _QWEN35_THINKING_SUFFIX)

            def apply_chat_template(self, _messages, **_kwargs):
                self.template_kwargs = _kwargs
                if self.chat_template.endswith((_QWEN35_THINKING_SUFFIX, _QWEN35_NONTHINKING_SUFFIX)):
                    return "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                return "<|im_start|>assistant\n"

        processor = Processor()
        digest = configure_nonthinking_template(processor)
        self.assertEqual(processor.chat_template, processor.tokenizer.chat_template)
        self.assertIs(processor.template_kwargs["enable_thinking"], False)
        self.assertTrue(processor.chat_template.endswith(_QWEN35_NONTHINKING_SUFFIX))
        self.assertNotIn("enable_thinking", processor.chat_template)
        self.assertEqual(len(digest), 64)

        plain = Processor()
        plain.chat_template = "other plain template"
        self.assertEqual(len(configure_nonthinking_template(plain)), 64)
        self.assertEqual(plain.tokenizer.chat_template, "other plain template")

        closed = Processor()
        closed.chat_template = "other closed template"
        closed.apply_chat_template = lambda *_args, **_kwargs: "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(len(configure_nonthinking_template(closed)), 64)

        unsupported = Processor()
        unsupported.chat_template = "other template"
        unsupported.apply_chat_template = lambda *_args, **_kwargs: "<|im_start|>assistant\n<think>\n"
        with self.assertRaisesRegex(ValueError, "nonthinking assistant prefix"):
            configure_nonthinking_template(unsupported)

    def _mock_runtime(self, *, successful_updates: int, share_scaler: bool = True) -> tuple[dict[str, object], list[tuple[str, object]], list[object]]:
        """Dependency stubs exercise launcher wiring without an optimizer run."""
        modules: dict[str, object] = {}
        loads: list[tuple[str, object]] = []
        trainers: list[object] = []

        torch = types.ModuleType("torch")
        torch.float32, torch.float16, torch.bfloat16 = "float32", "float16", "bfloat16"

        class FakeGradScaler:
            def __init__(self, device, *, init_scale=65536):
                self.device = device
                self.scale = float(init_scale)

            def is_enabled(self):
                return True

            def get_scale(self):
                return self.scale

        torch.amp = SimpleNamespace(GradScaler=FakeGradScaler)
        modules["torch"] = torch

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: SimpleNamespace(model_info=lambda *_args, **_kwargs: SimpleNamespace(sha="resolved"))
        modules["huggingface_hub"] = hub

        class FakeModel:
            def __init__(self) -> None:
                self.generation_config = SimpleNamespace(eos_token_id=5)
                self.parameter = SimpleNamespace(requires_grad_=lambda _flag: None)

            def eval(self):
                return self

            def parameters(self):
                return [self.parameter]

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        def load_model(_model_id, *, dtype, **_kwargs):
            loads.append(("model", dtype))
            return FakeModel()

        tokenizer = SimpleNamespace(
            chat_template="prefix" + _QWEN35_THINKING_SUFFIX,
            eos_token_id=7, all_special_tokens=["<|im_end|>"],
            unk_token_id=0, convert_tokens_to_ids=lambda _token: 11,
        )

        class FakeProcessor:
            def __init__(self):
                self.chat_template = None
                self.tokenizer = tokenizer
                self.image_processor = SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 16777216})

            def apply_chat_template(self, _messages, **_kwargs):
                if self.chat_template.endswith((_QWEN35_THINKING_SUFFIX, _QWEN35_NONTHINKING_SUFFIX)):
                    return "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                return "<|im_start|>assistant\n"

            def save_pretrained(self, path):
                (Path(path) / "fake_tokenizer_config.json").write_text(
                    json.dumps({"chat_template": self.tokenizer.chat_template}), encoding="utf-8"
                )

        processor = FakeProcessor()

        def load_processor(_model_id, **_kwargs):
            loads.append(("processor", _kwargs.get("padding_side")))
            return processor

        transformers = types.ModuleType("transformers")
        transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=load_model)
        transformers.AutoProcessor = SimpleNamespace(from_pretrained=load_processor)
        transformers.TrainerCallback = object
        modules["transformers"] = transformers

        trl = types.ModuleType("trl")
        experimental = types.ModuleType("trl.experimental")
        gold = types.ModuleType("trl.experimental.gold")
        gold.GOLDConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        modules.update({"trl": trl, "trl.experimental": experimental, "trl.experimental.gold": gold})

        evaluation = types.ModuleType("sure_vl.proxy_evaluation")
        evaluation.ProxyValidationCallback = lambda **kwargs: SimpleNamespace(**kwargs)
        modules["sure_vl.proxy_evaluation"] = evaluation

        class FakeTrainer:
            def __init__(self, **kwargs):
                self.args = kwargs["args"]
                self.model = kwargs["model"]
                # Trainer aligns the model EOS to the tokenizer, then GOLD
                # copies that clobbered value into its generation config.
                self.model.generation_config.eos_token_id = 7
                self.generation_config = SimpleNamespace(eos_token_id=7)
                self.generation_kwargs = {"eos_token_id": 7}
                self.teacher_model = kwargs["teacher_model"]
                self.callbacks = []
                self.state = SimpleNamespace(global_step=0)
                self.accelerator = SimpleNamespace(
                    is_main_process=True, unwrap_model=lambda model: model,
                    native_amp=True, mixed_precision="fp16", device=SimpleNamespace(type="cuda"),
                    _optimizers=[], scaler=FakeGradScaler("cuda"),
                )
                self.optimizer_evidence = SimpleNamespace(summary=lambda: {
                    "optimizer_attempted_steps": 2,
                    "optimizer_successful_updates": successful_updates,
                    "optimizer_skipped_updates": 2 - successful_updates,
                    "teacher_mode": "ema",
                    "teacher_ema_updates": successful_updates,
                })
                trainers.append(self)

            def add_callback(self, callback):
                self.callbacks.append(callback)

            def train(self):
                self.accelerator._optimizers = [SimpleNamespace(
                    scaler=self.accelerator.scaler if share_scaler else FakeGradScaler("cuda")
                )]
                for callback in self.callbacks:
                    on_begin = getattr(callback, "on_train_begin", None)
                    if callable(on_begin):
                        on_begin(self.args, self.state, None)
                self.state.global_step = 2
                path = Path(self.args.output_dir) / "proxy_validation_metrics.jsonl"
                path.write_text(''.join(json.dumps({
                    "optimizer_step": step, "actual_successful_optimizer_updates": step,
                    "optimizer_state_max_step": step, "optimizer_step_count_agrees": True,
                }) + '\n' for step in (0, 2)))
                return SimpleNamespace(global_step=2, metrics={"train_loss": 0.1})

            def save_model(self):
                self.saved = True

        trainer_module = types.ModuleType("sure_vl.proxy_trainer")
        trainer_module.ProxyGOLDTrainer = FakeTrainer
        modules["sure_vl.proxy_trainer"] = trainer_module
        return modules, loads, trainers

    def test_fp16_launch_loads_fp32_student_fp16_teacher_and_records_success(self) -> None:
        model_dir = self.root / "model"
        model_dir.mkdir()
        self.settings["model_id"] = str(model_dir)
        setting = self.settings["setting"]
        setting.update({
            "target_world_size": 1, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1, "generation_batch_size": 1,
            "num_generations": 1, "max_steps": 2, "bf16": False, "fp16": True,
            "fp16_initial_scale": 1.0,
        })
        self._save()
        plan = load_plan(self.config)
        modules, loads, trainers = self._mock_runtime(successful_updates=2)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}), patch.dict(sys.modules, modules), patch(
            "sure_vl.train_proxy.build_proxy_dataset", return_value=object()
        ):
            run_training(plan)
        self.assertEqual([dtype for kind, dtype in loads if kind == "model"], ["float32", "float16"])
        self.assertEqual(trainers[0].args.fp16, True)
        self.assertEqual(trainers[0].args.bf16, False)
        self.assertEqual(trainers[0].args.top_k, 0)
        self.assertEqual(trainers[0].args.top_p, 1.0)
        self.assertEqual(trainers[0].args.temperature, 1.0)
        self.assertTrue(trainers[0].args.disable_dropout)
        self.assertEqual(trainers[0].model.generation_config.eos_token_id, [5, 7, 11])
        self.assertEqual(trainers[0].generation_config.eos_token_id, [5, 7, 11])
        self.assertEqual(trainers[0].generation_kwargs["eos_token_id"], [5, 7, 11])
        self.assertEqual(len(trainers[0].callbacks), 2)
        self.assertIs(trainers[0].accelerator._optimizers[0].scaler, trainers[0].accelerator.scaler)
        saved_template = json.loads((plan.output_dir / "fake_tokenizer_config.json").read_text())["chat_template"]
        self.assertNotIn("enable_thinking", saved_template)
        self.assertTrue(saved_template.endswith(_QWEN35_NONTHINKING_SUFFIX))
        record = json.loads((plan.output_dir / "training_completed.json").read_text())
        self.assertEqual(record["optimizer_steps"], 2)
        self.assertEqual(record["optimizer_evidence"]["optimizer_successful_updates"], 2)
        self.assertEqual(record["validation_optimizer_steps"], [0, 2])
        self.assertEqual(record["generation_terminators"], [5, 7, 11])
        self.assertIn("source_commit", record)
        self.assertIn(record["source_worktree_dirty"], (True, False, None))
        self.assertEqual(record["runtime_versions"]["python"], sys.version.split()[0])
        for package in ("torch", "transformers", "trl", "accelerate"):
            self.assertIn(package, record["runtime_versions"])
        self.assertEqual(len(record["chat_template_sha256"]), 64)
        self.assertEqual(record["sampling"], {"temperature": 1.0, "top_p": 1.0, "top_k": 0})
        self.assertIs(record["enable_thinking"], False)
        self.assertEqual(record["fp16_initial_scale"], 1.0)
        self.assertEqual(record["fp16_final_scale"], 1.0)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "cannot resume or reuse a run"):
                run_training(plan)

    def test_scaler_identity_is_verified_before_any_update(self) -> None:
        model_dir = self.root / "model"
        model_dir.mkdir()
        self.settings["model_id"] = str(model_dir)
        self.settings["setting"].update({
            "target_world_size": 1, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1, "generation_batch_size": 1,
            "num_generations": 1, "max_steps": 2, "bf16": False, "fp16": True,
            "fp16_initial_scale": 1.0,
        })
        self._save()
        plan = load_plan(self.config)
        modules, _loads, trainers = self._mock_runtime(successful_updates=2, share_scaler=False)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}), patch.dict(sys.modules, modules), patch(
            "sure_vl.train_proxy.build_proxy_dataset", return_value=object()
        ):
            with self.assertRaisesRegex(RuntimeError, "does not share"):
                run_training(plan)
        self.assertEqual(trainers[0].state.global_step, 0)
        self.assertFalse((plan.output_dir / "training_completed.json").exists())

    def test_skipped_optimizer_update_cannot_be_reported_complete(self) -> None:
        model_dir = self.root / "model"
        model_dir.mkdir()
        self.settings["model_id"] = str(model_dir)
        self.settings["setting"].update({
            "target_world_size": 1, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1, "generation_batch_size": 1,
            "num_generations": 1, "max_steps": 2,
        })
        self._save()
        plan = load_plan(self.config)
        modules, _loads, trainers = self._mock_runtime(successful_updates=1)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}), patch.dict(sys.modules, modules), patch(
            "sure_vl.train_proxy.build_proxy_dataset", return_value=object()
        ):
            with self.assertRaisesRegex(RuntimeError, "only 1 successful updates"):
                run_training(plan)
        self.assertFalse((plan.output_dir / "training_completed.json").exists())
        self.assertFalse(hasattr(trainers[0], "saved"))


if __name__ == "__main__":
    unittest.main()
