"""Data and batch preflight for the model-free TRL entry point."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sure_vl.train_trl import DEFAULT_CONFIG, _ensure_processor_chat_template, load_plan, main, run_training


def _example(example_id: str, split: str, student: str, teacher: str) -> dict:
    return {
        "id": example_id,
        "split": split,
        "student_image": student,
        "teacher_image": teacher,
        "question": "What color is the square?",
        "required_visual_facts": {"color": "blue", "shape": "square"},
        "accepted_answers": ["blue"],
    }


class TrainingPreflightTests(unittest.TestCase):
    def test_processor_uses_tokenizer_template_only_when_missing(self) -> None:
        class Processor:
            chat_template = None

            def __init__(self):
                self.tokenizer = type("Tokenizer", (), {"chat_template": "tokenizer template"})()

        processor = Processor()
        _ensure_processor_chat_template(processor)
        self.assertEqual(processor.chat_template, "tokenizer template")
        processor.chat_template = "processor template"
        _ensure_processor_chat_template(processor)
        self.assertEqual(processor.chat_template, "processor template")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("train-blur.png", "train-clear.png", "dev-blur.png", "dev-clear.png"):
            (self.root / name).write_bytes(b"test image placeholder")
        self.train = self.root / "train.jsonl"
        self.dev = self.root / "dev.jsonl"
        self.train.write_text(
            json.dumps(_example("train-1", "train", "train-blur.png", "train-clear.png")) + "\n"
        )
        self.dev.write_text(
            json.dumps(_example("dev-1", "dev", "dev-blur.png", "dev-clear.png")) + "\n"
        )
        self.config = self.root / "config.json"
        self.settings = json.loads(DEFAULT_CONFIG.read_text())
        self.settings["train_manifest"] = "train.jsonl"
        self.settings["validation_manifest"] = "dev.jsonl"
        self.settings["output_dir"] = "run"
        self._save()

    def _save(self) -> None:
        self.config.write_text(json.dumps(self.settings))

    def test_preflight_reports_frozen_data_and_batch(self) -> None:
        plan = load_plan(self.config)
        report = plan.summary()
        self.assertEqual(report["global_batch_size"], 256)
        self.assertEqual(report["unique_prompts_per_rank_per_update"], 4)
        self.assertEqual(report["train_count"], 1)
        self.assertEqual(report["validation_count"], 1)
        self.assertEqual(len(report["train_sha256"]), 64)
        self.assertEqual(report["train_split"], "train")
        self.assertEqual(report["validation_split"], "dev")
        self.assertEqual(plan.output_dir, (self.root / "run").resolve())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--config", str(self.config), "--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["train_sha256"], report["train_sha256"])

    def test_batch_equation_is_enforced(self) -> None:
        self.settings["setting"]["num_generations"] = 4
        self._save()
        with self.assertRaisesRegex(ValueError, "generation_batch_size"):
            load_plan(self.config)

    def test_precision_modes_are_mutually_exclusive(self) -> None:
        self.settings["setting"]["fp16"] = True
        self._save()
        with self.assertRaisesRegex(ValueError, "cannot both"):
            load_plan(self.config)

    def test_validation_cadence_and_subset_are_checked(self) -> None:
        self.settings["validation"]["every_n_steps"] = 0
        self._save()
        with self.assertRaisesRegex(ValueError, "validation.every_n_steps"):
            load_plan(self.config)
        self.settings["validation"]["every_n_steps"] = 20
        self.settings["validation"]["subset_size"] = -1
        self._save()
        with self.assertRaisesRegex(ValueError, "validation.subset_size"):
            load_plan(self.config)

    def test_same_image_across_splits_is_rejected(self) -> None:
        self.dev.write_text(
            json.dumps(_example("dev-1", "dev", "train-blur.png", "dev-clear.png")) + "\n"
        )
        with self.assertRaisesRegex(ValueError, "overlap in image"):
            load_plan(self.config)

    def test_requires_present_manifest_paths(self) -> None:
        self.settings["train_manifest"] = None
        self._save()
        with self.assertRaisesRegex(ValueError, "train_manifest"):
            load_plan(self.config)

    def test_world_size_guard_precedes_model_loading(self) -> None:
        plan = load_plan(self.config)
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "WORLD_SIZE=1"):
                run_training(plan)


if __name__ == "__main__":
    unittest.main()
