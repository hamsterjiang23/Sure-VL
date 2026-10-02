"""Frozen validation and update evidence attached to the TRL Trainer lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from ...proxy_evaluation import _generation_seed, _optimizer_state_step, _subset_sha256, select_proxy_subset
from ...proxy_metrics import evaluate_proxy_records


class GRPOValidationCallback(TrainerCallback):
    def __init__(self, trainer: Any, rows: list[dict], output_dir: Path,
                 config: dict, tracker: Any):
        self.trainer, self.output_dir, self.config, self.tracker = trainer, output_dir, config, tracker
        self.rows = select_proxy_subset(rows, size=config["subset_size"], seed=config["subset_seed"])
        self.subset_hash = _subset_sha256(self.rows)
        self.steps: list[int] = []

    def _evaluate(self, step: int) -> None:
        import torch
        from PIL import Image
        trainer = self.trainer
        model = trainer.accelerator.unwrap_model(trainer.model)
        previous_mode = model.training
        previous = (trainer.generation_config.temperature, trainer.generation_config.top_p,
                    trainer.generation_config.max_new_tokens)
        trainer.generation_config.temperature = self.config["temperature"]
        trainer.generation_config.top_p = self.config["top_p"]
        trainer.generation_config.max_new_tokens = self.config["max_new_tokens"]
        model.eval()
        attempts = []
        try:
            device = trainer.accelerator.device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.no_grad():
                for row in self.rows:
                    with Image.open(row["image"]) as source:
                        image = source.convert("RGB")
                    prepared = {**row, "images": [image]}
                    with torch.random.fork_rng(devices=devices):
                        seed = _generation_seed(self.config["subset_seed"], row["example_id"])
                        torch.manual_seed(seed)
                        if devices:
                            torch.cuda.manual_seed(seed)
                        trainer._generate_and_score_completions([prepared])
                    if len(trainer.last_rollout_records) != 1:
                        raise RuntimeError("GRPO validation must assess exactly one generated completion per ID")
                    attempts.append(dict(trainer.last_rollout_records[0]))
        finally:
            trainer.generation_config.temperature, trainer.generation_config.top_p, trainer.generation_config.max_new_tokens = previous
            model.train(previous_mode)
            # Base GRPO's one-sample eval group has undefined reward std. Our
            # fixed-set reports below carry proper denominators instead.
            trainer._metrics["eval"].clear()
        report = evaluate_proxy_records(attempts)
        evidence = trainer.optimizer_evidence.summary()
        adam_step = _optimizer_state_step(trainer)
        report.update({
            "optimizer_step": step,
            "actual_successful_optimizer_updates": evidence["optimizer_successful_updates"],
            "optimizer_state_max_step": adam_step,
            "optimizer_step_count_agrees": adam_step == evidence["optimizer_successful_updates"],
            "optimizer_evidence": evidence,
            "subset_sha256": self.subset_hash,
            "subset_ids": [row["example_id"] for row in self.rows],
            "generation": {"sample_count": len(attempts), "temperature": self.config["temperature"],
                           "top_p": self.config["top_p"], "max_new_tokens": self.config["max_new_tokens"]},
        })
        if step == 0 and adam_step not in (None, 0):
            raise RuntimeError("validation before training found nonzero Adam state")
        path = self.output_dir / f"proxy_validation_attempts_step_{step:06d}.jsonl"
        with path.open("x", encoding="utf-8") as destination:
            for record in attempts:
                destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with (self.output_dir / "proxy_validation_metrics.jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        self.steps.append(step)
        self.tracker.log_validation(report, evidence, step)
        print(json.dumps({"validation_step": step, "answer_accuracy": report["answer_accuracy"],
                          "format_clean_count": report["output_coverage"]["format_clean_count"],
                          "visual_proxy_pair_count": report["visual_proxy_pair_count"],
                          "answer_confidence_count": report["answer_confidence_count"]}), flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self._evaluate(0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.config["every_n_steps"] == 0 or state.global_step == state.max_steps:
            self._evaluate(int(state.global_step))
        return control


class GradientMonitorCallback(TrainerCallback):
    """Measure the actual gradients *after* Trainer's configured norm clip."""
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        import torch
        norms = [parameter.grad.detach().float().norm() for parameter in self.trainer.model.parameters()
                 if parameter.grad is not None]
        norm = float(torch.stack(norms).norm()) if norms else 0.0
        if not norms or not torch.isfinite(torch.tensor(norm)):
            raise RuntimeError("missing or nonfinite post-clip gradients")
        self.trainer._metrics["train"]["grad_norm_after_clip"].append(norm)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        evidence = self.trainer.optimizer_evidence.summary()
        adam_step = _optimizer_state_step(self.trainer)
        if adam_step != evidence["optimizer_successful_updates"]:
            raise RuntimeError("actual Adam state differs from successful-update evidence")
        row = {"trainer_step": int(state.global_step), "optimizer_state_max_step": adam_step, **evidence}
        with (Path(args.output_dir) / "optimizer_evidence.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps(row, allow_nan=False) + "\n")
        self.trainer._metrics["train"]["optimizer_state_max_step"].append(float(adam_step))
        return control
