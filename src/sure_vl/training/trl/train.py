"""Launch the Sure-VL subclass of the installed TRL GRPOTrainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ...train_proxy import (TrainingPlan, _sha256, _source_provenance, _runtime_versions, configure_generation_terminators,
                            configure_nonthinking_template, load_plan)

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/trl/qwen35_08b_visionopd_100step.json"


def load_grpo_plan(path=DEFAULT_CONFIG) -> TrainingPlan:
    plan = load_plan(path)
    if plan.config.get("backend") != "trl_grpo":
        raise ValueError("TRL GRPO entry requires backend='trl_grpo'")
    setting = plan.config["setting"]
    if setting["target_world_size"] != 1 or setting["per_device_train_batch_size"] != 1:
        raise ValueError("this V100 recipe requires one GPU and microbatch size one")
    if setting["gradient_accumulation_steps"] != setting["generation_batch_size"]:
        raise ValueError("complete one fresh generation batch before each optimizer/EMA update")
    if setting["fp16"] or setting["bf16"]:
        raise ValueError("V100 Qwen3.5 recipe uses FP32")
    if plan.config.get("grpo") != {"loss_type": "grpo", "scale_rewards": "none", "num_iterations": 1}:
        raise ValueError("GRPO recipe requires loss_type=grpo, scale_rewards=none, num_iterations=1")
    if plan.output_dir.exists() and any(plan.output_dir.iterdir()):
        raise RuntimeError("cannot resume or reuse a nonempty training output directory")
    return plan


def _dataset(rows):
    from datasets import Dataset, Image, List
    records = [{"prompt": row["prompt"], "images": [row["image"]],
                "example_payload": row["example_payload"], "example_id": row["example_id"],
                "split": row["split"]} for row in rows]
    return Dataset.from_list(records).cast_column("images", List(Image(mode="RGB")))


def run_training(plan: TrainingPlan) -> None:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("this single-GPU GRPO recipe requires WORLD_SIZE=1")
    import torch
    from trl import GRPOConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ...teacher_ema import OptimizerEvidenceCallback
    from ...tracking import ExperimentTracker
    from ...proxy_tracking import ProxyTrackingCallback
    from .trainer import SureVLGRPOTrainer
    from .validation import GRPOValidationCallback, GradientMonitorCallback
    runtime_versions = _runtime_versions()
    if runtime_versions["trl"] != "1.14.1":
        raise RuntimeError("this adapter requires the audited TRL 1.14.1 implementation")
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    setting = plan.config["setting"]
    model_id = plan.config["model_id"]
    local_only = Path(model_id).is_dir()
    kwargs = {"local_files_only": local_only, "revision": plan.config.get("model_revision")}
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left", **kwargs)
    template_hash = configure_nonthinking_template(processor)
    processor.image_processor.size = {**dict(processor.image_processor.size), "longest_edge": setting["max_pixels"]}
    student = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32,
                                                        attn_implementation="eager", **kwargs)
    teacher = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32,
                                                        attn_implementation="eager", **kwargs)
    stops = configure_generation_terminators(student, processor.tokenizer)
    args = GRPOConfig(
        output_dir=str(plan.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=setting["gradient_accumulation_steps"],
        generation_batch_size=setting["generation_batch_size"],
        num_generations=setting["num_generations"], num_generations_eval=1,
        per_device_eval_batch_size=1, num_iterations=1,
        loss_type="grpo", scale_rewards="none", beta=0.0,
        learning_rate=setting["learning_rate"], max_steps=setting["max_steps"],
        num_train_epochs=setting["num_train_epochs"],
        temperature=1.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0,
        max_completion_length=setting["max_completion_length"],
        use_vllm=False, use_liger_kernel=False, disable_dropout=True,
        bf16=False, fp16=False, gradient_checkpointing=setting["gradient_checkpointing"],
        max_grad_norm=setting["max_grad_norm"], seed=setting["seed"],
        logging_steps=setting["logging_steps"], save_steps=setting["save_steps"],
        save_strategy="steps", eval_strategy="no", remove_unused_columns=False,
        # The tracking callback uploads native GRPO logs on the successful
        # update axis. The stock W&B callback would redefine train/* axes.
        report_to=[],
        run_name=plan.config["tracking"].get("run_name"), log_completions=False,
    )
    trainer = SureVLGRPOTrainer(model=student, teacher_model=teacher, args=args,
                                train_dataset=_dataset(plan.train_rows), processing_class=processor,
                                proxy_config=plan.config["proxy"], reward_config=plan.config["reward"],
                                teacher_config=plan.config["teacher"], loss_config=plan.config["loss"])
    trainer.generation_config.eos_token_id = list(stops)
    trainer.model.generation_config.eos_token_id = list(stops)
    # The official optimizer/scheduler/accumulation and checkpoint code is
    # retained. Count its successful steps and update Teacher only afterward.
    trainer.optimizer_evidence = OptimizerEvidenceCallback(
        trainer, teacher_mode=plan.config["teacher"]["mode"], ema_decay=plan.config["teacher"]["ema_decay"])
    trainer.add_callback(trainer.optimizer_evidence)
    trainer.add_callback(GradientMonitorCallback(trainer))
    manifest = plan.summary()
    manifest.update({"backend": "trl_grpo", "trainer_class": "SureVLGRPOTrainer(GRPOTrainer)",
                     "objective": "separate group-centered content/report GRPO advantages; content-only OPSD FK",
                     "chat_template_sha256": template_hash, "enable_thinking": False,
                     "generation_terminators": stops,
                     "runtime_versions": runtime_versions,
                     "gradient_clip_max_norm": setting["max_grad_norm"],
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    manifest["source_commit"], manifest["source_worktree_dirty"] = _source_provenance()
    root = Path(__file__).resolve().parents[2]
    manifest["source_files_sha256"] = {str(path.relative_to(root)): _sha256(path) for path in sorted(root.rglob("*.py"))}
    tracker = None
    try:
        tracker = ExperimentTracker(plan.config["tracking"], output_dir=plan.output_dir, run_config=manifest)
        trainer.experiment_tracker = tracker
        manifest["tracking_run_url"] = tracker.run_url
        validation = GRPOValidationCallback(trainer, plan.validation_rows, plan.output_dir,
                                            plan.config["validation"], tracker)
        trainer.add_callback(validation)
        trainer.add_callback(ProxyTrackingCallback(trainer, tracker))
        (plan.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        result = trainer.train()
        evidence = trainer.optimizer_evidence.summary()
        expected = setting["max_steps"]
        if int(result.global_step) != expected or evidence["optimizer_successful_updates"] != expected:
            raise RuntimeError("GRPO did not complete the requested successful optimizer updates")
        if evidence["optimizer_skipped_updates"] or (plan.config["teacher"]["mode"] == "ema" and evidence["teacher_ema_updates"] != expected):
            raise RuntimeError("skipped update or mismatched EMA evidence")
        from ...proxy_evaluation import _optimizer_state_step
        adam_step = _optimizer_state_step(trainer)
        if adam_step != expected or 0 not in validation.steps or expected not in validation.steps:
            raise RuntimeError("GRPO Adam/frozen-validation evidence does not agree")
        trainer.save_model()
        processor.save_pretrained(plan.output_dir)
        trainer.accelerator.unwrap_model(trainer.teacher_model).save_pretrained(plan.output_dir / "teacher_final")
        completed = {**manifest, "status": "completed", "optimizer_steps": expected,
                     "optimizer_evidence": evidence, "optimizer_state_max_step": adam_step,
                     "validation_optimizer_steps": validation.steps, "train_metrics": result.metrics}
        (plan.output_dir / "training_completed.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n")
        tracker.finish(exit_code=0)
    except BaseException as error:
        (plan.output_dir / "training_failed.json").write_text(json.dumps({
            "status": "failed", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False, indent=2) + "\n")
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sure-VL TRL GRPO training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    plan = load_grpo_plan(args.config)
    print(json.dumps(plan.summary(), ensure_ascii=False, indent=2), flush=True)
    if not args.check_only:
        run_training(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
