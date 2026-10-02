"""Preflight and launch the paired-image TRL GOLD training adapter.

The reference configuration deliberately has no data paths. A training run
requires frozen train and validation Sure-VL manifests with independently
declared visual facts and paired student/teacher image paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import ProtocolError
from .trl_data import assert_disjoint_manifests, build_gold_dataset, manifest_to_gold_rows


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sure_vl_trl_v1.json"


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be set to a path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_processor_chat_template(processor: Any) -> None:
    """Use a tokenizer template when a base VLM omits the processor copy.

    Some Qwen3.5 base snapshots ship a tokenizer chat template but no
    processor-level template. GOLD and validation both call the processor's
    apply_chat_template, so they must share the same explicit fallback.
    """
    if getattr(processor, "chat_template", None) is not None:
        return
    template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    if template is None:
        raise ValueError("processor and tokenizer both lack a chat template")
    processor.chat_template = template


@dataclass(frozen=True)
class TrainingPlan:
    config: dict[str, Any]
    config_path: Path
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        setting = self.config["setting"]
        return {
            "model_id": self.config["model_id"],
            "model_revision": self.config.get("model_revision"),
            "train_manifest": str(self.train_manifest),
            "train_sha256": _sha256(self.train_manifest),
            "train_count": len(self.train_rows),
            "train_split": self.train_rows[0]["split"],
            "validation_manifest": str(self.validation_manifest),
            "validation_sha256": _sha256(self.validation_manifest),
            "validation_count": len(self.validation_rows),
            "validation_split": self.validation_rows[0]["split"],
            "global_batch_size": (
                setting["target_world_size"]
                * setting["per_device_train_batch_size"]
                * setting["gradient_accumulation_steps"]
            ),
            "unique_prompts_per_rank_per_update": setting["generation_batch_size"],
            "output_dir": str(self.output_dir),
        }


def load_plan(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    train_manifest: str | Path | None = None,
    validation_manifest: str | Path | None = None,
) -> TrainingPlan:
    """Validate a complete data and configuration snapshot without ML imports."""
    path = Path(config_path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    revision = config.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model_revision must be a nonempty string or null")
    setting = config.get("setting")
    loss = config.get("loss")
    validation = config.get("validation")
    if not isinstance(setting, dict) or not isinstance(loss, dict) or not isinstance(validation, dict):
        raise ValueError("setting, loss, and validation must be JSON objects")
    for name in (
        "target_world_size", "per_device_train_batch_size", "gradient_accumulation_steps",
        "generation_batch_size", "num_generations", "max_completion_length", "max_pixels",
        "seed", "logging_steps", "save_steps",
    ):
        value = setting.get(name)
        if name == "seed":
            if type(value) is not int or value < 0:
                raise ValueError("seed must be a nonnegative integer")
        else:
            _positive_int(value, name)
    for name in ("num_train_epochs", "learning_rate", "rollout_temperature", "max_grad_norm"):
        _positive_number(setting.get(name), name)
    if type(setting.get("max_steps")) is not int or setting["max_steps"] == 0 or setting["max_steps"] < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    top_p = _positive_number(setting.get("rollout_top_p"), "rollout_top_p")
    if top_p > 1:
        raise ValueError("rollout_top_p must be at most 1")
    if type(setting.get("bf16")) is not bool or type(setting.get("fp16")) is not bool:
        raise ValueError("bf16 and fp16 must be booleans")
    if setting["bf16"] and setting["fp16"]:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    for name in ("policy_weight", "opsd_weight"):
        value = loss.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if loss["policy_weight"] == loss["opsd_weight"] == 0:
        raise ValueError("at least one loss weight must be positive")
    if loss.get("opsd_beta") != 0.0:
        raise ValueError("the v1 OPSD setting requires beta=0 forward KL")
    _positive_number(loss.get("opsd_temperature"), "opsd_temperature")
    _positive_number(loss.get("opsd_token_clip"), "opsd_token_clip")
    _positive_int(loss.get("diagnostic_tokens"), "diagnostic_tokens")
    for name in ("subset_size", "every_n_steps", "max_new_tokens"):
        _positive_int(validation.get(name), f"validation.{name}")
    if type(validation.get("subset_seed")) is not int or validation["subset_seed"] < 0:
        raise ValueError("validation.subset_seed must be a nonnegative integer")
    _positive_number(validation.get("temperature"), "validation.temperature")
    if _positive_number(validation.get("top_p"), "validation.top_p") > 1:
        raise ValueError("validation.top_p must be at most 1")
    local_batch = setting["per_device_train_batch_size"] * setting["gradient_accumulation_steps"]
    if setting["generation_batch_size"] * setting["num_generations"] != local_batch:
        raise ValueError("generation_batch_size * num_generations must equal the local optimizer batch")

    base = path.parent
    train_path = _path(str(train_manifest) if train_manifest is not None else config.get("train_manifest"), base, "train_manifest")
    validation_path = _path(
        str(validation_manifest) if validation_manifest is not None else config.get("validation_manifest"),
        base, "validation_manifest",
    )
    if train_path == validation_path:
        raise ValueError("train and validation manifests must be different files")
    output_dir = _path(config.get("output_dir"), base, "output_dir")
    train_rows = manifest_to_gold_rows(train_path)
    validation_rows = manifest_to_gold_rows(validation_path)
    assert_disjoint_manifests(train_rows, validation_rows)
    return TrainingPlan(config, path, train_path, validation_path, output_dir, train_rows, validation_rows)


def run_training(plan: TrainingPlan) -> None:
    """Load the same starting checkpoint twice and launch one TRL GOLD run."""
    setting = plan.config["setting"]
    loss = plan.config["loss"]
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != setting["target_world_size"]:
        raise RuntimeError(
            f"launch has WORLD_SIZE={actual_world_size}; config requires {setting['target_world_size']}"
        )
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from trl.experimental.gold import GOLDConfig
    except ImportError as error:
        raise RuntimeError("install the optional training dependencies with `uv sync --extra train`") from error

    from .trl_evaluation import FrozenValidationCallback
    from .trl_trainer import SureVLGOLDTrainer

    model_id = plan.config["model_id"]
    revision = plan.config.get("model_revision")
    if not Path(model_id).is_dir():
        revision = HfApi().model_info(model_id, revision=revision).sha
    load_kwargs = {"revision": revision} if revision else {}
    dtype = torch.bfloat16 if setting["bf16"] else (torch.float16 if setting["fp16"] else torch.float32)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left", **load_kwargs)
    _ensure_processor_chat_template(processor)
    image_processor = processor.image_processor
    image_size = dict(image_processor.size)
    image_size["longest_edge"] = setting["max_pixels"]
    if image_size.get("shortest_edge", 0) > setting["max_pixels"]:
        raise ValueError("max_pixels must be at least the image processor shortest_edge")
    image_processor.size = image_size
    student = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **load_kwargs)
    teacher = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **load_kwargs)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    dataset = build_gold_dataset(plan.train_rows)
    args = GOLDConfig(
        output_dir=str(plan.output_dir),
        per_device_train_batch_size=setting["per_device_train_batch_size"],
        gradient_accumulation_steps=setting["gradient_accumulation_steps"],
        generation_batch_size=setting["generation_batch_size"],
        num_generations=setting["num_generations"],
        num_train_epochs=setting["num_train_epochs"],
        max_steps=setting["max_steps"],
        learning_rate=setting["learning_rate"],
        temperature=setting["rollout_temperature"],
        top_p=setting["rollout_top_p"],
        max_length=None,
        max_completion_length=setting["max_completion_length"],
        max_grad_norm=setting["max_grad_norm"],
        bf16=setting["bf16"],
        fp16=setting["fp16"],
        seed=setting["seed"],
        logging_steps=setting["logging_steps"],
        save_steps=setting["save_steps"],
        save_strategy="steps",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        use_vllm=False,
        use_uld_loss=False,
        lmbda=1.0,
        beta=0.0,
        report_to=[],
    )
    trainer = SureVLGOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        policy_weight=loss["policy_weight"],
        opsd_weight=loss["opsd_weight"],
        opsd_temperature=loss["opsd_temperature"],
        opsd_token_clip=loss["opsd_token_clip"],
        diagnostic_tokens=loss["diagnostic_tokens"],
    )
    validation = plan.config["validation"]
    trainer.add_callback(FrozenValidationCallback(
        validation_rows=plan.validation_rows,
        processor=processor,
        output_dir=plan.output_dir,
        subset_size=validation["subset_size"],
        subset_seed=validation["subset_seed"],
        every_n_steps=validation["every_n_steps"],
        final_step=setting["max_steps"] if setting["max_steps"] > 0 else None,
        max_new_tokens=validation["max_new_tokens"],
        temperature=validation["temperature"],
        top_p=validation["top_p"],
    ))
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    run_record = plan.summary()
    run_record["resolved_model_revision"] = revision
    if trainer.accelerator.is_main_process:
        (plan.output_dir / "run_manifest.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    result = trainer.train()
    optimizer_steps = int(trainer.state.global_step)
    if optimizer_steps != int(result.global_step):
        raise RuntimeError(
            f"optimizer step evidence disagrees: state={optimizer_steps}, result={result.global_step}"
        )
    if setting["max_steps"] > 0 and optimizer_steps != setting["max_steps"]:
        raise RuntimeError(
            f"training stopped at {optimizer_steps} optimizer steps, expected {setting['max_steps']}"
        )
    trainer.save_model()
    if trainer.accelerator.is_main_process:
        validation_path = plan.output_dir / "validation_metrics.jsonl"
        if not validation_path.is_file():
            raise RuntimeError("frozen validation did not write its metrics")
        validation_steps = [
            int(json.loads(line)["optimizer_step"])
            for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if 0 not in validation_steps or optimizer_steps not in validation_steps:
            raise RuntimeError(
                f"frozen validation lacks step 0 or step {optimizer_steps}: {validation_steps}"
            )
        completed = {
            **run_record,
            "status": "completed",
            "optimizer_steps": optimizer_steps,
            "train_output_global_step": int(result.global_step),
            "validation_optimizer_steps": validation_steps,
            "train_metrics": result.metrics,
        }
        (plan.output_dir / "training_completed.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sure-VL TRL GOLD/OPSD trainer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--check-only", action="store_true", help="validate manifests without ML imports or model downloads")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(
            args.config,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
        )
        print(json.dumps(plan.summary(), ensure_ascii=False, indent=2))
        if not args.check_only:
            run_training(plan)
    except (OSError, ValueError, ProtocolError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
