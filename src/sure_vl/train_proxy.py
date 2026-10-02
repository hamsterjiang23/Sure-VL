"""Launch internal visual certainty and answer calibration with TRL GOLD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .proxy_data import assert_disjoint_proxy_manifests, build_proxy_dataset, manifest_to_proxy_rows
from .proxy_protocol import ProxyProtocolError
from .tracking import ExperimentTracker, validate_tracking_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sure_vl_proxy_v1.json"


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


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "trl", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_provenance() -> tuple[str | None, bool | None]:
    source_root = Path(__file__).resolve().parents[2]
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=source_root, capture_output=True, text=True,
    )
    if git_root.returncode != 0 or Path(git_root.stdout.strip()).resolve() != source_root:
        # A copied package beneath an ignored outputs directory must not
        # inherit the enclosing repository's unrelated clean commit identity.
        return None, None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True,
    )
    if commit.returncode != 0:
        return None, None
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_root, capture_output=True, text=True,
    )
    return commit.stdout.strip(), bool(status.stdout.strip()) if status.returncode == 0 else None


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


_QWEN35_THINKING_SUFFIX = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)
_QWEN35_NONTHINKING_SUFFIX = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}\n"
    "{%- endif %}"
)
_NONTHINKING_ASSISTANT_ENDINGS = (
    "<|im_start|>assistant\n",
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
)


def configure_nonthinking_template(processor: Any) -> str:
    """Freeze the cached Qwen3.5 assistant prefix to its nonthinking mode.

    Qwen3.5's native nonthinking prompt has a closed empty ``<think>`` block.
    Keep that prefix so the model directly generates the answer format. Other
    templates may use a plain assistant header, but an open ``<think>`` prefix
    would push the model into its builtin thinking mode. The processor and
    tokenizer share the final template for GOLD, rescoring, and validation.
    """
    _ensure_processor_chat_template(processor)
    template = processor.chat_template
    if not isinstance(template, str):
        raise ValueError("processor chat template must be a string")
    if template.endswith(_QWEN35_THINKING_SUFFIX):
        template = template[: -len(_QWEN35_THINKING_SUFFIX)] + _QWEN35_NONTHINKING_SUFFIX
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("processor must support apply_chat_template")
    rendered = apply_template(
        [{"role": "user", "content": "Sure-VL template preflight"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered.endswith(_NONTHINKING_ASSISTANT_ENDINGS):
        raise ValueError("chat template must end at a nonthinking assistant prefix")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def configure_generation_terminators(model: Any, tokenizer: Any) -> tuple[int, ...]:
    """Stop on the snapshot's EOS and its declared chat end marker."""
    stops: set[int] = set()
    for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(value, (list, tuple, set)):
            stops.update(int(item) for item in value)
        elif value is not None:
            stops.add(int(value))
    if "<|im_end|>" in getattr(tokenizer, "all_special_tokens", ()):
        token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if token is not None and token != getattr(tokenizer, "unk_token_id", None):
            stops.add(int(token))
    if not stops:
        raise ValueError("model/tokenizer has no generation terminator")
    model.generation_config.eos_token_id = sorted(stops)
    return tuple(sorted(stops))


def restore_training_generation_terminators(trainer: Any, terminators: tuple[int, ...]) -> tuple[int, ...]:
    """Restore EOS IDs after Trainer may align model config to the tokenizer.

    GOLD constructs its own generation config after Transformers Trainer has
    initialized the model. Both that config (used by generate and GOLD's
    completion mask) and the model copy must retain every original stop ID.
    """
    trainer.model.generation_config.eos_token_id = list(terminators)
    trainer.generation_config.eos_token_id = list(terminators)
    generation_kwargs = getattr(trainer, "generation_kwargs", None)
    if isinstance(generation_kwargs, dict):
        generation_kwargs["eos_token_id"] = list(terminators)
    return tuple(int(token) for token in trainer.generation_config.eos_token_id)


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
            "method": self.config["method"],
            "config_sha256": _sha256(self.config_path),
            "configuration": self.config,
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
            "unique_prompts_per_rank_per_update": (
                max(1, math.ceil(setting["generation_batch_size"] / setting["num_generations"]
                                 / setting["target_world_size"]))
                if self.config.get("backend") in {"trl_grpo", "verl_single_gpu"} else setting["generation_batch_size"]
            ),
            "completions_per_update": (
                setting["target_world_size"] * setting["per_device_train_batch_size"]
                * setting["gradient_accumulation_steps"]
            ),
            "unique_prompts_per_update": (
                setting["generation_batch_size"] // setting["num_generations"]
                if self.config.get("backend") in {"trl_grpo", "verl_single_gpu"}
                else setting["generation_batch_size"] * setting["target_world_size"]
            ),
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
    if config.get("method") != "internal_visual_proxy":
        raise ValueError("method must be internal_visual_proxy")
    model_id = config.get("model_id")
    config["tracking"] = validate_tracking_config(config.get("tracking"))
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
    if setting["rollout_temperature"] != 1 or top_p != 1:
        raise ValueError("proxy v1 score-function training requires rollout temperature=1 and top_p=1")
    if setting.get("enable_thinking") is not False:
        raise ValueError("proxy v1 requires enable_thinking=false for the model chat template")
    if type(setting.get("bf16")) is not bool or type(setting.get("fp16")) is not bool:
        raise ValueError("bf16 and fp16 must be booleans")
    if setting["bf16"] and setting["fp16"]:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    initial_scale = setting.get("fp16_initial_scale")
    if initial_scale is not None:
        _positive_number(initial_scale, "fp16_initial_scale")
        if not setting["fp16"]:
            raise ValueError("fp16_initial_scale requires fp16=true")
    if type(setting.get("gradient_checkpointing")) is not bool:
        raise ValueError("gradient_checkpointing must be a boolean")
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
    proxy = config.get("proxy")
    reward = config.get("reward")
    teacher = config.get("teacher")
    if not all(isinstance(item, dict) for item in (proxy, reward, teacher)):
        raise ValueError("proxy, reward, and teacher must be JSON objects")
    for name in ("alpha", "lambda_b"):
        value = proxy.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"proxy.{name} must be in [0,1]")
    _positive_number(proxy.get("tau_s"), "proxy.tau_s")
    for name in ("min_vision_tokens", "chunk_size"):
        _positive_int(proxy.get(name), f"proxy.{name}")
    for name in ("answer_utility", "rho_answer", "rho_visual", "format_penalty"):
        _positive_number(reward.get(name), f"reward.{name}")
    if reward["answer_utility"] < reward["rho_answer"]:
        raise ValueError("answer_utility must be at least rho_answer")
    if teacher.get("mode") not in {"fixed", "ema"}:
        raise ValueError("teacher.mode must be fixed or ema")
    decay = teacher.get("ema_decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("teacher.ema_decay must be in [0,1)")
    local_batch = setting["per_device_train_batch_size"] * setting["gradient_accumulation_steps"]
    if config.get("backend") in {"trl_grpo", "verl_single_gpu"}:
        if setting["generation_batch_size"] != local_batch * setting["target_world_size"]:
            raise ValueError("GRPO generation_batch_size must equal the global optimizer batch")
        if setting["num_generations"] < 2 or setting["generation_batch_size"] % setting["num_generations"]:
            raise ValueError("GRPO generation batch must contain complete groups of at least two samples")
    elif setting["generation_batch_size"] * setting["num_generations"] != local_batch:
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
    train_rows = manifest_to_proxy_rows(train_path)
    validation_rows = manifest_to_proxy_rows(validation_path)
    assert_disjoint_proxy_manifests(train_rows, validation_rows)
    return TrainingPlan(config, path, train_path, validation_path, output_dir, train_rows, validation_rows)


def run_training(plan: TrainingPlan) -> None:
    """Finish cloud logging after validation/checkpoint gates or on failure."""
    holder: dict[str, Any] = {}
    try:
        _run_training(plan, holder)
        tracker = holder.get("tracker")
        if tracker is not None:
            tracker.finish(exit_code=0)
    except BaseException:
        tracker = holder.get("tracker")
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise


def _run_training(plan: TrainingPlan, tracking_holder: dict[str, Any]) -> None:
    """Load the same starting checkpoint twice and launch one TRL GOLD run."""
    setting = plan.config["setting"]
    loss = plan.config["loss"]
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != setting["target_world_size"]:
        raise RuntimeError(
            f"launch has WORLD_SIZE={actual_world_size}; config requires {setting['target_world_size']}"
        )
    # Only rank 0 checks persisted artifacts. Otherwise a slower rank could
    # mistake the current run's freshly written manifest for an old run.
    if int(os.environ.get("RANK", "0")) == 0:
        for filename in ("run_manifest.json", "training_completed.json", "proxy_validation_metrics.jsonl"):
            if (plan.output_dir / filename).exists():
                raise RuntimeError(f"output_dir already contains {filename}; proxy v1 cannot resume or reuse a run")
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
        from trl.experimental.gold import GOLDConfig
    except ImportError as error:
        raise RuntimeError("install the optional training dependencies with `uv sync --extra train`") from error

    from .proxy_evaluation import ProxyValidationCallback
    from .proxy_trainer import ProxyGOLDTrainer

    model_id = plan.config["model_id"]
    revision = plan.config.get("model_revision")
    if not Path(model_id).is_dir():
        revision = HfApi().model_info(model_id, revision=revision).sha
    load_kwargs = {"revision": revision} if revision else {}
    dtype = torch.bfloat16 if setting["bf16"] else (torch.float16 if setting["fp16"] else torch.float32)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left", **load_kwargs)
    template_sha256 = configure_nonthinking_template(processor)
    image_processor = processor.image_processor
    image_size = dict(image_processor.size)
    image_size["longest_edge"] = setting["max_pixels"]
    if image_size.get("shortest_edge", 0) > setting["max_pixels"]:
        raise ValueError("max_pixels must be at least the image processor shortest_edge")
    image_processor.size = image_size
    # AMP fp16 needs FP32 trainable parameters/gradients for GradScaler.
    student_dtype = torch.float32 if setting["fp16"] else dtype
    student = AutoModelForImageTextToText.from_pretrained(model_id, dtype=student_dtype, **load_kwargs)
    teacher = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **load_kwargs)
    generation_terminators = configure_generation_terminators(student, processor.tokenizer)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    dataset = build_proxy_dataset(plan.train_rows)
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
        top_k=0,
        max_length=None,
        max_completion_length=setting["max_completion_length"],
        max_grad_norm=setting["max_grad_norm"],
        bf16=setting["bf16"],
        fp16=setting["fp16"],
        gradient_checkpointing=setting["gradient_checkpointing"],
        disable_dropout=True,
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
    trainer = ProxyGOLDTrainer(
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
        proxy_config=plan.config["proxy"],
        reward_config=plan.config["reward"],
        teacher_config=plan.config["teacher"],
    )
    generation_terminators = restore_training_generation_terminators(trainer, generation_terminators)
    initial_scale = setting.get("fp16_initial_scale")
    if initial_scale is not None:
        accelerator = trainer.accelerator
        if (
            not getattr(accelerator, "native_amp", False)
            or getattr(accelerator, "mixed_precision", None) != "fp16"
            or getattr(getattr(accelerator, "device", None), "type", None) != "cuda"
        ):
            raise RuntimeError("fp16_initial_scale requires CUDA native AMP")
        if getattr(accelerator, "_optimizers", ()) or getattr(trainer, "optimizer", None) is not None:
            raise RuntimeError("fp16 scaler must be set before accelerator prepares the optimizer")
        scaler = torch.amp.GradScaler("cuda", init_scale=float(initial_scale))
        if not scaler.is_enabled():
            raise RuntimeError("configured FP16 GradScaler is disabled")
        accelerator.scaler = scaler

        class VerifyPreparedScaler(TrainerCallback):
            """Fail before step 1 if AcceleratedOptimizer uses another scaler."""

            def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                prepared = getattr(accelerator, "_optimizers", ())
                if not prepared or any(getattr(optimizer, "scaler", None) is not scaler for optimizer in prepared):
                    raise RuntimeError("prepared optimizer does not share the configured FP16 scaler")
                return control

        trainer.add_callback(VerifyPreparedScaler())
    validation = plan.config["validation"]
    trainer.add_callback(ProxyValidationCallback(
        trainer=trainer,
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
    run_record["generation_terminators"] = generation_terminators
    run_record["chat_template_sha256"] = template_sha256
    run_record["enable_thinking"] = False
    run_record["confidence_report"] = {"minimum": 0, "maximum": 10, "integer": True, "normalization_divisor": 10}
    run_record["teacher_conditioning"] = {
        "privileged": "separate teacher template, paired teacher image, optional teacher_evidence",
        "baseline": "same EMA model with exact student prompt IDs and student image; no privileged evidence",
        "gap_scope": "image/evidence/template conditioning plus model drift; JS subtraction is heuristic",
    }
    run_record["sampling"] = {
        "temperature": setting["rollout_temperature"],
        "top_p": setting["rollout_top_p"],
        "top_k": 0,
    }
    run_record["fp16_initial_scale"] = (
        float(trainer.accelerator.scaler.get_scale()) if setting["fp16"] else None
    )
    run_record["source_commit"], run_record["source_worktree_dirty"] = _source_provenance()
    run_record["source_files_sha256"] = {
        path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    run_record["runtime_versions"] = _runtime_versions()
    run_record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda = getattr(torch, "cuda", None)
    run_record["device_name"] = cuda.get_device_name(trainer.accelerator.device) if cuda is not None and cuda.is_available() else "cpu"
    if trainer.accelerator.is_main_process:
        from .proxy_tracking import ProxyTrackingCallback
        tracker = ExperimentTracker(plan.config.get("tracking"), output_dir=plan.output_dir,
                                    run_config=run_record)
        tracking_holder["tracker"] = tracker
        trainer.experiment_tracker = tracker
        if tracker.enabled:
            trainer.add_callback(ProxyTrackingCallback(trainer, tracker))
            run_record["tracking_run_url"] = tracker.run_url
    if trainer.accelerator.is_main_process:
        with (plan.output_dir / "run_manifest.json").open("x", encoding="utf-8") as manifest:
            manifest.write(json.dumps(run_record, ensure_ascii=False, indent=2) + "\n")
    result = trainer.train()
    optimizer_steps = int(trainer.state.global_step)
    evidence = trainer.optimizer_evidence.summary()
    if optimizer_steps != int(result.global_step):
        raise RuntimeError(
            f"optimizer step evidence disagrees: state={optimizer_steps}, result={result.global_step}"
        )
    if setting["max_steps"] > 0 and optimizer_steps != setting["max_steps"]:
        raise RuntimeError(
            f"training stopped at {optimizer_steps} optimizer steps, expected {setting['max_steps']}"
        )
    if evidence["optimizer_successful_updates"] != optimizer_steps:
        raise RuntimeError(
            f"only {evidence['optimizer_successful_updates']} successful updates in {optimizer_steps} attempted steps; "
            f"skipped={evidence['optimizer_skipped_updates']}"
        )
    if plan.config["teacher"]["mode"] == "ema" and evidence["teacher_ema_updates"] != optimizer_steps:
        raise RuntimeError("EMA update count does not match successful optimizer updates")
    trainer.save_model()
    # Trainer.save_model usually saves processing_class too; do it explicitly
    # so a reloaded checkpoint retains the prompt without a prefixed think.
    if trainer.accelerator.is_main_process:
        processor.save_pretrained(plan.output_dir)
        validation_path = plan.output_dir / "proxy_validation_metrics.jsonl"
        if not validation_path.is_file():
            raise RuntimeError("frozen validation did not write its metrics")
        validation_reports = [
            json.loads(line)
            for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validation_steps = [int(report["optimizer_step"]) for report in validation_reports]
        if 0 not in validation_steps or optimizer_steps not in validation_steps:
            raise RuntimeError(
                f"frozen validation lacks step 0 or step {optimizer_steps}: {validation_steps}"
            )
        final_report = next(report for report in reversed(validation_reports) if int(report["optimizer_step"]) == optimizer_steps)
        if (final_report.get("actual_successful_optimizer_updates") != optimizer_steps
                or final_report.get("optimizer_state_max_step") != optimizer_steps
                or final_report.get("optimizer_step_count_agrees") is not True):
            raise RuntimeError("final validation optimizer evidence disagrees with Adam state")
        completed = {
            **run_record,
            "status": "completed",
            "optimizer_steps": optimizer_steps,
            "train_output_global_step": int(result.global_step),
            "validation_optimizer_steps": validation_steps,
            "train_metrics": result.metrics,
            "optimizer_evidence": evidence,
            "fp16_final_scale": (
                float(trainer.accelerator.scaler.get_scale()) if setting["fp16"] else None
            ),
        }
        final_teacher = trainer.accelerator.unwrap_model(trainer.teacher_model)
        final_teacher.save_pretrained(plan.output_dir / "teacher_final")
        (plan.output_dir / "training_completed.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sure-VL internal visual proxy TRL GOLD/OPSD trainer")
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
    except (OSError, ValueError, ProxyProtocolError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
