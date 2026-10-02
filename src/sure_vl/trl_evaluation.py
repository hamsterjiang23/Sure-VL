"""Frozen, labeled validation for the TRL vision-language training run.

The subset is selected once by a stable ID hash, then evaluated with the
student image and the same prompt, parser, and verifier used in training.
Only complete :class:`Example` labels are accepted. Generation/runtime errors
abort evaluation; malformed model text is recorded as an output attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import audit_attempts
from .objective import score
from .protocol import Example, OutputAttempt, ProtocolError, verify
from .trl_prompt import build_user_prompt, parse_student_completion, split_generated_eos

try:
    from transformers import TrainerCallback
except ImportError:  # Keep the protocol-only install usable without transformers.
    TrainerCallback = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class FrozenEvaluation:
    report: dict[str, Any]
    attempts: tuple[dict[str, Any], ...]


def _example_from_row(row: Mapping[str, Any]) -> Example:
    if not isinstance(row, Mapping) or not isinstance(row.get("example_payload"), str):
        raise ProtocolError("validation row requires a serialized, labeled example_payload")
    try:
        example = Example.from_dict(json.loads(row["example_payload"]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid validation example_payload: {error}") from error
    if row.get("example_id") != example.id:
        raise ProtocolError(f"validation row ID differs from its labels: {example.id}")
    if row.get("split") != example.split:
        raise ProtocolError(f"validation row split differs from its labels: {example.id}")
    if "image" not in row or row["image"] is None:
        raise ProtocolError(f"validation row lacks a student image: {example.id}")
    if isinstance(row["image"], (str, os.PathLike)) and Path(row["image"]).resolve() != Path(example.student_image).resolve():
        raise ProtocolError(f"validation image differs from the labeled student image: {example.id}")
    return example


def select_frozen_subset(
    validation_rows: Sequence[Mapping[str, Any]], *, size: int, seed: int
) -> tuple[dict[str, Any], ...]:
    """Choose an order-independent subset before observing any model output."""
    if type(size) is not int or size <= 0:
        raise ValueError("validation subset size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("validation subset seed must be a nonnegative integer")
    if not validation_rows:
        raise ProtocolError("validation rows are empty")
    examples = [_example_from_row(row) for row in validation_rows]
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        raise ProtocolError("validation rows have duplicate IDs")
    if len({example.split for example in examples}) != 1:
        raise ProtocolError("validation rows must belong to one split")
    ranked = sorted(
        zip(validation_rows, examples, strict=True),
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item[1].id}".encode()).hexdigest(),
            item[1].id,
        ),
    )
    return tuple(dict(row) for row, _ in ranked[:size])


def subset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint selected labels and image paths for cross-step comparison."""
    payloads = [row["example_payload"] for row in rows]
    return hashlib.sha256("\n".join(payloads).encode("utf-8")).hexdigest()


def attempt_from_text(example: Example, text: str) -> tuple[OutputAttempt, dict[str, Any]]:
    """Parse one raw completion and preserve the evidence behind the audit."""
    record: dict[str, Any] = {
        "id": example.id,
        "raw_completion": text,
        "parsed": False,
        "format_error": None,
        "visual_correct": None,
        "answer_correct": None,
    }
    try:
        parsed = parse_student_completion(example, text)
    except ProtocolError as error:
        record["format_error"] = str(error)
        return OutputAttempt(example.id, None, str(error)), record

    output = parsed.output
    checked = verify(example, output)
    reward = score(
        int(checked.visual_correct),
        int(checked.answer_correct),
        output.visual_confidence / 100.0,
        output.conditional_answer_confidence / 100.0,
    )
    record.update({
        "parsed": True,
        "visual_facts": dict(output.visual_facts),
        "answer": output.answer,
        "visual_confidence": output.visual_confidence,
        "conditional_answer_confidence": output.conditional_answer_confidence,
        "per_fact_correct": dict(checked.per_fact),
        "visual_correct": checked.visual_correct,
        "answer_correct": checked.answer_correct,
        "reward": {
            "utility": reward.utility,
            "calibration": reward.calibration,
            "total": reward.total,
        },
    })
    return OutputAttempt(example.id, output, None), record


def _student_image(value: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("frozen validation requires Pillow from the train extra") from error
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            return source.convert("RGB")
    raise ProtocolError("validation image must be a path or PIL image")


def _generation_seed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def evaluate_frozen_subset(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
) -> FrozenEvaluation:
    """Generate one sampled answer per frozen example with stable per-ID seeds.

    This is a held-out readback, not an optimizer update. The processor's
    multimodal prompt and generated-token decode mirror the training path.
    """
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not rows:
        raise ProtocolError("validation subset is empty")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("frozen validation requires torch from the train extra") from error

    examples = [_example_from_row(row) for row in rows]
    if len({example.id for example in examples}) != len(examples):
        raise ProtocolError("validation subset has duplicate IDs")
    if len({example.split for example in examples}) != 1:
        raise ProtocolError("validation subset must have one split")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor must expose its tokenizer")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters to determine its device") from error
    gpu_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    model.eval()
    attempts: list[OutputAttempt] = []
    records: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for row, example in zip(rows, examples, strict=True):
                image = _student_image(row["image"])
                prompt = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": build_user_prompt(example)},
                    ],
                }]
                prompt_text = processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                encoded = processor(
                    images=[[image]], text=[prompt_text], padding=True, return_tensors="pt"
                )
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                prompt_length = encoded["input_ids"].shape[1]
                with torch.random.fork_rng(devices=gpu_devices):
                    torch.manual_seed(_generation_seed(seed, example.id))
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        max_new_tokens=max_new_tokens,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        pad_token_id=(
                            tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        ),
                    )
                sequences = getattr(generated, "sequences", generated)
                if sequences.ndim != 2 or sequences.shape[0] != 1:
                    raise RuntimeError(f"generation returned invalid batch shape for {example.id}")
                generated_ids = sequences[0, prompt_length:].tolist()
                body_ids, eos_ids = split_generated_eos(
                    tokenizer, generated_ids,
                    generation_eos_token_id=getattr(
                        getattr(model, "generation_config", None), "eos_token_id", None
                    ),
                )
                text = tokenizer.decode(
                    list(body_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                attempt, record = attempt_from_text(example, text)
                record["generated_token_count"] = len(generated_ids)
                record["ended_with_eos"] = bool(eos_ids)
                record["hit_max_new_tokens"] = len(generated_ids) >= max_new_tokens and not eos_ids
                attempts.append(attempt)
                records.append(record)
    finally:
        model.train(was_training)

    report = audit_attempts(examples, attempts)
    report["subset_sha256"] = subset_sha256(rows)
    report["generation"] = {
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "sample_count": len(examples),
        "hit_max_new_tokens_count": sum(bool(record["hit_max_new_tokens"]) for record in records),
    }
    return FrozenEvaluation(report, tuple(records))


class FrozenValidationCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Evaluate at step 0, a fixed interval, the requested final step, and end.

    In distributed training every rank enters the same barriers; only rank 0
    generates and writes. Other ranks wait while rank 0 reads the fixed split.
    """

    def __init__(
        self,
        *,
        validation_rows: Sequence[Mapping[str, Any]],
        processor: Any,
        output_dir: str | Path,
        subset_size: int = 64,
        subset_seed: int = 42,
        every_n_steps: int = 20,
        final_step: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> None:
        if type(every_n_steps) is not int or every_n_steps <= 0:
            raise ValueError("every_n_steps must be a positive integer")
        if final_step is not None and (type(final_step) is not int or final_step <= 0):
            raise ValueError("final_step must be a positive integer or None")
        self.rows = select_frozen_subset(validation_rows, size=subset_size, seed=subset_seed)
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.final_step = final_step
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = subset_seed
        self._evaluated_steps: set[int] = set()

    def _evaluate(self, state: Any, model: Any) -> None:
        step = int(state.global_step)
        if step in self._evaluated_steps:
            return
        try:
            import torch.distributed as dist
        except ImportError:
            dist = None
        distributed = dist is not None and dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()
        try:
            if bool(state.is_world_process_zero):
                if model is None:
                    raise RuntimeError("Trainer did not pass model to validation callback")
                result = evaluate_frozen_subset(
                    model, self.processor, self.rows,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                attempt_path = self.output_dir / f"validation_attempts_step_{step:06d}.jsonl"
                temporary = attempt_path.with_suffix(".jsonl.tmp")
                with temporary.open("w", encoding="utf-8") as output:
                    for record in result.attempts:
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                temporary.replace(attempt_path)
                report = dict(result.report)
                report.update({
                    "optimizer_step": step,
                    "subset_ids": [row["example_id"] for row in self.rows],
                    "attempts_path": str(attempt_path),
                })
                with (self.output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
            self._evaluated_steps.add(step)
        finally:
            if distributed:
                dist.barrier()

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._evaluate(state, kwargs.get("model"))
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step > 0 and (step % self.every_n_steps == 0 or step == self.final_step):
            self._evaluate(state, kwargs.get("model"))
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if int(state.global_step) not in self._evaluated_steps:
            self._evaluate(state, kwargs.get("model"))
        return control
