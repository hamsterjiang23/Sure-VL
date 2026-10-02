"""Fixed-subset readback for the teacher-grounded visual proxy trainer.

Generation and scoring use the same sampled completion IDs. The Student
forward pass scores token ``t`` from logit position ``prompt_len + t - 1``;
the trainer's ``measure_rollout`` scores the paired clear-view Teacher along
that very prefix. This module performs no optimizer update.
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

from .proxy_metrics import evaluate_proxy_records
from .proxy_prompt import split_proxy_generated_eos

try:
    from transformers import TrainerCallback
except ImportError:  # Keep the lightweight protocol install importable.
    TrainerCallback = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ProxyEvaluation:
    report: dict[str, Any]
    attempts: tuple[dict[str, Any], ...]


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id", row.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("validation row requires a nonempty example_id")
    return value


def select_proxy_subset(
    validation_rows: Sequence[Mapping[str, Any]], *, size: int, seed: int,
) -> tuple[dict[str, Any], ...]:
    """Select by ID hash before observing model outputs; input order is irrelevant."""
    if type(size) is not int or size <= 0:
        raise ValueError("subset_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("subset_seed must be a nonnegative integer")
    if not validation_rows:
        raise ValueError("validation_rows must be nonempty")
    rows = [dict(row) for row in validation_rows]
    ids = [_row_id(row) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("validation rows contain duplicate example IDs")
    splits = {row.get("split") for row in rows}
    if len(splits) > 1:
        raise ValueError("validation rows must belong to one split")
    for row in rows:
        for required in ("prompt", "image", "teacher_image", "example_payload"):
            if required not in row or row[required] is None:
                raise ValueError(f"{_row_id(row)} validation row lacks {required}")
        payload = row["example_payload"]
        try:
            labeled = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError as error:
            raise ValueError(f"{_row_id(row)} has invalid example_payload JSON") from error
        if not isinstance(labeled, Mapping) or labeled.get("id") != _row_id(row):
            raise ValueError(f"{_row_id(row)} validation ID differs from example_payload")
        if row.get("split") is not None and labeled.get("split") is not None and row["split"] != labeled["split"]:
            raise ValueError(f"{_row_id(row)} validation split differs from example_payload")
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{_row_id(row)}".encode()).hexdigest(),
            _row_id(row),
        ),
    )
    return tuple(ranked[:size])


def _subset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    frozen = [
        {"id": _row_id(row), "example_payload": row["example_payload"]}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _image(value: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("proxy validation requires Pillow") from error
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            return source.convert("RGB")
    raise ValueError("validation image must be a path or PIL image")


def _generation_seed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _optimizer_state_step(trainer: Any) -> int | None:
    optimizer = getattr(trainer, "optimizer", None)
    states = getattr(optimizer, "state", None)
    if states is None:
        return None
    steps: list[int] = []
    for state in states.values():
        if not isinstance(state, Mapping) or "step" not in state:
            continue
        try:
            steps.append(int(state["step"]))
        except (TypeError, ValueError):
            continue
    if steps:
        return max(steps)
    state = getattr(trainer, "state", None)
    return 0 if getattr(state, "global_step", None) == 0 else None


def evaluate_proxy_subset(
    trainer: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
) -> ProxyEvaluation:
    """Read back a fixed held-out subset with aligned multimodal logits."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not rows:
        raise ValueError("validation subset must be nonempty")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("proxy validation requires torch") from error
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor must expose a tokenizer")
    model = trainer.accelerator.unwrap_model(trainer.model)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("student model has no parameters") from error
    gpu_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ] if device.type == "cuda" else []
    was_training = model.training
    model.eval()
    attempts: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for row in rows:
                example_id = _row_id(row)
                student_image = _image(row["image"])
                teacher_image = _image(row["teacher_image"])
                scoring_row = {
                    **row,
                    "student_image": student_image,
                    "teacher_image": teacher_image,
                }
                images, prepared_prompts = trainer._extract_images_and_prompts([{
                    "prompt": row["prompt"], "image": student_image,
                }])
                prompt_texts = processor.apply_chat_template(
                    prepared_prompts, tokenize=False, add_generation_prompt=True,
                )
                if isinstance(prompt_texts, str):
                    prompt_texts = [prompt_texts]
                encoded = processor(
                    images=images,
                    text=prompt_texts,
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                input_ids = encoded["input_ids"]
                if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                    raise RuntimeError(f"invalid tokenized prompt shape for {example_id}")
                prompt_length = int(input_ids.shape[1])
                if prompt_length == 0:
                    raise RuntimeError(f"empty prompt for {example_id}")
                scoring_row["_student_prompt_ids"] = input_ids[0, encoded["attention_mask"][0].bool()]
                with torch.random.fork_rng(devices=gpu_devices):
                    torch.manual_seed(_generation_seed(seed, example_id))
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
                if sequences.ndim != 2 or sequences.shape[0] != 1 or sequences.shape[1] < prompt_length:
                    raise RuntimeError(f"invalid generation shape for {example_id}")
                if not torch.equal(sequences[0, :prompt_length], input_ids[0]):
                    raise RuntimeError(f"generation changed the prompt prefix for {example_id}")
                completion_ids = sequences[0, prompt_length:]
                completion_length = int(completion_ids.numel())
                forward_inputs = dict(encoded)
                forward_inputs["input_ids"] = sequences
                forward_inputs["attention_mask"] = torch.cat((
                    encoded["attention_mask"],
                    encoded["attention_mask"].new_ones((1, completion_length)),
                ), dim=1)
                for key in ("mm_token_type_ids", "token_type_ids"):
                    if key in encoded:
                        prefix = encoded[key]
                        forward_inputs[key] = torch.cat((
                            prefix,
                            prefix.new_zeros((*prefix.shape[:-1], completion_length)),
                        ), dim=-1)
                student_outputs = model(**forward_inputs, use_cache=False)
                selected_logits = student_outputs.logits[
                    0, prompt_length - 1 : prompt_length - 1 + completion_length, :
                ]
                if selected_logits.shape[0] != completion_length:
                    raise RuntimeError(f"Student logits do not align with completion for {example_id}")
                measured = trainer.measure_rollout(
                    scoring_row, completion_ids, selected_logits, diagnostics=False,
                )
                record = dict(measured.record)
                if "id" in record and record["id"] != example_id:
                    raise RuntimeError(f"measured ID differs from validation row: {example_id}")
                body_ids, eos_ids = split_proxy_generated_eos(
                    tokenizer, completion_ids.tolist(),
                    generation_eos_token_id=getattr(
                        getattr(model, "generation_config", None), "eos_token_id", None
                    ),
                )
                record.update({
                    "id": example_id,
                    "raw_completion": tokenizer.decode(
                        list(body_ids), skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "generated_ids": completion_ids.tolist(),
                    "generated_token_count": completion_length,
                    "ended_with_eos": bool(eos_ids),
                    "hit_max_new_tokens": completion_length >= max_new_tokens and not eos_ids,
                })
                attempts.append(record)
    finally:
        model.train(was_training)
    report = evaluate_proxy_records(attempts)
    report["subset_sha256"] = _subset_sha256(rows)
    report["generation"] = {
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "sample_count": len(attempts),
        "hit_max_new_tokens_count": sum(attempt["hit_max_new_tokens"] for attempt in attempts),
    }
    return ProxyEvaluation(report=report, attempts=tuple(attempts))


class ProxyValidationCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Evaluate at step 0, an interval, the requested final step, and train end."""

    def __init__(
        self,
        *,
        trainer: Any,
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
        self.trainer = trainer
        self.rows = select_proxy_subset(validation_rows, size=subset_size, seed=subset_seed)
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.seed = subset_seed
        self.every_n_steps = every_n_steps
        self.final_step = final_step
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._evaluated_steps: set[int] = set()

    def _evaluate(self, state: Any) -> None:
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
        failure: Exception | None = None
        failure_text: str | None = None
        if bool(state.is_world_process_zero):
            try:
                result = evaluate_proxy_subset(
                    self.trainer, self.processor, self.rows,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                attempt_path = self.output_dir / f"proxy_validation_attempts_step_{step:06d}.jsonl"
                temporary = attempt_path.with_suffix(".jsonl.tmp")
                with temporary.open("w", encoding="utf-8") as output:
                    for record in result.attempts:
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                temporary.replace(attempt_path)
                evidence_callback = getattr(self.trainer, "optimizer_evidence", None)
                optimizer_evidence = (
                    evidence_callback.summary() if evidence_callback is not None else {}
                )
                adam_step = _optimizer_state_step(self.trainer)
                report = dict(result.report)
                report.update({
                    "optimizer_step": step,
                    "trainer_state_global_step": step,
                    "actual_successful_optimizer_updates": optimizer_evidence.get(
                        "optimizer_successful_updates"
                    ),
                    "optimizer_evidence": optimizer_evidence,
                    "optimizer_state_max_step": adam_step,
                    "optimizer_step_count_agrees": (
                        optimizer_evidence["optimizer_successful_updates"] == adam_step
                        if adam_step is not None and "optimizer_successful_updates" in optimizer_evidence
                        else None
                    ),
                    "requested_final_step": self.final_step,
                    "reached_requested_final_step": self.final_step is not None and step >= self.final_step,
                    "subset_ids": [_row_id(row) for row in self.rows],
                    "attempts_path": str(attempt_path),
                })
                with (self.output_dir / "proxy_validation_metrics.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as error:
                failure = error
                failure_text = f"{type(error).__name__}: {error}"
        if distributed:
            messages = [failure_text]
            dist.broadcast_object_list(messages, src=0)
            failure_text = messages[0]
            dist.barrier()
        if failure_text is not None:
            if failure is not None:
                raise failure
            raise RuntimeError(f"rank-0 proxy validation failed: {failure_text}")
        self._evaluated_steps.add(step)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._evaluate(state)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step > 0 and (step % self.every_n_steps == 0 or step == self.final_step):
            self._evaluate(state)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if int(state.global_step) not in self._evaluated_steps:
            self._evaluate(state)
        return control
