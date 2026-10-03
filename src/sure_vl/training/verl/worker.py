"""Single-GPU, genuine veRL Ray Worker for Sure-VL's proxy objective.

This is a custom Worker backend, not veRL's PPO/FSDP/vLLM trainer.  The model,
HF multimodal rollout, detached pre-update p/q+/q- score, differentiable joint
loss, Adam update, and gated EMA all run inside one Ray actor.  DataProto is the
only per-batch RPC protocol; the first version deliberately rejects world > 1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register

from ...proxy_protocol import ProxyExample
from .objective import (
    ScoredProxyRollout,
    prepare_proxy_rollout,
    score_proxy_rollout,
)


def _group_size(config: Mapping[str, Any]) -> int:
    setting = config["setting"]
    size = setting["num_generations"]
    if type(size) is not int or size < 2 or any(setting[key] != size for key in (
        "gradient_accumulation_steps", "generation_batch_size",
    )):
        raise ValueError("veRL group mode requires num_generations=accumulation=generation_batch_size>=2")
    return size


def _object_array(items: list[Any]) -> np.ndarray:
    result = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        result[index] = item
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _example(payload: Any) -> ProxyExample:
    if isinstance(payload, ProxyExample):
        return payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ProxyExample.from_dict(payload)


def _completions(proto: DataProto, size: int) -> tuple[tuple[int, ...], ...]:
    if proto.batch is None or "completion_ids" not in proto.batch or "completion_mask" not in proto.batch:
        raise ValueError("DataProto requires completion_ids and completion_mask tensors")
    ids, mask = proto.batch["completion_ids"], proto.batch["completion_mask"]
    if ids.ndim != 2 or ids.shape[0] != size or ids.shape != mask.shape or ids.shape[1] < 1:
        raise ValueError("veRL grouped completion tensor has wrong shape")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("completion_mask must be binary")
    result = []
    for row, active in zip(ids, mask, strict=True):
        n = int(active.sum())
        if n < 1 or not bool(active[:n].all()) or bool(active[n:].any()):
            raise ValueError("completion_mask must be a nonempty prefix")
        result.append(tuple(int(item) for item in row[:n].tolist()))
    return tuple(result)


def _pad_completion_group(ids: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, ids))
    values = torch.zeros((len(ids), width), dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.bool)
    for row_index, row in enumerate(ids):
        values[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[row_index, :len(row)] = True
    return values, mask


def _single_payload(proto: DataProto) -> ProxyExample:
    raw = proto.non_tensor_batch.get("example_payload")
    if raw is None or len(raw) != 1:
        raise ValueError("DataProto requires exactly one example_payload")
    return _example(raw[0])


def _check_single(proto: DataProto) -> None:
    if not isinstance(proto, DataProto) or len(proto) != 1:
        raise ValueError("first veRL backend requires DataProto batch size 1")


@dataclass
class _PendingGeneration:
    nonce: str
    model_version: int
    example_digest: str
    completion_ids: tuple[tuple[int, ...], ...]
    student_prompt_ids: tuple[int, ...]
    student_prompt_sha256: str


@dataclass
class _PendingScore:
    nonce: str
    model_version: int
    example_digest: str
    completion_ids: tuple[tuple[int, ...], ...]
    student_prompt_ids: tuple[int, ...]
    student_prompt_sha256: str
    scored: tuple[ScoredProxyRollout, ...]
    clear_teacher_logits: tuple[torch.Tensor | None, ...]  # detached CPU cache, fixed until update
    content_advantages: tuple[float, ...]
    report_advantages: tuple[float, ...]


@dataclass(frozen=True)
class _GroupJoint:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    opsd_loss: torch.Tensor
    score_function_token_mean_diagnostic: torch.Tensor
    content_mean_nll: torch.Tensor | None
    report_mean_nll: torch.Tensor | None
    policy_clipfrac: float
    policy_approx_kl: float
    policy_clipfrac_lower: float


@dataclass(frozen=True)
class _VanillaPolicyConfig:
    """Small adapter for official veRL's policy-loss function, world size one."""

    global_batch_size: int
    clip_ratio: float = 0.2
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None

    @property
    def global_batch_info(self) -> dict[str, int]:
        return {"dp_size": 1, "global_batch_size": self.global_batch_size}

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


class SureVLProxyWorker(Worker):
    """One Ray actor owns the full student, EMA teacher, optimizer, and data."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        if self.world_size != 1 or self.rank != 0:
            raise ValueError("Sure-VL veRL v1 supports world_size=1 only")
        self.config = dict(config)
        self.runtime: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: Any | None = None
        self._generated: _PendingGeneration | None = None
        self._scored: _PendingScore | None = None
        self.attempted_steps = 0
        self.successful_updates = 0
        self.skipped_updates = 0
        self.teacher_ema_updates = 0
        self.model_version = 0

    def _make_runtime(self) -> Any:
        from .vlm import VLMRuntime

        return VLMRuntime(self.config)

    def _require_ready(self) -> Any:
        if self.runtime is None or self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("initialize must finish before a worker RPC")
        return self.runtime

    def _check_idle(self) -> None:
        if self._generated is not None or self._scored is not None:
            raise RuntimeError("pending sampled rollout must be scored and updated first")

    def _evidence(self) -> dict[str, Any]:
        optimizer_state_max_step = self._adam_max_step()
        return {
            "optimizer_attempted_steps": self.attempted_steps,
            "optimizer_successful_updates": self.successful_updates,
            "optimizer_skipped_updates": self.skipped_updates,
            "teacher_ema_updates": self.teacher_ema_updates,
            "teacher_mode": self.config["teacher"]["mode"],
            "model_version": self.model_version,
            "adam_max_step": optimizer_state_max_step,
            "optimizer_state_max_step": optimizer_state_max_step,
            "scheduler_last_epoch": (
                None if self.lr_scheduler is None else int(self.lr_scheduler.last_epoch)
            ),
        }

    def _adam_max_step(self) -> int:
        if self.optimizer is None:
            return 0
        steps = []
        for state in self.optimizer.state.values():
            step = state.get("step")
            if step is not None:
                steps.append(int(step.item() if isinstance(step, torch.Tensor) else step))
        return max(steps, default=0)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self) -> dict[str, Any]:
        if self.runtime is not None:
            raise RuntimeError("worker is already initialized")
        setting = self.config["setting"]
        if setting["target_world_size"] != 1 or setting["per_device_train_batch_size"] != 1:
            raise ValueError("veRL group mode requires one GPU and per-device microbatch size 1")
        if type(setting.get("max_steps")) is not int or setting["max_steps"] < 1:
            raise ValueError("veRL linear schedule requires a positive max_steps budget")
        group_size = _group_size(self.config)
        if setting.get("bf16") or setting.get("fp16"):
            raise ValueError("veRL v1 is FP32 only")
        if setting["rollout_temperature"] != 1 or setting["rollout_top_p"] != 1:
            raise ValueError("on-policy score function requires natural temperature and untruncated sampling")
        if self.config["teacher"]["mode"] not in {"ema", "fixed"}:
            raise ValueError("teacher.mode must be ema or fixed")
        if self.config["loss"].get("opsd_beta") != 0.0:
            raise ValueError("veRL v1 uses original OPSD beta=0 forward KL")
        grpo = self.config.get("grpo")
        if (not isinstance(grpo, Mapping) or grpo.get("loss_type") != "grpo"
                or grpo.get("scale_rewards") != "none" or grpo.get("num_iterations") != 1):
            raise ValueError("veRL group prototype requires one-iteration GRPO with unscaled group centering")
        runtime = self._make_runtime()
        if runtime.student is runtime.teacher:
            raise ValueError("student and teacher must be separate model objects")
        student_params = dict(runtime.student.named_parameters())
        teacher_params = dict(runtime.teacher.named_parameters())
        if student_params.keys() != teacher_params.keys():
            raise ValueError("teacher and student parameter names differ")
        for name, parameter in teacher_params.items():
            if parameter.shape != student_params[name].shape:
                raise ValueError(f"teacher parameter shape differs: {name}")
            parameter.requires_grad_(False)
        runtime.teacher.eval()
        # VLMRuntime disables dropout and keeps train mode for optional gradient
        # checkpointing; its generate() temporarily switches to eval mode.
        self.runtime = runtime
        self.optimizer = torch.optim.AdamW(
            runtime.student.parameters(), lr=float(setting["learning_rate"]),
            weight_decay=0.0,
        )
        from transformers import get_scheduler

        self.lr_scheduler = get_scheduler(
            "linear", optimizer=self.optimizer, num_warmup_steps=0,
            num_training_steps=setting["max_steps"],
        )
        evidence = self._evidence()
        return {
            "backend": "custom_veRL_RayWorkerGroup_HF",
            "world_size": self.world_size,
            "num_generations": group_size,
            "rank": self.rank,
            "device": str(runtime.device),
            "template_sha256": runtime.template_sha256,
            "model_id": self.config["model_id"],
            **evidence,
            "optimizer_evidence": evidence,
        }

    def _generation_parameters(self, meta: Mapping[str, Any], *, training: bool) -> dict[str, Any]:
        setting = self.config["setting"]
        validation = self.config.get("validation", {})
        parameters = {
            "max_new_tokens": int(meta.get(
                "max_new_tokens", setting["max_completion_length"] if training else validation["max_new_tokens"]
            )),
            "temperature": float(meta.get("temperature", 1.0 if training else validation["temperature"])),
            "top_p": float(meta.get("top_p", 1.0 if training else validation["top_p"])),
            "top_k": int(meta.get("top_k", 0)),
            "seed": int(meta.get("seed", setting["seed"])),
        }
        if parameters["max_new_tokens"] <= 0 or parameters["seed"] < 0:
            raise ValueError("generation length/seed is invalid")
        if training and (parameters["temperature"], parameters["top_p"], parameters["top_k"]) != (1.0, 1.0, 0):
            raise ValueError("training rollout must use the natural, untruncated student distribution")
        return parameters

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, requests: DataProto) -> DataProto:
        runtime = self._require_ready()
        self._check_idle()
        _check_single(requests)
        example = _single_payload(requests)
        parameters = self._generation_parameters(requests.meta_info, training=True)
        encoded = runtime.encode_student(example)
        ids_group = []
        with torch.no_grad():
            for generation_index in range(_group_size(self.config)):
                per_sample = dict(parameters, seed=parameters["seed"] + generation_index)
                generated = runtime.generate(encoded, per_sample)
                if generated.ndim != 1 or not generated.numel():
                    raise RuntimeError("HF rollout returned no generated tokens")
                ids_group.append(tuple(int(item) for item in generated.tolist()))
        ids = tuple(ids_group)
        padded_ids, completion_mask = _pad_completion_group(ids)
        nonce = uuid.uuid4().hex
        self._generated = _PendingGeneration(
            nonce=nonce, model_version=self.model_version,
            example_digest=_digest(example.to_dict()), completion_ids=ids,
            student_prompt_ids=tuple(int(item) for item in encoded.input_ids.tolist()),
            student_prompt_sha256=encoded.sha256,
        )
        return DataProto.from_dict(
            tensors={"completion_ids": padded_ids, "completion_mask": completion_mask},
            non_tensors={"example_payload": _object_array([example.to_dict()] * len(ids))},
            meta_info={"generation_nonce": nonce, "model_version": self.model_version,
                       "student_prompt_sha256": encoded.sha256,
                       "sample_seeds": [parameters["seed"] + i for i in range(len(ids))]},
        )

    def _teacher_conditioning(self, example: ProxyExample, student: Any, teacher: Any) -> dict[str, Any]:
        return {
            "privileged_image": example.teacher_image,
            "teacher_evidence_present": example.teacher_evidence is not None,
            "teacher_evidence_sha256": (
                None if example.teacher_evidence is None else _digest(example.teacher_evidence)
            ),
            "student_prompt_sha256": student.sha256,
            "teacher_prompt_sha256": teacher.sha256,
            "baseline_prompt_sha256": student.sha256,
            "baseline_image": example.student_image,
            "privileged_input": {"prompt_tokens": int(teacher.input_ids.numel())},
            "baseline_input": {"prompt_tokens": int(student.input_ids.numel())},
            "baseline": "same EMA, exact student prompt and image, no privileged evidence",
            "gap_scope": "combined image, evidence, teacher-template and parameter differences; heuristic baseline correction",
        }

    def _score_ids(self, example: ProxyExample, ids: tuple[int, ...], expected_prompt_ids: tuple[int, ...],
                   expected_prompt_sha256: str,
                   ) -> tuple[ScoredProxyRollout, torch.Tensor | None]:
        runtime = self._require_ready()
        student = runtime.encode_student(example)
        actual_prompt_ids = tuple(int(item) for item in student.input_ids.tolist())
        if actual_prompt_ids != expected_prompt_ids or student.sha256 != expected_prompt_sha256:
            raise RuntimeError("scoring student prompt or image tensors differ from generation")
        prepared = prepare_proxy_rollout(
            example, runtime.tokenizer, ids, eos_token_ids=runtime.eos_token_ids,
        )
        teacher = runtime.encode_teacher(example)
        sample_ids = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            student_logits = runtime.forward_response(runtime.student, student, sample_ids)
            clear_logits = None
            if prepared.content_count and (
                self.config["loss"]["opsd_weight"] > 0 or prepared.vision_positions
            ):
                clear_logits = runtime.forward_response(
                    runtime.teacher, teacher,
                    torch.tensor(prepared.content_ids, dtype=torch.long),
                ).detach()
            restricted_logits = None
            if prepared.vision_positions and self.config["proxy"]["lambda_b"] > 0:
                last = prepared.vision_positions[-1] + 1
                baseline = runtime.encode_student(example)
                if (not torch.equal(baseline.input_ids, student.input_ids)
                        or baseline.sha256 != student.sha256):
                    raise RuntimeError("q- must reuse exact encoded student prompt and image tensors")
                restricted_logits = runtime.forward_response(
                    runtime.teacher, baseline,
                    torch.tensor(ids[:last], dtype=torch.long),
                ).detach()
            scored = score_proxy_rollout(
                example, prepared, student_logits, clear_logits, restricted_logits,
                proxy_config=self.config["proxy"], reward_config=self.config["reward"],
                teacher_conditioning=self._teacher_conditioning(example, student, teacher),
            )
            # Persist the exact sampled token sequence for later zero-update
            # parity audits; decoding and re-tokenizing cannot recover it.
            scored.record["sampled_completion_ids"] = list(ids)
            scored.record["student_prompt_ids_sha256"] = _digest(actual_prompt_ids)
            scored.record["teacher_conditioning"]["privileged_forward_executed"] = clear_logits is not None
            scored.record["teacher_conditioning"]["baseline_forward_executed"] = restricted_logits is not None
            if clear_logits is not None and prepared.content_count:
                temperature = float(self.config["loss"]["opsd_temperature"])
                p_log = torch.log_softmax(student_logits[:prepared.content_count].float() / temperature, -1)
                q_log = torch.log_softmax(clear_logits.float() / temperature, -1)
                q = q_log.exp()
                raw = (q * (q_log - p_log)).sum(-1)
                scored.record["opsd"].update(
                    sampled_positions=float(prepared.content_count),
                    raw_forward_kl_mean=float(raw.mean().item()),
                    top1_disagreement_rate=float((p_log.argmax(-1) != q_log.argmax(-1)).float().mean().item()),
                )
        return scored, None if clear_logits is None else clear_logits.detach().cpu().clone()

    @staticmethod
    def _group_advantages(rewards: list[float], example_digest: str) -> tuple[float, ...]:
        """Use official veRL GRPO outcome centering on one prompt group."""
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        values = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("group rewards must be finite")
        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=values,
            response_mask=torch.ones_like(values),
            index=np.asarray([example_digest] * len(rewards), dtype=object),
            epsilon=1e-4,
            norm_adv_by_std_in_grpo=False,  # TRL config uses scale_rewards='none'.
        )
        result = tuple(float(value) for value in advantages[:, 0].tolist())
        if not all(math.isfinite(value) for value in result):
            raise FloatingPointError("official veRL group advantages are nonfinite")
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def score_sequences(self, rollout: DataProto) -> DataProto:
        self._require_ready()
        size = _group_size(self.config)
        if not isinstance(rollout, DataProto) or len(rollout) != size:
            raise ValueError("score requires one complete prompt group")
        if self._generated is None or self._scored is not None:
            raise RuntimeError("score requires exactly one pending generation")
        pending = self._generated
        raw_payloads = rollout.non_tensor_batch.get("example_payload")
        if raw_payloads is None or len(raw_payloads) != size:
            raise ValueError("score requires a payload per generated completion")
        example = _example(raw_payloads[0])
        if any(_digest(_example(value).to_dict()) != pending.example_digest for value in raw_payloads):
            raise RuntimeError("generated group must contain one unchanged prompt")
        ids = _completions(rollout, size)
        if (rollout.meta_info.get("generation_nonce") != pending.nonce
                or rollout.meta_info.get("model_version") != self.model_version
                or rollout.meta_info.get("student_prompt_sha256") != pending.student_prompt_sha256
                or pending.model_version != self.model_version
                or _digest(example.to_dict()) != pending.example_digest
                or ids != pending.completion_ids):
            raise RuntimeError("stale or altered sampled rollout")
        assessments = [self._score_ids(
            example, row_ids, pending.student_prompt_ids, pending.student_prompt_sha256,
        ) for row_ids in ids]
        scored = tuple(item[0] for item in assessments)
        cached_teacher = tuple(item[1] for item in assessments)
        content_advantages = self._group_advantages(
            [item.content_reward for item in scored], pending.example_digest,
        )
        report_advantages = self._group_advantages(
            [item.report_reward for item in scored], pending.example_digest,
        )
        content_mask = torch.zeros_like(rollout.batch["completion_mask"], dtype=torch.bool)
        report_mask = torch.zeros_like(content_mask)
        for index, item in enumerate(scored):
            width = len(ids[index])
            content_mask[index, :width] = torch.tensor(item.prepared.content_mask, dtype=torch.bool)
            report_mask[index, :width] = torch.tensor(item.prepared.report_mask, dtype=torch.bool)
        if not torch.equal(content_mask | report_mask, rollout.batch["completion_mask"].bool()):
            raise RuntimeError("group content/report masks do not partition sampled IDs")
        nonce = uuid.uuid4().hex
        self._scored = _PendingScore(
            nonce=nonce, model_version=self.model_version,
            example_digest=pending.example_digest, completion_ids=ids,
            student_prompt_ids=pending.student_prompt_ids,
            student_prompt_sha256=pending.student_prompt_sha256,
            scored=scored, clear_teacher_logits=cached_teacher,
            content_advantages=content_advantages, report_advantages=report_advantages,
        )
        self._generated = None
        return DataProto.from_dict(
            tensors={
                "completion_ids": rollout.batch["completion_ids"].detach().cpu(),
                "completion_mask": rollout.batch["completion_mask"].detach().cpu(),
                "content_mask": content_mask.cpu(),
                "report_mask": report_mask.cpu(),
                "content_reward": torch.tensor([item.content_reward for item in scored], dtype=torch.float32),
                "report_reward": torch.tensor([item.report_reward for item in scored], dtype=torch.float32),
                "content_advantage": torch.tensor(content_advantages, dtype=torch.float32),
                "report_advantage": torch.tensor(report_advantages, dtype=torch.float32),
            },
            non_tensors={
                "example_payload": _object_array([example.to_dict()] * size),
                "record_json": _object_array([
                    json.dumps(item.record, ensure_ascii=False, allow_nan=False) for item in scored
                ]),
            },
            meta_info={"score_nonce": nonce, "model_version": self.model_version,
                       "generation_nonce": pending.nonce, "teacher_version": self.teacher_ema_updates,
                       "student_prompt_sha256": pending.student_prompt_sha256,
                       "group_size": size},
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def audit_zero_update(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Compare raw cached-generation logits with exact-ID full-forward logits.

        This method is only valid after a generate→score RPC pair and does not
        call backward, the optimizer, the LR scheduler, or EMA.
        """
        runtime = self._require_ready()
        pending = self._scored
        if pending is None or self._generated is not None:
            raise RuntimeError("zero-update audit requires a scored rollout")
        if any((self.attempted_steps, self.successful_updates, self.teacher_ema_updates)):
            raise RuntimeError("zero-update audit requires untouched model and optimizer")
        if not isinstance(request, Mapping):
            raise TypeError("audit request must be a mapping")
        example = _example(request.get("example_payload"))
        if _digest(example.to_dict()) != pending.example_digest:
            raise RuntimeError("audit example differs from scored rollout")
        parameters = self._generation_parameters(request, training=True)
        encoded = runtime.encode_student(example)
        if (tuple(int(item) for item in encoded.input_ids.tolist()) != pending.student_prompt_ids
                or encoded.sha256 != pending.student_prompt_sha256):
            raise RuntimeError("audit prompt or image differs from sampled rollout")
        with torch.no_grad():
            replay_ids, raw_logits = runtime.generate_with_logits(encoded, parameters)
            expected = torch.tensor(pending.completion_ids[0], dtype=torch.long)
            if not torch.equal(replay_ids, expected):
                raise RuntimeError("audit replay changed actual first-sample token IDs")
            full_logits = runtime.forward_response(runtime.student, encoded, expected)
            raw_log_prob = torch.log_softmax(raw_logits.float(), dim=-1)
            full_log_prob = torch.log_softmax(full_logits.detach().cpu().float(), dim=-1)
            if raw_log_prob.shape != full_log_prob.shape:
                raise RuntimeError("cached generation and full-forward vocab shapes differ")
            difference = (raw_log_prob - full_log_prob).abs()
            selected = difference.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
            content_rewards = torch.tensor(
                [item.content_reward for item in pending.scored], dtype=torch.float32
            )
            report_rewards = torch.tensor(
                [item.report_reward for item in pending.scored], dtype=torch.float32
            )
        teacher_parameters = list(runtime.teacher.parameters())
        student_parameters = list(runtime.student.parameters())
        evidence = self._evidence()
        return {
            "sampled_completion_ids": expected.tolist(),
            "sampled_completion_sha256": _digest(expected.tolist()),
            "student_prompt_sha256": encoded.sha256,
            "teacher_prompt_sha256": pending.scored[0].record["teacher_conditioning"]["teacher_prompt_sha256"],
            "baseline_prompt_sha256": pending.scored[0].record["teacher_conditioning"]["baseline_prompt_sha256"],
            "generation_vs_forward_max_abs_logprob": float(difference.max().item()),
            "generation_vs_forward_mean_abs_logprob": float(difference.mean().item()),
            "generation_vs_forward_max_abs_selected_logprob": float(selected.max().item()),
            "generation_logits_shape": list(raw_log_prob.shape),
            "group_size": len(pending.scored),
            "privileged_forward_count": sum(
                bool(item.record["teacher_conditioning"]["privileged_forward_executed"])
                for item in pending.scored
            ),
            "baseline_forward_count": sum(
                bool(item.record["teacher_conditioning"]["baseline_forward_executed"])
                for item in pending.scored
            ),
            "nonfallback_proxy_count": sum(not item.record["proxy_fallback"] for item in pending.scored),
            "content_advantage_mean": float(torch.tensor(pending.content_advantages).mean().item()),
            "report_advantage_mean": float(torch.tensor(pending.report_advantages).mean().item()),
            "content_advantage_matches_centered_rewards": bool(torch.allclose(
                torch.tensor(pending.content_advantages),
                content_rewards - content_rewards.mean(), atol=1e-6,
            )),
            "report_advantage_matches_centered_rewards": bool(torch.allclose(
                torch.tensor(pending.report_advantages),
                report_rewards - report_rewards.mean(), atol=1e-6,
            )),
            "teacher_frozen": all(not parameter.requires_grad and parameter.grad is None
                                  for parameter in teacher_parameters),
            "student_gradients_absent": all(parameter.grad is None for parameter in student_parameters),
            "optimizer_evidence": evidence,
            "optimizer_updates": 0,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_zero_update_audit(self) -> dict[str, Any]:
        """Discard an audited rollout so the untouched Worker can close."""
        if any((self.attempted_steps, self.successful_updates, self.teacher_ema_updates)):
            raise RuntimeError("cannot discard an audit after an optimizer attempt")
        self._generated = None
        self._scored = None
        return {"discarded_without_update": True, "optimizer_evidence": self._evidence()}

    def _update_ema(self) -> None:
        if self.config["teacher"]["mode"] != "ema":
            return
        runtime = self._require_ready()
        decay = float(self.config["teacher"]["ema_decay"])
        with torch.no_grad():
            student_params = dict(runtime.student.named_parameters())
            for name, target in runtime.teacher.named_parameters():
                target.mul_(decay).add_(student_params[name].detach().to(target), alpha=1.0 - decay)
            student_buffers = dict(runtime.student.named_buffers())
            for name, target in runtime.teacher.named_buffers():
                if name in student_buffers and target.shape == student_buffers[name].shape:
                    target.copy_(student_buffers[name].detach())
        runtime.teacher.eval()
        self.teacher_ema_updates += 1

    def _update_result(self, pending: _PendingScore, *, status: str, joint: _GroupJoint | None,
                       grad_norm: float | None, grad_norm_after_clip: float | None,
                       learning_rate_used: float | None, learning_rate_next: float,
                       gradient_components: Mapping[str, Any]) -> DataProto:
        def finite_scalar(tensor: Any | None) -> float | None:
            if tensor is None:
                return None
            scalar = float(tensor.detach())
            return scalar if math.isfinite(scalar) else None

        content_tokens = sum(item.prepared.content_count for item in pending.scored)
        report_tokens = sum(sum(item.prepared.report_mask) for item in pending.scored)
        generated_tokens = sum(len(ids) for ids in pending.completion_ids)
        loss_diagnostics = {
            "score_function_token_mean_diagnostic": (
                None if joint is None else finite_scalar(joint.score_function_token_mean_diagnostic)
            ),
            "content_mean_nll": (None if joint is None or joint.content_mean_nll is None
                                 else finite_scalar(joint.content_mean_nll)),
            "report_mean_nll": (None if joint is None or joint.report_mean_nll is None
                                else finite_scalar(joint.report_mean_nll)),
            "content_tokens": content_tokens,
            "report_tokens": report_tokens,
            "generated_tokens": generated_tokens,
            "group_size": len(pending.scored),
            "gradient_components": dict(gradient_components),
        }
        records = []
        for index, item in enumerate(pending.scored):
            record = dict(item.record)
            record.update({
                "optimizer_status": status,
                "trainer_step_before_update": pending.model_version,
                "group_index": index,
                "group_size": len(pending.scored),
                "content_advantage": pending.content_advantages[index],
                "report_advantage": pending.report_advantages[index],
                "loss_diagnostics": loss_diagnostics,
            })
            if joint is not None and finite_scalar(joint.opsd_loss) is not None:
                record["opsd"] = {**record["opsd"], "clipped_group_loss_mean": finite_scalar(joint.opsd_loss)}
            records.append(record)
        def value(tensor: Any | None) -> float:
            return float("nan") if tensor is None else float(tensor.detach())

        metrics = {
            "policy_loss": finite_scalar(None if joint is None else joint.policy_loss),
            "opsd_loss": finite_scalar(None if joint is None else joint.opsd_loss),
            "joint_loss": finite_scalar(None if joint is None else joint.loss),
            "score_function_token_mean_diagnostic": finite_scalar(
                None if joint is None else joint.score_function_token_mean_diagnostic
            ),
            "grad_norm": grad_norm,
            "grad_norm_after_clip": grad_norm_after_clip,
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": learning_rate_next,
            "scheduler_last_epoch": (
                None if self.lr_scheduler is None else int(self.lr_scheduler.last_epoch)
            ),
            "optimizer_state_max_step": self._adam_max_step(),
            "content_tokens": content_tokens,
            "report_tokens": report_tokens,
            "generated_tokens": generated_tokens,
            "group_size": len(pending.scored),
            "policy_clipfrac": None if joint is None else joint.policy_clipfrac,
            "policy_approx_kl": None if joint is None else joint.policy_approx_kl,
            "policy_clipfrac_lower": None if joint is None else joint.policy_clipfrac_lower,
            "content_advantage_variance": float(torch.tensor(pending.content_advantages).var(unbiased=False)),
            "report_advantage_variance": float(torch.tensor(pending.report_advantages).var(unbiased=False)),
            "zero_advantage_group": all(
                abs(value) < 1e-12 for value in (*pending.content_advantages, *pending.report_advantages)
            ),
        }

        return DataProto.from_dict(
            tensors={
                "loss": torch.tensor([value(None if joint is None else joint.loss)], dtype=torch.float32),
                "policy_loss": torch.tensor([value(None if joint is None else joint.policy_loss)], dtype=torch.float32),
                "score_function_token_mean_diagnostic": torch.tensor(
                    [value(None if joint is None else joint.score_function_token_mean_diagnostic)],
                    dtype=torch.float32,
                ),
                "opsd_loss": torch.tensor([value(None if joint is None else joint.opsd_loss)], dtype=torch.float32),
                "update_succeeded": torch.tensor([status == "updated"], dtype=torch.bool),
                "content_tokens": torch.tensor([content_tokens], dtype=torch.long),
                "report_tokens": torch.tensor([report_tokens], dtype=torch.long),
            },
            non_tensors={"group_records_json": _object_array([
                json.dumps(records, ensure_ascii=False, allow_nan=False)
            ])},
            meta_info={"evidence": self._evidence(), "optimizer_evidence": self._evidence(), "status": status,
                       "grad_norm": grad_norm, "score_nonce": pending.nonce,
                       "loss_diagnostics": loss_diagnostics, "metrics": metrics},
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, scored_batch: DataProto) -> DataProto:
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        from ...trl_distillation import opsd_content_loss

        runtime = self._require_ready()
        size = _group_size(self.config)
        if not isinstance(scored_batch, DataProto) or len(scored_batch) != size:
            raise ValueError("update requires a complete prompt group")
        pending = self._scored
        if pending is None or self._generated is not None:
            raise RuntimeError("update requires one previously scored rollout")
        raw_payloads = scored_batch.non_tensor_batch.get("example_payload")
        if raw_payloads is None or len(raw_payloads) != size:
            raise ValueError("update requires one payload per sampled completion")
        example = _example(raw_payloads[0])
        if any(_digest(_example(value).to_dict()) != pending.example_digest for value in raw_payloads):
            raise RuntimeError("scored group contains changed examples")
        ids = _completions(scored_batch, size)
        if (scored_batch.meta_info.get("score_nonce") != pending.nonce
                or scored_batch.meta_info.get("model_version") != pending.model_version
                or scored_batch.meta_info.get("student_prompt_sha256") != pending.student_prompt_sha256
                or pending.model_version != self.model_version
                or _digest(example.to_dict()) != pending.example_digest
                or ids != pending.completion_ids):
            raise RuntimeError("stale, replayed, or altered scored rollout")
        for index, assessment in enumerate(pending.scored):
            width = len(ids[index])
            if (not torch.equal(scored_batch.batch["content_mask"][index, :width].cpu(),
                                torch.tensor(assessment.prepared.content_mask))
                    or not torch.equal(scored_batch.batch["report_mask"][index, :width].cpu(),
                                       torch.tensor(assessment.prepared.report_mask))
                    or bool(scored_batch.batch["content_mask"][index, width:].any())
                    or bool(scored_batch.batch["report_mask"][index, width:].any())):
                raise RuntimeError("scored token masks were altered")
            for key, expected in (
                ("content_reward", assessment.content_reward),
                ("report_reward", assessment.report_reward),
                ("content_advantage", pending.content_advantages[index]),
                ("report_advantage", pending.report_advantages[index]),
            ):
                if not math.isclose(float(scored_batch.batch[key][index]), expected, abs_tol=1e-6):
                    raise RuntimeError(f"detached {key} was altered")
        encoded = runtime.encode_student(example)
        if (tuple(int(item) for item in encoded.input_ids.tolist()) != pending.student_prompt_ids
                or encoded.sha256 != pending.student_prompt_sha256):
            raise RuntimeError("update student prompt or image differs from sampled prompt")
        self.attempted_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        policy_total = 0.0
        policy_token_loss_sum = 0.0
        opsd_total = 0.0
        content_nll_sum = 0.0
        report_nll_sum = 0.0
        clipfrac_total = 0.0
        approx_kl_total = 0.0
        clipfrac_lower_total = 0.0
        content_tokens = 0
        report_tokens = 0
        finite_loss = True
        for index, response_ids in enumerate(ids):
            selected = runtime.forward_response(
                runtime.student, encoded, torch.tensor(response_ids, dtype=torch.long)
            )
            if not bool(torch.isfinite(selected.detach()).all()):
                finite_loss = False
                break
            targets = torch.tensor(response_ids, dtype=torch.long, device=selected.device)
            log_prob = torch.log_softmax(selected.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            content = torch.tensor(pending.scored[index].prepared.content_mask,
                                   dtype=torch.bool, device=selected.device)
            report = torch.tensor(pending.scored[index].prepared.report_mask,
                                  dtype=torch.bool, device=selected.device)
            advantages = torch.where(
                content, log_prob.new_tensor(pending.content_advantages[index]),
                log_prob.new_tensor(pending.report_advantages[index]),
            )
            policy, pg_metrics = compute_policy_loss_vanilla(
                old_log_prob=log_prob.detach().unsqueeze(0),
                log_prob=log_prob.unsqueeze(0),
                advantages=advantages.unsqueeze(0),
                response_mask=torch.ones_like(log_prob).unsqueeze(0),
                loss_agg_mode="seq-mean-token-mean",
                config=_VanillaPolicyConfig(global_batch_size=size),
            )
            count = pending.scored[index].prepared.content_count
            if count and self.config["loss"]["opsd_weight"] > 0:
                teacher_logits = pending.clear_teacher_logits[index]
                if teacher_logits is None or teacher_logits.shape[0] != count:
                    raise RuntimeError("content OPSD lacks aligned privileged teacher logits")
                opsd_mean = opsd_content_loss(
                    selected[:count].unsqueeze(0),
                    teacher_logits.to(selected.device).unsqueeze(0),
                    torch.ones((1, count), device=selected.device, dtype=torch.bool),
                    beta=0.0, temperature=float(self.config["loss"]["opsd_temperature"]),
                    pointwise_clip=float(self.config["loss"]["opsd_token_clip"]),
                    reduction="token_mean",
                )
                opsd_part = opsd_mean / size
            else:
                opsd_part = policy * 0.0
            sample_loss = (
                float(self.config["loss"]["policy_weight"]) * policy
                + float(self.config["loss"]["opsd_weight"]) * opsd_part
            )
            if not bool(torch.isfinite(sample_loss.detach()).all()):
                finite_loss = False
                break
            sample_loss.backward()  # Accumulate four microbatches; step only after the whole group.
            policy_total += float(policy.detach())
            # This token-weighted score-function quantity is diagnostic only;
            # the actual GRPO objective weights per-sequence token means.
            policy_token_loss_sum += float((-log_prob.detach() * advantages.detach()).sum())
            opsd_total += float(opsd_part.detach())
            content_nll_sum += float(-log_prob.detach()[content].sum())
            report_nll_sum += float(-log_prob.detach()[report].sum())
            content_tokens += int(content.sum())
            report_tokens += int(report.sum())
            clipfrac_total += pg_metrics["actor/pg_clipfrac"] / size
            approx_kl_total += pg_metrics["actor/ppo_kl"] / size
            clipfrac_lower_total += pg_metrics["actor/pg_clipfrac_lower"] / size
        gradient_components = {"measured": False, "scope": "all_student_parameters",
                               "reason": "group_sequential_backward",
                               "policy_grad_norm": None, "opsd_grad_norm": None}
        joint = None
        if finite_loss:
            joint = _GroupJoint(
                loss=torch.tensor(float(self.config["loss"]["policy_weight"]) * policy_total
                                  + float(self.config["loss"]["opsd_weight"]) * opsd_total),
                policy_loss=torch.tensor(policy_total), opsd_loss=torch.tensor(opsd_total),
                score_function_token_mean_diagnostic=torch.tensor(
                    policy_token_loss_sum / sum(map(len, ids))
                ),
                content_mean_nll=(torch.tensor(content_nll_sum / content_tokens) if content_tokens else None),
                report_mean_nll=(torch.tensor(report_nll_sum / report_tokens) if report_tokens else None),
                policy_clipfrac=clipfrac_total, policy_approx_kl=approx_kl_total,
                policy_clipfrac_lower=clipfrac_lower_total,
            )
        grad_norm: float | None = None
        grad_norm_after_clip: float | None = None
        learning_rate_used: float | None = None
        if finite_loss:
            gradients = [param.grad for param in runtime.student.parameters() if param.grad is not None]
            if not gradients:
                raise RuntimeError("joint update produced no student gradients")
            finite_grad = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            if finite_grad:
                norm = torch.nn.utils.clip_grad_norm_(
                    runtime.student.parameters(), float(self.config["setting"]["max_grad_norm"]),
                    error_if_nonfinite=False,
                )
                grad_norm = float(norm.detach().item())
                finite_grad = math.isfinite(grad_norm) and all(
                    bool(torch.isfinite(gradient).all()) for gradient in gradients
                )
                if finite_grad:
                    grad_norm_after_clip = math.sqrt(sum(
                        float(gradient.detach().float().norm().item()) ** 2
                        for gradient in gradients
                    ))
                    finite_grad = math.isfinite(grad_norm_after_clip)
        else:
            finite_grad = False
        if finite_loss and finite_grad:
            learning_rate_used = float(self.optimizer.param_groups[0]["lr"])
            self.optimizer.step()
            if not all(bool(torch.isfinite(parameter.detach()).all())
                       for parameter in runtime.student.parameters()):
                self._scored = None
                raise FloatingPointError("Adam produced nonfinite student parameters; EMA was not updated")
            self.successful_updates += 1
            self.model_version += 1
            self.lr_scheduler.step()
            self._update_ema()
            status = "updated"
        else:
            self.skipped_updates += 1
            status = "nonfinite_skipped"
        self.optimizer.zero_grad(set_to_none=True)
        self._scored = None
        return self._update_result(
            pending, status=status, joint=joint, grad_norm=grad_norm,
            grad_norm_after_clip=grad_norm_after_clip,
            learning_rate_used=learning_rate_used,
            learning_rate_next=float(self.optimizer.param_groups[0]["lr"]),
            gradient_components=gradient_components,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def evaluate(self, requests: DataProto) -> DataProto:
        runtime = self._require_ready()
        self._check_idle()
        if not isinstance(requests, DataProto) or len(requests) != 1:
            raise ValueError("evaluate requires one fixed-validation example per RPC")
        raw = requests.non_tensor_batch.get("example_payload")
        if raw is None or len(raw) != len(requests):
            raise ValueError("evaluate requires example_payload per row")
        parameters = self._generation_parameters(requests.meta_info, training=False)
        records = []
        with torch.no_grad():
            for value in raw:
                example = _example(value)
                encoded = runtime.encode_student(example)
                generated = runtime.generate(encoded, parameters)
                ids = tuple(int(item) for item in generated.tolist())
                scored, _ = self._score_ids(
                    example, ids, tuple(int(item) for item in encoded.input_ids.tolist()),
                    encoded.sha256,
                )
                records.append(json.dumps(scored.record, ensure_ascii=False, allow_nan=False))
        return DataProto.from_dict(
            non_tensors={"record_json": _object_array(records)},
            meta_info={"evidence": self._evidence(), "optimizer_evidence": self._evidence(),
                       "validation_rows": len(records)},
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str) -> dict[str, Any]:
        runtime = self._require_ready()
        self._check_idle()
        if int(self.lr_scheduler.last_epoch) != self.successful_updates:
            raise RuntimeError("scheduler steps differ from successful optimizer updates")
        target = Path(path).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"checkpoint path already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            for label, model in (("student", runtime.student), ("teacher", runtime.teacher)):
                destination = temporary / label
                destination.mkdir()
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(destination, safe_serialization=True)
                else:
                    torch.save(model.state_dict(), destination / "state_dict.pt")
            if hasattr(runtime.processor, "save_pretrained"):
                runtime.processor.save_pretrained(temporary / "processor")
            torch.save(self.optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(self.lr_scheduler.state_dict(), temporary / "scheduler.pt")
            manifest = {
                "backend": "custom_veRL_RayWorkerGroup_HF",
                "model_id": self.config["model_id"],
                "template_sha256": runtime.template_sha256,
                "evidence": self._evidence(),
                "config_sha256": _digest(self.config),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
            )
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {"path": str(target), "optimizer_evidence": manifest["evidence"],
                "optimizer_state_max_step": self._adam_max_step(), **manifest}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def close(self) -> dict[str, Any]:
        self._check_idle()
        evidence = self._evidence()
        self.runtime = None
        self.optimizer = None
        self.lr_scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return evidence
