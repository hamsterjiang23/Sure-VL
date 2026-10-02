"""Sure-VL proxy rewards and content distillation on TRL's native GRPOTrainer.

The upstream trainer still owns generation, group batching, advantage-aware
clipping, the optimizer, checkpointing, and logging.  This subclass scores
exactly the sampled completion IDs, replaces scalar advantages with the two
group-centred content/report advantages supported by upstream GRPO, and adds
an original-OPSD forward KL on content tokens from the same student forward.

This adapter is pinned to the audited TRL 1.14.1 implementation.  It supports
one or two DDP GPUs, one row per accumulation microbatch, num_iterations=1,
no Liger kernel, and no model/optimizer sharding.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from trl import GRPOTrainer
from trl.data_utils import prepare_multimodal_messages

from ...proxy_prompt import build_proxy_teacher_messages
from ...proxy_protocol import ProxyExample
from ...proxy_rollout import (
    ScoredProxyRollout,
    prepare_proxy_rollout,
    score_proxy_rollout,
)
from ...trl_distillation import opsd_content_loss, opsd_signal_diagnostics


_MODEL_IMAGE_KEYS = (
    "pixel_values", "image_grid_thw", "pixel_attention_mask", "spatial_shapes",
    "image_sizes", "image_position_ids", "num_tiles",
)
_SEQUENCE_KEYS = ("mm_token_type_ids", "token_type_ids")


@dataclass(frozen=True)
class ProxyAssessment:
    record: dict[str, Any]
    content_mask: tuple[bool, ...]
    report_mask: tuple[bool, ...]
    content_reward: float
    report_reward: float


@dataclass(frozen=True)
class _SampleScore:
    example_payload: str
    ids: tuple[int, ...]
    student_prompt_ids: tuple[int, ...]
    teacher_version: int
    scored: ScoredProxyRollout


def _example(payload: Any) -> ProxyExample:
    if isinstance(payload, ProxyExample):
        return payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ProxyExample.from_dict(payload)


def _payload_json(example: ProxyExample) -> str:
    return json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as source:
            return source.convert("RGB")
    raise ValueError("proxy image must be a PIL image or a path")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _group_center(rewards: torch.Tensor, group_size: int, scale_rewards: str) -> torch.Tensor:
    """Match GRPO's finite-reward group centering and optional std scaling."""
    if rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group_size:
        raise ValueError("rewards must contain complete generation groups")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("proxy rewards must be finite")
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_rewards == "none":
        return centered.reshape(-1)
    if scale_rewards == "group":
        if group_size < 2:
            return centered.reshape(-1)
        return (centered / (grouped.std(dim=1, unbiased=True, keepdim=True) + 1e-4)).reshape(-1)
    if scale_rewards == "batch":
        std = rewards.std(unbiased=True) if rewards.numel() > 1 else rewards.new_zeros(())
        return (centered / (std + 1e-4)).reshape(-1)
    raise ValueError("scale_rewards must be 'none', 'group', or 'batch'")


def _global_group_center(local_rewards: torch.Tensor, accelerator: Any, group_size: int,
                         scale_rewards: str, *, training: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center one complete GRPO group, then return this rank's contiguous slice."""
    local_count = local_rewards.numel()
    world = accelerator.num_processes if training else 1
    if not local_count or local_count * world != group_size:
        raise ValueError("local rewards do not form one complete global generation group")
    global_rewards = accelerator.gather(local_rewards) if training and world > 1 else local_rewards
    if global_rewards.numel() != group_size:
        raise RuntimeError("DDP gathered rewards do not match GRPO group size")
    global_advantages = _group_center(global_rewards, group_size, scale_rewards)
    rank = accelerator.process_index if training else 0
    start = rank * local_count
    return global_advantages[start:start + local_count], global_advantages, global_rewards


class ProxyGRPOTrainer(GRPOTrainer):
    """Native TRL GRPO with segment advantages and detached Teacher proxy."""

    def __init__(
        self,
        model: Any,
        teacher_model: Any,
        *,
        proxy_config: Mapping[str, Any],
        reward_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
        teacher_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        args = kwargs.get("args")
        if args is None:
            raise ValueError("ProxyGRPOTrainer requires an explicit GRPOConfig")
        if kwargs.get("reward_funcs") is not None:
            raise ValueError("Sure-VL owns its proxy reward function")
        if not isinstance(model, torch.nn.Module) or not isinstance(teacher_model, torch.nn.Module):
            raise TypeError("student and teacher must be instantiated torch models")
        if model is teacher_model:
            raise ValueError("student and teacher must be separate model instances")
        if getattr(args, "use_vllm", False) or getattr(args, "use_liger_kernel", False):
            raise ValueError("proxy GRPO v1 uses native HF generation and full logits")
        if getattr(args, "use_transformers_continuous_batching", False) or getattr(args, "mask_truncated_completions", False):
            raise ValueError("proxy GRPO v1 requires regular HF generation and unmasked sampled IDs")
        if getattr(args, "deepspeed", None) or getattr(args, "fsdp", None):
            raise ValueError("proxy GRPO v1 requires complete, unsharded models")
        if getattr(args, "per_device_train_batch_size", None) != 1:
            raise ValueError("proxy GRPO v1 requires per-device microbatch size 1")
        if getattr(args, "num_iterations", None) != 1:
            raise ValueError("proxy GRPO v1 requires num_iterations=1")
        if getattr(args, "loss_type", None) != "grpo" or float(getattr(args, "beta", math.nan)) != 0.0:
            raise ValueError("proxy GRPO v1 requires loss_type='grpo' and beta=0")
        if getattr(args, "scale_rewards", None) not in ("none", "group", "batch"):
            raise ValueError("GRPO scale_rewards must be 'none', 'group', or 'batch'")
        if getattr(args, "steps_per_generation", None) != getattr(args, "gradient_accumulation_steps", None):
            raise ValueError("group scoring and one EMA update require steps_per_generation=gradient_accumulation_steps")
        if getattr(args, "num_generations", 0) < 2:
            raise ValueError("proxy GRPO v1 requires at least two generations")
        if float(getattr(args, "temperature", math.nan)) != 1.0 or float(getattr(args, "top_p", math.nan)) != 1.0:
            raise ValueError("training requires natural temperature and untruncated top-p")
        if getattr(args, "top_k", 0) not in (0, None):
            raise ValueError("training top_k must be zero")
        if getattr(args, "gradient_checkpointing", False) and getattr(args, "gradient_checkpointing_kwargs", None):
            # A forward hook must observe the model's outputs. Standard Qwen
            # checkpointing is supported; custom checkpoint wrappers are not.
            if args.gradient_checkpointing_kwargs.get("use_reentrant", False):
                raise ValueError("reentrant gradient checkpointing is unsupported")
        self.teacher_model = teacher_model
        self.proxy_config = dict(proxy_config)
        self.reward_config = dict(reward_config)
        self.loss_config = dict(loss_config)
        self.teacher_config = dict(teacher_config)
        self.opsd_weight = float(self.loss_config["opsd_weight"])
        if self.opsd_weight < 0 or not math.isfinite(self.opsd_weight):
            raise ValueError("opsd_weight must be finite and nonnegative")
        if float(self.loss_config["opsd_beta"]) != 0.0:
            raise ValueError("original OPSD forward KL requires beta=0")
        if float(self.loss_config["policy_weight"]) != 1.0:
            raise ValueError("native GRPO policy coefficient must be one in this recipe")
        if self.teacher_config.get("mode") not in {"fixed", "ema"}:
            raise ValueError("teacher mode must be fixed or ema")
        self._latest_sample_scores: list[_SampleScore] | None = None
        self.last_rollout_records: list[dict[str, Any]] = []
        self._global_tracking_buffer: list[dict[str, Any]] = []
        super().__init__(model=model, reward_funcs=self._content_reward_func, **kwargs)
        if self.accelerator.num_processes not in (1, 2):
            raise ValueError("proxy GRPO v1 supports one or two DDP processes")
        if self.num_generations != args.steps_per_generation * self.accelerator.num_processes:
            raise ValueError("one complete global GRPO group must be consumed per optimizer update")
        if args.generation_batch_size != self.num_generations:
            raise ValueError("generation batch must contain exactly one global GRPO group")
        explicit_teacher_device = self.teacher_config.get("device")
        if explicit_teacher_device is not None:
            if explicit_teacher_device != "cuda:1":
                raise ValueError("the external Teacher device must be cuda:1")
            if (self.accelerator.num_processes != 1 or self.accelerator.device.type != "cuda"
                    or self.accelerator.device.index not in (None, 0)
                    or torch.cuda.current_device() != 0 or torch.cuda.device_count() < 2):
                raise ValueError("cuda:1 Teacher requires one process, Student cuda:0, and two visible GPUs")
            teacher_device = torch.device(explicit_teacher_device)
        else:
            teacher_device = self.accelerator.device
        self.teacher_device = teacher_device
        self.teacher_model.to(teacher_device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def _teacher_version(self) -> int:
        evidence = getattr(self, "optimizer_evidence", None)
        return 0 if evidence is None else int(evidence.summary()["teacher_ema_updates"])

    def _encode_messages(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Use the same processor and chat-template call as GRPO generation."""
        encoded = self.processing_class.apply_chat_template(
            conversation=[messages], tools=self.tools or None,
            chat_template=self.chat_template, add_generation_prompt=True,
            tokenize=True, return_dict=True, **self.chat_template_kwargs,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RuntimeError("VLM processor did not return prompt input_ids")
        result: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            result[key] = value.to(self.accelerator.device)
        ids = result["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise RuntimeError("encoded proxy prompt must be one nonempty sequence")
        attention = result.get("attention_mask", torch.ones_like(ids))
        if attention.shape != ids.shape:
            raise RuntimeError("processor attention mask differs from input_ids")
        active = attention[0].bool().nonzero(as_tuple=True)[0]
        if active.numel() < 1 or not bool(attention[0, int(active[0]):].bool().all()):
            raise RuntimeError("prompt padding must be a contiguous left prefix")
        first = int(active[0])
        if first:
            original = ids.shape[1]
            for key in ("input_ids", "attention_mask", *_SEQUENCE_KEYS):
                if key in result and result[key].ndim >= 2 and result[key].shape[-1] == original:
                    result[key] = result[key][..., first:]
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        return result

    def _teacher_messages(self, example: ProxyExample) -> list[dict[str, Any]]:
        messages = build_proxy_teacher_messages(
            example.teacher_question or example.question, example.teacher_evidence,
        )
        return prepare_multimodal_messages(messages, images=[_image(example.teacher_image)])

    @staticmethod
    def _causal_logits(model: Any, encoded: Mapping[str, torch.Tensor], response_ids: Sequence[int],
                       *, supports_logits_to_keep: bool) -> torch.Tensor:
        """Score response token t at prompt_len+t-1 under this view's own prefix."""
        input_device = encoded["input_ids"].device
        model_device = next(model.parameters()).device
        cross_device = model_device != input_device
        if cross_device and torch.is_grad_enabled():
            raise RuntimeError("cross-device Teacher logits require detached evaluation")

        def on_model(tensor: torch.Tensor) -> torch.Tensor:
            if not cross_device:
                return tensor
            # Staging through host memory avoids GPU peer transport/NCCL on
            # hosts where both cards work independently but cross-GPU fails.
            return tensor.detach().to("cpu").to(model_device)

        response = torch.as_tensor(response_ids, dtype=torch.long, device=input_device)
        if response.ndim != 1 or not response.numel():
            raise ValueError("sampled response must be nonempty")
        prefix_ids = on_model(encoded["input_ids"])
        response = on_model(response)
        prompt_length = prefix_ids.shape[1]
        model_inputs = {key: on_model(value) for key, value in encoded.items() if key != "input_ids"}
        model_inputs["input_ids"] = torch.cat((prefix_ids, response.unsqueeze(0)), dim=1)
        prefix_attention = model_inputs["attention_mask"]
        model_inputs["attention_mask"] = torch.cat((
            prefix_attention, prefix_attention.new_ones((1, response.numel())),
        ), dim=1)
        for key in _SEQUENCE_KEYS:
            if key in model_inputs:
                prefix = model_inputs[key]
                if prefix.ndim != 2 or prefix.shape != prefix_ids.shape:
                    raise RuntimeError(f"{key} does not align with encoded prompt")
                model_inputs[key] = torch.cat((prefix, prefix.new_zeros((1, response.numel()))), dim=1)
        if supports_logits_to_keep:
            model_inputs["logits_to_keep"] = response.numel() + 1
        output = model(**model_inputs, use_cache=False)
        logits = output.logits
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError("model did not return [1, sequence, vocab] logits")
        if logits.shape[1] == response.numel() + 1:
            selected = logits[0, :-1, :]
        elif logits.shape[1] >= prompt_length + response.numel():
            selected = logits[0, prompt_length - 1:prompt_length - 1 + response.numel(), :]
        else:
            raise RuntimeError("model logits do not cover sampled response positions")
        if selected.shape[0] != response.numel():
            raise RuntimeError("causal sampled-token alignment failed")
        if cross_device:
            return selected.detach().to("cpu").to(input_device)
        return selected

    def _score_one(self, example: ProxyExample, student_messages: list[dict[str, Any]],
                   ids: tuple[int, ...], student_logits: torch.Tensor | None = None
                   ) -> tuple[ScoredProxyRollout, tuple[int, ...]]:
        encoded_student = self._encode_messages(student_messages)
        prompt_ids = tuple(int(value) for value in encoded_student["input_ids"][0].tolist())
        prepared = prepare_proxy_rollout(
            example, self._tokenizer, ids,
            eos_token_ids=getattr(self.generation_config, "eos_token_id", None),
        )
        student_model = self.accelerator.unwrap_model(self.model)
        supports_keep = "logits_to_keep" in self.model_kwarg_keys
        was_training = student_model.training
        try:
            student_model.eval()
            with torch.no_grad():
                p_logits = (
                    self._causal_logits(student_model, encoded_student, ids,
                                        supports_logits_to_keep=supports_keep)
                    if student_logits is None else student_logits.detach()
                )
                q_plus = None
                q_minus = None
                teacher_encoding = None
                if prepared.content_count and (self.opsd_weight > 0 or prepared.vision_positions):
                    teacher_encoding = self._encode_messages(self._teacher_messages(example))
                    q_plus = self._causal_logits(
                        self.teacher_model, teacher_encoding, prepared.content_ids,
                        supports_logits_to_keep=supports_keep,
                    ).detach()
                if prepared.vision_positions and float(self.proxy_config["lambda_b"]) > 0:
                    last = prepared.vision_positions[-1] + 1
                    # The same encoded student prompt, including image tensors,
                    # is passed to EMA q-minus: exact same-view conditioning.
                    q_minus = self._causal_logits(
                        self.teacher_model, encoded_student, ids[:last],
                        supports_logits_to_keep=supports_keep,
                    ).detach()
                teacher_prompt_ids = None if teacher_encoding is None else (
                    teacher_encoding["input_ids"][0].tolist()
                )
                conditioning = {
                    "privileged_image": example.teacher_image,
                    "teacher_evidence_present": example.teacher_evidence is not None,
                    "teacher_evidence_sha256": (
                        None if example.teacher_evidence is None else _digest(example.teacher_evidence)
                    ),
                    "student_prompt_sha256": _digest(prompt_ids),
                    "teacher_prompt_sha256": (
                        None if teacher_prompt_ids is None else _digest(teacher_prompt_ids)
                    ),
                    "baseline_prompt_sha256": _digest(prompt_ids),
                    "baseline_image": example.student_image,
                    "privileged_input": {"prompt_tokens": len(teacher_prompt_ids or ())},
                    "baseline_input": {"prompt_tokens": len(prompt_ids)},
                    "baseline": "same EMA, exact student prompt and image, no privileged evidence",
                    "gap_scope": "joint image/evidence/template/model difference; heuristic JS correction",
                }
                scored = score_proxy_rollout(
                    example, prepared, p_logits, q_plus, q_minus,
                    proxy_config=self.proxy_config, reward_config=self.reward_config,
                    teacher_conditioning=conditioning,
                )
                if q_plus is not None and prepared.content_count:
                    # A bounded detached logit-space readback, separate from
                    # the actual full-content differentiable loss below.
                    with torch.enable_grad():
                        diagnostic = opsd_signal_diagnostics(
                            p_logits[:prepared.content_count], q_plus,
                            temperature=float(self.loss_config["opsd_temperature"]),
                            pointwise_clip=float(self.loss_config["opsd_token_clip"]),
                            weight=self.opsd_weight,
                            max_positions=int(self.loss_config.get("diagnostic_tokens", 4)),
                        )
                    scored.record["opsd"].update({
                        "sampled_positions": diagnostic.sampled_positions,
                        "raw_forward_kl_mean": diagnostic.raw_forward_kl_mean,
                        "clipped_loss_mean": diagnostic.clipped_loss_mean,
                        "clipped_vocabulary_fraction": diagnostic.clipped_vocabulary_fraction,
                        "top1_disagreement_rate": diagnostic.top1_disagreement_rate,
                        "weighted_logit_grad_l2": diagnostic.weighted_logit_grad_l2,
                    })
        finally:
            student_model.train(was_training)
        return scored, prompt_ids

    def _content_reward_func(self, prompts: list[Any], completions: list[Any],
                             completion_ids: list[Sequence[int]], **kwargs: Any) -> list[float]:
        payloads = kwargs.get("example_payload")
        if payloads is None or len(payloads) != len(completion_ids) or len(prompts) != len(completion_ids):
            raise ValueError("GRPO reward lacks aligned example_payload or prompts")
        scores: list[_SampleScore] = []
        for prompt, ids_raw, payload in zip(prompts, completion_ids, payloads, strict=True):
            example = _example(payload)
            ids = tuple(int(value) for value in ids_raw)
            scored, student_ids = self._score_one(example, prompt, ids)
            scores.append(_SampleScore(
                example_payload=_payload_json(example), ids=ids,
                student_prompt_ids=student_ids, teacher_version=self._teacher_version(),
                scored=scored,
            ))
        self._latest_sample_scores = scores
        return [entry.scored.content_reward for entry in scores]

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        self._latest_sample_scores = None
        batch = super()._generate_and_score_completions(inputs)
        scores = self._latest_sample_scores
        if scores is None or len(scores) != batch["completion_ids"].shape[0]:
            raise RuntimeError("proxy rewards did not align with GRPO sampled completions")
        count, width = batch["completion_ids"].shape
        content = torch.zeros((count, width), dtype=torch.bool, device=batch["completion_ids"].device)
        report = torch.zeros_like(content)
        payloads, records, prompt_ids, versions = [], [], [], []
        for index, entry in enumerate(scores):
            n = int(batch["completion_mask"][index].sum().item())
            actual = tuple(int(value) for value in batch["completion_ids"][index, :n].tolist())
            if actual != entry.ids:
                raise RuntimeError("GRPO returned IDs differ from reward-scored sampled IDs")
            actual_prompt = tuple(int(value) for value in batch["prompt_ids"][index][
                batch["prompt_mask"][index].bool()
            ].tolist())
            if actual_prompt != entry.student_prompt_ids:
                raise RuntimeError("reward Student prompt differs from actual GRPO generation prompt IDs")
            content[index, :n] = torch.tensor(entry.scored.prepared.content_mask, device=content.device)
            report[index, :n] = torch.tensor(entry.scored.prepared.report_mask, device=report.device)
            payloads.append(entry.example_payload)
            records.append(entry.scored.record)
            prompt_ids.append(entry.student_prompt_ids)
            versions.append(entry.teacher_version)
        if not bool(((content | report) == batch["completion_mask"].bool()).all()) or bool((content & report).any()):
            raise RuntimeError("proxy content/report masks do not partition actual completion IDs")
        training = self.model.training
        group_size = self.num_generations if training else self.num_generations_eval
        content_rewards = torch.tensor([entry.scored.content_reward for entry in scores],
                                       dtype=torch.float32, device=content.device)
        report_rewards = torch.tensor([entry.scored.report_reward for entry in scores],
                                      dtype=torch.float32, device=content.device)
        if training and self.accelerator.num_processes > 1:
            fingerprints = torch.tensor(
                [list(bytes.fromhex(_digest(payload))) for payload in payloads],
                dtype=torch.uint8, device=content.device,
            )
            global_fingerprints = self.accelerator.gather(fingerprints)
            if global_fingerprints.shape != (group_size, 32) or not bool(
                (global_fingerprints == global_fingerprints[:1]).all()
            ):
                raise RuntimeError("DDP completions are not from one identical global prompt")
        content_adv, global_content_adv, global_content_rewards = _global_group_center(
            content_rewards, self.accelerator, group_size, self.scale_rewards, training=training,
        )
        report_adv, global_report_adv, global_report_rewards = _global_group_center(
            report_rewards, self.accelerator, group_size, self.scale_rewards, training=training,
        )
        advantage = torch.where(content, content_adv[:, None], report_adv[:, None])
        advantage = advantage * batch["completion_mask"]
        for index, record in enumerate(records):
            record["grpo"] = {
                "content_advantage": float(content_adv[index]),
                "report_advantage": float(report_adv[index]),
                "active_policy_tokens": int((advantage[index].abs() > 0).sum()),
                "scale_rewards": self.scale_rewards,
                "loss_type": self.loss_type,
            }
        batch["advantages"] = advantage
        batch["_proxy_content_mask"] = content
        batch["_proxy_report_mask"] = report
        batch["_proxy_example_payloads"] = payloads
        batch["_proxy_expected_prompt_ids"] = prompt_ids
        batch["_proxy_teacher_version"] = torch.tensor(versions, dtype=torch.long, device=content.device)
        batch["_proxy_record_json"] = [json.dumps(record, ensure_ascii=False, allow_nan=False)
                                       for record in records]
        self.last_rollout_records = records
        mode = "train" if training else "eval"
        active = (advantage.abs() > 0) & batch["completion_mask"].bool()
        group_zero = (global_content_adv.view(-1, group_size).abs().sum(-1) == 0) & (
            global_report_adv.view(-1, group_size).abs().sum(-1) == 0
        )
        group_counts = torch.tensor([
            int(active.sum()), int(batch["completion_mask"].sum()),
            sum(not row["proxy_fallback"] for row in records), len(records),
        ], dtype=torch.long, device=content.device)
        if training and self.accelerator.num_processes > 1:
            group_counts = self.accelerator.gather(group_counts).reshape(-1, 4).sum(0)
        self._metrics[mode]["sure_vl/zero_advantage_group_fraction"].append(float(group_zero.float().mean()))
        self._metrics[mode]["sure_vl/active_policy_tokens"].append(float(group_counts[0]))
        self._metrics[mode]["sure_vl/policy_active_token_fraction"].append(
            float(group_counts[0] / group_counts[1]) if int(group_counts[1]) else 0.0
        )
        self._metrics[mode]["sure_vl/report_reward_mean"].append(float(global_report_rewards.mean()))
        self._metrics[mode]["sure_vl/proxy_usable_fraction"].append(
            float(group_counts[2] / group_counts[3])
        )
        self._latest_sample_scores = None
        return batch

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: Any = None) -> torch.Tensor:
        if return_outputs:
            raise ValueError("GRPOTrainer does not support return_outputs")
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs=False,
                                        num_items_in_batch=num_items_in_batch)
        payloads = inputs.get("_proxy_example_payloads")
        if payloads is None or len(payloads) != 1:
            raise RuntimeError("proxy metadata was lost before GRPO microbatch update")
        version = int(inputs["_proxy_teacher_version"][0])
        if version != self._teacher_version():
            raise RuntimeError("EMA Teacher changed between group scoring and update")
        captured: list[torch.Tensor] = []

        def capture_logits(_module: Any, _arguments: Any, output: Any) -> None:
            if not hasattr(output, "logits"):
                raise RuntimeError("GRPO Student forward did not return logits")
            captured.append(output.logits)

        handle = model.register_forward_hook(capture_logits) if self.opsd_weight > 0 else None
        try:
            grpo_loss = super().compute_loss(model, inputs, return_outputs=False,
                                             num_items_in_batch=num_items_in_batch)
        finally:
            if handle is not None:
                handle.remove()
        if self.opsd_weight > 0 and len(captured) != 1:
            raise RuntimeError("expected exactly one native GRPO Student forward")
        content = inputs["_proxy_content_mask"][0]
        count = int(content.sum())
        if count and (not bool(content[:count].all()) or bool(content[count:].any())):
            raise RuntimeError("OPSD content mask must be a sampled completion prefix")
        if self.opsd_weight > 0 and count:
            completion = inputs["completion_ids"][0]
            total = completion.shape[0]
            logits = captured[0]
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < total + 1:
                raise RuntimeError("captured native GRPO logits lack a causal completion prefix")
            student_logits = logits[0, -total - 1:-1, :][:count]
            example = _example(payloads[0])
            teacher_encoding = self._encode_messages(self._teacher_messages(example))
            with torch.no_grad():
                teacher_logits = self._causal_logits(
                    self.teacher_model, teacher_encoding,
                    tuple(int(value) for value in completion[:count].tolist()),
                    supports_logits_to_keep="logits_to_keep" in self.model_kwarg_keys,
                ).detach()
            opsd = opsd_content_loss(
                student_logits.unsqueeze(0), teacher_logits.unsqueeze(0),
                torch.ones((1, count), dtype=torch.bool, device=student_logits.device),
                beta=0.0, temperature=float(self.loss_config["opsd_temperature"]),
                pointwise_clip=float(self.loss_config["opsd_token_clip"]),
                top_k=None, reduction="token_mean",
            )
        else:
            opsd = grpo_loss * 0.0
        accumulation = float(self.current_gradient_accumulation_steps)
        total_loss = grpo_loss + self.opsd_weight * opsd / accumulation
        self._metrics["train"]["sure_vl/grpo_loss_scaled"].append(float(grpo_loss.detach()))
        self._metrics["train"]["sure_vl/grpo_loss_unscaled"].append(float(grpo_loss.detach()) * accumulation)
        self._metrics["train"]["sure_vl/opsd_loss_unscaled"].append(float(opsd.detach()))
        self._metrics["train"]["sure_vl/opsd_weighted_contribution_scaled"].append(
            self.opsd_weight * float(opsd.detach()) / accumulation
        )
        self._metrics["train"]["sure_vl/joint_loss_scaled"].append(float(total_loss.detach()))
        self._metrics["train"]["sure_vl/content_tokens"].append(float(count))
        active = (inputs["advantages"][0].abs() > 0) & inputs["completion_mask"][0].bool()
        sampled_tokens = int(inputs["completion_mask"][0].sum())
        self._metrics["train"]["sure_vl/policy_active_token_fraction_micro_local"].append(
            int(active.sum()) / sampled_tokens if sampled_tokens else 0.0
        )
        self._record_training_attempt(inputs, full_opsd=float(opsd.detach()) if count else None)
        return total_loss

    def _record_training_attempt(self, inputs: Mapping[str, Any], *, full_opsd: float | None = None) -> None:
        records = inputs.get("_proxy_record_json")
        if records is None:
            return
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"proxy_train_attempts_rank_{self.accelerator.process_index}.jsonl"
        with path.open("a", encoding="utf-8") as destination:
            for raw in records:
                record = json.loads(raw)
                if full_opsd is not None:
                    record["opsd"]["full_content_clipped_loss_mean"] = full_opsd
                attempt = {
                    "trainer_step_before_update": int(self.state.global_step),
                    "rank": int(self.accelerator.process_index), **record,
                }
                destination.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False) + "\n")
                self._global_tracking_buffer.append(attempt)

    def _extract_images_and_prompts(self, rows: Sequence[Mapping[str, Any]]) -> tuple[list[list[Image.Image]], list[Any]]:
        """Compatibility for the fixed-subset validation callback."""
        images, prompts = [], []
        for row in rows:
            image = _image(row["image"])
            images.append([image])
            prompts.append(prepare_multimodal_messages(row["prompt"], images=[image]))
        return images, prompts

    def measure_rollout(self, row: Mapping[str, Any], completion_ids: torch.Tensor,
                        selected_student_logits: torch.Tensor, *, diagnostics: bool = False) -> ProxyAssessment:
        """Score an externally sampled validation completion with the same method."""
        del diagnostics
        example = _example(row["example_payload"])
        image = _image(row.get("student_image", row["image"]))
        messages = prepare_multimodal_messages(row["prompt"], images=[image])
        ids = tuple(int(value) for value in completion_ids.tolist())
        scored, prompt_ids = self._score_one(example, messages, ids, selected_student_logits)
        expected = row.get("_student_prompt_ids")
        if expected is not None and prompt_ids != tuple(int(value) for value in expected.tolist()):
            raise RuntimeError("validation Student prompt IDs differ from actual generation")
        return ProxyAssessment(
            record=scored.record, content_mask=scored.prepared.content_mask,
            report_mask=scored.prepared.report_mask,
            content_reward=scored.content_reward, report_reward=scored.report_reward,
        )


# The entrypoint uses the project-facing name; keep the implementation name
# explicit for method tests and reviews.
SureVLGRPOTrainer = ProxyGRPOTrainer
