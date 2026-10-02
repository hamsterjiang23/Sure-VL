"""TRL GOLD backend for joint on-policy Sure-VL and OPSD training.

This adapter deliberately keeps the two losses separate. The policy term is a
score-function surrogate over the student's sampled content and confidence
tokens. The OPSD term is a full-vocabulary, teacher-to-student forward KL on
content tokens only. It is *not* the sampled reverse-KL estimator in
``train_step.py``; adding that estimator here would count teacher guidance a
second time.

The implementation targets ``trl==1.14.1`` and its VLM GOLDTrainer. GOLD owns
sampling, image-aware collation, accumulation, optimizer steps, and checkpoint
handling. This subclass supplies the paired teacher image, strict output
parsing, segment masks, rewards, and the joint loss. It requires ``lmbda=1``
and ``use_vllm=False`` so each optimizer window consumes fresh student samples.
Its teacher is a separately loaded, frozen copy of the student's initial model.

Malformed completions remain in the batch. Their entire generated span is
routed to the report term with a fixed reward of -3, and no teacher KL is
computed. This is an implementation choice below the minimum valid calibration
score (-2), not a result inferred from the Sure-VL objective.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from .objective import score
from .protocol import Example, ProtocolError, verify
from .trl_distillation import opsd_content_loss, opsd_signal_diagnostics
from .trl_prompt import (
    build_content_report_masks,
    parse_student_completion,
    split_generated_eos,
)


try:
    import torch
    import torch.nn.functional as F
    from trl.experimental.gold import GOLDTrainer
except ImportError as exc:  # Preserve the torch-free protocol/CLI install.
    _TRAIN_IMPORT_ERROR: ImportError | None = exc
    GOLDTrainer = object  # type: ignore[assignment,misc]
else:
    _TRAIN_IMPORT_ERROR = None


FORMAT_FAILURE_CONTENT_REWARD = -2.0
FORMAT_FAILURE_REPORT_REWARD = -3.0


def opsd_forward_kl_per_token(
    student_logits: Any,
    teacher_logits: Any,
    *,
    temperature: float = 1.1,
    pointwise_clip: float = 0.05,
) -> Any:
    """Original OPSD beta=0 loss before token reduction.

    Both tensors have shape ``[content_tokens, vocab]``. The pointwise cap is
    applied to each vocabulary summand *before* summing it, matching the OPSD
    reference implementation. Its clipped result may be negative.
    """
    if _TRAIN_IMPORT_ERROR is not None:
        raise ImportError("TRL training requires the optional training dependencies") from _TRAIN_IMPORT_ERROR
    if student_logits.ndim != 2 or teacher_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same [tokens, vocabulary] shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pointwise_clip) or pointwise_clip <= 0:
        raise ValueError("pointwise_clip must be finite and positive")

    count = student_logits.shape[0]
    if count == 0:
        return student_logits.new_zeros((0,), dtype=torch.float32)
    return opsd_content_loss(
        student_logits.unsqueeze(0),
        teacher_logits.unsqueeze(0),
        torch.ones((1, count), dtype=torch.bool, device=student_logits.device),
        beta=0.0,
        temperature=temperature,
        pointwise_clip=pointwise_clip,
        reduction="none",
    )[0]


def _model_origin(model: Any) -> str | None:
    if isinstance(model, str):
        return model
    config = getattr(model, "config", None)
    value = getattr(config, "_name_or_path", None)
    return value if isinstance(value, str) and value else None


class SureVLGOLDTrainer(GOLDTrainer):  # type: ignore[misc,valid-type]
    """VLM GOLD trainer with paired-image OPSD and dual-confidence RL.

    Dataset rows follow GOLD's VLM schema: raw conversational ``prompt``, an
    assistant ``completion`` placeholder, and a PIL ``image`` for the student.
    They additionally contain a PIL ``teacher_image`` and ``example_payload``
    holding a JSON-serialized :class:`sure_vl.protocol.Example`. The image in
    ``teacher_image`` is used only in the frozen teacher forward pass.

    ``GOLDConfig.temperature`` is the student rollout temperature. The OPSD
    softmax temperature is independently controlled by ``opsd_temperature``.
    Set GOLDConfig ``beta=0`` for teacher-to-student forward KL. Policy and
    distillation contributions are scaled by ``policy_weight`` and
    ``opsd_weight``. The policy term uses zero baselines, matching the core
    ``train_step`` default; ``num_generations`` supplies independent on-policy
    samples but no GRPO reward normalization is applied.
    """

    def __init__(
        self,
        *,
        model: Any,
        teacher_model: Any,
        args: Any,
        train_dataset: Any,
        processing_class: Any,
        opsd_weight: float = 1.0,
        policy_weight: float = 1.0,
        opsd_temperature: float = 1.1,
        opsd_token_clip: float = 0.05,
        diagnostic_tokens: int = 4,
        **gold_kwargs: Any,
    ) -> None:
        if _TRAIN_IMPORT_ERROR is not None:
            raise ImportError("SureVLGOLDTrainer requires torch and trl==1.14.1") from _TRAIN_IMPORT_ERROR
        if model is teacher_model:
            raise ValueError("student and teacher must be separately loaded model objects")
        student_origin, teacher_origin = _model_origin(model), _model_origin(teacher_model)
        if student_origin and teacher_origin and student_origin != teacher_origin:
            raise ValueError("student and teacher must share the same initial checkpoint and tokenizer")
        if getattr(args, "lmbda", None) != 1.0:
            raise ValueError("Sure-VL requires GOLDConfig(lmbda=1.0) for fully on-policy training")
        if getattr(args, "use_vllm", None):
            raise ValueError("Sure-VL v1 requires GOLDConfig(use_vllm=False)")
        if getattr(args, "use_uld_loss", None):
            raise ValueError("Sure-VL v1 requires same-tokenizer forward KL, not ULD")
        if getattr(args, "beta", None) != 0.0:
            raise ValueError("Sure-VL v1 requires GOLDConfig(beta=0.0) for OPSD forward KL")
        if getattr(args, "remove_unused_columns", False):
            raise ValueError("remove_unused_columns must be False to preserve paired-image metadata")
        if gold_kwargs.get("eval_dataset") is not None or str(getattr(args, "eval_strategy", "no")).lower() not in (
            "no", "intervalstrategy.no"
        ):
            raise ValueError("Sure-VL v1 uses separate frozen evaluation; GOLD off-policy eval is unsupported")
        for name, value in (
            ("opsd_weight", opsd_weight),
            ("policy_weight", policy_weight),
            ("opsd_temperature", opsd_temperature),
            ("opsd_token_clip", opsd_token_clip),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if opsd_weight < 0 or policy_weight < 0 or (opsd_weight == 0 and policy_weight == 0):
            raise ValueError("loss weights must be nonnegative and at least one must be positive")
        if opsd_temperature <= 0 or opsd_token_clip <= 0:
            raise ValueError("OPSD temperature and token clip must be positive")
        if type(diagnostic_tokens) is not int or diagnostic_tokens < 1:
            raise ValueError("diagnostic_tokens must be a positive integer")

        sample = next(iter(train_dataset), None)
        required = {"prompt", "completion", "image", "teacher_image", "example_payload"}
        if not isinstance(sample, Mapping) or not required.issubset(sample):
            raise ValueError(f"training rows must contain {', '.join(sorted(required))}")

        self.opsd_weight = float(opsd_weight)
        self.policy_weight = float(policy_weight)
        self.opsd_temperature = float(opsd_temperature)
        self.opsd_token_clip = float(opsd_token_clip)
        self.diagnostic_tokens = diagnostic_tokens
        super().__init__(
            model=model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **gold_kwargs,
        )
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def _generate_on_policy_vlm_slice(self, pending_slice: dict[str, Any]):
        # GOLD's inner VLM collator mutates copies of rows and drops non-model
        # fields. Preserve the paired image and labels in row order first.
        raw_examples = pending_slice["_gold_vlm_on_policy_raw_examples"]
        rows = [
            {
                "prompt": example["prompt"],
                "teacher_image": example["teacher_image"],
                "example_payload": example["example_payload"],
            }
            for example in raw_examples
        ]
        inputs, text_logs = super()._generate_on_policy_vlm_slice(pending_slice)
        inputs["_sure_vl_rows"] = rows
        return inputs, text_logs

    def _teacher_logits_for_content(self, row: Mapping[str, Any], content_ids: Any) -> Any:
        """Score only sampled content tokens under the paired clear image."""
        teacher_row = {"prompt": row["prompt"], "image": row["teacher_image"]}
        images, prepared_prompts = self._extract_images_and_prompts([teacher_row])
        prompt_texts = self.processing_class.apply_chat_template(
            prepared_prompts, tokenize=False, add_generation_prompt=True
        )
        prompt_batch = self.processing_class(
            images=images,
            text=prompt_texts,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = self.accelerator.device
        prompt_ids = prompt_batch["input_ids"]
        prompt_mask = prompt_batch["attention_mask"].bool()
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.as_tensor(prompt_ids)
        if not isinstance(prompt_mask, torch.Tensor):
            prompt_mask = torch.as_tensor(prompt_mask)
        real_prompt_ids = prompt_ids[0, prompt_mask[0]].to(device)
        if real_prompt_ids.numel() == 0:
            raise ValueError("teacher prompt has no tokens")
        expected_prompt_ids = row.get("_expected_student_prompt_ids")
        if expected_prompt_ids is not None and not torch.equal(
            real_prompt_ids, expected_prompt_ids.to(device)
        ):
            raise RuntimeError("same-view baseline must use the exact encoded student prompt IDs")
        trace = row.get("_teacher_input_trace")
        if isinstance(trace, dict):
            trace.update(prompt_tokens=int(real_prompt_ids.numel()),
                         sampled_prefix_tokens=int(content_ids.numel()))
        teacher_ids = torch.cat((real_prompt_ids, content_ids.to(device)), dim=0).unsqueeze(0)
        teacher_mask = torch.ones_like(teacher_ids)
        teacher_kwargs = self._get_model_forward_kwargs(prompt_batch, exclude=self._SEQUENCE_KEYS)
        teacher_kwargs = {key: value.to(device) for key, value in teacher_kwargs.items()}
        for key in self._SEQUENCE_KEYS:
            if key in prompt_batch:
                prefix = prompt_batch[key][0, prompt_mask[0]].to(device)
                suffix = torch.zeros(content_ids.numel(), dtype=prefix.dtype, device=device)
                teacher_kwargs[key] = torch.cat((prefix, suffix), dim=0).unsqueeze(0)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                use_cache=False,
                **teacher_kwargs,
            )
        start = real_prompt_ids.numel() - 1
        teacher_logits = teacher_outputs.logits[0, start : start + content_ids.numel(), :]
        if teacher_logits.shape[0] != content_ids.numel():
            raise RuntimeError("teacher logit positions do not align with sampled content tokens")
        return teacher_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training:
            raise RuntimeError("Sure-VL v1 computes the joint loss only on fresh on-policy training rollouts")
        rows = inputs.get("_sure_vl_rows")
        if rows is None:
            raise RuntimeError("paired row metadata was lost before Sure-VL loss computation")
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        if len(rows) != input_ids.shape[0]:
            raise ValueError("paired row count does not match the generated batch")
        student_kwargs = self._get_model_forward_kwargs(inputs)
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            **student_kwargs,
        )
        student_logits = student_outputs.logits
        policy_losses = []
        kl_sums = []
        content_token_count = 0
        format_failure_count = 0
        reward_total = 0.0
        reward_report = 0.0
        reward_utility = 0.0
        valid_count = 0
        visual_correct_count = 0
        answer_correct_count = 0
        generated_token_count = 0
        diagnostic_totals = {
            "raw_forward_kl_mean": 0.0,
            "clipped_loss_mean": 0.0,
            "clipped_vocabulary_fraction": 0.0,
            "top1_disagreement_rate": 0.0,
            "weighted_logit_grad_l2": 0.0,
        }
        diagnostic_positions = 0
        diagnostic_rows = 0

        for row_index, row in enumerate(rows):
            payload = row["example_payload"]
            try:
                example = Example.from_dict(json.loads(payload))
            except (TypeError, ValueError, json.JSONDecodeError, ProtocolError) as error:
                raise ValueError(f"invalid example_payload in row {row_index}: {error}") from error
            positions = (labels[row_index] != -100).nonzero(as_tuple=True)[0]
            generated_token_count += int(positions.numel())
            if (positions == 0).any():
                raise ValueError("generated completion includes sequence position zero")
            if positions.numel() and not torch.equal(
                positions,
                torch.arange(positions[0], positions[0] + positions.numel(), device=positions.device),
            ):
                raise ValueError("generated completion positions must be contiguous")
            completion_ids = input_ids[row_index, positions]
            body_ids, eos_ids = split_generated_eos(
                self._tokenizer, completion_ids.tolist(),
                generation_eos_token_id=getattr(
                    getattr(self, "generation_config", None), "eos_token_id", None
                ),
            )
            text = self._tokenizer.decode(
                list(body_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            try:
                parsed = parse_student_completion(example, text)
            except ProtocolError:
                parsed = None
            masks = build_content_report_masks(
                self._tokenizer,
                body_ids,
                text,
                parsed.report_start_char if parsed is not None else None,
            )
            valid = parsed is not None and masks.failure_reason is None
            if valid:
                valid_count += 1
                verification = verify(example, parsed.output)
                visual_correct_count += int(verification.visual_correct)
                answer_correct_count += int(verification.answer_correct)
                reward = score(
                    int(verification.visual_correct),
                    int(verification.answer_correct),
                    parsed.output.visual_confidence / 100.0,
                    parsed.output.conditional_answer_confidence / 100.0,
                )
                content_reward = reward.total
                report_reward = reward.calibration
                reward_utility += reward.utility
            else:
                format_failure_count += 1
                content_reward = FORMAT_FAILURE_CONTENT_REWARD
                report_reward = FORMAT_FAILURE_REPORT_REWARD

            reward_total += content_reward if valid else report_reward
            reward_report += report_reward
            content_mask_values = (*masks.content_mask, *((0,) * len(eos_ids)))
            report_mask_values = (*masks.report_mask, *((1,) * len(eos_ids)))
            if len(content_mask_values) != len(completion_ids):
                raise RuntimeError("content/report masks do not cover the generated completion")
            content_mask = torch.tensor(content_mask_values, dtype=torch.bool, device=input_ids.device)
            report_mask = torch.tensor(report_mask_values, dtype=torch.bool, device=input_ids.device)
            if torch.any(content_mask & report_mask) or not torch.all(content_mask | report_mask):
                raise RuntimeError("content/report masks must form a partition")
            if content_mask.any() and not torch.all(content_mask[: int(content_mask.sum().item())]):
                raise RuntimeError("content tokens must be a prefix of the sampled completion")

            if positions.numel() == 0:
                # Keep the example in the batch mean even though an empty
                # completion has no sampled action on which to place a gradient.
                policy_losses.append(student_logits[row_index].sum() * 0.0)
                continue
            selected_logits = student_logits[row_index, positions - 1, :]
            sampled_logps = F.log_softmax(selected_logits.float(), dim=-1).gather(
                -1, completion_ids.unsqueeze(-1)
            ).squeeze(-1)
            policy_losses.append(
                -content_reward * sampled_logps[content_mask].sum()
                - report_reward * sampled_logps[report_mask].sum()
            )

            if content_mask.any() and self.opsd_weight > 0:
                content_ids = completion_ids[content_mask]
                teacher_logits = self._teacher_logits_for_content(row, content_ids)
                per_token_kl = opsd_forward_kl_per_token(
                    selected_logits[content_mask],
                    teacher_logits,
                    temperature=self.opsd_temperature,
                    pointwise_clip=self.opsd_token_clip,
                )
                kl_sums.append(per_token_kl.sum())
                content_token_count += int(content_mask.sum().item())
                diagnostics = opsd_signal_diagnostics(
                    selected_logits[content_mask],
                    teacher_logits,
                    temperature=self.opsd_temperature,
                    pointwise_clip=self.opsd_token_clip,
                    weight=self.opsd_weight,
                    max_positions=self.diagnostic_tokens,
                )
                diagnostic_rows += 1
                diagnostic_positions += diagnostics.sampled_positions
                for key in diagnostic_totals:
                    diagnostic_totals[key] += getattr(diagnostics, key) * diagnostics.sampled_positions

        policy_loss = torch.stack(policy_losses).mean()
        opsd_loss = (
            torch.stack(kl_sums).sum() / content_token_count
            if content_token_count
            else policy_loss * 0.0
        )
        loss = self.policy_weight * policy_loss + self.opsd_weight * opsd_loss
        mode = "train"

        def log_local(name: str, value: float) -> None:
            # GOLD averages _metrics inside each rank but does not all-reduce
            # these custom fields. Make their scope explicit in every key.
            self._metrics[mode][f"sure_vl/rank_local/{name}"].append(float(value))

        log_local("policy_loss", float(policy_loss.detach()))
        log_local("opsd_loss", float(opsd_loss.detach()))
        log_local("format_failure_count", format_failure_count)
        log_local("format_failure_rate", format_failure_count / len(rows))
        log_local("effective_policy_reward", reward_total / len(rows))
        log_local("report_reward", reward_report / len(rows))
        log_local("reward_utility_valid_mean",
            reward_utility / valid_count if valid_count else 0.0
        )
        log_local("valid_count", valid_count)
        log_local("visual_correct_valid_rate",
            visual_correct_count / valid_count if valid_count else 0.0
        )
        log_local("answer_correct_valid_rate",
            answer_correct_count / valid_count if valid_count else 0.0
        )
        log_local("content_tokens", content_token_count)
        log_local("generated_tokens", generated_token_count)
        log_local("content_token_coverage",
            content_token_count / generated_token_count if generated_token_count else 0.0
        )
        log_local("opsd_rows", diagnostic_rows)
        log_local("opsd_diagnostic_positions", diagnostic_positions)
        for key, total in diagnostic_totals.items():
            log_local(f"opsd_{key}",
                total / diagnostic_positions if diagnostic_positions else 0.0
            )
        return (loss, student_outputs) if return_outputs else loss
