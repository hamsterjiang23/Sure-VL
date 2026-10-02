"""On-policy TRL training with a detached internal visual certainty target.

The student samples from the restricted view. The frozen-per-update teacher
scores the exact sampled content prefix with clear and restricted views. Only
vision-body distributions define S; original OPSD forward KL supervises all
content, including malformed outputs, and never the confidence report.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .proxy_method import empty_visual_proxy, visual_certainty_proxy
from .proxy_prompt import build_proxy_masks, parse_proxy_completion, split_proxy_generated_eos
from .proxy_protocol import ProxyExample, verify_proxy_answer
from .teacher_ema import OptimizerEvidenceCallback
from .trl_distillation import opsd_signal_diagnostics
from .trl_trainer import SureVLGOLDTrainer, opsd_forward_kl_per_token, _TRAIN_IMPORT_ERROR

if _TRAIN_IMPORT_ERROR is None:
    import torch
    import torch.nn.functional as F
    from trl.experimental.gold import GOLDTrainer


@dataclass
class RolloutAssessment:
    record: dict[str, Any]
    content_mask: Any
    report_mask: Any
    content_reward: float
    report_reward: float
    opsd_loss_sum: Any
    content_token_count: int


class ProxyGOLDTrainer(SureVLGOLDTrainer):
    """Full-model GOLD backend for the internal-proxy derivation.

    Policy loss is the unnormalized score-function sum per sampled completion,
    averaged across rows. OPSD is separately averaged across content tokens.
    No sampled reverse KL is added to the report or policy reward.
    """
    def __init__(self, *, proxy_config: Mapping[str, Any], reward_config: Mapping[str, Any],
                 teacher_config: Mapping[str, Any], **kwargs: Any) -> None:
        args = kwargs["args"]
        if getattr(args, "deepspeed", None) or getattr(args, "fsdp", None):
            raise ValueError("proxy v1 supports complete per-rank models, not sharded EMA parameters")
        if (getattr(args, "temperature", None) != 1.0 or getattr(args, "top_p", None) != 1.0
                or getattr(args, "top_k", None) not in (0, None)):
            raise ValueError("proxy policy score requires natural-temperature, untruncated sampling")
        self.proxy_config = dict(proxy_config)
        self.reward_config = dict(reward_config)
        self.teacher_config = dict(teacher_config)
        super().__init__(**kwargs)
        self.optimizer_evidence = OptimizerEvidenceCallback(
            self, teacher_mode=self.teacher_config["mode"], ema_decay=self.teacher_config["ema_decay"]
        )
        self.add_callback(self.optimizer_evidence)

    def _generate_on_policy_vlm_slice(self, pending_slice: dict[str, Any]):
        raw = pending_slice["_gold_vlm_on_policy_raw_examples"]
        rows = [{"prompt": row["prompt"], "student_image": row["image"],
                 "teacher_image": row["teacher_image"], "example_payload": row["example_payload"]}
                for row in raw]
        inputs, logs = GOLDTrainer._generate_on_policy_vlm_slice(self, pending_slice)
        inputs["_sure_vl_proxy_rows"] = rows
        return inputs, logs

    def _teacher_logits_for_content(self, row: Mapping[str, Any], content_ids: Any) -> Any:
        # The parent slice otherwise retains the full prompt-logit allocation.
        return super()._teacher_logits_for_content(row, content_ids).detach().clone()

    def measure_rollout(self, row: Mapping[str, Any], completion_ids: Any,
                        selected_student_logits: Any, *, diagnostics: bool = False) -> RolloutAssessment:
        """Score the actual sampled IDs; used unchanged in training and readback."""
        if selected_student_logits.ndim != 2 or len(completion_ids) != selected_student_logits.shape[0]:
            raise ValueError("sampled IDs and causal student logits must align")
        payload = row["example_payload"]
        example = ProxyExample.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
        body, terminal = split_proxy_generated_eos(
            self._tokenizer, completion_ids.tolist(),
            generation_eos_token_id=getattr(getattr(self, "generation_config", None), "eos_token_id", None),
        )
        text = self._tokenizer.decode(list(body), skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
        parsed = parse_proxy_completion(example, text)
        masks = build_proxy_masks(self._tokenizer, body, text, parsed)
        device = selected_student_logits.device
        content = torch.tensor((*masks.content_mask, *((0,) * len(terminal))), device=device, dtype=torch.bool)
        report = torch.tensor((*masks.report_mask, *((1,) * len(terminal))), device=device, dtype=torch.bool)
        vision = torch.tensor((*masks.vision_mask, *((0,) * len(terminal))), device=device, dtype=torch.bool)
        if not torch.all(content | report) or torch.any(content & report) or torch.any(vision & ~content):
            raise RuntimeError("invalid vision/content/report token partition")
        count = int(content.sum().item())
        if count and not bool(content[:count].all()):
            raise RuntimeError("teacher content must be a prefix of sampled completion")
        errors = list(parsed.format_errors)
        if masks.failure_reason:
            errors.append(masks.failure_reason)
        proxy = empty_visual_proxy()
        kl_sum = selected_student_logits.sum() * 0.0
        opsd: dict[str, float] = {"content_tokens": float(count), "sampled_positions": 0.0}
        clear_logits = None
        if count and (self.opsd_weight > 0 or bool(vision.any())):
            content_ids = completion_ids[:count]
            clear_logits = self._teacher_logits_for_content(row, content_ids)
            vision_positions = vision[:count].nonzero(as_tuple=True)[0]
            if vision_positions.numel():
                restricted_logits = None
                if self.proxy_config["lambda_b"] > 0:
                    # q- uses the same EMA teacher and the same sampled prefix.
                    # Only the prefix through the last vision position is needed.
                    last = int(vision_positions[-1].item()) + 1
                    restricted_row = dict(row, teacher_image=row["student_image"])
                    restricted_all = self._teacher_logits_for_content(restricted_row, content_ids[:last])
                    restricted_logits = restricted_all.index_select(0, vision_positions)
                    del restricted_all
                proxy = visual_certainty_proxy(
                    selected_student_logits[:count].index_select(0, vision_positions),
                    clear_logits.index_select(0, vision_positions),
                    torch.ones(vision_positions.numel(), device=device, dtype=torch.bool),
                    restricted_logits,
                    alpha=self.proxy_config["alpha"], tau_s=self.proxy_config["tau_s"],
                    lambda_b=self.proxy_config["lambda_b"], temperature=1.0,
                    min_vision_tokens=self.proxy_config["min_vision_tokens"],
                    chunk_size=self.proxy_config["chunk_size"],
                )
                del restricted_logits
            if self.opsd_weight > 0:
                per_token = opsd_forward_kl_per_token(
                    selected_student_logits[:count], clear_logits,
                    temperature=self.opsd_temperature, pointwise_clip=self.opsd_token_clip,
                )
                kl_sum = per_token.sum()
                opsd["clipped_loss_mean"] = float(per_token.detach().mean())
                if diagnostics:
                    measured = opsd_signal_diagnostics(
                        selected_student_logits[:count], clear_logits,
                        temperature=self.opsd_temperature, pointwise_clip=self.opsd_token_clip,
                        weight=self.opsd_weight, max_positions=self.diagnostic_tokens,
                    )
                    for key in ("sampled_positions", "raw_forward_kl_mean", "clipped_loss_mean",
                                "clipped_vocabulary_fraction", "top1_disagreement_rate", "weighted_logit_grad_l2"):
                        opsd[key] = float(getattr(measured, key))
        if proxy.fallback:
            errors.append("vision_proxy_fallback")
        answer_label = verify_proxy_answer(example, parsed.answer)
        correct = bool(answer_label) if answer_label is not None else False
        v = None if parsed.visual_confidence is None else parsed.visual_confidence / 10.0
        r = None if parsed.answer_confidence is None else parsed.answer_confidence / 10.0
        utility = self.reward_config["answer_utility"] * int(correct)
        # Missing reports/answer label have explicit maximal penalties, not
        # invented confidence values or labels used in calibration metrics.
        answer_score = -self.reward_config["rho_answer"] * (
            (r - int(correct)) ** 2 if r is not None and answer_label is not None else 1.0
        )
        visual_score = -self.reward_config["rho_visual"] * ((v - proxy.certainty) ** 2 if v is not None else 1.0)
        format_penalty = self.reward_config["format_penalty"] if errors else 0.0
        report_reward = answer_score + visual_score - format_penalty
        content_reward = utility + report_reward
        components = {key: value for key, value in {
            "raw_js": proxy.mean_raw_js, "baseline_js": proxy.mean_baseline_js,
            "corrected_gap": proxy.mean_corrected_gap, "teacher_entropy": proxy.mean_teacher_entropy,
            "uncertainty": proxy.mean_uncertainty,
        }.items() if value is not None}
        record = {
            "id": example.id, "raw_completion": text, "vision_text": parsed.vision_text,
            "answer": parsed.answer, "answer_correct": correct,
            "answer_label_available": answer_label is not None,
            "visual_confidence": v, "answer_confidence": r,
            "visual_confidence_score": parsed.visual_confidence,
            "answer_confidence_score": parsed.answer_confidence,
            "confidence_score_max": 10,
            "visual_proxy": proxy.certainty, "proxy_fallback": proxy.fallback,
            "vision_tokens": proxy.vision_token_count, "content_tokens": count,
            "generated_tokens": int(completion_ids.numel()), "format_errors": errors,
            "proxy_components": components, "reward": {
                "utility": utility, "answer_score": answer_score, "visual_score": visual_score,
                "report": report_reward, "total": content_reward, "format_penalty": format_penalty,
            }, "opsd": opsd,
        }
        return RolloutAssessment(record, content, report, content_reward, report_reward, kl_sum, count)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: Any = None) -> Any:
        if not model.training:
            raise RuntimeError("proxy joint loss requires fresh on-policy training rollouts")
        rows = inputs.get("_sure_vl_proxy_rows")
        ids, labels = inputs["input_ids"], inputs["labels"]
        if not rows or len(rows) != ids.shape[0]:
            raise ValueError("paired proxy metadata must match generated rows")
        outputs = model(input_ids=ids, attention_mask=inputs["attention_mask"], use_cache=False,
                        **self._get_model_forward_kwargs(inputs))
        policies, kl_sums, records = [], [], []
        content_tokens = 0
        for index, row in enumerate(rows):
            positions = (labels[index] != -100).nonzero(as_tuple=True)[0]
            if bool((positions == 0).any()) or (positions.numel() and not torch.equal(
                positions, torch.arange(positions[0], positions[0] + positions.numel(), device=positions.device)
            )):
                raise ValueError("sampled completion must be a contiguous span after its prompt")
            completion = ids[index, positions]
            selected = outputs.logits[index, positions - 1, :]
            measured = self.measure_rollout(row, completion, selected, diagnostics=True)
            records.append(measured.record)
            logps = F.log_softmax(selected.float(), dim=-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)
            policies.append(-measured.content_reward * logps[measured.content_mask].sum()
                            - measured.report_reward * logps[measured.report_mask].sum())
            kl_sums.append(measured.opsd_loss_sum)
            content_tokens += measured.content_token_count
        policy = torch.stack(policies).mean()
        opsd = torch.stack(kl_sums).sum() / content_tokens if content_tokens else policy * 0.0
        loss = self.policy_weight * policy + self.opsd_weight * opsd

        def log(name: str, value: float) -> None:
            self._metrics["train"][f"sure_vl/rank_local/{name}"].append(float(value))

        log("policy_loss", float(policy.detach()))
        log("opsd_loss", float(opsd.detach()))
        log("content_tokens", content_tokens)
        generated = sum(r["generated_tokens"] for r in records)
        log("generated_tokens", generated)
        log("content_token_coverage", content_tokens / generated if generated else 0.0)
        log("format_failure_rate", sum(bool(r["format_errors"]) for r in records) / len(records))
        log("proxy_fallback_rate", sum(r["proxy_fallback"] for r in records) / len(records))
        log("answer_correct_rate", sum(r["answer_correct"] for r in records) / len(records))
        usable = [r for r in records if not r["proxy_fallback"]]
        log("proxy_usable_rows", len(usable))
        for key in ("visual_proxy", "vision_tokens"):
            values = [r[key] for r in usable]
            log(key + "_mean", sum(values) / len(values) if values else 0.0)
        if usable:
            mean_s = sum(r["visual_proxy"] for r in usable) / len(usable)
            log("visual_proxy_variance", sum((r["visual_proxy"] - mean_s) ** 2 for r in usable) / len(usable))
        for key in ("utility", "answer_score", "visual_score", "report", "total", "format_penalty"):
            log("reward_" + key, sum(r["reward"][key] for r in records) / len(records))
        for key in ("raw_js", "baseline_js", "corrected_gap", "teacher_entropy", "uncertainty"):
            values = [r["proxy_components"][key] for r in records if key in r["proxy_components"]]
            log("proxy_" + key, sum(values) / len(values) if values else 0.0)
        positions = sum(r["opsd"]["sampled_positions"] for r in records)
        log("opsd_diagnostic_positions", positions)
        log("opsd_rows", sum(r["opsd"]["sampled_positions"] > 0 for r in records))
        for key in ("raw_forward_kl_mean", "clipped_loss_mean", "clipped_vocabulary_fraction",
                    "top1_disagreement_rate", "weighted_logit_grad_l2"):
            total = sum(r["opsd"].get(key, 0.0) * r["opsd"]["sampled_positions"] for r in records)
            log("opsd_" + key, total / positions if positions else 0.0)
        self.last_rollout_records = records
        # Keep per-sample evidence beside rank-local aggregate telemetry.
        # This records attempted windows; the callback separately counts
        # successful updates and AMP skips.
        from pathlib import Path
        path = Path(self.args.output_dir) / f"proxy_train_attempts_rank_{self.accelerator.process_index}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps({"trainer_step_before_update": int(self.state.global_step),
                                         "rank": int(self.accelerator.process_index), **record},
                                        ensure_ascii=False) + "\n")
        return (loss, outputs) if return_outputs else loss
