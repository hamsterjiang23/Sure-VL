"""Reproduce a real-Qwen GRPO sample/Teacher/backward audit without updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import GRPOConfig
from trl.trainer.utils import split_pixel_values_by_grid, split_tensor_dict, unsplit_pixel_values_by_grid
from transformers import AutoModelForImageTextToText, AutoProcessor

from sure_vl.proxy_data import manifest_to_proxy_rows
from sure_vl.train_proxy import configure_generation_terminators, configure_nonthinking_template
from sure_vl.training.trl.train import _dataset
from sure_vl.training.trl.trainer import SureVLGRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-update native GRPO p/q+/q− and backward audit")
    parser.add_argument("--model", type=Path, required=True, help="local Qwen3.5 model snapshot")
    parser.add_argument("--manifest", type=Path, required=True, help="official paired train manifest")
    parser.add_argument("--config", type=Path, required=True, help="Sure-VL training JSON")
    parser.add_argument("--output", type=Path, required=True, help="new JSON evidence path")
    parser.add_argument("--row-index", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("audit requires exactly one visible CUDA GPU")
    if cli.output.exists():
        raise FileExistsError(cli.output)
    before = time.monotonic()
    torch.manual_seed(42)
    config = json.loads(cli.config.read_text())
    rows = manifest_to_proxy_rows(cli.manifest)
    dataset = _dataset(rows)
    row = dataset[cli.row_index]
    processor = AutoProcessor.from_pretrained(cli.model, local_files_only=True, padding_side="left")
    template_sha = configure_nonthinking_template(processor)
    processor.image_processor.size = {**dict(processor.image_processor.size), "longest_edge": 65536}
    student = AutoModelForImageTextToText.from_pretrained(
        cli.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    )
    teacher = AutoModelForImageTextToText.from_pretrained(
        cli.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    )
    stops = configure_generation_terminators(student, processor.tokenizer)
    args = GRPOConfig(
        output_dir=str(cli.output.parent / "unused_trainer_output"),
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        generation_batch_size=4, num_generations=4, num_generations_eval=1,
        num_iterations=1, loss_type="grpo", scale_rewards="none", beta=0.0,
        learning_rate=1e-6, max_steps=1, temperature=1.0, top_p=1.0, top_k=0,
        max_completion_length=256, use_vllm=False, use_liger_kernel=False,
        disable_dropout=True, bf16=False, fp16=False, gradient_checkpointing=True,
        max_grad_norm=1.0, remove_unused_columns=False, report_to=[],
        save_strategy="no", eval_strategy="no", seed=42,
    )
    trainer = SureVLGRPOTrainer(
        model=student, teacher_model=teacher, args=args,
        train_dataset=dataset, processing_class=processor,
        proxy_config=config["proxy"], reward_config=config["reward"],
        teacher_config=config["teacher"], loss_config=config["loss"],
    )
    trainer.generation_config.eos_token_id = list(stops)
    trainer.model.generation_config.eos_token_id = list(stops)
    # Trainer.train normally sets this at each accumulation window. This
    # standalone zero-update audit invokes compute_loss directly.
    trainer.current_gradient_accumulation_steps = 4
    trainer._record_training_attempt = lambda _inputs, **_kwargs: None
    trainer.model.train()
    p_logps = []
    qminus_sameview_logp_deltas = []
    teacher_calls = {"q_plus": 0, "q_minus": 0}
    call_context = {"student_prompt_ids": None}
    original_causal = trainer._causal_logits

    def trace_causal(model, encoded, response_ids, *, supports_logits_to_keep):
        values = original_causal(
            model, encoded, response_ids,
            supports_logits_to_keep=supports_logits_to_keep,
        )
        prompt_ids = tuple(int(x) for x in encoded["input_ids"][0].tolist())
        if model is trainer.accelerator.unwrap_model(trainer.model):
            call_context["student_prompt_ids"] = prompt_ids
            ids_tensor = torch.as_tensor(response_ids, device=values.device, dtype=torch.long)
            logps = F.log_softmax(values.float(), -1).gather(-1, ids_tensor[:, None]).squeeze(-1)
            p_logps.append(logps.detach().cpu())
            # Direct diagnostic on actual sampled IDs, independent of reward
            # eligibility, which needs a parsed vision span.
            with torch.no_grad():
                qminus_logits = original_causal(
                    trainer.teacher_model, encoded, response_ids,
                    supports_logits_to_keep=supports_logits_to_keep,
                )
                qminus_logps = F.log_softmax(qminus_logits.float(), -1).gather(
                    -1, ids_tensor[:, None],
                ).squeeze(-1)
                qminus_sameview_logp_deltas.append(float((qminus_logps - logps).abs().max()))
        elif model is trainer.teacher_model:
            role = "q_minus" if prompt_ids == call_context["student_prompt_ids"] else "q_plus"
            teacher_calls[role] += 1
        return values

    trainer._causal_logits = trace_causal
    with torch.no_grad():
        batch = trainer._generate_and_score_completions([row] * 4)
    if len(p_logps) != 4 or len(qminus_sameview_logp_deltas) != 4 or len(trainer.last_rollout_records) != 4:
        raise RuntimeError("did not score four actual generated completions")
    if not all(torch.isfinite(values).all() for values in p_logps):
        raise FloatingPointError("nonfinite sampled p log probabilities")
    generated_tokens = [int(mask.sum()) for mask in batch["completion_mask"]]
    full_ids = torch.cat((batch["prompt_ids"], batch["completion_ids"]), dim=1)
    full_mask = torch.cat((batch["prompt_mask"], batch["completion_mask"]), dim=1)
    forward_kwargs = {key: batch[key] for key in (
        "pixel_values", "image_grid_thw", "pixel_attention_mask", "spatial_shapes",
        "image_sizes", "token_type_ids", "mm_token_type_ids", "image_position_ids", "num_images", "num_tiles",
    ) if key in batch}
    with torch.no_grad():
        base_logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, full_ids, full_mask, batch["completion_ids"].shape[1],
            batch_size=1, compute_entropy=False, **forward_kwargs,
        )
    logp_deltas = [
        (base_logps[i, :n].detach().cpu() - p_logps[i][:n]).abs()
        for i, n in enumerate(generated_tokens)
    ]
    max_p_logp_delta = max(float(delta.max()) for delta in logp_deltas)
    sampled_count = sum(generated_tokens)
    records = trainer.last_rollout_records
    mask_pass = bool(((batch["_proxy_content_mask"] | batch["_proxy_report_mask"])
                      == batch["completion_mask"].bool()).all())
    if not mask_pass:
        raise RuntimeError("actual-ID content/report partition failed")
    if teacher_calls["q_plus"] < 1:
        raise RuntimeError("privileged Teacher was never evaluated")
    trainer.model.zero_grad(set_to_none=True)
    split = split_pixel_values_by_grid(batch)
    microbatches = [unsplit_pixel_values_by_grid(part) for part in split_tensor_dict(split, 4)]
    losses = []
    opsd_losses = []
    for index, micro in enumerate(microbatches):
        loss = trainer.compute_loss(trainer.model, micro)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"nonfinite joint loss in microbatch {index}")
        losses.append(float(loss.detach()))
        opsd_losses.append(trainer._metrics["train"]["sure_vl/opsd_loss_unscaled"][-1])
        loss.backward()
        print(json.dumps({"event": "micro_backward", "index": index,
                          "loss_scaled": losses[-1], "opsd_unscaled": opsd_losses[-1],
                          "content_tokens": int(micro["_proxy_content_mask"].sum())}), flush=True)
    grads = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
    if not grads or not all(bool(torch.isfinite(gradient).all()) for gradient in grads):
        raise FloatingPointError("missing or nonfinite Student parameter gradients")
    grad_norm = math.sqrt(sum(float(gradient.detach().float().norm()) ** 2 for gradient in grads))
    teacher_grad_count = sum(parameter.grad is not None for parameter in trainer.teacher_model.parameters())
    if teacher_grad_count:
        raise RuntimeError("frozen Teacher received a gradient")
    report = {
        "kind": "real Qwen GRPO proxy G4 zero-update forward/backward audit",
        "example_id": row["example_id"], "template_sha256": template_sha,
        "student_prompt_json_sha256": hashlib.sha256(
            json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "prompt_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/sure_vl/proxy_prompt.py").read_bytes()
        ).hexdigest(),
        "generation": {"G": 4, "GA": 4, "max_new_tokens": 256, "temperature": 1.0,
                       "top_p": 1.0, "top_k": 0, "sampled_tokens_total": sampled_count,
                       "sampled_tokens_each": generated_tokens},
        "max_p_reward_vs_native_grpo_sample_logp_delta": max_p_logp_delta,
        "teacher_forward_counts": teacher_calls,
        "qminus_sameview_direct_diagnostic": {"actual_sampled_prefixes": len(qminus_sameview_logp_deltas), "max_abs_sampled_logp_delta": max(qminus_sameview_logp_deltas), "algorithm_reward_branch_calls": teacher_calls["q_minus"]},
        "content_report_partition_passed": mask_pass,
        "record_summary": [{"format_errors": r["format_errors"], "vision_tokens": r["vision_tokens"],
                            "proxy_fallback": r["proxy_fallback"],
                            "answer_correct": r["answer_correct"],
                            "content_tokens": r["content_tokens"],
                            "answer_confidence": r["answer_confidence"],
                            "visual_confidence": r["visual_confidence"],
                            "reward": r["reward"], "grpo": r["grpo"], "opsd": r["opsd"]}
                           for r in records],
        "micro_loss_scaled": losses, "micro_opsd_unscaled": opsd_losses,
        "student_grad_norm_before_clip": grad_norm,
        "student_grad_tensors": len(grads), "teacher_grad_tensors": teacher_grad_count,
        "optimizer_constructed": trainer.optimizer is not None,
        "optimizer_updates": 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_seconds": time.monotonic() - before,
    }
    if max_p_logp_delta > 0.05 or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("sampled p/native GRPO parity or Student gradient failed")
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "audit_completed", "output": str(cli.output),
                      "sampled_tokens": sampled_count, "q_minus_calls": teacher_calls["q_minus"],
                      "max_p_logp_delta": max_p_logp_delta, "student_grad_norm": grad_norm,
                      "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
