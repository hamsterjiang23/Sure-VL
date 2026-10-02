#!/usr/bin/env python3
"""Zero-update GRPO VLM sampling/forward parity on one real manifest row.

The probe invokes installed TRL's GRPOTrainer generation and keeps the actual
sampled token IDs and raw generation logits. It then scores those same IDs in
a full multimodal student forward. It neither constructs an optimizer nor
calls ``train``. Its report describes this new diagnostic rollout only; old
training JSONL records do not contain the IDs needed for a replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def compare_generation_and_forward_logits(
    generation_logits: tuple[Any, ...],
    forward_logits: Any,
    completion_ids: Any,
    completion_mask: Any,
) -> dict[str, float | int]:
    """Compare raw model.generate logits to causal logits on the *same* IDs."""
    import torch
    import torch.nn.functional as F

    if forward_logits.ndim != 3 or completion_ids.ndim != 2 or completion_mask.shape != completion_ids.shape:
        raise ValueError("expected logits [batch, completion, vocabulary] and matching ID/mask tensors")
    if forward_logits.shape[:2] != completion_ids.shape or len(generation_logits) < completion_ids.shape[1]:
        raise ValueError("generation and causal forward completion positions do not align")
    count = 0
    max_abs_logit = 0.0
    max_abs_sample_logp = 0.0
    sum_abs_sample_logp = 0.0
    sum_js = 0.0
    max_js = 0.0
    for position in range(completion_ids.shape[1]):
        active = completion_mask[:, position].bool()
        if not bool(active.any()):
            continue
        generated = generation_logits[position][active].float()
        rescored = forward_logits[active, position].float()
        if generated.shape != rescored.shape or not bool(torch.isfinite(generated).all()) or not bool(torch.isfinite(rescored).all()):
            raise ValueError("generation/forward logits differ in shape or contain nonfinite values")
        ids = completion_ids[active, position, None]
        generation_logp = F.log_softmax(generated, dim=-1)
        forward_logp = F.log_softmax(rescored, dim=-1)
        delta = (generation_logp.gather(-1, ids) - forward_logp.gather(-1, ids)).abs()
        log_mean = torch.logaddexp(generation_logp, forward_logp) - torch.log(
            generated.new_tensor(2.0)
        )
        js = 0.5 * (
            (generation_logp.exp() * (generation_logp - log_mean)).sum(-1)
            + (forward_logp.exp() * (forward_logp - log_mean)).sum(-1)
        )
        n = int(active.sum())
        count += n
        max_abs_logit = max(max_abs_logit, float((generated - rescored).abs().max()))
        max_abs_sample_logp = max(max_abs_sample_logp, float(delta.max()))
        sum_abs_sample_logp += float(delta.sum())
        max_js = max(max_js, float(js.max()))
        sum_js += float(js.sum())
    if not count:
        raise ValueError("no sampled completion token remains after GRPO masking")
    return {
        "compared_tokens": count,
        "max_abs_raw_logit_delta": max_abs_logit,
        "max_abs_sample_logprob_delta": max_abs_sample_logp,
        "mean_abs_sample_logprob_delta": sum_abs_sample_logp / count,
        "max_jensen_shannon_nats": max_js,
        "mean_jensen_shannon_nats": sum_js / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="existing local Qwen3.5 checkpoint")
    parser.add_argument("--manifest", required=True, help="frozen proxy train manifest")
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--output", required=True, help="new JSON report path")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-sampled-logprob-delta", type=float, default=0.05)
    parser.add_argument("--max-js-nats", type=float, default=0.001)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_pixels <= 0 or args.seed < 0:
        parser.error("max-new-tokens/max-pixels must be positive and seed nonnegative")
    model_path = Path(args.model).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not model_path.is_dir() or output.exists():
        parser.error("model must be an existing local directory and output must not exist")

    import torch
    import trl
    import transformers
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from sure_vl.proxy_data import build_proxy_dataset, manifest_to_proxy_rows
    from sure_vl.train_proxy import configure_generation_terminators, configure_nonthinking_template

    if not torch.cuda.is_available():
        raise RuntimeError("this bounded real-model diagnostic requires CUDA")
    rows = [row for row in manifest_to_proxy_rows(args.manifest) if row["example_id"] == args.example_id]
    if len(rows) != 1:
        raise ValueError(f"example ID must identify exactly one manifest row: {args.example_id}")
    row = rows[0]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, padding_side="left")
    template_sha256 = configure_nonthinking_template(processor)
    image_size = dict(processor.image_processor.size)
    image_size["longest_edge"] = args.max_pixels
    processor.image_processor.size = image_size
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to("cuda:0")
    if getattr(model.config, "model_type", None) != "qwen3_5":
        raise RuntimeError("this image-position parity probe currently targets Qwen3.5")
    model.eval()
    stops = configure_generation_terminators(model, processor.tokenizer)
    dataset = build_proxy_dataset([row, row])

    def constant_diagnostic_reward(completions, **_kwargs):
        # The numeric value is immaterial: only the actual sampled IDs and
        # GRPO's multimodal inputs are read. No loss/backward/optimizer follows.
        return [0.0] * len(completions)

    training_args = GRPOConfig(
        output_dir=str(output.parent / "grpo_audit_unused"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        num_generations=2,
        num_iterations=1,
        max_completion_length=args.max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        use_vllm=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        disable_dropout=True,
        remove_unused_columns=False,
        report_to=[],
        logging_steps=1,
        max_steps=1,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=constant_diagnostic_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    # GRPO's train/eval mode chooses the number of completions per prompt.
    # Dropout is disabled, so train mode retains distribution parity.
    trainer.model.train()
    captured: dict[str, Any] = {}
    original_generate = model.generate

    def capture_generate(*generation_args, **generation_kwargs):
        generation_kwargs["return_dict_in_generate"] = True
        generation_kwargs["output_logits"] = True
        generated = original_generate(*generation_args, **generation_kwargs)
        captured["generated"] = generated
        captured["kwargs_keys"] = sorted(generation_kwargs)
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"):
            captured[key] = generation_kwargs.get(key)
        return generated.sequences  # Preserve GRPOTrainer's expected API.

    model.generate = capture_generate
    try:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            batch = trainer._generate_and_score_completions([dataset[0], dataset[1]])
    finally:
        model.generate = original_generate
    generated = captured.get("generated")
    if generated is None or getattr(generated, "logits", None) is None:
        raise RuntimeError("GRPO generation did not expose raw logits for the parity audit")
    prompt_ids, completion_ids = batch["prompt_ids"], batch["completion_ids"]
    prompt_mask, completion_mask = batch["prompt_mask"], batch["completion_mask"]
    if captured.get("mm_token_type_ids") is None or batch.get("mm_token_type_ids") is None:
        raise RuntimeError("GRPO Qwen3.5 rollout/scoring omitted mm_token_type_ids")
    if not torch.equal(captured["input_ids"].to(prompt_ids.device), prompt_ids):
        raise RuntimeError("GRPO generation and scoring used different prompt token IDs")
    if not torch.equal(captured["attention_mask"].to(prompt_mask.device), prompt_mask):
        raise RuntimeError("GRPO generation and scoring used different prompt attention masks")
    if not torch.equal(captured["mm_token_type_ids"].to(prompt_ids.device),
                       batch["mm_token_type_ids"][:, :prompt_ids.shape[1]]):
        raise RuntimeError("GRPO generation and scoring used different prompt multimodal token types")
    for key in ("pixel_values", "image_grid_thw"):
        if captured.get(key) is None or batch.get(key) is None or not torch.equal(
            captured[key].to(batch[key].device), batch[key]
        ):
            raise RuntimeError(f"GRPO generation and scoring used different {key}")
    raw_completion = generated.sequences[:, prompt_ids.shape[1]:]
    for index in range(completion_ids.shape[0]):
        active = completion_mask[index].bool()
        if not torch.equal(completion_ids[index, active], raw_completion[index, :int(active.sum())]):
            raise RuntimeError("GRPO output completion IDs differ from the captured model.generate IDs")
    full_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    completion_width = completion_ids.shape[1]
    forward_keys = (
        "pixel_values", "image_grid_thw", "pixel_attention_mask", "spatial_shapes",
        "image_sizes", "token_type_ids", "mm_token_type_ids", "image_position_ids",
    )
    forward_kwargs = {key: batch[key] for key in forward_keys if key in batch}
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids, attention_mask=attention_mask,
            logits_to_keep=completion_width + 1, use_cache=False, **forward_kwargs,
        )
    selected = outputs.logits[:, :-1, :][:, -completion_width:, :]
    metrics = compare_generation_and_forward_logits(
        tuple(generated.logits), selected, completion_ids, completion_mask
    )
    passes = (
        metrics["max_abs_sample_logprob_delta"] <= args.max_sampled_logprob_delta
        and metrics["max_jensen_shannon_nats"] <= args.max_js_nats
    )
    report = {
        "kind": "fresh GRPOTrainer zero-update actual-ID image-position parity",
        "model": str(model_path), "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "example_id": args.example_id, "seed": args.seed,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__},
        "template_sha256": template_sha256,
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                       "eos_token_ids": stops, "max_new_tokens": args.max_new_tokens,
                       "kwargs_keys": captured["kwargs_keys"]},
        "actual_prompt_ids": [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, prompt_mask)],
        "actual_completion_ids": [ids[mask.bool()].tolist() for ids, mask in zip(completion_ids, completion_mask)],
        "prompt_mm_types_sha256": hashlib.sha256(
            captured["mm_token_type_ids"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "metrics": metrics,
        "thresholds": {"max_sampled_logprob_delta": args.max_sampled_logprob_delta,
                       "max_js_nats": args.max_js_nats},
        "parity_passed": bool(passes), "optimizer_updates": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "kind": report["kind"], "example_id": args.example_id,
        "metrics": metrics, "parity_passed": bool(passes),
        "generation_kwargs_keys": captured["kwargs_keys"],
        "versions": report["versions"], "output": str(output),
    }, ensure_ascii=False))
    if not passes:
        raise RuntimeError("GRPO generation/full-forward parity exceeded diagnostic thresholds")


if __name__ == "__main__":
    main()
