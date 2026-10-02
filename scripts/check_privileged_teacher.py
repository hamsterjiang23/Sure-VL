#!/usr/bin/env python3
"""Check actual VLM teacher conditioning on a synthetic crop/scene-graph fixture.

This uses a fixed diagnostic completion, no sampling or optimizer update. It
checks input routing and causal logit alignment, not task effectiveness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="existing local VLM checkpoint directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from sure_vl.proxy_prompt import build_proxy_prompt
    from sure_vl.proxy_protocol import ProxyExample
    from sure_vl.proxy_trainer import ProxyGOLDTrainer, proxy_teacher_row
    from sure_vl.teacher_view import build_teacher_view, STUDENT_FOCUS_HINT
    from sure_vl.train_proxy import configure_nonthinking_template

    if not Path(args.model).is_dir():
        raise ValueError("model must already exist locally")
    output = Path(args.output)
    if output.exists():
        raise ValueError("diagnostic output already exists")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, padding_side="left")
    template_hash = configure_nonthinking_template(processor)
    processor.image_processor.size = {**dict(processor.image_processor.size), "longest_edge": 65536}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = Image.new("RGB", (192, 128), "white")
    draw = ImageDraw.Draw(source)
    draw.ellipse((20, 28, 60, 68), fill="blue")
    draw.rectangle((125, 75, 155, 105), fill="green")
    views = build_teacher_view(source, [10, 18, 72, 80])
    example = ProxyExample(
        id="synthetic-routing-only", split="diagnostic", student_image="synthetic-full.png",
        teacher_image="synthetic-crop.png", question="What color is the circle?",
        accepted_answers=("blue",), student_image_hint=STUDENT_FOCUS_HINT,
        teacher_evidence={"source": "manually specified synthetic fixture, not CLEVR-Math",
                          "scene_graph": {"objects": [{"shape": "circle", "color": "blue"}]}},
    )
    row = {
        "prompt": [{"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": build_proxy_prompt(example)}]}],
        "student_image": views.student_image, "teacher_image": views.teacher_image,
        "example_payload": json.dumps(example.to_dict()),
    }
    trainer = object.__new__(ProxyGOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=device)
    trainer.processing_class = processor
    trainer._tokenizer = processor.tokenizer
    trainer.teacher_model = model
    trainer.generation_config = model.generation_config
    trainer.proxy_config = {"alpha": 0.5, "tau_s": 0.5, "lambda_b": 1.0,
                            "min_vision_tokens": 1, "chunk_size": 16}
    trainer.reward_config = {"answer_utility": 1.0, "rho_answer": 1.0,
                            "rho_visual": 1.0, "format_penalty": 1.0}
    trainer.opsd_weight, trainer.opsd_temperature, trainer.opsd_token_clip = 1.0, 1.1, 0.05
    trainer.diagnostic_tokens = 4
    images, prompts = trainer._extract_images_and_prompts([{"prompt": row["prompt"], "image": row["student_image"]}])
    text = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    encoded = processor(images=images, text=text, padding=True, padding_side="left",
                        add_special_tokens=False, return_tensors="pt").to(device)
    completion = ("<vision>A blue circle is visible.</vision><answer>blue</answer><confidence>"
                  "<visual_confidence>8</visual_confidence><answer_confidence>8</answer_confidence></confidence>")
    ids = torch.tensor(processor.tokenizer.encode(completion, add_special_tokens=False), device=device)
    row["_student_prompt_ids"] = encoded["input_ids"][0, encoded["attention_mask"][0].bool()]
    prompt_length = encoded["input_ids"].shape[1]
    forward = dict(encoded)
    forward["input_ids"] = torch.cat((encoded["input_ids"], ids.unsqueeze(0)), dim=1)
    forward["attention_mask"] = torch.ones_like(forward["input_ids"])
    for key in trainer._SEQUENCE_KEYS:
        if key in encoded:
            forward[key] = torch.cat((encoded[key], encoded[key].new_zeros((1, len(ids)))), dim=1)
    with torch.no_grad():
        student_logits = model(**forward, use_cache=False).logits[0, prompt_length - 1:prompt_length - 1 + len(ids)].clone()
        measured = trainer.measure_rollout(row, ids, student_logits)
    assert torch.isfinite(student_logits).all()
    assert not measured.record["proxy_fallback"]
    assert not measured.record["format_errors"]
    assert measured.record["teacher_conditioning"]["privileged_input"]["prompt_tokens"] > 0
    assert measured.record["teacher_conditioning"]["baseline_input"]["prompt_tokens"] == prompt_length
    assert "scene_graph" not in json.dumps(row["prompt"])
    assert "scene_graph" in json.dumps(proxy_teacher_row(row)["prompt"])
    record = {"validation_kind": "synthetic fixture; fixed completion; zero optimizer updates",
              "model": args.model, "chat_template_sha256": template_hash,
              "enable_thinking": False, "view": views.metadata,
              "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
              "finite_logits": True, "exact_baseline_prompt_ids_checked": True,
              "teacher_logits_detached": True, "assessment": measured.record}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
