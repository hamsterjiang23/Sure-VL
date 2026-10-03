"""Zero-update visual-proxy diagnostic on fixed Qwen completion token IDs.

``prepare`` generates the frozen validation completions once. ``score`` (below)
reuses those exact IDs under alternate Student/Teacher image and prompt views.
Neither mode constructs a trainer, optimizer, gradient, or EMA update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _rows(manifest: Path, ids_file: Path | None, max_samples: int | None):
    from sure_vl.proxy_data import manifest_to_proxy_rows
    from sure_vl.proxy_evaluation import select_proxy_subset

    all_rows = manifest_to_proxy_rows(manifest)
    if ids_file is None:
        selected = list(select_proxy_subset(all_rows, size=32, seed=42))
    else:
        source = ids_file.read_text(encoding="utf-8")
        try:
            parsed = json.loads(source)
        except json.JSONDecodeError:
            parsed = [json.loads(line) for line in source.splitlines() if line.strip()]
        if isinstance(parsed, dict):
            parsed = parsed.get("subset_ids", parsed.get("ids"))
        if not isinstance(parsed, list):
            raise ValueError("ids-file must be a JSON array or an object containing subset_ids/ids")
        ids = [item if isinstance(item, str) else item.get("id") for item in parsed]
        if len(ids) != 32 or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != 32:
            raise ValueError("ids-file must freeze exactly 32 distinct nonempty example IDs")
        by_id = {row["example_id"]: row for row in all_rows}
        if set(ids) - set(by_id):
            raise ValueError("ids-file contains IDs absent from the validation manifest")
        selected = [by_id[item] for item in ids]
    if max_samples is not None:
        if max_samples < 1 or max_samples > 32:
            raise ValueError("max-samples must be between 1 and 32")
        selected = selected[:max_samples]
    return all_rows, selected


def _load_processor(model_path: Path, max_pixels: int = 65536):
    from transformers import AutoProcessor
    from sure_vl.train_proxy import configure_nonthinking_template

    processor = AutoProcessor.from_pretrained(str(model_path), padding_side="left", local_files_only=True)
    template_hash = configure_nonthinking_template(processor)
    size = dict(processor.image_processor.size)
    if size.get("shortest_edge", 0) > max_pixels:
        raise ValueError("max_pixels smaller than processor shortest_edge")
    size["longest_edge"] = max_pixels
    processor.image_processor.size = size
    return processor, template_hash


def _load_model(model_path: Path, device: str):
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encoding_context(processor: Any, device: str):
    import torch
    return SimpleNamespace(
        processing_class=processor, tools=None, chat_template=None,
        chat_template_kwargs={"enable_thinking": False},
        accelerator=SimpleNamespace(device=torch.device(device)),
    )


def _encode_messages(context: Any, messages: list[dict[str, Any]]):
    from sure_vl.training.trl.trainer import ProxyGRPOTrainer
    return ProxyGRPOTrainer._encode_messages(context, messages)


def _prepared_messages(messages: list[dict[str, Any]], image: Any):
    from trl.data_utils import prepare_multimodal_messages
    return prepare_multimodal_messages(messages, images=[image])


def _read_rgb(path: str | Path):
    from PIL import Image
    with Image.open(path) as source:
        return source.convert("RGB")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def _prepare(args: argparse.Namespace) -> int:
    import torch
    from sure_vl.proxy_evaluation import _generation_seed
    from sure_vl.proxy_prompt import split_proxy_generated_eos
    from sure_vl.proxy_protocol import ProxyExample, grade_proxy_answer
    from sure_vl.proxy_rollout import prepare_proxy_rollout
    from sure_vl.train_proxy import configure_generation_terminators

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("prepare requires one explicitly selected CUDA device")
    all_rows, rows = _rows(args.manifest, args.ids_file, args.max_samples)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.tokenpack.parent.mkdir(parents=True, exist_ok=True)
    if args.tokenpack.exists():
        raise FileExistsError(args.tokenpack)
    processor, template_hash = _load_processor(args.model)
    model = _load_model(args.model, args.device)
    stops = configure_generation_terminators(model, processor.tokenizer)
    context = _encoding_context(processor, args.device)
    gpu_index = torch.device(args.device).index
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()
    count = 0
    with torch.inference_mode(), args.tokenpack.open("x", encoding="utf-8") as destination:
        for row in rows:
            example = ProxyExample.from_dict(json.loads(row["example_payload"]))
            image = _read_rgb(row["image"])
            teacher_image = _read_rgb(row["teacher_image"])
            encoded = _encode_messages(context, _prepared_messages(row["prompt"], image))
            prefix_ids = encoded["input_ids"]
            prefix_len = int(prefix_ids.shape[1])
            seed = _generation_seed(42, example.id)
            with torch.random.fork_rng(devices=[gpu_index]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                generated = model.generate(
                    **encoded, do_sample=True, max_new_tokens=256,
                    temperature=0.6, top_p=0.95, top_k=0,
                    eos_token_id=list(stops),
                    pad_token_id=(processor.tokenizer.pad_token_id
                                  if processor.tokenizer.pad_token_id is not None
                                  else processor.tokenizer.eos_token_id),
                )
            sequences = getattr(generated, "sequences", generated)
            if sequences.ndim != 2 or sequences.shape[0] != 1 or sequences.shape[1] <= prefix_len:
                raise RuntimeError(f"invalid generated sequence for {example.id}")
            if not torch.equal(sequences[0, :prefix_len], prefix_ids[0]):
                raise RuntimeError(f"generation prompt IDs differ for {example.id}")
            ids = [int(value) for value in sequences[0, prefix_len:].tolist()]
            prepared = prepare_proxy_rollout(example, processor.tokenizer, ids, eos_token_ids=stops)
            correct, _ = grade_proxy_answer(example, prepared.parsed.answer)
            body_ids, eos_ids = split_proxy_generated_eos(processor.tokenizer, ids,
                                                          generation_eos_token_id=stops)
            record = {
                "example_id": example.id, "completion_ids": ids,
                "completion_text": processor.tokenizer.decode(body_ids, skip_special_tokens=False,
                                                                 clean_up_tokenization_spaces=False),
                "prompt_token_sha256": _digest_json(prefix_ids[0].tolist()),
                "student_image_sha256": _sha256(Path(row["image"])),
                "teacher_image_sha256": _sha256(Path(row["teacher_image"])),
                "student_image_size": list(image.size), "teacher_image_size": list(teacher_image.size),
                "source_example_sha256": _digest_json(example.to_dict()),
                "generation_seed": seed, "content_count": prepared.content_count,
                "vision_positions": list(prepared.vision_positions), "answer_correct": correct,
                "ended_with_eos": bool(eos_ids),
                "hit_max_new_tokens": len(ids) >= 256 and not eos_ids,
            }
            destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            count += 1
            print(json.dumps({"mode": "prepare", "done": count, "total": len(rows),
                              "example_id": example.id, "completion_tokens": len(ids),
                              "vision_tokens": len(prepared.vision_positions)}), flush=True)
            del encoded, generated, sequences
    _write_json(args.output / "manifest.json", {
        "mode": "prepare", "status": "completed", "zero_update": True,
        "optimizer_updates": 0, "gradient_or_backward_calls": 0,
        "source_commit": _git_commit(), "script_sha256": _sha256(Path(__file__)),
        "validation_manifest": str(args.manifest.resolve()),
        "validation_manifest_sha256": _sha256(args.manifest),
        "ids_file": None if args.ids_file is None else str(args.ids_file.resolve()),
        "ids_file_sha256": None if args.ids_file is None else _sha256(args.ids_file),
        "selected_ids": [row["example_id"] for row in rows],
        "selected_count": len(rows), "validation_count": len(all_rows),
        "student_model": str(args.model.resolve()), "chat_template_sha256": template_hash,
        "tokenpack": str(args.tokenpack.resolve()), "tokenpack_sha256": _sha256(args.tokenpack),
        "generation": {"max_new_tokens": 256, "temperature": 0.6, "top_p": 0.95,
                       "top_k": 0, "subset_seed": 42, "eos_token_ids": list(stops),
                       "enable_thinking": False, "max_pixels": 65536},
        "device": args.device,
    })
    return 0


def _pixel_sha256(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"RGB:{image.width}x{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _student_variants(image: Any, example_id: str) -> tuple[dict[str, Any], int]:
    import numpy as np
    from PIL import Image

    width, height = image.size
    blur = image.resize((16, 16), Image.Resampling.BILINEAR).resize(
        image.size, Image.Resampling.BILINEAR,
    )
    blank = Image.new("RGB", image.size, (127, 127, 127))
    occluded = image.copy()
    x0, y0 = width // 4, height // 4
    x1, y1 = x0 + width // 2, y0 + height // 2
    occluded.paste((127, 127, 127), (x0, y0, x1, y1))
    noise_seed = int.from_bytes(hashlib.sha256(f"1234:{example_id}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(noise_seed)
    pixels = np.asarray(image, dtype=np.float32)
    noisy = np.clip(np.rint(pixels + rng.normal(0.0, 50.0, pixels.shape)), 0, 255).astype(np.uint8)
    noise = Image.fromarray(noisy, "RGB")
    return {"normal": image, "blur_student": blur, "blank_student": blank,
            "occluded_student": occluded, "noise_student": noise}, noise_seed


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("*.safetensors.index.json"))
    config = model_path / "config.json"
    if config.is_file():
        files.append(config)
    if not files:
        raise FileNotFoundError(f"no frozen model weights/config in {model_path}")
    return {"path": str(model_path.resolve()),
            "files": {file.name: _sha256(file) for file in files}}


def _tokenpack_rows(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not records or len(records) > 32:
        raise ValueError("tokenpack must contain 1..32 frozen validation samples")
    ids = [record.get("example_id") for record in records]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("tokenpack has empty or duplicated example IDs")
    for record in records:
        completion = record.get("completion_ids")
        if (not isinstance(completion, list) or not completion or len(completion) > 256
                or any(type(item) is not int or item < 0 for item in completion)):
            raise ValueError("tokenpack completion IDs must be a nonempty integer array of at most 256 tokens")
    return records


def _token_diagnostics(p_cpu: Any, q_plus_cpu: Any, q_minus_cpu: Any, *, device: str,
                       alpha: float = 0.5, lambda_b: float = 1.0, tau_s: float = 0.5,
                       chunk_size: int = 16) -> tuple[list[dict[str, float]], dict[str, Any]]:
    import torch

    count = 0 if p_cpu is None else p_cpu.shape[0]
    if count == 0:
        empty = {key: None for key in ("raw_js", "baseline_js", "corrected_gap", "teacher_entropy",
                                             "student_entropy", "uncertainty")}
        return [], {"vision_tokens": 0, "proxy_fallback": True, "visual_proxy": 0.0, **empty}
    if q_plus_cpu.shape != p_cpu.shape or q_minus_cpu.shape != p_cpu.shape:
        raise RuntimeError("selected full-vocabulary logits are not aligned")
    log_two = math.log(2.0)
    log_vocab = math.log(p_cpu.shape[1])
    token_rows: list[dict[str, float]] = []

    def js(log_a, log_b):
        mix = torch.logaddexp(log_a, log_b) - log_two
        a, b = log_a.exp(), log_b.exp()
        a_term = torch.where(a > 0, a * (log_a - mix), 0.0)
        b_term = torch.where(b > 0, b * (log_b - mix), 0.0)
        return ((a_term + b_term).sum(-1) / (2.0 * log_two)).clamp(0.0, 1.0)

    def entropy(log_a):
        a = log_a.exp()
        terms = torch.where(a > 0, a * log_a, 0.0)
        return (-terms.sum(-1) / log_vocab).clamp(0.0, 1.0)

    with torch.inference_mode():
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            p = torch.log_softmax(p_cpu[start:stop].to(device=device, dtype=torch.float32), -1)
            q_plus = torch.log_softmax(q_plus_cpu[start:stop].to(device=device, dtype=torch.float32), -1)
            q_minus = torch.log_softmax(q_minus_cpu[start:stop].to(device=device, dtype=torch.float32), -1)
            raw, baseline = js(p, q_plus), js(p, q_minus)
            gap = (raw - lambda_b * baseline).clamp(0.0, 1.0)
            teacher_h, student_h = entropy(q_plus), entropy(p)
            uncertainty = alpha * gap + (1.0 - alpha) * teacher_h
            columns = [value.detach().cpu().tolist() for value in
                       (raw, baseline, gap, teacher_h, student_h, uncertainty)]
            for values in zip(*columns, strict=True):
                token_rows.append(dict(zip(("raw_js", "baseline_js", "corrected_gap",
                                            "teacher_entropy", "student_entropy", "uncertainty"),
                                           map(float, values), strict=True)))
    means = {key: sum(row[key] for row in token_rows) / count for key in token_rows[0]}
    fallback = count < 8
    proxy = 0.0 if fallback else math.exp(-means["uncertainty"] / tau_s)
    return token_rows, {"vision_tokens": count, "proxy_fallback": fallback,
                        "visual_proxy": proxy, **means}


def _score(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image
    from sure_vl.proxy_prompt import build_proxy_teacher_messages
    from sure_vl.proxy_protocol import ProxyExample, grade_proxy_answer
    from sure_vl.proxy_rollout import prepare_proxy_rollout
    from sure_vl.proxy_method import visual_certainty_proxy
    from sure_vl.train_proxy import configure_generation_terminators
    from sure_vl.training.trl.trainer import ProxyGRPOTrainer

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("score requires one explicitly selected CUDA device")
    started = time.perf_counter()
    torch.cuda.set_device(torch.device(args.device))
    # Some CUDA builds reject reset_peak_memory_stats before the first
    # allocation has initialized the device's memory-stat counters.
    torch.empty(1, device=args.device)
    torch.cuda.reset_peak_memory_stats()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    all_rows, frozen_rows = _rows(args.manifest, args.ids_file, None)
    by_id = {row["example_id"]: row for row in all_rows}
    provenance_path = args.manifest.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("official provenance.json is required to audit wrong-scene teacher crops")
    source_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    scene_groups = {item["id"]: item["original_scene_group"]
                    for item in source_provenance["examples"]}
    tokenpack = _tokenpack_rows(args.tokenpack)
    frozen_ids = [row["example_id"] for row in frozen_rows]
    if [record["example_id"] for record in tokenpack] != frozen_ids[:len(tokenpack)]:
        raise ValueError("tokenpack IDs/order differ from the frozen validation set")
    if args.max_samples is not None:
        if not 1 <= args.max_samples <= len(tokenpack):
            raise ValueError("max-samples exceeds tokenpack or is not positive")
        tokenpack = tokenpack[:args.max_samples]
    assigned = tokenpack[args.shard_index::args.num_shards]
    if not assigned:
        raise ValueError("this score shard has no samples")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    processor, template_hash = _load_processor(args.model)
    student = _load_model(args.model, args.device)
    teacher = _load_model(args.teacher, args.device)
    stops = configure_generation_terminators(student, processor.tokenizer)
    context = _encoding_context(processor, args.device)
    supports_keep = "logits_to_keep" in __import__("inspect").signature(student.forward).parameters
    teacher_supports_keep = "logits_to_keep" in __import__("inspect").signature(teacher.forward).parameters
    forward_totals = {"student": 0, "teacher": 0}
    condition_names = ("normal", "blur_student", "blank_student", "occluded_student",
                       "noise_student", "wrong_teacher_crop", "same_template_crop",
                       "template_only_teacher", "identity_teacher")
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    with torch.inference_mode(), (args.output / "token_records.jsonl").open("x", encoding="utf-8") as token_file, \
            (args.output / "samples.jsonl").open("x", encoding="utf-8") as sample_file:
        for sample_index, packed in enumerate(assigned):
            example_id = packed["example_id"]
            row = by_id[example_id]
            example = ProxyExample.from_dict(json.loads(row["example_payload"]))
            if _digest_json(example.to_dict()) != packed["source_example_sha256"]:
                raise RuntimeError(f"frozen example changed after tokenpack generation: {example_id}")
            if (_sha256(Path(row["image"])) != packed["student_image_sha256"]
                    or _sha256(Path(row["teacher_image"])) != packed["teacher_image_sha256"]):
                raise RuntimeError(f"frozen image bytes changed after tokenpack generation: {example_id}")
            original = _read_rgb(row["image"])
            correct_crop = _read_rgb(row["teacher_image"])
            if list(original.size) != packed["student_image_size"] or list(correct_crop.size) != packed["teacher_image_size"]:
                raise RuntimeError(f"frozen image dimensions changed for {example_id}")
            variants, noise_seed = _student_variants(original, example_id)
            current_group = scene_groups.get(example_id)
            if not current_group:
                raise RuntimeError(f"official original-scene group missing for {example_id}")
            donor_id = None
            current_offset = frozen_ids.index(example_id)
            for jump in range(1, len(frozen_ids)):
                candidate = frozen_ids[(current_offset + jump) % len(frozen_ids)]
                if (scene_groups.get(candidate) and scene_groups[candidate] != current_group
                        and _sha256(Path(by_id[candidate]["image"])) != packed["student_image_sha256"]
                        and _sha256(Path(by_id[candidate]["teacher_image"])) != packed["teacher_image_sha256"]):
                    donor_id = candidate
                    break
            if donor_id is None:
                raise RuntimeError(f"no different-scene teacher crop for {example_id}")
            wrong_crop = _read_rgb(by_id[donor_id]["teacher_image"]).resize(
                correct_crop.size, Image.Resampling.LANCZOS,
            )
            if _pixel_sha256(wrong_crop) == _pixel_sha256(correct_crop):
                raise RuntimeError(f"wrong teacher crop pixels equal correct crop for {example_id}")
            if sample_index < 4:
                stem = f"{sample_index:02d}-{hashlib.sha256(example_id.encode()).hexdigest()[:12]}"
                for label, image in {**variants, "correct_teacher_crop": correct_crop,
                                     "wrong_teacher_crop": wrong_crop}.items():
                    image.save(image_dir / f"{stem}-{label}.png")
            ids = packed["completion_ids"]
            prepared = prepare_proxy_rollout(example, processor.tokenizer, ids, eos_token_ids=stops)
            correct, _ = grade_proxy_answer(example, prepared.parsed.answer)
            positions = list(prepared.vision_positions)
            if (positions != packed["vision_positions"] or prepared.content_count != packed["content_count"]
                    or correct is not packed["answer_correct"]):
                raise RuntimeError(f"tokenpack parsing/answer changed for {example_id}")
            last = positions[-1] + 1 if positions else 0
            selected = torch.as_tensor(positions, dtype=torch.long, device=args.device)
            cache: dict[str, Any] = {}
            sample_forwards = {"student": 0, "teacher": 0}
            student_messages = row["prompt"]
            teacher_messages = build_proxy_teacher_messages(
                example.teacher_question or example.question, example.teacher_evidence,
            )

            def logits_for(role: str, messages: list[dict[str, Any]], image: Any):
                encoded = _encode_messages(context, _prepared_messages(messages, image))
                prompt_hash = _digest_json(encoded["input_ids"][0].tolist())
                image_hash = _pixel_sha256(image)
                fingerprint = _digest_json({"role": role, "prompt_sha256": prompt_hash,
                                            "image_pixel_sha256": image_hash,
                                            "completion_sha256": _digest_json(ids[:last])})
                if not last:
                    return None, {"input_sha256": fingerprint, "prompt_sha256": prompt_hash,
                                  "image_pixel_sha256": image_hash}
                if fingerprint not in cache:
                    model = student if role == "student" else teacher
                    supports = supports_keep if role == "student" else teacher_supports_keep
                    full = ProxyGRPOTrainer._causal_logits(model, encoded, ids[:last],
                                                           supports_logits_to_keep=supports)
                    cache[fingerprint] = full.index_select(0, selected).detach().to("cpu")
                    del full
                    forward_totals[role] += 1
                    sample_forwards[role] += 1
                return cache[fingerprint], {"input_sha256": fingerprint, "prompt_sha256": prompt_hash,
                                            "image_pixel_sha256": image_hash}

            for condition in condition_names:
                before = dict(sample_forwards)
                student_image = variants.get(condition, original)
                if condition == "wrong_teacher_crop":
                    plus_image, plus_messages = wrong_crop, teacher_messages
                elif condition == "same_template_crop":
                    plus_image, plus_messages = correct_crop, student_messages
                elif condition == "template_only_teacher":
                    plus_image, plus_messages = original, teacher_messages
                else:
                    plus_image, plus_messages = correct_crop, teacher_messages
                p, p_fingerprint = logits_for("student", student_messages, student_image)
                q_minus, minus_fingerprint = logits_for("teacher", student_messages, student_image)
                if condition == "identity_teacher":
                    q_plus, plus_fingerprint = q_minus, dict(minus_fingerprint)
                else:
                    q_plus, plus_fingerprint = logits_for("teacher", plus_messages, plus_image)
                token_metrics, summary = _token_diagnostics(p, q_plus, q_minus, device=args.device)
                if condition == "normal" and positions:
                    reference = visual_certainty_proxy(
                        p.to(args.device), q_plus.to(args.device),
                        torch.ones(len(positions), dtype=torch.bool, device=args.device),
                        q_minus.to(args.device), alpha=0.5, tau_s=0.5, lambda_b=1.0,
                        temperature=1.0, min_vision_tokens=8, chunk_size=16,
                    )
                    for key, reference_value in (("raw_js", reference.mean_raw_js),
                                                 ("baseline_js", reference.mean_baseline_js),
                                                 ("corrected_gap", reference.mean_corrected_gap),
                                                 ("teacher_entropy", reference.mean_teacher_entropy),
                                                 ("uncertainty", reference.mean_uncertainty),
                                                 ("visual_proxy", reference.certainty)):
                        if not math.isclose(summary[key], reference_value, rel_tol=1e-5, abs_tol=1e-6):
                            raise RuntimeError(f"normal proxy parity failed for {example_id}: {key}")
                for token_index, metrics in zip(positions, token_metrics, strict=True):
                    token_id = ids[token_index]
                    record = {"example_id": example_id, "condition": condition,
                              "token_index": token_index, "token_id": token_id,
                              "token_text": processor.tokenizer.decode([token_id], skip_special_tokens=False,
                                                                      clean_up_tokenization_spaces=False),
                              **metrics}
                    token_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                sample = {"example_id": example_id, "condition": condition,
                          **summary, "answer_correct": correct,
                          "original_scene_group": current_group,
                          "wrong_crop_donor_id": donor_id if condition == "wrong_teacher_crop" else None,
                          "wrong_crop_donor_original_scene_group": (
                              scene_groups[donor_id] if condition == "wrong_teacher_crop" else None),
                          "wrong_crop_donor_file_sha256": (
                              _sha256(Path(by_id[donor_id]["teacher_image"]))
                              if condition == "wrong_teacher_crop" else None),
                          "noise_seed": noise_seed if condition == "noise_student" else None,
                          "input_fingerprints": {"p": p_fingerprint["input_sha256"],
                                                 "q_plus": plus_fingerprint["input_sha256"],
                                                 "q_minus": minus_fingerprint["input_sha256"]},
                          "image_pixel_sha256": {"student": p_fingerprint["image_pixel_sha256"],
                                                  "teacher_plus": plus_fingerprint["image_pixel_sha256"],
                                                  "teacher_minus": minus_fingerprint["image_pixel_sha256"]},
                          "prompt_sha256": {"student": p_fingerprint["prompt_sha256"],
                                            "teacher_plus": plus_fingerprint["prompt_sha256"],
                                            "teacher_minus": minus_fingerprint["prompt_sha256"]},
                          "forward_counts": {role: sample_forwards[role] - before[role]
                                             for role in sample_forwards}}
                sample_file.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            sample_file.flush()
            token_file.flush()
            print(json.dumps({"mode": "score", "shard_index": args.shard_index,
                              "done": sample_index + 1, "total": len(assigned),
                              "example_id": example_id, "vision_tokens": len(positions),
                              "forward_counts": sample_forwards}), flush=True)
            del cache
    grad_tensor_count = sum(parameter.grad is not None for model in (student, teacher)
                            for parameter in model.parameters())
    if grad_tensor_count:
        raise RuntimeError("zero-update diagnostic unexpectedly created parameter gradients")
    torch.cuda.synchronize()
    _write_json(args.output / "manifest.json", {
        "mode": "score", "status": "completed", "completed": True, "zero_update": True,
        "optimizer_updates": 0, "gradient_or_backward_calls": 0, "ema_updates": 0,
        "source_commit": _git_commit(), "script_sha256": _sha256(Path(__file__)),
        "validation_manifest": str(args.manifest.resolve()),
        "validation_manifest_sha256": _sha256(args.manifest),
        "official_provenance_sha256": _sha256(provenance_path),
        "ids_file_sha256": None if args.ids_file is None else _sha256(args.ids_file),
        "tokenpack": str(args.tokenpack.resolve()), "tokenpack_sha256": _sha256(args.tokenpack),
        "selected_ids": [record["example_id"] for record in assigned],
        "sample_count": len(assigned), "condition_count": len(condition_names),
        "conditions": list(condition_names),
        "student_model": _model_fingerprint(args.model),
        "teacher_model": _model_fingerprint(args.teacher),
        "chat_template_sha256": template_hash,
        "transform": {"blur_student": "16x16 BILINEAR downsample then original-size BILINEAR",
                      "blank_student": "RGB(127,127,127)",
                      "occluded_student": "center width/2 x height/2 RGB(127,127,127)",
                      "noise_student": "numpy Gaussian RGB std=50, rounded/clipped uint8; seed sha256('1234:'+example_id) first 8 bytes",
                      "wrong_teacher_crop": "next frozen ID teacher crop resized to current crop size LANCZOS",
                      "same_template_crop": "correct teacher crop with Student prompt",
                      "template_only_teacher": "full Student image with Teacher prompt",
                      "identity_teacher": "q_plus is exact cached q_minus logits"},
        "proxy_formula": {"temperature": 1.0, "alpha": 0.5, "lambda_b": 1.0,
                          "tau_s": 0.5, "min_vision_tokens": 8, "chunk_size": 16,
                          "js_normalization": "ln(2)", "entropy_normalization": "ln(vocab_size)"},
        "generation": {"max_new_tokens": 256, "temperature": 0.6, "top_p": 0.95,
                       "top_k": 0, "enable_thinking": False, "max_pixels": 65536,
                       "eos_token_ids": list(stops)},
        "forward_counts": forward_totals,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "parameter_grad_tensor_count": grad_tensor_count,
        "token_records_sha256": _sha256(args.output / "token_records.jsonl"),
        "samples_sha256": _sha256(args.output / "samples.jsonl"),
        "device": args.device, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "memory": {"student_and_teacher_fp32": True, "same_cuda_device": True,
                   "third_model_loaded": False, "selected_vision_logits_cached_on_cpu": True,
                   "response_forward_truncated_after_last_vision_token": True},
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "score"):
        command = sub.add_parser(mode)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--tokenpack", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--ids-file", type=Path)
        command.add_argument("--max-samples", type=int)
        command.add_argument("--device", default="cuda:0")
        if mode == "score":
            command.add_argument("--teacher", type=Path, required=True)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.model.is_dir() or not args.manifest.is_file():
        parser.error("model and manifest must exist locally")
    if args.ids_file is not None and not args.ids_file.is_file():
        parser.error("ids-file does not exist")
    if args.mode == "score" and (not args.teacher.is_dir() or not args.tokenpack.is_file()):
        parser.error("teacher and tokenpack must exist locally")
    return _prepare(args) if args.mode == "prepare" else _score(args)


if __name__ == "__main__":
    raise SystemExit(main())
