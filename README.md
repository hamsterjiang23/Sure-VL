# Sure-VL

Teacher-grounded internal visual confidence and answer confidence for vision-language models. The method is based on [the full derivation](教师学生差距构造内部视觉置信度_完整推导.md).

## Training backends

| Backend | Implementation | Configuration | Environment / entry point |
| --- | --- | --- | --- |
| TRL | `src/sure_vl/training/trl/` — a subclass of official `trl.GRPOTrainer` 1.14.1 | `configs/trl/` | `uv sync --extra train --extra tracking --frozen`; `sure-vl-train-trl` |
| veRL | `src/sure_vl/training/verl/` — extension using vendored official veRL | `configs/verl/` | Separate `envs/verl/`; see [backend status](docs/verl_backend.md) |

`sure-vl-train` is an alias for the TRL entry point. The previous GOLD and binary-label interfaces are retained as `sure-vl-train-gold-legacy` and `sure-vl-train-legacy`. Historical flat configs and `train_proxy.py` describe those legacy routes. They are not the active training recipe.

[Reference source snapshots](third_party/README.md) include OPSD, Vision-OPD, VL-Calibration, and **official veRL**, with original licenses, fixed commits and a per-file provenance manifest. The official veRL snapshot is separate from Vision-OPD's veRL fork.

## Model output and Teacher input

The Student receives the official full image with a red region marker and the question. Built-in thinking is disabled. The adapted VL-Calibration System Prompt requests a brief visual description, brief task deduction, an answer, and two **0–10 integer** reports:

```text
<vision>brief question-relevant visual description</vision>
<reason>brief deduction based on that visual evidence</reason>
<answer>A</answer>
<confidence><visual_confidence>8</visual_confidence><answer_confidence>7</answer_confidence></confidence>
```

Both roles use System + User messages and require these four blocks in order (`vision-reason-answer-confidence-v2`). The explicit `<reason>` is part of the answer protocol; the Qwen chat template still closes its built-in thinking prefix. Legacy responses missing `<reason>` remain readable but receive a format error.

The Teacher receives the **official enhanced crop**, a **different Teacher System Prompt**, the clean question, and optional additional evidence such as a scene graph. This adapts Vision-OPD's `bbox_images` replacement and optional `teacher_prompt` mechanism. The official Vision-OPD-6K data has no scene graph; the manifest supports `teacher_evidence` when supplied by another dataset. Answers used for grading never enter either prompt. See the [exact prompt protocol and source mapping](docs/prompt-protocol-v2.md).

The Teacher scores the exact Student-generated content token IDs. On visual-description tokens, Student/Teacher JS divergence, same-input Teacher baseline, and Teacher entropy form the detached internal target `S`. The baseline uses the exact Student prompt IDs and image. This target reflects image, evidence, template conditioning and model differences; it is a confidence proxy, not a factual visual-accuracy label. Visual spans with fewer than eight tokens fall back to `S=0` with explicit coverage counts.

The report integers are divided by 10 for reward calculation:

- Content reward: `Y − (r−Y)² − (v−S)² − format_penalty`.
- Report reward: `−(r−Y)² − (v−S)² − format_penalty`.
- Missing reports receive the corresponding maximum penalty. Known ground truth with an absent answer has `Y=0`.

For A–D data, a response such as `C. sneakers` is graded correct only if the description exactly matches option C in the question; its answer format is still noncanonical. Missing reports stay missing in calibration metrics.

## Native TRL training

The TRL Trainer owns sampling, accumulation, optimizer, scheduler, clipping, checkpoints and native training logs. The Sure-VL subclass supplies Teacher scoring, separate content/report advantages, and content-only OPSD.

Each question produces four actual sampled responses. Content and report rewards are **separately centered within that group**, without dividing by reward standard deviation. Native GRPO uses sequence mean of token mean losses. A constant reward group therefore has zero policy advantage. This is an explicit training variant of the derivation's raw score-function sum.

OPSD follows the original repository's full-vocabulary forward KL, temperature 1.1 and pointwise vocabulary cap 0.05, averaged over content tokens (`vision`, `reason`, and `answer`, including their delimiters). It uses the **same differentiable Student forward** as GRPO. The visual proxy uses only the `vision` body. Confidence tokens receive no distillation. Teacher parameters are frozen during a group and updated by FP32 EMA only after a successful optimizer step.

The V100 recipe uses the cached Qwen3.5-0.8B model, FP32, one GPU, microbatch 1, accumulation 4, learning rate `1e-6`, at most 256 generated tokens and 65,536 image pixels.

Two V100s were probed on the test server. Full-model DDP failed during NCCL broadcast; placing Student on GPU 0 and Teacher on GPU 1 also encountered a CUDA launch timeout during generation, before any optimizer update. The default therefore uses one visible GPU. `qwen35_08b_visionopd_ddp2_100step.json` and `qwen35_08b_visionopd_teacher_gpu1_100step.json` preserve the attempted configurations; neither has passed the GPU runtime gate on this server.

## Official data and launch

The data source is [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K), fixed at `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4`. All seven complete LFS archives and all **6,241 paired PNGs** have passed verification on the test server. The default recipe and `qwen35_08b_visionopd_full_100step.json` use frozen **5,985 training / 256 held-out** examples. [The full-data audit](docs/evidence/visionopd_full_v1.json) records archive, image, manifest and provenance checks.

The earlier prefix recipe remains available as `qwen35_08b_visionopd_prefix640_100step.json`: **512 training / 128 held-out** examples, with a fixed Student archive prefix and complete verified Teacher archive. Prefix availability introduces selection bias. Both are project holdouts from the official training data, with original-scene and actual-image-hash separation. A frozen run's selection never changes.

```bash
cd /data/LHJ/Sure-VL
uv sync --extra train --extra tracking --frozen
uv run --extra train --extra tracking sure-vl-train-trl \
  --config configs/trl/qwen35_08b_visionopd_protocol_v2_100step.json --check-only
CUDA_VISIBLE_DEVICES=0 uv run --no-sync --frozen --extra train --extra tracking sure-vl-train-trl \
  --config configs/trl/qwen35_08b_visionopd_protocol_v2_100step.json
```

The recipe requests **100 successful optimizer updates** and online W&B. Validation uses the same frozen 32 held-out IDs at step 0, every 20 updates, and step 100. `training_completed.json` is written only after Trainer, Adam state, successful updates, EMA and validation gates agree. A command, process start, rollout or source test is not completion evidence.

See the [metric registry](docs/metrics_registry.md) for loss scale, gradient norms, active policy tokens, output coverage, integer report histograms, calibration denominators, Teacher JS/entropy, OPSD diagnostics and update counts. All cloud families use the successful-update axis through one tracking callback.

## Training audit

The earlier official-data GOLD run was **stopped at 80 attempts** after confirming that its Qwen3.5 generation dropped `mm_token_type_ids` while rescoring used multimodal positions. Its results are invalid for on-policy training. [The audit](docs/trl_training_audit.md) records the mechanism and the native GRPO sampling parity check. That check used real images and actual sampled IDs, with **zero optimizer updates**; subsequent update evidence is reported separately.

Historical [100-update pilot evidence](docs/validation-proxy-100step-v1.md) demonstrated optimizer execution but failed output quality. Neither those runs nor source tests establish calibration improvement.

The historical [full official-data TRL run](docs/validation-full-grpo-100step-v1.md) completed **100 successful updates**, with Adam and Teacher EMA both at 100, zero skips, 400 actual rollouts and all six fixed assessments. Its W&B state is finished. It used the earlier output protocol without `<reason>`; its 32/32 format coverage does not validate the current four-block protocol. The report preserves paired calibration denominators and the limits of this small-model result.
