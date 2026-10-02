# Sure-VL

Sure-VL studies verbal confidence in vision-language models trained with a privileged visual teacher. The current method implements the **teacher-grounded internal visual certainty proxy** in [the complete derivation](./教师学生差距构造内部视觉置信度_完整推导.md). See [runtime validation](docs/validation-proxy-v1.md) for tested pipeline behavior and its limits; the small pilot does not establish effectiveness.

## Current method

The Student sees a restricted image `I-` and a question. This first version generates a brief free-text visual description, an answer, and two integer reports from 0 to 10, without a reasoning or thinking block:

```text
<vision>brief visual description</vision>
<answer>final answer</answer>
<confidence><visual_confidence>v</visual_confidence><answer_confidence>r</answer_confidence></confidence>
```

The two report integers are divided by 10 before scoring, so `v` and `r` in the reward equation below lie in `[0,1]`.

The linked derivation writes content as `(Z,T,y)` and shows a `<reasoning>` section. The active nonthinking pilot sets `T` to the empty sequence; its visual proxy depends on `Z` tokens and the answer check depends on `y`, so neither requires generated chain-of-thought text. The parser can recover fields from older `<think>` outputs for audits, but flags them as noncanonical for this pilot.

An EMA Teacher sees a privileged image `I+`, optional evidence `E+` (for example, a scene graph), and a **separate Teacher prompt**. It scores the **same Student completion token prefixes** without generating a replacement trajectory. On `<vision>` tokens, normalized Student–Teacher Jensen–Shannon divergence, an optional same-view Teacher baseline, and privileged-view Teacher entropy form a detached visual certainty target `S_vis`. A missing or too-short visual span receives `S_vis=0`. The target is recomputed from each rollout; it is not a stored label.

The baseline `q-` uses the same EMA model with the **exact Student prompt IDs, Student image, and no privileged evidence**. Its prompt IDs are checked at runtime. With different Teacher/Student templates, the corrected gap reflects image, evidence, and template conditioning as well as model drift. JS subtraction is a heuristic correction; it does not isolate a purely visual causal effect.

- `visual_confidence` (`v`) estimates **teacher-grounded internal visual certainty**. It is **not** the probability that the free-text visual description is factually correct.
- `answer_confidence` (`r`) estimates the **unconditional probability that the final answer is correct**. It is not conditioned on a visual-correctness event.

The first-version task reward is `R = Y - (r-Y)^2 - (v-S_vis)^2`, where `Y` is checked against frozen accepted answers. Content and report spans receive separate policy weights. The adapter also applies the [original OPSD](https://github.com/siyan-zhao/OPSD/blob/main/opsd_trainer.py) **full-vocabulary forward KL** to content tokens only, with main-launcher defaults `beta=0`, temperature `1.1`, and per-vocabulary contribution cap `0.05`. The derivation presents a reverse-KL constrained objective; OPSD forward KL is this repository's explicit implementation choice, not an algebraic identity with it. Confidence tokens receive no OPSD loss. The EMA Teacher updates only after a successful Student optimizer update.

The formula above applies to valid reports and extractable answers. A missing report or unavailable answer label receives the corresponding maximal squared-score penalty; malformed structure or an unusable visual span adds a configurable format penalty (default 1). The content policy receives answer utility plus report penalties, while the report policy receives the penalties. These fallbacks are explicit implementation choices. Missing labels and fallback proxy values are excluded from the corresponding calibration denominators.

The output order and 0–10 scale follow [VL-Calibration's published prompt](https://github.com/Mr-Loevan/VL-Calibration/blob/main/examples/format_prompt/Standard_Decouple.jinja). Sure-VL adapts its visual section to a direct, brief description and changes the two report meanings to internal visual certainty and unconditional answer correctness. Broad training scale takes inspiration from [VL-Calibration](https://github.com/Mr-Loevan/VL-Calibration), regional/global inputs from [Vision-OPD](https://github.com/VisionOPD/Vision-OPD), and the distribution loss from original OPSD. [TRL GOLD 1.14.1](https://huggingface.co/docs/trl/v1.14.1/gold_trainer) handles VLM sampling and optimizer accumulation. This TRL API is experimental and pinned in `pyproject.toml`.

## Privileged image and Teacher prompt

[Vision-OPD section 3.2](https://arxiv.org/html/2605.18740v1) isolates a question-relevant evidence region and resizes the crop by **2x in width and height**. The Student receives the full image with a red bounding box and a spatial hint; the Teacher receives the crop. This is a regional perception advantage. The public [data preparation code](https://github.com/VisionOPD/Vision-OPD/blob/06860e69b5ed9dc24e96ca5c855f3a4ef25976aa/scripts/prepare_data.py) consumes precomputed `teacher_images`; it does not publish the crop-generation or interpolation implementation. Sure-VL implements the stated crop/2x method with LANCZOS interpolation as an explicit project choice.

The Teacher template in `proxy_prompt.py` instructs it to describe question-relevant facts from its local image and available evidence, respect the visible region's scope, and avoid inferring unseen global facts. The Student template asks for evidence from its ordinary image. Optional `teacher_evidence` is serialized into the Teacher prompt only; `accepted_answers` is used solely by the answer verifier. Both templates keep direct output and the same 0–10 integer report contract.

Build a new paired dataset from JSONL with `id`, `split`, `source_image`, `question`, `accepted_answers`, `evidence_bbox_xyxy` (source-image pixel coordinates), and optional `teacher_evidence`:

```bash
uv run --extra train python scripts/build_privileged_proxy_data.py \
  --input-jsonl /path/to/source.jsonl \
  --output-dir /data/LHJ/Sure-VL/data/privileged_proxy_v1
```

The evidence region must contain the information needed for the question. The builder requires an explicit region; it does not invent a question-relevant box. `--allow-no-roi` explicitly permits an unchanged image and records `no_evidence_roi`. Image transforms, source/manifest hashes, and evidence presence are recorded in `provenance.json`. Evidence can be text or a JSON object/list, including a dataset-provided scene graph. This input route does not itself validate a CLEVR-Math scene graph source or generate missing annotations.

## Data contract

Each JSONL manifest contains one split and six required fields per example, plus optional `student_image_hint` and `teacher_evidence`:

```json
{"id":"sample-1","split":"train","student_image":"images/sample-1.restricted.png","teacher_image":"images/sample-1.clear.png","question":"What shape is shown?","accepted_answers":["circle"]}
```

Image paths are resolved relative to the manifest. Student and Teacher paths must be distinct; train and validation cannot reuse IDs or image paths. `accepted_answers` is frozen before training and checked by normalized exact matching. This strict rule can reject semantically equivalent answers with extra units, punctuation, or unlisted aliases; audit those cases before treating pilot accuracy as a research result. **No visual-fact slots, binary visual label `V`, or static proxy target are required.** An unextractable answer is not silently assigned `Y=0`. A malformed or absent confidence block does not disable content Teacher supervision.

The [official VL-Calibration-12K dataset](https://modelscope.cn/datasets/xiaowenyi/VL-Calibration-12K) supplies questions, answers, and images, but no restricted/clear pairs. `scripts/build_vlcalib_proxy_pilot.py` uses a frozen 16-train/8-validation selection, verifies source files and rows, and creates a restricted Student view by downsampling then upsampling the image. It writes manifests, image hashes, and `provenance.json`. This tiny selected pilot tests the pipeline; it is not a representative effectiveness benchmark. Its selection file contains older visual-fact annotations, but the proxy builder does not read or emit them.

## Install, build data, and preflight

Use [uv](https://docs.astral.sh/uv/) for environments. The core package needs no training dependencies; the `train` extra supplies PyTorch, Transformers, datasets, Accelerate, and TRL.

```bash
uv sync --extra train --frozen
uv run --extra train python -m unittest discover -s tests -q
uv run --extra train --with pyarrow python scripts/build_vlcalib_proxy_pilot.py \
  --output-dir /data/LHJ/Sure-VL/data/vlcalib_proxy_pilot_v1
```

Pass `--train-parquet` and `--validation-parquet` to the builder if the official files are already local. Otherwise it fetches them and verifies their pinned hashes. `--check-only` validates the config, paths, split separation, dataset hashes, and batch equation without loading a model:

```bash
uv run --extra train sure-vl-train \
  --config configs/sure_vl_proxy_v1.json \
  --train-manifest /data/LHJ/Sure-VL/data/vlcalib_proxy_pilot_v1/train.jsonl \
  --validation-manifest /data/LHJ/Sure-VL/data/vlcalib_proxy_pilot_v1/validation.jsonl \
  --check-only
```

The checked-in `configs/sure_vl_proxy_v1.json` is a **reference-scale** Qwen3-VL-4B setting based on VL-Calibration: 8 GPUs, 15 epochs, eight generations, and global batch 256. Use the smaller configs below for the V100 pilot.

## Single-V100 smoke, then 100 steps

The two pilot configs point to the ordinary Qwen3.5-0.8B snapshot at `/data/LHJ/PGR-Probe/.hf_cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`. They use one GPU, batch size 1, accumulation 1, one generation, 65,536 maximum image pixels, and 256 maximum completion tokens. Build the paired pilot data above, then check and run the **one-step smoke** under `/data/LHJ/Sure-VL`:

```bash
ssh v100-2-hamster
cd /data/LHJ/Sure-VL
uv sync --extra train --frozen
CUDA_VISIBLE_DEVICES=0 uv run --extra train sure-vl-train \
  --config configs/qwen35_08b_v100_proxy_smoke.json --check-only
CUDA_VISIBLE_DEVICES=0 uv run --extra train python scripts/train_trl.py \
  --config configs/qwen35_08b_v100_proxy_smoke.json
```

After confirming one **successful** optimizer update, the smoke validation records, and memory headroom, check and launch the separate 100-step run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra train sure-vl-train \
  --config configs/qwen35_08b_v100_proxy_100step.json --check-only
CUDA_VISIBLE_DEVICES=0 uv run --extra train python scripts/train_trl.py \
  --config configs/qwen35_08b_v100_proxy_100step.json
```

Both configs set `enable_thinking=false`. The processor and tokenizer use the same Qwen nonthinking chat template: its empty `<think></think>` prefill stays in the prompt, and generated tokens start with the direct `<vision>` format above. The current small-model V100 recipe uses FP32 for Student and Teacher (`fp16=false`, `bf16=false`) and keeps FP32 EMA master weights, so check memory headroom during the smoke. V100 does not support bf16. The trainer rejects nonfinite Student gradients before an optimizer update, and EMA advances only after a successful update.

An earlier fp16 nonthinking probe produced a usable visual proxy, but backward gradients became NaN and the optimizer update was skipped. It did **not** complete a training step. fp16 AMP remains optional; that path keeps Student trainable parameters in FP32, runs the Teacher in fp16, and keeps FP32 EMA masters. Enable it only after a bounded smoke verifies finite gradients and zero skipped updates.

These are launch instructions, not completed-run evidence. A completed 100-step claim requires `training_completed.json`, 100 **successful** optimizer updates in `optimizer_evidence`, and same-subset validation at step 0 and step 100. `TrainerState.global_step`, rollout count, or a launch command alone are insufficient evidence.

## Fixed validation and evidence

The proxy run selects a held-out subset by ID hash before observing outputs, then evaluates the same examples at step 0, every `validation.every_n_steps`, and the requested final step with fixed per-example generation seeds. The output directory records `run_manifest.json`, `proxy_validation_metrics.jsonl`, per-step raw attempt JSONL, training telemetry, and `training_completed.json` only after completion checks pass.

The audit includes answer accuracy and answer-confidence Brier/ECE with effective label counts; visual-report squared/binned error **against the internal proxy**; proxy coverage and distribution; Teacher gap and entropy components; vision length, format coverage, and OPSD signal. Visual-proxy error is not factual visual calibration. Compare checkpoints only with the same frozen subset, split, proxy settings, and image restriction. The 16/8 pilot and a 100-step smoke can establish pipeline behavior, not benchmark improvement.

The [original paired-view 100-update audit](docs/validation-proxy-100step-v1.md) confirms 100 successful optimizer and EMA updates with zero skips, but all eight validation outputs lose extractable answers and usable vision spans from step 20 onward. This is a failed output-quality diagnostic. It predates the new crop/evidence/Teacher-template implementation. The new [actual-model routing fixture](docs/evidence/privileged_teacher_fixture.json) checks a synthetic crop and scene graph with a fixed completion and zero optimizer updates; it is not a CLEVR-Math evaluation or an effectiveness result.

## Layout and legacy interface

| Path | Purpose |
| --- | --- |
| `src/sure_vl/proxy_protocol.py`, `proxy_prompt.py`, `proxy_data.py` | Answer-only manifest, free-text output, paired GOLD rows |
| `src/sure_vl/proxy_method.py`, `proxy_trainer.py`, `teacher_ema.py` | Detached proxy, joint training, successful-step EMA |
| `src/sure_vl/proxy_metrics.py`, `proxy_evaluation.py` | Fixed-subset readback and diagnostics |
| `src/sure_vl/train_proxy.py`, `configs/sure_vl_proxy_v1.json` | Proxy preflight and launch |
| `src/sure_vl/trl_distillation.py` | Original OPSD loss |

The earlier binary-`V` design remains under `protocol.py`, `objective.py`, `trl_prompt.py`, `trl_data.py`, `trl_trainer.py`, `train_trl.py`, and `configs/sure_vl_trl_v1.json` for comparison. Its `sure-vl` audit command and `sure-vl-train-legacy` entry point still require visual-fact labels and **conditional** answer confidence. They are not the default proxy method. The earlier derivation is [archived here](./双置信度RL与视觉教师约束_逐步推导.md).
