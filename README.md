# SURE-VL

SURE-VL is a research repository for **dual-confidence vision-language reinforcement learning with a privileged visual teacher**. The current source of truth is [the derivation](%E5%8F%8C%E7%BD%AE%E4%BF%A1%E5%BA%A6RL%E4%B8%8E%E8%A7%86%E8%A7%89%E6%95%99%E5%B8%88%E7%BA%A6%E6%9D%9F_%E9%80%90%E6%AD%A5%E6%8E%A8%E5%AF%BC.md), copied from the [research discussion](https://chatgpt.com/share/6abfa273-25e4-83ea-8d52-7819107abdcb). It is a conditional mathematical design, not a trained result.

This repository contains the data contract, frozen verifier, reward, offline audit, and a **TRL GOLD training adapter** for paired restricted/clear images. The adapter combines on-policy dual-confidence RL with the original OPSD distribution loss in one optimizer update. It has not been run on a real paired dataset, so the repository makes no effectiveness claim.

**Current status:** The published VL-Calibration question/answer data does not provide the binary visual-fact correctness label `V` required by this reward. The included 16-train/8-validation pilot freezes manually reviewed visual facts for a tiny selected subset, but it does not establish a general source of `V` supervision. Training is paused while that supervision or the objective is reconsidered. No optimizer-step result is claimed.

## Method contract

For each example, the student sees a restricted image and question, while a **frozen copy of the same-origin model** sees the paired clear image and the same question. The student generates content first, then two verbal confidence reports:

```text
Visual facts: Z
Reasoning: T
Answer: y
Visual confidence: v
Conditional answer confidence: r
```

The first three fields are the **content segment**. The last two fields are the **report segment**. There is one model and one shared parameter set; these are token spans, not separate confidence heads. The implementation stores `v` and `r` as integer percentages from 0 through 100, then divides by 100 for scoring.

Before any rollout, freeze the required visual fact slots and accepted answers. `V=1` only when every required fact is correct; a missing slot is incorrect. `Y=1` when the candidate answer matches the frozen answer rule. The intended probability meanings are `v ≈ P(V=1 | H)` and `r ≈ P(Y=1 | V=1,H)`. The second report is scored only when `V=1`. As the derivation notes, a fully observed `H=(x,τ)` with deterministic checks makes these probabilities 0 or 1; nontrivial calibration requires an information-limited or population-level reading of `H`.

The reference reward is

```text
S = -(v - V)^2 - V (r - Y)^2
R = 2 V + Y + S
```

The content-only teacher term uses the sampled **student-to-teacher reverse KL** log-ratio, with the teacher evaluated on each student content prefix:

```text
k = Σ_content_tokens [log π_student(token | restricted image, prefix)
                     - log μ_teacher(token | clear image, same prefix)]
J = E[R - β k]
```

`k` on one rollout is a sampled log-ratio, not a nonnegative KL measurement. Its expectation under the current student policy is the sequence reverse KL. The equation above and `src/sure_vl/train_step.py` preserve the original derivation as a reference implementation.

### TRL v1 objective

The requested TRL implementation uses **original OPSD forward KL** on content tokens in place of the sampled reverse-KL term above. It samples a completion from the current student, evaluates the same content token IDs under the restricted-image student and frozen clear-image teacher, and minimizes

```text
L = policy_weight * mean[-R Σ_content log π(token) - S Σ_report log π(token)]
  + opsd_weight * mean_content_tokens[clipped KL(μ_teacher || π_student)]
```

The OPSD default is full-vocabulary KL with `beta=0`, softmax temperature `1.1`, and a per-vocabulary-contribution upper clip of `0.05`, matching the [original OPSD implementation](https://github.com/siyan-zhao/OPSD/blob/main/opsd_trainer.py) and [main launcher](https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_4b.sh). Clipping each contribution can make the reported loss negative. Teacher logits are detached, and report tokens receive no teacher loss. The reference reverse-KL weight is not added to this TRL objective.

The prompt uses the `<think><vision>...<reasoning>...` and boxed-answer structure from [VL-Calibration](https://github.com/Mr-Loevan/VL-Calibration), with Sure-VL's own two 0–100 confidence meanings. The paired restricted/clear image setup follows [Vision-OPD](https://github.com/VisionOPD/Vision-OPD); the distribution loss follows the **original OPSD** repository. [TRL GOLD v1.14.1](https://huggingface.co/docs/trl/v1.14.1/gold_trainer) provides VLM sampling, collation, accumulation, and checkpoints. It is an experimental TRL API, so the dependency is pinned.

## Repository layout

```text
src/sure_vl/protocol.py   Paired examples, student outputs, frozen V/Y checks
src/sure_vl/objective.py  Reward and sampled score-function accounting
src/sure_vl/train_step.py One backward and optimizer step for supplied rollouts
src/sure_vl/trl_prompt.py   Structured prompt, strict parser, token masks
src/sure_vl/trl_data.py     Frozen paired-image manifest to GOLD dataset
src/sure_vl/trl_distillation.py Original OPSD distribution loss
src/sure_vl/trl_trainer.py TRL GOLD joint policy and OPSD trainer
src/sure_vl/train_trl.py   Model-free preflight and GPU launcher
src/sure_vl/metrics.py    Split-level offline audit
src/sure_vl/cli.py        Validate and audit JSONL records
examples/                Arithmetic-only synthetic fixture
tests/                   Protocol, objective, update, and audit checks
```

## Quick start

Python 3.10+ is sufficient for the reference package; the core has no runtime dependencies.

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run sure-vl validate --examples examples/synthetic_examples.jsonl
uv run sure-vl evaluate --examples examples/synthetic_examples.jsonl --outputs examples/synthetic_outputs.jsonl
```

The synthetic image URIs are labels for testing the arithmetic and audit path; they are not image assets or training data.

For the TRL path, prepare one JSONL manifest per split with **real local image files** in `student_image` and `teacher_image`. Required visual fact values and accepted answers must be set before training. Use the same JSONL schema shown below. Then run:

```bash
uv sync --extra train
uv run --extra train sure-vl-train --config configs/sure_vl_trl_v1.json \
  --train-manifest /absolute/path/train.jsonl \
  --validation-manifest /absolute/path/validation.jsonl --check-only
accelerate launch --num_processes 8 scripts/train_trl.py \
  --config configs/sure_vl_trl_v1.json \
  --train-manifest /absolute/path/train.jsonl \
  --validation-manifest /absolute/path/validation.jsonl
```

The default setting tracks [VL-Calibration's training scale and Qwen3-VL-4B model](https://github.com/Mr-Loevan/VL-Calibration): 15 epochs, learning rate `1e-6`, eight generations, and global batch `8 GPUs × 8 samples × 4 accumulation steps = 256`. It uses `max_length=None` to preserve VLM image tokens, a 4096-token completion cap, and a 1,000,000-pixel image limit. It is a starting setting, not a completed experiment. The preflight verifies paths, split disjointness, dataset digests, and the batch equation without downloading a model. The launch resolves the model revision, loads separate student and frozen teacher copies from it, and writes `run_manifest.json` to the output directory. A GPU environment and substantial memory are required for the configured full-parameter run.

The [published VL-Calibration-12K dataset](https://modelscope.cn/datasets/xiaowenyi/VL-Calibration-12K) exposes `problem`, `answer`, and `images`; it does not provide the necessary visual-fact labels or explicit restricted/clear image pairs. The training command therefore requires a curated, frozen Sure-VL manifest rather than treating that dataset as directly trainable. `configs/vl_calibration_reference.json` records source settings; `configs/sure_vl_trl_v1.json` is the runnable configuration after data paths are supplied.

`sure_vl.train_step.train_step(rollouts, optimizer, beta=...)` remains the backend-independent reference for the sampled reverse-KL derivation. It accepts supplied token log-probabilities and does not generate from a VLM. The TRL path uses the distinct OPSD objective above. The core package has no PyTorch requirement; the `train` extra supplies the VLM training stack.

## Data format and frozen protocol

Each example is one JSON object per line:

```json
{"id":"toy-1","split":"dev","student_image":"synthetic://restricted/1","teacher_image":"synthetic://clear/1","question":"What color is the square?","required_visual_facts":{"object":"square","color":{"canonical":"blue","aliases":["azure"]}},"accepted_answers":["blue"]}
```

Each output uses the same `id` as metadata, followed by the five generated fields:

```json
{"id":"toy-1","visual_facts":{"object":"square","color":"blue"},"reasoning":"The square is blue.","answer":"blue","visual_confidence":80,"conditional_answer_confidence":90}
```

The initial verifier uses Unicode NFKC normalization, case folding, and whitespace collapsing before exact matching. This is a **repository implementation choice**, not a claim that exact matching is adequate for every VQA dataset. Set aliases before training and keep the example file fixed for the experiment. The audit includes its SHA-256 digest to identify the exact protocol/data snapshot. Keep train, validation, and test examples in separate files and do not change frozen answer aliases after looking at rollout results. The TRL adapter enforces the stated generation order and content/report token boundary; malformed completions stay in RL accounting with a format penalty and no OPSD loss.

`evaluate` takes one split per invocation. It preserves attempts with a valid metadata `id` even if their generated fields are malformed, and counts absent attempts separately. Both remain in correctness and required-fact denominators as failures. A present parsed output with a missing required visual fact also has `V=0`. Unknown or duplicate IDs, or lines with no assignable ID, raise an error. This audit does not assign an invented RL reward to malformed output; reward and calibration means state their effective parsed-output counts.

## Audit fields

The audit reports sample count, `V=1` and `Y=1` counts/rates, required fact accuracy, fact-count distribution, reward, visual Brier/ECE, conditional answer Brier/ECE and its effective `V=1` count, high-confidence errors, output coverage, and format failures. Conditional metrics are `null` when no parsed `V=1` example exists. These are offline checks on supplied outputs; the synthetic fixture is not evidence of model performance.

Before a real run, freeze the paired image source, image degradation, fact and answer annotations, and split boundaries. Only comparisons with the same frozen protocol can support an empirical claim.
