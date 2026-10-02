# SURE-VL

SURE-VL is a research repository for **dual-confidence vision-language reinforcement learning with a privileged visual teacher**. The current source of truth is [the derivation](%E5%8F%8C%E7%BD%AE%E4%BF%A1%E5%BA%A6RL%E4%B8%8E%E8%A7%86%E8%A7%89%E6%95%99%E5%B8%88%E7%BA%A6%E6%9D%9F_%E9%80%90%E6%AD%A5%E6%8E%A8%E5%AF%BC.md), copied from the [research discussion](https://chatgpt.com/share/6abfa273-25e4-83ea-8d52-7819107abdcb). It is a conditional mathematical design, not a trained result.

This first repository version makes the data contract, frozen exact-match verifier, reward, sampled reverse-KL accounting, one batch update from supplied token log-probabilities, and offline audit executable. No model, dataset, or GPU launcher was specified in the derivation, so rollout generation and VLM integration remain model-specific work.

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

`k` on one rollout is a sampled log-ratio, not a nonnegative KL measurement. Its expectation under the current student policy is the sequence reverse KL. Standard OPSD uses a different, forward-KL objective and must be labeled separately.

For a first training adapter aligned with the derivation: sample from the current policy, preserve the content/report token boundary, compute teacher log-probabilities only for content tokens, use `R - βk` for the content score function and `S` for the report score function, then perform one backward pass and optimizer step per rollout batch. Do not add another direct loss for the same reverse-KL term. PPO/GRPO clipping or repeated updates would be a surrogate, not this exact on-policy estimator.

## Repository layout

```text
src/sure_vl/protocol.py   Paired examples, student outputs, frozen V/Y checks
src/sure_vl/objective.py  Reward and sampled score-function accounting
src/sure_vl/train_step.py One backward and optimizer step for supplied rollouts
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

`sure_vl.train_step.train_step(rollouts, optimizer, beta=...)` accepts `SampledRollout` records with differentiable student content/report token log-probabilities and detached teacher content log-probabilities. It combines both segment losses before exactly one `backward()` and one `optimizer.step()`. This interface does not sample a VLM, parse generated text, or score the teacher; a model adapter must supply those values from the current policy. The core package has no PyTorch requirement; its tensor interface is compatible with PyTorch, while the optional real-PyTorch test runs when PyTorch is installed.

## Data format and frozen protocol

Each example is one JSON object per line:

```json
{"id":"toy-1","split":"dev","student_image":"synthetic://restricted/1","teacher_image":"synthetic://clear/1","question":"What color is the square?","required_visual_facts":{"object":"square","color":{"canonical":"blue","aliases":["azure"]}},"accepted_answers":["blue"]}
```

Each output uses the same `id` as metadata, followed by the five generated fields:

```json
{"id":"toy-1","visual_facts":{"object":"square","color":"blue"},"reasoning":"The square is blue.","answer":"blue","visual_confidence":80,"conditional_answer_confidence":90}
```

The initial verifier uses Unicode NFKC normalization, case folding, and whitespace collapsing before exact matching. This is a **repository implementation choice**, not a claim that exact matching is adequate for every VQA dataset. Set aliases before training and keep the example file fixed for the experiment. The audit includes its SHA-256 digest to identify the exact protocol/data snapshot. Keep train, validation, and test examples in separate files and do not change frozen answer aliases after looking at rollout results. The JSONL output is a parsed record; a future model adapter must enforce the stated generation order and preserve the exact content/report token boundary.

`evaluate` takes one split per invocation. It preserves attempts with a valid metadata `id` even if their generated fields are malformed, and counts absent attempts separately. Both remain in correctness and required-fact denominators as failures. A present parsed output with a missing required visual fact also has `V=0`. Unknown or duplicate IDs, or lines with no assignable ID, raise an error. This audit does not assign an invented RL reward to malformed output; reward and calibration means state their effective parsed-output counts.

## Audit fields

The audit reports sample count, `V=1` and `Y=1` counts/rates, required fact accuracy, fact-count distribution, reward, visual Brier/ECE, conditional answer Brier/ECE and its effective `V=1` count, high-confidence errors, output coverage, and format failures. Conditional metrics are `null` when no parsed `V=1` example exists. These are offline checks on supplied outputs; the synthetic fixture is not evidence of model performance.

Before a real run, choose and document the student/teacher checkpoints, paired image source, question and answer dataset, image degradation, split boundaries, generation format, token masks, rollout sampling, KL coefficient or dual budget, and compute budget. The derivation proposes reward and teacher ablations; only comparisons with the same frozen protocol can support an empirical claim.
