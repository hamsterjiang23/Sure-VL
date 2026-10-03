# Sure-VL experiment metrics

`tracking.backend: wandb` publishes finite scalar summaries to an online W&B
run. `tracking_run.json` records the run ID and URL after online initialization.
The startup `run_config` is sent once to W&B as the run's frozen configuration,
including caller-supplied protocol settings, provenance, and input hashes.
The local `proxy_train_attempts_rank_*.jsonl` and
`proxy_validation_metrics.jsonl` files remain the detailed audit records.
Tracking disabled (`backend: none` or no tracking config) imports no W&B SDK.
The caller creates the W&B tracker on world rank 0; W&B training windows cover
that rank's on-policy attempts. Every rank retains its own
`proxy_train_attempts_rank_*.jsonl`, so W&B rank-local curves must not be read
as pooled multi-rank estimates.

Every W&B log payload contains `optimizer/successful_updates`, and the
`train/*`, `validation/*`, and `trainer/*` chart families use that metric as
their X axis. This is the number of **successful optimizer updates**, including
the EMA update only after a successful step. W&B's internal history step is
never supplied explicitly. A skipped optimizer attempt can therefore produce
another history event at the same X value. `trainer/state_global_step` and
`optimizer/attempted_steps`, `optimizer/skipped_updates`, and
`optimizer/teacher_ema_updates` expose the distinction.

| Family | Source and scope | Main keys and denominators |
| --- | --- | --- |
| `train/*` | Buffered **training attempts** since the prior Trainer log event. Includes attempts from skipped updates; no claim of held-out performance. | `attempt_count` and `sample_count` count buffered completions. `answer_correct_count / sample_count` gives `answer_accuracy`; the active GRPO path has known ground truth for every row. An unextractable answer is `Y=0`; absence of a report is counted separately. Historical GOLD records used different label-availability semantics and must not be pooled with these runs. `output_coverage/*` counts and rates use `sample_count`, except answer-report coverage where the eligible label count is explicit. |
| `train/visual_proxy_*` | Student visual text compared with the frozen teacher-grounded internal proxy S. It is **not** factual visual accuracy. | `visual_proxy_eligible_count` excludes fallback S=0; `visual_proxy_pair_count` also requires a valid visual report. `visual_proxy_stats/mean` and `/variance` use eligible rows only. MSE, binned error, correlation, and report coverage use their named eligible/pair counts; missing values are omitted. |
| `train/proxy_components/*` | Visual-token distribution diagnostics for nonfallback rows. | Each component has `/mean` and `/sample_count`, including `raw_js`, `baseline_js`, `corrected_gap`, `teacher_entropy`, and `uncertainty` when present. `corrected_gap` is the bounded heuristic correction, not a causal image-only estimate. |
| `train/opsd_components/*` and `train/opsd_diagnostic_*` | Forward-KL and sparse diagnostic readings on generated **content** tokens, not confidence-report tokens. | Every component has its own `/sample_count`, including `raw_forward_kl_mean`, `clipped_loss_mean`, `clipped_vocabulary_fraction`, `top1_disagreement_rate`, and `weighted_logit_grad_l2` when measured. The last is a weighted gradient norm with respect to sampled **logits**, not model parameters or an observed update. `opsd_diagnostic_rows` counts rows with sampled diagnostic positions, `opsd_diagnostic_positions` sums those positions, and `opsd_diagnostic_coverage` divides by `attempt_count`. A zero position count means no sparse diagnostic sample, not zero KL. |
| `train/visual_confidence_score/*`, `train/answer_confidence_score/*` | Parser-valid raw reports on the 0..10 integer scale, separate from normalized 0..1 calibration metrics. | `/valid_count`, `/coverage` (valid reports / `attempt_count`), `/mean` over valid reports only, and `/count_0` through `/count_10`. A missing mean remains absent when no report is valid; bins with zero observed reports are genuine zero counts. |
| `train/reward_components/*` | The actual bounded answer, visual-proxy, format, and total reward terms on attempted completions. | Each `/mean` includes its own `/sample_count`; missing reports receive the trainer's explicit reward penalty, while calibration metrics leave their values missing. |
| `train/answer_*`, `train/high_confidence_answer_errors/*` | Answer event Y and valid answer-confidence reports. | Brier/ECE10 use `answer_confidence_count` valid answer-report/label pairs. High-confidence error rate uses `high_confidence_answer_errors/sample_count`, which can be zero; then the rate is absent. |
| `train/output_coverage/*`, `train/format_error_*_count`, `train/vision_tokens/*` | Structural output quality and token counts for the buffered attempts. | Counts and rates show format, report, nonfallback, and label coverage. Vision-token distribution uses all attempts, including zero-token outputs. |
| `validation/*` | A frozen held-out subset at step 0 and configured intervals. | Same metric definitions as the training summary, but computed only on validation attempts. `validation/generation/hit_max_new_tokens_count` counts generation-cap hits and should be read with `validation/generation/sample_count`. The subset ID/hash and full attempts stay in local validation artifacts. |
| `train/visual_proxy_stats/*`, `validation/visual_proxy_stats/*` | The original exponential visual proxy. | Population `std` is in normalized [0,1] units; multiply by 10 for score units. Distribution statistics exclude fallback and should be read with valid sample count and coverage. |
| `trainer/*` | Hugging Face Trainer logs and local loss measurements. | `trainer/loss`, `trainer/grad_norm`, `trainer/learning_rate`, `trainer/epoch`, and available `trainer/rank_local_policy_loss` / `rank_local_opsd_loss` are optimization telemetry. Rank-local means with empty-denominator placeholder zeros are intentionally not uploaded; aggregate windows above provide explicit counts. |

Nested scalar keys from the summary, such as
`validation/output_coverage/format_clean_rate` and
`train/reward_components/total/mean`, are flattened under their family. Arrays
such as calibration bins and risk-coverage curves, IDs, paths, strings,
booleans, `None`, NaN, and infinity are not uploaded as scalar metrics. A metric
without eligible samples is absent from W&B; the associated count remains 0.
The JSONL audit artifacts retain full report structures and generated text.

W&B online initialization must return a usable run ID and URL. Offline or
disabled initialization raises an error and does not write
`tracking_run.json`; it is not presented as an uploaded experiment. The module
does not read or print API keys.


## Native TRL GRPO additions

The TRL callback sends the upstream Trainer logs through the same tracker;
`report_to=[]` prevents the stock W&B callback from redefining `train/*` axes.
Native sampling and loss computation remain in `trl.GRPOTrainer`.

| Cloud key (under `trainer/`) | Meaning |
| --- | --- |
| `loss` | Native accumulated joint loss for the logged optimizer interval |
| `sure_vl_grpo_loss_scaled`, `sure_vl_grpo_loss_unscaled` | Per-microbatch native GRPO loss, respectively after and before accumulation division |
| `sure_vl_opsd_loss_unscaled`, `sure_vl_opsd_weighted_contribution_scaled` | Content-token OPSD mean, and its actual weighted contribution to the accumulated loss |
| `sure_vl_joint_loss_scaled` | Sum of native GRPO and OPSD contributions for a microbatch |
| `sure_vl_zero_advantage_group_fraction` | Fraction of question groups with no content or report reward differences |
| `sure_vl_active_policy_tokens`, `sure_vl_policy_active_token_fraction` | Sampled tokens receiving nonzero segment advantage |
| `grad_norm`, `grad_norm_after_clip` | Parameter gradient norm before native clipping, and actual norm entering Adam |
| `optimizer_state_max_step` | Actual maximum Adam state step, checked against successful-update evidence after each step |
| Native `entropy`, reward std and clip-region logs | Official GRPO rollout/optimization telemetry; empty calibration denominators remain separately visible |

At fresh on-policy ratio 1, a group-centered policy loss may be near zero
while the gradient is nonzero. Read active tokens and reward spread as well
as the scalar loss. An identical-reward group has no policy signal; its
content OPSD can still provide a gradient. `optimizer_evidence.jsonl`
retains actual Adam/EMA/update counts at every successful update.
