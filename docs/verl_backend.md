# Isolated veRL backend

Sure-VL's first veRL backend uses the vendored `verl.protocol.DataProto` and
`verl.single_controller.ray.RayWorkerGroup` for actual Ray actor calls. A
custom `SureVLProxyWorker` owns the Student, EMA Teacher, AdamW optimizer,
sampling, scoring, and update on **one CUDA GPU**. It does not instantiate a
TRL Trainer or veRL's PPO/FSDP/vLLM trainer. The current config uses world
size one, one prompt per update, four sampled completions per prompt, four
sequential gradient accumulations, FP32, and Qwen's nonthinking mode.

## Environment and source

From the repository root, use the separately locked uv project. Its
`sure-vl` and `verl` dependencies must be editable paths to this checkout, so
edits under `src/sure_vl/` and the independent, official upstream snapshot
`third_party/verl/verl/` are used directly. The Vision-OPD repository also
contains a veRL fork as reference code; it is a different source tree.

```bash
uv sync --project envs/verl --frozen
uv run --project envs/verl --frozen python -c 'import torch, tensordict, ray, verl; print(torch.__version__, tensordict.__version__, ray.__version__, verl.__file__)'
env SURE_VL_RUN_RAY_TRANSPORT=1 uv run --project envs/verl --frozen python -m unittest discover -s tests -p 'test_verl_transport.py' -v
```

The last command is a CPU-only transport smoke: a two-row `DataProto` travels
through a real one-worker Ray placement group and returns through veRL's
dispatch/collect path. It does not load a model. Keep it separate from the
regular test suite because it starts a local Ray instance. A clean import and
this test are the runtime compatibility gate for the locked Torch 2.6.0 and
TensorDict 0.10.0 pair. TensorDict's package metadata does not constrain its
Torch version; resolution alone cannot establish compatibility.

This isolated lock points its editable `verl` dependency to
`third_party/verl`. The upstream package specifies `transformers<5.13`,
whereas the Qwen3.5 model used here requires the pinned 5.18.0 runtime. The
project records an explicit uv dependency override for this prototype;
successful resolution is not evidence that the imported veRL code and model
work together. Keep the import, Ray round trip, and bounded model check as
separate runtime gates.

The custom WorkerGroup/DataProto import path needs Torch, NumPy, Ray,
TensorDict, Packaging, and Transformers plus their transitive packages.
Pandas, Hydra, and TorchData are declared by the official vendored `verl`
package, even though this custom transport path does not import their PPO,
SFT, or dataset modules. An editable install still resolves those declared
package dependencies. The isolated project separately pins Pillow,
torchvision, Accelerate, and W&B for model/image execution and tracking.
Its explicit CPU API dependencies include `TransferQueue==0.1.10`,
`msgspec==0.22.0`, and `pyzmq==27.2.0`; the vendored `DataProto` import reaches
veRL's TransferQueue compatibility module.

## One update

The driver validates the frozen train and validation manifests before Ray
starts. It creates a single GPU `RayWorkerGroup` and calls these registered
worker methods with veRL `DataProto` objects:

1. `generate_sequences` samples four completions for the same prompt from
   the Student at temperature 1, top-p 1, top-k 0, preserving sampled token
   IDs and using a distinct seed for each sample.
2. `score_sequences` evaluates the pre-update Student and frozen Teacher on
   all four sets of IDs. The privileged Teacher receives the official Vision-OPD crop
   and clean question (plus optional evidence when present). The baseline
   Teacher receives the Student image and exact Student prompt IDs, without
   privileged evidence. It computes a detached internal visual certainty
   proxy and detached answer/report rewards. It centers content and report
   rewards separately within the four-sample prompt group through official
   veRL `compute_grpo_outcome_advantage`, without standard deviation scaling.
3. `update_actor` checks the rollout nonce, model version, example, masks,
   rewards, and advantages, then recomputes Student logits with gradients.
   It calls official veRL `compute_policy_loss_vanilla` in
   `seq-mean-token-mean` mode for each sample, using the detached log
   probability from the same forward pass as `old_log_prob` for this
   one-iteration update. Content-only OPSD forward KL is averaged over each
   sample's content tokens, then over the group. Four backwards accumulate
   before one AdamW step; EMA advances only after a finite, successful step.
   A skipped update cannot silently count as successful.

The corrected JS gap is a heuristic for the *combined* image, evidence,
template, and model difference. Its certainty `S` is an internal proxy, not
factual visual correctness or a ground-truth `V` label. The worker does not
use FSDP or vLLM rollout. The policy-loss function computes a ratio, but its
forward value is one because `old_log_prob` is detached from the current
log probability in the single update iteration.

The logged `policy_loss` is the actual official ratio-based GRPO objective:
for each of four completions it averages `-advantage` over that completion's
generated tokens, then divides by four. Its gradient flows through the ratio.
`opsd_loss` likewise averages each completion's content-token loss before the
four-sample average. `score_function_token_mean_diagnostic` weights all tokens
together and is only a diagnostic; it is not the optimized policy loss.
`grad_norm` is the norm returned by PyTorch's clipping call before clipping,
while `grad_norm_after_clip` measures the gradients afterward.
The Worker uses the same linear learning-rate recipe as the TRL run: no warmup
and decay over the configured successful-update budget. It advances the
scheduler only after `optimizer.step()` succeeds. `learning_rate_used` records
the rate applied to that update; `learning_rate_next` records the rate after
the scheduler advances. Checkpoint evidence includes `scheduler_last_epoch`
and the scheduler state, which must agree with successful optimizer updates.

## Launch and evidence

The driver entry point is `sure-vl-train-verl`. The current 100-update config
points to the frozen full official 5,985-row training split and disjoint 256-row
validation split. Its source and split hashes are recorded in
[`visionopd_full_v1.json`](evidence/visionopd_full_v1.json). Inspect the config on the server without starting Ray,
then launch it in a fresh output directory:

```bash
uv run --project envs/verl --frozen sure-vl-train-verl --config configs/verl/qwen35_08b_visionopd_100step.json --check-only
CUDA_VISIBLE_DEVICES=0 uv run --project envs/verl --frozen sure-vl-train-verl --config configs/verl/qwen35_08b_visionopd_100step.json
```

The controller refuses a nonempty output directory and has no resume path.
It records source/config/data hashes, vendored veRL Python hashes, runtime
versions, template hash, raw train/validation attempts, optimizer metrics,
and checkpoints. Fixed validation runs before any update and at configured
successful-update intervals. `training_completed.json` is written only after
the requested successful-update count, matching Adam step and EMA count,
final checkpoint, and validation have passed. A launch, a Ray actor start, or
an attempted step alone is not a completed experiment. Even a completed
small-model run requires held-out coverage and calibration results before an
effectiveness claim.

The separate audit environment passed 14 unique CPU tests, including official
`DataProto`/Ray WorkerGroup transport, TinyWorker RPCs, and the grouped
objective checks. See
[`verl_cpu_runtime_v1.json`](evidence/verl_cpu_runtime_v1.json) for the exact
versions and tested file hashes. That audit reused the root environment's
packages through a read-only path; the fully independent
`uv sync --project envs/verl --frozen` has not completed. It loaded no Qwen model and made zero
GPU optimizer updates.

The separate [real Qwen zero-update audit](evidence/verl_qwen_zero_update_v1.json)
then passed through the official Ray WorkerGroup on GPU 0. It sampled four
actual responses with the training length limit of 256, scored the privileged
Teacher four times and the same-input Teacher baseline three times, and obtained
two nonfallback visual proxies. Raw generation versus full-forward sampled-token
log-probabilities differ by at most `1.60e-5` (`1.31e-4` over the full vocabulary),
within the fixed `1e-3` gate. Group advantages match separate reward centering,
Teacher parameters stay frozen, Student gradients are absent, and Adam/EMA/LR
update counters remain zero. The audit discards its pending group before closing.

The real Qwen GPU backward/update path and a full veRL training run have not
been executed. The launch command above remains a recipe for that remaining
runtime gate. The completed 100-update experiment in this repository used TRL.
