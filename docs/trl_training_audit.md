# TRL training correction audit

## Confirmed failure in the previous run

The official-data GOLD run [9rwk29y0](https://wandb.ai/jiangcangshu0-nanjing-university/Sure-VL/runs/9rwk29y0), source `7f6ed86fa922b83fc79573d3683afa15ba887412`, was stopped after 80 recorded attempts. It did not complete 100 updates. The [abort artifact](evidence/visionopd_gold_aborted_v1.json) preserves the reason and counters.

TRL GOLD's generation input filtering removed `mm_token_type_ids`. Transformers Qwen3.5 then used the text-position fallback during generation; the loss forward received multimodal types and used M-RoPE. The sampled trajectory and its scored policy were therefore different. This was an implementation error, not merely a chart-scale issue. The legacy adapter has a regression fix, but the active trainer now uses native TRL GRPO.

The previous raw negative-reward score-function sum also scaled with completion length. At cloud step 45, the loss was about −632 and the pre-clip gradient norm about 25,986; the native clipping threshold was 1. A real Adam update alone did not establish a useful training signal. Many outputs had the same −3 reward and unusable confidence structure.

## Native sampling parity

The [GRPO parity artifact](evidence/grpo_sampling_audit_v1.json) records a fresh official Qwen3.5-0.8B model, official paired-data row, native `GRPOTrainer`, and two actual sampled completions.

Generation and scoring had identical prompt IDs, attention mask, image tensors, grid dimensions and multimodal token types. On 32 generated tokens:

| Check | Observed value |
| --- | --- |
| Maximum absolute raw-logit difference | `9.92e-5` |
| Maximum sampled-token log-probability difference | `4.29e-5` |
| Maximum JS divergence, nats | `9.52e-8` |
| Optimizer updates | **0** |

This validates native generation/rescoring alignment. It does not validate the custom Teacher reward, its backward path or training effectiveness.

## Changes to the actual training objective

`training/trl/trainer.py` subclasses the pinned official `GRPOTrainer`. It keeps native generation, accumulation, ratio/clipping, length reduction, optimizer and scheduler. The Sure-VL additions are:

1. Score exact generated IDs under frozen Student/Teacher distributions, with each view's own causal prompt offset.
2. Center content and report rewards separately within each four-response question group; do not scale by standard deviation.
3. Supply per-token segment advantages to native GRPO. Constant reward groups have no policy gradient signal.
4. Add original OPSD content-token forward KL using logits from that same differentiable Student forward, scaled for the same accumulation factor.
5. Update Teacher EMA only after a successful optimizer update.

This changes the optimization estimator and length scaling relative to the raw sum in the derivation. It is not a cosmetic division of displayed loss.

The answer grader now accepts a labeled description only when it exactly matches that option in the question, and logs its noncanonical format separately. An absent answer with known ground truth is a failure event `Y=0`; it does not remove an otherwise valid confidence report from the answer-calibration denominator.

## Interpretation of corrected monitoring

With fresh on-policy ratios equal to one, group-centered policy losses can be numerically near zero while their gradients are nonzero. Read them together with active advantage tokens, reward spread, gradients, OPSD and output coverage. A group with identical content/report rewards has zero policy advantages; OPSD can still supply content gradients.

Native `grad_norm` is measured before clipping. `grad_norm_after_clip` measures the actual gradients entering the optimizer. Losses marked `scaled` include gradient-accumulation division; `unscaled` losses show each microbatch's original contribution. The tracker uploads native GRPO and Sure-VL logs once on the actual successful-update axis.

Runtime evidence must distinguish actual sampled-ID alignment, finite backward gradients, Adam parameter updates, completed 100-update runs, and held-out calibration results. None implies the others.


## Teacher scoring and backward audit

The [actual-model subclass audit](evidence/grpo_teacher_backward_final_prompt_v1.json)
uses the selected Student template, four actual completions and four native
GRPO microbatch backward calls, with no optimizer construction or update.
It records final-template actual samples, four reward-path q-minus forwards, eight
q-plus forwards, complete mask partition, finite Student gradients and zero
Teacher gradients. The largest reward-rescoring versus native sampled-token
log-probability difference is `2.48e-5`. Format coverage remains limited:
this final-template group has four eligible visual spans, two complete dual reports and one clean structure.

The earlier [format probe](evidence/grpo_prompt_probe_v2.json) selected a short system
example plus a repeated tag contract after the user question by output
coverage, not answer accuracy. Before adding the explicit report meanings and description limit, its eight actual samples yielded five dual
reports and four clean structures. This bounded prompt diagnostic is not a
training result or a generalization claim.

The pointwise OPSD vocabulary cap can make its summed modified loss negative:
it clips positive vocabulary contributions while retaining negative terms.
The uncapped forward KL and the clipped objective are logged separately;
a small negative clipped value is not a negative raw KL or a nonfinite loss.

## Two-GPU runtime attempts

Both V100s passed individual allocation and synchronization checks. The full Qwen DDP model failed during NCCL parameter broadcast even with peer/shared-memory transports disabled. A single Student process with a separately placed Teacher loaded both models, but actual generation encountered a CUDA launch timeout. These bounded attempts made zero optimizer updates. The active 100-update recipe consequently exposes only GPU 0. CPU Gloo tests cover the new distributed callback aggregation; they do not establish working GPU DDP.

The earlier corrected single-GPU launch was interrupted at zero updates to investigate the requested two-GPU setup. Its output is preserved; the fallback uses a fresh `single-100step-wandb-v2` output directory.
