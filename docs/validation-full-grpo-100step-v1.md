# Full official-data TRL run: 100 successful updates

> Historical protocol: this run generated `<vision>`, `<answer>`, and `<confidence>` without `<reason>`. Its format coverage and results apply to that earlier protocol. It has not trained or validated the current `vision-reason-answer-confidence-v2` prompt.

The [W&B run](https://wandb.ai/jiangcangshu0-nanjing-university/Sure-VL/runs/ep7oqk7d) finished on the frozen official Vision-OPD split: 5,985 training examples and 256 held-out examples. Source: `b1dbc0177cf50546b6afd0bde08020c7c386c338`. Training used Qwen3.5-0.8B, one V100, FP32, native TRL GRPO, four actual completions per question and 100 successful updates. This budget does not traverse every training example.

## Execution and monitoring

- Adam state, attempted updates, successful updates and Teacher EMA: **100** each; skipped updates: **0**.
- Actual rollout records: **400**, exactly four per update. All checked numbers are finite.
- Native checkpoint includes model, optimizer, scheduler and Trainer state; scheduler epoch and Trainer step are 100. Final Student and Teacher models are present.
- Six assessments use identical ordered held-out IDs at steps 0, 20, 40, 60, 80 and 100.
- W&B history contains all 100 training rows and six assessments. Joint loss ranges from −0.00821 to 0.19995; pre-clip gradient norm from 5.56 to 131.36; actual post-clip norm is approximately 1.

The native Adam-state cloud field was filtered by the running source. Every actual Adam step was retained and verified in local JSONL, with the final verified value written and read back in W&B summary. The tracker fix in `8d95f29` exposes this field in subsequent runs.

## Fixed assessment results

| Metric | Step 0 | Step 100 | Denominator |
| --- | ---: | ---: | --- |
| Answer accuracy | 5/32 | 9/32 | All fixed IDs |
| Both integer confidence reports | 22/32 | 32/32 | All fixed IDs |
| Clean output structure | 17/32 | 32/32 | All fixed IDs |
| Answer Brier | 0.6232 | 0.2355 | Same 22 IDs with valid answer reports at both endpoints |
| Answer accuracy on those IDs | 5/22 | 5/22 | Same 22 IDs |
| Mean answer confidence | 0.7045 | 0.1545 | Same 22 IDs |
| Visual proxy MSE | 0.1492 | 0.00365 | Same 21 IDs with valid reports and nonfallback proxy at both endpoints |

The mean visual proxy on the paired 21 IDs changes from 0.6510 to 0.7344 as the policy and EMA Teacher change. Its MSE measures agreement with that internal target, not factual visual correctness. The answer-confidence improvement on the common 22 IDs accompanies reduced confidence while answer accuracy on those IDs stays unchanged. These are small, single-run process results; they do not establish general calibration or accuracy gains.

The [complete audit](evidence/grpo_full_run_completed_v1.json) preserves exact data/source hashes, checkpoint evidence, cloud ranges, validation coverage, paired denominators and raw-record hashes. The bounded [two-GPU attempts](evidence/dual_gpu_attempts_v1.json) failed before updates; this completed run used one visible GPU.
