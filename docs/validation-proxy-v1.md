# Internal visual proxy: runtime validation

Date: 2026-10-02. This record covers pipeline checks on the frozen VL-Calibration 16/8 pilot, not a benchmark result.

## Implemented path

The default TRL entry point uses free-text vision, an answer, and two verbal reports scored as integers 0–10 and divided by 10 for the reward. Qwen's built-in thinking is disabled. On aligned Student prefixes, the clear EMA Teacher and optional restricted-view Teacher define detached normalized JS/entropy certainty. The report targets that internal certainty; it is not factual visual correctness. The answer report targets frozen answer verification. Original OPSD forward KL supervises content and excludes reports, including on malformed trajectories.

The single-V100 recipes use the cached Qwen3.5-0.8B snapshot, FP32 models and EMA masters, gradient checkpointing, a 256-token cap and 65,536-pixel processing budget. Training samples use temperature 1, top-p 1 and top-k 0. FP16 probes produced nonfinite gradients; FP32 is the tested recipe. A pre-optimizer gate rejects nonfinite gradients, and successful updates are counted separately from Trainer steps.

## One-step evidence

- Server: `v100-2-hamster`, `/data/LHJ/Sure-VL`, physical GPU 0 (V100S 32GB).
- Environment: uv lock; Torch 2.6.0+cu124, Transformers 5.18.0, TRL 1.14.1, Accelerate 1.15.0.
- Server tests: 138 tests passed, including the optional tensor and TRL tests, with no skips.
- Final bounded smoke: 1 attempted step, **1 successful optimizer update, 0 skips, 1 EMA update**. Student and Teacher checkpoints were saved.
- Same fixed two-example validation ran at steps 0 and 1. At step 1 one of two visual spans was usable. Both verbal report fields were recoverable as 0–10 integers, but neither final answer was recoverable under the strict tag contract. Format compliance remains limited on this small model.
- The sampled training output had a recoverable correct answer but no usable visual section or dual report: its proxy fell back to zero and it received explicit format/report penalties, while content OPSD remained active. The finite gradient norm was approximately 15,210. These checks do not establish correct visual grounding or improved calibration.

Machine-readable evidence: [completed smoke](evidence/proxy_smoke_completed.json) and [validation readback](evidence/proxy_smoke_validation.jsonl). The smoke ran before the new method's commit: its previous HEAD plus `source_worktree_dirty=true` is deliberately preserved. It is not presented as a run from a clean committed source snapshot.

Earlier diagnostics are preserved on the server. GPU1 first reported a CUDA launch timeout during model placement, while standalone copies and small matmuls passed on both cards; its cause was not established. GPU0 subsequently ran the full smoke. The nonthinking FP16 probe computed visual proxies but skipped its only update after NaN gradients, so no completion marker was accepted.

## 100-update acceptance criteria

The pinned run must produce `training_completed.json` with 100 successful updates, 0 skips, 100 EMA updates, agreement with Adam state step counts, and the identical held-out subset at steps 0/20/40/60/80/100. Report proxy coverage, output format coverage, answer-confidence label counts, proxy/Teacher components and OPSD diagnostics together. Pilot readback cannot support a general effectiveness claim.
