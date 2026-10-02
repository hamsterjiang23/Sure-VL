# Internal visual proxy: 100-update pilot readback

Audited 2026-10-02. This is a pipeline and failure-mode readback for the frozen 16-train/8-validation VL-Calibration pilot, not an effectiveness result. The complete compact counts and hashes are in [the JSON evidence](evidence/proxy_100step_summary.json); original JSONL attempts and checkpoints remain on `v100-2-hamster` under `/data/LHJ/Sure-VL/outputs/qwen35-08b-proxy-100step-fp32-score10-v6`.

## Run identity and completion

- Clean source commit `7d8e2948e1abc9a7c7ad2cf0880dbca60f0010ce`; config SHA-256 `12108a7daaf63c221e98d3c153547570075f556c01c4fcc1b2ea6adc3e261062`. Frozen train and validation manifests contain 16 and 8 different examples. The same eight validation IDs and subset hash `1770f3bfea7585ca06df18e8fe4915ccb52d54629720c304cd3bf1d0f6c0821e` were used at every readback.
- Qwen3.5-0.8B on physical GPU 0, Tesla V100S 32 GB, using uv, FP32 Student and EMA Teacher. The config requested 100 steps at learning rate `1e-6`, one generated sample per update, 256-token cap, and 0–10 integer reports normalized by 10. Proxy settings were `alpha=0.5`, `tau_s=0.5`, `lambda_b=1`, with at least eight vision tokens required.
- `training_completed.json` records **100 attempted steps, 100 successful optimizer updates, zero skipped updates, and 100 EMA updates**. The final Adam state step is 100. All six validation records agree with the corresponding successful-update, EMA, and Adam step counts. The completed marker lists exactly `0,20,40,60,80,100`; model and Teacher checkpoints were saved remotely.

## Fixed validation

Every row below has eight attempted examples. `Labels` counts extractable answers; accuracy uses all eight attempts, so an unextractable answer fails that denominator. `S` counts *nonfallback* internal visual proxy values. A report count only means the integer field parsed, not that its target was available.

| Actual updates | Correct / 8 | Labels / 8 | S / 8 | Visual / answer reports | Both reports | Clean format | 256-token cap | Valid OPSD diagnostics |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 4 | 3 | 0 / 1 | 0 | 0 | 0 | 8 |
| 20 | 0 | 0 | 0 | 8 / 8 | 8 | 0 | 0 | 3 |
| 40 | 0 | 0 | 0 | 4 / 4 | 4 | 0 | 4 | 8 |
| 60 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 6 | 8 |
| 80 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 5 | 8 |
| 100 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 3 | 8 |

At step 0, the three nonfallback S values have mean `0.919984`, population variance `0.000744`, p05 `0.887078`, median `0.937711`, and p95 `0.940482`. Their mean normalized raw JS is `0.01000420`, same-view baseline JS `0.0000000122`, corrected gap `0.01000418`, and clear-Teacher normalized entropy `0.0738403`. These quantities are undefined at steps 20–100 because **no** validation output has a usable vision span. The step-0 S–answer relation has two labeled, nonfallback examples, both correct; a correlation or correct-versus-incorrect gap is therefore undefined. Later relation denominators are zero.

No checkpoint has a valid visual-report/proxy pair, so visual-proxy MSE and binned report error remain undefined. No checkpoint has both an extractable answer label and a valid answer report, so answer Brier and ECE10 also remain undefined. A reported zero here would be misleading. These metrics concern an internal Teacher proxy, not factual visual correctness.

The raw validation outputs show a format failure rather than a calibrated predictor. At step 20, all eight samples emitted both confidence fields, but none provided an extractable answer or usable vision span. By step 40, four outputs reached the generation cap; by step 100, all eight still lacked a usable vision span and answer, and three reached the cap. The all-attempt accuracy moved from 2/8 at step 0 to 0/8 at each later checkpoint. This tiny pilot does not isolate why the model shifted behavior.

## Training-attempt coverage

The 100 logged attempts have eight nonfallback S values, all before update steps 1–10. Their S mean is `0.897484`, population variance `0.000991`; mean raw JS `0.0365442`, baseline JS `0.0244274`, corrected gap `0.0148035`, and Teacher entropy `0.0939756` (all denominators **8**). Only two attempts have extractable answer labels, and only one overlaps a nonfallback S; this cannot estimate an S–Y relationship. Both confidence reports parsed in 16/100 attempts, no attempt had clean canonical format, and 94/100 attempts had positive content-token OPSD counts. Parsed raw confidence scores were integers within 0–10. These counts show computation and output coverage, not a useful calibration signal.

## Method boundary

The input pair in this run is the original VL-Calibration image for the Teacher and a downsampled then upsampled version for the Student. The run has **no question crop or external evidence E**, and does not implement the proposed new Vision-OPD image/evidence enhancement. It should be described as an original-versus-blurred paired-view pilot. The small model and eight held-out examples cannot support a general effectiveness claim, especially given the vanished valid S/report denominators.

For the next `I+ / E+` design, the same-view baseline `q−` should use the EMA Teacher with the Student's exact prompt/token IDs, restricted image, no E, and the same generated content prefix. This estimates Student-versus-EMA/model-lag disagreement under the same input. A privileged `q+` can use a dedicated Teacher template, crop, and evidence, but its raw JS then reflects the **joint** change in image, evidence, and template. Subtracting the `q−` JS is an explicit heuristic, not an exact decomposition of JS divergence and not proof that the remainder is only visual information. The next run should log each prompt/template hash and token length, image transform/crop identity, E presence and source, aligned content-prefix length and vision positions, raw and baseline JS, corrected JS, Teacher entropy, and valid denominators. A same-template paired-view ablation is needed to separate template changes from the additional visual condition.
