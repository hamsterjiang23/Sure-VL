import json
import math
import tempfile
import unittest
from pathlib import Path

from sure_vl.tracking import ExperimentTracker, validate_tracking_config


class FakeRun:
    def __init__(self, *, mode="online", url="https://wandb.ai/team/sure-vl/runs/run-1"):
        self.id = "run-1"
        self.url = url
        self.settings = type("Settings", (), {"mode": mode})()
        self.offline = mode == "offline"
        self.defined = []
        self.logged = []
        self.finished = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, payload, **kwargs):
        self.logged.append((dict(payload), dict(kwargs)))

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class FakeSDK:
    def __init__(self, run=None):
        self.run = run if run is not None else FakeRun()
        self.init_args = []

    def init(self, **kwargs):
        self.init_args.append(kwargs)
        return self.run


def _config():
    return {
        "backend": "wandb", "mode": "online", "project": "sure-vl",
        "entity": "test-team", "run_name": "test-run",
    }


def _evidence(successful=2):
    return {
        "optimizer_attempted_steps": successful + 1,
        "optimizer_successful_updates": successful,
        "optimizer_skipped_updates": 1,
        "teacher_mode": "ema", "teacher_ema_updates": successful,
    }


def _record(*, fallback=False):
    return {
        "id": "same-sample", "answer_correct": not fallback,
        "answer_label_available": not fallback,
        "answer_confidence": None if fallback else 0.8,
        "visual_confidence": None if fallback else 0.9,
        "answer_confidence_score": None if fallback else 8,
        "visual_confidence_score": None if fallback else 9,
        "visual_proxy": 0.0 if fallback else 0.85,
        "proxy_fallback": fallback, "vision_tokens": 0 if fallback else 12,
        "format_errors": ["missing_confidence"] if fallback else [],
        "proxy_components": {} if fallback else {
            "raw_js": 0.12, "baseline_js": 0.02, "corrected_gap": 0.10,
            "teacher_entropy": 0.40,
        },
        "reward": {"total": -2.0 if fallback else 1.0, "format_penalty": int(fallback)},
        "opsd": {"sampled_positions": 0 if fallback else 4,
                 **({} if fallback else {
                     "raw_forward_kl_mean": 0.07,
                     "clipped_vocabulary_fraction": 0.03,
                     "weighted_logit_grad_l2": 0.42,
                 })},
    }


class TrackingTests(unittest.TestCase):
    def test_disabled_backend_is_a_noop_without_output_or_sdk(self):
        self.assertEqual(validate_tracking_config(None), {"backend": "none"})
        with tempfile.TemporaryDirectory() as temporary:
            tracker = ExperimentTracker(None, output_dir=temporary)
            tracker.record_rollout({"malformed": object()})
            tracker.log_training({"loss": math.nan}, {}, 0)
            tracker.log_validation({"answer_brier": None}, {}, 0)
            tracker.log_counters({}, 0)
            tracker.finish()
            self.assertFalse(tracker.enabled)
            self.assertFalse((Path(temporary) / "tracking_run.json").exists())

    def test_tracking_config_rejects_unintentional_offline_and_missing_fields(self):
        for config in (
            {"backend": "wandb", "mode": "offline", "project": "p", "run_name": "r"},
            {"backend": "wandb", "project": "p", "run_name": "r"},
            {"backend": "wandb", "mode": "online", "run_name": "r"},
            {"backend": "wandb", "mode": "online", "project": "p"},
            {"backend": "none", "project": "p"},
            {"backend": "unknown"},
            {"backend": "wandb", "mode": "online", "project": "p", "run_name": "r", "typo": True},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_tracking_config(config)

    def test_online_run_metadata_axes_and_training_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            provenance = {"model": "Qwen3.5-0.8B", "manifest_sha256": "abc123",
                          "method": {"alpha": 0.5}}
            tracker = ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk,
                                        run_config=provenance)
            self.assertTrue(tracker.enabled)
            self.assertEqual(sdk.init_args[0]["mode"], "online")
            self.assertEqual(sdk.init_args[0]["entity"], "test-team")
            self.assertEqual(sdk.init_args[0]["config"], provenance)
            self.assertEqual(tracker.run_url, sdk.run.url)
            metadata = json.loads((Path(temporary) / "tracking_run.json").read_text())
            self.assertEqual(metadata["id"], "run-1")
            self.assertEqual(metadata["url"], sdk.run.url)
            self.assertEqual(sdk.run.defined, [
                (("optimizer/successful_updates",), {}),
                (("train/*",), {"step_metric": "optimizer/successful_updates"}),
                (("validation/*",), {"step_metric": "optimizer/successful_updates"}),
                (("trainer/*",), {"step_metric": "optimizer/successful_updates"}),
            ])
            first = _record()
            tracker.record_rollout(first)
            first["proxy_components"]["raw_js"] = 999  # buffered copy is stable
            tracker.record_rollout(_record(fallback=True))  # duplicate source ID
            self.assertEqual(tracker.pending_rollout_count, 2)
            tracker.log_training({
                "loss": 1.25, "grad_norm": math.nan,
                "optimizer_state_max_step": 2, "optimizer_successful_updates": 999,
                "sure_vl/rank_local/policy_loss": 0.5,
                "sure_vl/rank_local/visual_proxy_mean": 0.0,
            }, _evidence(), 3)
            payload, options = sdk.run.logged[0]
            self.assertEqual(options, {})  # never SDK step=, even when callbacks share X
            self.assertEqual(payload["optimizer/successful_updates"], 2)
            self.assertEqual(payload["trainer/state_global_step"], 3)
            self.assertEqual(payload["optimizer/skipped_updates"], 1)
            self.assertEqual(payload["train/attempt_count"], 2)
            self.assertEqual(payload["train/visual_proxy_eligible_count"], 1)
            self.assertEqual(payload["train/visual_proxy_fallback_count"], 1)
            self.assertEqual(payload["train/visual_proxy_stats/mean"], 0.85)
            self.assertEqual(payload["train/visual_proxy_stats/variance"], 0.0)
            self.assertEqual(payload["train/proxy_components/raw_js/mean"], 0.12)
            self.assertEqual(payload["train/proxy_components/raw_js/sample_count"], 1)
            self.assertEqual(payload["train/reward_components/total/sample_count"], 2)
            self.assertEqual(payload["train/opsd_diagnostic_rows"], 1)
            self.assertEqual(payload["train/opsd_diagnostic_positions"], 4)
            self.assertEqual(payload["train/opsd_diagnostic_coverage"], 0.5)
            self.assertEqual(payload["train/opsd_components/clipped_vocabulary_fraction/mean"], 0.03)
            self.assertEqual(payload["train/opsd_components/weighted_logit_grad_l2/mean"], 0.42)
            self.assertEqual(payload["train/visual_confidence_score/mean"], 9)
            self.assertEqual(payload["train/visual_confidence_score/valid_count"], 1)
            self.assertEqual(payload["train/visual_confidence_score/coverage"], 0.5)
            self.assertEqual(payload["train/visual_confidence_score/count_9"], 1)
            self.assertEqual(payload["train/visual_confidence_score/count_8"], 0)
            self.assertEqual(payload["train/answer_confidence_score/mean"], 8)
            self.assertEqual(payload["train/answer_confidence_score/count_8"], 1)
            self.assertEqual(payload["train/format_error_missing_confidence_count"], 1)
            self.assertEqual(payload["trainer/loss"], 1.25)
            self.assertEqual(payload["trainer/optimizer_state_max_step"], 2)
            self.assertNotIn("trainer/optimizer_successful_updates", payload)
            self.assertEqual(payload["trainer/rank_local_policy_loss"], 0.5)
            self.assertNotIn("trainer/grad_norm", payload)
            self.assertNotIn("trainer/rank_local_visual_proxy_mean", payload)
            self.assertNotIn("train/answer_risk_coverage", payload)  # array stays in JSONL
            self.assertNotIn("train/visual_proxy_answer_relation/correlation", payload)
            tracker.log_training({}, _evidence(), 3)
            self.assertEqual(tracker.pending_rollout_count, 0)
            self.assertNotIn("train/attempt_count", sdk.run.logged[-1][0])
            tracker.finish()
            self.assertEqual(sdk.run.finished, [0])

    def test_validation_keeps_missing_metrics_missing_at_same_optimizer_x(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            tracker = ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk)
            report = {
                "sample_count": 8, "answer_confidence_count": 0,
                "answer_brier": None, "answer_ece10": math.nan,
                "visual_proxy_eligible_count": 0,
                "visual_proxy_stats": {"sample_count": 0, "mean": None, "variance": None},
                "output_coverage": {"format_clean_rate": 0.25, "format_clean_count": 2},
                "generation": {"sample_count": 8, "hit_max_new_tokens_count": 3,
                               "max_new_tokens": 256},
                "subset_ids": ["private-id"], "optimizer_step": 2,
                "optimizer_evidence": {"optimizer_successful_updates": 999},
            }
            tracker.log_validation(report, _evidence(), 3)
            payload, options = sdk.run.logged[-1]
            self.assertEqual(options, {})
            self.assertEqual(payload["optimizer/successful_updates"], 2)
            self.assertEqual(payload["validation/sample_count"], 8)
            self.assertEqual(payload["validation/visual_proxy_stats/sample_count"], 0)
            self.assertEqual(payload["validation/generation/hit_max_new_tokens_count"], 3)
            self.assertNotIn("validation/answer_brier", payload)
            self.assertNotIn("validation/answer_ece10", payload)
            self.assertNotIn("validation/visual_proxy_stats/mean", payload)
            self.assertNotIn("validation/optimizer_evidence/optimizer_successful_updates", payload)
            self.assertNotIn("validation/generation/max_new_tokens", payload)
            tracker.log_counters(_evidence(), 3)
            self.assertEqual(sdk.run.logged[-1][0]["optimizer/successful_updates"], 2)
            tracker.finish()

    def test_offline_or_missing_url_fails_closed_without_metadata(self):
        for run in (FakeRun(mode="offline"), FakeRun(url=None)):
            with tempfile.TemporaryDirectory() as temporary:
                with self.subTest(run=run), self.assertRaisesRegex(RuntimeError, "no upload|no usable"):
                    ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=FakeSDK(run))
                self.assertEqual(run.finished, [1])
                self.assertFalse((Path(temporary) / "tracking_run.json").exists())

    def test_missing_optimizer_evidence_and_unflushed_success_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            tracker = ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk)
            with self.assertRaisesRegex(ValueError, "successful-update"):
                tracker.log_counters({}, 1)
            self.assertEqual(sdk.run.logged, [])
            tracker.record_rollout(_record())
            with self.assertRaisesRegex(RuntimeError, "unflushed"):
                tracker.finish()
            tracker.log_training({}, _evidence(), 3)
            tracker.finish(exit_code=0)
            with self.assertRaisesRegex(RuntimeError, "already finished"):
                tracker.record_rollout(_record())

    def test_startup_config_rejects_nonfinite_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk,
                                  run_config={"source": {"score": math.nan}})
            self.assertEqual(sdk.init_args, [])

    def test_invalid_raw_report_score_is_not_uploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            tracker = ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk)
            record = _record()
            record["visual_confidence_score"] = 11
            tracker.record_rollout(record)
            with self.assertRaisesRegex(ValueError, "visual_confidence_score"):
                tracker.log_training({}, _evidence(), 3)
            self.assertEqual(sdk.run.logged, [])
            tracker.finish(exit_code=1)

    def test_all_missing_training_reports_keep_means_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = FakeSDK()
            tracker = ExperimentTracker(_config(), output_dir=temporary, wandb_sdk=sdk)
            tracker.record_rollout(_record(fallback=True))
            tracker.log_training({}, _evidence(0), 0)
            payload = sdk.run.logged[-1][0]
            self.assertEqual(payload["train/visual_proxy_eligible_count"], 0)
            self.assertEqual(payload["train/visual_proxy_stats/sample_count"], 0)
            self.assertNotIn("train/visual_proxy_stats/mean", payload)
            self.assertEqual(payload["train/visual_confidence_score/valid_count"], 0)
            self.assertEqual(payload["train/visual_confidence_score/coverage"], 0)
            self.assertNotIn("train/visual_confidence_score/mean", payload)
            self.assertEqual(payload["train/visual_confidence_score/count_0"], 0)
            self.assertNotIn("train/answer_brier", payload)
            tracker.finish()


if __name__ == "__main__":
    unittest.main()
