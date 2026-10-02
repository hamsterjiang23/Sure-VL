"""Optional experiment telemetry on the *successful optimizer update* axis.

The JSONL attempt and validation files remain the audit record. This module
publishes finite scalar summaries only; missing metrics retain their explicit
sample counts and are never replaced with zero. W&B's implicit history step is
deliberately separate from the optimizer-update X axis.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .proxy_metrics import evaluate_proxy_records


_REPORT_FIELDS = frozenset({
    "sample_count", "answer_correct_count", "answer_accuracy",
    "answer_label_available_count", "answer_label_unavailable_count",
    "answer_confidence_count", "answer_confidence_coverage", "answer_brier",
    "answer_ece", "answer_ece10", "high_confidence_answer_errors",
    "visual_proxy_eligible_count", "visual_proxy_fallback_count",
    "visual_proxy_pair_count", "visual_proxy_report_coverage",
    "visual_proxy_mse", "visual_proxy_binned_mean_error",
    "visual_proxy_binned_mean_error10", "visual_proxy_report_correlation",
    "visual_proxy_stats", "visual_proxy_answer_relation", "proxy_components",
    "reward_components", "opsd_components", "output_coverage", "vision_tokens",
})
_RANK_LOCAL_LOG_FIELDS = frozenset({
    "policy_loss", "opsd_loss", "content_tokens", "generated_tokens",
})
_EVIDENCE_FIELDS = {
    "optimizer_attempted_steps": "optimizer/attempted_steps",
    "optimizer_successful_updates": "optimizer/successful_updates",
    "optimizer_skipped_updates": "optimizer/skipped_updates",
    "teacher_ema_updates": "optimizer/teacher_ema_updates",
}


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"tracking {name} must be a nonempty string")
    return value.strip()


def validate_tracking_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the explicit online W&B choice; absent config disables tracking."""
    if raw is None:
        return {"backend": "none"}
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError("tracking config must be an object with string keys")
    allowed = {"backend", "mode", "project", "entity", "run_name", "output_dir"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown tracking config fields: {sorted(unknown)}")
    backend = raw.get("backend", "none")
    if backend == "none":
        if set(raw) - {"backend"}:
            raise ValueError("disabled tracking cannot specify W&B settings")
        return {"backend": "none"}
    if backend != "wandb":
        raise ValueError("tracking backend must be 'none' or 'wandb'")
    if raw.get("mode") != "online":
        raise ValueError("W&B tracking requires explicit mode='online'")
    project = _nonempty(raw.get("project"), "project")
    run_name = _nonempty(raw.get("run_name"), "run_name")
    entity = raw.get("entity")
    if entity is not None:
        entity = _nonempty(entity, "entity")
    output_dir = raw.get("output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
            raise ValueError("tracking output_dir must be a nonempty path")
        output_dir = str(output_dir)
    return {
        "backend": "wandb", "mode": "online", "project": project,
        "entity": entity, "run_name": run_name, "output_dir": output_dir,
    }


def _finite_scalar(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _flatten_scalars(prefix: str, value: Any) -> dict[str, int | float]:
    """Flatten dictionaries; arrays, bools, strings, None, and NaNs are omitted."""
    if isinstance(value, Mapping):
        flattened: dict[str, int | float] = {}
        for key, item in value.items():
            if isinstance(key, str) and key:
                flattened.update(_flatten_scalars(f"{prefix}/{key}", item))
        return flattened
    scalar = _finite_scalar(value)
    return {prefix: scalar} if scalar is not None else {}


def _raw_score_metrics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    """Count each parser-valid 0..10 score without imputing missing reports."""
    payload: dict[str, int | float] = {}
    for field in ("visual_confidence_score", "answer_confidence_score"):
        if not any(field in record for record in records):
            continue
        scores: list[int] = []
        for record in records:
            value = record.get(field)
            if value is None:
                continue
            if type(value) is not int or not 0 <= value <= 10:
                raise ValueError(f"{field} must be an integer from 0 to 10 or None")
            scores.append(value)
        prefix = f"train/{field}"
        payload[f"{prefix}/valid_count"] = len(scores)
        payload[f"{prefix}/coverage"] = len(scores) / len(records)
        if scores:
            payload[f"{prefix}/mean"] = sum(scores) / len(scores)
        counts = Counter(scores)
        for score in range(11):
            payload[f"{prefix}/count_{score}"] = counts[score]
    return payload


class ExperimentTracker:
    """Keep a bounded rollout window and publish its aggregate at each log event.

    ``wandb_sdk`` permits a no-network fake in tests. It is imported lazily only
    for an enabled W&B run. The caller should initialize and use this tracker on
    the world-zero process only.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        output_dir: str | Path | None = None,
        wandb_sdk: Any = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = validate_tracking_config(config)
        self.backend = self.config["backend"]
        self.enabled = self.backend == "wandb"
        self.run_url: str | None = None
        self.run_id: str | None = None
        self._run: Any = None
        self._closed = False
        self._buffer: list[dict[str, Any]] = []
        self._last_successful_updates: int | None = None
        if not self.enabled:
            return
        destination = output_dir if output_dir is not None else self.config["output_dir"]
        if destination is None or not str(destination).strip():
            raise ValueError("W&B tracking requires an output_dir")
        self.output_dir = Path(destination).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_dir / "tracking_run.json"
        if metadata_path.exists():
            raise ValueError(f"tracking metadata already exists: {metadata_path}")
        if wandb_sdk is None:
            try:
                import wandb as wandb_sdk
            except ImportError as error:
                raise RuntimeError("W&B tracking requires the optional wandb dependency") from error
        init_kwargs = {
            "project": self.config["project"],
            "name": self.config["run_name"],
            "mode": "online",
            "dir": str(self.output_dir),
        }
        if run_config is not None:
            if not isinstance(run_config, Mapping) or any(not isinstance(key, str) for key in run_config):
                raise ValueError("W&B run_config must be an object with string keys")
            try:
                # Freeze the startup configuration and reject non-JSON or
                # nonfinite values before giving it to the SDK.
                init_kwargs["config"] = json.loads(json.dumps(
                    dict(run_config), ensure_ascii=False, allow_nan=False,
                ))
            except (TypeError, ValueError) as error:
                raise ValueError("W&B run_config must contain finite JSON values") from error
        if self.config["entity"] is not None:
            init_kwargs["entity"] = self.config["entity"]
        run = wandb_sdk.init(**init_kwargs)
        try:
            self._check_online_run(run)
            run.define_metric("optimizer/successful_updates")
            for group in ("train/*", "validation/*", "trainer/*"):
                run.define_metric(group, step_metric="optimizer/successful_updates")
            self.run_id = run.id
            self.run_url = run.url
            metadata = {
                "backend": "wandb", "mode": "online", "project": self.config["project"],
                "entity": self.config["entity"], "run_name": self.config["run_name"],
                "id": self.run_id, "url": self.run_url,
            }
            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
                                 encoding="utf-8")
            temporary.replace(metadata_path)
        except Exception:
            if run is not None:
                run.finish(exit_code=1)
            raise
        self._run = run

    @staticmethod
    def _check_online_run(run: Any) -> None:
        if run is None:
            raise RuntimeError("W&B did not create a run")
        settings = getattr(run, "settings", getattr(run, "_settings", None))
        mode = getattr(settings, "mode", None)
        mode = getattr(mode, "value", mode)
        if mode is not None and str(mode).lower() != "online":
            raise RuntimeError("W&B initialized outside online mode; no upload is claimed")
        if getattr(run, "offline", False) is True:
            raise RuntimeError("W&B initialized offline; no upload is claimed")
        if not isinstance(getattr(run, "id", None), str) or not run.id:
            raise RuntimeError("W&B online run has no ID")
        url = getattr(run, "url", None)
        parsed = urlparse(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("W&B run has no usable online URL; no upload is claimed")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("experiment tracker has already finished")

    @property
    def pending_rollout_count(self) -> int:
        """Number of training attempts awaiting the next summary log."""
        return len(self._buffer)

    def _evidence_payload(
        self, evidence: Mapping[str, Any], state_global_step: int,
    ) -> dict[str, int | float]:
        if not isinstance(evidence, Mapping):
            raise ValueError("optimizer evidence must be a mapping")
        if type(state_global_step) is not int or state_global_step < 0:
            raise ValueError("trainer global step must be a nonnegative integer")
        successful = evidence.get("optimizer_successful_updates")
        if type(successful) is not int or successful < 0:
            raise ValueError("optimizer evidence requires a nonnegative successful-update count")
        if self._last_successful_updates is not None and successful < self._last_successful_updates:
            raise ValueError("successful optimizer updates cannot move backwards")
        payload: dict[str, int | float] = {
            "optimizer/successful_updates": successful,
            "trainer/state_global_step": state_global_step,
        }
        for source, destination in _EVIDENCE_FIELDS.items():
            if source == "optimizer_successful_updates":
                continue
            value = evidence.get(source)
            if value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"optimizer evidence {source} must be a nonnegative integer")
            payload[destination] = value
        self._last_successful_updates = successful
        return payload

    def record_rollout(self, record: Mapping[str, Any]) -> None:
        """Buffer one raw training attempt until the next training-log event."""
        if not self.enabled:
            return
        self._require_open()
        if not isinstance(record, Mapping):
            raise ValueError("rollout record must be a mapping")
        self._buffer.append(copy.deepcopy(dict(record)))

    def _log(self, payload: dict[str, int | float]) -> None:
        self._require_open()
        if self.enabled:
            # No SDK step= argument: validation and Trainer callbacks may both
            # write at the same successful-update count.
            self._run.log(payload)

    def log_training(
        self, logs: Mapping[str, Any] | None, evidence: Mapping[str, Any],
        state_global_step: int,
    ) -> None:
        if not self.enabled:
            return
        payload = self._evidence_payload(evidence, state_global_step)
        if logs is not None:
            if not isinstance(logs, Mapping):
                raise ValueError("trainer logs must be a mapping or None")
            for name, value in logs.items():
                if not isinstance(name, str):
                    continue
                if name.startswith("sure_vl/rank_local/"):
                    local_name = name.removeprefix("sure_vl/rank_local/")
                    if local_name not in _RANK_LOCAL_LOG_FIELDS:
                        continue
                    destination = f"trainer/rank_local_{local_name}"
                elif name.startswith("optimizer_") or name.startswith("teacher_"):
                    continue  # The evidence callback is authoritative here.
                else:
                    destination = f"trainer/{name.replace('/', '_')}"
                payload.update(_flatten_scalars(destination, value))
        if self._buffer:
            # A repeated dataset ID is a distinct on-policy attempt. The metrics
            # helper requires unique IDs, so use a window-local synthetic ID.
            metrics_input = [dict(record, id=f"attempt-{index}")
                             for index, record in enumerate(self._buffer)]
            report = evaluate_proxy_records(metrics_input)
            for field in _REPORT_FIELDS:
                if field in report:
                    payload.update(_flatten_scalars(f"train/{field}", report[field]))
            payload["train/attempt_count"] = len(self._buffer)
            payload.update(_raw_score_metrics(self._buffer))
            positions = [float((record.get("opsd") or {}).get("sampled_positions", 0))
                         for record in self._buffer]
            diagnostic_rows = sum(position > 0 for position in positions)
            payload["train/opsd_diagnostic_rows"] = diagnostic_rows
            payload["train/opsd_diagnostic_positions"] = sum(positions)
            payload["train/opsd_diagnostic_coverage"] = diagnostic_rows / len(self._buffer)
            for error, count in Counter(
                error for record in self._buffer for error in record["format_errors"]
            ).items():
                if error and all(character.isalnum() or character == "_" for character in error):
                    payload[f"train/format_error_{error}_count"] = count
        self._log(payload)
        self._buffer.clear()

    def log_validation(
        self, report: Mapping[str, Any], evidence: Mapping[str, Any],
        state_global_step: int,
    ) -> None:
        if not self.enabled:
            return
        if not isinstance(report, Mapping):
            raise ValueError("validation report must be a mapping")
        payload = self._evidence_payload(evidence, state_global_step)
        for field in _REPORT_FIELDS:
            if field in report:
                payload.update(_flatten_scalars(f"validation/{field}", report[field]))
        generation = report.get("generation")
        if isinstance(generation, Mapping):
            for field in ("sample_count", "hit_max_new_tokens_count"):
                if field in generation:
                    payload.update(_flatten_scalars(f"validation/generation/{field}", generation[field]))
        self._log(payload)

    def log_counters(self, evidence: Mapping[str, Any], state_global_step: int) -> None:
        if not self.enabled:
            return
        self._log(self._evidence_payload(evidence, state_global_step))

    def finish(self, exit_code: int = 0) -> None:
        if type(exit_code) is not int:
            raise ValueError("exit_code must be an integer")
        if not self.enabled or self._closed:
            return
        if exit_code == 0 and self._buffer:
            raise RuntimeError("unflushed rollout records; call log_training before successful finish")
        self._run.finish(exit_code=exit_code)
        self._closed = True
