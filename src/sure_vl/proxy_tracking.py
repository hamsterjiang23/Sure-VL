"""Trainer events for explicitly configured cloud experiment tracking."""
from __future__ import annotations

from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object


class ProxyTrackingCallback(TrainerCallback):
    def __init__(self, trainer: Any, tracker: Any) -> None:
        self.trainer, self.tracker = trainer, tracker

    def _evidence(self) -> dict[str, Any]:
        callback = getattr(self.trainer, "optimizer_evidence", None)
        return callback.summary() if callback is not None else {}

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.tracker.log_counters(self._evidence(), int(state.global_step))
        return control

    def on_log(self, args: Any, state: Any, control: Any,
               logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.tracker.log_training(logs or {}, self._evidence(), int(state.global_step))
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.tracker.pending_rollout_count:
            self.tracker.log_training({}, self._evidence(), int(state.global_step))
        return control
