"""Trainer events for explicitly configured cloud experiment tracking."""
from __future__ import annotations

import math
from collections import Counter
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

    def _rank_world(self) -> tuple[int, int]:
        accelerator = self.trainer.accelerator
        rank = int(getattr(accelerator, "process_index", 0))
        world = int(getattr(accelerator, "num_processes", 1))
        if world < 1 or not 0 <= rank < world:
            raise RuntimeError("invalid tracking rank/world size")
        if world > 1:
            import torch.distributed as distributed
            if (not distributed.is_available() or not distributed.is_initialized()
                    or distributed.get_rank() != rank or distributed.get_world_size() != world):
                raise RuntimeError("tracking requires the matching initialized process group")
        return rank, world

    def _gather_rollouts(self) -> list[dict[str, Any]]:
        rank, world = self._rank_world()
        buffer = getattr(self.trainer, "_global_tracking_buffer", None)
        if buffer is None:
            if world > 1:
                raise RuntimeError("distributed trainer lacks its rank-local rollout buffer")
            return []
        if not isinstance(buffer, list):
            raise TypeError("rank-local rollout buffer must be a list")
        local = [dict(record) for record in buffer]
        if any(record.get("rank", rank) != rank for record in local):
            raise RuntimeError("rank-local rollout buffer contains another rank's record")
        if world > 1:
            import torch.distributed as distributed
            shards: list[Any] = [None] * world
            distributed.all_gather_object(shards, local)
            if any(not isinstance(shard, list) for shard in shards):
                raise RuntimeError("distributed tracking did not gather all rollout shards")
            records = []
            for source_rank, shard in enumerate(shards):
                for record in shard:
                    if not isinstance(record, dict) or record.get("rank") != source_rank:
                        raise RuntimeError("gathered rollout has an invalid source rank")
                    records.append(record)
        else:
            records = local
        group_size = getattr(self.trainer, "num_generations", None)
        if records and type(group_size) is int and group_size > 0:
            steps = Counter(record.get("trainer_step_before_update") for record in records)
            if any(type(step) is not int or count != group_size for step, count in steps.items()):
                raise RuntimeError("tracking window lacks a complete global prompt group")
        buffer.clear()
        return records if rank == 0 else []

    def _mean_rank_logs(self, logs: dict[str, Any]) -> dict[str, Any]:
        rank, world = self._rank_world()
        if world == 1:
            return logs
        import torch.distributed as distributed
        shards: list[Any] = [None] * world
        distributed.all_gather_object(shards, logs)
        if any(not isinstance(shard, dict) for shard in shards):
            raise RuntimeError("distributed tracking did not gather all Trainer logs")
        merged: dict[str, Any] = {}
        for key in set().union(*(shard.keys() for shard in shards)):
            if not isinstance(key, str):
                continue
            values = [shard.get(key) for shard in shards]
            if all(type(value) in (int, float) and math.isfinite(value) for value in values):
                destination = (key.replace("sure_vl/rank_local/", "sure_vl/global/", 1)
                               if key.startswith("sure_vl/rank_local/") else key)
                merged[destination] = sum(float(value) for value in values) / world
            elif any(type(value) in (int, float) for value in values):
                raise RuntimeError(f"numeric Trainer log differs in coverage across ranks: {key}")
        return merged if rank == 0 else {}

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self._rank_world()[0] == 0:
            self.tracker.log_counters(self._evidence(), int(state.global_step))
        return control

    def on_log(self, args: Any, state: Any, control: Any,
               logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        records = self._gather_rollouts()
        merged_logs = self._mean_rank_logs(logs or {})
        if self._rank_world()[0] == 0:
            for record in records:
                self.tracker.record_rollout(record)
            self.tracker.log_training(merged_logs, self._evidence(), int(state.global_step))
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        records = self._gather_rollouts()
        if self._rank_world()[0] == 0:
            for record in records:
                self.tracker.record_rollout(record)
            if self.tracker.pending_rollout_count:
                self.tracker.log_training({}, self._evidence(), int(state.global_step))
        return control
