"""Opt-in CPU Gloo check for both TRL ranks' validation and tracking RPCs."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _run_rank(rank: int, directory: str) -> None:
    import torch
    import torch.distributed as distributed
    from PIL import Image

    from sure_vl.proxy_tracking import ProxyTrackingCallback
    from sure_vl.training.trl.validation import (
        GRPOValidationCallback, GradientMonitorCallback, _consistent_evidence,
    )

    root = Path(directory)
    distributed.init_process_group(
        backend="gloo", init_method=f"file://{root / 'process-group'}",
        rank=rank, world_size=2,
    )
    try:
        image = root / "tiny.png"
        rows = []
        for index in range(4):
            example_id = f"heldout-{index}"
            rows.append({
                "example_id": example_id, "split": "validation", "image": str(image),
                "teacher_image": str(image), "prompt": "count the objects",
                "example_payload": json.dumps({"id": example_id, "split": "validation"}),
            })

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = torch.nn.Linear(1, 1)

        class Trainer:
            def __init__(self) -> None:
                self.model = Model()
                self.accelerator = SimpleNamespace(
                    process_index=rank, num_processes=2, device=torch.device("cpu"),
                    unwrap_model=lambda model: model.core,
                )
                self.generation_config = SimpleNamespace(temperature=1.0, top_p=1.0, max_new_tokens=8)
                self.optimizer = SimpleNamespace(state={})
                self.state = SimpleNamespace(global_step=0)
                self._metrics = {"eval": {"old": [1]}, "train": defaultdict(list)}
                self.last_rollout_records = []
                self.num_generations = 4
                self._global_tracking_buffer: list[dict] = []
                self.counters = {
                    "optimizer_attempted_steps": 0, "optimizer_successful_updates": 0,
                    "optimizer_skipped_updates": 0, "teacher_ema_updates": 0,
                }
                self.optimizer_evidence = SimpleNamespace(summary=lambda: dict(self.counters))

            def _generate_and_score_completions(self, prepared):
                if self.model.training or self.model.core.training:
                    raise RuntimeError("validation entered native GRPO in training mode")
                local = torch.tensor([rank], dtype=torch.int64)
                shards = [torch.empty_like(local) for _ in range(2)]
                distributed.all_gather(shards, local)
                if [int(item.item()) for item in shards] != [0, 1]:
                    raise RuntimeError("one DDP rank did not enter native generation")
                self.last_rollout_records = [{"id": prepared[0]["example_id"]}]

        class Tracker:
            def __init__(self) -> None:
                self.validation = []
                self.training = []
                self.counters = []
                self.records = []

            @property
            def pending_rollout_count(self):
                return len(self.records)

            def log_validation(self, report, evidence, step):
                self.validation.append((report, evidence, step))

            def record_rollout(self, record):
                self.records.append(record)

            def log_training(self, logs, evidence, step):
                self.training.append((dict(logs), dict(evidence), step, len(self.records)))
                self.records.clear()

            def log_counters(self, evidence, step):
                self.counters.append((dict(evidence), step))

        trainer = Trainer()
        tracker = Tracker()
        callback = GRPOValidationCallback(
            trainer, rows, root, {
                "subset_size": 4, "subset_seed": 42, "temperature": 0.6,
                "top_p": 0.95, "max_new_tokens": 16, "every_n_steps": 1,
            }, tracker,
        )
        report = {
            "answer_accuracy": 0.5, "output_coverage": {"format_clean_count": 4},
            "visual_proxy_pair_count": 4, "answer_confidence_count": 4,
        }
        with patch("sure_vl.training.trl.validation.evaluate_proxy_records", return_value=report):
            callback.on_train_begin(None, trainer.state, None)
        distributed.barrier()
        assert callback.steps == [0]
        assert trainer.model.training and trainer.model.core.training
        assert len(tracker.validation) == (1 if rank == 0 else 0)
        if rank == 0:
            attempts = [json.loads(line) for line in (root / "proxy_validation_attempts_step_000000.jsonl").read_text().splitlines()]
            assert [item["id"] for item in attempts] == [item["example_id"] for item in callback.rows]
            assert len((root / "proxy_validation_metrics.jsonl").read_text().splitlines()) == 1

        trainer.state.global_step = 1
        trainer.counters.update(optimizer_attempted_steps=1, optimizer_successful_updates=1,
                                teacher_ema_updates=1)
        trainer.optimizer.state = {"parameter": {"step": 1}}
        monitor = GradientMonitorCallback(trainer)
        monitor.on_step_end(SimpleNamespace(output_dir=str(root)), trainer.state, None)
        distributed.barrier()
        assert len((root / f"optimizer_evidence_rank_{rank}.jsonl").read_text().splitlines()) == 1
        if rank == 0:
            assert len((root / "optimizer_evidence.jsonl").read_text().splitlines()) == 1

        # A local counter disagreement must be detected by both ranks before
        # it can be written as one apparently consistent global run.
        if rank == 1:
            trainer.counters["optimizer_skipped_updates"] = 1
        try:
            _consistent_evidence(trainer, trainer.state.global_step)
        except RuntimeError as error:
            assert "counters differ" in str(error)
        else:
            raise AssertionError("cross-rank optimizer evidence disagreement was accepted")
        trainer.counters["optimizer_skipped_updates"] = 0

        trainer._global_tracking_buffer = [
            {"rank": rank, "trainer_step_before_update": 0, "id": f"sample-{rank}-{index}"}
            for index in range(2)
        ]
        tracking = ProxyTrackingCallback(trainer, tracker)
        tracking.on_step_end(None, trainer.state, None)
        tracking.on_log(None, trainer.state, None, logs={
            "loss": 1.0 + rank, "sure_vl/rank_local/opsd_loss": 1.0 + 2 * rank,
        })
        assert not trainer._global_tracking_buffer
        if rank == 0:
            assert len(tracker.counters) == 1
            logs, _, step, attempts = tracker.training[-1]
            assert step == 1 and attempts == 4
            assert logs == {"loss": 1.5, "sure_vl/global/opsd_loss": 2.0}
        else:
            assert not tracker.counters and not tracker.training
        (root / f"rank-{rank}-passed").write_text("ok\n")
    finally:
        distributed.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("SURE_VL_RUN_DDP_CALLBACK_TEST") == "1"
    and all(importlib.util.find_spec(name) for name in ("torch", "transformers", "PIL")),
    "set SURE_VL_RUN_DDP_CALLBACK_TEST=1 in a CPU torch environment",
)
class DDPCallbackTests(unittest.TestCase):
    def test_two_rank_validation_evidence_and_cloud_aggregation(self) -> None:
        import torch.multiprocessing as multiprocessing
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (2, 2), "red").save(root / "tiny.png")
            multiprocessing.spawn(_run_rank, args=(temporary,), nprocs=2, join=True)
            self.assertEqual((root / "rank-0-passed").read_text(), "ok\n")
            self.assertEqual((root / "rank-1-passed").read_text(), "ok\n")


if __name__ == "__main__":
    unittest.main()
