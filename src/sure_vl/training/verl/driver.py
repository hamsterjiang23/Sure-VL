"""Sure-VL's native veRL Ray WorkerGroup controller.

The first implementation uses one complete FP32 Student and EMA Teacher in
one GPU worker. Rollout, detached proxy scoring, and the joint actor update
are separate DataProto RPCs. It does not use a TRL Trainer or veRL PPO.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from ...proxy_metrics import evaluate_proxy_records
from ...tracking import ExperimentTracker
from ...train_proxy import TrainingPlan, _sha256, _source_provenance, load_plan

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/verl/qwen35_08b_visionopd_100step.json"


def load_verl_plan(config_path: str | Path = DEFAULT_CONFIG) -> TrainingPlan:
    plan = load_plan(config_path)
    config = plan.config
    if config.get("backend") != "verl_single_gpu":
        raise ValueError("veRL controller requires backend='verl_single_gpu'")
    setting = config["setting"]
    for key in ("target_world_size", "per_device_train_batch_size"):
        if setting[key] != 1:
            raise ValueError(f"veRL group prototype requires setting.{key}=1")
    group_size = setting["num_generations"]
    if (group_size < 2 or setting["gradient_accumulation_steps"] != group_size
            or setting["generation_batch_size"] != group_size):
        raise ValueError("veRL group prototype requires num_generations=accumulation=generation_batch_size>=2")
    if setting["fp16"] or setting["bf16"]:
        raise ValueError("veRL V100 implementation requires FP32")
    if setting["max_steps"] < 1:
        raise ValueError("veRL controller requires a positive max_steps budget")
    if config.get("grpo") != {"loss_type": "grpo", "scale_rewards": "none", "num_iterations": 1}:
        raise ValueError("veRL prototype requires one-iteration GRPO with unscaled group centering")
    runtime = config.get("verl")
    if not isinstance(runtime, dict) or set(runtime) != {"ray_num_cpus", "object_store_memory_mb"}:
        raise ValueError("verl requires ray_num_cpus and object_store_memory_mb")
    for key, minimum in (("ray_num_cpus", 2), ("object_store_memory_mb", 128)):
        if type(runtime[key]) is not int or runtime[key] < minimum:
            raise ValueError(f"verl.{key} must be an integer >= {minimum}")
    if plan.output_dir.exists() and any(plan.output_dir.iterdir()):
        raise RuntimeError("cannot resume or reuse a nonempty veRL run directory")
    return plan


def _one_worker(results: Any) -> dict[str, Any]:
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise RuntimeError("veRL ONE_TO_ALL must return one worker result")
    return results[0]


def _record(output: Any) -> dict[str, Any]:
    entries = output.non_tensor_batch.get("record_json")
    if entries is None or len(entries) != 1:
        raise RuntimeError("veRL RPC must return one raw assessment record")
    raw = entries[0]
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise RuntimeError("assessment record must be an object")
    return value


def _group_records(output: Any, expected_size: int) -> list[dict[str, Any]]:
    entries = output.non_tensor_batch.get("group_records_json")
    if entries is None or len(entries) != 1:
        raise RuntimeError("veRL update must return one grouped assessment record")
    records = json.loads(entries[0])
    if (not isinstance(records, list) or len(records) != expected_size
            or not all(isinstance(item, dict) for item in records)):
        raise RuntimeError("veRL update returned an incomplete prompt group")
    return records


def _assert_evidence(evidence: Mapping[str, Any], step: int, teacher_mode: str) -> None:
    for key in ("optimizer_attempted_steps", "optimizer_successful_updates",
                "scheduler_last_epoch", "optimizer_state_max_step"):
        if type(evidence.get(key)) is not int or evidence[key] != step:
            raise RuntimeError(f"veRL {key} does not match successful step {step}")
    if evidence.get("optimizer_skipped_updates") != 0:
        raise RuntimeError("veRL run contains skipped optimizer updates")
    if teacher_mode == "ema" and evidence.get("teacher_ema_updates") != step:
        raise RuntimeError("veRL EMA count does not match successful optimizer updates")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _validation_subset(plan: TrainingPlan) -> tuple[list[dict[str, Any]], str]:
    # Same ID-based selection and payload hash used by the TRL validation.
    from ...proxy_evaluation import _subset_sha256, select_proxy_subset
    validation = plan.config["validation"]
    rows = list(select_proxy_subset(plan.validation_rows, size=validation["subset_size"],
                                    seed=validation["subset_seed"]))
    return rows, _subset_sha256(rows)


def _request(row: Mapping[str, Any], generation: Mapping[str, Any]) -> Any:
    import numpy as np
    from verl import DataProto
    return DataProto.from_dict(non_tensors={
        "example_payload": np.array([row["example_payload"]], dtype=object),
    }, meta_info=dict(generation))


def _validate(group: Any, plan: TrainingPlan, rows: list[dict[str, Any]],
              subset_sha256: str, step: int, evidence: dict[str, Any],
              tracker: ExperimentTracker) -> dict[str, Any]:
    from ...proxy_evaluation import _generation_seed
    validation = plan.config["validation"]
    attempts = []
    for row in rows:
        output = group.evaluate(_request(row, {
            "max_new_tokens": validation["max_new_tokens"],
            "temperature": validation["temperature"], "top_p": validation["top_p"],
            "seed": _generation_seed(validation["subset_seed"], row["example_id"]),
        }))
        attempts.append(_record(output))
    report = evaluate_proxy_records(attempts)
    report.update({
        "optimizer_step": step, "actual_successful_optimizer_updates": step,
        "subset_ids": [row["example_id"] for row in rows],
        "subset_sha256": subset_sha256, "optimizer_evidence": evidence,
        "generation": {"sample_count": len(attempts), **validation},
    })
    attempts_path = plan.output_dir / f"proxy_validation_attempts_step_{step:06d}.jsonl"
    with attempts_path.open("x", encoding="utf-8") as destination:
        for attempt in attempts:
            destination.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False) + "\n")
    _append_jsonl(plan.output_dir / "proxy_validation_metrics.jsonl", report)
    tracker.log_validation(report, evidence, step)
    print(json.dumps({"event": "validation", "step": step,
                      "answer_accuracy": report["answer_accuracy"],
                      "visual_proxy_pair_count": report["visual_proxy_pair_count"],
                      "answer_confidence_count": report["answer_confidence_count"]}), flush=True)
    return report


def run_verl_training(plan: TrainingPlan) -> None:
    # Enforce these guards before importing ML packages or starting Ray.
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("run the veRL driver directly; WORLD_SIZE must be 1")
    if plan.output_dir.exists() and any(plan.output_dir.iterdir()):
        raise RuntimeError("cannot resume or reuse a nonempty veRL run directory")
    try:
        import ray
        import verl
        from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from .worker import SureVLProxyWorker
    except ImportError as error:
        raise RuntimeError("veRL requires its isolated uv environment: uv run --project envs/verl --frozen") from error
    source_root = Path(__file__).resolve().parents[4]
    expected_verl = (source_root / "third_party/verl/verl").resolve()
    if Path(verl.__file__).resolve().parent != expected_verl:
        raise RuntimeError(f"wrong veRL implementation imported; expected {expected_verl}")
    if ray.is_initialized():
        raise RuntimeError("veRL driver requires its own local Ray instance")
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    setting = plan.config["setting"]
    runtime = plan.config["verl"]
    record = plan.summary()
    record.update({"backend": "verl_single_gpu", "enable_thinking": False,
                   "confidence_report": {"minimum": 0, "maximum": 10, "integer": True,
                                         "normalization_divisor": 10},
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "framework_boundary": "prototype native Ray WorkerGroup/DataProto with official veRL GRPO advantage and policy loss; world_size=1, FP32 full models",
                   "num_generations_per_prompt": setting["num_generations"],
                   "teacher_conditioning": {
                       "privileged": "separate teacher template, paired teacher image, optional evidence",
                       "baseline": "same EMA, exact student prompt IDs and image, no evidence",
                       "gap_scope": "joint image/evidence/template/model difference; heuristic JS correction"}})
    record["source_commit"], record["source_worktree_dirty"] = _source_provenance()
    source_package = source_root / "src/sure_vl"
    record["source_files_sha256"] = {
        str(path.relative_to(source_package)): _sha256(path)
        for path in sorted(source_package.rglob("*.py"))
    }
    record["vendored_verl_files_sha256"] = {
        str(path.relative_to(expected_verl)): _sha256(path)
        for path in sorted(expected_verl.rglob("*.py"))
    }
    record["runtime_versions"] = {name: metadata.version(name) for name in
                                  ("torch", "torchvision", "transformers", "ray", "tensordict", "verl")}
    tracker = None
    group = None
    ray_started = False
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sure-vl-ray-") as ray_temp:
        try:
            ray.init(num_cpus=runtime["ray_num_cpus"], num_gpus=1, include_dashboard=False,
                     object_store_memory=runtime["object_store_memory_mb"] * 1024 * 1024,
                     _temp_dir=ray_temp, log_to_driver=True)
            ray_started = True
            pool = RayResourcePool(process_on_nodes=[1], use_gpu=True, max_colocate_count=1)
            remote_class = RayClassWithInitArgs(ray.remote(SureVLProxyWorker), config=plan.config)
            group = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=remote_class, device_name="cuda")
            startup = _one_worker(group.initialize())
            record["worker_startup"] = startup
            tracker = ExperimentTracker(plan.config.get("tracking"), output_dir=plan.output_dir,
                                        run_config=record)
            record["tracking_run_url"] = tracker.run_url
            with (plan.output_dir / "run_manifest.json").open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            rows, subset_hash = _validation_subset(plan)
            evidence = startup["optimizer_evidence"]
            _assert_evidence(evidence, 0, plan.config["teacher"]["mode"])
            validation_steps = [0]
            _validate(group, plan, rows, subset_hash, 0, evidence, tracker)
            order = list(range(len(plan.train_rows)))
            rng = random.Random(setting["seed"])
            epoch_order: list[int] = []
            for step in range(1, setting["max_steps"] + 1):
                if not epoch_order:
                    epoch_order = order.copy()
                    rng.shuffle(epoch_order)
                row = plan.train_rows[epoch_order.pop()]
                rollout = group.generate_sequences(_request(row, {
                    "max_new_tokens": setting["max_completion_length"],
                    "temperature": 1.0, "top_p": 1.0,
                    "seed": setting["seed"] + step,
                }))
                scored = group.score_sequences(rollout)
                result = group.update_actor(scored)
                evidence = result.meta_info["optimizer_evidence"]
                metrics = result.meta_info["metrics"]
                attempts = _group_records(result, setting["num_generations"])
                for attempt in attempts:
                    _append_jsonl(plan.output_dir / "proxy_train_attempts_rank_0.jsonl",
                                  {**attempt, "attempted_optimizer_step": step})
                    tracker.record_rollout(attempt)
                _assert_evidence(evidence, step, plan.config["teacher"]["mode"])
                tracker.log_counters(evidence, step)
                _append_jsonl(plan.output_dir / "optimizer_metrics.jsonl",
                              {"optimizer_step": step, **metrics, **evidence})
                if step % setting["logging_steps"] == 0 or step == setting["max_steps"]:
                    tracker.log_training(metrics, evidence, step)
                    print(json.dumps({"event": "update", "step": step, **metrics, **evidence}), flush=True)
                if step % plan.config["validation"]["every_n_steps"] == 0 or step == setting["max_steps"]:
                    validation_steps.append(step)
                    _validate(group, plan, rows, subset_hash, step, evidence, tracker)
                if step % setting["save_steps"] == 0 and step != setting["max_steps"]:
                    _one_worker(group.save_checkpoint(str(plan.output_dir / f"checkpoint-{step}")))
            checkpoint = _one_worker(group.save_checkpoint(str(plan.output_dir / "checkpoint-final")))
            _assert_evidence(checkpoint["optimizer_evidence"], setting["max_steps"], plan.config["teacher"]["mode"])
            if checkpoint["optimizer_state_max_step"] != setting["max_steps"]:
                raise RuntimeError("saved Adam step does not match successful updates")
            completed = {**record, "status": "completed", "optimizer_steps": setting["max_steps"],
                         "optimizer_evidence": evidence, "checkpoint": checkpoint,
                         "validation_optimizer_steps": validation_steps,
                         "validation_subset_sha256": subset_hash,
                         "runtime_seconds": time.monotonic() - started}
            (plan.output_dir / "training_completed.json").write_text(
                json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tracker.finish(exit_code=0)
        except BaseException as error:
            (plan.output_dir / "training_failed.json").write_text(json.dumps({
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "runtime_seconds": time.monotonic() - started,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if tracker is not None:
                with contextlib.suppress(Exception):
                    tracker.finish(exit_code=1)
            raise
        finally:
            if group is not None:
                with contextlib.suppress(Exception):
                    group.close()
                for handle in group.workers:
                    with contextlib.suppress(Exception):
                        ray.kill(handle, no_restart=True)
            if ray_started:
                ray.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sure-VL single-GPU veRL Ray Worker trainer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = load_verl_plan(args.config)
        print(json.dumps(plan.summary(), ensure_ascii=False, indent=2), flush=True)
        if not args.check_only:
            run_verl_training(plan)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
