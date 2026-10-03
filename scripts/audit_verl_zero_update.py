"""Bounded real-Qwen veRL Ray generate→score audit with zero optimizer updates.

Run only when the selected GPU is idle. This script creates an isolated Ray
worker, samples one four-completion group, scores it, and compares raw HF
generation logits with full-forward logits on the same sampled token IDs.
It never calls update_actor or writes a training completion marker.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--train-row-index", type=int, default=0)
    parser.add_argument("--max-abs-logprob", type=float, default=1e-3)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 256:
        parser.error("max-new-tokens must be between 1 and 256 for this bounded probe")
    if args.output.exists():
        parser.error("output already exists; preserve prior audit artifacts")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("set CUDA_VISIBLE_DEVICES to one explicitly idle GPU")

    import ray
    import torch
    import verl
    from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    from sure_vl.training.verl.driver import _request, load_verl_plan
    from sure_vl.training.verl.worker import SureVLProxyWorker

    plan = load_verl_plan(args.config)
    if args.train_row_index < 0 or args.train_row_index >= len(plan.train_rows):
        parser.error("train-row-index is outside the frozen manifest")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("audit requires exactly one visible CUDA device")
    source_root = Path(__file__).resolve().parents[1]
    expected_vendor = (source_root / "third_party/verl/verl").resolve()
    if Path(verl.__file__).resolve().parent != expected_vendor:
        raise RuntimeError(f"wrong veRL source; expected {expected_vendor}")
    row = plan.train_rows[args.train_row_index]
    seed = int(plan.config["setting"]["seed"])
    request_meta = {"max_new_tokens": args.max_new_tokens,
                    "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": seed}
    python_path = os.pathsep.join(filter(None, (
        str(source_root / "src"), str(source_root / "third_party/verl"),
        os.environ.get("PYTHONPATH", ""),
    )))
    group = None
    ray_started = False
    report: dict[str, object] = {
        "status": "started", "kind": "real_qwen_verl_zero_update_audit",
        "config_sha256": _hash(args.config), "source_root": str(source_root),
        "official_verl_file": str(Path(verl.__file__).resolve()),
        "train_manifest_sha256": _hash(plan.train_manifest),
        "example_id": row["example_id"], "seed": seed,
        "max_new_tokens": args.max_new_tokens,
        "gpu_visible": os.environ["CUDA_VISIBLE_DEVICES"],
        "optimizer_updates": 0,
    }
    try:
        ray.init(num_cpus=plan.config["verl"]["ray_num_cpus"], num_gpus=1,
                 include_dashboard=False, log_to_driver=False,
                 object_store_memory=plan.config["verl"]["object_store_memory_mb"] * 1024 * 1024,
                 runtime_env={"env_vars": {"PYTHONPATH": python_path}})
        ray_started = True
        pool = RayResourcePool(process_on_nodes=[1], use_gpu=True, max_colocate_count=1)
        group = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(ray.remote(SureVLProxyWorker), config=plan.config),
            device_name="cuda", worker_env={"PYTHONPATH": python_path},
        )
        startup = group.initialize()[0]
        rollout = group.generate_sequences(_request(row, request_meta))
        scored = group.score_sequences(rollout)
        audit = group.audit_zero_update({"example_payload": row["example_payload"], **request_meta})[0]
        report.update({
            "startup": startup, "generation_nonce": rollout.meta_info["generation_nonce"],
            "score_nonce": scored.meta_info["score_nonce"],
            "sampled_group_ids": [
                rollout.batch["completion_ids"][index, rollout.batch["completion_mask"][index]].tolist()
                for index in range(len(rollout))
            ],
            "score_content_rewards": scored.batch["content_reward"].tolist(),
            "score_report_rewards": scored.batch["report_reward"].tolist(),
            "score_content_advantages": scored.batch["content_advantage"].tolist(),
            "score_report_advantages": scored.batch["report_advantage"].tolist(),
            "audit": audit,
        })
        if not (audit["content_advantage_matches_centered_rewards"]
                and audit["report_advantage_matches_centered_rewards"]
                and audit["teacher_frozen"] and audit["student_gradients_absent"]
                and audit["privileged_forward_count"] > 0
                and audit["baseline_forward_count"] > 0
                and audit["nonfallback_proxy_count"] > 0
                and audit["optimizer_evidence"]["optimizer_state_max_step"] == 0
                and audit["optimizer_evidence"]["teacher_ema_updates"] == 0
                and audit["generation_vs_forward_max_abs_logprob"] <= args.max_abs_logprob):
            raise RuntimeError("zero-update parity, advantage, or frozen-state gate failed")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if group is not None:
            try:
                report["audit_discard"] = group.abort_zero_update_audit()[0]
            except Exception as error:
                report["audit_discard_error"] = f"{type(error).__name__}: {error}"
            try:
                report["close_evidence"] = group.close()[0]
            except Exception as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
            for handle in group.workers:
                with contextlib.suppress(Exception):
                    ray.kill(handle, no_restart=True)
        if ray_started:
            with contextlib.suppress(Exception):
                ray.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write("\n")
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "optimizer_updates": 0,
                          "max_abs_logprob": (report.get("audit") or {}).get(
                              "generation_vs_forward_max_abs_logprob")},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
