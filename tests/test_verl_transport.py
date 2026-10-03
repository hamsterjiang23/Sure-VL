"""Opt-in CPU Ray RPC smoke for the vendored veRL transport primitives.

Run with ``SURE_VL_RUN_RAY_TRANSPORT=1`` in the isolated veRL environment.
This does not construct or update a model.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


VENDORED_VERL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "verl"


@unittest.skipUnless(
    os.environ.get("SURE_VL_RUN_RAY_TRANSPORT") == "1",
    "set SURE_VL_RUN_RAY_TRANSPORT=1 in the isolated veRL environment",
)
class VendoredVerlTransportTests(unittest.TestCase):
    def test_dataproto_round_trip_through_one_cpu_ray_worker(self) -> None:
        # The Ray worker is a separate interpreter. Give it the same checked-in
        # veRL package path as the test driver, without installing the vendor tree.
        vendor_path = str(VENDORED_VERL_ROOT)
        sys.path.insert(0, vendor_path)
        try:
            import numpy as np
            import ray
            import torch
            from verl.protocol import DataProto
            from verl.single_controller.base import Worker
            from verl.single_controller.base.decorator import Dispatch, register
            from verl.single_controller.ray.base import (
                RayClassWithInitArgs,
                RayResourcePool,
                RayWorkerGroup,
            )

            @ray.remote(num_cpus=1)
            class EchoWorker(Worker):
                @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
                def increment(self, batch: DataProto) -> DataProto:
                    assert self.world_size == 1
                    return DataProto.from_dict(
                        tensors={"input_ids": batch.batch["input_ids"] + 1},
                        non_tensors={"sample_id": batch.non_tensor_batch["sample_id"]},
                        meta_info={"worker_rank": self.rank},
                    )

            python_path = os.pathsep.join(filter(None, (vendor_path, os.environ.get("PYTHONPATH", ""))))
            ray.init(
                num_cpus=2,
                include_dashboard=False,
                log_to_driver=False,
                runtime_env={"env_vars": {"PYTHONPATH": python_path}},
            )
            try:
                pool = RayResourcePool(process_on_nodes=[1], use_gpu=False, max_colocate_count=1)
                group = RayWorkerGroup(
                    resource_pool=pool,
                    ray_cls_with_init=RayClassWithInitArgs(EchoWorker),
                    device_name="cpu",
                    worker_env={"PYTHONPATH": python_path},
                )
                payload = DataProto.from_dict(
                    tensors={"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)},
                    non_tensors={"sample_id": np.array(["first", "second"], dtype=object)},
                    meta_info={"source": "cpu-transport-smoke"},
                )
                returned = group.increment(payload)

                self.assertIsInstance(returned, DataProto)
                self.assertEqual(group.world_size, 1)
                self.assertEqual(len(returned), 2)
                self.assertEqual(returned.batch["input_ids"].tolist(), [[2, 3], [4, 5]])
                self.assertEqual(returned.non_tensor_batch["sample_id"].tolist(), ["first", "second"])
                self.assertEqual(returned.meta_info["worker_rank"], 0)
            finally:
                ray.shutdown()
        finally:
            sys.path.remove(vendor_path)


if __name__ == "__main__":
    unittest.main()
