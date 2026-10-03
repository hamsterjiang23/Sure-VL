"""CPU-only checks for the isolated veRL controller's group contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sure_vl.training.verl.driver import _group_records, load_verl_plan


class VerlDriverTests(unittest.TestCase):
    def test_group_records_preserve_all_four_sample_assessments(self):
        rows = [{"id": "same-prompt", "group_index": index} for index in range(4)]
        result = SimpleNamespace(non_tensor_batch={"group_records_json": [json.dumps(rows)]})
        self.assertEqual(_group_records(result, 4), rows)
        with self.assertRaisesRegex(RuntimeError, "incomplete prompt group"):
            _group_records(result, 3)

    def test_plan_accepts_g4_and_rejects_single_completion(self):
        source = Path(__file__).resolve().parents[1] / "configs/verl/qwen35_08b_visionopd_100step.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation"):
                image = root / f"{split}-student.png"
                teacher = root / f"{split}-teacher.png"
                image.touch()
                teacher.touch()
                example = {
                    "id": f"{split}-1", "split": split,
                    "student_image": str(image), "teacher_image": str(teacher),
                    "question": "Choose A or B", "accepted_answers": ["A"],
                }
                (root / f"{split}.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
                config[f"{split}_manifest"] = str(root / f"{split}.jsonl")
            config["output_dir"] = str(root / "new-run")
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plan = load_verl_plan(path)
            self.assertEqual(plan.config["setting"]["num_generations"], 4)
            config["setting"].update(num_generations=1, generation_batch_size=1,
                                     gradient_accumulation_steps=1)
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete groups|group prototype"):
                load_verl_plan(path)


if __name__ == "__main__":
    unittest.main()
