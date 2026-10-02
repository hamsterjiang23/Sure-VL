import importlib.util
import tempfile
import unittest
from pathlib import Path

from sure_vl.protocol import load_examples_jsonl
from sure_vl.trl_data import assert_disjoint_manifests, manifest_to_gold_rows


class SmokeFixtureTests(unittest.TestCase):
    def test_fixture_is_valid_and_disjoint(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "make_smoke_fixture.py"
        spec = importlib.util.spec_from_file_location("make_smoke_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            train, dev = module.create_fixture(directory)
            assert_disjoint_manifests(manifest_to_gold_rows(train), manifest_to_gold_rows(dev))
            self.assertEqual(load_examples_jsonl(train)[0].accepted_answers, ("blue",))
            self.assertEqual((Path(directory) / "train-clear.ppm").read_bytes()[:2], b"P6")


if __name__ == "__main__":
    unittest.main()
