"""CPU-only checks for the frozen visual-proxy diagnostic tokenpack contract."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_visual_proxy_diagnostic as diagnostic
from sure_vl.proxy_prompt import PROXY_OUTPUT_PROTOCOL


PROMPT_SOURCE = Path(__file__).resolve().parents[1] / "src/sure_vl/proxy_prompt.py"


class VisualProxyDiagnosticProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenpack = self.root / "tokenpack.jsonl"
        self.prepare_manifest = self.root / "manifest.json"
        self.prompt_source_sha256 = hashlib.sha256(PROMPT_SOURCE.read_bytes()).hexdigest()

    def _write_pack(
        self,
        *,
        row_changes: dict | None = None,
        manifest_changes: dict | None = None,
        drop_row_fields: tuple[str, ...] = (),
        drop_manifest_fields: tuple[str, ...] = (),
    ) -> tuple[dict, dict]:
        row = {
            "example_id": "validation-1",
            "completion_ids": [123, 456],
            "prompt_token_sha256": "a" * 64,
            "proxy_output_protocol": PROXY_OUTPUT_PROTOCOL,
            "proxy_prompt_source_sha256": self.prompt_source_sha256,
            "reason_text": "A short deduction.",
            "format_errors": [],
        }
        row.update(row_changes or {})
        for field in drop_row_fields:
            row.pop(field)
        self.tokenpack.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = {
            "mode": "prepare",
            "status": "completed",
            "tokenpack_sha256": hashlib.sha256(self.tokenpack.read_bytes()).hexdigest(),
            "selected_ids": [row["example_id"]],
            "selected_count": 1,
            "proxy_output_protocol": PROXY_OUTPUT_PROTOCOL,
            "proxy_prompt_source_sha256": self.prompt_source_sha256,
        }
        manifest.update(manifest_changes or {})
        for field in drop_manifest_fields:
            manifest.pop(field)
        self.prepare_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return row, manifest

    def _validate(self) -> tuple[list[dict], dict]:
        return diagnostic._validate_tokenpack_protocol(
            self.tokenpack,
            self.prepare_manifest,
            expected_protocol=PROXY_OUTPUT_PROTOCOL,
            expected_prompt_source_sha256=self.prompt_source_sha256,
        )

    def test_current_protocol_tokenpack_is_accepted_without_model_dependencies(self) -> None:
        row, manifest = self._write_pack()
        records, returned_manifest = self._validate()
        self.assertEqual(records, [row])
        self.assertEqual(returned_manifest, manifest)

    def test_legacy_tokenpack_without_protocol_metadata_is_rejected(self) -> None:
        for location in ("both", "row_only"):
            with self.subTest(location=location):
                fields = ("proxy_output_protocol", "proxy_prompt_source_sha256")
                self._write_pack(
                    drop_row_fields=fields,
                    drop_manifest_fields=fields if location == "both" else (),
                )
                with self.assertRaises(ValueError):
                    self._validate()

    def test_score_rejects_legacy_pack_before_model_loading(self) -> None:
        fields = ("proxy_output_protocol", "proxy_prompt_source_sha256")
        self._write_pack(drop_row_fields=fields, drop_manifest_fields=fields)
        args = SimpleNamespace(
            tokenpack=self.tokenpack,
            prepare_manifest=self.prepare_manifest,
            shard_index=0,
            num_shards=1,
        )
        with patch.object(diagnostic, "_load_model") as load_model, patch.object(
            diagnostic, "_load_processor"
        ) as load_processor, patch.dict(sys.modules, {"torch": None, "PIL": None}):
            with self.assertRaises(ValueError):
                diagnostic._score(args)
            load_model.assert_not_called()
            load_processor.assert_not_called()

    def test_prompt_source_drift_is_rejected_even_when_pack_and_manifest_agree(self) -> None:
        old_digest = "0" * 64
        self._write_pack(
            row_changes={"proxy_prompt_source_sha256": old_digest},
            manifest_changes={"proxy_prompt_source_sha256": old_digest},
        )
        with self.assertRaises(ValueError):
            self._validate()

    def test_prepare_manifest_must_match_tokenpack_bytes_and_order(self) -> None:
        for manifest_changes in (
            {"tokenpack_sha256": "0" * 64},
            {"selected_ids": ["another-example"]},
            {"mode": "score"},
        ):
            with self.subTest(manifest_changes=manifest_changes):
                self._write_pack(manifest_changes=manifest_changes)
                with self.assertRaises(ValueError):
                    self._validate()

    def test_reason_field_types_are_validated(self) -> None:
        self._write_pack(drop_row_fields=("reason_text",))
        with self.assertRaises(ValueError):
            self._validate()
        for row_changes in (
            {"reason_text": 123},
            {"format_errors": "missing_reason"},
            {"format_errors": ["missing_reason", 123]},
        ):
            with self.subTest(row_changes=row_changes):
                self._write_pack(row_changes=row_changes)
                with self.assertRaises(ValueError):
                    self._validate()

    def test_student_prompt_token_hash_detects_changed_prefix(self) -> None:
        original_ids = [1, 2, 3]
        expected_sha256 = hashlib.sha256(b"[1,2,3]").hexdigest()
        self.assertEqual(
            diagnostic._assert_student_prompt_hash(original_ids, expected_sha256, "validation-1"),
            expected_sha256,
        )
        with self.assertRaises(ValueError):
            diagnostic._assert_student_prompt_hash([1, 2, 4], expected_sha256, "validation-1")


if __name__ == "__main__":
    unittest.main()
