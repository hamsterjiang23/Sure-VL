"""Command-line validation and offline audit for frozen JSONL protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from .metrics import audit_attempts
from .protocol import ProtocolError, load_attempts_jsonl, load_examples_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(report: dict, destination: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination is None:
        sys.stdout.write(rendered)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sure-vl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a frozen example JSONL file")
    validate.add_argument("--examples", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    audit = subparsers.add_parser("evaluate", help="audit one complete split of structured outputs")
    audit.add_argument("--examples", type=Path, required=True)
    audit.add_argument("--outputs", type=Path, required=True)
    audit.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    try:
        examples = load_examples_jsonl(args.examples)
        protocol = {"examples_sha256": _sha256(args.examples)}
        if args.command == "validate":
            report = {
                "valid": True,
                "sample_count": len(examples),
                "splits": dict(sorted(Counter(example.split for example in examples).items())),
                **protocol,
            }
        else:
            attempts = load_attempts_jsonl(args.outputs)
            report = {
                **audit_attempts(examples, attempts),
                **protocol,
                "outputs_sha256": _sha256(args.outputs),
            }
        _emit(report, args.output)
    except (OSError, ProtocolError, ValueError) as error:
        parser.exit(2, f"sure-vl: {error}\n")
    return 0
