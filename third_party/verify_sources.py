#!/usr/bin/env python3
"""Compare vendored source files with the pinned upstream snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    differences: list[str] = []
    for source in manifest["sources"]:
        directory = root / source["name"]
        expected = {item["path"]: item for item in source["files"]}
        actual: dict[str, Path] = {}
        for path in directory.rglob("*"):
            if path.is_dir():
                continue
            actual[path.relative_to(directory).as_posix()] = path
        for rel in sorted(expected.keys() - actual.keys()):
            differences.append(f"{source['name']}: missing {rel}")
        for rel in sorted(actual.keys() - expected.keys()):
            differences.append(f"{source['name']}: added {rel}")
        for rel in sorted(expected.keys() & actual.keys()):
            path = actual[rel]
            if path.is_symlink() or not path.is_file():
                differences.append(f"{source['name']}: non-regular {rel}")
                continue
            content = path.read_bytes()
            recorded = expected[rel]
            if (hashlib.sha256(content).hexdigest() != recorded["sha256"]
                    or len(content) != recorded["size"]
                    or bool(path.stat().st_mode & 0o111) != recorded["executable"]):
                differences.append(f"{source['name']}: modified {rel}")
    if differences:
        print("\n".join(differences))
        return 1
    print("All vendored files match the three pinned upstream snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
