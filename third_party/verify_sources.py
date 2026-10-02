#!/usr/bin/env python3
"""Compare vendored source files with the pinned upstream snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _is_generated(relative: Path) -> bool:
    return any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"}
               or part.endswith(".egg-info") for part in relative.parts)


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    differences: list[str] = []
    for source in manifest["sources"]:
        directory = root / source["name"]
        expected = {item["path"]: item for item in source["files"]}
        if len(expected) != source["file_count"]:
            differences.append(f"{source['name']}: file_count differs from file manifest")
        if sum(item["size"] for item in expected.values()) != source["total_bytes"]:
            differences.append(f"{source['name']}: total_bytes differs from file manifest")
        if max((item["size"] for item in expected.values()), default=0) != source["largest_file_bytes"]:
            differences.append(f"{source['name']}: largest_file_bytes differs from file manifest")
        actual: dict[str, Path] = {}
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if _is_generated(relative):
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            actual[relative.as_posix()] = path
        for gitlink in source.get("gitlinks", []):
            rel = gitlink["path"]
            path = directory / rel
            if path.is_symlink() or path.is_file() or (path.is_dir() and any(path.iterdir())):
                differences.append(f"{source['name']}: unmaterialized gitlink has contents {rel}")
        for rel in sorted(expected.keys() - actual.keys()):
            differences.append(f"{source['name']}: missing {rel}")
        for rel in sorted(actual.keys() - expected.keys()):
            differences.append(f"{source['name']}: added {rel}")
        for rel in sorted(expected.keys() & actual.keys()):
            path = actual[rel]
            recorded = expected[rel]
            if recorded.get("kind") == "symlink":
                if not path.is_symlink():
                    differences.append(f"{source['name']}: non-symlink {rel}")
                    continue
                target = os.readlink(path)
                content = os.fsencode(target)
                if target != recorded["target"]:
                    differences.append(f"{source['name']}: modified {rel}")
                    continue
            elif path.is_symlink() or not path.is_file():
                differences.append(f"{source['name']}: non-regular {rel}")
                continue
            else:
                content = path.read_bytes()
            if (hashlib.sha256(content).hexdigest() != recorded["sha256"]
                    or len(content) != recorded["size"]
                    or (not path.is_symlink()
                        and bool(path.stat().st_mode & 0o111) != recorded["executable"])):
                differences.append(f"{source['name']}: modified {rel}")
    if differences:
        print("\n".join(differences))
        return 1
    print(f"All vendored files match the {len(manifest['sources'])} pinned upstream snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
