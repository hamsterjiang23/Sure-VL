#!/usr/bin/env python3
"""Create tiny synthetic paired-image manifests for one-step interface checks.

These images are deliberately trivial. They verify file decoding and trainer
plumbing; they are not research training data or an effectiveness benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ppm_square(path: Path, *, color: tuple[int, int, int], size: int) -> None:
    background = (235, 235, 235)
    low, high = size // 4, 3 * size // 4
    pixels = bytearray()
    for row in range(size):
        for column in range(size):
            pixels.extend(color if low <= row < high and low <= column < high else background)
    path.write_bytes(f"P6\n{size} {size}\n255\n".encode("ascii") + pixels)


def create_fixture(directory: str | Path) -> tuple[Path, Path]:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    specifications = (("train", "blue", (25, 70, 220)), ("dev", "red", (220, 45, 45)))
    manifests = []
    for split, answer, rgb in specifications:
        restricted = root / f"{split}-restricted.ppm"
        clear = root / f"{split}-clear.ppm"
        _ppm_square(restricted, color=rgb, size=32)
        _ppm_square(clear, color=rgb, size=128)
        manifest = root / f"{split}.jsonl"
        record = {
            "id": f"synthetic-{split}-square",
            "split": split,
            "student_image": restricted.name,
            "teacher_image": clear.name,
            "question": "What color is the square?",
            "required_visual_facts": {"shape": "square", "color": answer},
            "accepted_answers": [answer],
        }
        manifest.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        manifests.append(manifest)
    return manifests[0], manifests[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="make paired Sure-VL smoke data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train, dev = create_fixture(args.output_dir)
    print(json.dumps({"train_manifest": str(train), "validation_manifest": str(dev)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
