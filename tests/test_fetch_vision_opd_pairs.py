"""Bounded HTTP Range cache and safe extraction of official image members."""

from __future__ import annotations

import hashlib
import fcntl
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_vision_opd_pairs import MIRROR_ROOT, download_prefix, extract_complete_pngs

try:
    import requests
    from PIL import Image
except ImportError:
    requests = None
    Image = None


class _FakeResponse:
    def __init__(self, data: bytes, start: int, end: int, *, fail_after_first_chunk: bool):
        self.data = data
        self.start = start
        self.end = end
        self.fail_after_first_chunk = fail_after_first_chunk
        self.status_code = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
            "ETag": '"fixed-official-blob"',
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size: int):
        yield self.data[self.start:min(self.start + 3, self.end + 1)]
        if self.fail_after_first_chunk:
            raise requests.ConnectionError("temporary transfer reset")
        if self.start + 3 <= self.end:
            yield self.data[self.start + 3:self.end + 1]


class _FakeSession:
    def __init__(self, data: bytes):
        self.data = data
        self.starts: list[int] = []

    def get(self, _url, *, headers, stream, timeout):
        range_value = headers["Range"].removeprefix("bytes=")
        start, end = map(int, range_value.split("-"))
        self.starts.append(start)
        return _FakeResponse(
            self.data, start, end, fail_after_first_chunk=(len(self.starts) == 1),
        )


@unittest.skipIf(requests is None, "requests is needed for HTTP Range fetch")
class RangeDownloadTests(unittest.TestCase):
    def test_interrupted_request_resumes_exact_next_byte_and_records_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = b"the-official-archive-prefix"
            cache = Path(temporary) / "teacher.prefix"
            session = _FakeSession(data)
            result = download_prefix(
                label="test", url=MIRROR_ROOT + "/teacher_images/teacher_images.tar.gz",
                cache_path=cache, target_bytes=17, archive_size=len(data),
                session=session, max_attempts=2,
            )
            self.assertEqual(session.starts, [0, 3])
            self.assertEqual(cache.read_bytes(), data[:17])
            self.assertEqual(result["prefix_sha256"], hashlib.sha256(data[:17]).hexdigest())
            sidecar = json.loads(cache.with_suffix(".prefix.json").read_text())
            self.assertEqual(sidecar["downloaded_bytes"], 17)
            self.assertEqual(sidecar["prefix_sha256"], result["prefix_sha256"])
            self.assertEqual(sidecar["hf_revision"], MIRROR_ROOT.rsplit("/", 1)[1])
            cache.write_bytes(b"X" + cache.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "range cache SHA256 changed"):
                download_prefix(
                    label="test", url=MIRROR_ROOT + "/teacher_images/teacher_images.tar.gz",
                    cache_path=cache, target_bytes=17, archive_size=len(data),
                    session=session,
                )

    def test_refuses_another_writer_of_the_same_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "student.prefix"
            lock_path = cache.with_suffix(cache.suffix + ".lock")
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already has a writer"):
                    download_prefix(
                        label="student", url=MIRROR_ROOT + "/images/images.tar.gz00",
                        cache_path=cache, target_bytes=3, archive_size=3,
                        session=_FakeSession(b"abc"),
                    )


@unittest.skipIf(Image is None, "Pillow is needed for PNG verification")
class SafeExtractionTests(unittest.TestCase):
    @staticmethod
    def _png(size: tuple[int, int]) -> bytes:
        image = Image.frombytes("RGB", size, bytes((i * 37) % 256 for i in range(size[0] * size[1] * 3)))
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def _add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
        entry = tarfile.TarInfo(name)
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))

    def test_rejects_traversal_and_incomplete_final_png(self) -> None:
        small = self._png((2, 2))
        large = self._png((512, 512))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "archive.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                self._add_file(archive, "./small.png", small)
                self._add_file(archive, "../../escape.png", small)
                entry = tarfile.TarInfo("./symlink.png")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "../../outside"
                archive.addfile(entry)
                self._add_file(archive, "./large.png", large)
            prefix = root / "prefix.gz"
            content = archive_path.read_bytes()
            prefix.write_bytes(content[:len(content) // 2])
            output = root / "images"
            result = extract_complete_pngs(
                label="test", prefix_path=prefix, output_dir=output,
                allowed_filenames={"small.png", "large.png", "escape.png", "symlink.png"},
            )
            self.assertEqual((output / "small.png").read_bytes(), small)
            self.assertFalse((output / "large.png").exists())
            self.assertFalse((root / "escape.png").exists())
            self.assertFalse((output / "symlink.png").exists())
            self.assertEqual(result["truncated_prefix"], 1)
            with self.assertRaisesRegex(ValueError, "complete archive ended"):
                extract_complete_pngs(
                    label="test", prefix_path=prefix, output_dir=output,
                    allowed_filenames={"small.png", "large.png"}, complete_archive=True,
                )

    def test_complete_gzip_tar_can_cross_split_archive_byte_boundaries(self) -> None:
        first = self._png((5, 5))
        second = self._png((8, 8))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "archive.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                self._add_file(archive, "./first.png", first)
                self._add_file(archive, "./second.png", second)
            content = archive_path.read_bytes()
            cuts = (len(content) // 3, len(content) // 3 + 1)
            parts = (root / "part00", root / "part01", root / "part02")
            for path, data in zip(parts, (content[:cuts[0]], content[cuts[0]:cuts[1]], content[cuts[1]:]), strict=True):
                path.write_bytes(data)
            output = root / "images"
            result = extract_complete_pngs(
                label="split", prefix_path=parts, output_dir=output,
                allowed_filenames={"first.png", "second.png"}, complete_archive=True,
            )
            self.assertEqual(result["extracted"], 2)
            self.assertEqual(result["truncated_prefix"], 0)
            self.assertEqual((output / "first.png").read_bytes(), first)
            self.assertEqual((output / "second.png").read_bytes(), second)


if __name__ == "__main__":
    unittest.main()
