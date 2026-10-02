"""Build an explicit evidence crop for the privileged teacher view.

Vision-OPD uses an evidence-centered bounding box: the student receives the
full image with a red box, and the teacher receives only the box crop resized
by 2x. Its released training code consumes prebuilt images, so interpolation
and rectangle width here are documented Sure-VL implementation choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


STUDENT_FOCUS_HINT = "Only focus on the region inside the red bounding box."
VISION_OPD_REFERENCE_COMMIT = "06860e69b5ed9dc24e96ca5c855f3a4ef25976aa"


@dataclass(frozen=True)
class TeacherView:
    student_image: Any
    teacher_image: Any
    metadata: dict[str, Any]


def validate_source_bbox(
    bbox_xyxy: Sequence[int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Validate a half-open pixel box in EXIF-oriented image coordinates."""
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        raise ValueError("evidence_bbox_xyxy must contain four integer pixel coordinates")
    if any(type(value) is not int for value in bbox_xyxy):
        raise ValueError("evidence_bbox_xyxy must contain four integer pixel coordinates")
    left, top, right, bottom = bbox_xyxy
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("evidence_bbox_xyxy must have positive area inside the source image")
    return left, top, right, bottom


def build_teacher_view(
    source_image: Any,
    evidence_bbox_xyxy: Sequence[int] | None,
    *,
    allow_no_roi: bool = False,
    no_roi_reason: str = "source_bbox_unavailable",
    box_width: int = 2,
) -> TeacherView:
    """Return full-image student and crop-only teacher images with provenance.

    Coordinates refer to the RGB image after EXIF orientation is applied.
    ``[left, top, right, bottom]`` is half-open, matching Pillow ``crop``.
    No center/random crop is inferred when the source has no evidence box.
    """
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as error:
        raise RuntimeError("teacher views require Pillow; use `uv run --extra train`") from error
    if not isinstance(source_image, Image.Image):
        raise TypeError("source_image must be a Pillow image")
    if type(box_width) is not int or box_width < 1:
        raise ValueError("box_width must be a positive integer")
    oriented = ImageOps.exif_transpose(source_image).convert("RGB")
    width, height = oriented.size
    if evidence_bbox_xyxy is None:
        if not allow_no_roi:
            raise ValueError("evidence_bbox_xyxy is required unless allow_no_roi=True")
        if not isinstance(no_roi_reason, str) or not no_roi_reason.strip():
            raise ValueError("no_roi_reason must be a nonempty string")
        return TeacherView(
            student_image=oriented.copy(),
            teacher_image=oriented.copy(),
            metadata={
                "kind": "no_evidence_roi",
                "reason": no_roi_reason,
                "source_image_size": [width, height],
                "bbox_coordinate_space": "EXIF-oriented source RGB pixels, half-open xyxy",
            },
        )
    left, top, right, bottom = validate_source_bbox(evidence_bbox_xyxy, oriented.size)
    crop = oriented.crop((left, top, right, bottom))
    teacher = crop.resize((2 * (right - left), 2 * (bottom - top)), Image.Resampling.LANCZOS)
    student = oriented.copy()
    ImageDraw.Draw(student).rectangle(
        (left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=box_width,
    )
    return TeacherView(
        student_image=student,
        teacher_image=teacher,
        metadata={
            "kind": "vision_opd_evidence_crop_2x",
            "source_bbox_xyxy": [left, top, right, bottom],
            "source_image_size": [width, height],
            "bbox_coordinate_space": "EXIF-oriented source RGB pixels, half-open xyxy",
            "crop_size": [right - left, bottom - top],
            "teacher_image_size": list(teacher.size),
            "crop_resize_scale": 2,
            "crop_interpolation": "Pillow.Image.Resampling.LANCZOS",
            "student_overlay": {"color_rgb": [255, 0, 0], "line_width": box_width},
            "student_image_hint": STUDENT_FOCUS_HINT,
        },
    )
