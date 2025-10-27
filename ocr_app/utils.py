# ocr_app/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple
import cv2

from .types import NDArray, MarkdownTexts


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def to_rgb(img_bgr: NDArray) -> NDArray:
    """
    Convert a BGR OpenCV image to RGB.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_markdown(texts: MarkdownTexts, save_path: Path) -> None:
    """
    Persist markdown fragments or a single string to disk.
    """
    if isinstance(texts, (list, tuple)):
        content = "\n\n".join(str(t) for t in texts)
    else:
        content = str(texts)
    save_path.write_text(content, encoding="utf-8")


def get_imgs_in_doc(page_obj: Any) -> List[dict]:
    """
    Retrieve image blocks from a PPStructureV3 page result.

    PPStructureV3 may return an object that doesn't expose all keys as attributes.
    Try attribute -> mapping -> __dict__.
    """
    # Attribute path
    imgs = getattr(page_obj, "imgs_in_doc", None)
    if imgs is not None:
        return list(imgs)

    # Mapping path
    if isinstance(page_obj, dict):
        return list(page_obj.get("imgs_in_doc", []) or [])

    # NOTE: Fallback: inspect __dict__
    d = getattr(page_obj, "__dict__", None)
    if isinstance(d, dict):
        return list(d.get("imgs_in_doc", []) or [])

    return []


def crop_with_coord(img_rgb: NDArray, coord: Tuple[int, int, int, int]) -> NDArray:
    """
    Crop an RGB image using (x1, y1, x2, y2) with bounds safety.
    """
    x1, y1, x2, y2 = map(int, coord)
    h, w = img_rgb.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return img_rgb[y1:y2, x1:x2]


def is_pdf(path: Path) -> bool:
    """
    True if path looks like a PDF file. (A simple logic)
    """
    return path.suffix.lower() == ".pdf"

