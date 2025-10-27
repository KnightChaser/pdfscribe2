# ocr_app/pdf.py
from __future__ import annotations

from pathlib import Path
from typing import List
import tempfile

from pdf2image import convert_from_path


def pdf_to_images(pdf_path: Path, dpi: int) -> List[Path]:
    """
    Convert a PDF into a list of PNG files (one per page).

    Returns paths in a temporary directory (callers can copy/move as needed).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
    pages = convert_from_path(str(pdf_path), dpi=dpi, output_folder=str(tmpdir), fmt="png")

    out_paths: List[Path] = []
    for index, pil_image in enumerate(pages, start=1):
        p = tmpdir / f"page_{index:04d}.png"
        pil_image.save(p, "PNG")
        out_paths.append(p)
    return out_paths
