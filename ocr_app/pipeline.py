# ocr_app/pipeline.py
from __future__ import annotations

from pathlib import Path
import cv2

from paddleocr import PPStructureV3

from .utils import (
    get_imgs_in_doc,
    save_markdown,
    crop_with_coord,
)
from .types import NDArray


def run_pipeline_on_image(
    pipeline: PPStructureV3,
    img_rgb: NDArray,
    page_index: int,
    out_md_dir: Path,
    out_img_dir: Path,
) -> None:
    """
    Run PPStructureV3 on a single page image.

    Writes:
      - Markdown fragments per sub-page: out_md_dir / "page_####_##.md"
      - Extracted figure images: out_img_dir / "page_####_##_img_##.png"
    """
    results = pipeline.predict(img_rgb)

    for sub_idx, page in enumerate(results, start=1):
        # 1) Markdown
        md = getattr(page, "markdown", {}) or {}
        texts = md.get("markdown_texts", [])
        md_save_path = out_md_dir / f"page_{page_index:04d}_{sub_idx:02d}.md"
        save_markdown(texts, md_save_path)
        print(f"[+] Markdown saved: {md_save_path}")

        # 2) Images (figures)
        imgs_info = get_imgs_in_doc(page)
        for img_idx, img_info in enumerate(imgs_info, start=1):
            pil_img = img_info.get("img")
            coord = img_info.get("coordinate")

            if pil_img is not None:
                save_path = out_img_dir / f"page_{page_index:04d}_{sub_idx:02d}_img_{img_idx:02d}.png"
                pil_img.save(save_path)
                print(f"[+] Figure saved: {save_path}")
            elif coord is not None:
                # NOTE: Fallback routine
                crop = crop_with_coord(img_rgb, coord)
                save_path = out_img_dir / f"page_{page_index:04d}_{sub_idx:02d}_img_{img_idx:02d}_fallback.png"
                # OpenCV expects BGR
                cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"[+] Figure (fallback) saved: {save_path}")

