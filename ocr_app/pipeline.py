# ocr_app/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List
import cv2
import os

from paddleocr import PPStructureV3

from .markdown_utils import join_texts, rewrite_image_srcs
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
      - All <img src="..."> and MD image links are rewritten to our saved figure paths.
    """
    results = pipeline.predict(img_rgb)

    for sub_idx, page in enumerate(results, start=1):
        # 1) Extract figures and save with canonical names
        imgs_info = get_imgs_in_doc(page)
        saved_paths: List[Path] = []
        original_srcs: List[str] = []

        for img_idx, info in enumerate(imgs_info, start=1):
            pil_image = info.get("img")
            coordinate = info.get("coordinate")
            md_src = info.get("path") # e.g., 'imgs/img_in_image_box_...jpg'
            if isinstance(md_src, str):
                original_srcs.append(md_src)

            save_path = out_img_dir / f"page_{page_index:04d}_{sub_idx:02d}_img_{img_idx:02d}.png"
            if pil_image is not None:
                # NOTE: Expected routine in normal cases
                pil_image.save(save_path)
            elif coordinate is not None:
                # NOTE: fallback routine
                crop = crop_with_coord(img_rgb, coordinate)
                cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            else:
                # nothing to save for this slot; keep alignment with original_srcs
                continue
            saved_paths.append(save_path)
            print(f"[+] Figure saved: {save_path}")

        # 2) Build a mapping original --> our exported rela (from MD file's folder)
        md_path = out_md_dir / f"page_{page_index:04d}_{sub_idx:02d}.md"
        mapping = {}
        for original, path in zip(original_srcs, saved_paths):
            # WARNING:
            # relative_to() fails unless 'path' is under md_path.parent; relpath works across siblings.
            rel = os.path.relpath(str(path), start=str(md_path.parent))
            mapping[original] = rel

        # 3) Markdown --> join + rewrite image srcs --> Finally save
        md_obj = getattr(page, "markdown", {}) or {}
        texts = md_obj.get("markdown_texts", [])
        blob  = join_texts(texts)
        blob  = rewrite_image_srcs(blob, mapping)

        save_markdown(blob, md_path)
        print(f"[+] MD saved (rewritten): {md_path}")
