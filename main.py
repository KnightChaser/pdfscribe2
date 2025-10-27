# main.py
from __future__ import annotations

import os
import cv2
import typer
import tempfile
from typing import List, Tuple, Any, Iterable
from warnings import filterwarnings

from paddleocr import PPStructureV3
from pdf2image import convert_from_path

app = typer.Typer(add_completion=False)
filterwarnings("ignore")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_rgb(img_bgr: Any) -> Any:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_markdown(texts: Iterable[str], save_path: str) -> None:
    # texts can be a list of md fragments or a string (depending on pipeline)
    if isinstance(texts, (list, tuple)):
        content = "\n\n".join([str(t) for t in texts])
    else:
        content = str(texts)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

def get_imgs_in_doc(page_obj: Any) -> List[dict]:
    """
    PPStructureV3 sometimes returns an object (LayoutParsingResultV2) that
    doesn't expose all keys as attributes, even though keys exist.
    This accessor tries attribute -> dict -> __dict__.
    """
    # Attribute path
    imgs = getattr(page_obj, "imgs_in_doc", None)
    if imgs is not None:
        return imgs

    # Dict-style path
    if isinstance(page_obj, dict):
        return page_obj.get("imgs_in_doc", []) or []

    # Fallback: inspect __dict__
    d = getattr(page_obj, "__dict__", None)
    if isinstance(d, dict):
        return d.get("imgs_in_doc", []) or []

    return []


def crop_with_coord(img_rgb, coord: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = map(int, coord)
    h, w = img_rgb.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return img_rgb[y1:y2, x1:x2]

def run_pipeline_on_image(
    pipeline: PPStructureV3,
    img_rgb,
    page_index: int,
    out_md_dir: str,
    out_img_dir: str,
) -> None:
    results = pipeline.predict(img_rgb)

    for sub_idx, page in enumerate(results):
        # 1) Markdown
        md = getattr(page, "markdown", {}) or {}
        texts = md.get("markdown_texts", [])
        md_save_path = os.path.join(out_md_dir, f"page_{page_index:04d}_{sub_idx+1:02d}.md")
        save_markdown(texts, md_save_path)
        print(f"[+] MD saved: {md_save_path}")

        # 2) Images
        imgs_info = get_imgs_in_doc(page)
        for img_index, img_info in enumerate(imgs_info, start=1):
            pil_image = img_info.get("img")
            coord = img_info.get("coordinate")
            if pil_image is not None:
                save_path = os.path.join(
                    out_img_dir,
                    f"page_{page_index:04d}_{sub_idx+1:02d}_img_{img_index:02d}.png",
                )
                pil_image.save(save_path)
                print(f"[+] Figure saved: {save_path}")
            elif coord is not None:
                # NOTE: Fallback: crop from original image
                crop = crop_with_coord(img_rgb, coord)
                save_path = os.path.join(
                    out_img_dir,
                    f"page_{page_index:04d}_{sub_idx+1:02d}_img_{img_index:02d}_fallback.png",
                )
                cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"[+] Figure (fallback) saved: {save_path}")

def pdf_to_images(pdf_path: str, dpi: int) -> List[str]:
    """
    Convert a PDF into a list of PNG paths (one per page). Uses a temp dir; caller should copy/save outputs if needed.
    """
    tmpdir = tempfile.mkdtemp(prefix="pdf_ocr_")
    pages = convert_from_path(pdf_path, dpi=dpi, output_folder=tmpdir, fmt="png")
    img_paths: List[str] = []
    for index, pil_image in enumerate(pages, start=1):
        path = os.path.join(tmpdir, f"page_{index:04d}.png")
        pil_image.save(path, "PNG")
        img_paths.append(path)
    return img_paths


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


@app.command()
def run(
    input_document: str = typer.Option(..., "--input-document", "-i", help="Path to PDF or image."),
    out_dir: str = typer.Option("./output", "--out-dir", "-o", help="Output root directory."),
    dpi: int = typer.Option(300, "--dpi", help="DPI for PDF → image."),
    lang: str = typer.Option("korean", "--lang", help="Language code for PaddleOCR."),
    use_doc_orientation_classify: bool = typer.Option(
        False, "--doc-orient", help="Enable document orientation classification."
    ),
    use_doc_unwarping: bool = typer.Option(
        False, "--doc-unwarp", help="Enable text image unwarping."
    ),
    use_textline_orientation: bool = typer.Option(
        False, "--textline-orient", help="Enable text line orientation classification."
    ),
    enable_mkldnn: bool = typer.Option(
        True, "--mkldnn/--no-mkldnn", help="Enable MKLDNN for CPU inference."
    ),
):
    """
    PDF/Image --> PPStructureV3 --> per-page Markdown + extracted figures.
    """
    if not os.path.exists(input_document):
        typer.secho(f"Input not found: {input_document}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    out_md_dir = os.path.join(out_dir, "output_md")
    out_img_dir = os.path.join(out_dir, "output_images")
    ensure_dir(out_md_dir)
    ensure_dir(out_img_dir)

    # Init pipeline
    pipeline = PPStructureV3(
        lang=lang,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        enable_mkldnn=enable_mkldnn,
    )

    # Gather image pages
    image_paths: List[str] = []
    if is_pdf(input_document):
        typer.secho(f"[+] Converting PDF --> images @ {dpi} dpi …", fg=typer.colors.CYAN)
        image_paths = pdf_to_images(input_document, dpi)
    else:
        image_paths = [input_document]

    # Process each page
    for page_no, img_path in enumerate(image_paths, start=1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            typer.secho(f"[!] Skipping unreadable image: {img_path}", fg=typer.colors.YELLOW)
            continue
        img_rgb = to_rgb(img_bgr)
        typer.secho(f"[+] Processing page {page_no} ({os.path.basename(img_path)})", fg=typer.colors.GREEN)
        run_pipeline_on_image(pipeline, img_rgb, page_no, out_md_dir, out_img_dir)

    typer.secho(f"[✓] Done. Output --> {out_dir}", fg=typer.colors.BRIGHT_GREEN)


if __name__ == "__main__":
    app()
