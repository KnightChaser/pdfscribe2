# ocr_app/cli.py
from __future__ import annotations

import cv2
import typer
import logging
from pathlib import Path

from paddleocr import PPStructureV3

from .pipeline import run_pipeline_on_image
from .pdf import pdf_to_images
from .utils import ensure_dir, is_pdf, to_rgb

# Silence noisy libs if desired
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

app = typer.Typer(add_completion=False)

@app.command()
def run(
    input_document: str = typer.Option(..., "--input-document", "-i", help="Path to PDF or image."),
    out_dir: str = typer.Option("./output", "--out-dir", "-o", help="Output root directory."),
    dpi: int = typer.Option(300, "--dpi", help="DPI for PDF → image conversion."),
    lang: str = typer.Option("korean", "--lang", help="PaddleOCR language."),
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
) -> None:
    """
    PDF/Image --> PPStructureV3 --> per-page Markdown + extracted figures.
    No config file. All parameters via CLI flags.
    """
    input_path = Path(input_document)
    if not input_path.exists():
        typer.secho(f"Input not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    out_root = Path(out_dir)
    out_md_dir = out_root / "output_md"
    out_img_dir = out_root / "output_images"
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
    if is_pdf(input_path):
        typer.secho(f"[+] Converting PDF --> images @ {dpi} dpi ...", fg=typer.colors.CYAN)
        image_paths = pdf_to_images(input_path, dpi)
    else:
        image_paths = [input_path]

    # Process each page
    for page_no, img_path in enumerate(image_paths, start=1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            typer.secho(f"[!] Skipping unreadable image: {img_path}", fg=typer.colors.YELLOW)
            continue
        img_rgb = to_rgb(img_bgr)
        typer.secho(f"[+] Processing page {page_no} ({img_path.name})", fg=typer.colors.GREEN)
        run_pipeline_on_image(pipeline, img_rgb, page_no, out_md_dir, out_img_dir)

    typer.secho(f"[✓] Done. Output --> {out_root}", fg=typer.colors.BRIGHT_GREEN)

