from paddleocr import PPStructureV3
from pdf2image import convert_from_path
from warnings import filterwarnings
import cv2
import os
import tempfile

filterwarnings("ignore")

pipeline = PPStructureV3(
    lang="korean",
    use_doc_orientation_classify=False,  # Disable document orientation classification model
    use_doc_unwarping=False,  # Disable text image unwarping model
    use_textline_orientation=False,  # Disable text line orientation classification model
    enable_mkldnn=True,  # speed optimization on CPU if available
)

pdf_path = "./example/example_report.pdf"
output_md_folder = "./output_md"
output_img_folder = "./output_images"
os.makedirs(output_md_folder, exist_ok=True)
os.makedirs(output_img_folder, exist_ok=True)

# 1) Convert PDF -> images (one image per page)
with tempfile.TemporaryDirectory() as tmpdir:
    pages = convert_from_path(pdf_path, dpi=300, output_folder=tmpdir, fmt="png")

    for page_index, pil_img in enumerate(pages):
        # Save temporary image or convert for OpenCV
        img_path = os.path.join(tmpdir, f"page_{page_index+1}.png")
        pil_img.save(img_path, "PNG")

        # Read via OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found after conversion: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) Run OCR/layout pipeline for each page image
        results = pipeline.predict(img_rgb)

        # 3) Process output
        for sub_page_idx, page in enumerate(results):
            md = page.markdown
            texts = md.get("markdown_texts", [])
            md_path = os.path.join(
                output_md_folder, f"pdfpage_{page_index+1}_sub_{sub_page_idx+1}.md"
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(texts)
            print(f"Markdown for page {page_index+1} saved to {md_path}")

            # Extract image blocks if any
            # note: attribute access might need dict-style as discussed
            imgs_info = []
            if hasattr(page, "imgs_in_doc"):
                imgs_info = getattr(page, "imgs_in_doc")
            elif isinstance(page, dict):
                imgs_info = page.get("imgs_in_doc", [])

            for img_index, img_info in enumerate(imgs_info):
                pil_img_block = img_info.get("img")
                if pil_img_block:
                    img_save_path = os.path.join(
                        output_img_folder,
                        f"page_{page_index+1}_img_{img_index+1}.png"
                    )
                    pil_img_block.save(img_save_path)
                    print(f"Saved image block: {img_save_path}")
                else:
                    # fallback: crop original img_rgb using coordinates
                    coord = img_info.get("coordinate")
                    if coord:
                        x1, y1, x2, y2 = map(int, coord)
                        crop = img_rgb[y1:y2, x1:x2]
                        crop_path = os.path.join(
                            output_img_folder,
                            f"page_{page_index+1}_img_{img_index+1}_fallback.png"
                        )
                        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                        print(f"Saved fallback cropped image: {crop_path}")
