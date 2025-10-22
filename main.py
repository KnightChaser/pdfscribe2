from paddleocr import PPStructureV3
from warnings import filterwarnings
import cv2
import os

filterwarnings("ignore")

pipeline = PPStructureV3(
    lang="korean",
    use_doc_orientation_classify=False,  # Disable document orientation classification model
    use_doc_unwarping=False,  # Disable text image unwarping model
    use_textline_orientation=False,  # Disable text line orientation classification model
    enable_mkldnn=True,  # speed optimization on CPU if available
)

img_path = "./example/document_screenshot.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = pipeline.predict(img_rgb)

output_md_folder = "./output_md"
output_img_folder = "./output_images"
os.makedirs(output_md_folder, exist_ok=True)
os.makedirs(output_img_folder, exist_ok=True)

for page_index, page in enumerate(results):
    # Extract markdown texts
    md = page.markdown
    texts = md.get("markdown_texts", [])
    md_path = os.path.join(output_md_folder, f"page_{page_index + 1}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(texts)
    print(f"Markdown for page {page_index + 1} saved to {md_path}")

    # Extract and save images
    imgs_info = getattr(page, "imgs_in_doc", None)
    if imgs_info is None:
        imgs_info = page.get("imgs_in_doc", []) if isinstance(page, dict) else []

    for img_index, img_info in enumerate(imgs_info):
        pil_img = img_info.get("img")
        if pil_img is None:
            # TODO: Make a fallback logic
            continue

        img_save_path = os.path.join(
            output_img_folder, f"page_{page_index + 1}_img_{img_index + 1}.png"
        )
        pil_img.save(img_save_path)
        print(
            f"Image for page {page_index + 1}, image {img_index + 1} saved to {img_save_path}"
        )
