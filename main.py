from paddleocr import PaddleOCR
from warnings import filterwarnings
import cv2

filterwarnings("ignore")

ocr = PaddleOCR(
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

result = ocr.predict(img_rgb)

data = []
if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
    # This matches your debug output
    for page_res in result:
        texts = page_res.get("rec_texts", [])
        scores = page_res.get("rec_scores", [])
        polys = page_res.get("rec_polys", [])
        for t, s, p in zip(texts, scores, polys):
            data.append(
                {
                    "text": t,
                    "score": float(s),
                    "poly": p.tolist() if hasattr(p, "tolist") else p,
                }
            )
else:
    # Fallback to older format
    for line in result:
        box = line[0]
        text, score = line[1]
        data.append({"text": text, "score": float(score), "box": box})

# Print only the extracted text on the console
for item in data:
    print(item["text"])
