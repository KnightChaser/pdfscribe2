# ocr_app/markdown_utils.py
from __future__ import annotations

import re
from typing import Dict, Iterable

# HTML: capture "<img ... src=\"" as group 1, the URL as group 2, and the trailing quote+attrs "..." as group 3
_HTML_IMG = re.compile(r'(<img\s+[^>]*\bsrc=["\'])([^"\']+)(["\'][^>]*>)', re.IGNORECASE)

# Markdown: capture "![](" as group 1, the URL as group 2, and the closing ")" (possibly with title) as group 3
_MD_IMG = re.compile(r'(!\[[^\]]*\]\()([^)]+)(\))')

def join_texts(texts: Iterable[str]) -> str:
    """
    Join markdown/HTML fragments into a single string.
    """
    return "".join(str(t) for t in texts)

def rewrite_image_srcs(md_text: str, mapping: Dict[str, str]) -> str:
    """
    Rewrite all image sources in the given Markdown/HTML content using `mapping`.
    Keys: original src (e.g., 'imgs/img_in_image_box_...jpg')
    Values: final relative path (e.g., 'output_images/page_0001_01_img_01.png')
    """
    def _html(m: re.Match) -> str:
        pre, src, post = m.groups()
        return f'{pre}{mapping.get(src, src)}{post}'

    def _md(m: re.Match) -> str:
        pre, src, post = m.groups()
        return f'{pre}{mapping.get(src, src)}{post}'

    text = _HTML_IMG.sub(_html, md_text)
    text = _MD_IMG.sub(_md, text)
    return text
