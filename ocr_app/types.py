# ocr_app/types.py
from __future__ import annotations
from typing import Any, Iterable, Tuple, TypedDict, Union
import numpy as np

NDArray = np.ndarray

class ImgInfo(TypedDict, total=False):
    img: Any
    coordinate: Tuple[int, int, int, int]
    path: str
    score: float

MarkdownTexts = Union[str, Iterable[str]]
PageObject = Any
