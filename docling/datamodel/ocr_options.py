from typing import ClassVar, List
from pydantic import BaseModel


class BaseOptions(BaseModel):
    """Base class for options."""

    kind: ClassVar[str]

class OcrOptions(BaseOptions):
    """OCR options."""

    lang: List[str]
    force_full_page_ocr: bool = False  # If enabled a full page OCR is always applied
    bitmap_area_threshold: float = (
        0.05  # percentage of the area for a bitmap to processed with OCR
    )