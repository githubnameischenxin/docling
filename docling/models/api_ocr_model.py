from typing import Iterable, Type
import numpy as np
from docling_core.types.doc.page import TextCell, BoundingRectangle
from docling_core.types.doc import CoordOrigin, BoundingBox
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
from docling.datamodel.settings import settings
from docling.datamodel.api_ocr_options import ApiOcrOptions 
import logging

_log = logging.getLogger(__name__)

class ApiOcrModel(BaseOcrModel):
    """OCR model that delegates to ApiOcrOptions.run_ocr."""
    KIND = "api_ocr"

    def __init__(self, enabled: bool, artifacts_path, options: ApiOcrOptions, accelerator_options):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options = options
        self.scale = options.scale or 2.0

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    high_res_image = page._backend.get_page_image(scale=self.scale, cropbox=ocr_rect)
                    im = np.array(high_res_image)

                    # Convert to PNG bytes
                    from io import BytesIO
                    import PIL.Image
                    buf = BytesIO()
                    PIL.Image.fromarray(im).save(buf, format="PNG")
                    image_bytes = buf.getvalue()

                    _log.info("Running OCR on image...start")
                    # Use remote API
                    try:
                        markdown_text = self.options.run_ocr(image_bytes)
                    except Exception as e:
                        _log.warning(f"OCR API failed: {e}")
                        markdown_text = ""

                    _log.info("Running OCR on image...end")
                    # Create a fake cell for now (position = bounding box)
                    if markdown_text.strip():
                        cell = TextCell(
                            index=0,
                            text=markdown_text,
                            orig=markdown_text,
                            from_ocr=True,
                            confidence=1.0,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    (
                                        ocr_rect.l,
                                        ocr_rect.t,
                                        ocr_rect.r,
                                        ocr_rect.b
                                    ),
                                    origin=CoordOrigin.TOPLEFT
                                )
                            ),
                        )
                        all_ocr_cells.append(cell)

                self.post_process_cells(all_ocr_cells, page)

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[ApiOcrOptions]:
        return ApiOcrOptions
