from typing import Any, ClassVar, Dict, Optional, Union, List, Callable, Literal
from pydantic import AnyUrl
from docling.datamodel.pipeline_options import OcrOptions
from docling_core.types.doc.page import SegmentedPage
from openai import OpenAI
import base64
import logging
import re

_log = logging.getLogger(__name__)

class ApiOcrOptions(OcrOptions):
    """Use a remote OpenAI-compatible Vision LLM endpoint as OCR backend."""
    kind: ClassVar[Literal["api_ocr"]] = "api_ocr"

    # Inherited OcrOptions fields with defaults
    lang: List[str] = [
        "english",
        "chinese",
    ]
    force_full_page_ocr: bool = False
    bitmap_area_threshold: float = 0.05

    # Remote API settings
    url: AnyUrl
    model: str
    api_key: str
    # headers: Dict[str, str] = {}
    # params: Dict[str, Any] = {}
    timeout: float = 60.0
    concurrency: int = 1

    # VLM prompt and image preprocessing
    prompt: Union[str, Callable[[Optional[SegmentedPage]], str]]
    scale: float = 2.0
    max_size: Optional[int] = None
    temperature: float = 0.0

    class Config:
        extra = "forbid"

    def run_ocr(self, image_bytes: bytes) -> str:
        """
        Call the remote API (OpenAI-compatible Vision LLM) to perform OCR on image_bytes, return extracted text.
        """
        # Encode image to base64
        base64_image = self.encode_image(image_bytes)
        # Prepare OpenAI client
        client = OpenAI(
            api_key=self.api_key,
            base_url=str(self.url),
        )
        prompt_text = self.prompt(None) if callable(self.prompt) else self.prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            # stream=True,
            # stream_options={"include_usage": True}
        )
        md = completion.model_dump()["choices"][0]["message"]["content"]
        _log.info(f"vlm response: \n{md}")
        return md
        # result = []
        # for chunk in completion:
        #     # You can customize this: print(chunk.model_dump()) or accumulate results
        #     result.append(chunk.model_dump())
        # return "\n".join([str(r) for r in result])

    
    def encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')
