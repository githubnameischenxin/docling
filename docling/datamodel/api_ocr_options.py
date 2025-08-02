from typing import ClassVar, Optional, Union, List, Callable, Literal
from pydantic import AnyUrl
from docling.datamodel.pipeline_options import OcrOptions
from docling_core.types.doc.page import SegmentedPage
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from docling.grpclient.inference_client import InferenceClient
from docling.proto.inference_pb2 import (
    CompletionRequest,
    ModelParams,
    ModelTrait,
    SamplerParams,
    UserMessage,
    UserContent,
    Image,
    ImageMediaType
)

_log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8)

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

    # Remote gRPC API settings
    mTLS_grpc_address: str
    mTLS_ca_cert_path: str
    mTLS_client_cert_path: str
    mTLS_client_key_path: str
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
        Call mTLS grpc api to perform OCR on image_bytes, return extracted text.
        """
        def runner():
            return asyncio.run(self._run_ocr_async(image_bytes))
        
        return _executor.submit(runner).result()
    
    async def _run_ocr_async(self, image_bytes: bytes) -> str:
        """
        Async call mTLS grpc api to perform OCR on image_bytes, return extracted text.
        """
        prompt_text = self.prompt(None) if callable(self.prompt) else self.prompt
        client = InferenceClient(
            self.mTLS_grpc_address,
            self.mTLS_ca_cert_path,
            self.mTLS_client_cert_path,
            self.mTLS_client_key_path,
        )
        await client.connect()
        request = CompletionRequest(
            model_params=ModelParams(
                model_trait=ModelTrait.MODEL_TRAIT_IMAGE_TO_TEXT,
                preamble="you are a helpful assistant",
                sampler_params=SamplerParams(
                    max_tokens=10000,
                    temperature=1.0,
                )
            ),
            user_message=UserMessage(
                user_contents=[
                    UserContent(
                        text=prompt_text
                    ),
                    UserContent(
                        image=Image(
                            media_type=ImageMediaType.IMAGE_MEDIA_TYPE_JPEG,
                            data=image_bytes
                        )
                    )
                ]
            ),
        )
        response = await client.completion(request)
        await client.close()
        md = response.message
        _log.info(f"vlm response: \n{md}")
        return md
        

    # def run_ocr(self, image_bytes: bytes) -> str:
    #     """
    #     Call the remote API (OpenAI-compatible Vision LLM) to perform OCR on image_bytes, return extracted text.
    #     """
    #     # Encode image to base64
    #     base64_image = self.encode_image(image_bytes)
    #     # Prepare OpenAI client
    #     client = OpenAI(
    #         api_key=self.api_key,
    #         base_url=str(self.url),
    #     )
    #     prompt_text = self.prompt(None) if callable(self.prompt) else self.prompt
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt_text},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ]
    #     completion = client.chat.completions.create(
    #         model=self.model,
    #         messages=messages,
    #         # stream=True,
    #         # stream_options={"include_usage": True}
    #     )
    #     md = completion.model_dump()["choices"][0]["message"]["content"]
    #     _log.info(f"vlm response: \n{md}")
    #     return md
    #     # result = []
    #     # for chunk in completion:
    #     #     # You can customize this: print(chunk.model_dump()) or accumulate results
    #     #     result.append(chunk.model_dump())
    #     # return "\n".join([str(r) for r in result])

    
    # def encode_image(self, image_bytes):
    #     return base64.b64encode(image_bytes).decode('utf-8')
