import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from docling.datamodel.api_ocr_options import ApiOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
_log = logging.getLogger(__name__)

app = FastAPI(
    title="MarkItDown service",
    description="convert document to markdown",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MTLS_GRPC_ADDRESS = os.getenv("MTLS_GRPC_ADDRESS", "127.0.0.1:9988")
MTLS_CA_CERT_PATH = os.getenv("MTLS_CA_CERT_PATH", "/etc/docling/certs/ca.crt")
MTLS_CLIENT_CERT_PATH = os.getenv("MTLS_CLIENT_CERT_PATH", "/etc/docling/certs/client.crt")
MTLS_CLIENT_KEY_PATH = os.getenv("MTLS_CLIENT_KEY_PATH", "/etc/docling/certs/client.key")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxx")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-vl-72b-instruct")
# prompt = "Convert this image to markdown. Do not miss any text and only output the bare markdown!"
prompt="Extract or describe the content of the image and convert it into markdown format, trying to ensure the information is complete and not add other information."

api_ocr_options = ApiOcrOptions(
    mTLS_grpc_address=MTLS_GRPC_ADDRESS,
    mTLS_ca_cert_path=MTLS_CA_CERT_PATH,
    mTLS_client_cert_path=MTLS_CLIENT_CERT_PATH,
    mTLS_client_key_path=MTLS_CLIENT_KEY_PATH,
    url=BASE_URL,
    model=LLM_MODEL,
    api_key=API_KEY,
    prompt=prompt,
    timeout=90,
    scale=1.0,
)

api_vlm_options = ApiVlmOptions(
    url=f"{BASE_URL}/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    params=dict(model=LLM_MODEL),
    prompt=prompt,
    timeout=90,
    scale=1.0,
    response_format=ResponseFormat.MARKDOWN,
)

converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.PPTX,
        InputFormat.ASCIIDOC,
        InputFormat.CSV,
        InputFormat.XLSX,
        InputFormat.MD,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
            backend=PyPdfiumDocumentBackend,
            pipeline_options=PdfPipelineOptions(
                enable_remote_services=True,
                do_code_enrichment=False,
                do_formula_enrichment=False,
                do_picture_classification=False,
                do_picture_description=False,
                ocr_options=api_ocr_options,
            ),
        ),
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(
                enable_remote_services=True,
                vlm_options=api_vlm_options
            ),
        )
    }
)


executor = ThreadPoolExecutor(max_workers=8)

@app.post("/convert", response_class=PlainTextResponse)
async def ocr_file(file: UploadFile = File(...)):
    try:
        if file:
            file_name = file.filename
            contents = await file.read()
            file_stream = BytesIO(contents)
            stream = DocumentStream(name=file_name, stream=file_stream)
            loop = asyncio.get_running_loop()
            convert_sync = partial(converter.convert, stream)
            conversion_result = await loop.run_in_executor(executor, convert_sync)
            doc = conversion_result.document
            md = doc.export_to_markdown()
            return md
        else:
            raise HTTPException(status_code=400, detail="Provide either a file")
    except Exception as e:
        _log.error(f"Conversion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Conversion failed")


@app.get("/")
async def root():
    """check health"""
    return {"status": "ok", "message": "Docling service is healthy"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8800"))
    
    uvicorn.run(app, host=host, port=port)