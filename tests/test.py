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
import os
import logging

logging.basicConfig(level=logging.INFO)

_log = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
model="qwen2.5-vl-72b-instruct"
prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!"

source = "./dh.pdf"

api_ocr_options = ApiOcrOptions(
    url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=model,
    api_key=api_key,
    prompt=prompt,
    timeout=90, 
    scale=1.0,
)

api_vlm_options = ApiVlmOptions(
    url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    params=dict(model=model),
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

doc = converter.convert(source).document
_log.info(f"markdown: \n{doc.export_to_markdown()}")
