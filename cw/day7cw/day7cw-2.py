from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter,PdfFormatOption


pdf_options=PdfPipelineOptions(
    do_ocr=False,

)
doc_converter=DocumentConverter(
    format_options={
        InputFormat.PDF:PdfFormatOption(pipeline_options=pdf_options) 
    }
) 
result=doc_converter.convert("example.pdf") 
print(result.document.export_to_markdown())