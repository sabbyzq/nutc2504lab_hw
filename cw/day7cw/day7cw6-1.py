import logging
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions,ResponseFormat
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

def olmocr2_vlm_options(
    model:str="olmocr",
    hostname_and_port:str="localhost:8000/v1",
    prompt:str="Convert this page to markdown.",
    max_tokens:int=4096,
    temperature:float=0.0,
    api_key:str="",)-> ApiVlmOptions:

    headers={}
    if api_key:
        headers["Authorization"]=f"Bearer{api_key}" 

    options=ApiVlmOptions(
        url=f"http://{hostname_and_port}/chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=120,
        scale=2.0,
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

def convert_pdf_to_markdown(
    input_pdf_path:str,
    output_md_path:str,
    vllm_hostname:str="localhost:8000",
    model_name:str="olmocr",):
        
    logging.basicConfig(level=logging.INFO)
    logger=logging.getLogger(__name__)

    input_path=Path(input_pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{input_pdf_path}")
    logger.info(f"開始處理 PDF：{input_pdf_path}")

    pipeline_options=VlmPipelineOptions(
        enable_remote_services=True
    )

    pipeline_options.vlm_options=olmocr2_vlm_options(
        model=model_name,
        hostname_and_port=vllm_hostname,
        prompt="Convert this page to clean,readable markdown format.",
        temperature=0.0,
    )

    doc_converter=DocumentConverter(
        format_options={
            InputFormat.PDF:PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    logger.info("正在轉換文件：")
    result=doc_converter.convert(input_pdf_path)

    markdown_content=result.document.export_to_markdown()

    output_path=Path(output_md_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(markdown_content,encoding="utf-8")

    logger.info(f"轉換完成！Markdown以儲存至：{output_md_path}")
    logger.info(f"總頁數：{len(result.document.pages)}")
    return markdown_content

def main():
    input_pdf="./sample_table.pdf"
    output_markdown="./sample_table.md"

    try:
        convert_pdf_to_markdown(
            input_pdf_path=input_pdf,
            output_md_path=output_markdown,
            vllm_hostname="https://ws-01.wade0426.me/v1/chat/completions",
            model_name='allenai/olmOCR-2-7B-1025-FP8',
        )
    except Exception as e:
        logging.error(f"轉換失敗：{e}")
        raise

if __name__=="__main__":
    main()
