import os
import csv
import pdfplumber
from PIL import Image
import pytesseract
from docx import Document
import pandas as pd
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

vllm_hostname="ws-01.wade0426.me:443"
model_name=""
api_key=""

PDF_FILES = ["1.pdf", "2.pdf", "3.pdf"]
IMG_FILES = ["4.png"]
DOCX_FILES = ["5.docx"]

QUESTIONS_CSV = "questions.csv"        
GROUND_TRUTH_CSV = "questions_answer.csv" 
OUTPUT_CSV = "test_dataset.csv"


SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "you are chatgpt",
    "system prompt",
    "forget all rules",
    "do not answer"
]

def detect_prompt_injection(text: str) -> bool:
    text = text.lower()
    return any(pattern in text for pattern in SUSPICIOUS_PATTERNS)


def extract_text_from_pdf(pdf_path):
    text_all = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                text_all += text + "\n"
            else:
                text_all += pytesseract.image_to_string(page.to_image(resolution=300).original) + "\n"
    return text_all

def extract_text_from_img(img_path):
    return pytesseract.image_to_string(Image.open(img_path), lang="chi_tra+eng")

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs)


def llm(prompt: str) -> str:
    return "這是一個示例回答（請換成真實 LLM 回傳）"


def rag_answer(question, retrieved_docs):
    context = "\n".join([doc["text"] for doc in retrieved_docs])
    prompt = f"""
    請根據以下文件內容回答問題：
    {context}

    問題：{question}
    """
    answer = llm(prompt)
    source = retrieved_docs[0]["source"]
    return answer, source


def evaluate_with_deepeval(question, answer, ground_truth, contexts):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=contexts
    )

    metrics = [
        AnswerRelevancyMetric(),
        FaithfulnessMetric(),
        ContextualPrecisionMetric(),
        ContextualRecallMetric()
    ]

    print(f"\nDeepeval 評測結果 (問題: {question}):")
    for metric in metrics:
        metric.measure(test_case)
        print(metric.__class__.__name__, ":", metric.score)


def save_to_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "questions", "answer", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":

    documents = {}
    injection_results = {}

    for pdf in PDF_FILES:
        text = extract_text_from_pdf(pdf)
        documents[pdf] = text
        injection_results[pdf] = detect_prompt_injection(text)

    for img in IMG_FILES:
        text = extract_text_from_img(img)
        documents[img] = text
        injection_results[img] = detect_prompt_injection(text)

    for docx_file in DOCX_FILES:
        text = extract_text_from_docx(docx_file)
        documents[docx_file] = text
        injection_results[docx_file] = detect_prompt_injection(text)

    print("\n=== Prompt Injection 檢測結果 ===")
    for name, is_injection in injection_results.items():
        print(f"{name}: {'含有惡意提示詞' if is_injection else '安全'}")




    questions_df = pd.read_csv(QUESTIONS_CSV)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)

    csv_rows = []


    for idx, row in questions_df.iterrows():
        id = row['id']
        question = row['questions']

  
        retrieved_docs = [{"text": t, "source": s} for s, t in documents.items()]

        answer, source = rag_answer(question, retrieved_docs)
        gt_answer_row = ground_truth_df.loc[ground_truth_df["id"].astype(str) == str(id), "answer"]
        if not gt_answer_row.empty:
            gt_answer = gt_answer_row.values[0]
        else:
            gt_answer = "No ground truth found"
            print(f"警告: CSV 中找不到 q_id={id} 的答案")
        contexts = [d["text"] for d in retrieved_docs]
        evaluate_with_deepeval(question, answer, gt_answer, contexts)

  
        csv_rows.append({
            "id": id,
            "questions": question,
            "answer": answer,
            "source": source
        })


    save_to_csv(OUTPUT_CSV, csv_rows)
    print(f"\n已產生 CSV: {OUTPUT_CSV}")

