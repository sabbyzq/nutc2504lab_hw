import os
import csv
import uuid
import requests
import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

STUDENT_ID = "1111132002"
DATA_DIR = "./day5"
QUESTION_FILE = "questions.csv"
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"

API_URL = "https://ws-04.wade0426.me/embed"
API_KEY = ""

FIXED_CHUNK_SIZE = 300
SLIDING_CHUNK_SIZE = 300
SLIDING_OVERLAP = 100
SEMANTIC_THRESHOLD = 0.3


def load_texts() -> Dict[str, str]:
    texts = {}
    for i in range(1, 6):
        fname = f"data_0{i}.txt"
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            texts[fname] = f.read()
    return texts


def load_questions() -> pd.DataFrame:
    return pd.read_csv(QUESTION_FILE)


def fixed_chunking(text: str, size: int) -> List[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]


def sliding_window_chunking(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i+size])
    return chunks


def semantic_chunking(text: str) -> List[str]:
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        return paragraphs

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(paragraphs)

    chunks = []
    current_chunk = paragraphs[0]

    for i in range(1, len(paragraphs)):
        sim = cosine_similarity(vectors[i-1], vectors[i])[0][0]
        if sim < SEMANTIC_THRESHOLD:
            chunks.append(current_chunk)
            current_chunk = paragraphs[i]
        else:
            current_chunk += "\n" + paragraphs[i]

    chunks.append(current_chunk)
    return chunks


def retrieve_best_chunk(question: str, chunks: List[str]) -> str:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + chunks)
    sims = cosine_similarity(vectors[0:1], vectors[1:])[0]
    best_idx = sims.argmax()
    return chunks[best_idx]


def call_score_api(question: str, context: str) -> float:
    payload = {
        "question": question,
        "context": context
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("score", 0)

def build_csv(results: List[Dict]):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    texts = load_texts()
    questions = load_questions()
    results = []

    for _, q in questions.iterrows():
        q_id = q["id"]
        question = q["question"]

        for source, text in texts.items():
            # 固定切塊
            fixed_chunks = fixed_chunking(text, FIXED_CHUNK_SIZE)
            fixed_text = retrieve_best_chunk(question, fixed_chunks)
            fixed_score = call_score_api(question, fixed_text)

            results.append({
                "id": str(uuid.uuid4()),
                "q_id": q_id,
                "method": "固定大小",
                "retrieve_text": fixed_text,
                "score": fixed_score,
                "source": source
            })

            # 滑動視窗
            sliding_chunks = sliding_window_chunking(
                text, SLIDING_CHUNK_SIZE, SLIDING_OVERLAP
            )
            sliding_text = retrieve_best_chunk(question, sliding_chunks)
            sliding_score = call_score_api(question, sliding_text)

            results.append({
                "id": str(uuid.uuid4()),
                "q_id": q_id,
                "method": "滑動視窗",
                "retrieve_text": sliding_text,
                "score": sliding_score,
                "source": source
            })

            # 語意切塊
            semantic_chunks = semantic_chunking(text)
            semantic_text = retrieve_best_chunk(question, semantic_chunks)
            semantic_score = call_score_api(question, semantic_text)

            results.append({
                "id": str(uuid.uuid4()),
                "q_id": q_id,
                "method": "語意切塊",
                "retrieve_text": semantic_text,
                "score": semantic_score,
                "source": source
            })

    build_csv(results)
    print(" CSV 建立完成：", OUTPUT_CSV)


if __name__ == "__main__":
    main()
