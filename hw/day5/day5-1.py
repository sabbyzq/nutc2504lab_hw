import os
import csv
import uuid
import requests
import pandas as pd
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------- API 設定 ----------
API_URL = "https://ws-04.wade0426.me/embed"

def get_embedding(text: str) -> list:
    """使用 API 取得文字向量 (embedding)"""
    response = requests.post(
        API_URL,
        json={
            "texts": [text],
            "task_description": "檢索技術文件",
            "normalize": True
        }
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

# ---------- 資料路徑 ----------
STUDENT_ID = "1111132002"
DATA_DIR = "./day5"
QUESTION_FILE = "questions.csv"
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"

FIXED_CHUNK_SIZE = 300
SLIDING_CHUNK_SIZE = 300
SLIDING_OVERLAP = 100

# ---------- 連線 Qdrant ----------
QDRANT = QdrantClient(url="http://localhost:6333")

def ensure_collection(collection_name, vector_size=4096):
    """建立 Qdrant collection，如果不存在"""
    if not QDRANT.collection_exists(collection_name):
        QDRANT.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' 已建立")
    else:
        print(f"Collection '{collection_name}' 已存在")

# ---------- 讀取文字 ----------
def load_texts() -> Dict[str, str]:
    texts = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts[fname] = f.read()
    return texts

def load_questions():
    df = pd.read_csv(QUESTION_FILE)
    df = df.rename(columns={
        "q_id": "id",
        "questions": "question",
        "answer": "answer",
        "source": "source"
    })
    return df

# ---------- 切塊方法 ----------
def fixed_chunks(text: str, chunk_size: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def sliding_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def sentence_chunks(text: str) -> List[str]:
    # 直接以換行或句號切分
    import re
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------- 上傳 Qdrant ----------
def upload_chunks(collection_name: str, chunks: List[str], source_file: str):
    points = []
    for chunk in chunks:
        vector = get_embedding(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"source": source_file, "text": chunk}
        ))
    QDRANT.upsert(collection_name=collection_name, points=points)
    print(f"{len(points)} chunks 已上傳到 '{collection_name}'")

# ---------- 主程式 ----------
def main():
    texts = load_texts()
    questions = load_questions()
    print("CSV 欄位：", list(questions.columns))

    # 建立三個 collection
    ensure_collection("fixed_chunk")
    ensure_collection("sliding_chunk")
    ensure_collection("sentence_chunk")

    # 處理每個檔案
    for fname, content in texts.items():
        # 固定切塊
        fixed = fixed_chunks(content, FIXED_CHUNK_SIZE)
        upload_chunks("fixed_chunk", fixed, fname)

        # 滑動切塊
        sliding = sliding_chunks(content, SLIDING_CHUNK_SIZE, SLIDING_OVERLAP)
        upload_chunks("sliding_chunk", sliding, fname)

        # 語句切塊
        sentences = sentence_chunks(content)
        upload_chunks("sentence_chunk", sentences, fname)

if __name__ == "__main__":
    main()

