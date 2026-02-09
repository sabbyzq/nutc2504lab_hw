import requests
from qdrant_client import QdrantClient
import pandas as pd
client = QdrantClient(url="http://localhost:6333")

API_URL = "https://ws-04.wade0426.me/embed"

def get_embedding(text: str) -> list:
    """使用 API 取得文字向量 (embedding)"""
    response = requests.post(
        API_URL,
        json={
            "texts": text,
            "task_description": "檢索技術文件",
            "normalize": True
        }
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

file= "questions.csv"
df = pd.read_csv(file)
texts = [file]
query_vector = get_embedding(texts)

# 執行搜尋
search_result = client.query_points(
    collection_name="fixed_chunk",
    query=query_vector,
    limit=3
)


#3.顯示結果
for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"內容: {point.payload['text']}")
    print(f"來源: {point.payload['source']}")
    print("---")

SERVER_URL = "https://hw-01.wade0426.me/submit_answer"


def submit_homework(q_id, answer):
    payload = {
        "q_id": [point.id],
        "student_answer": point.payload['text'],
        "source_file": point.payload['source']

    }
