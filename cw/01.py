import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

API_URL = "https://ws-04.wade0426.me/embed"


client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(
        size=4096,
        distance=Distance.COSINE
    )
)


texts = [
    "人工智慧很有趣",
    "AI 可以幫助醫療診斷",
    "向量資料庫適合語意搜尋",
    "Qdrant 是一種向量資料庫",
    "大型語言模型很強大"
]


response = requests.post(API_URL, json={
    "texts": texts,
    "normalize": True,
    "batch_size": 32
})
response.raise_for_status()

result = response.json()
print("Embedding 維度：", result["dimension"])
#批次
points = []
for i, vec in enumerate(result["embeddings"]):
    points.append(
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "text": texts[i]
            }
        )
    )

client.upsert(
    collection_name="test_collection",
    points=points
)


query_text = ["AI有什麼好處"]

query_response = requests.post(API_URL, json={
    "texts": query_text,
    "normalize": True,
    "batch_size": 32
})
query_response.raise_for_status()

query_vector = query_response.json()["embeddings"][0]

search_result = client.query_points(
    collection_name="test_collection",
    query=query_vector,
    limit=3
)

print("\n搜尋結果:")
for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"相似度分數: {point.score:.4f}")
    print(f"內容: {point.payload['text']}")
    print("-" * 30)

