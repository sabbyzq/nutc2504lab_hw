import os
import pandas as pd
import requests
from rank_bm25 import BM25Okapi
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from openai import OpenAI

client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key=""
)

embedder = SentenceTransformer("intfloat/multilingual-e5-base")


docs = open("qa_data.txt", encoding="utf-8").read().split("\n\n")
tokenized_docs = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized_docs)
doc_embeddings = embedder.encode(docs, convert_to_tensor=True)

def rewrite_query(history, query):
    prompt = f"""
你是客服助手，根據對話歷史，改寫使用者問題成明確單一問題。
對話歷史：
{history}

問題：{query}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

def hybrid_search(query, top_k=5):
    q_tokens = query.split()
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    q_emb = embedder.encode(query, convert_to_tensor=True)
    emb_scores = util.cos_sim(q_emb, doc_embeddings)[0]
    emb_top = emb_scores.topk(top_k).indices.tolist()

    return list(set(bm25_top + emb_top))

def get_embedding(text: str) -> list:
    response = requests.post(
        "http://ws-04.wade0426.me/embed",
        json={"texts": [text]}
    )
    return response.json()["embeddings"][0]


def rerank(query, doc_ids):
    scored = []
    for i in doc_ids:
        score = util.cos_sim(
            embedder.encode(query, convert_to_tensor=True),
            doc_embeddings[i]
        )
        scored.append((i, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [docs[i] for i, _ in scored[:3]]

def answer_llm(context, question):
    prompt = f"""
你是台水 AI 客服，只能根據以下資料回答，不可編造。

資料：
{context}

問題：
{question}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

metrics = [
    FaithfulnessMetric(),
    AnswerRelevancyMetric(),
]

questions = pd.read_csv("questions.csv")
answers = pd.read_csv("questions_answer.csv")

rows = []
history = ""

for i, q in enumerate(questions["question"]):
    rewritten = rewrite_query(history, q)
    doc_ids = hybrid_search(rewritten)
    contexts = rerank(rewritten, doc_ids)
    answer = answer_llm("\n".join(contexts), rewritten)

    test_case = LLMTestCase(
        input=q,
        actual_output=answer,
        expected_output=answers.iloc[i]["answer"],
        retrieval_context=contexts
    )

    scores = {}
    for m in metrics:
        m.measure(test_case)
        scores[m.__class__.__name__] = m.score

    rows.append([
        i + 1,
        q,
        answer,
        scores["FaithfulnessMetric"],
        scores["AnswerRelevancyMetric"],
        scores["ContextualRecallMetric"],
        scores["ContextualPrecisionMetric"],
        scores["ContextualRelevancyMetric"],
    ])

    history += f"Q:{q}\nA:{answer}\n"

df = pd.DataFrame(rows, columns=[
    "q_id", "questions", "answer",
    "Faithfulness（忠實度）",
    "Answer_Relevancy（答案相關性）",
    "Contextual_Recall（上下文召回率）",
    "Contextual_Precision（上下文精確度）",
    "Contextual_Relevancy（上下文相關性）"
])

df.to_csv("day6_HW_questions.csv", index=False, encoding="utf-8-sig")
print(" Day6 作業完成")
