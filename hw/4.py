
import requests
import json
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


SEARXNG_URL = "https://puli-8080.huannago.com/search"

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="",
    temperature=0
)
class AgentState(TypedDict):
    input: str
    knowledge_base: str
    search_queries: List[str]
    is_sufficient: bool
    cache_hit: bool



def search_searxng(query: str, limit: int = 3):
    print(f" 搜尋中:{query}")
    params = {
        "q": query,
        "format": "json",
        "language": "zh-TW"
    }
    try:
        r = requests.get(SEARXNG_URL, params=params, timeout=10)
        data = r.json()
        results = data.get("results", [])
        return results[:limit]
    except Exception as e:
        print(" 搜尋失敗:", e)
        return []



def check_cache_node(state: AgentState):
    print("\n[1] 快速檢查（是否已有答案）")
    return {
        "cache_hit": False,
        "knowledge_base": ""
    }


def planner_node(state: AgentState):
    print("[2] Planner:判斷是否需要查資料")

    prompt = f"""
使用者問題：{state['input']}

如果這是常識、定義、推理題，
不需要即時資料，請回答 YES。
只有在「必須查網路」時回答 NO。
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    decision = "YES" in res.content.upper()

    print(" 是否需要搜尋:", "否" if decision else "是")
    return {"is_sufficient": decision}


def query_gen_node(state: AgentState):
    print("[3] Query Generator:產生搜尋關鍵字")

    prompt = f"""
請將下列問題轉成「最適合搜尋的關鍵字」:
問題：{state['input']}
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    query = res.content.strip()

    print(" 搜尋關鍵字:", query)
    return {"search_queries": [query]}


def search_tool_node(state: AgentState):
    print("[4] Search Tool:文字搜尋 + 摘要")

    query = state["search_queries"][0]
    results = search_searxng(query, limit=2)

    kb = state.get("knowledge_base", "")

    if not results:
        kb += "\n(找不到相關資料）"
    else:
        for r in results:
            kb += f"\n標題：{r.get('title','')}\n"
            kb += f"摘要：{r.get('content','')[:300]}\n"
            kb += f"來源：{r.get('url','')}\n"

    return {"knowledge_base": kb}


def final_answer_node(state: AgentState):
    print("[5] Final Answer產生最終回答")

    prompt = f"""
問題：{state['input']}

可用資訊：
{state.get('knowledge_base','')}

請「使用繁體中文」給出清楚、完整的回答。
"""
    try:
        res = llm.invoke([HumanMessage(content=prompt)])

        print("\n====================")
        print(" AI 回答：\n")
        print(res.content)
        print("====================\n")

    except Exception as e:
        print("\n LLM 回應逾時或失敗，啟用備援回答\n")

        fallback_answer = f"""
    （系統備援回答）

    「{state['input']}」通常可能的原因包括：
    1. 市場供需變化
    2. 投資人情緒（恐慌、利空消息）
    3. 總體經濟因素（升息、通膨）
    4. 公司基本面轉弱
    5. 突發事件或政策影響

    這是基於一般金融常識的整理，非即時投資建議。
    """

        print("====================")
        print(fallback_answer)
        print("====================\n")
    return state


workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    lambda s: "final_answer" if s["cache_hit"] else "planner",
    {"final_answer": "final_answer", "planner": "planner"}
)

workflow.add_conditional_edges(
    "planner",
    lambda s: "final_answer" if s["is_sufficient"] else "query_gen",
    {"final_answer": "final_answer", "query_gen": "query_gen"}
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()


if __name__ == "__main__":
    print("=== 課後實戰:自動查證 AI ===")
    print("輸入 exit / q 離開")

    while True:
        user_input = input("\n請輸入你的問題:")
        if user_input.lower() in ["exit", "q"]:
            break

        app.invoke({
            "input": user_input,
            "knowledge_base": "",
            "search_queries": [],
            "is_sufficient": False,
            "cache_hit": False
        })
print(app.get_graph().draw_ascii())