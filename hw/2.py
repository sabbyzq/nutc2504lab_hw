import json 
from typing import Annotatd, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

llm=ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="",
    temperature=0
)
class MeetingState(TypedDict):
    messages: Annotated[list, add_messages]
    transcript: str
    detailed_minutes: str
    summary: str

@tool
@tool
def asr_tool(audio_path: str) -> str:
    """
    將語音轉成逐字稿（作業版模擬）
    """
    return """
00:00:00 今天我們來討論文化 podcast 的經營方向。
00:00:05 目前最大的問題是內容產出不穩定。
00:00:10 希望能用 AI 來協助整理會議紀錄。
""".strip()
def minutes_taker_tool(transcript: str) -> str:
    """
    整理詳細逐字稿
    """
    return f"【會議逐字稿】\n{transcript}"
def summarizer_tool(transcript: str) -> str:
    """
    產生會議重點摘要
    """
    return """
【重點摘要】
- 主題：文化 podcast 經營
- 問題：內容產出不穩定
- 解法：導入 AI 自動化會議紀錄

【Action Items】
- 建立 ASR 流程
- 使用 LangGraph 管理節點
""".strip()
tools = [
    asr_tool,
    minutes_taker_tool,
    summarizer_tool
]

tool_node = ToolNode(tools)
def asr_node(state: MeetingState):
    result = asr_tool.invoke("meeting.wav")
    return {"transcript": result}
def minutes_node(state: MeetingState):
    result = minutes_taker_tool.invoke(state["transcript"])
    return {"detailed_minutes": result}
def summary_node(state: MeetingState):
    result = summarizer_tool.invoke(state["transcript"])
    return {"summary": result}
def writer_node(state: MeetingState):
    final_doc = f"""
==============================
智慧會議記錄
==============================

{state['detailed_minutes']}

------------------------------

{state['summary']}
"""
    print(final_doc)
    return {}
graph = StateGraph(MeetingState)

graph.add_node("asr", asr_node)
graph.add_node("minutes_taker", minutes_node)
graph.add_node("summarizer", summary_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("asr")

graph.add_edge("asr", "minutes_taker")
graph.add_edge("asr", "summarizer")

graph.add_edge("minutes_taker", "writer")
graph.add_edge("summarizer", "writer")

graph.add_edge("writer", END)

app = graph.compile()
if __name__ == "__main__":
    app.invoke({
        "messages": []
    })
