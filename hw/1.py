import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm=ChatOpenAI(
    temperature=0,
    streaming=True
)


stylea=ChatPromptTemplate.from_template(
    "請以理性的風格撰寫一篇關於 {topic} 的貼文。"
)

styleb=ChatPromptTemplate.from_template(
    "請以有趣的風格，撰寫一篇關於 {topic} 的貼文。"
)

stylea_chain = stylea | llm
styleb_chain = styleb | llm

parallel_chain=RunnableParallel(
    professional=stylea_chain,
    casual=styleb_chain
)

def stream_output(topic):
    print("\n Streaming Output \n")
    for chunk in parallel_chain.stream({"topic": topic}):
        for key, value in chunk.items():
            print(f"[{key}] {value.content}", end="", flush=True)
    print("\n")

def batch_output(topic):
    print("\n Batch Output \n")
    start_time = time.time()
    result = parallel_chain.invoke({"topic": topic})
    end_time = time.time()

    for key, value in result.items():
        print(f"\n[{key}]\n{value.content}")

    print(f"\n Batch Processing Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    topic = input("請輸入主題 Topic:")

    stream_output(topic)
    batch_output(topic)

