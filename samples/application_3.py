from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.1,
    format="json",          # 强制 JSON
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是技术文档编写者。只输出 JSON，包含 answer 字段，不要多余说明。"),
    ("user", "{input}")
])

chain = prompt | llm | JsonOutputParser()

q = "LangChain是什么？"
resp = chain.invoke({"input": q})
resp["question"] = q        # 不依赖模型复述

print("问题：", resp["question"])
print("回答：", resp["answer"])