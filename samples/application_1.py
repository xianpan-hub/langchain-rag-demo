from dotenv import load_dotenv
load_dotenv()
# from langchain_openai import ChatOpenAI
#
# #llm = ChatOpenAI()
# llm = ChatOpenAI(model="deepseek-v4-flash"
#
# 本地Ollama版本（没有RAG，最简单问答）
from langchain_ollama import ChatOllama

# 实例化本地模型对象
llm = ChatOllama(
    model="qwen2.5:7b", # ollama里你pull下来的模型名字
    base_url="http://127.0.0.1:11434", # 本机ollama服务地址
    temperature=0.1
)

response = llm.invoke("什么是大模型？还有你的模型版本是多少？")
print(response)
print(response.content)
