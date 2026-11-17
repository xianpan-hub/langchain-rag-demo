from dotenv import load_dotenv
load_dotenv()

# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# llm = ChatOpenAI(model="deepseek-v4-flash")

# 本地Ollama版本（没有RAG，最简单问答）
from langchain_ollama import ChatOllama

# 实例化本地模型对象
llm = ChatOllama(
    model="qwen2.5:7b", # ollama里你pull下来的模型名字
    base_url="http://127.0.0.1:11434", # 本机ollama服务地址
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "您是世界级的技术文档编写者"),
    ("user", "{input}")
])

# 构建链条
chain = prompt | llm

# 执行链条
response = chain.invoke({"input": "大模型中的LangChain是什么?"})
print(response)
print(response.content)
