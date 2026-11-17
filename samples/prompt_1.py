from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

template = "你是专业程序员\n对于{text}知识进行形象准确的描述"

prompt=PromptTemplate.from_template(template)

prompt_1=ChatPromptTemplate.from_messages([
    ("system",template),
    ("user","{question}")
])

llm=ChatOllama(
    model="qwen2.5:7b",  # ollama里你pull下来的模型名字
    base_url="http://127.0.0.1:11434",  # 本机ollama服务地址
    temperature=0.1
)

paser=StrOutputParser()

chain=prompt_1 | llm | paser

out=chain.invoke({"text":"编程","question":"什么是顺序表"})

# print(prompt)
# print("="*50)
# print(prompt.format(text="python"))

print(out)

