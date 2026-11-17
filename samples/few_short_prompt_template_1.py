from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b",  # ollama里你pull下来的模型名字
    base_url="http://127.0.0.1:11434",  # 本机ollama服务地址
    temperature=0.0
)

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples=[
    {"input":"2+2","output":"4","description":"加法"},
    {"input":"2-1","output":"1","description":"减法"}
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human","算式:{input}"),
    ("ai","结果:{output},说明:{description}")
])

few_shot_msg_template=FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

prompt_template=ChatPromptTemplate.from_messages([
    ("system","你是一个{subject}大师"),
    few_shot_msg_template,
    ("user","{question}"),
])

from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()

chain=prompt_template | llm | parser

out=chain.invoke({"subject":"数学","question":"5*2"})

print(out)

