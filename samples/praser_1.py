from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

class Res(BaseModel):
    answer: str=Field(description="结果")
    des:str=Field(description="算式类型")

parser=PydanticOutputParser(pydantic_object=Res)
format_instructions=parser.get_format_instructions()

from langchain_core.prompts import ChatPromptTemplate

prompt_template=ChatPromptTemplate.from_template(
    partial_variables={"format_instructions":format_instructions},
    template="回答:{question}\n{format_instructions}"
)

prompt=ChatPromptTemplate.from_messages([
    ("system","严格按照(结果，算式类型)输出"),
    prompt_template,
])

from langchain_ollama import ChatOllama

llm=ChatOllama(
    model="qwen2.5:7b",  # ollama里你pull下来的模型名字
    base_url="http://127.0.0.1:11434",  # 本机ollama服务地址
    temperature=0.0
)

chain=prompt | llm | parser

out=chain.invoke({"question":"2*5"})

print(out)




