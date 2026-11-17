from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

# from enum import Enum
#
# class LlmTask(Enum):
#     ANSWER = "answer"
#     REWRITE = "rewrite"

def query_rewrite(query:str) -> str:
    rule="作为智能问题扩展器，将问题扩展或改为适合文本检索的格式，回答不附带无关内容"
    return ollama(rule,"",query)

def rag_answer(context_text:str,query:str) -> str:
    rule="作为智能文本检索助手，严格按照上下文回答，回答不附带无关内容，没有信息则回答没有足够资料，不知道"
    return ollama(rule,context_text,query)

def ollama(rule:str,context_text:str,query:str) -> str:
    llm = ChatOpenAI(
        model="qwen2.5:7b", # ollama里你pull下来的模型名字
        base_url="http://127.0.0.1:11434/v1", # 本机ollama服务地址
        temperature=0.5
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{ruler}"),
        ("human", "上下文：{context}\n问题：{question}")
    ])
    parser = StrOutputParser()
    chain = (
            prompt
            | llm
            | parser
    )
    answer = chain.invoke(
        {"ruler": rule, "context": context_text, "question": query})
    print(f"{answer}")
    return answer
