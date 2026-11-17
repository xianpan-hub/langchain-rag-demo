import os
import time
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ============================================================
# 1、网页加载（静态博客，WebBaseLoader 直接抓）
# ============================================================
print("\n" + "=" * 70)
print("【第 1 步】网页加载")
print("=" * 70)

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

t0 = time.time()
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_path=URL)
docs = loader.load()
print(f"✅ 加载完成，耗时 {time.time() - t0:.1f} 秒")
print(f"文档数：{len(docs)}，正文长度：{len(docs[0].page_content)} 字符")
print(f"正文前 300 字：\n{docs[0].page_content[:300]}")

if len(docs[0].page_content.strip()) < 100:
    raise RuntimeError("正文过短，抓取失败。")

# ============================================================
# 2、文档切分
# ============================================================
print("\n" + "=" * 70)
print("【第 2 步】文档切分")
print("=" * 70)

from langchain_text_splitters import RecursiveCharacterTextSplitter

t0 = time.time()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
print(f"✅ 切分完成，耗时 {time.time() - t0:.1f} 秒，块数：{len(documents)}")

# ============================================================
# 3、向量嵌入模型
# ============================================================
print("\n" + "=" * 70)
print("【第 3 步】向量嵌入模型")
print("=" * 70)

from langchain_ollama import OllamaEmbeddings

t0 = time.time()
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)
dim = len(embeddings.embed_query("test"))
print(f"✅ 嵌入维度：{dim}，耗时 {time.time() - t0:.1f} 秒")

# ============================================================
# 4、本地大模型
# ============================================================
print("\n" + "=" * 70)
print("【第 4 步】本地大模型")
print("=" * 70)

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.1,
    num_ctx=2048,
    num_predict=512,
)
print("✅ ChatOllama 就绪")

# ============================================================
# 5、构建 FAISS 向量库
# ============================================================
print("\n" + "=" * 70)
print("【第 5 步】构建 FAISS 向量库")
print("=" * 70)

from langchain_community.vectorstores import FAISS

print(f"开始对 {len(documents)} 个块嵌入并构建 FAISS...")
t0 = time.time()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever(search_kwargs={"k": 2})
print(f"✅ FAISS 就绪，耗时 {time.time() - t0:.1f} 秒")

# ============================================================
# 6、RAG 链组装
# ============================================================
print("\n" + "=" * 70)
print("【第 6 步】组装 RAG 链")
print("=" * 70)

prompt = ChatPromptTemplate.from_template("""
你**只能**使用下面【上下文】里的信息回答用户问题。
禁止使用你自身预训练知识库，绝对不要编造不存在的信息。
如果上下文没有相关信息，直接回复：不知道。

上下文：{context}
问题：{question}
""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG 链就绪")

# ============================================================
# 7、提问测试
# ============================================================
print("\n" + "=" * 70)
print("【第 7 步】提问测试")
print("=" * 70)

query = "什么是LLM Agent?"
print(f"\n📝 提问：{query}")

t0 = time.time()
hit_docs = retriever.invoke(query)
print(f"\n🔍 检索到 {len(hit_docs)} 个 chunk，耗时 {time.time() - t0:.1f} 秒：")
for i, d in enumerate(hit_docs):
    print(f"\n--- 命中 {i+1} ---")
    print(d.page_content[:400])

t0 = time.time()
answer = rag_chain.invoke(query)
print(f"\n🤖 RAG 回答（耗时 {time.time() - t0:.1f} 秒）：\n{answer}")