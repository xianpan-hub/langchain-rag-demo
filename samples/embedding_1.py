from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ========== 1.加载PDF ==========
loader = PyMuPDFLoader("../data/demo.pdf")
docs = loader.load()
print(f"原始加载：一共 {len(docs)} 个Document，也就是 {len(docs)} 页")

# ========== 2.文本切分 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        "。",
        "，",
        " ",
        ""
    ]
)
split_docs = text_splitter.split_documents(docs)
print(f"切分完成：一共得到 {len(split_docs)} 个文本块chunk")

# ========== 3. Ollama Embedding 向量模型 ==========
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# ========== 4.构建FAISS向量库 ==========
vector_store = FAISS.from_documents(split_docs, embeddings)
print("✅ 本地向量库构建完成！")

# ========== 5. 检索 ==========
query = "Release Date是什么时候？"
search_docs = vector_store.similarity_search(query, k=2)

# 把检索出来的多个文本块拼接成上下文字符串
context_text = "\n".join([doc.page_content for doc in search_docs])

# ========== 6. 初始化对话大模型 qwen2.5:7b ==========
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)

# ========== 7. 写提示词模板：规定大模型只能根据给的上下文回答 ==========
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个文档问答助手，严格只使用提供的上下文内容回答问题。如果上下文没有答案，就说不知道，不要编造。"),
    ("human", "上下文：{context}\n问题：{question}")
])

# 组装prompt
prompt = prompt_template.invoke({"context": context_text, "question": query})

# ========== 8. 发给大模型，拿到回答 ==========
response = llm.invoke(prompt)

print("\n===== 检索拿到的上下文 =====")
print(context_text)
print("\n===== Qwen 的回答 =====")
print(response.content)

# ========== 保存向量索引 ==========
vector_store.save_local("./faiss_ollama_index")
print("\n✅ 向量索引保存到 faiss_ollama_index 文件夹")
