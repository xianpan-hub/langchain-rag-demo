from langchain_community.document_loaders import TextLoader

from langchain_ollama import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.documents import Document

def load_text() -> list[Document]:
    loader = TextLoader("data/test.txt", encoding="utf-8")
    docs = loader.load()
    return docs

def load_pdf() -> list[Document]:
    loader = PyPDFLoader("data/demo.pdf")
    docs = loader.load()  # load()执行加载，返回Document列表
    return docs

def rag(query:str,docs:list[Document]) -> str:
    splitter = RecursiveCharacterTextSplitter(
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
    split_docs = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    vector_store = FAISS.from_documents(split_docs, embeddings)
    search_docs = vector_store.similarity_search(query, k=2)
    context_text = "\n".join([doc.page_content for doc in search_docs])
    return context_text












