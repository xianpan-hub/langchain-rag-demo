from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../data/demo.pdf")
docs = loader.load()

# PDF会一页一个Document
for doc in docs:
    print("页码：", doc.metadata["page"])
    print(doc.page_content)
