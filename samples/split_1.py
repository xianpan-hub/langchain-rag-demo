from langchain_community.document_loaders import PyMuPDFLoader

# 1.加载PDF
loader = PyMuPDFLoader("../data/demo.pdf")
docs = loader.load()
print(f"原始加载：一共 {len(docs)} 个Document，也就是 {len(docs)} 页")

# 2.初始化递归字符切分器
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

# 3.执行切分
split_docs = text_splitter.split_documents(docs)
print(f"切分完成：一共得到 {len(split_docs)} 个文本块chunk")

# 4.打印前3块，防止刷屏
for idx, chunk in enumerate(split_docs[:3]):
    print("-" * 60)
    print(f"【第 {idx+1} 块】来源页码：{chunk.metadata['page']}")
    print(f"块内容：{chunk.page_content}")
