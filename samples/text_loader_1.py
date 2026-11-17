from langchain_community.document_loaders import TextLoader

# 传入文件路径
loader = TextLoader("../data/test.txt", encoding="utf-8")
docs = loader.load()  # load()执行加载，返回Document列表

print(docs[0].page_content) # 打印文件文字
print(docs[0].metadata)     # 打印元信息（文件名路径）
