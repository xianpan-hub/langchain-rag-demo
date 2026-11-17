from fastapi import FastAPI

import ollama_service

import rag_service

def main() -> None:
    doc=rag_service.load_pdf()
    query=input()
    query=ollama_service.query_rewrite(query)
    context=rag_service.rag(query,doc)
    ollama_service.rag_answer(context,query)

app = FastAPI()

docs = rag_service.load_pdf()

@app.get("/")
def read_item(q: str):
    q=ollama_service.query_rewrite(q)
    context=rag_service.rag(q,docs)
    answer=ollama_service.rag_answer(context,q)
    return {"answer":answer}

if __name__ == "__main__":
    main()
