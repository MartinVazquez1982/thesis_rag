import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from factories.embeddings_factory import get_embeddings
from factories.llm_factory import get_llm
from factories.vectorstore_factory import get_vectorstore
from services.indexing_service import IndexingService
from services.rag_service import RAGService

load_dotenv()

app = FastAPI(title="RAG Tesis API")

# Factories
EMB = get_embeddings()
LLM = get_llm()
VS = get_vectorstore()

DATA_PDF = os.path.join("data", "paper.pdf")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Services
INDEX = IndexingService(VS, EMB, CHUNK_SIZE, CHUNK_OVERLAP)


class QueryRequest(BaseModel):
    question: str
    k: int = 4
    
class QueryResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, Any]]

@app.on_event("startup")
def on_startup():
    try:
        count = INDEX.index_pdf(DATA_PDF, force=False)
        print(f"[RAG] {count} fragments have been indexed from {DATA_PDF}")
    except Exception as e:
        print(f"[RAG] failed to index on startup: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    out = RAGService.query(req.question, k=req.k)
    return QueryResponse(**out)
