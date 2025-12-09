# backend_server.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import logging
from typing import List, Any

# Import your RAG pipeline class (ensure rag.py exposes RAGPipeline)
from rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

API_KEY = os.environ.get("BACKEND_API_KEY", "change_me")

app = FastAPI(title="RAG Backend")

class Query(BaseModel):
    q: str

_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing RAGPipeline (this may take a while on first start)...")
        _pipeline = RAGPipeline()  # ensures ingest/chroma loaded
    return _pipeline

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/rag")
async def rag(query: Query, x_api_key: str | None = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    pipeline = get_pipeline()
    answer, docs = pipeline.ask(query.q)
    # docs may be LangChain Document objects; convert to serializable format
    sources = []
    for d in docs:
        try:
            meta = d.metadata if hasattr(d, "metadata") else {}
        except Exception:
            meta = {}
        sources.append({"text": getattr(d, "page_content", str(d)), "metadata": meta})
    return {"answer": answer, "sources": sources}
