"""FastAPI routes for RAG API."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
from pathlib import Path

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.store import create_vector_store, load_vector_store
from src.generation.rag_chain import ask

app = FastAPI(title="RAG Document Assistant API", version="0.1.0")


class QuestionRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        chunks = chunk_documents(docs)
        create_vector_store(chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": f"Processed {len(chunks)} chunks from {file.filename}"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents."""
    try:
        store = load_vector_store()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No documents found: {e}")

    result = ask(store, request.question, k=request.k)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
