"""
FastAPI application — Enterprise Knowledge Assistant API.

Endpoints:
  GET  /health   — liveness check
  GET  /stats    — vector store statistics
  POST /ask      — ask a question, get a grounded answer
  POST /ingest   — upload and index a document
"""

import os
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline, EXTRACTORS
from app.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# App-level singletons
# ──────────────────────────────────────────────
_pipeline:  RAGPipeline | None   = None
_generator: AnswerGenerator | None = None


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _generator

    logger.info("Starting up — initialising RAG pipeline…")
    _pipeline  = RAGPipeline()
    _generator = AnswerGenerator()

    # Auto-ingest sample docs when the index is empty
    sample_dir = os.getenv("SAMPLE_DOCS_DIR", "./data/sample_docs")
    if (
        _pipeline.collection_stats()["total_chunks"] == 0
        and Path(sample_dir).exists()
    ):
        logger.info("Index empty — ingesting sample docs from %s", sample_dir)
        _pipeline.ingest_directory(sample_dir)

    yield   # ← application runs here

    logger.info("Shutting down.")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG-based Q&A over internal company documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────
class AskRequest(BaseModel):
    question:             str
    top_k:                int = 5
    conversation_history: list[dict] | None = None


class Source(BaseModel):
    document: str
    page:     int | str


class AskResponse(BaseModel):
    answer:     str
    sources:    list[Source]
    confidence: float
    error:      str | None = None   # ← surface errors to the client


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.get("/stats", tags=["System"])
def stats():
    """Return vector store statistics."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready yet.")
    return _pipeline.collection_stats()


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(body: AskRequest):
    """Ask a question; returns a grounded answer with source citations."""
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if _pipeline is None or _generator is None:
        raise HTTPException(503, "Service not ready yet.")

    chunks = _pipeline.retrieve(body.question, top_k=body.top_k)
    result = _generator.generate(
        question=body.question,
        chunks=chunks,
        conversation_history=body.conversation_history,
    )
    return AskResponse(**result)


@app.post("/ingest", tags=["Documents"])
async def ingest(file: UploadFile = File(...)):
    """Upload a document and index it into the knowledge base."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready yet.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in EXTRACTORS:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {list(EXTRACTORS.keys())}",
        )

    # Write to a temp file, ingest, then clean up
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        count = _pipeline.ingest_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "filename":     file.filename,
        "chunks_added": count,
        "message":      f"Successfully indexed {count} chunks from '{file.filename}'.",
    }
