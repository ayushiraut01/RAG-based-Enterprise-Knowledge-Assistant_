"""
RAG Pipeline: Document ingestion, chunking, embedding, and retrieval.
Uses ChromaDB as vector store and sentence-transformers for embeddings.
"""

import os
import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME  = "knowledge_base"

CHUNK_SIZE    = 512   # characters per chunk
CHUNK_OVERLAP = 64    # overlap between consecutive chunks
TOP_K         = 5     # default number of chunks to retrieve


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping fixed-size chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 30:          # drop tiny tail fragments
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _doc_id(source: str, chunk_index: int) -> str:
    """Stable, collision-resistant ID for a chunk."""
    key = f"{source}::{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


# ──────────────────────────────────────────────
# Text extractors
# ──────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> list[dict]:
    """Returns list of {text, page} dicts — one per PDF page."""
    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i})
    return pages


def extract_text_from_docx(path: str) -> list[dict]:
    """Returns a single page dict with all paragraph text joined."""
    doc       = docx.Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page": 1}]


def extract_text_from_txt(path: str) -> list[dict]:
    """Returns a single page dict for plain-text / markdown files."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [{"text": text, "page": 1}]


# Map file extension → extractor function
EXTRACTORS: dict = {
    ".pdf":  extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt":  extract_text_from_txt,
    ".md":   extract_text_from_txt,
}


# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d chunks",
            COLLECTION_NAME,
            self.collection.count(),
        )

    # ── Ingestion ──────────────────────────────────────────────────────────
    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file into the vector store.
        Returns the number of *new* chunks added (skips duplicates).
        """
        path = Path(file_path)
        ext  = path.suffix.lower()

        if ext not in EXTRACTORS:
            logger.warning("Unsupported file type skipped: %s", ext)
            return 0

        pages    = EXTRACTORS[ext](str(path))
        filename = path.name
        added    = 0

        for page_data in pages:
            chunks = _chunk_text(page_data["text"])
            for i, chunk in enumerate(chunks):
                doc_id = _doc_id(f"{filename}::p{page_data['page']}", i)

                # Skip duplicates
                if self.collection.get(ids=[doc_id])["ids"]:
                    continue

                embedding = self.embedder.encode(chunk).tolist()
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source":      filename,
                        "page":        page_data["page"],
                        "chunk_index": i,
                    }],
                )
                added += 1

        logger.info("Ingested '%s' → %d new chunks added", filename, added)
        return added

    def ingest_directory(self, dir_path: str) -> int:
        """Recursively ingest all supported files in *dir_path*."""
        total = 0
        for p in Path(dir_path).rglob("*"):
            if p.suffix.lower() in EXTRACTORS and p.is_file():
                total += self.ingest_file(str(p))
        logger.info("Directory ingest complete — %d total chunks added", total)
        return total

    # ── Retrieval ──────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Return the *top_k* most semantically relevant chunks for *query*.
        Each chunk dict contains: text, source, page, score (cosine similarity).
        """
        if self.collection.count() == 0:
            logger.warning("retrieve() called on empty collection")
            return []

        q_embedding = self.embedder.encode(query).tolist()
        n           = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":   doc,
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", "?"),
                "score":  round(1 - dist, 4),   # distance → cosine similarity
            })

        return chunks

    # ── Stats ──────────────────────────────────────────────────────────────
    def collection_stats(self) -> dict:
        return {
            "total_chunks":  self.collection.count(),
            "embed_model":   EMBED_MODEL_NAME,
            "chunk_size":    CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
