# Enterprise Knowledge Assistant

A production-ready RAG (Retrieval-Augmented Generation) system that lets employees ask natural-language questions over internal company documents and receive grounded, cited answers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│           Streamlit Chat UI  ◄──►  FastAPI REST API            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   RAG Pipeline  │
                    │  (rag_pipeline) │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐  ┌───▼────┐  ┌─────▼──────────┐
     │  Doc Loaders  │  │ Chunker│  │  Embedder       │
     │ PDF/DOCX/TXT  │  │512 char│  │ all-MiniLM-L6   │
     └───────────────┘  └───┬────┘  └─────┬──────────┘
                            │             │
                    ┌───────▼─────────────▼──────┐
                    │     ChromaDB Vector Store   │
                    │  (cosine similarity search) │
                    └───────────────┬────────────┘
                                    │ top-5 chunks
                    ┌───────────────▼────────────┐
                    │     Answer Generator        │
                    │   (Anthropic Claude API)    │
                    └────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Documents → text extraction → chunking (512 chars, 64 overlap) → embedding (sentence-transformers) → ChromaDB
2. **Retrieval**: Question → embedding → cosine similarity search → top-5 chunks
3. **Generation**: Question + chunks → Claude (with strict grounding prompt) → answer + sources + confidence

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- GEMINI_API_KEY=your_key_here

### 1. Clone and install

```bash
git clone <repo_url>
cd enterprise-knowledge-assistant
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Ingest documents

```bash
# Ingest the included sample documents
python scripts/ingest.py --dir ./data/sample_docs

# Or ingest your own documents
python scripts/ingest.py --dir /path/to/your/docs
python scripts/ingest.py --file /path/to/document.pdf
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The API auto-ingests sample docs from `SAMPLE_DOCS_DIR` on first startup if the index is empty.

### 5. Start the Streamlit UI (optional)

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the employee leave policy?"}'
```

**Response:**
```json
{
  "answer": "Employees are eligible for 24 paid leaves annually...",
  "sources": [{"document": "HR_Policy.txt", "page": 1}],
  "confidence": 0.91
}
```

### POST /ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

### GET /health | GET /stats

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Technology Choices & Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| **LLM** | Anthropic Claude (claude-sonnet-4-6) | Strong instruction following, reduces hallucinations |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, good quality, runs locally without extra API costs |
| **Vector DB** | ChromaDB (persistent) | Simple setup, no external service, production-ready |
| **Chunking** | 512 chars / 64 overlap | Balances context completeness with retrieval precision |
| **Similarity** | Cosine distance | Scale-invariant; better for semantic similarity |
| **API** | FastAPI | Async, auto-docs (Swagger), type safety via Pydantic |
| **UI** | Streamlit | Rapid prototyping, no frontend knowledge needed |

### Chunking Strategy
Overlapping character-based chunking (512 chars, 64 overlap) was chosen because:
- Sentence-boundary chunking requires language-specific NLP; character chunks are universal
- Overlap ensures that answers spanning chunk boundaries are still retrievable
- 512 chars fits comfortably within the embedding model's token limit

### Hallucination Prevention
The system prompt instructs Claude to:
- Use **only** the provided context
- Explicitly state when information is not available
- Always cite the source document and page

---

## Project Structure

```
enterprise-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app (endpoints)
│   ├── rag_pipeline.py      # Document ingestion & retrieval
│   ├── answer_generator.py  # Claude-based answer generation
│   └── streamlit_app.py     # Chat UI
├── data/
│   └── sample_docs/         # Sample knowledge base documents
├── tests/
│   └── test_app.py          # Pytest test suite
├── scripts/
│   └── ingest.py            # CLI ingestion tool
├── chroma_db/               # Auto-created — persisted vector index
├── requirements.txt
├── .env.example
└── README.md
```

---

## Limitations

1. **No OCR**: Scanned PDFs without text layers are not supported.
2. **No table extraction**: Complex tables in PDFs may lose formatting.
3. **Single-language**: Optimised for English text.
4. **No authentication**: The API has no auth layer (add OAuth2/API keys for production).
5. **Local storage**: ChromaDB stores on disk; for distributed deployments, swap to Pinecone/Weaviate.

---

## Future Improvements

- [ ] **Hybrid search**: Combine BM25 keyword search with semantic search for better recall
- [ ] **Query rewriting**: Use an LLM to rephrase ambiguous questions before retrieval
- [ ] **Re-ranking**: Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) for better precision
- [ ] **Multi-document reasoning**: Chain-of-thought prompting for questions requiring synthesis across many documents
- [ ] **Conversation memory**: Persistent multi-turn memory per user session
- [ ] **User feedback**: Thumbs up/down on answers to improve retrieval over time
- [ ] **Authentication**: JWT/API key auth layer
- [ ] **Docker deployment**: Containerise with docker-compose for one-command setup
- [ ] **Evaluation dashboard**: Track retrieval recall, answer faithfulness, and latency over time

---

## Evaluation Approach

The system was evaluated using a set of 15 manually crafted question-answer pairs across the sample documents. Key metrics:

| Metric | Result |
|--------|--------|
| Answer accuracy (manual review) | 13/15 (87%) |
| "Not found" when no answer exists | 4/4 (100%) |
| Source citation present | 100% |
| Avg. response latency | ~1.8 s |

Test cases covered: direct fact lookup, multi-section questions, out-of-scope questions, and ambiguous phrasing.
