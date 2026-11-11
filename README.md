# Research Assistant for PDFs — Backend

## 🗂️ Folder Structure

```
backend/
├── main.py                        # FastAPI app entry point
├── requirements.txt               # All dependencies with pinned versions
├── .env                           # Secret keys (never commit!)
├── .env.example                   # Safe template for teammates
├── .gitignore
│
├── core/
│   ├── config.py                  # Pydantic settings (reads from .env)
│   └── dependencies.py            # FastAPI dependency injection (DB, Pinecone)
│
├── db/
│   ├── database.py                # SQLAlchemy async engine + session setup
│   ├── models.py                  # DB table definitions (Paper, ChatSession, Message)
│   └── crud.py                    # All DB query functions (create/read/update/delete)
│
├── api/
│   └── routes/
│       ├── papers.py              # POST /upload, GET /summary, DELETE /{id}
│       ├── chat.py                # POST /sessions, POST /message, SSE /stream
│       └── health.py              # GET /health (service status)
│
├── agents/
│   ├── state.py                   # LangGraph AgentState TypedDict
│   ├── graph.py                   # LangGraph graph with all nodes + edges
│   └── nodes/
│       ├── input_guard.py         # Node 1: NeMo input safety check
│       ├── query_analyzer.py      # Node 2: Classify query type
│       ├── query_rewriter.py      # Node 3: Improve query for retrieval
│       ├── multi_query.py         # Node 4: Generate 3 query variants
│       ├── hyde_generator.py      # Node 5: Hypothetical Document Embedding
│       ├── retriever.py           # Node 6: Hybrid search (dense + sparse BM25)
│       ├── reranker.py            # Node 7: Cohere reranker (top 20 → top 5)
│       ├── synthesizer.py         # Node 8: Generate answer via Portkey
│       ├── output_guard.py        # Node 9: NeMo output safety check
│       └── citation_formatter.py  # Node 10: Add page citations to answer
│
├── services/
│   ├── pdf_processor.py           # PDF parsing + semantic/fixed chunking
│   ├── embedder.py                # OpenAI embeddings (batch)
│   ├── pinecone_client.py         # Pinecone hybrid search + upsert
│   ├── reranker.py                # Cohere + FlashRank reranking
│   ├── portkey_client.py          # LLM gateway (OpenAI → Groq fallback)
│   └── summarizer.py              # Structured 6-section summary generation
│
├── guardrails/
│   ├── config.yml                 # NeMo Guardrails LLM + rails config
│   └── main.co                    # Colang flow definitions (jailbreak, off-topic)
│
├── tracing/
│   ├── logfire_setup.py           # Pydantic Logfire setup + FastAPI instrumentation
│   └── langsmith_setup.py         # LangSmith environment configuration
│
├── utils/
│   ├── file_utils.py              # File save, delete, validate helpers
│   └── text_utils.py              # Token counting, text cleaning, RRF fusion
│
└── uploads/                       # Uploaded PDFs stored here (gitignored)
    └── .gitkeep
```

## 🚀 Setup Instructions

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# → Fill in your API keys in .env

# 4. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. View API docs
# Open: http://localhost:8000/docs
```

## 🔄 Implementation Phases

| Phase | Tasks | Status |
|-------|-------|--------|
| **Phase 1** | FastAPI + PDF upload + Basic RAG | ⬜ |
| **Phase 2** | Hybrid Search + HyDE + Reranking | ⬜ |
| **Phase 3** | LangGraph Agentic Flow | ⬜ |
| **Phase 4** | NeMo Guardrails + Portkey + Tracing | ⬜ |
| **Phase 5** | Polish + Streaming + Citations | ⬜ |

