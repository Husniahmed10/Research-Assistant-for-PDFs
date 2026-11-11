# Research Assistant for PDFs — Backend

## Hybrid retrieval

New uploads fit BM25 on that paper's chunk texts and upsert dense embeddings
alongside sparse values. The fitted parameters are saved atomically as
`uploads/bm25/<paper-id-hash>/bm25_params.json` and loaded on startup. Each paper
keeps separate parameters so later uploads do not change its retrieval behavior.
Standalone calls to `fit_bm25(corpus)` save `uploads/bm25_params.json`.

Chat uses hybrid retrieval. `HYBRID_ALPHA` controls query weighting: `1` uses
dense retrieval, `0` uses sparse retrieval, and the default `0.7` favors dense
similarity. Weighting is applied to query vectors before calling Pinecone.
Old papers without fitted parameters continue using dense search; delete and
re-upload a paper to generate its sparse vectors. Keep the existing dotproduct
index; no new index or embedding model is required. Deletion removes the paper's
saved encoder parameters after its vectors are deleted.

BM25 uses NLTK tokenization resources. On a fresh environment, provision them once:

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

Offline hybrid tests (real BM25 encoders, fake Pinecone index):

```powershell
python -m utils.test_hybrid
```

## Reranking and rank fusion helpers

`services.reranker.rerank_chunks(query, chunks, top_n)` tries Cohere's configured
model (default `rerank-english-v3.0`), then FlashRank if credentials are absent or
the API fails. It returns copies of the chunks sorted by `relevance_score` while
preserving their source metadata. Use a thread pool when calling it from an async
route. These helpers are available for later pipeline wiring; simple RAG currently
uses the top hybrid-search results directly.

FlashRank's first use needs a model download. Prepare the ignored local cache
before running offline or deploying without network access:

```powershell
python -c "from services.reranker import prepare_flashrank; prepare_flashrank()"
```

Keep `FLASHRANK_CACHE_DIR` (default `.cache/flashrank`) available on the machine
that runs inference. A fresh checkout does not contain these cached weights.

`utils.text_utils.apply_rrf(ranked_lists, k=60)` merges ranked string IDs or chunk
dictionaries. It sums `1 / (k + rank)` using ranks starting at 1, deduplicates IDs,
and adds `rrf_score` to dictionary results without replacing retrieval scores.

```powershell
python -m utils.test_reranking
```

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

