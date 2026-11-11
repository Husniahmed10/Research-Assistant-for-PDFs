"""
tracing/logfire_setup.py — Pydantic Logfire Configuration & Instrumentation
============================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import logfire and settings

Step 2:  Write setup_logfire(app: FastAPI) → None:
            - Called once in main.py during startup
            - Configure Logfire:
                logfire.configure(token=settings.LOGFIRE_TOKEN)
            - Instrument FastAPI automatically:
                logfire.instrument_fastapi(app)
                → This auto-traces every HTTP request (method, path, status, latency)
            - Instrument SQLAlchemy:
                logfire.instrument_sqlalchemy(engine)
                → Auto-traces every DB query
            - Instrument HTTPX:
                logfire.instrument_httpx()
                → Traces outbound HTTP calls (OpenAI, Cohere, Pinecone)

Step 3:  How to add manual spans in service files:
            with logfire.span("operation_name", key=value):
                # code here
            → Creates a named span with attributes visible in Logfire dashboard

Step 4:  How to add structured logs:
            logfire.info("PDF processed", pages=12, chunks=48, tokens=4320)
            logfire.error("Pinecone upsert failed", paper_id=paper_id, error=str(e))

WHAT LOGFIRE DASHBOARD SHOWS:
    - Request timeline (waterfall view)
    - Latency per step (PDF extraction, embedding, retrieval, reranking, LLM)
    - Error rates and stack traces
    - Token usage per request
    - Database query performance
"""

