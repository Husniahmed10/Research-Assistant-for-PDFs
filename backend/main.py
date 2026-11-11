"""
main.py — FastAPI Application Entry Point
===========================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import FastAPI, CORS middleware, and all routers
Step 2:  Load settings from core/config.py
Step 3:  Initialize Logfire tracing (instrument FastAPI automatically)
Step 4:  Initialize LangSmith tracing (set environment variables)
Step 5:  Create the FastAPI app instance with metadata (title, version, docs URL)
Step 6:  Add CORS middleware — allow React frontend (localhost:3000) to call this API
Step 7:  Add a startup event handler:
            - Connect to SQLite DB and run migrations (create tables)
            - Initialize Pinecone index (create if not exists)
            - Log "Server ready" via Logfire
Step 8:  Add a shutdown event handler:
            - Close DB connection
            - Flush any pending Logfire/LangSmith traces
Step 9:  Include all API routers:
            - /api/v1/papers   → paper upload, list, delete
            - /api/v1/chat     → Q&A chatbot endpoint
            - /api/v1/health   → health check
Step 10: Add a root GET / route that returns API info
Step 11: If __name__ == "__main__": run uvicorn with hot-reload in dev mode
"""

