"""
core/config.py — Application Configuration & Settings
=======================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import BaseSettings from pydantic_settings
Step 2:  Define a Settings class that inherits from BaseSettings
Step 3:  Declare all settings fields with types and default values:
            - App settings: APP_NAME, APP_ENV, DEBUG, APP_HOST, APP_PORT
            - Upload settings: UPLOAD_DIR, MAX_FILE_SIZE_MB
            - OpenAI settings: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
            - Groq settings: GROQ_API_KEY, GROQ_MODEL
            - Portkey settings: PORTKEY_API_KEY, PORTKEY_VIRTUAL_KEY_OPENAI, PORTKEY_VIRTUAL_KEY_GROQ
            - Pinecone settings: PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT, PINECONE_DIMENSION
            - Cohere settings: COHERE_API_KEY
            - Database: DATABASE_URL
            - LangSmith: LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
            - Logfire: LOGFIRE_TOKEN
            - RAG settings: CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, TOP_K_RERANK, HYBRID_ALPHA
Step 4:  Add a model_config to tell pydantic to read from .env file automatically
Step 5:  Create a module-level `settings = Settings()` singleton instance
Step 6:  Other modules import `from core.config import settings` to access all configs
"""

