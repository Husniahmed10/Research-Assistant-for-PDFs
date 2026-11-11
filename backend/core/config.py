"""Typed settings loaded from .env when running from the backend directory."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shared configuration; environment variables override the .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application and uploads
    APP_NAME: str = "Research Assistant for PDFs"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50

    # Language models and embeddings
    OPENAI_API_KEY: str = Field(default="", repr=False)
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GROQ_API_KEY: str = Field(default="", repr=False)
    GROQ_MODEL: str = "llama-3.1-70b-versatile"

    # Gateway
    PORTKEY_API_KEY: str = Field(default="", repr=False)
    PORTKEY_VIRTUAL_KEY_OPENAI: str = Field(default="", repr=False)
    PORTKEY_VIRTUAL_KEY_GROQ: str = Field(default="", repr=False)

    # Vector storage and reranking
    PINECONE_API_KEY: str = Field(default="", repr=False)
    PINECONE_INDEX_NAME: str = "research-assistant"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_DIMENSION: int = 1536
    COHERE_API_KEY: str = Field(default="", repr=False)

    # Persistence
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./research_assistant.db", repr=False
    )

    # Observability
    LANGCHAIN_API_KEY: str = Field(default="", repr=False)
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "research-assistant"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LOGFIRE_TOKEN: str = Field(default="", repr=False)

    # Retrieval
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    HYBRID_ALPHA: float = 0.7


settings = Settings()

