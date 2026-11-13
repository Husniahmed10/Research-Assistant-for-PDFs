"""Configure automatic LangGraph tracing before importing the graph/routes."""

import os

from core.config import settings


def setup_langsmith() -> None:
    """Respect the configured tracing switch; no key means no remote tracing."""
    enabled = settings.LANGCHAIN_TRACING_V2 and bool(settings.LANGCHAIN_API_KEY.strip())
    os.environ["LANGCHAIN_TRACING_V2"] = str(enabled).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
