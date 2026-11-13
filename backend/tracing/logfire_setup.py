"""Configure Logfire once and instrument each app/async database engine."""

from weakref import WeakSet

import logfire
from fastapi import FastAPI

from core.config import settings
from db.database import engine

_configured = False
_httpx_instrumented = False
_engines = WeakSet()


def setup_logfire(app: FastAPI) -> None:
    """Call before ASGI startup; absent credentials select local-only tracing."""
    global _configured, _httpx_instrumented
    if getattr(app.state, "logfire_instrumented", False):
        return
    if not _configured:
        token = settings.LOGFIRE_TOKEN.strip()
        logfire.configure(token=token or None, send_to_logfire=bool(token),
                          service_name=settings.APP_NAME, console=False)
        _configured = True
    # SQLAlchemy's instrumentation expects the sync proxy of an AsyncEngine.
    if engine.sync_engine not in _engines:
        logfire.instrument_sqlalchemy(engine=engine.sync_engine)
        _engines.add(engine.sync_engine)
    if not _httpx_instrumented:
        logfire.instrument_httpx()
        _httpx_instrumented = True
    logfire.instrument_fastapi(app)
    app.state.logfire_instrumented = True
