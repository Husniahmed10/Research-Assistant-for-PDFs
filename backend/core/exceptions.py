"""Shared API error handling, including dependencies and response validation."""

import logfire
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException


def log_internal_error(request: Request, exc: Exception) -> None:
    """Record error location/type without request bodies, query strings or secrets."""
    route = request.scope.get("route")
    logfire.error("API request failed", method=request.method,
                  route=getattr(route, "path", "unmatched"), error_type=type(exc).__name__)


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log_internal_error(request, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code >= 500:
        log_internal_error(request, exc)
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": jsonable_encoder(exc.detail)}, headers=exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


class ErrorHandlingRoute(APIRoute):
    """Apply one try/except boundary to every route without changing its schema."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await original(request)
            except (HTTPException, RequestValidationError):
                raise  # Preserve explicit errors and FastAPI validation handling.
            except Exception as exc:
                return await internal_error_handler(request, exc)

        return handler
