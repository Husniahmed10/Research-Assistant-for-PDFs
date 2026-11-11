"""Async OpenAI-primary/Groq-secondary chat through the Portkey gateway."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from portkey_ai import AsyncPortkey

from core.config import settings


def _configured(value: str) -> bool:
    return bool(value.strip()) and not any(
        placeholder in value.lower() for placeholder in ("your-", "your_", "...", "-here")
    )


def build_fallback_config() -> dict[str, Any]:
    """Build ordered targets from Model Catalog names, virtual keys or API keys.

    A model named @provider-slug/model uses Portkey Model Catalog. Otherwise,
    prefer a configured virtual key, falling back to provider API credentials.
    Never log this dictionary: raw provider credentials may be included.
    """
    targets = []
    for provider, model, virtual_key, api_key in (
        ("openai", settings.OPENAI_MODEL, settings.PORTKEY_VIRTUAL_KEY_OPENAI, settings.OPENAI_API_KEY),
        ("groq", settings.GROQ_MODEL, settings.PORTKEY_VIRTUAL_KEY_GROQ, settings.GROQ_API_KEY),
    ):
        target = {"override_params": {"model": model, "temperature": 0.1}}
        if model.startswith("@"):
            pass  # Provider identity is part of the Model Catalog model name.
        elif _configured(virtual_key):
            target["virtual_key"] = virtual_key
        elif _configured(api_key):
            target.update(provider=provider, api_key=api_key)
        else:
            raise ValueError(f"Configure {provider.upper()} provider credentials or a Model Catalog model")
        targets.append(target)
    return {"strategy": {"mode": "fallback"}, "targets": targets}


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("At least one chat message is required")
    for message in messages:
        if (not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()):
            raise ValueError("Messages require a system/user/assistant role and non-empty text")


def get_gateway_config() -> dict[str, Any] | str:
    """Prefer a saved config for workspaces that prohibit inline configurations.

    The saved config must define OpenAI first and Groq second, with provider
    credentials and model overrides configured in Portkey's dashboard.
    """
    if _configured(settings.PORTKEY_CONFIG_ID):
        return settings.PORTKEY_CONFIG_ID
    return build_fallback_config()


@asynccontextmanager
async def _gateway() -> AsyncIterator[AsyncPortkey]:
    if not _configured(settings.PORTKEY_API_KEY):
        raise ValueError("Configure PORTKEY_API_KEY before making gateway requests")
    config = get_gateway_config()
    # Sharing the HTTP client also closes the pinned SDK's internal OpenAI
    # transport when this request or stream ends.
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http_client:
        async with AsyncPortkey(
            api_key=settings.PORTKEY_API_KEY,
            config=config,
            http_client=http_client,
            request_timeout=60_000,
        ) as gateway:
            yield gateway


async def chat_completion(
    messages: list[dict[str, Any]], stream: bool = False
) -> str:
    """Return the full reply. With stream=True, collect streamed text into a string.

    For incremental delivery, iterate stream_completion(messages) instead.
    Gateway/provider failures propagate; no fabricated fallback answer is returned.
    """
    _validate_messages(messages)
    if stream:
        return "".join([token async for token in stream_completion(messages)]).strip()
    async with _gateway() as gateway:
        response = await gateway.chat.completions.create(
            model=settings.OPENAI_MODEL, messages=messages, stream=False, max_tokens=2048,
        )
        if not response.choices:
            raise RuntimeError("Gateway returned no completion choices")
        choice = response.choices[0]
        if choice.message is None:
            raise RuntimeError("Gateway returned no message")
        if getattr(choice.message, "refusal", None):
            raise RuntimeError("The model declined the request")
        if choice.finish_reason != "stop":
            raise RuntimeError("Chat completion did not finish successfully")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Gateway returned an empty text completion")
        return content.strip()


async def stream_completion(messages: list[dict[str, Any]]) -> AsyncIterator[str]:
    """Yield non-empty text deltas and close the transport when finished.

    Mid-stream errors propagate. Do not replay a stream after emitting text,
    because doing so can duplicate content or splice answers from two models.
    """
    _validate_messages(messages)
    async with _gateway() as gateway:
        stream = await gateway.chat.completions.create(
            model=settings.OPENAI_MODEL, messages=messages, stream=True, max_tokens=2048,
        )
        emitted = False
        completed = False
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason not in (None, "stop"):
                    raise RuntimeError("Stream generation did not finish successfully")
                if choice.finish_reason == "stop":
                    completed = True
                content = getattr(choice.delta, "content", None)
                if isinstance(content, str) and content:
                    emitted = True
                    yield content
        finally:
            await stream.aclose()
        if not emitted or not completed:
            raise RuntimeError("Gateway returned an empty or incomplete stream")
