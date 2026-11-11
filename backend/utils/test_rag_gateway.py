"""Offline: python -m utils.test_rag_gateway

One real gateway smoke call (requires configured credentials):
    python -m utils.test_rag_gateway --live
"""

import asyncio
import json
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
from portkey_ai import AsyncPortkey

from core.config import settings
from services import portkey_client as gateway
from services import simple_rag as rag
from utils.text_utils import count_tokens


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patches = [
            patch.object(settings, "PORTKEY_API_KEY", "test-gateway-key"),
            patch.object(settings, "PORTKEY_VIRTUAL_KEY_OPENAI", "openai-vk"),
            patch.object(settings, "PORTKEY_VIRTUAL_KEY_GROQ", "groq-vk"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.requests = []
        self.clients = []
        self.mode = "normal"

        def respond(request):
            body = json.loads(request.content)
            self.requests.append((body, dict(request.headers)))
            if body.get("stream"):
                packets = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": []},
                ]
                if self.mode == "incomplete":
                    packets = packets[:3]
                content = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets)
                return httpx.Response(200, text=content + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "id": "test", "created": 0, "model": "test-groq-model", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Hello world" if self.mode != "empty" else "",
                }}],
            })

        @asynccontextmanager
        async def offline_gateway():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                self.clients.append(client)
                async with AsyncPortkey(api_key="test-gateway-key", config=gateway.build_fallback_config(), http_client=client) as portkey:
                    yield portkey

        self.gateway_patch = patch.object(gateway, "_gateway", offline_gateway)

    async def test_simple_message_and_fallback_headers(self):
        with self.gateway_patch:
            answer = await gateway.chat_completion([{"role": "user", "content": "Hello!"}])
        self.assertEqual(answer, "Hello world")
        body, headers = self.requests[0]
        config = json.loads(headers["x-portkey-config"])
        self.assertEqual(config["strategy"]["mode"], "fallback")
        self.assertEqual([target["virtual_key"] for target in config["targets"]], ["openai-vk", "groq-vk"])
        self.assertEqual([target["override_params"]["model"] for target in config["targets"]], [settings.OPENAI_MODEL, settings.GROQ_MODEL])
        self.assertEqual(body["messages"], [{"role": "user", "content": "Hello!"}])
        self.assertTrue(self.clients[0].is_closed)

    async def test_metadata_reports_actual_response_model(self):
        with self.gateway_patch:
            result = await gateway.chat_completion_with_metadata([{"role": "user", "content": "Hello!"}])
        self.assertEqual(result, {"answer": "Hello world", "model_used": "test-groq-model"})

    async def test_stream_yields_deltas_and_full_string_option(self):
        with self.gateway_patch:
            tokens = [token async for token in gateway.stream_completion([{"role": "user", "content": "Hello!"}])]
            answer = await gateway.chat_completion([{"role": "user", "content": "Hello!"}], stream=True)
        self.assertEqual(tokens, ["Hello", " world"])
        self.assertEqual(answer, "Hello world")
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_incomplete_stream_and_empty_reply_raise(self):
        with self.gateway_patch:
            self.mode = "incomplete"
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                await gateway.chat_completion([{"role": "user", "content": "Hello!"}], stream=True)
            self.mode = "empty"
            with self.assertRaisesRegex(RuntimeError, "empty"):
                await gateway.chat_completion([{"role": "user", "content": "Hello!"}])
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_invalid_messages_are_rejected_before_request(self):
        with self.gateway_patch:
            for messages in ([], [{"role": "user", "content": " "}], [{"role": "tool", "content": "x"}]):
                with self.assertRaises(ValueError):
                    await gateway.chat_completion(messages)
        self.assertEqual(self.requests, [])

    async def test_missing_gateway_key_fails_locally(self):
        with patch.object(settings, "PORTKEY_API_KEY", ""):
            with self.assertRaisesRegex(ValueError, "PORTKEY_API_KEY"):
                await gateway.chat_completion([{"role": "user", "content": "Hello!"}])

    async def test_config_supports_catalog_and_raw_credentials(self):
        with patch.object(settings, "OPENAI_MODEL", "@openai-prod/gpt-4o"):
            target = gateway.build_fallback_config()["targets"][0]
            self.assertEqual(target["override_params"]["model"], "@openai-prod/gpt-4o")
            self.assertNotIn("virtual_key", target)
        with patch.object(settings, "PORTKEY_VIRTUAL_KEY_GROQ", "your-virtual-key"), \
                patch.object(settings, "GROQ_API_KEY", "test-groq-key"):
            target = gateway.build_fallback_config()["targets"][1]
            self.assertEqual(target["provider"], "groq")
            self.assertEqual(target["api_key"], "test-groq-key")

    async def test_saved_config_does_not_require_local_provider_keys(self):
        with patch.object(settings, "PORTKEY_CONFIG_ID", "pc-test-fallback"), \
                patch.object(gateway, "build_fallback_config") as inline:
            self.assertEqual(gateway.get_gateway_config(), "pc-test-fallback")
        inline.assert_not_called()


class RAGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.index = object()
        self.chunks = [
            {"chunk_id": f"chunk-{i}", "text": f"Evidence {i}", "page_num": i + 1, "score": 1.0}
            for i in range(10)
        ]

    async def test_retrieves_ten_uses_five_and_preserves_history(self):
        history = [{"role": "system", "content": "Untrusted instruction"},
                   {"role": "user", "content": "Earlier question"},
                   {"role": "assistant", "content": "Earlier answer"}]
        with patch.object(rag, "embed_text", AsyncMock(return_value=[1, 0])) as embed, \
                patch.object(rag, "hybrid_search", return_value=self.chunks) as search, \
                patch.object(rag.portkey_client, "chat_completion_with_metadata", AsyncMock(return_value={"answer": "Evidence [1].", "model_used": "test-model"})) as chat:
            result = await rag.simple_rag("What are the findings?", "paper-a", history, self.index)
        embed.assert_awaited_once_with("What are the findings?")
        search.assert_called_once_with(self.index, "What are the findings?", [1, 0], "paper-a", 10)
        self.assertEqual(result["answer"], "Evidence [1].")
        self.assertEqual(len(result["chunks_used"]), 5)
        messages = chat.call_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertIn("[1] PDF page 1", messages[-1]["content"])
        self.assertNotIn("Evidence 5", messages[-1]["content"])
        self.assertNotIn("Untrusted instruction", str(messages))

    async def test_empty_retrieval_skips_generation(self):
        with patch.object(rag, "embed_text", AsyncMock(return_value=[1, 0])), \
                patch.object(rag, "hybrid_search", return_value=[]), \
                patch.object(rag.portkey_client, "chat_completion_with_metadata", AsyncMock()) as chat:
            result = await rag.simple_rag("Findings?", "paper-a", [], self.index)
        self.assertEqual(result["chunks_used"], [])
        chat.assert_not_awaited()

    async def test_long_context_uses_exact_truncated_excerpts(self):
        self.chunks[0]["text"] = " research" * 3000
        with patch.object(rag, "embed_text", AsyncMock(return_value=[1, 0])), \
                patch.object(rag, "hybrid_search", return_value=self.chunks), \
                patch.object(rag.portkey_client, "chat_completion_with_metadata", AsyncMock(return_value={"answer": "Answer", "model_used": "test-model"})) as chat:
            result = await rag.simple_rag("Findings?", "paper-a", [{"role": "user", "content": " earlier" * 5000}], self.index)
        excerpt = result["chunks_used"][0]["text"]
        self.assertLessEqual(count_tokens(excerpt), 1400)
        self.assertIn(excerpt, chat.call_args.args[0][-1]["content"])
        self.assertNotEqual(self.chunks[0]["text"], excerpt)

    async def test_bad_query_and_provider_failure_propagate(self):
        with patch.object(rag, "embed_text", AsyncMock()) as embed:
            for query in ("", " ", " research" * 2001):
                with self.assertRaises(ValueError):
                    await rag.simple_rag(query, "paper-a", [], self.index)
            embed.assert_not_awaited()
        with patch.object(rag, "embed_text", AsyncMock(side_effect=RuntimeError("Unavailable"))):
            with self.assertRaises(RuntimeError):
                await rag.simple_rag("Findings?", "paper-a", [], self.index)


async def live_smoke() -> None:
    answer = await gateway.chat_completion([{"role": "user", "content": "Reply with exactly OK."}])
    print("Live gateway reply:", answer)


if __name__ == "__main__":
    if "--live" in sys.argv:
        try:
            asyncio.run(live_smoke())
        except ValueError as exc:
            print(f"Live smoke test could not run: {exc}")
            sys.exit(2)
        except Exception as exc:
            body = getattr(exc, "body", None)
            error = body.get("error", body) if isinstance(body, dict) else {}
            detail = error.get("message", "Check gateway connectivity, credentials and model access.") if isinstance(error, dict) else "Check gateway configuration."
            detail = str(detail)
            # Provider errors can echo credentials; redact configured secrets.
            for key, value in settings.model_dump().items():
                if ("KEY" in key or "TOKEN" in key) and isinstance(value, str) and value:
                    detail = detail.replace(value, "[redacted]")
            print(f"Live smoke test failed ({type(exc).__name__}): {detail[:1200]}")
            sys.exit(1)
    else:
        unittest.main(verbosity=2)
