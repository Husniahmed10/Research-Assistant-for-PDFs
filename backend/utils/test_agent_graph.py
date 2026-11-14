"""Run from backend: python -m utils.test_agent_graph.

Tests compiled graph routing with model, guardrail and vector-service mocks.
"""

import copy
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langsmith import tracing_context

from agents.graph import agent
from agents.nodes._shared import UNVERIFIED
from services import embedder, guardrails_service, pinecone_client, portkey_client, reranker


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.state = {"query": "What accuracy is reported?", "paper_id": "paper-a",
                      "session_id": "session-a", "chat_history": [], "steps": [], "error": "",
                      "generation_attempts": 0, "retry_count": 0, "max_generation_attempts": 2}
        self.chunks = [{"chunk_id": "a", "page_num": 3, "text": "Accuracy is 95%.", "score": 0.9}]
        self.input_check = AsyncMock(return_value=True)
        self.output_check = AsyncMock(return_value=True)
        self.generate = AsyncMock(return_value={"answer": "Accuracy is 95% [1].", "model_used": "test-model"})

        async def plan(messages):
            instruction = messages[0]["content"]
            if instruction.startswith("Classify"):
                return "factual"
            if instruction.startswith("Rewrite"):
                return self.state["query"]
            if instruction.startswith("Generate exactly"):
                return json.dumps(["What are the accuracy results?", "How accurate is the model?", "What performance is reported?"])
            return "Hypothetical retrieval passage."

        self.planner = AsyncMock(side_effect=plan)
        self.search_patch = patch.object(pinecone_client, "hybrid_search", side_effect=lambda *args: copy.deepcopy(self.chunks))
        self.rerank_patch = patch.object(reranker, "rerank_chunks", side_effect=lambda query, chunks, top_n: chunks[:top_n])
        self.patches = [
            patch.object(guardrails_service, "check_input", self.input_check),
            patch.object(guardrails_service, "check_output", self.output_check),
            patch.object(portkey_client, "chat_completion", self.planner),
            patch.object(portkey_client, "chat_completion_with_metadata", self.generate),
            patch.object(embedder, "embed_text", AsyncMock(return_value=[0.5, 0.5])),
            patch.object(embedder, "embed_batch", AsyncMock(side_effect=lambda texts: [[1, 0] for _ in texts])),
            self.search_patch, self.rerank_patch,
        ]
        self.mocks = [item.start() for item in self.patches]

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tracing.__exit__(None, None, None)

    async def run_agent(self):
        return await agent.ainvoke(self.state, config={
            "configurable": {"pinecone_index": object()}, "recursion_limit": 40,
        })

    async def test_all_ten_nodes_and_accumulated_progress(self):
        result = await self.run_agent()
        self.assertEqual(len(result["steps"]), 10)
        self.assertTrue(result["is_output_safe"])
        self.assertEqual(result["generation_attempts"], 1)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["citations"][0]["page_num"], 3)
        self.assertIn("Sources: Page 3", result["final_answer"])

    async def test_real_langchain_tracer_accepts_all_routing_runs(self):
        # Keep the actual Run schema/callbacks; only transport is mocked so this
        # catches SDK validation errors hidden by tests with tracing disabled.
        from langchain_core.tracers.langchain import LangChainTracer
        transport = Mock()
        tracer = LangChainTracer(client=transport, project_name="local-test")
        tracer.raise_error = True
        tracer.run_inline = True
        result = await agent.ainvoke(self.state, config={
            "configurable": {"pinecone_index": object()}, "recursion_limit": 40,
            "callbacks": [tracer],
        })
        self.assertTrue(result["is_output_safe"])
        self.assertEqual(len(result["steps"]), 10)
        self.assertIsNotNone(tracer.latest_run)
        self.assertFalse(tracer.run_map)
        transport.create_run.assert_called()

    async def test_unsafe_input_stops_before_planning(self):
        self.input_check.return_value = False
        result = await self.run_agent()
        self.assertFalse(result["is_safe"])
        self.assertEqual(result["steps"], ["Input blocked"])
        self.planner.assert_not_awaited()
        self.generate.assert_not_awaited()
        self.mocks[-2].assert_not_called()

    async def test_fatal_retrieval_error_stops_before_rerank(self):
        self.mocks[-2].side_effect = RuntimeError("Private service error")
        result = await self.run_agent()
        self.assertTrue(result["error"])
        self.assertNotIn("Private service error", result["error"])
        self.mocks[-1].assert_not_called()
        self.generate.assert_not_awaited()
        self.assertEqual(result["final_answer"], UNVERIFIED)

    async def test_output_failure_retries_then_succeeds(self):
        self.output_check.side_effect = [False, True]
        result = await self.run_agent()
        self.assertTrue(result["is_output_safe"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(self.generate.await_count, 2)
        self.mocks[-1].assert_called_once()
        self.assertEqual(len(result["citations"]), 1)

    async def test_retry_guard_stops_at_budget(self):
        self.state["max_generation_attempts"] = 3
        self.output_check.return_value = False
        result = await self.run_agent()
        self.assertEqual(self.generate.await_count, 3)
        self.assertEqual(result["generation_attempts"], 3)
        self.assertEqual(result["retry_count"], 2)
        self.assertTrue(result["error"])
        self.assertFalse(result["is_output_safe"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["final_answer"], UNVERIFIED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
