"""Run from backend: python -m utils.test_pinecone.

Uses mocked control-plane calls and an in-memory fake index; no cloud requests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pinecone.core.client.model.query_response import QueryResponse
from pinecone.core.client.model.scored_vector import ScoredVector

from core.config import settings
from services import pinecone_client as service


def chunk(index=0, **overrides):
    return {
        "chunk_id": f"chunk-{index}", "embedding": [1.0, 0.0, 0.0],
        "text": f"Evidence {index}", "page_num": 1, "chunk_index": index,
        **overrides,
    }


class FakeIndex:
    def __init__(self):
        self.records = {}

    def upsert(self, vectors):
        self.records.update({vector["id"]: vector for vector in vectors})

    def query(self, vector, filter, top_k, include_metadata):
        matches = [
            {"id": key, "metadata": row["metadata"],
             "score": sum(a * b for a, b in zip(vector, row["values"]))}
            for key, row in self.records.items()
            if row["metadata"]["paper_id"] == filter["paper_id"]
        ]
        return {"matches": sorted(matches, key=lambda row: row["score"], reverse=True)[:top_k]}

    def delete(self, filter):
        self.records = {
            key: row for key, row in self.records.items()
            if row["metadata"]["paper_id"] != filter["paper_id"]
        }


class PineconeTests(unittest.TestCase):
    def setUp(self):
        self.dimension = patch.object(settings, "PINECONE_DIMENSION", 3)
        self.dimension.start()
        self.addCleanup(self.dimension.stop)

    def control_plane(self, existing=True, **description):
        pc = Mock()
        pc.list_indexes.return_value.names.return_value = [settings.PINECONE_INDEX_NAME] if existing else []
        pc.describe_index.return_value = SimpleNamespace(**{
            "dimension": 3, "metric": "dotproduct", "status": {"ready": True},
            **description,
        })
        return pc

    def test_create_index_and_return_handle(self):
        pc = self.control_plane(existing=False)
        with patch.object(settings, "PINECONE_API_KEY", "test-key"), \
                patch.object(service, "Pinecone", return_value=pc) as factory:
            self.assertIs(service.init_pinecone_index(), pc.Index.return_value)
        factory.assert_called_once_with(api_key="test-key")
        options = pc.create_index.call_args.kwargs
        self.assertEqual(options["name"], settings.PINECONE_INDEX_NAME)
        self.assertEqual(options["dimension"], 3)
        self.assertEqual(options["metric"], "dotproduct")
        self.assertEqual(options["spec"].asdict(), {
            "serverless": {"cloud": "aws", "region": settings.PINECONE_ENVIRONMENT},
        })
        self.assertEqual(options["timeout"], 60)
        pc.Index.assert_called_once_with(settings.PINECONE_INDEX_NAME)

    def test_reuse_existing_index(self):
        pc = self.control_plane()
        with patch.object(settings, "PINECONE_API_KEY", "test-key"), \
                patch.object(service, "Pinecone", return_value=pc):
            service.init_pinecone_index()
        pc.create_index.assert_not_called()

    def test_invalid_existing_index_is_rejected(self):
        for description, exception in (
            ({"dimension": 10}, ValueError), ({"metric": "cosine"}, ValueError),
            ({"status": {"ready": False}}, RuntimeError),
        ):
            pc = self.control_plane(**description)
            with patch.object(settings, "PINECONE_API_KEY", "test-key"), \
                    patch.object(service, "Pinecone", return_value=pc), \
                    self.assertRaises(exception):
                service.init_pinecone_index()
            pc.Index.assert_not_called()

    def test_missing_key_makes_no_request(self):
        with patch.object(settings, "PINECONE_API_KEY", ""), \
                patch.object(service, "Pinecone") as factory, self.assertRaises(ValueError):
            service.init_pinecone_index()
        factory.assert_not_called()

    def test_upsert_batching_and_metadata(self):
        index = Mock()
        self.assertEqual(service.upsert_chunks(index, [chunk(i) for i in range(205)], "paper-a"), 205)
        self.assertEqual([len(call.kwargs["vectors"]) for call in index.upsert.call_args_list], [100, 100, 5])
        first = index.upsert.call_args_list[0].kwargs["vectors"][0]
        self.assertEqual(first, {
            "id": "chunk-0", "values": [1.0, 0.0, 0.0],
            "metadata": {"paper_id": "paper-a", "text": "Evidence 0", "page_num": 1, "chunk_index": 0},
        })

    def test_byte_budget_reduces_batch_size(self):
        index = Mock()
        with patch.object(service, "_MAX_BATCH_BYTES", 600):
            service.upsert_chunks(index, [chunk(i) for i in range(10)], "paper-a")
        self.assertGreater(index.upsert.call_count, 1)
        self.assertEqual(sum(len(call.kwargs["vectors"]) for call in index.upsert.call_args_list), 10)

    def test_upsert_validation_precedes_all_writes(self):
        for invalid in (
            chunk(1, embedding=[1.0]), chunk(1, embedding=[float("nan"), 0, 0]),
            chunk(1, text=""), chunk(1, text="x" * 42000),
            chunk(1, page_num=0), chunk(1, chunk_index=-1), chunk(1, chunk_id="chunk-0"),
        ):
            index = Mock()
            with self.assertRaises(ValueError):
                service.upsert_chunks(index, [chunk(), invalid], "paper-a")
            index.upsert.assert_not_called()
        index = Mock()
        self.assertEqual(service.upsert_chunks(index, [], "paper-a"), 0)
        index.upsert.assert_not_called()

    def test_dense_search_with_actual_sdk_response_models(self):
        index = Mock()
        index.query.return_value = QueryResponse(matches=[
            ScoredVector(id="chunk-0", score=0.75, metadata={
                "paper_id": "paper-a", "text": "Evidence", "page_num": 2,
            }),
            ScoredVector(id="other-paper", score=1.0, metadata={
                "paper_id": "paper-b", "text": "Not allowed", "page_num": 1,
            }),
        ])
        self.assertEqual(service.dense_search(index, [1, 0, 0], "paper-a", 5), [
            {"text": "Evidence", "page_num": 2, "score": 0.75, "chunk_id": "chunk-0"},
        ])
        index.query.assert_called_once_with(
            vector=[1, 0, 0], filter={"paper_id": "paper-a"}, top_k=5, include_metadata=True,
        )

    def test_two_paper_search_and_delete_isolation(self):
        index = FakeIndex()
        service.upsert_chunks(index, [chunk(0), chunk(1, embedding=[0.5, 0, 0])], "paper-a")
        service.upsert_chunks(index, [chunk(2, embedding=[10, 0, 0])], "paper-b")
        results = service.dense_search(index, [1, 0, 0], "paper-a", 10)
        self.assertEqual([row["chunk_id"] for row in results], ["chunk-0", "chunk-1"])
        service.delete_paper_vectors(index, "paper-a")
        self.assertEqual(service.dense_search(index, [1, 0, 0], "paper-a", 10), [])
        self.assertEqual(list(index.records), ["chunk-2"])

    def test_bad_search_or_delete_never_calls_index(self):
        index = Mock()
        for paper_id in ("", " "):
            with self.assertRaises(ValueError):
                service.dense_search(index, [1, 0, 0], paper_id, 5)
            with self.assertRaises(ValueError):
                service.delete_paper_vectors(index, paper_id)
        with self.assertRaises(ValueError):
            service.dense_search(index, [1, 0, 0], "paper-a", 0)
        with self.assertRaises(ValueError):
            service.dense_search(index, [1], "paper-a", 5)
        index.query.assert_not_called()
        index.delete.assert_not_called()

    def test_provider_errors_propagate(self):
        index = Mock()
        index.upsert.side_effect = RuntimeError("Offline")
        with self.assertRaises(RuntimeError):
            service.upsert_chunks(index, [chunk()], "paper-a")
        index.delete.side_effect = RuntimeError("Offline")
        with self.assertRaises(RuntimeError):
            service.delete_paper_vectors(index, "paper-a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
