"""Run from backend: python -m utils.test_reranking.

Mocks both reranking engines; no model download or billable API requests.
"""

import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.config import settings
from services import reranker
from utils.text_utils import apply_rrf


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        reranker._get_local_ranker.cache_clear()
        for name, value in (("COHERE_API_KEY", "test-key"), ("FLASHRANK_CACHE_DIR", self.directory.name)):
            item = patch.object(settings, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.chunks = [
            {"chunk_id": "a", "text": "Bananas grow on plants.", "page_num": 2, "score": 0.8},
            {"chunk_id": "b", "text": "Mars is the red planet.", "page_num": 5, "score": 0.5},
        ]

    def test_cohere_mapping_sorted_and_no_input_mutation(self):
        original = copy.deepcopy(self.chunks)
        co = Mock()
        co.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=0, relevance_score=0.1),
            SimpleNamespace(index=1, relevance_score=0.9),
        ])
        with patch.object(reranker.cohere, "Client", return_value=co), \
                patch.object(reranker, "Ranker") as local:
            results = reranker.rerank_chunks("Which planet is red?", self.chunks, 2)
        self.assertEqual([chunk["chunk_id"] for chunk in results], ["b", "a"])
        self.assertEqual(results[0]["relevance_score"], 0.9)
        self.assertEqual(results[0]["page_num"], 5)
        self.assertEqual(results[0]["score"], 0.5)
        self.assertEqual(self.chunks, original)
        co.rerank.assert_called_once_with(
            model="rerank-english-v3.0", query="Which planet is red?",
            documents=[chunk["text"] for chunk in self.chunks], top_n=2,
        )
        local.assert_not_called()

    def test_cohere_failure_falls_back_to_local(self):
        def local_results(request):
            # Mimic FlashRank's in-place mutation and out-of-order document IDs.
            request.passages[0]["score"] = 0.1
            request.passages[1]["score"] = 0.95
            return request.passages[::-1]

        co = Mock()
        co.rerank.side_effect = RuntimeError("private provider error")
        ranker = Mock()
        ranker.rerank.side_effect = local_results
        original = copy.deepcopy(self.chunks)
        with patch.object(reranker.cohere, "Client", return_value=co), \
                patch.object(reranker, "Ranker", return_value=ranker), \
                self.assertLogs(reranker.logger, level="WARNING") as logs:
            results = reranker.rerank_chunks("Which planet is red?", self.chunks, 1)
        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertEqual(results[0]["relevance_score"], 0.95)
        self.assertEqual(self.chunks, original)
        self.assertNotIn("private provider error", str(logs.output))

    def test_missing_key_uses_cached_local_ranker(self):
        local = Mock()
        local.rerank.return_value = [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.1}]
        with patch.object(settings, "COHERE_API_KEY", ""), \
                patch.object(reranker.cohere, "Client") as co, \
                patch.object(reranker, "Ranker", return_value=local) as factory:
            reranker.prepare_flashrank()
            self.assertEqual(len(reranker.rerank_chunks("Mars?", self.chunks, 99)), 2)
            reranker.rerank_chunks("Mars?", self.chunks, 1)
        factory.assert_called_once()
        co.assert_not_called()

    def test_invalid_primary_result_triggers_fallback(self):
        co = Mock()
        co.rerank.return_value = SimpleNamespace(results=[SimpleNamespace(index=99, relevance_score=0.9)])
        local = Mock()
        local.rerank.return_value = [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.1}]
        with patch.object(reranker.cohere, "Client", return_value=co), \
                patch.object(reranker, "Ranker", return_value=local), \
                self.assertLogs(reranker.logger, level="WARNING"):
            self.assertEqual(reranker.rerank_chunks("Mars?", self.chunks, 1)[0]["chunk_id"], "b")

    def test_empty_zero_invalid_and_failed_local(self):
        with patch.object(reranker.cohere, "Client") as co, patch.object(reranker, "Ranker") as local:
            self.assertEqual(reranker.rerank_chunks("Mars?", [], 2), [])
            self.assertEqual(reranker.rerank_chunks("Mars?", self.chunks, 0), [])
            for query, chunks, top_n in ((" ", self.chunks, 1), ("Mars?", self.chunks, -1), ("Mars?", [{"text": ""}], 1)):
                with self.assertRaises(ValueError):
                    reranker.rerank_chunks(query, chunks, top_n)
            co.assert_not_called()
            local.assert_not_called()
        with patch.object(settings, "COHERE_API_KEY", ""), \
                patch.object(reranker, "Ranker", side_effect=OSError("No cached model")), \
                self.assertRaisesRegex(RuntimeError, "prepare the FlashRank cache"):
            reranker.rerank_chunks("Mars?", self.chunks, 1)


class RRFTests(unittest.TestCase):
    def test_ids_merge_by_rank(self):
        self.assertEqual(apply_rrf([["a", "b"], ["b", "c"]]), ["b", "a", "c"])
        self.assertEqual(apply_rrf([]), [])
        self.assertEqual(apply_rrf([[], []]), [])

    def test_chunk_scores_metadata_and_duplicates(self):
        a = {"chunk_id": "a", "text": "first", "score": 0.9}
        b = {"chunk_id": "b", "text": "second", "score": 0.1}
        inputs = [[a, a, b], [{"chunk_id": "b", "text": "other copy"}]]
        original = copy.deepcopy(inputs)
        result = apply_rrf(inputs)
        self.assertEqual([chunk["chunk_id"] for chunk in result], ["b", "a"])
        self.assertAlmostEqual(result[0]["rrf_score"], 1 / 63 + 1 / 61)
        self.assertAlmostEqual(result[1]["rrf_score"], 1 / 61)
        self.assertEqual(result[0]["text"], "second")
        self.assertEqual(result[0]["score"], 0.1)
        self.assertEqual(inputs, original)

    def test_ties_zero_k_and_id_alias(self):
        self.assertEqual(apply_rrf([["b"], ["a"]]), ["b", "a"])
        result = apply_rrf([[{"id": "x"}], [{"chunk_id": "x"}]], k=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rrf_score"], 2)

    def test_invalid_inputs(self):
        for k in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                apply_rrf([], k)
        for ranked in ([[{}]], [[{"text": "No ID"}]], [[""]], [[None]]):
            with self.assertRaises(ValueError):
                apply_rrf(ranked)


if __name__ == "__main__":
    unittest.main(verbosity=2)
