"""Run from backend: python -m utils.test_hybrid.

Fits real BM25 encoders against a fake vector index; no cloud requests.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.config import settings
from services import pinecone_client as service
from utils.test_pinecone import FakeIndex, chunk


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name, value in (("UPLOAD_DIR", self.directory.name), ("PINECONE_DIMENSION", 3), ("HYBRID_ALPHA", 0.7)):
            item = patch.object(settings, name, value)
            item.start()
            self.addCleanup(item.stop)
        service._load_bm25.cache_clear()

    def test_fit_dump_load_and_restart_equivalence(self):
        encoder = service.fit_bm25(["quasar telescope", "banana orchard"])
        path = Path(self.directory.name) / "bm25_params.json"
        self.assertTrue(path.is_file())
        expected = encoder.encode_queries(["quasar"])
        service._load_bm25.cache_clear()
        self.assertEqual(service.load_bm25_encoders(), 1)
        loaded = service._load_bm25(str(path), path.stat().st_mtime_ns)
        self.assertEqual(loaded.encode_queries(["quasar"]), expected)

    def test_client_side_alpha_scaling_and_filter(self):
        service.fit_bm25(["quasar telescope", "banana orchard"], "paper-a")
        index = Mock()
        index.query.return_value = {"matches": []}
        result = service.hybrid_search(index, "quasar", [1, 0.5, 0], "paper-a", 10)
        self.assertEqual(result, [])
        arguments = index.query.call_args.kwargs
        self.assertEqual(arguments["vector"], [0.7, 0.35, 0.0])
        self.assertNotIn("alpha", arguments)
        self.assertEqual(arguments["filter"], {"paper_id": "paper-a"})
        encoder = service._load_bm25(str(service._bm25_path("paper-a")), service._bm25_path("paper-a").stat().st_mtime_ns)
        original = encoder.encode_queries(["quasar"])[0]
        self.assertEqual(arguments["sparse_vector"]["indices"], original["indices"])
        for actual, value in zip(arguments["sparse_vector"]["values"], original["values"]):
            self.assertAlmostEqual(actual, value * 0.3)

    def test_sparse_signal_changes_ranking_and_papers_stay_isolated(self):
        index = FakeIndex()
        service.upsert_chunks(index, [
            chunk(0, text="quasar telescope", embedding=[0.1, 0, 0]),
            chunk(1, text="banana orchard", embedding=[1, 0, 0]),
        ], "paper-a")
        parameters = service._bm25_path("paper-a").read_bytes()
        service.upsert_chunks(index, [chunk(2, text="quasar quasar quasar", embedding=[50, 0, 0])], "paper-b")
        self.assertEqual(service._bm25_path("paper-a").read_bytes(), parameters)
        with patch.object(settings, "HYBRID_ALPHA", 1):
            self.assertEqual(service.hybrid_search(index, "quasar", [1, 0, 0], "paper-a", 1)[0]["chunk_id"], "chunk-1")
        with patch.object(settings, "HYBRID_ALPHA", 0):
            self.assertEqual(service.hybrid_search(index, "quasar", [1, 0, 0], "paper-a", 1)[0]["chunk_id"], "chunk-0")
        service._load_bm25.cache_clear()
        self.assertEqual(service.load_bm25_encoders(), 2)
        service.delete_paper_vectors(index, "paper-a")
        self.assertFalse(service._bm25_path("paper-a").exists())
        self.assertTrue(service._bm25_path("paper-b").exists())
        self.assertEqual(list(index.records), ["chunk-2"])

    def test_dense_fallback_for_legacy_papers_or_stopword_queries(self):
        index = Mock()
        index.query.return_value = {"matches": []}
        service.hybrid_search(index, "quasar", [1, 0, 0], "legacy", 5)
        self.assertNotIn("sparse_vector", index.query.call_args.kwargs)
        service.fit_bm25(["quasar telescope"], "paper-a")
        service.hybrid_search(index, "the and", [1, 0, 0], "paper-a", 5)
        self.assertNotIn("sparse_vector", index.query.call_args.kwargs)

    def test_invalid_configuration_and_corpus(self):
        for corpus in ([], [""], ["valid", None], ["the and"]):
            with self.assertRaises(ValueError):
                service.fit_bm25(corpus)
        with patch.object(settings, "HYBRID_ALPHA", 1.5), self.assertRaises(ValueError):
            service.hybrid_search(Mock(), "quasar", [1, 0, 0], "paper-a", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
