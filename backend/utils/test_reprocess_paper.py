"""Retry an existing failed paper with mocked providers and temporary storage."""

from unittest.mock import patch

from utils import reprocess_paper as retry
from utils.test_papers import PaperAPIFixture


class RetryTests(PaperAPIFixture):
    async def test_failed_paper_is_retried_in_place(self):
        self.summary_mock.side_effect = RuntimeError("Summary unavailable")
        with self.assertLogs("api.routes.papers", level="ERROR"):
            upload = await self.upload()
        paper_id = upload.json()["paper_id"]
        self.summary_mock.side_effect = None
        with patch.object(retry, "AsyncSessionLocal", self.sessions), \
                patch.object(retry, "init_pinecone_index", return_value=self.index):
            result = await retry.reprocess_paper(paper_id)
        self.assertEqual(result["paper_id"], paper_id)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len((await self.client.get("/api/v1/papers/")).json()), 1)
        self.assertEqual(len(list(self.uploads.glob("*.pdf"))), 1)

    async def test_ready_paper_is_not_overwritten(self):
        upload = await self.upload()
        with patch.object(retry, "AsyncSessionLocal", self.sessions), \
                patch.object(retry, "init_pinecone_index") as index:
            with self.assertRaisesRegex(ValueError, "Only failed papers"):
                await retry.reprocess_paper(upload.json()["paper_id"])
        index.assert_not_called()
