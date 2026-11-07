"""Real SQLite + FastAPI integration coverage; no model inference is mocked as live verification."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.cli import main as cli_main
from src.contracts import CorpusResult, UDLException
from src.runtime import Runtime
from src.web import create_app


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/raw").mkdir(parents=True)
        (self.root / "data/assets").mkdir(parents=True)
        (self.root / "data/raw/book.pdf").write_bytes(b"%PDF integration fixture")
        (self.root / "data/assets/figure.png").write_bytes(b"fixture crop")
        self.runtime = Runtime(self.root / "data/test.sqlite3", self.root)
        self.client = TestClient(create_app(project_root=self.root, runtime=self.runtime))
        self.addCleanup(self.client.close)
        self.corpus = CorpusResult(
            sources=[{"source_id": "book", "source_type": "pdf", "path": "data/raw/book.pdf",
                      "sha256": hashlib.sha256(b"%PDF integration fixture").hexdigest(), "revision": "fixture", "title": "Fixture book",
                      "metadata": {"pages": 2}}],
            chunks=[
                {"chunk_id": "text-1", "source_id": "book", "source_type": "pdf", "content_type": "text",
                 "content": "Gradient descent updates weights using the loss gradient.",
                 "metadata": {"chapter": "1", "pdf_page": 1}},
                {"chunk_id": "figure-1", "source_id": "book", "source_type": "pdf", "content_type": "figure",
                 "content": "Figure 1.1 neural network diagram.",
                 "metadata": {"chapter": "1", "pdf_page": 1, "figure_number": "1.1", "asset_path": "data/assets/figure.png", "review_status": "verified"}},
                {"chunk_id": "equation-1", "source_id": "book", "source_type": "pdf", "content_type": "equation",
                 "content": "a = b + c", "metadata": {"chapter": "1", "pdf_page": 1, "equation_number": "1.1", "asset_path": "data/assets/figure.png", "review_status": "pending"}},
            ], report={})
        self.runtime.store.build(self.corpus)

    def test_status_search_chunk_and_allowlisted_source_and_asset_use_sqlite(self):
        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json()["lexical_ready"])
        self.assertFalse(status.json()["dense_ready"])
        self.assertIn("job", status.json())
        self.assertTrue(all("api_key" not in p for p in status.json()["providers"]))

        result = self.client.post("/api/search", json={"query": "gradient descent weights", "mode": "hybrid"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["hits"][0]["chunk_id"], "text-1")
        self.assertEqual(result.json()["mode"], "lexical")
        self.assertTrue(result.json()["warnings"])

        chunk = self.client.get("/api/chunks/text-1")
        self.assertEqual(chunk.status_code, 200)
        self.assertEqual(chunk.json()["content"], self.runtime.store.get_chunk("text-1")["content"])
        self.assertEqual(self.client.get("/api/source", params={"source_id": "book"}).content, b"%PDF integration fixture")
        self.assertEqual(self.client.get("/api/asset", params={"chunk_id": "figure-1"}).content, b"fixture crop")
        self.assertEqual(self.client.get("/api/asset", params={"chunk_id": "text-1"}).status_code, 404)
        self.assertEqual(self.client.get("/api/chunks/text-1", params={"build_id": "missing"}).status_code, 404)

    def test_cli_status_and_search_read_the_same_sqlite_database(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli_main(["status", "--db", str(self.runtime.db_path)]), 0)
        self.assertEqual(json.loads(output.getvalue())["active_build_id"],
                         self.runtime.store.status()["active_build_id"])
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli_main(["search", "--db", str(self.runtime.db_path), "gradient descent", "--mode", "lexical"]), 0)
        self.assertEqual(json.loads(output.getvalue())["hits"][0]["chunk_id"], "text-1")

    def test_math_context_gate_omits_pending_equation(self):
        from src.tutor import context_from_hits
        equation = self.runtime.store.get_chunk("equation-1")
        text, sources = context_from_hits([equation])
        self.assertEqual(text, "")
        self.assertEqual(sources, [])

    def test_failed_staged_build_keeps_previous_active_sqlite_build(self):
        before = self.runtime.store.status()["active_build_id"]
        broken = CorpusResult(
            sources=self.corpus.sources,
            chunks=[dict(self.corpus.chunks[0], source_id="missing-source")],
        )
        with self.assertRaises(UDLException):
            self.runtime.store.build(broken)
        after = self.runtime.store.status()
        self.assertEqual(after["active_build_id"], before)
        self.assertTrue(after["lexical_ready"])
        self.assertIsNotNone(self.runtime.store.get_chunk("text-1", build_id=before))

    def test_background_build_reports_completion_through_status(self):
        with patch("src.corpus.ensure_sources"), patch("src.corpus.extract_corpus", return_value=self.corpus):
            response = self.client.post("/api/build", json={"semantic": False, "offline": True})
        self.assertEqual(response.status_code, 200, response.text)
        deadline = time.monotonic() + 5
        status = response.json()
        while status["job"].get("running") and time.monotonic() < deadline:
            time.sleep(0.01)
            status = self.client.get("/api/status").json()
        self.assertFalse(status["job"].get("running"), status["job"])
        self.assertEqual(status["job"].get("phase"), "complete", status["job"])

    def test_source_membership_and_path_containment_are_enforced(self):
        outside = self.root / "secret.pdf"
        outside.write_bytes(b"secret")
        hostile = CorpusResult(
            sources=[dict(self.corpus.sources[0], path="secret.pdf")],
            chunks=[dict(self.corpus.chunks[0], chunk_id="hostile-text")],
        )
        self.runtime.store.build(hostile)
        self.assertEqual(self.client.get("/api/source", params={"source_id": "book"}).status_code, 404)
        self.assertEqual(self.client.get("/api/source", params={"source_id": "unknown"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
