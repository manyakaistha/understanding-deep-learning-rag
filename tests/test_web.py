from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from src.web import create_app


class FakeStore:
    def __init__(self):
        self.source = None

    def get_source(self, source_id, build_id=None):
        return self.source

    def get_chunk(self, chunk_id, build_id=None):
        return None

    def list_reviews(self, build_id=None):
        return []


class FakeRuntime:
    def __init__(self):
        self.store = FakeStore()

    def status(self):
        return {"lexical_ready": False, "dense_ready": False}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = FakeRuntime()
        self.client = TestClient(create_app(project_root=self.root, runtime=self.runtime))
        self.addCleanup(self.client.close)

    def test_validation_errors_do_not_echo_api_keys(self):
        response = self.client.post("/api/ask", json={"query": "x", "provider": "groq", "api_key": "DO-NOT-ECHO", "top_k": "DO-NOT-ECHO"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("DO-NOT-ECHO", response.text)

    def test_cross_origin_mutations_rejected(self):
        response = self.client.post("/api/build", json={}, headers={"origin": "https://external.example"})
        self.assertEqual(response.status_code, 403)

    def test_arbitrary_file_reads_and_symlinks_rejected(self):
        target = self.root / "private.txt"
        target.write_text("PRIVATE")
        self.runtime.store.source = {"path": "private.txt"}
        response = self.client.get("/api/source", params={"source_id": "book"})
        self.assertEqual(response.status_code, 404)
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        (raw / "escape.pdf").symlink_to(target)
        self.runtime.store.source = {"path": "data/raw/escape.pdf"}
        self.assertEqual(self.client.get("/api/source", params={"source_id": "book"}).status_code, 404)

    def test_known_source_served_and_unknown_chunk_fails(self):
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        (raw / "book.pdf").write_bytes(b"%PDF-test")
        self.runtime.store.source = {"path": "data/raw/book.pdf"}
        response = self.client.get("/api/source", params={"source_id": "book"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-test")
        self.assertEqual(self.client.get("/api/chunks/missing").status_code, 404)

    def test_status_exposes_readiness_without_secrets(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["dense_ready"])
        self.assertTrue(all("api_key" not in p for p in response.json()["providers"]))


if __name__ == "__main__":
    unittest.main()
