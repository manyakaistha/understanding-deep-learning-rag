import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.contracts import BOOK_SHA256, CorpusResult, UDLException
from src.review_bundle import apply_review_bundle, assemble_review_bundle, text_sha256


class ReviewBundleTests(unittest.TestCase):
    def test_new_hashed_review_replaces_stale_overlay_without_invalidating_old_build(self):
        from src.store import SQLiteStore
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "data/assets"
            assets.mkdir(parents=True)
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            def reviewed(version):
                image = assets / (version + ".png")
                image.write_bytes(version.encode())
                text = "y=x" if version == "old" else "y=2x"
                crop_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                corpus = CorpusResult(sources=[dict(source_id="book", source_type="book",
                                      path="data/raw/book.pdf", sha256=BOOK_SHA256)],
                    chunks=[dict(chunk_id="eq", source_id="book", source_type="book",
                        content_type="equation", content="raw", raw_content="raw",
                        metadata=dict(asset_path=f"data/assets/{version}.png", equation_number="1.1", pdf_page=1))])
                review = dict(chunk_id="eq", decision="verified", source_sha256=BOOK_SHA256,
                    crop_sha256=crop_hash, transcription=text, transcription_sha256=text_sha256(text),
                    reviewer="Independent reviewer", method="source comparison")
                batch = root / (version + ".json")
                batch.write_text(json.dumps([review]))
                bundle = root / "resources/book_math_reviews.json"
                assemble_review_bundle([batch], bundle)
                return apply_review_bundle(corpus, root)
            old = store.build(reviewed("old"))
            self.assertEqual(store.get_chunk("eq")["metadata"]["review_status"], "verified")
            fresh = reviewed("new")
            class FailingEmbedder:
                profile = {"model_name": "test", "revision": "test", "dimension": 4}
                fingerprint = "test"
                def encode_documents(self, texts):
                    raise RuntimeError("simulated inference failure")
            with self.assertRaises(RuntimeError):
                store.build(fresh, embedder=FailingEmbedder())
            self.assertEqual(store.status()["build_id"], old["build_id"])
            self.assertEqual(store.get_chunk("eq")["metadata"]["review_status"], "verified")
            new = store.build(fresh)
            self.assertNotEqual(new["build_id"], old["build_id"])
            self.assertEqual(store.get_chunk("eq")["content"], "y=2x")
            self.assertEqual(store.get_chunk("eq")["metadata"]["review_status"], "verified")
            historical = store.list_reviews(build_id=old["build_id"])
            self.assertEqual(historical[0]["metadata"]["review_status"], "verified")

    def test_only_exact_reviewed_source_transcription_is_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "data/assets/equation.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"source crop")
            crop_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            corpus = CorpusResult(sources=[{"source_id": "book", "sha256": BOOK_SHA256}],
                                  chunks=[{"chunk_id": "eq", "content_type": "equation", "content": "raw", "raw_content": "raw", "embedding_text": "old", "metadata": {"asset_path": "data/assets/equation.png", "review_status": "pending"}}])
            review = dict(chunk_id="eq", decision="verified", source_sha256=BOOK_SHA256,
                          crop_sha256=crop_hash, transcription=r"y=f[x]", transcription_sha256=text_sha256(r"y=f[x]"),
                          reviewer="Independent source comparison", method="visual comparison")
            batch = root / "batch.json"
            batch.write_text(json.dumps([review]))
            bundle = root / "resources/book_math_reviews.json"
            assemble_review_bundle([batch], bundle)
            reviewed = apply_review_bundle(corpus, root)
            self.assertEqual(reviewed.chunks[0]["metadata"]["review_status"], "verified")
            self.assertEqual(reviewed.chunks[0]["content"], r"y=f[x]")
            self.assertEqual(reviewed.chunks[0]["raw_content"], "raw")
            self.assertNotIn("embedding_text", reviewed.chunks[0])
            self.assertEqual(corpus.chunks[0]["metadata"]["review_status"], "pending")
            image.write_bytes(b"new geometry")
            invalidated = apply_review_bundle(corpus, root)
            self.assertEqual(invalidated.report["review_bundle"]["invalidated"], ["eq"])
            self.assertEqual(invalidated.chunks[0]["metadata"]["review_status"], "pending")

    def test_bad_hash_cannot_turn_a_draft_into_a_verified_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch.json"
            batch.write_text(json.dumps([dict(chunk_id="eq", decision="verified", source_sha256=BOOK_SHA256, transcription="changed", transcription_sha256="not-the-hash")]))
            with self.assertRaises(UDLException):
                assemble_review_bundle([batch], root / "bundle.json")
            batch.write_text(json.dumps([dict(chunk_id="eq", decision="pending", transcription="draft")]))
            result = assemble_review_bundle([batch], root / "bundle.json")
            self.assertEqual(result["verified_records"], 0)


if __name__ == "__main__":
    unittest.main()
