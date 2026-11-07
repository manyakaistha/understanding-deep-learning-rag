"""Real SQLite transaction, migration, staging, review, and cache tests for `src/store.py`."""

import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from src.contracts import CorpusResult, UDLException
from src.dense import compute_profile_fingerprint, l2_normalize
from src.store import SQLiteStore


class _DeterministicFakeEmbedder:
    """Contract-compliant test embedder (unit test double, not real model verification)."""

    def __init__(self, dimension: int = 8, revision: str = "test_rev_1") -> None:
        self.dimension = dimension
        self.revision = revision
        self.doc_calls = []
        self.query_calls = []
        self.fail_on_call = False

    @property
    def profile(self):
        return {
            "backend": "torch+transformers",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "revision": self.revision,
            "configured_revision": self.revision,
            "dimension": self.dimension,
            "pooling": "last_valid_token",
            "normalize": "l2",
            "dtype": "float32",
            "max_length": 4096,
            "overflow_policy": "sliding_window",
            "window_overlap": 128,
            "window_aggregation": "l2_normalized_mean",
            "query_overflow_policy": "reject",
            "query_instruction": "Given a web search query, retrieve relevant passages that answer the query",
            "query_prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            "document_prompt": "",
        }

    @property
    def fingerprint(self) -> str:
        return compute_profile_fingerprint(self.profile)

    def _hash_vec(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(digest[i % len(digest)]) + 1.0 for i in range(self.dimension)]
        return l2_normalize(raw)

    def encode_documents(self, texts):
        if self.fail_on_call:
            raise RuntimeError("Simulated embedder crash during build")
        self.doc_calls.append(list(texts))
        return [self._hash_vec(f"doc:{t}") for t in texts]

    def encode_queries(self, texts):
        self.query_calls.append(list(texts))
        return [self._hash_vec(f"query:{t}") for t in texts]


def _make_sample_corpus(root: Path) -> CorpusResult:
    assets_dir = root / "data" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "fig_7_5.png"
    fig_path.write_bytes(b"PNG_FIGURE_7_5_BYTES")
    eq_path = assets_dir / "eq_7_1.png"
    eq_path.write_bytes(b"PNG_EQUATION_7_1_BYTES")

    sources = [
        {
            "source_id": "book",
            "source_type": "pdf",
            "path": "data/raw/UnderstandingDeepLearning.pdf",
            "sha256": "9c0ea2e831e662a687d267dd9bb2d50ea743e58271e90b96a4fac4d585288e04",
            "revision": "v5.0.2",
            "title": "Understanding Deep Learning",
            "metadata": {"pages": 541},
        },
        {
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "path": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "sha256": "nb_sha_6_2",
            "revision": "44a3e5f",
            "title": "6.2 Gradient Descent",
            "metadata": {"chapter": "6"},
        },
    ]

    chunks = [
        {
            "chunk_id": "book_text_1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "text",
            "content": "Backpropagation computes gradients of the loss function with respect to weights.",
            "raw_content": "Backpropagation computes gradients of the loss function with respect to weights.",
            "metadata": {"chapter": "7", "section": "7.2", "pdf_page": 116, "book_page": 104},
        },
        {
            "chunk_id": "book_fig_7_5",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "figure",
            "content": "Figure 7.5 Backward pass through a two-layer neural network.",
            "raw_content": "Figure 7.5 Backward pass through a two-layer neural network.",
            "metadata": {
                "chapter": "7",
                "pdf_page": 117,
                "figure_number": "7.5",
                "asset_path": "data/assets/fig_7_5.png",
                "review_status": "verified",
            },
        },
        {
            "chunk_id": "book_eq_7_1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "equation",
            "content": r"\frac{\partial L}{\partial \Omega_k} = \frac{\partial l_i}{\partial f} \frac{\partial f}{\partial \Omega_k}",
            "raw_content": r"\frac{\partial L}{\partial \Omega_k} = \frac{\partial l_i}{\partial f} \frac{\partial f}{\partial \Omega_k}",
            "metadata": {
                "chapter": "7",
                "pdf_page": 118,
                "equation_number": "7.1",
                "asset_path": "data/assets/eq_7_1.png",
                "review_status": "pending",
            },
        },
        {
            "chunk_id": "nb_code_0",
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "content_type": "code",
            "content": "import numpy as np",
            "raw_content": "import numpy as np",
            "metadata": {
                "chapter": "6",
                "cell_index": 0,
                "start_line": 1,
                "end_line": 1,
                "symbol_names": ["np"],
            },
        },
        {
            "chunk_id": "nb_code_1_duplicate_content",
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "content_type": "code",
            "content": "import numpy as np",
            "raw_content": "import numpy as np",
            "metadata": {
                "chapter": "6",
                "cell_index": 1,
                "start_line": 1,
                "end_line": 1,
                "symbol_names": ["np"],
            },
        },
    ]

    return CorpusResult(sources=sources, chunks=chunks, report={})


class StoreTests(unittest.TestCase):
    def test_visual_snapshot_integrity_blocks_release_despite_stored_crop_hash(self):
        for kind in ("figure", "algorithm"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
                corpus = _make_sample_corpus(root)
                equation = next(c for c in corpus.chunks if c["content_type"] == "equation")
                equation["metadata"].update(review_status="verified", reviewer="independent reviewer")
                if kind == "figure":
                    visual = next(c for c in corpus.chunks if c["content_type"] == kind)
                else:
                    visual = {
                        "chunk_id": "book_algorithm_1", "source_id": "book",
                        "source_type": "pdf", "content_type": "algorithm",
                        "content": "Algorithm 1: gradient descent", "raw_content": "Algorithm 1: gradient descent",
                        "metadata": {"pdf_page": 100, "review_status": "verified"},
                    }
                    corpus.chunks.append(visual)
                crop_bytes = f"reviewed {kind} snapshot bytes".encode()
                snapshot = root / "data" / "assets" / "snapshots" / f"{kind}.png"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(crop_bytes)
                visual["metadata"].update(
                    asset_path=str(snapshot.relative_to(root)),
                    crop_sha256=hashlib.sha256(crop_bytes).hexdigest(),
                )
                before = store.build(corpus, embedder=_DeterministicFakeEmbedder())
                self.assertTrue(before["release_ready"])

                # The persisted digest is proof of the reviewed bytes, not a
                # substitute for the image required to inspect a citation.
                snapshot.unlink()
                missing = store.status()
                self.assertFalse(missing["release_ready"])
                self.assertEqual(missing["gates"]["missing_asset_files"], 1)
                self.assertEqual(missing["counts"][f"{kind}s_pending"], 1)
                self.assertEqual(missing["counts"]["equations_verified"], 1)

                snapshot.write_bytes(b"different unreviewed image bytes")
                changed = store.status()
                self.assertFalse(changed["release_ready"])
                self.assertEqual(changed["counts"][f"{kind}s_pending"], 1)
                self.assertEqual(changed["gates"]["missing_asset_files"], 0)
                snapshot.write_bytes(crop_bytes)
                self.assertTrue(store.status()["release_ready"])

    def test_migrations_wal_foreign_keys_and_safe_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "data" / "udl_rag.sqlite3"
            store = SQLiteStore(db_path, project_root=root)

            conn = sqlite3.connect(str(db_path))
            try:
                journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                self.assertEqual(journal_mode.lower(), "wal")
                mig_count = conn.execute(
                    "SELECT COUNT(*) FROM schema_migrations"
                ).fetchone()[0]
                self.assertGreaterEqual(mig_count, 1)
            finally:
                conn.close()

            corpus = _make_sample_corpus(root)
            store.build(corpus, embedder=None)

            backup_file = store.backup()
            self.assertTrue(backup_file.is_file())
            backup_store = SQLiteStore(backup_file, project_root=root)
            self.assertEqual(len(backup_store.list_chunks()), 5)
            self.assertEqual(len(backup_store.list_sources()), 2)

    def test_lexical_build_duplicate_content_provenance_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            corpus = _make_sample_corpus(root)

            status = store.build(corpus, embedder=None)
            self.assertTrue(status["initialized"])
            self.assertTrue(status["lexical_ready"])
            self.assertFalse(status["dense_ready"])
            self.assertFalse(status["release_ready"])
            self.assertEqual(status["counts"]["chunks"], 5)
            self.assertIn("dense_index_missing", status["gates"]["blockers"])
            self.assertIn("pending_equation_reviews:1", status["gates"]["blockers"])

            sources = store.list_sources()
            self.assertEqual(
                [s["source_id"] for s in sources],
                [
                    "book",
                    "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
                ],
            )
            self.assertEqual(
                store.get_source("book")["title"], "Understanding Deep Learning"
            )
            self.assertIsNone(store.get_source("nonexistent"))

            chunks = store.list_chunks()
            self.assertEqual(len(chunks), 5)
            c0 = store.get_chunk("nb_code_0")
            c1 = store.get_chunk("nb_code_1_duplicate_content")
            self.assertEqual(c0["content"], c1["content"])
            self.assertEqual(c0["metadata"]["cell_index"], 0)
            self.assertEqual(c1["metadata"]["cell_index"], 1)

    def test_atomic_staging_preserves_prior_build_on_failure_and_recovers_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "data" / "udl_rag.sqlite3"
            store = SQLiteStore(db_path, project_root=root)
            corpus = _make_sample_corpus(root)

            embedder = _DeterministicFakeEmbedder(dimension=8)
            first_status = store.build(corpus, embedder=embedder)
            first_build_id = first_status["active_build_id"]
            self.assertTrue(first_status["dense_ready"])

            # Simulate an embedder failure during a second build with modified chunk
            bad_corpus = _make_sample_corpus(root)
            bad_corpus.chunks[0]["content"] = "Modified content triggering re-encode"
            embedder.fail_on_call = True
            with self.assertRaises(RuntimeError):
                store.build(bad_corpus, embedder=embedder)

            # Prior active build must remain intact and active
            after_fail = store.status()
            self.assertEqual(after_fail["active_build_id"], first_build_id)
            self.assertTrue(after_fail["lexical_ready"])
            self.assertTrue(after_fail["dense_ready"])
            self.assertIn(
                "Backpropagation",
                store.get_chunk("book_text_1")["content"],
            )

            # Simulate an active writer lock from current process
            conn = sqlite3.connect(str(db_path))
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO writer_locks(lock_name, owner_id, pid, build_id, acquired_at, expires_at)
                        VALUES ('build', 'manual_owner', ?, 'manual_build', ?, ?)
                        """,
                        (os.getpid(), time.time(), time.time() + 600),
                    )
            finally:
                conn.close()

            embedder.fail_on_call = False
            with self.assertRaises(UDLException) as ctx:
                store.build(bad_corpus, embedder=embedder)
            self.assertEqual(ctx.exception.code, "build_busy")

            # Simulate expired lock and leftover staging build; recover() must clean them up
            conn = sqlite3.connect(str(db_path))
            try:
                with conn:
                    conn.execute(
                        "UPDATE writer_locks SET expires_at = ? WHERE lock_name = 'build'",
                        (time.time() - 10,),
                    )
                    conn.execute(
                        """
                        INSERT INTO builds(build_id, status, created_at, corpus_version, corpus_fingerprint)
                        VALUES ('crashed_staging_build', 'staging', '2026-01-01T00:00:00Z', 'v1', 'fp')
                        """
                    )
            finally:
                conn.close()

            rec = store.recover()
            self.assertEqual(rec["recovered_locks"], 1)
            self.assertIn("crashed_staging_build", rec["aborted_builds"])
            self.assertEqual(store.status()["active_build_id"], first_build_id)

    def test_exact_noop_deletes_and_embedding_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            corpus = _make_sample_corpus(root)
            embedder = _DeterministicFakeEmbedder(dimension=8)

            s1 = store.build(corpus, embedder=embedder)
            b1 = s1["active_build_id"]
            self.assertFalse(s1["last_build"]["noop"])
            self.assertEqual(s1["last_build"]["computed_embeddings"], 5)
            self.assertEqual(s1["last_build"]["cached_embeddings"], 0)
            self.assertEqual(len(embedder.doc_calls), 1)

            # Re-running identical build is an exact no-op
            s2 = store.build(corpus, embedder=embedder)
            self.assertEqual(s2["active_build_id"], b1)
            self.assertTrue(s2["last_build"]["noop"])
            self.assertEqual(len(embedder.doc_calls), 1)

            # Modify 1 chunk and delete 1 chunk: 3 unchanged chunks should hit embedding_cache
            updated_corpus = _make_sample_corpus(root)
            updated_corpus.chunks = updated_corpus.chunks[:-1]
            updated_corpus.chunks[0]["content"] = "Updated backpropagation explanation."

            s3 = store.build(updated_corpus, embedder=embedder)
            b3 = s3["active_build_id"]
            self.assertNotEqual(b3, b1)
            self.assertFalse(s3["last_build"]["noop"])
            self.assertEqual(s3["last_build"]["deleted_chunks"], 1)
            self.assertEqual(s3["last_build"]["updated_chunks"], 1)
            self.assertEqual(s3["last_build"]["unchanged_chunks"], 3)
            self.assertEqual(s3["last_build"]["cached_embeddings"], 3)
            self.assertEqual(s3["last_build"]["computed_embeddings"], 1)
            self.assertEqual(len(embedder.doc_calls[-1]), 1)
            self.assertIsNone(store.get_chunk("nb_code_1_duplicate_content"))
            self.assertIsNotNone(
                store.get_chunk("nb_code_1_duplicate_content", build_id=b1)
            )

    def test_equation_reviews_verification_dense_staleness_and_hash_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            corpus = _make_sample_corpus(root)
            embedder = _DeterministicFakeEmbedder(dimension=8)

            status = store.build(corpus, embedder=embedder)
            self.assertTrue(status["dense_ready"])
            self.assertFalse(status["dense_stale"])
            self.assertFalse(status["release_ready"])

            reviews = store.list_reviews()
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0]["chunk_id"], "book_eq_7_1")
            self.assertEqual(reviews[0]["review_status"], "pending")

            # Empty reviewer must be rejected when decision == 'verified'
            with self.assertRaises(UDLException) as ctx:
                store.review_equation(
                    "book_eq_7_1",
                    transcription=r"\nabla_{\Omega} L = J^T \nabla_f l_i",
                    reviewer="   ",
                    decision="verified",
                )
            self.assertEqual(ctx.exception.code, "invalid_reviewer")

            # Valid verification with edited transcription marks dense_stale=True until rebuild
            rev_result = store.review_equation(
                "book_eq_7_1",
                transcription=r"\nabla_{\Omega} L = J^T \nabla_f l_i",
                reviewer="math_reviewer_1",
                decision="verified",
            )
            self.assertEqual(rev_result["review_status"], "verified")
            self.assertEqual(rev_result["reviewer"], "math_reviewer_1")
            self.assertTrue(rev_result["hashes_valid"])

            after_review_status = store.status()
            self.assertTrue(after_review_status["dense_stale"])
            self.assertFalse(after_review_status["dense_ready"])
            self.assertFalse(after_review_status["release_ready"])
            self.assertIn("dense_index_stale", after_review_status["gates"]["blockers"])

            # Rebuilding with the original corpus applies the persisted review overlay and re-embeds only the edited equation
            calls_before = len(embedder.doc_calls)
            rebuilt_status = store.build(corpus, embedder=embedder)
            self.assertEqual(len(embedder.doc_calls), calls_before + 1)
            self.assertEqual(len(embedder.doc_calls[-1]), 1)
            self.assertTrue(rebuilt_status["dense_ready"])
            self.assertFalse(rebuilt_status["dense_stale"])
            self.assertTrue(rebuilt_status["release_ready"])
            self.assertEqual(
                store.get_chunk("book_eq_7_1")["content"],
                r"\nabla_{\Omega} L = J^T \nabla_f l_i",
            )

            # Modifying the equation crop file on disk invalidates the verified review
            eq_crop_file = root / "data" / "assets" / "eq_7_1.png"
            eq_crop_file.write_bytes(b"TAMPERED_CROP_BYTES")

            post_tamper_reviews = store.list_reviews()
            self.assertEqual(post_tamper_reviews[0]["review_status"], "pending")
            self.assertFalse(post_tamper_reviews[0]["hashes_valid"])
            self.assertFalse(store.status()["release_ready"])

    def test_stat_keyed_file_sha256_cache_hashes_source_once_across_many_equations_and_restats_on_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "data" / "raw"
            assets_dir = root / "data" / "assets"
            raw_dir.mkdir(parents=True, exist_ok=True)
            assets_dir.mkdir(parents=True, exist_ok=True)

            book_file = raw_dir / "UnderstandingDeepLearning.pdf"
            book_bytes = b"SIMULATED_BOOK_PDF_PAYLOAD_VERSION_1"
            book_file.write_bytes(book_bytes)
            book_sha = hashlib.sha256(book_bytes).hexdigest()

            sources = [
                {
                    "source_id": "book",
                    "source_type": "pdf",
                    "path": "data/raw/UnderstandingDeepLearning.pdf",
                    "sha256": book_sha,
                    "revision": "v5.0.2",
                    "title": "Understanding Deep Learning",
                    "metadata": {},
                }
            ]

            num_equations = 50
            chunks = []
            for idx in range(1, num_equations + 1):
                crop_rel = f"data/assets/eq_1_{idx}.png"
                crop_file = root / crop_rel
                crop_bytes = f"CROP_IMAGE_BYTES_{idx}".encode("utf-8")
                crop_file.write_bytes(crop_bytes)
                chunks.append(
                    {
                        "chunk_id": f"book_eq_1_{idx}",
                        "source_id": "book",
                        "source_type": "pdf",
                        "content_type": "equation",
                        "content": f"Equation 1.{idx}: y = w_{idx} x + b_{idx}",
                        "raw_content": f"Equation 1.{idx}: y = w_{idx} x + b_{idx}",
                        "metadata": {
                            "chapter": "1",
                            "pdf_page": 10 + (idx // 5),
                            "equation_number": f"1.{idx}",
                            "asset_path": crop_rel,
                            "review_status": "verified",
                            "reviewer": "auto_verifier",
                        },
                    }
                )

            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            corpus = CorpusResult(sources=sources, chunks=chunks, report={})

            # Building 50 equations sharing 1 book PDF and 50 distinct crop PNGs
            # must read/hash only 1 + 50 = 51 files total (not 50 * 4 = 200 reads).
            store.build(corpus, embedder=None)
            self.assertEqual(store._file_hash_read_count, 1 + num_equations)

            # Repeated status() and list_reviews() calls when files have not changed
            # on disk must do 0 additional file byte reads.
            for _ in range(5):
                st = store.status()
                self.assertEqual(st["counts"]["equations_verified"], num_equations)
                revs = store.list_reviews()
                self.assertEqual(len(revs), num_equations)
            self.assertEqual(store._file_hash_read_count, 1 + num_equations)

            # Modifying 1 crop file on disk updates its stat metadata, triggering
            # exactly 1 new file hash read and invalidating that equation review.
            tampered_crop = assets_dir / "eq_1_1.png"
            tampered_crop.write_bytes(b"MODIFIED_CROP_1_BYTES_LONGER")
            st_after_crop_edit = store.status()
            self.assertEqual(store._file_hash_read_count, 1 + num_equations + 1)
            self.assertEqual(
                st_after_crop_edit["counts"]["equations_verified"],
                num_equations - 1,
            )
            self.assertEqual(st_after_crop_edit["counts"]["equations_pending"], 1)

            # Modifying the shared book PDF on disk updates its stat metadata,
            # triggering exactly 1 new file hash read (not 49) and invalidating
            # all remaining verified equations for that book.
            book_file.write_bytes(b"SIMULATED_BOOK_PDF_PAYLOAD_VERSION_2_CHANGED")
            st_after_book_edit = store.status()
            self.assertEqual(store._file_hash_read_count, 1 + num_equations + 2)
            self.assertEqual(st_after_book_edit["counts"]["equations_verified"], 0)
            self.assertEqual(
                st_after_book_edit["counts"]["equations_pending"], num_equations
            )


if __name__ == "__main__":
    unittest.main()
