"""Composition root shared by command line and web; no RAG framework."""

import threading
import time
import hashlib
import json
from pathlib import Path

from src.contracts import DEFAULT_DB, PROJECT_ROOT, UDLException


class Runtime:
    def __init__(self, db_path=DEFAULT_DB, project_root=PROJECT_ROOT):
        from src.store import SQLiteStore
        self.db_path = Path(db_path)
        self.project_root = Path(project_root).resolve()
        self.store = SQLiteStore(self.db_path, project_root=self.project_root)
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._lock = threading.Lock()
        self._job = {"running": False, "phase": "idle", "error": None}

    def status(self):
        return dict(self._qualify_release(self.store.status()), job=dict(self._job))

    def _qualify_release(self, status):
        """A usable index is not a release until this exact build passed holdout."""
        status = dict(status)
        passed = False
        report_path = self.project_root / "data/evaluation/release.json"
        suite_path = self.project_root / "resources/evaluation.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
            passed = (report.get("passed") is True and report.get("split") == "heldout"
                      and report.get("mode") == "hybrid"
                      and report.get("build_id") == status.get("build_id")
                      and report.get("suite_sha256") == suite_sha)
        except (OSError, ValueError, TypeError):
            pass
        status["retrieval_evaluation_passed"] = passed
        status["release_ready"] = bool(status.get("release_ready") and passed)
        gates = dict(status.get("gates") or {})
        gates["retrieval_evaluation_passed"] = passed
        gates["release_ready"] = status["release_ready"]
        gates["blockers"] = list(gates.get("blockers") or [])
        if not passed:
            gates["blockers"].append("heldout_retrieval_evaluation_missing_or_failed")
        status["gates"] = gates
        status["release_validation"] = dict(gates)
        return status

    def embedder(self, allow_download=False):
        with self._embedder_lock:
            if self._embedder is None:
                from src.dense import LocalEmbedder
                self._embedder = LocalEmbedder(batch_size=1, max_length=2048, local_files_only=not allow_download)
            self._embedder.local_files_only = not allow_download
            return self._embedder

    def service(self, mode="hybrid"):
        from src.search import HybridRetriever
        from src.tutor import TutorService
        ready = self.store.status().get("dense_ready", False)
        embedder = self.embedder() if mode != "lexical" and ready else None
        return TutorService(HybridRetriever(self.store, embedder=embedder))

    def build(self, semantic=False, offline=False):
        if not self._lock.acquire(blocking=False):
            raise UDLException("build_busy", "An index build is already running.", True)
        self._job = {"running": True, "phase": "sources", "error": None, "started_at": time.time()}
        try:
            from src.corpus import ensure_sources, extract_corpus
            ensure_sources(self.project_root, offline=offline)
            self._job["phase"] = "extracting"
            corpus = extract_corpus(project_root=self.project_root, render_assets=True)
            from src.review_bundle import apply_review_bundle
            corpus = apply_review_bundle(corpus, self.project_root)
            from src.snapshots import snapshot_corpus
            corpus = snapshot_corpus(corpus, self.project_root)
            from src.snapshots import file_sha256
            corpus.report["pipeline_fingerprints"] = {
                name: file_sha256(Path(__file__).parent / name)
                for name in ("corpus.py", "search.py", "snapshots.py", "review_bundle.py")
            }
            self._job["phase"] = "embedding" if semantic else "indexing"
            report = self._qualify_release(self.store.build(corpus, embedder=self.embedder(allow_download=not offline) if semantic else None))
            self._job.update(running=False, phase="complete", report=report, finished_at=time.time())
            return report
        except Exception as exc:
            error = exc.as_dict() if isinstance(exc, UDLException) else {"code": "build_failed", "message": "The build failed. Run the CLI build command for a diagnostic traceback.", "retryable": True}
            self._job.update(running=False, phase="failed", error=error, finished_at=time.time())
            raise
        finally:
            self._lock.release()

    def start_build(self, semantic=False, offline=False):
        if self._job.get("running"):
            raise UDLException("build_busy", "An index build is already running.", True)
        # Set before starting the thread to prevent a second request racing it.
        self._job = {"running": True, "phase": "starting", "error": None}

        def run():
            try:
                self.build(semantic, offline)
            except Exception:
                pass  # Sanitized error is available through status().
        threading.Thread(target=run, name="udl-index-build", daemon=True).start()
        return self.status()
