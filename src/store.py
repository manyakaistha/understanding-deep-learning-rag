"""Authoritative SQLite store for the UDL retrieval pipeline.

Provides:
- Schema migrations with WAL mode, foreign keys, and automatic safe SQLite backups.
- Single-writer lock with crash recovery for interrupted staging builds.
- Atomic staged builds with an active build pointer (`store_state.active_build_id`)
  that preserves the prior usable build on any failure.
- Distinct provenance keys for duplicate-content chunks, source/chunk deletions,
  and exact no-op build detection.
- Full-input + model-profile embedding cache (`embedding_cache`) and float32 BLOB
  vector storage (`chunk_embeddings`).
- Equation review persistence (`list_reviews`, `review_equation`) with source,
  crop, and transcription SHA-256 verification, overlay persistence, automatic
  invalidation on hash changes, and marking dense embeddings stale when review
  changes a chunk's embedding input.
- Separate readiness reporting for usable lexical search, usable dense search,
  and validated release readiness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

from src.contracts import (
    CORPUS_VERSION,
    DEFAULT_DB,
    PROJECT_ROOT,
    CorpusResult,
    UDLException,
)
from src.dense import (
    canonical_profile_dict,
    compute_profile_fingerprint,
    embedding_cache_key,
    pack_vector,
    unpack_vector,
)
from src.search import (
    chunk_matches_modality,
    embedding_text,
    extract_chunk_anchors,
    extract_metadata_tokens,
    locator,
    tokenize,
)

LOCK_TTL_SECONDS = 1800.0

MIGRATIONS: List[Tuple[int, str, str]] = [
    (
        1,
        "initial_udl_store_schema",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            checksum TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS writer_locks (
            lock_name TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            pid INTEGER NOT NULL,
            build_id TEXT,
            acquired_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS builds (
            build_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('staging', 'ready', 'failed', 'aborted', 'superseded')),
            created_at TEXT NOT NULL,
            completed_at TEXT,
            corpus_version TEXT NOT NULL,
            corpus_fingerprint TEXT NOT NULL,
            lexical_ready INTEGER NOT NULL DEFAULT 0,
            dense_ready INTEGER NOT NULL DEFAULT 0,
            dense_stale INTEGER NOT NULL DEFAULT 0,
            release_ready INTEGER NOT NULL DEFAULT 0,
            embedding_model TEXT,
            embedding_revision TEXT,
            embedding_dimension INTEGER,
            embedding_fingerprint TEXT,
            embedding_profile_json TEXT,
            stats_json TEXT NOT NULL DEFAULT '{}',
            report_json TEXT NOT NULL DEFAULT '{}',
            parent_build_id TEXT
        );

        CREATE TABLE IF NOT EXISTS store_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            active_build_id TEXT REFERENCES builds(build_id) ON DELETE SET NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            build_id TEXT NOT NULL REFERENCES builds(build_id) ON DELETE CASCADE,
            source_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            revision TEXT NOT NULL,
            title TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (build_id, source_id)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            build_id TEXT NOT NULL REFERENCES builds(build_id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content_type TEXT NOT NULL,
            content TEXT NOT NULL,
            raw_content TEXT,
            embedding_text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            provenance_key TEXT NOT NULL,
            review_status TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (build_id, chunk_id),
            FOREIGN KEY (build_id, source_id) REFERENCES sources(build_id, source_id) ON DELETE CASCADE,
            UNIQUE (build_id, provenance_key)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_build_ordinal
            ON chunks(build_id, ordinal);
        CREATE INDEX IF NOT EXISTS idx_chunks_build_source
            ON chunks(build_id, source_id, ordinal);

        CREATE TABLE IF NOT EXISTS chunk_stats (
            build_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            doc_length REAL NOT NULL,
            content_token_count INTEGER NOT NULL,
            meta_token_count INTEGER NOT NULL,
            anchors_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (build_id, chunk_id),
            FOREIGN KEY (build_id, chunk_id) REFERENCES chunks(build_id, chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS bm25_postings (
            build_id TEXT NOT NULL,
            term TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            tf_content INTEGER NOT NULL,
            tf_meta INTEGER NOT NULL,
            PRIMARY KEY (build_id, term, chunk_id),
            FOREIGN KEY (build_id, chunk_id) REFERENCES chunks(build_id, chunk_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_bm25_postings_lookup
            ON bm25_postings(build_id, term);
        CREATE INDEX IF NOT EXISTS idx_bm25_postings_chunk
            ON bm25_postings(build_id, chunk_id);

        CREATE TABLE IF NOT EXISTS bm25_terms (
            build_id TEXT NOT NULL REFERENCES builds(build_id) ON DELETE CASCADE,
            term TEXT NOT NULL,
            df INTEGER NOT NULL,
            PRIMARY KEY (build_id, term)
        );

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            build_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            embedding_fingerprint TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (build_id, chunk_id),
            FOREIGN KEY (build_id, chunk_id) REFERENCES chunks(build_id, chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            embedding_fingerprint TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS equation_reviews (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            equation_number TEXT,
            transcription TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('pending', 'verified')),
            source_sha256 TEXT NOT NULL,
            crop_sha256 TEXT NOT NULL,
            transcription_sha256 TEXT NOT NULL,
            original_transcription_sha256 TEXT NOT NULL,
            build_id TEXT,
            updated_at TEXT NOT NULL
        );
        """,
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _is_equation_chunk(chunk: Mapping[str, Any]) -> bool:
    ctype = str(chunk.get("content_type") or "").strip().lower()
    meta = chunk.get("metadata") or {}
    return (
        ctype in {"equation", "math", "display_equation", "formula"}
        or meta.get("equation_number") is not None
    )


def _is_figure_chunk(chunk: Mapping[str, Any]) -> bool:
    ctype = str(chunk.get("content_type") or "").strip().lower()
    meta = chunk.get("metadata") or {}
    if meta.get("algorithm_number") is not None or ctype == "algorithm":
        return False
    return (
        ctype in {"figure", "image", "table"}
        or meta.get("figure_number") is not None
        or meta.get("figure_name") is not None
    )


def _is_algorithm_chunk(chunk: Mapping[str, Any]) -> bool:
    ctype = str(chunk.get("content_type") or "").strip().lower()
    meta = chunk.get("metadata") or {}
    return ctype == "algorithm" or meta.get("algorithm_number") is not None


def _compute_provenance_key(chunk: Mapping[str, Any]) -> str:
    """Compute a unique provenance key that remains distinct for identical content."""
    meta = chunk.get("metadata") or {}
    payload = {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "source_id": str(chunk.get("source_id") or ""),
        "source_type": str(chunk.get("source_type") or ""),
        "content_type": str(chunk.get("content_type") or ""),
        "pdf_page": meta.get("pdf_page"),
        "book_page": meta.get("book_page"),
        "cell_index": meta.get("cell_index"),
        "start_line": meta.get("start_line"),
        "end_line": meta.get("end_line"),
        "bbox": meta.get("bbox"),
        "figure_number": meta.get("figure_number") or meta.get("figure_name"),
        "equation_number": meta.get("equation_number"),
        "algorithm_number": meta.get("algorithm_number"),
    }
    return _sha256_text(_canonical_json(payload))


class SQLiteStore:
    """SQLite primary store for sources, chunks, BM25 postings, vectors, and reviews."""

    def __init__(
        self,
        path: Union[str, Path] = DEFAULT_DB,
        *,
        project_root: Optional[Union[str, Path]] = None,
        lock_ttl_seconds: float = LOCK_TTL_SECONDS,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = PROJECT_ROOT
        self.lock_ttl_seconds = float(lock_ttl_seconds)

        self._search_index_cache: Dict[str, Dict[str, Any]] = {}
        self._dense_vector_cache: Dict[str, Dict[str, Tuple[float, ...]]] = {}
        self._file_sha256_cache: Dict[Tuple[str, int, int, int, int, int], str] = {}
        self._file_hash_read_count: int = 0

        self._initialize_database()
        self.recover()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def backup(self, target_path: Optional[Union[str, Path]] = None) -> Path:
        """Create a consistent online backup of the SQLite database."""
        if target_path is None:
            backup_dir = self.path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = backup_dir / f"{self.path.stem}_{stamp}_{uuid.uuid4().hex[:6]}.sqlite3"
        else:
            dest = Path(target_path).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)

        src_conn = self._connect()
        dst_conn = sqlite3.connect(str(dest))
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()
        return dest

    def _initialize_database(self) -> None:
        has_existing_data = self.path.exists() and self.path.stat().st_size > 0
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                """
            )
            applied = {
                int(row["version"]): str(row["checksum"])
                for row in conn.execute(
                    "SELECT version, checksum FROM schema_migrations"
                ).fetchall()
            }
            pending = [
                (ver, name, sql)
                for ver, name, sql in MIGRATIONS
                if ver not in applied
            ]
            for ver, name, sql in MIGRATIONS:
                checksum = _sha256_text(sql.strip())
                if ver in applied and applied[ver] != checksum:
                    raise UDLException(
                        "schema_migration_mismatch",
                        f"Migration {ver} ({name}) checksum mismatch in {self.path}.",
                        retryable=False,
                    )
        finally:
            conn.close()

        if pending and has_existing_data:
            self.backup()

        if pending:
            conn = self._connect()
            try:
                with conn:
                    for ver, name, sql in pending:
                        checksum = _sha256_text(sql.strip())
                        conn.executescript(sql)
                        conn.execute("PRAGMA foreign_keys = ON;")
                        conn.execute(
                            """
                            INSERT INTO schema_migrations(version, name, applied_at, checksum)
                            VALUES (?, ?, ?, ?)
                            """,
                            (ver, name, _utc_now(), checksum),
                        )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO store_state(id, active_build_id, updated_at)
                        VALUES (1, NULL, ?)
                        """,
                        (_utc_now(),),
                    )
            finally:
                conn.close()
        else:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO store_state(id, active_build_id, updated_at)
                        VALUES (1, NULL, ?)
                        """,
                        (_utc_now(),),
                    )
            finally:
                conn.close()

    def recover(self) -> Dict[str, Any]:
        """Recover from interrupted builds and expired/dead-process writer locks.

        Never modifies `store_state.active_build_id` or committed ready builds.
        """
        now = time.time()
        recovered_locks = 0
        aborted_builds: List[str] = []

        conn = self._connect()
        try:
            with conn:
                locks = conn.execute(
                    "SELECT * FROM writer_locks WHERE lock_name = 'build'"
                ).fetchall()
                active_lock_held = False
                for lock in locks:
                    expired = float(lock["expires_at"]) <= now
                    dead_proc = not _is_pid_alive(int(lock["pid"]))
                    if expired or dead_proc:
                        conn.execute(
                            "DELETE FROM writer_locks WHERE lock_name = ?",
                            (lock["lock_name"],),
                        )
                        recovered_locks += 1
                    else:
                        active_lock_held = True

                if not active_lock_held:
                    staging_rows = conn.execute(
                        "SELECT build_id FROM builds WHERE status = 'staging'"
                    ).fetchall()
                    for row in staging_rows:
                        bid = str(row["build_id"])
                        aborted_builds.append(bid)
                        conn.execute(
                            "DELETE FROM builds WHERE build_id = ? AND status = 'staging'",
                            (bid,),
                        )
        finally:
            conn.close()

        return {
            "recovered_locks": recovered_locks,
            "aborted_builds": aborted_builds,
        }

    def _acquire_writer_lock(self, build_id: str) -> str:
        owner_id = uuid.uuid4().hex
        now = time.time()
        expires = now + self.lock_ttl_seconds
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM writer_locks WHERE lock_name = 'build'"
            ).fetchone()
            if row is not None:
                is_expired = float(row["expires_at"]) <= now
                is_dead = not _is_pid_alive(int(row["pid"]))
                if not is_expired and not is_dead:
                    conn.rollback()
                    raise UDLException(
                        "build_busy",
                        f"Another build ('{row['build_id']}') is currently holding the writer lock.",
                        retryable=True,
                    )
                conn.execute("DELETE FROM writer_locks WHERE lock_name = 'build'")

            conn.execute(
                """
                INSERT INTO writer_locks(lock_name, owner_id, pid, build_id, acquired_at, expires_at)
                VALUES ('build', ?, ?, ?, ?, ?)
                """,
                (owner_id, os.getpid(), build_id, now, expires),
            )
            conn.commit()
            return owner_id
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _release_writer_lock(self, owner_id: str) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM writer_locks WHERE lock_name = 'build' AND owner_id = ?",
                    (owner_id,),
                )
        finally:
            conn.close()

    def _resolve_asset_file(self, asset_path: Optional[str]) -> Optional[Path]:
        if not asset_path or not str(asset_path).strip():
            return None
        raw = Path(str(asset_path).strip())
        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    self.project_root / raw,
                    self.path.parent.parent / raw,
                    self.path.parent / raw,
                    PROJECT_ROOT / raw,
                ]
            )
        for cand in candidates:
            try:
                if cand.is_file():
                    return cand.resolve()
            except OSError:
                continue
        return None

    def _hash_file_cached(self, resolved_path: Path) -> Optional[str]:
        """Compute and cache file SHA-256 keyed by canonical path and stat metadata.

        Always restats the file on each call so any in-place edit, size change,
        inode replacement, or timestamp update immediately invalidates the entry.
        """
        try:
            canonical = resolved_path.resolve()
            st = canonical.stat()
        except OSError:
            return None
        cache_key = (
            str(canonical),
            int(st.st_dev),
            int(st.st_ino),
            int(st.st_size),
            int(st.st_mtime_ns),
            int(st.st_ctime_ns),
        )
        cached = self._file_sha256_cache.get(cache_key)
        if cached is not None:
            return cached

        digest = hashlib.sha256()
        try:
            with canonical.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None

        sha = digest.hexdigest()
        self._file_hash_read_count += 1
        # Evict any older stat entries for the same canonical path
        stale_keys = [
            k for k in self._file_sha256_cache if k[0] == str(canonical) and k != cache_key
        ]
        for k in stale_keys:
            self._file_sha256_cache.pop(k, None)
        self._file_sha256_cache[cache_key] = sha
        return sha

    def _compute_crop_sha256(self, chunk: Mapping[str, Any]) -> Optional[str]:
        meta = chunk.get("metadata") or {}
        asset_path = meta.get("asset_path") or chunk.get("asset_path")
        resolved = self._resolve_asset_file(asset_path)
        if resolved is not None:
            hashed = self._hash_file_cached(resolved)
            if hashed is not None:
                return hashed
        if asset_path and meta.get("_crop_from_file"):
            return None
        explicit_hash = meta.get("crop_sha256")
        if explicit_hash and str(explicit_hash).strip():
            return str(explicit_hash).strip()
        return None

    def _compute_source_sha256(
        self,
        chunk: Mapping[str, Any],
        sources_by_id: Mapping[str, Mapping[str, Any]],
    ) -> Optional[str]:
        source_id = str(chunk.get("source_id") or "")
        src = sources_by_id.get(source_id) or {}
        src_path = src.get("path")
        if src_path:
            resolved_src = self._resolve_asset_file(str(src_path))
            if resolved_src is not None:
                hashed = self._hash_file_cached(resolved_src)
                if hashed is not None:
                    return hashed
        src_sha = src.get("sha256")
        if src_sha and str(src_sha).strip():
            return str(src_sha).strip()
        meta = chunk.get("metadata") or {}
        if meta.get("source_sha256"):
            return str(meta["source_sha256"]).strip()
        return None

    def _normalize_and_validate_corpus(
        self, corpus: Union[CorpusResult, Mapping[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if isinstance(corpus, CorpusResult):
            raw_sources = corpus.sources
            raw_chunks = corpus.chunks
            raw_report = corpus.report or {}
        elif isinstance(corpus, Mapping):
            raw_sources = corpus.get("sources") or []
            raw_chunks = corpus.get("chunks") or []
            raw_report = corpus.get("report") or {}
        else:
            raise UDLException(
                "invalid_corpus",
                "build() requires a CorpusResult or dict with 'sources' and 'chunks'.",
                retryable=False,
            )

        if not isinstance(raw_sources, Sequence) or not isinstance(
            raw_chunks, Sequence
        ):
            raise UDLException(
                "invalid_corpus",
                "Corpus 'sources' and 'chunks' must be sequences of dictionaries.",
                retryable=False,
            )

        sources: List[Dict[str, Any]] = []
        seen_sources: set[str] = set()
        for idx, src in enumerate(raw_sources):
            if not isinstance(src, Mapping):
                raise UDLException(
                    "invalid_source_record",
                    f"Source at index {idx} must be a dict.",
                    retryable=False,
                )
            source_id = str(src.get("source_id") or "").strip()
            source_type = str(src.get("source_type") or "").strip()
            path_str = str(src.get("path") or source_id).strip()
            if not source_id or not source_type or not path_str:
                raise UDLException(
                    "invalid_source_record",
                    f"Source at index {idx} is missing required fields (source_id, source_type, path).",
                    retryable=False,
                )
            if source_id in seen_sources:
                raise UDLException(
                    "duplicate_source_id",
                    f"Duplicate source_id '{source_id}' in corpus.",
                    retryable=False,
                )
            seen_sources.add(source_id)
            sha256 = str(src.get("sha256") or _sha256_text(f"{source_id}:{path_str}")).strip()
            revision = str(src.get("revision") or CORPUS_VERSION).strip()
            title = str(src.get("title") or source_id).strip()
            meta = dict(src.get("metadata") or {})
            sources.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "path": path_str,
                    "sha256": sha256,
                    "revision": revision,
                    "title": title,
                    "metadata": meta,
                }
            )

        chunks: List[Dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_provenance: set[str] = set()
        for idx, ch in enumerate(raw_chunks):
            if not isinstance(ch, Mapping):
                raise UDLException(
                    "invalid_chunk_record",
                    f"Chunk at index {idx} must be a dict.",
                    retryable=False,
                )
            chunk_id = str(ch.get("chunk_id") or "").strip()
            source_id = str(ch.get("source_id") or "").strip()
            source_type = str(ch.get("source_type") or "").strip()
            content_type = str(ch.get("content_type") or "").strip()
            content = ch.get("content")
            if (
                not chunk_id
                or not source_id
                or not source_type
                or not content_type
                or content is None
            ):
                raise UDLException(
                    "invalid_chunk_record",
                    f"Chunk at index {idx} ('{chunk_id}') is missing required fields.",
                    retryable=False,
                )
            if chunk_id in seen_chunks:
                raise UDLException(
                    "duplicate_chunk_id",
                    f"Duplicate chunk_id '{chunk_id}' in corpus.",
                    retryable=False,
                )
            if source_id not in seen_sources:
                raise UDLException(
                    "unknown_chunk_source",
                    f"Chunk '{chunk_id}' references unknown source_id '{source_id}'.",
                    retryable=False,
                )
            seen_chunks.add(chunk_id)

            clean_content = str(content).replace("\x00", "")
            raw_content = ch.get("raw_content")
            clean_raw = (
                str(raw_content).replace("\x00", "")
                if raw_content is not None
                else clean_content
            )
            meta = json.loads(_canonical_json(dict(ch.get("metadata") or {})))
            asset_path = meta.get("asset_path") or ch.get("asset_path")
            if asset_path is not None:
                meta["asset_path"] = str(asset_path)
                if self._resolve_asset_file(str(asset_path)) is not None:
                    meta["_crop_from_file"] = True

            norm_chunk: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "source_type": source_type,
                "content_type": content_type,
                "content": clean_content,
                "raw_content": clean_raw,
                "metadata": meta,
                "embedding_text": ch.get("embedding_text"),
            }
            prov_key = _compute_provenance_key(norm_chunk)
            if prov_key in seen_provenance:
                raise UDLException(
                    "duplicate_provenance_key",
                    f"Chunk '{chunk_id}' has a colliding provenance key.",
                    retryable=False,
                )
            seen_provenance.add(prov_key)
            chunks.append(norm_chunk)

        return sources, chunks, dict(raw_report)

    def _apply_review_overlays(
        self,
        conn: sqlite3.Connection,
        sources: Sequence[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Apply persisted equation review overlays and validate source/crop/transcription hashes."""
        sources_by_id = {s["source_id"]: s for s in sources}
        persisted_rows = conn.execute("SELECT * FROM equation_reviews").fetchall()
        persisted_by_id = {str(r["chunk_id"]): dict(r) for r in persisted_rows}

        for chunk in chunks:
            if not _is_equation_chunk(chunk):
                continue

            cid = chunk["chunk_id"]
            meta = chunk["metadata"]
            src_hash = self._compute_source_sha256(chunk, sources_by_id)
            crop_hash = self._compute_crop_sha256(chunk)
            incoming_trans_hash = _sha256_text(chunk["content"].strip())

            overlay = persisted_by_id.get(cid)
            # A newly source-compared bundle can legitimately replace a review
            # for an older crop/transcription. Do not let that stale overlay
            # invalidate the fresh, independently hash-validated review.
            if (overlay is not None
                    and meta.get("review_status") == "verified"
                    and str(meta.get("reviewer") or "").strip()
                    and str(meta.get("review_method") or "").strip()
                    and meta.get("source_sha256") == src_hash
                    and meta.get("crop_sha256") == crop_hash
                    and meta.get("transcription_sha256") == incoming_trans_hash
                    and (src_hash, crop_hash, incoming_trans_hash) != (
                        overlay["source_sha256"], overlay["crop_sha256"], overlay["transcription_sha256"])):
                overlay = None
            if overlay is not None:
                hashes_match = (
                    bool(src_hash)
                    and bool(crop_hash)
                    and src_hash == overlay["source_sha256"]
                    and crop_hash == overlay["crop_sha256"]
                    and incoming_trans_hash
                    in {
                        overlay["original_transcription_sha256"],
                        overlay["transcription_sha256"],
                    }
                )
                if hashes_match:
                    had_custom_emb = bool(chunk.get("embedding_text"))
                    old_content = chunk["content"]
                    new_content = str(overlay["transcription"])
                    chunk["content"] = new_content
                    meta["transcription"] = new_content
                    meta["reviewer"] = overlay["reviewer"]
                    meta["source_sha256"] = src_hash
                    meta["crop_sha256"] = crop_hash
                    meta["transcription_sha256"] = overlay["transcription_sha256"]

                    if (
                        overlay["decision"] == "verified"
                        and str(overlay["reviewer"]).strip()
                    ):
                        meta["review_status"] = "verified"
                    else:
                        meta["review_status"] = "pending"

                    if new_content != old_content or not had_custom_emb:
                        chunk["embedding_text"] = embedding_text(
                            chunk, force_recompute=True
                        )
                else:
                    meta["review_status"] = "pending"
                    meta["review_invalidated"] = True
                    meta["source_sha256"] = src_hash
                    meta["crop_sha256"] = crop_hash
                    meta["transcription_sha256"] = incoming_trans_hash
                    conn.execute(
                        """
                        UPDATE equation_reviews
                        SET decision = 'pending', updated_at = ?
                        WHERE chunk_id = ?
                        """,
                        (_utc_now(), cid),
                    )
            else:
                reviewer = str(meta.get("reviewer") or "").strip()
                claimed_status = str(meta.get("review_status") or "pending").strip().lower()
                if (
                    claimed_status == "verified"
                    and reviewer
                    and src_hash
                    and crop_hash
                ):
                    meta["review_status"] = "verified"
                    meta["source_sha256"] = src_hash
                    meta["crop_sha256"] = crop_hash
                    meta["transcription_sha256"] = incoming_trans_hash
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO equation_reviews(
                            chunk_id, source_id, equation_number, transcription,
                            reviewer, decision, source_sha256, crop_sha256,
                            transcription_sha256, original_transcription_sha256,
                            build_id, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?, NULL, ?)
                        """,
                        (
                            cid,
                            chunk["source_id"],
                            str(meta.get("equation_number") or ""),
                            chunk["content"],
                            reviewer,
                            src_hash,
                            crop_hash,
                            incoming_trans_hash,
                            incoming_trans_hash,
                            _utc_now(),
                        ),
                    )
                else:
                    meta["review_status"] = "pending"
                    meta["source_sha256"] = src_hash
                    meta["crop_sha256"] = crop_hash
                    meta["transcription_sha256"] = incoming_trans_hash

            if not chunk.get("embedding_text"):
                chunk["embedding_text"] = embedding_text(chunk)

        for chunk in chunks:
            if not chunk.get("embedding_text"):
                chunk["embedding_text"] = embedding_text(chunk)

    def _compute_corpus_fingerprint(
        self,
        sources: Sequence[Mapping[str, Any]],
        chunks: Sequence[Mapping[str, Any]],
        report: Mapping[str, Any],
    ) -> str:
        payload = {
            "corpus_version": CORPUS_VERSION,
            "sources": [
                {
                    "source_id": s["source_id"],
                    "source_type": s["source_type"],
                    "path": s["path"],
                    "sha256": s["sha256"],
                    "revision": s["revision"],
                    "title": s["title"],
                    "metadata": s["metadata"],
                }
                for s in sources
            ],
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "source_id": c["source_id"],
                    "source_type": c["source_type"],
                    "content_type": c["content_type"],
                    "content": c["content"],
                    "raw_content": c["raw_content"],
                    "embedding_text": c["embedding_text"],
                    "metadata": c["metadata"],
                }
                for c in chunks
            ],
            "report": report,
        }
        return _sha256_text(_canonical_json(payload))

    def _evaluate_release_gates(
        self,
        sources: Sequence[Mapping[str, Any]],
        chunks: Sequence[Mapping[str, Any]],
        report: Mapping[str, Any],
        *,
        lexical_ready: bool,
        dense_ready: bool,
        dense_stale: bool,
    ) -> Dict[str, Any]:
        sources_by_id = {s["source_id"]: s for s in sources}
        by_content_type: Counter[str] = Counter()
        by_modality: Counter[str] = Counter()

        equations_total = 0
        equations_verified = 0
        equations_pending = 0

        figures_total = 0
        figures_verified = 0
        figures_pending = 0

        algorithms_total = 0
        algorithms_verified = 0
        algorithms_pending = 0

        missing_asset_files = 0

        for ch in chunks:
            ctype = str(ch.get("content_type") or "unknown")
            by_content_type[ctype] += 1
            for mod in ("text", "code", "figure", "exercise", "equation"):
                if chunk_matches_modality(ch, mod):
                    by_modality[mod] += 1

            meta = ch.get("metadata") or {}
            asset_path = meta.get("asset_path") or ch.get("asset_path")

            if _is_equation_chunk(ch):
                equations_total += 1
                src_hash = self._compute_source_sha256(ch, sources_by_id)
                crop_hash = self._compute_crop_sha256(ch)
                trans_hash = _sha256_text(str(ch.get("content") or "").strip())
                reviewer = str(meta.get("reviewer") or "").strip()
                status_val = str(meta.get("review_status") or "pending").strip().lower()
                if not crop_hash:
                    missing_asset_files += 1
                if (
                    status_val == "verified"
                    and reviewer
                    and src_hash
                    and crop_hash
                    and src_hash == meta.get("source_sha256")
                    and crop_hash == meta.get("crop_sha256")
                    and trans_hash == meta.get("transcription_sha256")
                ):
                    equations_verified += 1
                else:
                    equations_pending += 1

            elif _is_algorithm_chunk(ch):
                algorithms_total += 1
                # A recorded digest cannot replace the actual citation crop.
                # Read the file directly so missing/unreadable snapshots and
                # changed reviewed bytes both close the release gate.
                crop_file = self._resolve_asset_file(asset_path)
                crop_hash = self._hash_file_cached(crop_file) if crop_file else None
                status_val = str(meta.get("review_status") or "").strip().lower()
                if not crop_hash:
                    missing_asset_files += 1
                    algorithms_pending += 1
                elif (
                    status_val == "pending"
                    or (meta.get("crop_sha256") and crop_hash != meta["crop_sha256"])
                ):
                    algorithms_pending += 1
                else:
                    algorithms_verified += 1

            elif _is_figure_chunk(ch):
                figures_total += 1
                crop_file = self._resolve_asset_file(asset_path)
                crop_hash = self._hash_file_cached(crop_file) if crop_file else None
                status_val = str(meta.get("review_status") or "").strip().lower()
                if not crop_hash:
                    missing_asset_files += 1
                    figures_pending += 1
                elif (
                    status_val == "pending"
                    or (meta.get("crop_sha256") and crop_hash != meta["crop_sha256"])
                ):
                    figures_pending += 1
                else:
                    figures_verified += 1

        is_full_book = any(s["source_id"] == "book" for s in sources) and len(chunks) >= 500
        expected_figures = int(
            report.get("expected_figures", 285 if is_full_book else figures_total)
        )
        expected_algorithms = int(
            report.get("expected_algorithms", 2 if is_full_book else algorithms_total)
        )
        notebook_coverage_complete = bool(
            report.get("notebook_coverage_complete", True)
        )

        figure_target_met = (
            figures_pending == 0 and figures_verified >= expected_figures
        )
        algorithm_target_met = (
            algorithms_pending == 0 and algorithms_verified >= expected_algorithms
        )
        equation_reviews_complete = equations_pending == 0 and (
            not is_full_book or equations_total > 0
        )

        blockers: List[str] = []
        if not lexical_ready:
            blockers.append("lexical_index_not_ready")
        if not dense_ready:
            blockers.append("dense_index_missing")
        if dense_stale:
            blockers.append("dense_index_stale")
        if equations_pending > 0:
            blockers.append(f"pending_equation_reviews:{equations_pending}")
        if is_full_book and equations_total == 0:
            blockers.append("display_equations_not_extracted")
        if figures_pending > 0:
            blockers.append(f"pending_figure_reviews:{figures_pending}")
        if figures_verified < expected_figures:
            blockers.append(
                f"figure_crops_incomplete:{figures_verified}/{expected_figures}"
            )
        if algorithms_pending > 0:
            blockers.append(f"pending_algorithm_reviews:{algorithms_pending}")
        if algorithms_verified < expected_algorithms:
            blockers.append(
                f"algorithm_crops_incomplete:{algorithms_verified}/{expected_algorithms}"
            )
        if missing_asset_files > 0:
            blockers.append(f"missing_asset_files:{missing_asset_files}")
        if not notebook_coverage_complete:
            blockers.append("notebook_cell_coverage_incomplete")
        for extra_blocker in report.get("release_blockers") or []:
            if str(extra_blocker) not in blockers:
                blockers.append(str(extra_blocker))

        release_ready = len(blockers) == 0

        return {
            "counts": {
                "sources": len(sources),
                "chunks": len(chunks),
                "by_content_type": dict(sorted(by_content_type.items())),
                "by_modality": dict(sorted(by_modality.items())),
                "equations_total": equations_total,
                "equations_verified": equations_verified,
                "equations_pending": equations_pending,
                "figures_total": figures_total,
                "figures_verified": figures_verified,
                "figures_pending": figures_pending,
                "algorithms_total": algorithms_total,
                "algorithms_verified": algorithms_verified,
                "algorithms_pending": algorithms_pending,
            },
            "gates": {
                "lexical_usable": bool(lexical_ready),
                "dense_usable": bool(dense_ready and not dense_stale),
                "dense_stale": bool(dense_stale),
                "pending_equation_reviews": equations_pending,
                "pending_figure_reviews": figures_pending,
                "pending_algorithm_reviews": algorithms_pending,
                "missing_asset_files": missing_asset_files,
                "notebook_coverage_complete": notebook_coverage_complete,
                "figure_target_met": figure_target_met,
                "algorithm_target_met": algorithm_target_met,
                "equation_reviews_complete": equation_reviews_complete,
                "release_ready": release_ready,
                "blockers": blockers,
            },
        }

    def _get_active_build_row(
        self, conn: sqlite3.Connection, build_id: Optional[str] = None
    ) -> Optional[sqlite3.Row]:
        if build_id is not None:
            return conn.execute(
                "SELECT * FROM builds WHERE build_id = ? AND status IN ('ready', 'superseded')",
                (str(build_id),),
            ).fetchone()
        state = conn.execute(
            "SELECT active_build_id FROM store_state WHERE id = 1"
        ).fetchone()
        if not state or not state["active_build_id"]:
            return None
        return conn.execute(
            "SELECT * FROM builds WHERE build_id = ?",
            (state["active_build_id"],),
        ).fetchone()

    def build(
        self,
        corpus: Union[CorpusResult, Mapping[str, Any]],
        embedder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Stage and atomically activate a corpus build with optional dense embeddings."""
        self.recover()
        sources, chunks, report = self._normalize_and_validate_corpus(corpus)

        new_build_id = f"build_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        lock_owner = self._acquire_writer_lock(new_build_id)

        try:
            conn = self._connect()
            try:
                self._apply_review_overlays(conn, sources, chunks)
                corpus_fp = self._compute_corpus_fingerprint(
                    sources, chunks, report
                )

                emb_profile: Optional[Dict[str, Any]] = None
                emb_fp: Optional[str] = None
                emb_dim: Optional[int] = None
                emb_model: Optional[str] = None
                emb_rev: Optional[str] = None
                if embedder is not None:
                    raw_prof = getattr(embedder, "profile", None)
                    if callable(raw_prof):
                        raw_prof = raw_prof()
                    if not isinstance(raw_prof, Mapping):
                        raise UDLException(
                            "invalid_embedder",
                            "Embedder must expose a JSON-compatible 'profile' dictionary.",
                            retryable=False,
                        )
                    emb_profile = canonical_profile_dict(raw_prof)
                    emb_fp = str(
                        getattr(embedder, "fingerprint", None)
                        or compute_profile_fingerprint(emb_profile)
                    )
                    emb_dim = int(emb_profile["dimension"])
                    emb_model = str(emb_profile["model_name"])
                    emb_rev = str(emb_profile["revision"])

                prev_build = self._get_active_build_row(conn)
                prev_build_id = (
                    str(prev_build["build_id"]) if prev_build is not None else None
                )

                # Check for exact no-op update against active build
                if prev_build is not None and str(prev_build["corpus_fingerprint"]) == corpus_fp:
                    prev_dense_ready = bool(prev_build["dense_ready"]) and not bool(
                        prev_build["dense_stale"]
                    )
                    prev_emb_fp = prev_build["embedding_fingerprint"]
                    if embedder is None and bool(prev_build["lexical_ready"]):
                        stats = json.loads(prev_build["stats_json"] or "{}")
                        stats["noop"] = True
                        stats["added_chunks"] = 0
                        stats["updated_chunks"] = 0
                        stats["deleted_chunks"] = 0
                        stats["unchanged_chunks"] = len(chunks)
                        stats["added_sources"] = 0
                        stats["updated_sources"] = 0
                        stats["deleted_sources"] = 0
                        stats["computed_embeddings"] = 0
                        stats["cached_embeddings"] = (
                            len(chunks) if prev_dense_ready else 0
                        )
                        with conn:
                            conn.execute(
                                "UPDATE builds SET stats_json = ? WHERE build_id = ?",
                                (_canonical_json(stats), prev_build_id),
                            )
                        return self.status()

                    if (
                        embedder is not None
                        and prev_dense_ready
                        and prev_emb_fp == emb_fp
                    ):
                        stats = json.loads(prev_build["stats_json"] or "{}")
                        stats["noop"] = True
                        stats["added_chunks"] = 0
                        stats["updated_chunks"] = 0
                        stats["deleted_chunks"] = 0
                        stats["unchanged_chunks"] = len(chunks)
                        stats["added_sources"] = 0
                        stats["updated_sources"] = 0
                        stats["deleted_sources"] = 0
                        stats["computed_embeddings"] = 0
                        stats["cached_embeddings"] = len(chunks)
                        with conn:
                            conn.execute(
                                "UPDATE builds SET stats_json = ? WHERE build_id = ?",
                                (_canonical_json(stats), prev_build_id),
                            )
                        return self.status()

                prev_sources_map: Dict[str, str] = {}
                prev_chunks_map: Dict[str, str] = {}
                if prev_build_id is not None:
                    for r in conn.execute(
                        "SELECT source_id, sha256, revision, path, title, metadata_json FROM sources WHERE build_id = ?",
                        (prev_build_id,),
                    ).fetchall():
                        prev_sources_map[str(r["source_id"])] = _sha256_text(
                            f"{r['sha256']}:{r['revision']}:{r['path']}:{r['title']}:{r['metadata_json']}"
                        )
                    for r in conn.execute(
                        "SELECT chunk_id, content_hash, embedding_text, metadata_json FROM chunks WHERE build_id = ?",
                        (prev_build_id,),
                    ).fetchall():
                        prev_chunks_map[str(r["chunk_id"])] = _sha256_text(
                            f"{r['content_hash']}:{r['embedding_text']}:{r['metadata_json']}"
                        )

                cur_sources_map = {
                    s["source_id"]: _sha256_text(
                        f"{s['sha256']}:{s['revision']}:{s['path']}:{s['title']}:{_canonical_json(s['metadata'])}"
                    )
                    for s in sources
                }
                cur_chunks_map = {
                    c["chunk_id"]: _sha256_text(
                        f"{_sha256_text(c['content'])}:{c['embedding_text']}:{_canonical_json(c['metadata'])}"
                    )
                    for c in chunks
                }

                added_sources = sum(
                    1 for sid in cur_sources_map if sid not in prev_sources_map
                )
                updated_sources = sum(
                    1
                    for sid, h in cur_sources_map.items()
                    if sid in prev_sources_map and prev_sources_map[sid] != h
                )
                deleted_sources = sum(
                    1 for sid in prev_sources_map if sid not in cur_sources_map
                )

                added_chunks = sum(
                    1 for cid in cur_chunks_map if cid not in prev_chunks_map
                )
                updated_chunks = sum(
                    1
                    for cid, h in cur_chunks_map.items()
                    if cid in prev_chunks_map and prev_chunks_map[cid] != h
                )
                deleted_chunks = sum(
                    1 for cid in prev_chunks_map if cid not in cur_chunks_map
                )
                unchanged_chunks = sum(
                    1
                    for cid, h in cur_chunks_map.items()
                    if cid in prev_chunks_map and prev_chunks_map[cid] == h
                )

                created_at = _utc_now()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO builds(
                            build_id, status, created_at, corpus_version,
                            corpus_fingerprint, lexical_ready, dense_ready,
                            dense_stale, release_ready, embedding_model,
                            embedding_revision, embedding_dimension,
                            embedding_fingerprint, embedding_profile_json,
                            stats_json, report_json, parent_build_id
                        )
                        VALUES (?, 'staging', ?, ?, ?, 0, 0, 0, 0, ?, ?, ?, ?, ?, '{}', ?, ?)
                        """,
                        (
                            new_build_id,
                            created_at,
                            CORPUS_VERSION,
                            corpus_fp,
                            emb_model,
                            emb_rev,
                            emb_dim,
                            emb_fp,
                            _canonical_json(emb_profile)
                            if emb_profile is not None
                            else None,
                            _canonical_json(report),
                            prev_build_id,
                        ),
                    )

                cached_embeddings = 0
                computed_embeddings = 0
                chunk_vector_rows: List[Tuple[str, str, str, str, str, int, bytes]] = []

                if (
                    embedder is not None
                    and emb_profile is not None
                    and emb_fp is not None
                    and emb_dim is not None
                ):
                    cache_keys_by_cid: Dict[str, str] = {}
                    input_sha_by_cid: Dict[str, str] = {}
                    for ch in chunks:
                        cid = ch["chunk_id"]
                        emb_in = str(ch["embedding_text"])
                        ckey = embedding_cache_key(
                            emb_in, emb_profile, is_query=False
                        )
                        cache_keys_by_cid[cid] = ckey
                        input_sha_by_cid[cid] = _sha256_text(emb_in)

                    existing_blobs: Dict[str, bytes] = {}
                    unique_ckeys = sorted(set(cache_keys_by_cid.values()))
                    batch_step = 400
                    for start in range(0, len(unique_ckeys), batch_step):
                        sub_keys = unique_ckeys[start : start + batch_step]
                        placeholders = ",".join("?" for _ in sub_keys)
                        rows = conn.execute(
                            f"""
                            SELECT cache_key, dimension, vector
                            FROM embedding_cache
                            WHERE cache_key IN ({placeholders})
                              AND embedding_fingerprint = ?
                            """,
                            (*sub_keys, emb_fp),
                        ).fetchall()
                        for r in rows:
                            blob = bytes(r["vector"])
                            unpack_vector(blob, expected_dim=emb_dim)
                            existing_blobs[str(r["cache_key"])] = blob

                    missing_chunks: List[Dict[str, Any]] = []
                    for ch in chunks:
                        ckey = cache_keys_by_cid[ch["chunk_id"]]
                        if ckey in existing_blobs:
                            cached_embeddings += 1
                        else:
                            missing_chunks.append(ch)

                    newly_encoded_blobs: Dict[str, bytes] = {}
                    if missing_chunks:
                        slice_size = (
                            64 if len(missing_chunks) > 64 else len(missing_chunks)
                        )
                        for slice_start in range(0, len(missing_chunks), slice_size):
                            slice_chunks = missing_chunks[
                                slice_start : slice_start + slice_size
                            ]
                            texts_to_encode = [
                                str(ch["embedding_text"]) for ch in slice_chunks
                            ]
                            vectors = embedder.encode_documents(texts_to_encode)
                            if len(vectors) != len(slice_chunks):
                                raise UDLException(
                                    "embedding_count_mismatch",
                                    f"Embedder returned {len(vectors)} vectors for {len(slice_chunks)} chunks.",
                                    retryable=False,
                                )
                            slice_cache_rows: List[
                                Tuple[str, str, str, int, bytes, str]
                            ] = []
                            now_ts = _utc_now()
                            for ch, vec in zip(slice_chunks, vectors):
                                blob = pack_vector(vec, expected_dim=emb_dim)
                                cid = ch["chunk_id"]
                                ckey = cache_keys_by_cid[cid]
                                newly_encoded_blobs[ckey] = blob
                                computed_embeddings += 1
                                slice_cache_rows.append(
                                    (
                                        ckey,
                                        emb_fp,
                                        input_sha_by_cid[cid],
                                        emb_dim,
                                        blob,
                                        now_ts,
                                    )
                                )
                            if len(missing_chunks) > 64 and slice_cache_rows:
                                with conn:
                                    conn.executemany(
                                        """
                                        INSERT OR REPLACE INTO embedding_cache(
                                            cache_key, embedding_fingerprint, input_sha256,
                                            dimension, vector, created_at
                                        )
                                        VALUES (?, ?, ?, ?, ?, ?)
                                        """,
                                        slice_cache_rows,
                                    )
                                    conn.execute(
                                        "UPDATE writer_locks SET expires_at = ? WHERE lock_name = 'build' AND owner_id = ?",
                                        (
                                            time.time() + self.lock_ttl_seconds,
                                            lock_owner,
                                        ),
                                    )

                    for ch in chunks:
                        cid = ch["chunk_id"]
                        ckey = cache_keys_by_cid[cid]
                        blob = existing_blobs.get(ckey) or newly_encoded_blobs[ckey]
                        chunk_vector_rows.append(
                            (
                                new_build_id,
                                cid,
                                ckey,
                                emb_fp,
                                input_sha_by_cid[cid],
                                emb_dim,
                                blob,
                            )
                        )

                doc_lengths: List[float] = []
                chunk_stat_rows: List[Tuple[str, str, float, int, int, str]] = []
                posting_rows: List[Tuple[str, str, str, int, int]] = []
                df_counter: Counter[str] = Counter()

                for ch in chunks:
                    cid = ch["chunk_id"]
                    c_tokens = tokenize(ch["content"])
                    m_tokens = extract_metadata_tokens(ch)
                    c_counts = Counter(c_tokens)
                    m_counts = Counter(m_tokens)
                    all_terms = sorted(set(c_counts) | set(m_counts))
                    dl = float(len(c_tokens)) + 0.5 * float(len(m_tokens))
                    doc_lengths.append(dl)
                    anchors = extract_chunk_anchors(ch)
                    chunk_stat_rows.append(
                        (
                            new_build_id,
                            cid,
                            dl,
                            len(c_tokens),
                            len(m_tokens),
                            _canonical_json(anchors),
                        )
                    )
                    for term in all_terms:
                        df_counter[term] += 1
                        posting_rows.append(
                            (
                                new_build_id,
                                term,
                                cid,
                                int(c_counts.get(term, 0)),
                                int(m_counts.get(term, 0)),
                            )
                        )

                avg_doc_length = (
                    sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
                )

                lexical_ready = len(chunks) > 0
                dense_ready = (
                    embedder is not None
                    and len(chunk_vector_rows) == len(chunks)
                    and len(chunks) > 0
                )
                dense_stale = False

                eval_result = self._evaluate_release_gates(
                    sources,
                    chunks,
                    report,
                    lexical_ready=lexical_ready,
                    dense_ready=dense_ready,
                    dense_stale=dense_stale,
                )
                release_ready = bool(eval_result["gates"]["release_ready"])

                build_stats = {
                    "noop": False,
                    "doc_count": len(chunks),
                    "source_count": len(sources),
                    "avg_doc_length": avg_doc_length,
                    "vocabulary_size": len(df_counter),
                    "added_sources": added_sources,
                    "updated_sources": updated_sources,
                    "deleted_sources": deleted_sources,
                    "added_chunks": added_chunks,
                    "updated_chunks": updated_chunks,
                    "deleted_chunks": deleted_chunks,
                    "unchanged_chunks": unchanged_chunks,
                    "cached_embeddings": cached_embeddings,
                    "computed_embeddings": computed_embeddings,
                    "counts": eval_result["counts"],
                    "gates": eval_result["gates"],
                }

                with conn:
                    conn.executemany(
                        """
                        INSERT INTO sources(
                            build_id, source_id, ordinal, source_type,
                            path, sha256, revision, title, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                new_build_id,
                                s["source_id"],
                                idx,
                                s["source_type"],
                                s["path"],
                                s["sha256"],
                                s["revision"],
                                s["title"],
                                _canonical_json(s["metadata"]),
                            )
                            for idx, s in enumerate(sources)
                        ],
                    )

                    conn.executemany(
                        """
                        INSERT INTO chunks(
                            build_id, chunk_id, ordinal, source_id, source_type,
                            content_type, content, raw_content, embedding_text,
                            content_hash, provenance_key, review_status, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                new_build_id,
                                c["chunk_id"],
                                idx,
                                c["source_id"],
                                c["source_type"],
                                c["content_type"],
                                c["content"],
                                c["raw_content"],
                                str(c["embedding_text"]),
                                _sha256_text(c["content"]),
                                _compute_provenance_key(c),
                                c["metadata"].get("review_status"),
                                _canonical_json(c["metadata"]),
                            )
                            for idx, c in enumerate(chunks)
                        ],
                    )

                    conn.executemany(
                        """
                        INSERT INTO chunk_stats(
                            build_id, chunk_id, doc_length,
                            content_token_count, meta_token_count, anchors_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        chunk_stat_rows,
                    )

                    conn.executemany(
                        """
                        INSERT INTO bm25_postings(
                            build_id, term, chunk_id, tf_content, tf_meta
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        posting_rows,
                    )

                    conn.executemany(
                        """
                        INSERT INTO bm25_terms(build_id, term, df)
                        VALUES (?, ?, ?)
                        """,
                        [
                            (new_build_id, term, int(df))
                            for term, df in sorted(df_counter.items())
                        ],
                    )

                    if chunk_vector_rows:
                        now_ts = _utc_now()
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO embedding_cache(
                                cache_key, embedding_fingerprint, input_sha256,
                                dimension, vector, created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            [
                                (ckey, efp, in_sha, dim, blob, now_ts)
                                for (_bid, _cid, ckey, efp, in_sha, dim, blob) in chunk_vector_rows
                            ],
                        )
                        conn.executemany(
                            """
                            INSERT INTO chunk_embeddings(
                                build_id, chunk_id, cache_key,
                                embedding_fingerprint, input_sha256,
                                dimension, vector, stale
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                            """,
                            chunk_vector_rows,
                        )

                    completed_at = _utc_now()
                    conn.execute(
                        """
                        UPDATE builds
                        SET status = 'ready',
                            completed_at = ?,
                            lexical_ready = ?,
                            dense_ready = ?,
                            dense_stale = 0,
                            release_ready = ?,
                            stats_json = ?
                        WHERE build_id = ?
                        """,
                        (
                            completed_at,
                            1 if lexical_ready else 0,
                            1 if dense_ready else 0,
                            1 if release_ready else 0,
                            _canonical_json(build_stats),
                            new_build_id,
                        ),
                    )
                    if prev_build_id is not None and prev_build_id != new_build_id:
                        conn.execute(
                            """
                            UPDATE builds
                            SET status = 'superseded'
                            WHERE build_id = ? AND status = 'ready'
                            """,
                            (prev_build_id,),
                        )
                    conn.execute(
                        """
                        UPDATE store_state
                        SET active_build_id = ?, updated_at = ?
                        WHERE id = 1
                        """,
                        (new_build_id, completed_at),
                    )
                    conn.execute(
                        """
                        UPDATE equation_reviews
                        SET build_id = ?
                        WHERE chunk_id IN (
                            SELECT chunk_id FROM chunks WHERE build_id = ?
                        )
                        """,
                        (new_build_id, new_build_id),
                    )

                self._search_index_cache.pop(new_build_id, None)
                self._dense_vector_cache.pop(new_build_id, None)
                return self.status()

            except Exception:
                try:
                    with conn:
                        conn.execute(
                            "DELETE FROM builds WHERE build_id = ? AND status = 'staging'",
                            (new_build_id,),
                        )
                except Exception:
                    pass
                raise
            finally:
                conn.close()
        finally:
            self._release_writer_lock(lock_owner)

    def _row_to_source_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "source_id": str(row["source_id"]),
            "source_type": str(row["source_type"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "revision": str(row["revision"]),
            "title": str(row["title"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def _row_to_chunk_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        meta = json.loads(row["metadata_json"] or "{}")
        meta.pop("_crop_from_file", None)
        if row["review_status"] is not None:
            meta["review_status"] = str(row["review_status"])
        return {
            "chunk_id": str(row["chunk_id"]),
            "source_id": str(row["source_id"]),
            "source_type": str(row["source_type"]),
            "content_type": str(row["content_type"]),
            "content": str(row["content"]),
            "raw_content": (
                str(row["raw_content"])
                if row["raw_content"] is not None
                else str(row["content"])
            ),
            "metadata": meta,
            "embedding_text": str(row["embedding_text"]),
        }

    def list_sources(self, build_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all source records in the specified (or active) build."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                return []
            target_bid = str(b_row["build_id"])
            rows = conn.execute(
                """
                SELECT * FROM sources
                WHERE build_id = ?
                ORDER BY ordinal ASC, source_id ASC
                """,
                (target_bid,),
            ).fetchall()
            return [self._row_to_source_dict(r) for r in rows]
        finally:
            conn.close()

    def get_source(
        self, source_id: str, build_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single source record by `source_id`."""
        if not source_id:
            return None
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                return None
            target_bid = str(b_row["build_id"])
            row = conn.execute(
                """
                SELECT * FROM sources
                WHERE build_id = ? AND source_id = ?
                """,
                (target_bid, str(source_id)),
            ).fetchone()
            return self._row_to_source_dict(row) if row is not None else None
        finally:
            conn.close()

    def list_chunks(self, build_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all chunk records in the specified (or active) build."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                return []
            target_bid = str(b_row["build_id"])
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE build_id = ?
                ORDER BY ordinal ASC, chunk_id ASC
                """,
                (target_bid,),
            ).fetchall()
            return [self._row_to_chunk_dict(r) for r in rows]
        finally:
            conn.close()

    def get_chunk(
        self, chunk_id: str, build_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single chunk record by `chunk_id`."""
        if not chunk_id:
            return None
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                return None
            target_bid = str(b_row["build_id"])
            row = conn.execute(
                """
                SELECT * FROM chunks
                WHERE build_id = ? AND chunk_id = ?
                """,
                (target_bid, str(chunk_id)),
            ).fetchone()
            return self._row_to_chunk_dict(row) if row is not None else None
        finally:
            conn.close()

    def _synchronize_review_hashes(
        self, conn: sqlite3.Connection, target_bid: str
    ) -> None:
        """Re-check stored source/crop/transcription hashes for equations and invalidate if changed."""
        src_rows = conn.execute(
            "SELECT * FROM sources WHERE build_id = ?", (target_bid,)
        ).fetchall()
        sources_by_id = {
            str(r["source_id"]): self._row_to_source_dict(r) for r in src_rows
        }
        chunk_rows = conn.execute(
            "SELECT * FROM chunks WHERE build_id = ? ORDER BY ordinal ASC",
            (target_bid,),
        ).fetchall()

        reviews_by_cid = {
            str(r["chunk_id"]): dict(r)
            for r in conn.execute("SELECT * FROM equation_reviews").fetchall()
        }

        invalidated_any = False
        for crow in chunk_rows:
            raw_meta = json.loads(crow["metadata_json"] or "{}")
            chunk_dict = {
                "chunk_id": str(crow["chunk_id"]),
                "source_id": str(crow["source_id"]),
                "source_type": str(crow["source_type"]),
                "content_type": str(crow["content_type"]),
                "content": str(crow["content"]),
                "raw_content": crow["raw_content"],
                "metadata": raw_meta,
                "embedding_text": str(crow["embedding_text"]),
            }
            if not _is_equation_chunk(chunk_dict):
                continue

            cid = chunk_dict["chunk_id"]
            cur_status = str(
                crow["review_status"] or raw_meta.get("review_status") or "pending"
            ).lower()
            if cur_status != "verified":
                continue

            src_hash = self._compute_source_sha256(chunk_dict, sources_by_id)
            crop_hash = self._compute_crop_sha256(chunk_dict)
            trans_hash = _sha256_text(chunk_dict["content"].strip())

            rev = reviews_by_cid.get(cid)
            # Each build retains its own proof hashes. A later review of the
            # same equation must not rewrite historical source validation.
            expected_src = raw_meta.get("source_sha256") or (rev["source_sha256"] if rev else None)
            expected_crop = raw_meta.get("crop_sha256") or (rev["crop_sha256"] if rev else None)
            expected_trans = raw_meta.get("transcription_sha256") or (rev["transcription_sha256"] if rev else None)
            reviewer = str(raw_meta.get("reviewer") or (rev["reviewer"] if rev else "")).strip()

            valid = (
                bool(reviewer)
                and bool(src_hash)
                and bool(crop_hash)
                and src_hash == expected_src
                and crop_hash == expected_crop
                and trans_hash == expected_trans
            )
            if not valid:
                invalidated_any = True
                raw_meta["review_status"] = "pending"
                raw_meta["review_invalidated"] = True
                with conn:
                    conn.execute(
                        """
                        UPDATE chunks
                        SET review_status = 'pending', metadata_json = ?
                        WHERE build_id = ? AND chunk_id = ?
                        """,
                        (_canonical_json(raw_meta), target_bid, cid),
                    )
                    conn.execute(
                        """
                        UPDATE equation_reviews
                        SET decision = 'pending', updated_at = ?
                        WHERE chunk_id = ? AND source_sha256 = ?
                          AND crop_sha256 = ? AND transcription_sha256 = ?
                        """,
                        (_utc_now(), cid, expected_src, expected_crop, expected_trans),
                    )
                    conn.execute(
                        """
                        UPDATE builds
                        SET release_ready = 0
                        WHERE build_id = ?
                        """,
                        (target_bid,),
                    )

        if invalidated_any:
            self._search_index_cache.pop(target_bid, None)

    def list_reviews(self, build_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all equation review items for the specified (or active) build."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                return []
            target_bid = str(b_row["build_id"])
            self._synchronize_review_hashes(conn, target_bid)

            src_rows = conn.execute(
                "SELECT * FROM sources WHERE build_id = ?", (target_bid,)
            ).fetchall()
            sources_by_id = {
                str(r["source_id"]): self._row_to_source_dict(r) for r in src_rows
            }
            reviews_by_cid = {
                str(r["chunk_id"]): dict(r)
                for r in conn.execute("SELECT * FROM equation_reviews").fetchall()
            }
            chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE build_id = ? ORDER BY ordinal ASC, chunk_id ASC",
                (target_bid,),
            ).fetchall()

            results: List[Dict[str, Any]] = []
            for crow in chunk_rows:
                chunk_dict = self._row_to_chunk_dict(crow)
                if not _is_equation_chunk(chunk_dict):
                    continue
                cid = chunk_dict["chunk_id"]
                meta = chunk_dict["metadata"]
                rev = reviews_by_cid.get(cid)

                cur_src_hash = self._compute_source_sha256(chunk_dict, sources_by_id)
                cur_crop_hash = self._compute_crop_sha256(chunk_dict)
                cur_trans_hash = _sha256_text(chunk_dict["content"].strip())

                stored_src_hash = (
                    rev["source_sha256"] if rev else meta.get("source_sha256")
                )
                stored_crop_hash = (
                    rev["crop_sha256"] if rev else meta.get("crop_sha256")
                )
                stored_trans_hash = (
                    rev["transcription_sha256"]
                    if rev
                    else meta.get("transcription_sha256")
                )
                reviewer = rev["reviewer"] if rev else meta.get("reviewer")
                reviewed_at = rev["updated_at"] if rev else meta.get("reviewed_at")

                hashes_valid = (
                    bool(cur_src_hash)
                    and bool(cur_crop_hash)
                    and cur_src_hash == stored_src_hash
                    and cur_crop_hash == stored_crop_hash
                    and cur_trans_hash == stored_trans_hash
                )
                status_val = str(meta.get("review_status") or "pending").lower()
                if status_val == "verified" and not (
                    hashes_valid and reviewer and str(reviewer).strip()
                ):
                    status_val = "pending"

                results.append(
                    {
                        "chunk_id": cid,
                        "build_id": target_bid,
                        "source_id": chunk_dict["source_id"],
                        "equation_number": meta.get("equation_number"),
                        "chapter": meta.get("chapter"),
                        "section": meta.get("section"),
                        "pdf_page": meta.get("pdf_page"),
                        "book_page": meta.get("book_page"),
                        "bbox": meta.get("bbox"),
                        "asset_path": meta.get("asset_path"),
                        "transcription": chunk_dict["content"],
                        "content": chunk_dict["content"],
                        "raw_content": chunk_dict["raw_content"],
                        "review_status": status_val,
                        "decision": status_val,
                        "reviewer": reviewer,
                        "reviewed_at": reviewed_at,
                        "source_sha256": stored_src_hash or cur_src_hash,
                        "crop_sha256": stored_crop_hash or cur_crop_hash,
                        "transcription_sha256": stored_trans_hash or cur_trans_hash,
                        "hashes_valid": hashes_valid,
                        "locator": locator(chunk_dict),
                        "metadata": meta,
                    }
                )
            return results
        finally:
            conn.close()

    def review_equation(
        self,
        chunk_id: str,
        transcription: str,
        reviewer: str,
        decision: str,
        build_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist an equation review decision and update the active SQLite overlay."""
        clean_decision = str(decision or "").strip().lower()
        if clean_decision not in {"verified", "pending"}:
            raise UDLException(
                "review_invalid",
                f"Invalid review decision '{decision}'. Must be 'verified' or 'pending'.",
                retryable=False,
            )
        if transcription is None or not str(transcription).strip():
            raise UDLException(
                "review_invalid",
                "Equation transcription cannot be empty.",
                retryable=False,
            )
        clean_transcription = str(transcription).replace("\x00", "").strip()
        clean_reviewer = str(reviewer or "").strip()
        if clean_decision == "verified" and not clean_reviewer:
            raise UDLException(
                "invalid_reviewer",
                "Marking an equation 'verified' requires a non-empty reviewer.",
                retryable=False,
            )

        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                raise UDLException(
                    "index_empty",
                    "Cannot review equation: no active build in SQLiteStore.",
                    retryable=False,
                )
            target_bid = str(b_row["build_id"])
            crow = conn.execute(
                "SELECT * FROM chunks WHERE build_id = ? AND chunk_id = ?",
                (target_bid, str(chunk_id)),
            ).fetchone()
            if crow is None:
                raise UDLException(
                    "chunk_not_found",
                    f"Chunk '{chunk_id}' was not found in build '{target_bid}'.",
                    retryable=False,
                )

            raw_meta = json.loads(crow["metadata_json"] or "{}")
            chunk_dict = {
                "chunk_id": str(crow["chunk_id"]),
                "source_id": str(crow["source_id"]),
                "source_type": str(crow["source_type"]),
                "content_type": str(crow["content_type"]),
                "content": str(crow["content"]),
                "raw_content": crow["raw_content"],
                "metadata": raw_meta,
                "embedding_text": str(crow["embedding_text"]),
            }
            if not _is_equation_chunk(chunk_dict):
                raise UDLException(
                    "not_an_equation",
                    f"Chunk '{chunk_id}' is not an equation chunk.",
                    retryable=False,
                )

            src_rows = conn.execute(
                "SELECT * FROM sources WHERE build_id = ?", (target_bid,)
            ).fetchall()
            sources_by_id = {
                str(r["source_id"]): self._row_to_source_dict(r) for r in src_rows
            }

            src_hash = self._compute_source_sha256(chunk_dict, sources_by_id)
            crop_hash = self._compute_crop_sha256(chunk_dict)
            trans_hash = _sha256_text(clean_transcription)

            if clean_decision == "verified":
                if not src_hash:
                    raise UDLException(
                        "missing_equation_source_hash",
                        f"Cannot mark equation '{chunk_id}' verified without a source SHA-256 hash.",
                        retryable=False,
                    )
                if not crop_hash:
                    raise UDLException(
                        "missing_equation_crop",
                        f"Cannot mark equation '{chunk_id}' verified without a valid crop asset and crop SHA-256 hash.",
                        retryable=False,
                    )

            existing_rev = conn.execute(
                "SELECT original_transcription_sha256 FROM equation_reviews WHERE chunk_id = ?",
                (str(chunk_id),),
            ).fetchone()
            orig_trans_hash = (
                str(existing_rev["original_transcription_sha256"])
                if existing_rev is not None
                else _sha256_text(str(crow["raw_content"] or crow["content"]).strip())
            )

            old_emb_text = str(crow["embedding_text"])
            old_content = str(crow["content"])

            now_ts = _utc_now()
            raw_meta["transcription"] = clean_transcription
            raw_meta["review_status"] = clean_decision
            raw_meta["reviewer"] = clean_reviewer
            raw_meta["reviewed_at"] = now_ts
            raw_meta["source_sha256"] = src_hash or ""
            raw_meta["crop_sha256"] = crop_hash or ""
            raw_meta["transcription_sha256"] = trans_hash
            raw_meta.pop("review_invalidated", None)

            updated_chunk = {
                **chunk_dict,
                "content": clean_transcription,
                "metadata": raw_meta,
            }
            if clean_transcription != old_content:
                new_emb_text = embedding_text(updated_chunk, force_recompute=True)
            else:
                new_emb_text = old_emb_text
            updated_chunk["embedding_text"] = new_emb_text

            emb_input_changed = _sha256_text(new_emb_text) != _sha256_text(old_emb_text)

            c_tokens = tokenize(clean_transcription)
            m_tokens = extract_metadata_tokens(updated_chunk)
            c_counts = Counter(c_tokens)
            m_counts = Counter(m_tokens)
            all_terms = sorted(set(c_counts) | set(m_counts))
            dl = float(len(c_tokens)) + 0.5 * float(len(m_tokens))
            anchors = extract_chunk_anchors(updated_chunk)

            old_terms = [
                str(r["term"])
                for r in conn.execute(
                    "SELECT term FROM bm25_postings WHERE build_id = ? AND chunk_id = ?",
                    (target_bid, str(chunk_id)),
                ).fetchall()
            ]
            affected_terms = sorted(set(old_terms) | set(all_terms))

            with conn:
                conn.execute(
                    """
                    UPDATE chunks
                    SET content = ?,
                        embedding_text = ?,
                        content_hash = ?,
                        review_status = ?,
                        metadata_json = ?
                    WHERE build_id = ? AND chunk_id = ?
                    """,
                    (
                        clean_transcription,
                        new_emb_text,
                        _sha256_text(clean_transcription),
                        clean_decision,
                        _canonical_json(raw_meta),
                        target_bid,
                        str(chunk_id),
                    ),
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunk_stats(
                        build_id, chunk_id, doc_length,
                        content_token_count, meta_token_count, anchors_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_bid,
                        str(chunk_id),
                        dl,
                        len(c_tokens),
                        len(m_tokens),
                        _canonical_json(anchors),
                    ),
                )

                conn.execute(
                    "DELETE FROM bm25_postings WHERE build_id = ? AND chunk_id = ?",
                    (target_bid, str(chunk_id)),
                )
                conn.executemany(
                    """
                    INSERT INTO bm25_postings(build_id, term, chunk_id, tf_content, tf_meta)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            target_bid,
                            term,
                            str(chunk_id),
                            int(c_counts.get(term, 0)),
                            int(m_counts.get(term, 0)),
                        )
                        for term in all_terms
                    ],
                )

                for term in affected_terms:
                    cnt_row = conn.execute(
                        """
                        SELECT COUNT(*) AS df
                        FROM bm25_postings
                        WHERE build_id = ? AND term = ?
                        """,
                        (target_bid, term),
                    ).fetchone()
                    df_val = int(cnt_row["df"]) if cnt_row else 0
                    if df_val > 0:
                        conn.execute(
                            "INSERT OR REPLACE INTO bm25_terms(build_id, term, df) VALUES (?, ?, ?)",
                            (target_bid, term, df_val),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM bm25_terms WHERE build_id = ? AND term = ?",
                            (target_bid, term),
                        )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO equation_reviews(
                        chunk_id, source_id, equation_number, transcription,
                        reviewer, decision, source_sha256, crop_sha256,
                        transcription_sha256, original_transcription_sha256,
                        build_id, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(chunk_id),
                        chunk_dict["source_id"],
                        str(raw_meta.get("equation_number") or ""),
                        clean_transcription,
                        clean_reviewer,
                        clean_decision,
                        src_hash or "",
                        crop_hash or "",
                        trans_hash,
                        orig_trans_hash,
                        target_bid,
                        now_ts,
                    ),
                )

                if emb_input_changed:
                    has_emb_row = conn.execute(
                        "SELECT 1 FROM chunk_embeddings WHERE build_id = ? LIMIT 1",
                        (target_bid,),
                    ).fetchone()
                    if has_emb_row is not None or bool(b_row["dense_ready"]):
                        conn.execute(
                            """
                            UPDATE chunk_embeddings
                            SET stale = 1
                            WHERE build_id = ? AND chunk_id = ?
                            """,
                            (target_bid, str(chunk_id)),
                        )
                        conn.execute(
                            """
                            UPDATE builds
                            SET dense_stale = 1,
                                dense_ready = 0,
                                release_ready = 0
                            WHERE build_id = ?
                            """,
                            (target_bid,),
                        )
                        self._dense_vector_cache.pop(target_bid, None)

            self._search_index_cache.pop(target_bid, None)
            self.status()

            matching = [
                r
                for r in self.list_reviews(build_id=target_bid)
                if r["chunk_id"] == str(chunk_id)
            ]
            return matching[0]
        finally:
            conn.close()

    def status(self) -> Dict[str, Any]:
        """Return detailed store readiness, counts, embedding profile, and release validation gates."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn)
            if b_row is None:
                empty_counts = {
                    "sources": 0,
                    "chunks": 0,
                    "embeddings": 0,
                    "by_content_type": {},
                    "by_modality": {},
                    "equations_total": 0,
                    "equations_verified": 0,
                    "equations_pending": 0,
                    "figures_total": 0,
                    "figures_verified": 0,
                    "figures_pending": 0,
                    "algorithms_total": 0,
                    "algorithms_verified": 0,
                    "algorithms_pending": 0,
                }
                empty_gates = {
                    "lexical_usable": False,
                    "dense_usable": False,
                    "dense_stale": False,
                    "pending_equation_reviews": 0,
                    "pending_figure_reviews": 0,
                    "pending_algorithm_reviews": 0,
                    "missing_asset_files": 0,
                    "notebook_coverage_complete": False,
                    "figure_target_met": False,
                    "algorithm_target_met": False,
                    "equation_reviews_complete": False,
                    "release_ready": False,
                    "blockers": ["index_empty"],
                }
                return {
                    "initialized": False,
                    "db_path": str(self.path),
                    "active_build_id": None,
                    "build_id": None,
                    "corpus_version": CORPUS_VERSION,
                    "corpus_fingerprint": None,
                    "lexical_ready": False,
                    "dense_ready": False,
                    "dense_stale": False,
                    "release_ready": False,
                    "counts": empty_counts,
                    "gates": empty_gates,
                    "release_validation": empty_gates,
                    "embedding": None,
                    "last_build": None,
                }

            target_bid = str(b_row["build_id"])
            self._synchronize_review_hashes(conn, target_bid)
            b_row = self._get_active_build_row(conn, build_id=target_bid)
            assert b_row is not None

            sources = [
                self._row_to_source_dict(r)
                for r in conn.execute(
                    "SELECT * FROM sources WHERE build_id = ? ORDER BY ordinal ASC",
                    (target_bid,),
                ).fetchall()
            ]
            chunks_raw = conn.execute(
                "SELECT * FROM chunks WHERE build_id = ? ORDER BY ordinal ASC",
                (target_bid,),
            ).fetchall()
            chunks: List[Dict[str, Any]] = []
            for r in chunks_raw:
                meta = json.loads(r["metadata_json"] or "{}")
                if r["review_status"] is not None:
                    meta["review_status"] = str(r["review_status"])
                chunks.append(
                    {
                        "chunk_id": str(r["chunk_id"]),
                        "source_id": str(r["source_id"]),
                        "source_type": str(r["source_type"]),
                        "content_type": str(r["content_type"]),
                        "content": str(r["content"]),
                        "raw_content": r["raw_content"],
                        "metadata": meta,
                        "embedding_text": str(r["embedding_text"]),
                    }
                )

            emb_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chunk_embeddings WHERE build_id = ? AND stale = 0",
                (target_bid,),
            ).fetchone()
            valid_emb_count = int(emb_count_row["cnt"]) if emb_count_row else 0

            stale_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chunk_embeddings WHERE build_id = ? AND stale = 1",
                (target_bid,),
            ).fetchone()
            stale_emb_count = int(stale_count_row["cnt"]) if stale_count_row else 0

            lexical_ready = bool(b_row["lexical_ready"]) and len(chunks) > 0
            dense_stale = bool(b_row["dense_stale"]) or stale_emb_count > 0
            dense_ready = (
                bool(b_row["dense_ready"])
                and not dense_stale
                and len(chunks) > 0
                and valid_emb_count == len(chunks)
            )

            report = json.loads(b_row["report_json"] or "{}")
            eval_result = self._evaluate_release_gates(
                sources,
                chunks,
                report,
                lexical_ready=lexical_ready,
                dense_ready=dense_ready,
                dense_stale=dense_stale,
            )
            counts = eval_result["counts"]
            counts["embeddings"] = valid_emb_count
            gates = eval_result["gates"]
            release_ready = bool(gates["release_ready"])

            if int(b_row["release_ready"]) != (1 if release_ready else 0):
                with conn:
                    conn.execute(
                        "UPDATE builds SET release_ready = ? WHERE build_id = ?",
                        (1 if release_ready else 0, target_bid),
                    )

            emb_profile = (
                json.loads(b_row["embedding_profile_json"])
                if b_row["embedding_profile_json"]
                else None
            )
            embedding_info = (
                {
                    "model_name": b_row["embedding_model"],
                    "revision": b_row["embedding_revision"],
                    "dimension": b_row["embedding_dimension"],
                    "fingerprint": b_row["embedding_fingerprint"],
                    "profile": emb_profile,
                    "stale": dense_stale,
                    "ready": dense_ready,
                    "indexed_vectors": valid_emb_count,
                }
                if b_row["embedding_model"] is not None
                else None
            )

            raw_stats = json.loads(b_row["stats_json"] or "{}")
            last_build = {
                "build_id": target_bid,
                "created_at": b_row["created_at"],
                "completed_at": b_row["completed_at"],
                "parent_build_id": b_row["parent_build_id"],
                "noop": bool(raw_stats.get("noop", False)),
                "added_sources": int(raw_stats.get("added_sources", 0)),
                "updated_sources": int(raw_stats.get("updated_sources", 0)),
                "deleted_sources": int(raw_stats.get("deleted_sources", 0)),
                "added_chunks": int(raw_stats.get("added_chunks", 0)),
                "updated_chunks": int(raw_stats.get("updated_chunks", 0)),
                "deleted_chunks": int(raw_stats.get("deleted_chunks", 0)),
                "unchanged_chunks": int(raw_stats.get("unchanged_chunks", 0)),
                "cached_embeddings": int(raw_stats.get("cached_embeddings", 0)),
                "computed_embeddings": int(raw_stats.get("computed_embeddings", 0)),
            }

            return {
                "initialized": True,
                "db_path": str(self.path),
                "active_build_id": target_bid,
                "build_id": target_bid,
                "corpus_version": str(b_row["corpus_version"]),
                "corpus_fingerprint": str(b_row["corpus_fingerprint"]),
                "lexical_ready": lexical_ready,
                "dense_ready": dense_ready,
                "dense_stale": dense_stale,
                "release_ready": release_ready,
                "counts": counts,
                "gates": gates,
                "release_validation": gates,
                "embedding": embedding_info,
                "last_build": last_build,
            }
        finally:
            conn.close()

    def _get_search_index(self, build_id: Optional[str] = None) -> Dict[str, Any]:
        """Load BM25 postings, document frequencies, lengths, and anchors for `build_id`."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                raise UDLException(
                    "index_empty",
                    "No active corpus build found in SQLiteStore.",
                    retryable=False,
                )
            target_bid = str(b_row["build_id"])
            cached = self._search_index_cache.get(target_bid)
            if cached is not None:
                return cached

            chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE build_id = ? ORDER BY ordinal ASC",
                (target_bid,),
            ).fetchall()
            chunks_by_id = {
                str(r["chunk_id"]): self._row_to_chunk_dict(r) for r in chunk_rows
            }

            stat_rows = conn.execute(
                "SELECT * FROM chunk_stats WHERE build_id = ?", (target_bid,)
            ).fetchall()
            doc_lengths: Dict[str, float] = {}
            anchors_by_chunk: Dict[str, Dict[str, List[str]]] = {}
            for r in stat_rows:
                cid = str(r["chunk_id"])
                doc_lengths[cid] = float(r["doc_length"])
                anchors_by_chunk[cid] = json.loads(r["anchors_json"] or "{}")

            avg_dl = (
                sum(doc_lengths.values()) / len(doc_lengths)
                if doc_lengths
                else 1.0
            )

            df_rows = conn.execute(
                "SELECT term, df FROM bm25_terms WHERE build_id = ?",
                (target_bid,),
            ).fetchall()
            df_map = {str(r["term"]): int(r["df"]) for r in df_rows}

            postings: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
            post_rows = conn.execute(
                "SELECT term, chunk_id, tf_content, tf_meta FROM bm25_postings WHERE build_id = ?",
                (target_bid,),
            ).fetchall()
            for r in post_rows:
                postings[str(r["term"])][str(r["chunk_id"])] = (
                    int(r["tf_content"]),
                    int(r["tf_meta"]),
                )

            data = {
                "build_id": target_bid,
                "build_info": {
                    "build_id": target_bid,
                    "embedding_model": b_row["embedding_model"],
                    "embedding_revision": b_row["embedding_revision"],
                    "embedding_dimension": b_row["embedding_dimension"],
                    "embedding_fingerprint": b_row["embedding_fingerprint"],
                    "dense_ready": bool(b_row["dense_ready"]),
                    "dense_stale": bool(b_row["dense_stale"]),
                },
                "doc_count": len(chunks_by_id),
                "avg_doc_length": avg_dl,
                "chunks_by_id": chunks_by_id,
                "doc_lengths": doc_lengths,
                "anchors_by_chunk": anchors_by_chunk,
                "df": df_map,
                "postings": dict(postings),
            }
            self._search_index_cache[target_bid] = data
            return data
        finally:
            conn.close()

    def _get_dense_vectors(
        self, build_id: Optional[str] = None
    ) -> Dict[str, Tuple[float, ...]]:
        """Unpack and return float32 vector BLOBs for all chunks in `build_id`."""
        conn = self._connect()
        try:
            b_row = self._get_active_build_row(conn, build_id=build_id)
            if b_row is None:
                raise UDLException(
                    "index_empty",
                    "No active corpus build found in SQLiteStore.",
                    retryable=False,
                )
            target_bid = str(b_row["build_id"])
            if bool(b_row["dense_stale"]):
                raise UDLException(
                    "dense_stale",
                    f"Dense embeddings for build '{target_bid}' are marked stale.",
                    retryable=False,
                )
            if not bool(b_row["dense_ready"]):
                raise UDLException(
                    "dense_missing",
                    f"Dense embeddings for build '{target_bid}' are missing.",
                    retryable=False,
                )

            cached = self._dense_vector_cache.get(target_bid)
            if cached is not None:
                return cached

            expected_dim = (
                int(b_row["embedding_dimension"])
                if b_row["embedding_dimension"] is not None
                else None
            )
            rows = conn.execute(
                """
                SELECT chunk_id, dimension, vector, stale
                FROM chunk_embeddings
                WHERE build_id = ?
                """,
                (target_bid,),
            ).fetchall()

            vectors: Dict[str, Tuple[float, ...]] = {}
            for r in rows:
                if int(r["stale"]) != 0:
                    raise UDLException(
                        "dense_stale",
                        f"Dense embedding for chunk '{r['chunk_id']}' is stale.",
                        retryable=False,
                    )
                dim = expected_dim or int(r["dimension"])
                vectors[str(r["chunk_id"])] = unpack_vector(
                    bytes(r["vector"]), expected_dim=dim
                )

            self._dense_vector_cache[target_bid] = vectors
            return vectors
        finally:
            conn.close()
