"""Preserve the exact source bytes and crops referenced by historical builds."""

import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from src.contracts import CorpusResult, UDLException


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_content_addressed(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != expected:
            raise UDLException("snapshot_corrupt", "A saved source snapshot has changed. Restore it from backup before rebuilding.")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".snapshot-", delete=False) as out:
            temporary = Path(out.name)
            with source.open("rb") as inp:
                shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if file_sha256(temporary) != expected:
            raise UDLException("source_changed", "A source changed during the build. Retry after the source update finishes.", True)
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def snapshot_corpus(corpus: CorpusResult, project_root):
    """Return transport records pointing at immutable content-addressed copies.

    Originals remain available for updates. Review and citation provenance refer
    to the bytes copied here, so a later update cannot change an earlier citation.
    """
    root = Path(project_root).resolve()
    result = copy.deepcopy(corpus)

    def resolve(path, allowed):
        path = Path(path)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or not any(path.is_relative_to(root / prefix) for prefix in allowed):
            raise UDLException("source_path_invalid", "An extracted source or crop is missing or outside the approved data directories.")
        return path

    for source in result.sources:
        original = resolve(source["path"], ["data/raw", "data/snapshots"])
        actual = file_sha256(original)
        if source.get("sha256") != actual:
            raise UDLException("source_changed", "A source file changed after extraction. Re-run the build.", True)
        saved = root / "data/snapshots" / actual / original.name
        _copy_content_addressed(original, saved, actual)
        source.setdefault("metadata", {})["original_path"] = source["path"]
        source["path"] = saved.relative_to(root).as_posix()
    for chunk in result.chunks:
        meta = chunk.get("metadata") or {}
        if not meta.get("asset_path"):
            continue
        original = resolve(meta["asset_path"], ["data/assets"])
        actual = file_sha256(original)
        if meta.get("crop_sha256") and meta["crop_sha256"] != actual:
            raise UDLException("crop_changed", "An equation or figure crop changed after extraction. Re-run the build.", True)
        saved = root / "data/assets/snapshots" / (actual + ".png")
        _copy_content_addressed(original, saved, actual)
        meta["original_asset_path"] = meta["asset_path"]
        meta["asset_path"] = saved.relative_to(root).as_posix()
        meta["crop_sha256"] = actual
        chunk["metadata"] = meta
    return result
