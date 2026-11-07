"""Source-hashed, independently reviewed equation transcriptions for a release.

Draft/OCR candidates never enter this bundle automatically. A verified record
must retain its reviewer and source/crop/transcription hashes. Changed geometry
invalidates a review while leaving ordinary search available.
"""

import copy
import hashlib
import json
from pathlib import Path

from src.contracts import BOOK_SHA256, UDLException
from src.snapshots import file_sha256


def text_sha256(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def apply_review_bundle(corpus, project_root, bundle_path=None):
    root = Path(project_root).resolve()
    path = Path(bundle_path) if bundle_path else root / "resources/book_math_reviews.json"
    if not path.is_file():
        return corpus
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise UDLException("review_bundle_invalid", "The mathematical review bundle is unreadable.") from exc
    if payload.get("source_sha256") != BOOK_SHA256 or not isinstance(payload.get("reviews"), list):
        raise UDLException("review_bundle_invalid", "The review bundle must identify the pinned book and a list of reviewed equations.")
    result = copy.deepcopy(corpus)
    book = next((s for s in result.sources if s["source_id"] == "book"), None)
    if not book or book.get("sha256") != payload["source_sha256"]:
        raise UDLException("review_bundle_invalid", "The reviewed book does not match this corpus.")
    chunks = {c["chunk_id"]: c for c in result.chunks if c["content_type"] == "equation"}
    seen, applied, invalidated = set(), [], []
    for review in payload["reviews"]:
        cid = review.get("chunk_id")
        if not cid or cid in seen:
            raise UDLException("review_bundle_invalid", "The review bundle contains an empty or repeated equation identity.")
        seen.add(cid)
        if review.get("decision") != "verified":
            continue
        transcription = review.get("transcription", "")
        if (not isinstance(transcription, str) or not transcription.strip()
                or not str(review.get("reviewer", "")).strip()
                or not str(review.get("method", "")).strip()
                or review.get("source_sha256") != BOOK_SHA256
                or review.get("transcription_sha256") != text_sha256(transcription)):
            raise UDLException("review_bundle_invalid", "A verified review lacks its exact transcription, hash, reviewer, or method.")
        chunk = chunks.get(cid)
        if chunk is None:
            invalidated.append(cid)
            continue
        meta = chunk["metadata"]
        asset = (root / meta.get("asset_path", "")).resolve()
        valid_asset = asset.is_relative_to(root / "data/assets") and asset.is_file()
        if not valid_asset or file_sha256(asset) != review.get("crop_sha256"):
            meta["review_status"] = "pending"
            meta["review_invalidated"] = True
            invalidated.append(cid)
            continue
        chunk["content"] = transcription.strip()
        chunk.pop("embedding_text", None)
        meta.update(review_status="verified", reviewer=review["reviewer"],
                    source_sha256=BOOK_SHA256, crop_sha256=review["crop_sha256"],
                    transcription_sha256=review["transcription_sha256"],
                    transcription=chunk["content"], review_method=review["method"])
        applied.append(cid)
    result.report["review_bundle"] = {
        "version": payload.get("version"), "sha256": file_sha256(path),
        "applied": len(applied), "invalidated": invalidated,
    }
    return result


def assemble_review_bundle(record_paths, output_path, source_sha256=BOOK_SHA256):
    """Assemble reviewed records without promoting pending/draft entries."""
    reviews = {}
    for path in record_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("reviews", [])
        for row in rows:
            if row.get("decision") != "verified":
                continue
            if row.get("source_sha256") != source_sha256 or row.get("transcription_sha256") != text_sha256(row.get("transcription", "")):
                raise UDLException("review_bundle_invalid", "A batch review has mismatched source/transcription hashes.")
            cid = row["chunk_id"]
            if cid in reviews and reviews[cid] != row:
                raise UDLException("review_bundle_invalid", "Two batches contain conflicting reviews for an equation.")
            reviews[cid] = row
    result = {"version": "udl-v5.0.2-math-1", "source_sha256": source_sha256,
              "reviews": sorted(reviews.values(), key=lambda r: r["chunk_id"])}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"verified_records": len(reviews), "path": str(path)}
