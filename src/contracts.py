"""Small shared contracts for the handwritten UDL pipeline.

Chunks and sources remain plain JSON-compatible dictionaries at subsystem
boundaries. SQLite is authoritative; these objects are transport records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "udl_rag.sqlite3"
BOOK_SHA256 = "9c0ea2e831e662a687d267dd9bb2d50ea743e58271e90b96a4fac4d585288e04"
NOTEBOOK_COMMIT = "44a3e5f6786d57a3362d1480c9f6ca09f495aaa3"
BOOK_URL = "https://github.com/udlbook/udlbook/releases/download/v5.0.2/UnderstandingDeepLearning_05_29_25_C.pdf"
NOTEBOOK_REPO = "https://github.com/udlbook/udlbook.git"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CORPUS_VERSION = "udl-v5.0.2-1"


class UDLException(Exception):
    """Actionable error safe to expose without a traceback or credentials."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def as_dict(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


@dataclass
class CorpusResult:
    # source: source_id, source_type, path (relative to project root), sha256,
    # revision, title, metadata. source_id = 'book' or 'Notebooks/ChapXX/...'.
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # chunk: chunk_id, source_id, source_type, content_type, content,
    # raw_content, metadata, embedding_text (optional, otherwise content).
    # metadata: chapter, section, pdf_page (ONE-based), book_page, bbox,
    # cell_index (ZERO-based), start_line/end_line (ONE-based), symbol_names,
    # figure_number/equation_number/algorithm_number, asset_path (relative),
    # review_status ('pending'/'verified'), is_template, previous_id/next_id.
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRequest:
    query: str
    top_k: int = 6
    mode: str = "hybrid"  # lexical / semantic / hybrid
    modality: str = "all"  # all / text / code / figure / exercise / equation
    chapter: Optional[str] = None


@dataclass
class SearchResult:
    # hits contain chunk fields plus citation, locator, bm25_score,
    # semantic_score (nullable), retrieval_score, and build_id.
    hits: List[Dict[str, Any]] = field(default_factory=list)
    build_id: Optional[str] = None
    mode: str = "lexical"
    warnings: List[str] = field(default_factory=list)
    abstained: bool = False
    reason: Optional[str] = None
    # Each required facet must retain at least one of these source chunks in
    # the final answer context (OR within a group, AND between groups).
    required_evidence_groups: List[List[str]] = field(default_factory=list)

    def as_dict(self):
        return dict(hits=self.hits, build_id=self.build_id, mode=self.mode,
                    warnings=self.warnings, abstained=self.abstained, reason=self.reason,
                    required_evidence_groups=self.required_evidence_groups)
