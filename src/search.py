"""Handwritten BM25, exact cosine, RRF fusion, abstention, and chunk helpers.

Implements:
- `tokenize`: identifier-aware Unicode tokenization (snake_case and CamelCase).
- `locator`: human-readable citation locators for book and notebook chunks.
- `embedding_text`: canonical embedding text representation for chunks.
- `rrf_fuse`: deterministic Reciprocal Rank Fusion (default k=60).
- `HybridRetriever`: lexical (BM25 + exact anchors), semantic (exact cosine),
  and hybrid (BM25 + cosine + RRF) search over `SQLiteStore`, with scope filters,
  missing-dense fallback warning for hybrid, actionable error for semantic-only,
  and evidence-sensitive abstention using original scores rather than RRF.
- Source neighbor expansion helpers (`get_chunk_neighbors`, `expand_neighbors`).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import math
import re
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.contracts import SearchRequest, SearchResult, UDLException
from src.dense import cosine_similarity

VALID_MODES = {"lexical", "semantic", "hybrid"}
VALID_MODALITIES = {"all", "text", "code", "figure", "exercise", "equation"}

STOPWORDS: Set[str] = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "between",
    "both",
    "by",
    "can",
    "describe",
    "do",
    "does",
    "each",
    "every",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "more",
    "most",
    "not",
    "of",
    "on",
    "one",
    "or",
    "other",
    "our",
    "out",
    "over",
    "same",
    "show",
    "so",
    "some",
    "such",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "udl",
    "under",
    "up",
    "use",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "within",
    "without",
}

QUERY_FRAMING_WORDS: Set[str] = {
    "algorithm",
    "associated",
    "assignment",
    "book",
    "cell",
    "chap",
    "chapter",
    "cite",
    "code",
    "connect",
    "contents",
    "corresponding",
    "covered",
    "define",
    "defined",
    "defines",
    "definition",
    "definitions",
    "diagram",
    "diagrams",
    "drawn",
    "equation",
    "exact",
    "example",
    "exercise",
    "expression",
    "expressions",
    "explained",
    "figure",
    "file",
    "form",
    "formula",
    "formulate",
    "formulated",
    "formulation",
    "function",
    "helper",
    "identify",
    "illustrate",
    "illustrates",
    "illustration",
    "implement",
    "implementation",
    "implemented",
    "introduced",
    "locate",
    "mathematical",
    "mathematically",
    "method",
    "model",
    "models",
    "notebook",
    "notebooks",
    "page",
    "part",
    "problem",
    "python",
    "question",
    "related",
    "relates",
    "routine",
    "say",
    "says",
    "section",
    "source",
    "summarize",
    "summary",
    "text",
    "textbook",
    "topic",
    "treatment",
    "understanding",
    "utility",
}

AUXILIARY_MANNER_WORDS: Set[str] = {
    "avoid",
    "avoided",
    "avoiding",
    "enforce",
    "enforced",
    "enforcing",
    "ensure",
    "ensured",
    "ensuring",
    "guarantee",
    "guaranteed",
    "guaranteeing",
    "keep",
    "keeping",
    "kept",
    "maintain",
    "maintained",
    "maintaining",
    "preserve",
    "preserved",
    "preserving",
    "prevent",
    "prevented",
    "preventing",
    "remain",
    "remained",
    "remaining",
    "respect",
    "respected",
    "respecting",
    "retain",
    "retained",
    "retaining",
    "satisfy",
    "satisfied",
    "satisfying",
}


_CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_FIGURE_REF_RE = re.compile(
    r"\b(?:figure|fig\.?)\s*(\d+(?:\.\d+)+[a-z]?)\b", re.IGNORECASE
)
_EQUATION_REF_RE = re.compile(
    r"\b(?:equation|eq\.?|eqn\.?)\s*\(?(\d+(?:\.\d+)+[a-z]?)\)?(?!\.\d)",
    re.IGNORECASE,
)
_PROBLEM_REF_RE = re.compile(
    r"\b(?:problem|prob\.?|exercise|ex\.?|question)\s*(\d+(?:\.\d+)+[a-z]?)\b",
    re.IGNORECASE,
)
_ALGORITHM_REF_RE = re.compile(
    r"\b(?:algorithm|alg\.?)\s*(\d+(?:\.\d+)*[a-z]?)\b", re.IGNORECASE
)
_CHAPTER_REF_RE = re.compile(
    r"\b(?:chapter|chap\.?)\s*0*(\d+)\b", re.IGNORECASE
)
_NOTEBOOK_REF_RE = re.compile(
    r"\bnotebook\s*0*(\d+)\.0*(\d+)\b", re.IGNORECASE
)
_PAGE_REF_RE = re.compile(
    r"\bpage\s*(\d+)\b", re.IGNORECASE
)
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_DEF_SYMBOL_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_CALLABLE_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)\s*\(")
_CODE_WORDING_BEFORE_RE = re.compile(
    r"\b(?:function|functions|method|methods|class|classes|helper|helpers|routine|routines|callable|symbol|identifier)\s+([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_CODE_WORDING_AFTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)\s+(?:function|method|class|helper|routine|callable|implementation)\b",
    re.IGNORECASE,
)
_CODE_INTENT_QUERY_RE = re.compile(
    r"\b(?:function|functions|method|methods|class|classes|helper|helpers|routine|routines|callable|code|notebook|python|def|implement|implementation|implemented)\b",
    re.IGNORECASE,
)
_SNAKE_IDENTIFIER_RE = re.compile(
    r"\b([a-zA-Z]+(?:_[a-zA-Z0-9]+)+)(?!\s*(?:\^|\b(?:squared|cubed)\b))\b",
    re.IGNORECASE,
)

_GREEK_LATEX_NAMES: Set[str] = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "varpi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "ell",
    "nabla",
    "partial",
    "hat",
    "bar",
    "tilde",
    "vec",
    "dot",
    "star",
    "min",
    "max",
    "opt",
    "true",
    "pred",
    "prior",
    "post",
}

# Explicitly unindexed or nonexistent external sources per IMPLEMENTATION_CONTRACT.md
_UNSUPPORTED_EXTERNAL_RE = re.compile(
    r"\b(?:"
    r"answer\s+booklet|"
    r"solution\s+manual|"
    r"instructor\s+solution|"
    r"errata|"
    r"lecture\s+slides|"
    r"course\s+(?:lecture\s+)?slides|"
    r"author\s+blog|"
    r"blog\s+post|"
    r"unpublished\s+appendix|"
    r"hypothetical\s+\d{4}\s+edition|"
    r"nonexistent\s+\w+|"
    r"published\s+this\s+morning"
    r")\b",
    re.IGNORECASE,
)


def _is_math_subscript_identifier(name: str) -> bool:
    """Return True when `name` looks like a LaTeX/math subscript variable (e.g. `sigma_phi`, `theta_0`, `x_i`)."""
    parts = [p.lower() for p in name.split("_") if p]
    if not parts:
        return False
    if any(p in _GREEK_LATEX_NAMES for p in parts):
        return True
    return all(len(p) == 1 or p.isdigit() for p in parts)


def _has_code_symbol_intent(query: str, modality: str = "all") -> bool:
    """Return True only when the query or modality exhibits genuine code-symbol intent."""
    if (modality or "all").strip().lower() == "code":
        return True
    q = str(query or "")
    if _BACKTICK_SYMBOL_RE.search(q) or _DEF_SYMBOL_RE.search(q) or _CALLABLE_SYMBOL_RE.search(q):
        return True
    if _CODE_WORDING_BEFORE_RE.search(q) or _CODE_WORDING_AFTER_RE.search(q):
        return True
    if _CODE_INTENT_QUERY_RE.search(q):
        return True
    return False


def _defined_callable_names(chunk: Mapping[str, Any]) -> Set[str]:
    """Read actual function/class definitions, rather than guessing from words."""
    meta = chunk.get("metadata") or {}
    names = {
        str(symbol).split(":", 1)[1].casefold()
        for symbol in meta.get("defined_symbols", [])
        if str(symbol).partition(":")[0] in {"function", "class"}
        and ":" in str(symbol)
    }
    if "defined_symbols" not in meta and chunk_matches_modality(chunk, "code"):
        try:
            tree = ast.parse(textwrap.dedent(str(
                chunk.get("raw_content") or chunk.get("content") or "")))
        except (SyntaxError, ValueError):
            return names
        names.update(node.name.casefold() for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    return names


def _known_query_symbols(query, anchors, chunks, modality="all"):
    """Recognize unquoted names such as CamelCase or short function names.

    Ordinary prose is never promoted to an identifier merely because it follows
    'function'. A name must be defined by indexed source code, or already have
    explicit identifier syntax handled by extract_query_anchors.
    """
    if not _has_code_symbol_intent(query, modality):
        return anchors
    explicit_code = modality == "code" or bool(re.search(
        r"\b(?:notebooks?|code|python|implement\w*)\b", query, re.I))
    if _wants_formula(query) and not explicit_code:
        return anchors
    raw_words = {m.group(0) for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", query)}
    words = {word.casefold() for word in raw_words}
    definitions = set().union(*(_defined_callable_names(c) for c in chunks))
    known = (words & definitions) - STOPWORDS - QUERY_FRAMING_WORDS
    naming = r"(?:function|method|class|helper|routine|callable)"
    known = {name for name in known if (
        re.search(r"\b" + naming + r"\s+" + re.escape(name) + r"\b", query, re.I)
        or re.search(r"\b" + re.escape(name) + r"\s+" + naming + r"\b", query, re.I)
        or (explicit_code and re.search(r"\b" + re.escape(name)
            + r"\s+(?:defined|implemented|called|invoked)\b", query, re.I))
        or (explicit_code and any(word.casefold() == name and (
            "_" in word or re.search(r"[a-z][A-Z]|[A-Z]{2}|\d", word)) for word in raw_words))
    )}
    return {**anchors,
            "explicit_symbol": sorted(set(anchors.get("explicit_symbol", [])) | known),
            "symbol": sorted(set(anchors.get("symbol", [])) | known)}


def _is_notebook(chunk: Mapping[str, Any]) -> bool:
    source = str(chunk.get("source_id") or "")
    return (chunk.get("source_type") == "notebook"
            or source.startswith("Notebooks/") or source.endswith(".ipynb"))


def _wants_formula(query: str) -> bool:
    return bool(re.search(
        r"\b(?:equations?|formulas?|formulations?|mathematically)\b|"
        r"\b(?:mathematical|functional|formal)\s+(?:expressions?|forms?)\b|"
        r"\bformally\s+express\w*\b|"
        r"\b(?:derivative|gradient)\s+expressions?\b", query, re.IGNORECASE))

def _term_stem(term: str) -> str:
    """Conservative inflection matching for query-time lexical evidence."""
    if len(term) <= 3 or not term.isalpha():
        return term
    if term.endswith("ies") and len(term) > 4:
        term = term[:-3] + "y"
    elif term.endswith("s") and not term.endswith(("ss", "us", "is")):
        term = term[:-1]
    for suffix in ("ization", "isation", "ation", "ing", "ed"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            # initialize/initializing/initialization share 'initializ'.
            term = term[:-len(suffix)] + ("iz" if suffix == "ization" else "is" if suffix == "isation" else "")
            break
    return term[:-1] if term.endswith("e") and len(term) > 4 else term


def tokenize(text: Optional[str]) -> List[str]:
    """Tokenize text into lowercase alphanumeric and identifier components.

    Splits both `snake_case` (e.g. `gradient_descent` -> `['gradient', 'descent']`)
    and `CamelCase` (e.g. `SelfAttention` -> `['self', 'attention']`).
    """
    if not text:
        return []
    expanded = _CAMEL_BOUNDARY_RE.sub(" ", str(text))
    return _TOKEN_RE.findall(expanded.lower())


def normalize_chapter(chapter: Optional[Any]) -> Optional[str]:
    """Normalize chapter identifiers (e.g. 'Chap07', 'Chapter 7', '07' -> '7')."""
    if chapter is None:
        return None
    raw = str(chapter).strip()
    if not raw:
        return None
    match = re.search(
        r"(?:chap(?:ter)?\s*0*(\d+)|^0*(\d+)$)", raw, re.IGNORECASE
    )
    if match:
        num = match.group(1) or match.group(2)
        return str(int(num))
    app_match = re.search(r"(?:appendix\s*([a-z])|^([a-z])$)", raw, re.IGNORECASE)
    if app_match:
        letter = app_match.group(1) or app_match.group(2)
        return letter.upper()
    return raw.lower()


def chunk_chapter(chunk: Mapping[str, Any]) -> Optional[str]:
    """Extract normalized chapter identifier from a chunk record."""
    meta = chunk.get("metadata") or {}
    for candidate in (
        meta.get("chapter"),
        meta.get("chapter_id"),
        chunk.get("chapter"),
    ):
        norm = normalize_chapter(candidate)
        if norm is not None:
            return norm
    source_id = str(
        chunk.get("source_id")
        or meta.get("notebook_path")
        or chunk.get("source_name")
        or ""
    )
    chap_match = re.search(r"Chap0*(\d+)", source_id, re.IGNORECASE)
    if chap_match:
        return str(int(chap_match.group(1)))
    return None


def chunk_matches_modality(chunk: Mapping[str, Any], modality: str) -> bool:
    """Return True if `chunk` belongs to the requested `modality` filter."""
    mod = (modality or "all").strip().lower()
    if mod == "all":
        return True
    ctype = str(
        chunk.get("content_type") or chunk.get("modality") or ""
    ).strip().lower()
    meta = chunk.get("metadata") or {}

    if mod == "code":
        return ctype in {"code", "import", "notebook_code"}
    if mod == "figure":
        return (
            ctype in {"figure", "algorithm", "table", "image"}
            or meta.get("figure_number") is not None
            or meta.get("figure_name") is not None
            or meta.get("algorithm_number") is not None
        )
    if mod == "equation":
        return (
            ctype in {"equation", "math", "display_equation", "formula"}
            or meta.get("equation_number") is not None
        )
    if mod == "exercise":
        return (
            ctype
            in {
                "exercise",
                "eoc_question",
                "problem",
                "question",
                "notebook_exercise",
            }
            or bool(meta.get("is_template"))
            or meta.get("problem_number") is not None
            or meta.get("exercise_number") is not None
        )
    if mod == "text":
        return ctype in {"text", "markdown", "prose", "section", "narrative"}
    return False


def locator(chunk: Mapping[str, Any]) -> str:
    """Build a human-readable provenance locator for a book or notebook chunk."""
    meta = chunk.get("metadata") or {}
    source_type = str(chunk.get("source_type") or "").strip().lower()
    source_id = str(
        chunk.get("source_id")
        or meta.get("notebook_path")
        or chunk.get("source_name")
        or chunk.get("notebook_path")
        or ""
    ).strip()

    is_notebook = (
        source_type == "notebook"
        or source_id.startswith("Notebooks/")
        or source_id.endswith(".ipynb")
    )
    if is_notebook:
        label = source_id or "Notebook"
        cell = meta.get("cell_index")
        if cell is None:
            cell = meta.get("source_cell", chunk.get("cell_index"))
        parts = [label]
        if cell is not None:
            parts.append(f"cell {int(cell)}")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")
        if start_line is not None and end_line is not None:
            s_line = int(start_line)
            e_line = int(end_line)
            if s_line == e_line:
                parts.append(f"line {s_line}")
            else:
                parts.append(f"lines {s_line}-{e_line}")
        elif start_line is not None:
            parts.append(f"line {int(start_line)}")
        return ", ".join(parts)

    parts = ["UnderstandingDeepLearning.pdf"]
    chapter = meta.get("chapter") or meta.get("chapter_id")
    section = meta.get("section")
    if section:
        parts.append(f"Section {section}")
    elif chapter:
        parts.append(f"Ch. {chapter}")

    pdf_page = meta.get("pdf_page")
    if pdf_page is None:
        legacy_page = meta.get("page", meta.get("page_number", chunk.get("page_number")))
        if legacy_page is not None:
            pdf_page = int(legacy_page) + 1
    if pdf_page is not None:
        page_str = f"PDF page {int(pdf_page)}"
        book_page = meta.get("book_page")
        if book_page is not None:
            page_str += f" (book p. {book_page})"
        parts.append(page_str)

    fig_num = meta.get("figure_number") or meta.get("figure_name")
    alg_num = meta.get("algorithm_number")
    eq_num = meta.get("equation_number")
    prob_num = meta.get("problem_number") or meta.get("exercise_number")

    if fig_num is not None:
        parts.append(f"Figure {fig_num}")
    if alg_num is not None:
        parts.append(f"Algorithm {alg_num}")
    if eq_num is not None:
        parts.append(f"Equation {eq_num}")
    if prob_num is not None:
        parts.append(f"Problem {prob_num}")

    return ", ".join(parts)


def embedding_text(
    chunk: Mapping[str, Any], *, force_recompute: bool = False
) -> str:
    """Return the canonical text input used to embed a chunk.

    Uses `chunk['embedding_text']` when explicitly supplied (unless
    `force_recompute=True`), otherwise combines structured metadata anchors
    (if present and not already in `content`) with `chunk['content']`.
    """
    if not force_recompute:
        explicit = chunk.get("embedding_text")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

    content = str(chunk.get("content") or "").strip()
    meta = chunk.get("metadata") or {}
    if not meta:
        return content

    # A formula's semantic meaning depends on the surrounding source prose.
    # Keep the reviewed transcription itself separate in metadata, while
    # embedding its source context as well as its mathematical notation.
    if chunk.get("content_type") == "equation":
        for key in ("context_before", "context_after"):
            context = str(meta.get(key) or "").strip()
            if context and context not in content:
                content += "\nSource context: " + context

    content_lower = content.lower()
    tags: List[str] = []

    chapter = meta.get("chapter") or meta.get("chapter_id")
    if chapter and f"chapter {chapter}".lower() not in content_lower:
        tags.append(f"Chapter {chapter}")
    for key in ("chapter_title", "subsection"):
        title = str(meta.get(key) or "").strip()
        if title and title.casefold() not in content_lower:
            tags.append(title)

    section = meta.get("section")
    if section and str(section).lower() not in content_lower:
        tags.append(f"Section {section}")

    fig_num = meta.get("figure_number") or meta.get("figure_name")
    if fig_num and f"figure {fig_num}".lower() not in content_lower:
        tags.append(f"Figure {fig_num}")

    alg_num = meta.get("algorithm_number")
    if alg_num and f"algorithm {alg_num}".lower() not in content_lower:
        tags.append(f"Algorithm {alg_num}")

    eq_num = meta.get("equation_number")
    if eq_num and f"equation {eq_num}".lower() not in content_lower:
        tags.append(f"Equation {eq_num}")

    prob_num = meta.get("problem_number") or meta.get("exercise_number")
    if prob_num and f"problem {prob_num}".lower() not in content_lower:
        tags.append(f"Problem {prob_num}")

    symbols = meta.get("symbol_names") or []
    if isinstance(symbols, (list, tuple)):
        missing_syms = [
            str(s)
            for s in symbols
            if s and str(s).lower() not in content_lower
        ]
        if missing_syms:
            tags.append(f"Symbols: {', '.join(missing_syms)}")

    if not tags:
        return content
    header = " | ".join(tags)
    return f"{header}\n{content}" if content else header


def extract_chunk_anchors(chunk: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Extract normalized exact anchors (figure, equation, problem, algorithm, symbol) from a chunk."""
    meta = chunk.get("metadata") or {}
    content = str(chunk.get("content") or "")
    raw_content = str(chunk.get("raw_content") or "")
    combined = f"{content}\n{raw_content}"

    figures: Set[str] = set()
    equations: Set[str] = set()
    problems: Set[str] = set()
    algorithms: Set[str] = set()
    symbols: Set[str] = set()

    fig_val = meta.get("figure_number") or meta.get("figure_name")
    if fig_val is not None:
        cleaned = re.sub(r"^(?:figure|fig\.?)\s*", "", str(fig_val).strip(), flags=re.I)
        if cleaned:
            figures.add(cleaned.lower())
    for m in _FIGURE_REF_RE.finditer(combined):
        figures.add(m.group(1).lower())

    eq_val = meta.get("equation_number")
    if eq_val is not None:
        cleaned = re.sub(
            r"^(?:equation|eq\.?|eqn\.?)\s*",
            "",
            str(eq_val).strip("() "),
            flags=re.I,
        )
        if cleaned:
            equations.add(cleaned.lower())
    for m in _EQUATION_REF_RE.finditer(combined):
        equations.add(m.group(1).lower())

    prob_val = meta.get("problem_number") or meta.get("exercise_number")
    if prob_val is not None:
        cleaned = re.sub(
            r"^(?:problem|prob\.?|exercise|ex\.?|question)\s*",
            "",
            str(prob_val).strip(),
            flags=re.I,
        )
        if cleaned:
            problems.add(cleaned.lower())
    for m in _PROBLEM_REF_RE.finditer(combined):
        problems.add(m.group(1).lower())

    alg_val = meta.get("algorithm_number")
    if alg_val is not None:
        cleaned = re.sub(
            r"^(?:algorithm|alg\.?)\s*", "", str(alg_val).strip(), flags=re.I
        )
        if cleaned:
            algorithms.add(cleaned.lower())
    for m in _ALGORITHM_REF_RE.finditer(combined):
        algorithms.add(m.group(1).lower())

    meta_symbols = meta.get("symbol_names") or chunk.get("symbol_names") or []
    if isinstance(meta_symbols, (list, tuple)):
        for sym in meta_symbols:
            if sym and str(sym).strip():
                symbols.add(str(sym).strip().lower())
    for m in _DEF_SYMBOL_RE.finditer(combined):
        symbols.add(m.group(1).lower())
    for m in _SNAKE_IDENTIFIER_RE.finditer(combined):
        cand = m.group(1).lower()
        if not _is_math_subscript_identifier(cand):
            symbols.add(cand)

    return {
        "figure": sorted(figures),
        "equation": sorted(equations),
        "problem": sorted(problems),
        "algorithm": sorted(algorithms),
        "symbol": sorted(symbols),
    }


def extract_query_anchors(query: str) -> Dict[str, List[str]]:
    """Extract explicit anchors from a user query string.

    Distinguishes strict item anchors (`figure`, `equation`, `problem`,
    `algorithm`, `explicit_symbol`), `chapter` anchors, `notebook` section
    anchors, `page` anchors, and `symbol` candidates.
    """
    q = str(query or "")
    figures = sorted({m.group(1).lower() for m in _FIGURE_REF_RE.finditer(q)})
    equations = sorted({m.group(1).lower() for m in _EQUATION_REF_RE.finditer(q)})
    problems = sorted({m.group(1).lower() for m in _PROBLEM_REF_RE.finditer(q)})
    algorithms = sorted({m.group(1).lower() for m in _ALGORITHM_REF_RE.finditer(q)})
    chapters = sorted({str(int(m.group(1))) for m in _CHAPTER_REF_RE.finditer(q)})
    notebooks = sorted(
        {f"{int(m.group(1))}.{int(m.group(2))}" for m in _NOTEBOOK_REF_RE.finditer(q)}
    )
    pages = sorted({str(int(m.group(1))) for m in _PAGE_REF_RE.finditer(q)})

    explicit_symbols: Set[str] = set()
    for m in _BACKTICK_SYMBOL_RE.finditer(q):
        sym = m.group(1).lower()
        if not _is_math_subscript_identifier(sym):
            explicit_symbols.add(sym)
    for m in _DEF_SYMBOL_RE.finditer(q):
        sym = m.group(1).lower()
        if (
            sym not in QUERY_FRAMING_WORDS
            and sym not in STOPWORDS
            and not _is_math_subscript_identifier(sym)
        ):
            explicit_symbols.add(sym)
    for m in _CALLABLE_SYMBOL_RE.finditer(q):
        sym = m.group(1).lower()
        if not _is_math_subscript_identifier(sym):
            explicit_symbols.add(sym)
    for pattern in (_CODE_WORDING_BEFORE_RE, _CODE_WORDING_AFTER_RE):
        for m in pattern.finditer(q):
            sym = m.group(1).lower()
            if (
                sym not in QUERY_FRAMING_WORDS
                and sym not in STOPWORDS
                and not _is_math_subscript_identifier(sym)
            ):
                explicit_symbols.add(sym)

    soft_symbols: Set[str] = set(explicit_symbols)
    for m in _SNAKE_IDENTIFIER_RE.finditer(q):
        cand = m.group(1).lower()
        if not _is_math_subscript_identifier(cand):
            soft_symbols.add(cand)

    return {
        "figure": figures,
        "equation": equations,
        "problem": problems,
        "algorithm": algorithms,
        "chapter": chapters,
        "notebook": notebooks,
        "page": pages,
        "explicit_symbol": sorted(explicit_symbols),
        "symbol": sorted(soft_symbols),
    }


def extract_metadata_tokens(chunk: Mapping[str, Any]) -> List[str]:
    """Tokenize searchable metadata fields for a chunk."""
    meta = chunk.get("metadata") or {}
    parts: List[str] = []
    for key in ("source_id", "source_type", "content_type"):
        val = chunk.get(key)
        if val:
            parts.append(str(val))
    for key in (
        "chapter",
        "chapter_id",
        "chapter_title",
        "section",
        "subsection",
        "title",
        "notebook_path",
        "figure_number",
        "figure_name",
        "equation_number",
        "algorithm_number",
        "problem_number",
        "exercise_number",
    ):
        val = meta.get(key)
        if val is not None:
            parts.append(str(val))
    symbols = meta.get("symbol_names") or chunk.get("symbol_names") or []
    if isinstance(symbols, (list, tuple)):
        parts.extend(str(s) for s in symbols if s)
    return tokenize(" ".join(parts))


def rrf_fuse(
    *rankings: Sequence[Tuple[str, float]], k: int = 60
) -> List[Tuple[str, float]]:
    """Fuse multiple `(chunk_id, score)` rankings using Reciprocal Rank Fusion.

    Tie-breaking is completely deterministic:
    1. Higher RRF score (rounded to 12 decimal places to avoid float jitter)
    2. Better (smaller) best rank across input rankings
    3. Better (smaller) rank in the first ranking where the item appears
    4. Higher original score in the first ranking where the item appears
    5. Lexicographical `chunk_id`
    """
    if k <= 0:
        raise ValueError("RRF parameter k must be positive")
    fused_scores: Dict[str, float] = defaultdict(float)
    best_rank: Dict[str, int] = {}
    primary_rank: Dict[str, Tuple[int, int]] = {}
    primary_score: Dict[str, float] = {}

    for list_idx, ranking in enumerate(rankings):
        seen_in_list: Set[str] = set()
        for rank, (chunk_id, orig_score) in enumerate(ranking, start=1):
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            fused_scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                best_rank[chunk_id] = rank
            if chunk_id not in primary_rank:
                primary_rank[chunk_id] = (list_idx, rank)
                primary_score[chunk_id] = float(orig_score)

    return sorted(
        fused_scores.items(),
        key=lambda item: (
            -round(item[1], 12),
            best_rank.get(item[0], 10**9),
            primary_rank.get(item[0], (10**9, 10**9)),
            -round(primary_score.get(item[0], 0.0), 12),
            item[0],
        ),
    )


@dataclass
class AbstentionConfig:
    """Evidence-sensitive abstention thresholds based on original scores.

    Calibrated for `Qwen/Qwen3-Embedding-0.6B` (relevant ~0.83, unrelated ~0.38)
    and BM25 content-term evidence. RRF scores are never used as confidence.
    """

    min_bm25_score: float = 0.20
    min_semantic_score: float = 0.42
    strong_semantic_score: float = 0.62
    min_hybrid_semantic_score: float = 0.40
    min_content_term_coverage: float = 0.25


class HybridRetriever:
    """Handwritten BM25 + exact cosine + RRF retriever backed by `SQLiteStore`."""

    def __init__(
        self,
        store: Any,
        embedder: Optional[Any] = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        rrf_k: int = 60,
        meta_weight: float = 2.0,
        abstention: Optional[AbstentionConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.k1 = float(k1)
        self.b = float(b)
        self.rrf_k = int(rrf_k)
        self.meta_weight = float(meta_weight)
        self.abstention = abstention or AbstentionConfig()

    def _validate_request(self, request: SearchRequest) -> Tuple[str, str, int]:
        if not isinstance(request, SearchRequest):
            raise UDLException(
                "invalid_search_request",
                "search() requires a SearchRequest instance.",
                retryable=False,
            )
        mode = (request.mode or "hybrid").strip().lower()
        if mode not in VALID_MODES:
            raise UDLException(
                "mode_invalid",
                f"Unsupported search mode '{request.mode}'. Must be one of {sorted(VALID_MODES)}.",
                retryable=False,
            )
        modality = (request.modality or "all").strip().lower()
        if modality not in VALID_MODALITIES:
            raise UDLException(
                "modality_invalid",
                f"Unsupported modality '{request.modality}'. Must be one of {sorted(VALID_MODALITIES)}.",
                retryable=False,
            )
        top_k = int(request.top_k if request.top_k is not None else 6)
        if top_k < 1:
            raise UDLException(
                "invalid_top_k",
                "top_k must be >= 1.",
                retryable=False,
            )
        return mode, modality, top_k

    def _check_unsupported_external_query(
        self,
        raw_query: str,
        candidate_ids: Sequence[str],
        chunks_by_id: Mapping[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Abstain when the query asks for an explicitly unindexed external artifact."""
        match = _UNSUPPORTED_EXTERNAL_RE.search(raw_query)
        if not match:
            return None
        phrase = match.group(0).strip().lower()
        for cid in candidate_ids:
            chunk = chunks_by_id.get(cid) or {}
            # A book mentioning an external artifact is not that artifact.
            # Only an actually indexed source identity can satisfy provenance.
            identity = re.sub(r"[_/.-]+", " ", str(chunk.get("source_id") or "")).casefold()
            if phrase in identity:
                return None
        return f"Query requests unindexed or nonexistent external material ('{phrase}')."

    def _check_named_requirements(self, query, candidate_ids, chunks):
        """A known general subject cannot substantiate an absent named item.

        This differs from OOV phrasing: deliberate acronym/CamelCase/proper-name
        spelling identifies a requested entity, while ordinary lowercase words
        can remain semantic paraphrases.
        """
        names = []
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*\b", query):
            word = match.group(0)
            if (word.casefold() in STOPWORDS | QUERY_FRAMING_WORDS
                    or len(word) <= 2 or match.start() == 0):
                continue
            deliberate = bool(re.search(r"[a-z][A-Z]|[A-Z]{2}|[A-Za-z]\d", word))
            proper_name = word[0].isupper() and word[1:].islower()
            if deliberate or proper_name:
                names.append(word)
        if not names:
            return None
        corpus_words = set()
        corpus_passages = []
        source_ids = set()
        for cid in candidate_ids:
            chunk = chunks[cid]
            source_ids.add(chunk.get("source_id"))
            corpus_passages.append(" " + " ".join(tokenize(str(chunk.get("content") or ""))) + " ")
            corpus_words.update(word.casefold() for word in re.findall(
                r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*", str(chunk.get("content") or "")))
            corpus_words.update(str(name).casefold() for name in (chunk.get("metadata") or {}).get("symbol_names", []))
        if hasattr(self.store, "get_source"):
            for source_id in source_ids:
                source = self.store.get_source(source_id) or {}
                identity = " ".join(str(source.get(key) or "") for key in ("source_id", "title", "path"))
                corpus_passages.append(" " + " ".join(tokenize(identity)) + " ")
                corpus_words.update(word.casefold() for word in re.findall(
                    r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*", identity))
        for name in names:
            if name.casefold() not in corpus_words:
                return f"Named requirement '{name}' was not found in the requested source scope."
        # A named compound identifies one concept. Finding its component words
        # in unrelated passages does not establish that the concept is indexed.
        for match in re.finditer(r"\b[A-Z][a-z]+(?:[ -]+[A-Z][a-z]+)+\b", query):
            words = [word for word in tokenize(match.group())
                     if word not in STOPWORDS | QUERY_FRAMING_WORDS]
            if len(words) > 1 and not any(" " + " ".join(words) + " " in passage
                                          for passage in corpus_passages):
                return f"Named requirement '{match.group()}' was not found in the requested source scope."
        return None

    def _check_anchor_matches(
        self,
        raw_query: str,
        modality: str,
        query_anchors: Dict[str, List[str]],
        candidate_ids: Sequence[str],
        anchors_by_chunk: Mapping[str, Dict[str, List[str]]],
        chunks_by_id: Mapping[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, float], Optional[str]]:
        """Verify exact anchors in query and return `(item_boosts, chapter_boosts, abstain_reason)`.

        If the query specifies an explicit numbered reference (figure, equation,
        problem, algorithm, chapter) or an explicit code symbol (with genuine
        code intent: backticks, callable/function/class wording, or code
        modality) that does not exist in any candidate chunk, returns an
        abstention reason string. Unquoted underscore tokens without code intent
        soft-boost matching chunks without hard-excluding the query.
        """
        item_boosts: Dict[str, float] = defaultdict(float)
        chapter_boosts: Dict[str, float] = defaultdict(float)

        strict_types = (
            ("figure", "Figure", 35.0),
            ("equation", "Equation", 35.0),
            ("problem", "Problem", 35.0),
            ("algorithm", "Algorithm", 35.0),
            ("explicit_symbol", "Symbol", 25.0),
        )
        for key, label, boost_val in strict_types:
            requested_vals = query_anchors.get(key) or []
            lookup_key = "symbol" if key == "explicit_symbol" else key
            for ref_val in requested_vals:
                matched_any = False
                for cid in candidate_ids:
                    chunk_anc = anchors_by_chunk.get(cid) or {}
                    chunk_text = str(
                        (chunks_by_id.get(cid) or {}).get("content") or ""
                    ).lower()
                    if ref_val in (chunk_anc.get(lookup_key) or []) or (
                        key == "explicit_symbol"
                        and re.search(r"(?<!\w)" + re.escape(ref_val) + r"(?!\w)", chunk_text)
                    ):
                        item_boosts[cid] += boost_val
                        matched_any = True
                if not matched_any:
                    return (
                        {},
                        {},
                        f"Exact {label.lower()} reference '{ref_val}' was not found in the indexed corpus.",
                    )

        # Verify soft snake_case symbol identifiers: only hard-exclude when the
        # query has genuine code intent (backticks, function/class/callable
        # wording, or code modality).
        code_intent = _has_code_symbol_intent(raw_query, modality)
        for sym_val in query_anchors.get("symbol") or []:
            if sym_val in (query_anchors.get("explicit_symbol") or []):
                continue
            matched_sym = False
            for cid in candidate_ids:
                chunk_anc = anchors_by_chunk.get(cid) or {}
                chunk_text = str(
                    (chunks_by_id.get(cid) or {}).get("content") or ""
                ).lower()
                if sym_val in (chunk_anc.get("symbol") or []) or re.search(
                    r"(?<!\w)" + re.escape(sym_val) + r"(?!\w)", chunk_text):
                    item_boosts[cid] += 15.0
                    matched_sym = True
            if not matched_sym and code_intent:
                return (
                    {},
                    {},
                    f"Exact symbol reference '{sym_val}' was not found in the indexed corpus.",
                )

        # Verify explicit chapter references in query text (e.g. "In chapter 14", "chapter 99")
        for chap_val in query_anchors.get("chapter") or []:
            matched_chap = False
            for cid in candidate_ids:
                chunk = chunks_by_id.get(cid) or {}
                if chunk_chapter(chunk) == chap_val:
                    chapter_boosts[cid] += 12.0
                    matched_chap = True
            if not matched_chap:
                return (
                    {},
                    {},
                    f"Exact chapter reference '{chap_val}' was not found in the indexed corpus.",
                )

        # A requested notebook is an exact source reference, including when its
        # names happen to share terms with another notebook.
        for nb_ref in query_anchors.get("notebook") or []:
            parts = nb_ref.split(".")
            if len(parts) == 2:
                nb_prefix = f"/{parts[0]}_{parts[1]}_"
                matched_notebook = False
                for cid in candidate_ids:
                    chunk = chunks_by_id.get(cid) or {}
                    src_id = str(chunk.get("source_id") or "")
                    if nb_prefix in src_id:
                        chapter_boosts[cid] += 14.0
                        matched_notebook = True
                if not matched_notebook:
                    return {}, {}, f"Exact notebook reference '{nb_ref}' was not found in the indexed corpus."

        # Boost explicit page references (e.g. "page 141")
        for page_str in query_anchors.get("page") or []:
            page_num = int(page_str)
            matched_page = False
            for cid in candidate_ids:
                chunk = chunks_by_id.get(cid) or {}
                meta = chunk.get("metadata") or {}
                pdf_p = meta.get("pdf_page")
                book_p = meta.get("book_page")
                pdf_pages = meta.get("pdf_pages") or []
                if (
                    pdf_p == page_num
                    or book_p == page_num
                    or page_num in pdf_pages
                ):
                    item_boosts[cid] += 20.0
                    matched_page = True
            if not matched_page:
                return {}, {}, f"Exact page reference '{page_str}' was not found in the indexed corpus."

        return item_boosts, chapter_boosts, None

    def _score_lexical(
        self,
        query_tokens: Sequence[str],
        candidate_ids: Sequence[str],
        index_data: Mapping[str, Any],
        item_boosts: Mapping[str, float],
        chapter_boosts: Mapping[str, float],
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Set[str]]]:
        """Score candidate chunks using handwritten BM25 over SQLite postings + anchor boosts."""
        candidate_set = set(candidate_ids)
        total_docs = max(1, int(index_data.get("doc_count") or len(candidate_ids) or 1))
        avg_dl = max(1.0, float(index_data.get("avg_doc_length") or 1.0))
        doc_lengths: Mapping[str, float] = index_data.get("doc_lengths") or {}
        postings: Mapping[str, Mapping[str, Tuple[int, int]]] = (
            index_data.get("postings") or {}
        )
        df_map: Mapping[str, int] = index_data.get("df") or {}

        scores: Dict[str, float] = defaultdict(float)
        matched_terms_by_chunk: Dict[str, Set[str]] = defaultdict(set)

        for cid, boost in item_boosts.items():
            if cid in candidate_set and boost > 0.0:
                scores[cid] += float(boost)

        unique_terms = list(dict.fromkeys(query_tokens))
        for term in unique_terms:
            term_postings = postings.get(term)
            if not term_postings:
                continue
            df = int(df_map.get(term) or len(term_postings))
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            if term in STOPWORDS:
                term_weight = 0.05
            elif term in QUERY_FRAMING_WORDS:
                term_weight = 0.20
            elif "\x00" in term:
                term_weight = 0.50
            else:
                term_weight = 1.0

            for cid, (tf_content, tf_meta) in term_postings.items():
                if cid not in candidate_set:
                    continue
                tf = float(tf_content) + self.meta_weight * float(tf_meta)
                if tf <= 0.0:
                    continue
                dl = float(doc_lengths.get(cid, avg_dl))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
                scores[cid] += term_weight * idf * (tf * (self.k1 + 1.0)) / denom
                matched_terms_by_chunk[cid].add(term)

        # Apply chapter boost only to chunks that also matched at least one query term
        for cid, c_boost in chapter_boosts.items():
            if cid in candidate_set and c_boost > 0.0 and scores.get(cid, 0.0) > 0.0:
                scores[cid] += float(c_boost)

        ranked = sorted(
            ((cid, sc) for cid, sc in scores.items() if sc > 0.0),
            key=lambda item: (
                -round(item[1], 12),
                -len(matched_terms_by_chunk.get(item[0], ())),
                item[0],
            ),
        )
        return ranked, matched_terms_by_chunk

    def _score_semantic(
        self,
        query: str,
        candidate_ids: Sequence[str],
        vectors_by_chunk: Mapping[str, Tuple[float, ...]],
        build_info: Mapping[str, Any],
        bm25_map: Mapping[str, float],
    ) -> List[Tuple[str, float]]:
        """Score candidate chunks using exact cosine similarity with deterministic tie-breaking."""
        if self.embedder is None:
            raise UDLException(
                "dense_missing",
                "Semantic retrieval requires a configured LocalEmbedder.",
                retryable=False,
            )

        expected_dim = build_info.get("embedding_dimension")
        build_fp = build_info.get("embedding_fingerprint")
        embedder_fp = getattr(self.embedder, "fingerprint", None)
        if build_fp and embedder_fp and str(build_fp) != str(embedder_fp):
            raise UDLException(
                "dense_stale",
                "Active store embedding model/profile fingerprint does not match the query embedder.",
                retryable=False,
            )

        query_vectors = self.embedder.encode_queries([query])
        if not query_vectors or not query_vectors[0]:
            raise UDLException(
                "embedding_query_failed",
                "Query embedder returned no vector.",
                retryable=False,
            )
        q_vec = query_vectors[0]
        if expected_dim is not None and len(q_vec) != int(expected_dim):
            raise UDLException(
                "embedding_dimension_mismatch",
                f"Query embedding dimension {len(q_vec)} != store dimension {expected_dim}.",
                retryable=False,
            )

        scored: List[Tuple[str, float]] = []
        for cid in candidate_ids:
            doc_vec = vectors_by_chunk.get(cid)
            if doc_vec is None:
                raise UDLException(
                    "dense_missing",
                    f"Missing dense embedding vector for chunk '{cid}'.",
                    retryable=False,
                )
            sim = cosine_similarity(q_vec, doc_vec)
            scored.append((cid, float(sim)))

        return sorted(
            scored,
            key=lambda item: (
                -round(item[1], 12),
                -round(float(bm25_map.get(item[0], 0.0)), 12),
                item[0],
            ),
        )

    def _should_abstain(
        self,
        *,
        effective_mode: str,
        query_tokens: Sequence[str],
        content_terms: Sequence[str],
        core_subject_terms: Sequence[str],
        item_boosts: Mapping[str, float],
        top_chunk_ids: Sequence[str],
        bm25_map: Mapping[str, float],
        semantic_map: Mapping[str, Optional[float]],
        matched_terms_by_chunk: Mapping[str, Set[str]],
        df_map: Mapping[str, int],
        anchor_locator: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Apply conservative evidence-sensitive abstention using original scores."""
        if not top_chunk_ids:
            return True, "No matching evidence found in the indexed UDL corpus."

        has_item_anchor_match = any(
            item_boosts.get(cid, 0.0) > 0.0 for cid in top_chunk_ids
        )
        if has_item_anchor_match and anchor_locator:
            return False, None

        top_cid = top_chunk_ids[0]
        top_bm25 = float(bm25_map.get(top_cid, 0.0))
        max_bm25 = max(
            (float(bm25_map.get(cid, 0.0)) for cid in top_chunk_ids), default=0.0
        )
        top_sem = semantic_map.get(top_cid)
        max_sem = max(
            (
                float(semantic_map[cid])
                for cid in top_chunk_ids
                if semantic_map.get(cid) is not None
            ),
            default=-1.0,
        )

        content_set = set(content_terms)
        subject_set = set(core_subject_terms)

        best_content_overlap = 0
        best_subject_overlap = 0
        for cid in top_chunk_ids[:3]:
            matched = matched_terms_by_chunk.get(cid) or set()
            c_ov = len(matched & content_set) if content_set else len(matched)
            s_ov = len(matched & subject_set) if subject_set else c_ov
            if c_ov > best_content_overlap:
                best_content_overlap = c_ov
            if s_ov > best_subject_overlap:
                best_subject_overlap = s_ov

        # Check how many core subject terms exist anywhere in the corpus vocabulary
        oov_subjects = [
            t for t in subject_set if df_map.get(t, 0) == 0 and len(t) >= 3
        ]

        coverage = (
            best_content_overlap / len(content_set)
            if content_set
            else (1.0 if top_bm25 > 0 else 0.0)
        )
        subject_coverage = (
            best_subject_overlap / len(subject_set)
            if subject_set
            else coverage
        )
        if has_item_anchor_match and (not subject_set or (
            not oov_subjects and subject_coverage >= self.abstention.min_content_term_coverage
        )):
            return False, None

        # Unknown wording can be a paraphrase, not a missing topic. Require
        # cosine and lexical subject support in the SAME piece of evidence;
        # unrelated dense and lexical hits must not validate each other.
        strong_subject_evidence = any(
            float(semantic_map.get(cid) or -1.0) >= self.abstention.strong_semantic_score
            and bool((matched_terms_by_chunk.get(cid) or set()) & subject_set)
            and len((matched_terms_by_chunk.get(cid) or set()) & subject_set)
                / max(1, len(subject_set)) >= self.abstention.min_content_term_coverage
            for cid in top_chunk_ids[:3]
        )

        if effective_mode == "lexical":
            if content_set and best_content_overlap == 0:
                return (
                    True,
                    "No non-stopword query terms matched any indexed chunk.",
                )
            if subject_set and best_subject_overlap == 0:
                return (
                    True,
                    "None of the core subject terms in the query matched the corpus.",
                )
            if oov_subjects:
                return (
                    True,
                    f"Core query terms not present in UDL corpus: {sorted(oov_subjects)}.",
                )
            if max_bm25 < self.abstention.min_bm25_score:
                return (
                    True,
                    f"Lexical BM25 score ({max_bm25:.3f}) is below abstention threshold ({self.abstention.min_bm25_score:.3f}).",
                )
            if (
                len(subject_set) >= 2
                and subject_coverage < self.abstention.min_content_term_coverage
            ):
                return (
                    True,
                    f"Query subject term coverage ({subject_coverage:.2f}) is insufficient for grounded retrieval.",
                )
            return False, None

        if effective_mode == "semantic":
            if top_sem is None or max_sem < self.abstention.min_semantic_score:
                return (
                    True,
                    f"Semantic cosine similarity ({max_sem:.3f}) is below abstention threshold ({self.abstention.min_semantic_score:.3f}).",
                )
            if oov_subjects and not strong_subject_evidence:
                return (
                    True,
                    f"Core query terms not present in UDL corpus: {sorted(oov_subjects)}.",
                )
            if (
                max_sem < self.abstention.strong_semantic_score
                and subject_set
                and best_subject_overlap == 0
            ):
                return (
                    True,
                    "Moderate semantic similarity without sufficient lexical subject-term support.",
                )
            return False, None

        # effective_mode == "hybrid" (never uses RRF score for abstention)
        has_strong_semantic = strong_subject_evidence
        has_solid_lexical = (
            max_bm25 >= self.abstention.min_bm25_score
            and (not subject_set or best_subject_overlap > 0)
            and not oov_subjects
            and (
                len(subject_set) < 2
                or subject_coverage >= self.abstention.min_content_term_coverage
            )
        )
        has_combined_evidence = (
            best_subject_overlap > 0
            and not oov_subjects
            and max_bm25 >= self.abstention.min_bm25_score
            and max_sem >= self.abstention.min_hybrid_semantic_score
        )

        if has_strong_semantic or has_solid_lexical or has_combined_evidence:
            return False, None

        return (
            True,
            "Insufficient original lexical (BM25) and semantic (cosine) evidence for grounded answer.",
        )

    def _source_subqueries(self, query: str, modality: str, chunks):
        """Separate requested book reasoning and notebook work by clause syntax.

        A shared chapter is inherited only when a clause has no chapter or
        notebook reference of its own. Explicit cross-chapter requests survive.
        """
        if modality != "all":
            return []
        code_marker = r"\b(?:notebooks?|code|python|implement\w*)\b"
        book_marker = r"\b(?:book|textbook|theory|theoretical|conceptual|mathematical|mathematically|equation|formula|formulation)\b"
        concept_marker = r"\b(?:book|textbook|theory|theoretical|conceptual|mathematical|mathematically|equation|formula|formulation|explain\w*|describ\w*|summar\w*)\b"
        clauses = [p.strip(" ,;?") for p in re.split(
            r"[;?]+|,\s*and\s+|\band\s+(?=(?:how|where|what|which|show|find|locate)\b)|"
            r"\b(?:with|to|and)\s+(?=(?:(?:the|its|their|a)\s+)?(?:corresponding\s+)?"
            r"(?:(?:assignment|example|source)\s+)?(?:notebook|code|implementation|implemented)\b)",
            query, flags=re.I) if p.strip(" ,;?")]
        coordinated_explanation = bool(re.match(r"\s*(?:explain|describe|summarize)\b", query, re.I)
            and re.search(r"\band\b", query, re.I))
        distinct_concept_and_notebook = (
            any(bool(re.search(concept_marker, c, re.I) or _wants_formula(c)) and not re.search(code_marker, c, re.I) for c in clauses)
            and any(bool(re.search(code_marker, c, re.I)) for c in clauses)
        )
        if not re.search(code_marker, query, re.I) or not (
            re.search(book_marker, query, re.I)
            or _wants_formula(query)
            or coordinated_explanation
            or distinct_concept_and_notebook
            or re.search(r"\b(?:connect|relate|compare)\b", query, re.I)
        ):
            return []
        # Mathematical wording inside a notebook-only question does not ask
        # for a second source. A separate theory/math clause or an explicit
        # book/connection request is needed before allocating a book pool.
        if all(re.search(code_marker, clause, re.I) for clause in clauses) and not coordinated_explanation and not re.search(
            r"\b(?:book|textbook|connect|relate|compare)\b", query, re.I):
            return []
        book_parts, notebook_parts = [], []
        for clause in clauses:
            code = bool(re.search(code_marker, clause, re.I))
            book = bool(re.search(concept_marker if distinct_concept_and_notebook else book_marker, clause, re.I) or _wants_formula(clause))
            if code and not book:
                notebook_parts.append(clause)
            elif book and not code:
                book_parts.append(clause)
            else:
                # A shared introductory subject belongs to both facets.
                book_parts.append(clause)
                notebook_parts.append(clause)
        book_query = "; ".join(book_parts) or query
        notebook_query = "; ".join(notebook_parts) or query
        notebook_subjects = [term for term in tokenize(notebook_query)
            if term not in STOPWORDS | QUERY_FRAMING_WORDS and not term.isdigit()]
        if not notebook_subjects:
            # 'show its implementation in the corresponding notebook' has no
            # independent topic. Carry the stated book subject, rather than
            # accepting arbitrary code on generic implementation wording.
            shared_subject = book_query
            for pattern in (_FIGURE_REF_RE, _EQUATION_REF_RE, _PROBLEM_REF_RE,
                            _ALGORITHM_REF_RE, _CHAPTER_REF_RE, _PAGE_REF_RE):
                shared_subject = pattern.sub(" ", shared_subject)
            shared_subject = " ".join(term for term in tokenize(shared_subject)
                if term not in STOPWORDS | QUERY_FRAMING_WORDS and not term.isdigit())
            if shared_subject:
                notebook_query = shared_subject + "; " + notebook_query
        global_anchors = extract_query_anchors(query)
        def referenced_book_chapters(anchors):
            return sorted({chapter for chunk in chunks.values()
                if chunk.get("source_id") == "book"
                and any(str((chunk.get("metadata") or {}).get(key + "_number", "")).casefold()
                        in anchors[key] for key in ("equation", "figure", "problem", "algorithm"))
                if (chapter := chunk_chapter(chunk)) is not None})
        shared_chapters = (global_anchors["chapter"]
            or referenced_book_chapters(global_anchors)
            or sorted({ref.split(".")[0] for ref in global_anchors["notebook"]}))
        plans = []
        for source, subquery in (("book", book_query), ("notebook", notebook_query)):
            local = extract_query_anchors(subquery)
            chapters = (local["chapter"]
                or sorted({ref.split(".")[0] for ref in local["notebook"]})
                or referenced_book_chapters(local))
            if not chapters and len(shared_chapters) == 1:
                chapters = shared_chapters
            plans.append((source, subquery, chapters, local["notebook"]))
        return plans

    @staticmethod
    def _interleave_source_pools(pools, top_k):
        """Bound the total context while giving each requested source a turn."""
        result, seen = [], set()
        for rank in range(max((len(pool) for pool in pools), default=0)):
            for pool in pools:
                if rank < len(pool) and pool[rank][0] not in seen:
                    result.append(pool[rank])
                    seen.add(pool[rank][0])
                    if len(result) >= top_k:
                        return result
        return result

    def _rank_evidence(self, query, query_tokens, candidate_ids, index_data,
                       item_boosts, chapter_boosts, effective_mode,
                       vectors_by_chunk, top_k):
        """Rank one query within one evidence pool, retaining original scores."""
        contextual_index = self._contextual_lexical_terms(query_tokens, candidate_ids, index_data)
        lexical, matched = self._score_lexical(
            contextual_index.get("query_tokens", query_tokens), candidate_ids, contextual_index, item_boosts, chapter_boosts)
        bm25 = dict(lexical)
        semantic = []
        semantic_map = {cid: None for cid in candidate_ids}
        if effective_mode in {"semantic", "hybrid"}:
            semantic = self._score_semantic(
                query, candidate_ids, vectors_by_chunk, index_data["build_info"], bm25)
            semantic_map.update(semantic)
        if effective_mode == "lexical":
            ranking = lexical
        elif effective_mode == "semantic":
            ranking = sorted(semantic, key=lambda item: (
                -(item_boosts.get(item[0], 0) > 0),
                -(chapter_boosts.get(item[0], 0) > 0),
                -round(item[1], 12), -round(bm25.get(item[0], 0), 12), item[0]))
        else:
            depth = max(30, top_k * 6)
            ranking = rrf_fuse(lexical[:depth], semantic[:depth], k=self.rrf_k)
        return list(ranking), bm25, semantic_map, matched

    def _contextual_lexical_terms(self, query_tokens, candidate_ids, index_data):
        """Score complete source passages that explain small equation/code chunks.

        Formula notation and initialization statements rarely repeat the words
        in the question. Resolve only existing same-source links, leaving stored
        content, provenance and embeddings untouched.
        """
        chunks = index_data["chunks_by_id"]
        terms = set(query_tokens) - STOPWORDS - QUERY_FRAMING_WORDS
        effective_lengths = dict(index_data.get("doc_lengths") or {})
        query_stems = defaultdict(set)
        for term in terms:
            query_stems[_term_stem(term)].add(term)
        postings = {term: dict(rows) for term, rows in index_data["postings"].items()
                    if term in query_tokens}
        figure_owners = defaultdict(list)
        phrases = {(a, b) for a, b in zip(query_tokens, query_tokens[1:])
            if a not in STOPWORDS and b not in STOPWORDS
            and not (a in QUERY_FRAMING_WORDS and b in QUERY_FRAMING_WORDS)
            and not (a.isdigit() or b.isdigit())}
        phrase_stems = {(_term_stem(a), _term_stem(b)): a + "\x00" + b for a, b in phrases}
        phrase_postings = defaultdict(dict)
        for owner in chunks.values():
            if owner.get("content_type") == "text":
                for figure in extract_chunk_anchors(owner)["figure"]:
                    figure_owners[(owner.get("source_id"), figure)].append(owner)
        for indexed_term, rows in index_data["postings"].items():
            for query_term in query_stems.get(_term_stem(indexed_term), ()):
                if query_term == indexed_term:
                    continue
                expanded = postings.setdefault(query_term, {})
                for cid, (content_tf, meta_tf) in rows.items():
                    prev = expanded.get(cid, (0, 0))
                    expanded[cid] = (max(prev[0], content_tf), max(prev[1], meta_tf))
        for cid in candidate_ids:
            chunk = chunks[cid]
            meta = chunk.get("metadata") or {}
            context = ""
            if chunk.get("content_type") == "equation":
                context = " ".join(str(meta.get(key) or "") for key in ("context_before", "context_after"))
                before = chunks.get(meta.get("previous_id")) or {}
                if (before.get("content_type") == "text"
                        and before.get("source_id") == chunk.get("source_id")
                        and str(meta.get("equation_number") or "") in extract_chunk_anchors(before)["equation"]):
                    context += " " + str(before.get("content") or "")
            elif chunk.get("content_type") == "figure":
                ref = str(meta.get("figure_number") or "")
                page = meta.get("pdf_page")
                contexts = []
                for owner in figure_owners.get((chunk.get("source_id"), ref), []):
                    owner_meta = owner.get("metadata") or {}
                    start = owner_meta.get("pdf_page")
                    end = owner_meta.get("end_pdf_page", start)
                    if (page is not None and start is not None and end is not None
                            and int(start) - 1 <= int(page) <= int(end) + 1):
                        contexts.append(str(owner.get("content") or ""))
                context = " ".join(dict.fromkeys(contexts))
            elif chunk_matches_modality(chunk, "code"):
                before = chunks.get(meta.get("previous_id")) or {}
                if before.get("content_type") == "markdown" and before.get("source_id") == chunk.get("source_id"):
                    context = str(before.get("content") or "")
            if context:
                # Borrowed passage words must carry their passage length;
                # otherwise a tiny display receives an artificial BM25 boost
                # for hundreds of words it does not actually contain.
                effective_lengths[cid] = max(float(effective_lengths.get(cid, 0)),
                    len(tokenize(str(chunk.get("content") or ""))) + len(tokenize(context)))
            for context_term in set(tokenize(context)):
                for term in query_stems.get(_term_stem(context_term), ()):
                    rows = postings.setdefault(term, {})
                    content_tf, meta_tf = rows.get(cid, (0, 0))
                    rows[cid] = (max(content_tf, 1), meta_tf)
            # Ordered phrases preserve the relation between words. Inverting
            # 'input gradient' to 'gradient input', or matching both words in
            # unrelated sentences, should not receive the same evidence.
            evidence_tokens = [_term_stem(term) for term in tokenize(
                str(chunk.get("content") or "") + " " + context)]
            counts = defaultdict(int)
            for pair in zip(evidence_tokens, evidence_tokens[1:]):
                phrase = phrase_stems.get(pair)
                if phrase:
                    counts[phrase] += 1
            for phrase, count in counts.items():
                phrase_postings[phrase][cid] = (count, 0)
        # Keep original document frequencies so metadata context does not
        # redefine the corpus vocabulary or the OOV abstention guard.
        phrase_df = {phrase: len(rows) for phrase, rows in phrase_postings.items()}
        return {**index_data, "postings": {**index_data["postings"], **postings, **phrase_postings},
                "df": {**index_data["df"], **phrase_df}, "doc_lengths": effective_lengths,
                "query_tokens": list(query_tokens) + sorted(phrase_postings)}

    def _select_source_passages(self, query, ranking, chunks, candidate_ids, top_k, modality="all"):
        """Keep one complete owning passage before repeated dependent snippets.

        Numbered lookups and named functions retain exact chunk priority. For
        conceptual/formula questions a prose passage can own several consecutive
        displays. Resolve that explicit relationship instead of spending all
        five slots on nearly identical factors from the same explanation.
        """
        anchors = extract_query_anchors(query)
        if modality != "all":
            return ranking
        if any(anchors[key] for key in ("equation", "figure", "problem", "algorithm", "explicit_symbol")):
            return ranking
        if re.search(r"\b(?:figure|diagram)\b", query, re.I):
            return ranking
        candidates = set(candidate_ids)
        score_by_id = dict(ranking)
        formula = _wants_formula(query)
        setup = bool(re.search(r"\b(?:defined|introduced|initiali[sz]\w*|loaded|created|constructed)\b", query, re.I))
        notebook = bool(re.search(r"\b(?:notebooks?|code|python|implement\w*)\b", query, re.I))
        if notebook:
            if not setup:
                return ranking
            selected, seen = [], set()
            for item in ranking:
                cid = item[0]
                if cid in seen:
                    continue
                selected.append(item)
                seen.add(cid)
                chunk = chunks[cid]
                following = chunks.get((chunk.get("metadata") or {}).get("next_id")) or {}
                if (chunk.get("content_type") == "markdown" and following.get("chunk_id") in candidates
                        and following.get("source_id") == chunk.get("source_id")
                        and chunk_matches_modality(following, "code")
                        and any(str(s).startswith("variable:") for s in (following.get("metadata") or {}).get("defined_symbols", []))):
                    next_id = following["chunk_id"]
                    if next_id not in seen:
                        selected.append((next_id, score_by_id.get(next_id, 0.0)))
                        seen.add(next_id)
                if len(selected) >= top_k:
                    break
            return selected[:top_k] + [item for item in ranking if item[0] not in seen]
        owners_by_ref = defaultdict(list)
        for cid in candidate_ids:
            source_chunk = chunks[cid]
            if source_chunk.get("content_type") == "text":
                for ref in extract_chunk_anchors(source_chunk)["equation"]:
                    owners_by_ref[(source_chunk.get("source_id"), ref)].append(cid)
        selected, seen, owners = [], set(), set()
        deferred = []
        for item in ranking:
            cid = item[0]
            chunk = chunks[cid]
            owner = None
            if chunk.get("content_type") == "equation":
                ref = str((chunk.get("metadata") or {}).get("equation_number") or "")
                meta = chunk.get("metadata") or {}
                owning = [candidate for candidate in owners_by_ref.get((chunk.get("source_id"), ref), [])
                    if (chunks[candidate].get("metadata") or {}).get("pdf_page") == meta.get("pdf_page")]
                if owning:
                    owner = owning[0]
                current = chunk
                for _ in range(0 if owner else 8):
                    previous = chunks.get((current.get("metadata") or {}).get("previous_id"))
                    if not previous or previous.get("source_id") != chunk.get("source_id"):
                        break
                    if previous.get("content_type") == "text":
                        if ref and ref in extract_chunk_anchors(previous)["equation"]:
                            owner = previous["chunk_id"]
                        break
                    current = previous
            if owner and owner in candidates:
                if owner in owners:
                    deferred.append(item)
                    continue
                owners.add(owner)
                if owner not in seen:
                    selected.append((owner, score_by_id.get(owner, 0.0)))
                    seen.add(owner)
                deferred.append(item)
            elif cid not in seen:
                selected.append(item)
                seen.add(cid)
                if not formula and chunk.get("content_type") == "text":
                    meta = chunk.get("metadata") or {}
                    following = chunks.get(meta.get("next_id")) or {}
                    next_meta = following.get("metadata") or {}
                    continuation = (meta.get("section") == next_meta.get("section")
                        and meta.get("subsection") == next_meta.get("subsection"))
                    introduction = meta.get("section") is None and chunk_chapter(chunk) == chunk_chapter(following)
                    next_id = following.get("chunk_id")
                    if (next_id in candidates and next_id not in seen
                            and following.get("content_type") == "text"
                            and following.get("source_id") == chunk.get("source_id")
                            and (continuation or introduction)):
                        selected.append((next_id, score_by_id.get(next_id, 0.0)))
                        seen.add(next_id)
            if len(selected) >= top_k:
                break
        return selected[:top_k] + [item for item in deferred + list(ranking) if item[0] not in seen]

    def _facet_evidence_pools(self, query, candidate_ids, index_data, modality):
        """Preserve both substantive parts of a same-source 'while' question.

        A common section is preferred only if actual source text supports each
        facet independently. Shared chapter titles or topic-name lists cannot
        manufacture that connection. Otherwise each facet keeps its own pool.
        """
        anchors = extract_query_anchors(query)
        clauses = [clause.strip() for clause in re.split(r"\bwhile\b", query, flags=re.I) if clause.strip()]
        if (modality != "all" or len(clauses) != 2 or len(anchors["chapter"]) > 1
                or any(anchors[key] for key in ("equation", "figure", "problem", "algorithm", "explicit_symbol"))
                or re.search(r"\b(?:notebooks?|code|python)\b", query, re.I)):
            return None
        terms = [{_term_stem(term) for term in tokenize(clause)
            if term not in STOPWORDS and term not in QUERY_FRAMING_WORDS
            and term not in AUXILIARY_MANNER_WORDS and not term.isdigit()} for clause in clauses]
        if any(len(facet) < 2 for facet in terms):
            return None
        chunks = index_data["chunks_by_id"]
        eligible = [cid for cid in candidate_ids if chunks[cid].get("source_id") == "book"]
        owners = defaultdict(list)
        for cid in eligible:
            chunk = chunks[cid]
            if chunk.get("content_type") == "text":
                for ref in extract_chunk_anchors(chunk)["equation"]:
                    owners[(chunk.get("source_id"), ref)].append(cid)

        def owner_id(cid):
            chunk = chunks[cid]
            meta = chunk.get("metadata") or {}
            if chunk.get("content_type") == "equation":
                for owner in owners.get((chunk.get("source_id"), str(meta.get("equation_number"))), []):
                    if (chunks[owner].get("metadata") or {}).get("pdf_page") == meta.get("pdf_page"):
                        return owner
            return cid

        def covers(cid, facet):
            chunk = chunks[cid]
            meta = chunk.get("metadata") or {}
            actual = {_term_stem(term) for term in tokenize(str(chunk.get("content") or "")
                + " " + str(meta.get("subsection") or ""))}
            overlap = len(actual & terms[facet])
            proper = {word.casefold() for word in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", clauses[facet])
                if word.casefold() not in STOPWORDS and word.casefold() not in QUERY_FRAMING_WORDS and word.casefold() not in AUXILIARY_MANNER_WORDS}
            source_names = {word.casefold() for word in re.findall(r"[A-Za-z0-9_]+", str(chunk.get("content") or ""))}
            return overlap >= 2 and overlap / len(terms[facet]) >= .4 and proper <= source_names

        def ranked(facet, scope):
            tokens = tokenize(clauses[facet])
            contextual = self._contextual_lexical_terms(tokens, scope, index_data)
            ranking, _ = self._score_lexical(tokens, scope, contextual, {}, {})
            output, seen = [], set()
            for cid, score in ranking:
                cid = owner_id(cid)
                if cid not in seen:
                    output.append((cid, score))
                    seen.add(cid)
            return output

        pools = [ranked(facet, eligible) for facet in range(2)]
        votes = []
        for facet, pool in enumerate(pools):
            sections = {}
            for rank, (cid, _) in enumerate(pool[:20], 1):
                section = str((chunks[cid].get("metadata") or {}).get("section") or "")
                if re.fullmatch(r"[0-9A-Z]+(?:\.[0-9]+)+", section) and covers(cid, facet):
                    sections.setdefault((chunks[cid].get("source_id"), section), 1 / rank)
            votes.append(sections)
        shared = set(votes[0]) & set(votes[1])
        if shared:
            section = min(shared, key=lambda key: (-sum(vote[key] for vote in votes), key))
            scope = [cid for cid in eligible if (chunks[cid].get("source_id"),
                str((chunks[cid].get("metadata") or {}).get("section"))) == section]
            pools = [ranked(facet, scope) for facet in range(2)]
        return [[item for item in pool if covers(item[0], facet)] for facet, pool in enumerate(pools)]

    def _prioritize_requested_evidence(self, query, ranking, chunks, anchors, modality):
        """Respect explicit source/representation intent before rank fusion ties.

        RRF can otherwise bury the actual numbered equation under chunks merely
        mentioning its number, or fill a conceptual answer with notebook imports.
        This uses query syntax and source metadata, never benchmark identities.
        """
        q = query.casefold()
        wants_code = (modality == "code"
            or bool(re.search(r"\b(?:notebooks?|code|python|implement\w*)\b", q))
            or bool(anchors.get("symbol") and _has_code_symbol_intent(query, modality)))
        wants_figure = bool(re.search(r"\b(?:figure|diagram|illustrat\w*)\b", q))
        wants_equation = _wants_formula(q)
        data_definition = wants_code and not anchors.get("explicit_symbol") and bool(re.search(
            r"\b(?:introduced|initiali[sz]\w*|loaded|transposed|defined|created|constructed)\b", q))
        drawing = (wants_code and bool(re.search(
            r"\b(?:draw|drawn|drawing|plots?|plotted|plotting|visuali[sz]\w*)\b", q))
            and not re.search(r"\b(?:described|explained|instructions|documentation)\b", q))
        def priority(item):
            chunk = chunks[item[0]]
            meta = chunk.get("metadata") or {}
            exact = any(str(meta.get(key + "_number", "")).casefold() in anchors.get(key, [])
                        for key in ("equation", "figure", "problem", "algorithm"))
            if exact:
                return 3
            if wants_code and _is_notebook(chunk):
                # A code connection to an explicit numbered item is stronger
                # than an unrelated definition of another symbol in the query.
                mentioned = extract_chunk_anchors(chunk)
                if any(set(mentioned[key]) & set(anchors.get(key, []))
                       for key in ("equation", "figure", "problem", "algorithm")):
                    return 2.5
            if wants_code and anchors.get("symbol"):
                defined = _defined_callable_names(chunk)
                if defined.intersection(anchors["symbol"]):
                    return 2
            if drawing:
                for name in _defined_callable_names(chunk):
                    words = set(tokenize(name))
                    if (words & {"draw", "plot", "visualize", "visualise"}
                            and words & set(tokenize(q)) - {"draw", "plot", "function"}):
                        return 2
            if data_definition and any(str(s).startswith("variable:") for s in meta.get("defined_symbols", [])):
                return 1.5
            ctype = chunk.get("content_type")
            if wants_code and ctype in {"code", "notebook_code"}:
                return 1
            if wants_figure and ctype == "figure":
                return 1
            if wants_equation and ctype == "equation":
                return 1
            if not (wants_code or wants_figure or wants_equation) and chunk.get("source_id") == "book" and ctype == "text":
                return 1
            return 0
        # Exact references and definitions are categorical. Representation
        # preferences are bounded so a relevant explanation/figure can beat
        # an unrelated formula and a relevant formula can beat generic prose.
        def evidence_score(indexed):
            rank, item = indexed
            tier = priority(item)
            if tier == 1 and (wants_equation or wants_figure):
                return (1, 1.0 / (self.rrf_k + rank + 1))
            return (tier if tier >= 2 else 0,
                1.0 / (self.rrf_k + rank + 1) + (0.0005 * tier if tier < 2 else 0))
        ranked = [item for _, item in sorted(enumerate(ranking), key=evidence_score, reverse=True)]
        if (not any(anchors.get(key) for key in ("equation", "figure", "problem", "algorithm", "explicit_symbol"))
                and not re.search(r"\b(?:summary|summarize|notes|history|bibliography|references?|contents)\b", q)):
            def background(item):
                chunk = chunks[item[0]]
                meta = chunk.get("metadata") or {}
                text = str(chunk.get("content") or "")
                contents = not meta.get("chapter") and bool(re.search(r"\.(?:\s*\.){4}", text))
                return (contents or str(meta.get("section") or "").casefold() in {"notes", "references", "bibliography"}
                        or str(meta.get("subsection") or "").casefold() == "summary")
            ranked.sort(key=background)
        return ranked

    def search(self, request: SearchRequest) -> SearchResult:
        """Execute a lexical, semantic, or hybrid search request against `SQLiteStore`."""
        requested_mode, modality, top_k = self._validate_request(request)
        raw_query = (request.query or "").strip()
        warnings: List[str] = []

        status = self.store.status()
        build_id = status.get("active_build_id") or status.get("build_id")
        if not build_id or not status.get("lexical_ready"):
            if requested_mode == "semantic":
                raise UDLException(
                    "index_empty",
                    "The SQLite corpus store has no active build for semantic search.",
                    retryable=False,
                )
            return SearchResult(
                hits=[],
                build_id=build_id,
                mode=requested_mode,
                warnings=["Corpus store has no active build."],
                abstained=True,
                reason="Corpus store has no active build.",
            )

        query_tokens = tokenize(raw_query)
        query_anchors = extract_query_anchors(raw_query)
        has_any_anchor = any(bool(v) for v in query_anchors.values())

        if not raw_query or (not query_tokens and not has_any_anchor):
            return SearchResult(
                hits=[],
                build_id=build_id,
                mode="lexical" if requested_mode == "lexical" else requested_mode,
                warnings=warnings,
                abstained=True,
                reason="Query is empty or contains no searchable terms.",
            )

        index_data = self.store._get_search_index(build_id)
        chunks_by_id: Mapping[str, Dict[str, Any]] = index_data["chunks_by_id"]
        anchors_by_chunk: Mapping[str, Dict[str, List[str]]] = index_data[
            "anchors_by_chunk"
        ]
        df_map: Mapping[str, int] = index_data.get("df") or {}
        query_anchors = _known_query_symbols(
            raw_query, query_anchors, chunks_by_id.values(), modality)

        # Apply optional scope filters (modality and chapter)
        req_chapter = normalize_chapter(request.chapter)
        candidate_ids: List[str] = []
        for cid, chunk in chunks_by_id.items():
            if not chunk_matches_modality(chunk, modality):
                continue
            if req_chapter is not None and chunk_chapter(chunk) != req_chapter:
                continue
            candidate_ids.append(cid)

        # Resolve effective mode & dense readiness BEFORE returning early on scope/anchors
        effective_mode = requested_mode
        dense_ready = bool(status.get("dense_ready")) and not bool(
            status.get("dense_stale")
        )
        vectors_by_chunk: Optional[Mapping[str, Tuple[float, ...]]] = None

        if requested_mode == "semantic":
            if status.get("dense_stale"):
                raise UDLException(
                    "dense_stale",
                    "Semantic search cannot run because the dense index is marked stale after equation review; rebuild required.",
                    retryable=False,
                )
            if not dense_ready or self.embedder is None:
                raise UDLException(
                    "dense_missing",
                    "Semantic search requires an active dense embedding index and a configured LocalEmbedder.",
                    retryable=False,
                )
            vectors_by_chunk = self.store._get_dense_vectors(build_id)
        elif requested_mode == "hybrid":
            if not dense_ready or self.embedder is None:
                effective_mode = "lexical"
                if status.get("dense_stale"):
                    warnings.append(
                        "Dense embeddings are stale after equation review; falling back to lexical BM25 retrieval."
                    )
                elif not status.get("dense_ready"):
                    warnings.append(
                        "Dense embeddings are not built for the active corpus; falling back to lexical BM25 retrieval."
                    )
                else:
                    warnings.append(
                        "No query embedder configured; falling back to lexical BM25 retrieval."
                    )
            else:
                try:
                    vectors_by_chunk = self.store._get_dense_vectors(build_id)
                except UDLException as exc:
                    effective_mode = "lexical"
                    warnings.append(
                        f"Dense index unavailable ({exc.message}); falling back to lexical BM25 retrieval."
                    )

        if not candidate_ids:
            return SearchResult(
                hits=[],
                build_id=build_id,
                mode=effective_mode,
                warnings=warnings,
                abstained=True,
                reason="No indexed chunks match the requested scope filters.",
            )

        # Check for explicitly unindexed external sources (answer booklet, errata, slides, blog, etc.)
        unsupported_ext_reason = self._check_unsupported_external_query(
            raw_query, candidate_ids, chunks_by_id
        )
        if unsupported_ext_reason is None:
            unsupported_ext_reason = self._check_named_requirements(raw_query, candidate_ids, chunks_by_id)
        if unsupported_ext_reason is not None:
            return SearchResult(
                hits=[],
                build_id=build_id,
                mode=effective_mode,
                warnings=warnings,
                abstained=True,
                reason=unsupported_ext_reason,
            )

        # Verify exact anchors (nonexistent exact reference must abstain)
        _, _, anchor_abstain_reason = (
            self._check_anchor_matches(
                raw_query,
                modality,
                query_anchors,
                candidate_ids,
                anchors_by_chunk,
                chunks_by_id,
            )
        )
        if anchor_abstain_reason is not None:
            return SearchResult(
                hits=[],
                build_id=build_id,
                mode=effective_mode,
                warnings=warnings,
                abstained=True,
                reason=anchor_abstain_reason,
            )

        plans = self._source_subqueries(raw_query, modality, chunks_by_id)
        if not plans:
            # An explicit notebook or book request should not compete with the
            # other source merely because their vocabulary is similar.
            source = "all"
            if re.search(r"\b(?:notebooks?|code|python)\b", raw_query, re.I) and not re.search(
                r"\b(?:book|textbook)\b", raw_query, re.I):
                source = "notebook"
            elif re.search(r"\b(?:book|textbook)\b", raw_query, re.I):
                source = "book"
            chapters = query_anchors["chapter"] or sorted({
                ref.split(".")[0] for ref in query_anchors["notebook"]})
            plans = [(source, raw_query, chapters, query_anchors["notebook"])]
        elif top_k < len(plans):
            warnings.append("top_k is too small to include each requested source; use top_k >= 2 for both book and notebook evidence.")

        # Score each facet using its own query, anchors, source and chapter.
        # This also keeps abstention and reported cosine/BM25 scores local to
        # the question that the evidence is meant to answer.
        while True:
            pools = []
            required_pools = []
            bm25_map, semantic_map, query_by_chunk = {}, {}, {}
            try:
                for source, subquery, chapters, notebook_refs in plans:
                    scoped_ids = [cid for cid in candidate_ids
                        if (source == "all" or (source == "book" and chunks_by_id[cid].get("source_id") == "book")
                            or (source == "notebook" and _is_notebook(chunks_by_id[cid])))
                        and (not chapters or chunk_chapter(chunks_by_id[cid]) in chapters)
                        and (source != "notebook" or not notebook_refs or any(
                            "/" + ref.replace(".", "_") + "_" in str(chunks_by_id[cid].get("source_id") or "")
                            for ref in notebook_refs))]
                    local_anchors = _known_query_symbols(subquery, extract_query_anchors(subquery),
                        (chunks_by_id[cid] for cid in candidate_ids), modality)
                    local_items, local_chapters, local_error = self._check_anchor_matches(
                        subquery, modality, local_anchors, scoped_ids, anchors_by_chunk, chunks_by_id)
                    if local_error:
                        return SearchResult(hits=[], build_id=build_id, mode=effective_mode,
                            warnings=warnings, abstained=True, reason=local_error)
                    named_error = self._check_named_requirements(subquery, scoped_ids, chunks_by_id)
                    if named_error:
                        return SearchResult(hits=[], build_id=build_id, mode=effective_mode,
                            warnings=warnings, abstained=True, reason=named_error)
                    tokens = tokenize(subquery)
                    ranking, lexical, semantic, matched = self._rank_evidence(
                        subquery, tokens, scoped_ids, index_data, local_items,
                        local_chapters, effective_mode, vectors_by_chunk, top_k)
                    ranking = self._prioritize_requested_evidence(
                        subquery, ranking, chunks_by_id, local_anchors, modality)
                    ranking = self._select_source_passages(
                        subquery, ranking, chunks_by_id, scoped_ids, top_k, modality)
                    facets = self._facet_evidence_pools(subquery, scoped_ids, index_data, modality)
                    if facets is not None:
                        if any(not pool for pool in facets):
                            return SearchResult(hits=[], build_id=build_id, mode=effective_mode,
                                warnings=warnings, abstained=True,
                                reason="The indexed sources do not provide grounded evidence for every requested facet.")
                        facet_ranking = self._interleave_source_pools(facets, top_k)
                        original_scores = dict(ranking)
                        facet_ids = {cid for cid, _ in facet_ranking}
                        ranking = [(cid, original_scores.get(cid, 0.0)) for cid, _ in facet_ranking]
                        required_pools.extend(facets)
                    numbered = any(local_anchors.get(key) for key in ("figure", "equation", "problem", "algorithm"))
                    if numbered and not _has_code_symbol_intent(subquery, modality):
                        ranking = [item for item in ranking if local_items.get(item[0], 0) > 0]
                    content = [t for t in tokens if t not in STOPWORDS]
                    anchor_terms = set(tokenize(" ".join(local_anchors.get("explicit_symbol", [])
                        + local_anchors.get("symbol", []))))
                    subjects = [t for t in content if t not in QUERY_FRAMING_WORDS
                        and not t.isdigit() and t not in anchor_terms]
                    abstained, reason = self._should_abstain(
                        effective_mode=effective_mode, query_tokens=tokens,
                        content_terms=content, core_subject_terms=subjects,
                        item_boosts=local_items, top_chunk_ids=[cid for cid, _ in ranking[:top_k]],
                        bm25_map=lexical, semantic_map=semantic,
                        matched_terms_by_chunk=matched, df_map=df_map,
                        anchor_locator=bool(re.match(r"\s*(?:where|locate|find)\b", subquery, re.I)
                            and re.search(r"\b(?:define[ds]?|implemented|located)\b", subquery, re.I)
                            and not re.search(r"\b(?:that|which)\b|[;?].+|\band\s+(?:what|how|why|give|report|measure|show)\b", subquery, re.I)))
                    if abstained:
                        prefix = f"Insufficient {source} evidence: " if len(plans) > 1 else ""
                        return SearchResult(hits=[], build_id=build_id, mode=effective_mode,
                            warnings=warnings, abstained=True, reason=prefix + str(reason))
                    pools.append(ranking[:top_k])
                    if facets is None:
                        required_pools.append(ranking[:top_k])
                    bm25_map.update(lexical)
                    semantic_map.update(semantic)
                    query_by_chunk.update({cid: subquery for cid, _ in ranking[:top_k]})
                break
            except UDLException as exc:
                if requested_mode == "semantic" or effective_mode == "lexical":
                    raise
                effective_mode = "lexical"
                warnings.append(f"Dense query encoding unavailable ({exc.message}); falling back to lexical BM25 retrieval.")

        final_ranking = self._interleave_source_pools(pools, top_k)

        hits: List[Dict[str, Any]] = []
        for rank, (cid, ret_score) in enumerate(final_ranking, start=1):
            base_chunk = dict(chunks_by_id[cid])
            loc = locator(base_chunk)
            bm25_val = float(bm25_map.get(cid, 0.0))
            sem_val = semantic_map.get(cid)
            hit = {
                **base_chunk,
                "citation": f"S{rank}",
                "locator": loc,
                "source_locator": loc,
                "bm25_score": bm25_val,
                "semantic_score": float(sem_val) if sem_val is not None else None,
                "retrieval_score": float(ret_score),
                "retrieval_query": query_by_chunk[cid],
                "build_id": build_id,
            }
            hits.append(hit)

        final_ids = {cid for cid, _ in final_ranking}
        groups = [[cid for cid, _ in pool if cid in final_ids] for pool in required_pools] if len(required_pools) > 1 else []
        return SearchResult(
            hits=hits,
            build_id=build_id,
            mode=effective_mode,
            warnings=warnings,
            abstained=False,
            reason=None,
            required_evidence_groups=groups,
        )


def get_chunk_neighbors(
    store: Any,
    chunk_id: str,
    *,
    window: int = 1,
    build_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return previous and next chunks in the same source for context expansion."""
    if window <= 0:
        return {"previous": [], "next": []}
    target = store.get_chunk(chunk_id, build_id=build_id)
    if not target:
        return {"previous": [], "next": []}

    source_id = target.get("source_id")
    all_chunks = store.list_chunks(build_id=build_id)
    by_id = {c["chunk_id"]: c for c in all_chunks}

    prev_list: List[Dict[str, Any]] = []
    next_list: List[Dict[str, Any]] = []

    curr = target
    for _ in range(window):
        prev_id = (curr.get("metadata") or {}).get("previous_id")
        if prev_id and prev_id in by_id:
            prev_chunk = by_id[prev_id]
            if prev_chunk.get("source_id") == source_id:
                prev_list.insert(0, prev_chunk)
                curr = prev_chunk
                continue
        break

    curr = target
    for _ in range(window):
        next_id = (curr.get("metadata") or {}).get("next_id")
        if next_id and next_id in by_id:
            next_chunk = by_id[next_id]
            if next_chunk.get("source_id") == source_id:
                next_list.append(next_chunk)
                curr = next_chunk
                continue
        break

    if prev_list or next_list:
        return {"previous": prev_list, "next": next_list}

    same_source = [c for c in all_chunks if c.get("source_id") == source_id]
    try:
        idx = next(
            i for i, c in enumerate(same_source) if c["chunk_id"] == chunk_id
        )
    except StopIteration:
        return {"previous": [], "next": []}

    start_idx = max(0, idx - window)
    end_idx = min(len(same_source), idx + window + 1)
    return {
        "previous": same_source[start_idx:idx],
        "next": same_source[idx + 1 : end_idx],
    }


def expand_neighbors(
    store: Any,
    hits: Sequence[Mapping[str, Any]],
    *,
    window: int = 1,
    same_modality_only: bool = False,
    max_total: Optional[int] = None,
    build_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Expand search hits with adjacent chunks from the same source without duplicates."""
    if not hits or window <= 0:
        return [dict(h) for h in hits]

    expanded: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for hit in hits:
        cid = str(hit.get("chunk_id") or "")
        if not cid:
            continue
        effective_build = build_id or hit.get("build_id")
        neighbors = get_chunk_neighbors(
            store, cid, window=window, build_id=effective_build
        )
        hit_ctype = hit.get("content_type")

        for prev_chunk in neighbors["previous"]:
            pcid = prev_chunk["chunk_id"]
            if pcid in seen:
                continue
            if same_modality_only and prev_chunk.get("content_type") != hit_ctype:
                continue
            seen.add(pcid)
            expanded.append(
                {
                    **prev_chunk,
                    "locator": locator(prev_chunk),
                    "is_neighbor": True,
                    "neighbor_of": cid,
                    "build_id": effective_build,
                }
            )
            if max_total is not None and len(expanded) >= max_total:
                return expanded

        if cid not in seen:
            seen.add(cid)
            expanded.append(dict(hit))
            if max_total is not None and len(expanded) >= max_total:
                return expanded

        for next_chunk in neighbors["next"]:
            ncid = next_chunk["chunk_id"]
            if ncid in seen:
                continue
            if same_modality_only and next_chunk.get("content_type") != hit_ctype:
                continue
            seen.add(ncid)
            expanded.append(
                {
                    **next_chunk,
                    "locator": locator(next_chunk),
                    "is_neighbor": True,
                    "neighbor_of": cid,
                    "build_id": effective_build,
                }
            )
            if max_total is not None and len(expanded) >= max_total:
                return expanded

    return expanded


def build_context_block(
    hits: Iterable[Mapping[str, Any]], *, max_characters: int = 18000
) -> str:
    """Format retrieved hits into a cited context block (`[S1] ...`) for prompting."""
    parts: List[str] = []
    remaining = int(max_characters)
    for idx, hit in enumerate(hits, start=1):
        citation = hit.get("citation") or f"S{idx}"
        loc = hit.get("locator") or locator(hit)
        header = f"[{citation}] (Source: {loc})\n"
        available = remaining - len(header) - 2
        if available <= 0:
            break
        content = str(hit.get("content") or "")[:available]
        block = f"{header}{content}"
        parts.append(block)
        remaining -= len(block) + 2
    return "\n\n".join(parts)
