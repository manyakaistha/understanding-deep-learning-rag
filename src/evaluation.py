"""Fixed, provenance-aware retrieval evaluation independently grounded in original UDL sources."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.contracts import PROJECT_ROOT, SearchRequest


def normalize_text(text: str) -> str:
    """Normalize text for whitespace and dehyphenation comparisons."""
    if not text:
        return ""
    # Remove hyphenation at line breaks: e.g. 'differ-\n ential' -> 'differential'
    dehyphenated = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', str(text))
    # Collapse all whitespace and newlines into single spaces and casefold
    return ' '.join(dehyphenated.split()).casefold()


def is_valid_gold_target(expected: Any) -> bool:
    """Validate that a gold target is not a vague chapter-only placeholder.
    
    A supported gold target must specify:
    - For book: pdf_page (or pdf_pages) AND at least one specific anchor
      (quotes, figure_number, equation_number, problem_number, or source_spans).
    - For notebook: cell_index (or cell_indices) AND at least one specific anchor
      (quotes, symbol, symbol_names, or source_spans).
    - For groups with alternatives: a non-empty list of valid gold targets (or an
      'alternatives' list of valid targets within a target dict).
    Chapter-only labels without page or cell locators are explicitly rejected.
    """
    if isinstance(expected, list):
        if not expected:
            return False
        return all(is_valid_gold_target(item) for item in expected)

    if not isinstance(expected, dict) or not expected:
        return False

    if "alternatives" in expected:
        alts = expected["alternatives"]
        if not isinstance(alts, list) or not alts:
            return False
        if not all(is_valid_gold_target(alt) for alt in alts):
            return False
        if "source_id" in expected:
            base = {k: v for k, v in expected.items() if k != "alternatives"}
            if not is_valid_gold_target(base):
                return False
        return True

    source_id = str(expected.get("source_id") or "").strip()
    if not source_id:
        return False

    if source_id == "book":
        has_page = (
            expected.get("pdf_page") is not None
            or bool(expected.get("pdf_pages"))
            or (
                isinstance(expected.get("source_spans"), list)
                and any(isinstance(s, dict) and "pdf_page" in s for s in expected["source_spans"])
            )
        )
        if not has_page:
            return False
        has_anchor = bool(
            expected.get("quotes")
            or expected.get("figure_number")
            or expected.get("equation_number")
            or expected.get("problem_number")
            or expected.get("source_spans")
        )
        return has_anchor

    if source_id.startswith("Notebooks/") or source_id.endswith(".ipynb"):
        has_cell = (
            expected.get("cell_index") is not None
            or bool(expected.get("cell_indices"))
            or (
                isinstance(expected.get("source_spans"), list)
                and any(isinstance(s, dict) and "cell_index" in s for s in expected["source_spans"])
            )
        )
        if not has_cell:
            return False
        has_anchor = bool(
            expected.get("quotes")
            or expected.get("symbol")
            or expected.get("symbol_names")
            or expected.get("source_spans")
        )
        return has_anchor

    return False


def _matches(hit: Dict[str, Any], expected: Any) -> bool:
    """Check if a single search hit satisfies the expected gold criteria."""
    if not is_valid_gold_target(expected):
        return False

    if isinstance(expected, list):
        return any(_matches(hit, alt) for alt in expected)

    if isinstance(expected, dict) and "alternatives" in expected:
        candidates = list(expected["alternatives"])
        if "source_id" in expected:
            base = {k: v for k, v in expected.items() if k != "alternatives"}
            candidates.insert(0, base)
        return any(_matches(hit, cand) for cand in candidates)

    meta = hit.get("metadata") or {}
    source = str(hit.get("source_id") or "").strip()
    exp_source = str(expected.get("source_id") or "").strip()
    if exp_source and source != exp_source:
        return False

    # Chapter check: if expected specifies a chapter, hit must match if chapter is present
    if "chapter" in expected:
        exp_ch = str(expected["chapter"]).strip().lstrip("0")
        hit_ch = str(meta.get("chapter") or hit.get("chapter") or "").strip().lstrip("0")
        if hit_ch and hit_ch != exp_ch:
            return False

    # Page check for book targets (supporting single pages, page sets, and cross-page spans)
    if "pdf_page" in expected or "pdf_pages" in expected:
        exp_pages: Set[int] = set()
        if expected.get("pdf_page") is not None:
            p = expected["pdf_page"]
            if isinstance(p, (list, tuple, set)):
                exp_pages.update(int(x) for x in p)
            else:
                exp_pages.add(int(p))
        if expected.get("pdf_pages"):
            exp_pages.update(int(x) for x in expected["pdf_pages"])

        hit_pages: Set[int] = set()
        if meta.get("pdf_page") is not None:
            hp = meta["pdf_page"]
            if isinstance(hp, (list, tuple, set)):
                hit_pages.update(int(x) for x in hp)
            else:
                hit_pages.add(int(hp))
        if meta.get("pdf_pages"):
            hit_pages.update(int(x) for x in meta["pdf_pages"])
        if isinstance(meta.get("pdf_page"), int) and isinstance(meta.get("end_pdf_page"), int):
            start, end = meta["pdf_page"], meta["end_pdf_page"]
            if start <= end <= start + 3:
                hit_pages.update(range(start, end + 1))
        for span in meta.get("source_spans") or []:
            if isinstance(span, dict) and span.get("pdf_page") is not None:
                hit_pages.add(int(span["pdf_page"]))

        if not hit_pages or not (hit_pages & exp_pages):
            return False

    # Cell index check for notebook targets
    if "cell_index" in expected or "cell_indices" in expected:
        exp_cells: Set[int] = set()
        if expected.get("cell_index") is not None:
            c = expected["cell_index"]
            if isinstance(c, (list, tuple, set)):
                exp_cells.update(int(x) for x in c)
            else:
                exp_cells.add(int(c))
        if expected.get("cell_indices"):
            exp_cells.update(int(x) for x in expected["cell_indices"])

        hit_cells: Set[int] = set()
        if meta.get("cell_index") is not None:
            hc = meta["cell_index"]
            if isinstance(hc, (list, tuple, set)):
                hit_cells.update(int(x) for x in hc)
            else:
                hit_cells.add(int(hc))
        for span in meta.get("source_spans") or []:
            if isinstance(span, dict) and span.get("cell_index") is not None:
                hit_cells.add(int(span["cell_index"]))

        if not hit_cells or not (hit_cells & exp_cells):
            return False

    # Specific anchor number checks
    if "figure_number" in expected:
        exp_fig = str(expected["figure_number"]).strip()
        hit_fig = str(meta.get("figure_number") or "").strip()
        if hit_fig:
            if hit_fig != exp_fig:
                return False
        else:
            raw_c = str(hit.get("raw_content") or hit.get("content") or "")
            if f"figure {exp_fig}".casefold() not in normalize_text(raw_c):
                return False

    if "equation_number" in expected:
        exp_eq = str(expected["equation_number"]).strip()
        hit_eq = str(meta.get("equation_number") or "").strip()
        if hit_eq:
            if hit_eq != exp_eq:
                return False
        else:
            raw_c = str(hit.get("raw_content") or hit.get("content") or "")
            norm_c = normalize_text(raw_c)
            if f"({exp_eq})".casefold() not in norm_c and f"equation {exp_eq}".casefold() not in norm_c:
                return False

    if "problem_number" in expected:
        exp_pr = str(expected["problem_number"]).strip()
        hit_pr = str(meta.get("problem_number") or "").strip()
        if hit_pr:
            if hit_pr != exp_pr:
                return False
        else:
            raw_c = str(hit.get("raw_content") or hit.get("content") or "")
            if f"problem {exp_pr}".casefold() not in normalize_text(raw_c):
                return False

    if "symbol" in expected or "symbol_names" in expected:
        exp_syms: Set[str] = set()
        if expected.get("symbol"):
            exp_syms.add(str(expected["symbol"]).strip())
        if expected.get("symbol_names"):
            exp_syms.update(str(s).strip() for s in expected["symbol_names"])
        hit_syms = set(meta.get("symbol_names") or [])
        raw_c = str(hit.get("raw_content") or hit.get("content") or "")
        if not (exp_syms & hit_syms) and not any(s in raw_c for s in exp_syms):
            return False

    # Span matching: a hit must match ALL keys in at least one expected span!
    spans = expected.get("source_spans")
    if spans:
        candidates = [meta]
        if isinstance(meta.get("source_spans"), list):
            candidates.extend(s for s in meta["source_spans"] if isinstance(s, dict))
        span_matched = False
        for exp_span in spans:
            if not isinstance(exp_span, dict) or not exp_span:
                continue
            for cand in candidates:
                if all(cand.get(k) == v for k, v in exp_span.items()):
                    span_matched = True
                    break
            if span_matched:
                break
        if not span_matched:
            return False

    # Quote matching checks source text plus explicitly attached source context.
    # Equation chunks may keep only glyphs in raw_content while the parser stores
    # adjacent prose in context_before/context_after; context is source evidence,
    # unlike embedding_text which can contain synthetic labels or summaries.
    quotes = expected.get("quotes", [])
    if quotes:
        evidence_parts = [hit.get("raw_content"), hit.get("content"), meta.get("context_before"), meta.get("context_after")]
        evidence = "\n".join(str(part) for part in evidence_parts if part)
        norm_content = normalize_text(evidence)
        for q in quotes:
            norm_q = normalize_text(q)
            if norm_q not in norm_content:
                return False

    return True


def _group_matches(hits, expected):
    """Allow a source region's evidence to span several retrieved chunks.

    Every quote still requires its declared source/page/cell. An attached
    equation/figure anchor must also be retrieved. This measures evidence-set
    coverage independently of where a parser partitions a source paragraph.
    """
    alternatives = expected.get("alternatives") or []
    base = {k: v for k, v in expected.items() if k != "alternatives"}
    if alternatives:
        candidates = ([base] if base.get("source_id") else []) + alternatives
        return any(_group_matches(hits, candidate) for candidate in candidates)
    if any(_matches(hit, base) for hit in hits):
        return True
    quotes = base.get("quotes") or []
    if not quotes:
        return False
    anchor_keys = {"equation_number", "figure_number", "problem_number", "symbol", "symbol_names", "source_spans"}
    anchor = {k: v for k, v in base.items() if k != "quotes"}
    if any(key in anchor for key in anchor_keys) and not any(_matches(hit, anchor) for hit in hits):
        return False
    location = {k: v for k, v in base.items() if k not in anchor_keys and k != "quotes"}
    return all(any(_matches(hit, dict(location, quotes=[quote])) for hit in hits) for quote in quotes)


def score_case(hits: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
    """Score retrieval results for a single benchmark case across all required evidence groups."""
    groups = case.get("expected", [])
    if not groups or case.get("unsupported", False):
        return {
            "hit": False,
            "matched_groups": 0,
            "groups": len(groups),
            "first_correct_rank": None,
        }

    # Any unverified or chapter-only label invalidates the case
    for g in groups:
        if not is_valid_gold_target(g):
            return {
                "hit": False,
                "matched_groups": 0,
                "groups": len(groups),
                "first_correct_rank": None,
                "unverified_label": True,
            }

    found = [_group_matches(hits, g) for g in groups]
    all_matched = bool(groups) and all(found)

    first_rank: Optional[int] = None
    if all_matched:
        for idx, h in enumerate(hits, start=1):
            if any(_matches(h, g) for g in groups):
                first_rank = idx
                break

    return {
        "hit": all_matched,
        "matched_groups": sum(found),
        "groups": len(groups),
        "first_correct_rank": first_rank,
    }


def resolve_citation(hit: Dict[str, Any], store: Any = None, build_id: Optional[str] = None) -> bool:
    """Verify that a hit citation resolves against real store records."""
    has_label = bool(hit.get("citation") or hit.get("locator"))
    chunk_id = hit.get("chunk_id")
    source_id = hit.get("source_id")
    if not (has_label and chunk_id and source_id):
        return False
    if store is None or not hasattr(store, "get_chunk") or not hasattr(store, "get_source"):
        return False

    target_bid = build_id or hit.get("build_id")
    try:
        chunk = store.get_chunk(str(chunk_id), build_id=target_bid)
        if chunk is None:
            return False
        source = store.get_source(str(source_id), build_id=target_bid)
        if source is None:
            return False
        return chunk.get("source_id") == str(source_id)
    except Exception:
        return False


def evaluate(
    runtime: Any,
    suite_path: Path | str = PROJECT_ROOT / "resources/evaluation.json",
    mode: str = "hybrid",
    split: str = "heldout",
) -> Dict[str, Any]:
    """Execute fixed provenance benchmark evaluation, calculating Hit@5, abstention, and citation resolution."""
    raw_suite = Path(suite_path).read_text(encoding="utf-8")
    suite = json.loads(raw_suite)
    all_cases = suite.get("cases", [])
    cases = [c for c in all_cases if split == "all" or c.get("split") == split]

    status = runtime.status() if hasattr(runtime, "status") else {}
    store = getattr(runtime, "store", None)
    dense_ready = bool(status.get("dense_ready", False))
    build_id = status.get("build_id")

    resolved_config = {
        "suite_version": suite.get("version"),
        "mode": mode,
        "split": split,
        "top_k": 5,
        "dense_ready": dense_ready,
        "build_id": build_id,
    }

    rows: List[Dict[str, Any]] = []
    has_unverified_labels = False
    has_semantic_fallback = False

    for case in cases:
        unsupported = bool(case.get("unsupported", False))
        expected = case.get("expected", [])
        if not unsupported:
            for g in expected:
                if not is_valid_gold_target(g):
                    has_unverified_labels = True

        service = runtime.service(mode)
        result = service.search(SearchRequest(case["query"], top_k=5, mode=mode))
        hits = result.hits or []

        actual_mode = getattr(result, "mode", mode)
        if mode in {"hybrid", "semantic"} and actual_mode != mode:
            has_semantic_fallback = True

        s = score_case(hits, case)
        abstained = bool(getattr(result, "abstained", False))

        resolved_count = 0
        target_bid = build_id or getattr(result, "build_id", None)
        for h in hits:
            if resolve_citation(h, store=store, build_id=target_bid):
                resolved_count += 1

        s.update(
            id=case["id"],
            category=case["category"],
            unsupported=unsupported,
            abstained=abstained,
            unsupported_correct=(abstained if unsupported else None),
            citation_resolution=resolved_count,
            citations=len(hits),
            dense_used=dense_ready and actual_mode in ("hybrid", "semantic"),
        )
        rows.append(s)

    supported = [r for r in rows if not r["unsupported"]]
    categories: Dict[str, float] = {}
    for cat in sorted({r["category"] for r in supported}):
        vals = [r["hit"] for r in supported if r["category"] == cat]
        categories[cat] = sum(vals) / len(vals) if vals else 0.0

    unsupported_rows = [r for r in rows if r["unsupported"]]
    abst_vals = [r["unsupported_correct"] for r in unsupported_rows if r["unsupported_correct"] is not None]
    abstention_accuracy = (sum(abst_vals) / len(abst_vals)) if abst_vals else None

    hit_at_5 = (sum(r["hit"] for r in supported) / len(supported)) if supported else 0.0
    total_citations = sum(r["citations"] for r in rows)
    total_resolved = sum(r["citation_resolution"] for r in rows)
    citation_rate = (total_resolved / total_citations) if total_citations > 0 else (1.0 if not supported else 0.0)

    rank_values = [r["first_correct_rank"] for r in supported if r["first_correct_rank"] is not None]
    rank_mean = (sum(rank_values) / len(rank_values)) if rank_values else None

    # Gate verification:
    # 1. Non-empty test set
    # 2. No unverified/chapter-only gold labels
    # 3. Dense must be ready for hybrid/semantic; no fallback in semantic
    # 4. Overall supported Hit@5 >= 0.90
    # 5. All category Hit@5 >= 0.80
    # 6. Unsupported abstention accuracy >= 0.90
    # 7. Citation resolution rate == 100%
    dense_ok = (dense_ready and not has_semantic_fallback) if mode in ("hybrid", "semantic") else True
    passed = (
        bool(rows)
        and not has_unverified_labels
        and dense_ok
        and hit_at_5 >= 0.90
        and all(score >= 0.80 for score in categories.values())
        and (abstention_accuracy is not None and abstention_accuracy >= 0.90)
        and (citation_rate == 1.0 and total_citations > 0)
    )

    failures = [
        r["id"]
        for r in rows
        if (r["unsupported"] and not r["unsupported_correct"])
        or (not r["unsupported"] and not r["hit"])
    ]

    return {
        "suite_version": suite.get("version"),
        "mode": mode,
        "split": split,
        "build_id": build_id,
        "resolved_config": resolved_config,
        "dense_ready": dense_ready,
        "case_count": len(rows),
        "supported_hit_at_5": hit_at_5,
        "category_hit_at_5": categories,
        "unsupported_abstention_accuracy": abstention_accuracy,
        "first_correct_rank_mean": rank_mean,
        "citation_resolution_rate": citation_rate,
        "passed": passed,
        "failures": failures,
        "cases": rows,
    }
