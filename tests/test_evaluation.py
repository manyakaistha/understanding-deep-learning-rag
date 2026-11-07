"""Unit tests for retrieval evaluation, matching semantics, citation resolution, and source gold validation."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import fitz

from src.contracts import PROJECT_ROOT
from src.evaluation import (
    _matches,
    is_valid_gold_target,
    normalize_text,
    resolve_citation,
    score_case,
)


class MockStore:
    def __init__(self, valid_chunks: Optional[Dict[str, Any]] = None, valid_sources: Optional[Dict[str, Any]] = None):
        self.chunks = valid_chunks or {}
        self.sources = valid_sources or {}

    def get_chunk(self, chunk_id: str, build_id: Optional[str] = None):
        return self.chunks.get(chunk_id)

    def get_source(self, source_id: str, build_id: Optional[str] = None):
        return self.sources.get(source_id)


class TestEvaluationMatching(unittest.TestCase):
    def test_cross_page_problem_requires_the_retrieved_continuation(self):
        case = {"expected": [{"source_id": "book", "pdf_pages": [128, 129],
                              "problem_number": "7.1",
                              "quotes": ["two hidden units", "each of the 13 parameters"]}]}
        opening = {"source_id": "book", "metadata": {"pdf_page": 128, "problem_number": "7.1"},
                   "content": "Problem 7.1: a network with two hidden units."}
        continuation = {"source_id": "book", "metadata": {"pdf_page": 129},
                        "content": "Compute derivatives for each of the 13 parameters."}
        self.assertFalse(score_case([opening], case)["hit"])
        self.assertFalse(score_case([continuation], case)["hit"])
        self.assertTrue(score_case([opening, continuation], case)["hit"])
        continuation["metadata"]["pdf_page"] = 130
        self.assertFalse(score_case([opening, continuation], case)["hit"])

    def test_pooled_quotes_still_require_the_numbered_anchor(self):
        case = {"expected": [{"source_id": "book", "pdf_page": 228,
                              "equation_number": "12.9",
                              "quotes": ["Scaled dot-product self-attention", "square root"]}]}
        prose = {"source_id": "book", "metadata": {"pdf_page": 228},
                 "content": "Scaled dot-product self-attention divides by the square root."}
        equation = {"source_id": "book", "metadata": {"pdf_page": 228, "equation_number": "12.9"},
                    "content": "V Softmax[K^T Q / sqrt(D)]"}
        self.assertFalse(score_case([prose], case)["hit"])
        self.assertTrue(score_case([prose, equation], case)["hit"])
        equation["metadata"]["equation_number"] = "12.5"
        self.assertFalse(score_case([prose, equation], case)["hit"])

    def test_source_evidence_can_span_two_retrieved_chunks_on_its_declared_page(self):
        case = {'expected': [{'source_id': 'book', 'pdf_page': 42, 'quotes': ['First support.', 'Second support.']}]}
        hits = [{'source_id': 'book', 'content': text, 'metadata': {'pdf_page': page}}
                for text, page in [('First support.', 42), ('Second support.', 42)]]
        self.assertTrue(score_case(hits, case)['hit'])
        hits[1]['metadata']['pdf_page'] = 43
        self.assertFalse(score_case(hits, case)['hit'])

    def test_equation_quote_can_match_attached_source_context(self):
        expected = {"source_id": "book", "pdf_page": 33, "equation_number": "2.5", "quotes": ["least-squares loss"]}
        hit = {"source_id": "book", "metadata": {"pdf_page": 33, "equation_number": "2.5", "context_before": "We call this the least-squares loss."}, "raw_content": "\\sum_i (f[x_i]-y_i)^2"}
        self.assertTrue(_matches(hit, expected))

    def test_numbered_equation_does_not_require_unrelated_prose_quote(self):
        expected = {"source_id": "book", "pdf_page": 33, "equation_number": "2.5"}
        hit = {"source_id": "book", "metadata": {"pdf_page": 33, "equation_number": "2.5"}, "raw_content": "\\sum_i (f[x_i]-y_i)^2"}
        self.assertTrue(_matches(hit, expected))

    def test_false_positive_wrong_chapter(self):
        expected = {
            "source_id": "book",
            "chapter": "2",
            "pdf_page": 31,
            "quotes": ["supervised learning"],
        }
        hit_wrong_ch = {
            "source_id": "book",
            "metadata": {"chapter": "1", "pdf_page": 31},
            "content": "Supervised learning mapping",
        }
        hit_correct_ch = {
            "source_id": "book",
            "metadata": {"chapter": "2", "pdf_page": 31},
            "content": "Supervised learning mapping",
        }
        self.assertFalse(_matches(hit_wrong_ch, expected))
        self.assertTrue(_matches(hit_correct_ch, expected))

    def test_false_positive_wrong_page(self):
        expected = {
            "source_id": "book",
            "pdf_page": 31,
            "quotes": ["supervised learning"],
        }
        hit_wrong_page = {
            "source_id": "book",
            "metadata": {"pdf_page": 32},
            "content": "Supervised learning mapping",
        }
        hit_correct_page = {
            "source_id": "book",
            "metadata": {"pdf_page": 31},
            "content": "Supervised learning mapping",
        }
        self.assertFalse(_matches(hit_wrong_page, expected))
        self.assertTrue(_matches(hit_correct_page, expected))

    def test_false_positive_wrong_source(self):
        expected = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "cell_index": 4,
            "quotes": ["def f("],
        }
        hit_wrong_src = {
            "source_id": "book",
            "metadata": {"cell_index": 4},
            "content": "def f(x, phi0, phi1):",
        }
        hit_correct_src = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "metadata": {"cell_index": 4},
            "content": "def f(x, phi0, phi1):",
        }
        self.assertFalse(_matches(hit_wrong_src, expected))
        self.assertTrue(_matches(hit_correct_src, expected))

    def test_false_positive_wrong_cell(self):
        expected = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "cell_index": 4,
            "quotes": ["def f("],
        }
        hit_wrong_cell = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "metadata": {"cell_index": 7},
            "content": "def f(x, phi0, phi1):",
        }
        hit_correct_cell = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "metadata": {"cell_index": 4},
            "content": "def f(x, phi0, phi1):",
        }
        self.assertFalse(_matches(hit_wrong_cell, expected))
        self.assertTrue(_matches(hit_correct_cell, expected))

    def test_span_matching_requires_all_keys(self):
        expected = {
            "source_id": "book",
            "pdf_page": 32,
            "quotes": ["regression"],
            "source_spans": [{"pdf_page": 32, "section": "2.2"}],
        }
        # Hit only matches pdf_page, but section differs ("2.1" vs "2.2")
        partial_span_hit = {
            "source_id": "book",
            "metadata": {"pdf_page": 32, "section": "2.1"},
            "content": "linear regression model",
        }
        full_span_hit = {
            "source_id": "book",
            "metadata": {"pdf_page": 32, "section": "2.2"},
            "content": "linear regression model",
        }
        self.assertFalse(_matches(partial_span_hit, expected))
        self.assertTrue(_matches(full_span_hit, expected))

    def test_multi_evidence_requires_all_groups(self):
        case = {
            "id": "test-mixed",
            "category": "mixed",
            "expected": [
                {"source_id": "book", "pdf_page": 33, "quotes": ["least-squares loss"]},
                {"source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb", "cell_index": 7, "quotes": ["compute_loss"]},
            ],
        }
        book_hit = {
            "source_id": "book",
            "metadata": {"pdf_page": 33},
            "content": "we call this a least-squares loss",
        }
        nb_hit = {
            "source_id": "Notebooks/Chap02/2_1_Supervised_Learning.ipynb",
            "metadata": {"cell_index": 7},
            "content": "def compute_loss(x,y,phi0,phi1):",
        }

        # Only book evidence -> fail
        res_book_only = score_case([book_hit], case)
        self.assertFalse(res_book_only["hit"])
        self.assertEqual(res_book_only["matched_groups"], 1)

        # Only notebook evidence -> fail
        res_nb_only = score_case([nb_hit], case)
        self.assertFalse(res_nb_only["hit"])
        self.assertEqual(res_nb_only["matched_groups"], 1)

        # Both evidence items present -> pass
        res_both = score_case([book_hit, nb_hit], case)
        self.assertTrue(res_both["hit"])
        self.assertEqual(res_both["matched_groups"], 2)

    def test_alternatives_within_group_or_matching(self):
        case = {
            "id": "test-alt",
            "category": "theory",
            "expected": [
                {
                    "source_id": "book",
                    "chapter": "4",
                    "pdf_page": 59,
                    "quotes": ["two-layer network"],
                    "alternatives": [
                        {
                            "source_id": "book",
                            "chapter": "4",
                            "pdf_page": 57,
                            "quotes": ["composing networks to deep networks"],
                        }
                    ],
                }
            ],
        }
        hit_primary = {
            "source_id": "book",
            "metadata": {"chapter": "4", "pdf_page": 59},
            "content": "yielding a two-layer network example",
        }
        hit_alt = {
            "source_id": "book",
            "metadata": {"chapter": "4", "pdf_page": 57},
            "content": "from composing networks to deep networks",
        }
        hit_wrong = {
            "source_id": "book",
            "metadata": {"chapter": "4", "pdf_page": 58},
            "content": "unrelated deep network diagram",
        }

        # Match primary -> pass
        score_prim = score_case([hit_primary], case)
        self.assertTrue(score_prim["hit"])
        self.assertEqual(score_prim["first_correct_rank"], 1)

        # Match alternative -> pass
        score_alt = score_case([hit_alt], case)
        self.assertTrue(score_alt["hit"])
        self.assertEqual(score_alt["first_correct_rank"], 1)

        # Match neither -> fail
        score_wrong = score_case([hit_wrong], case)
        self.assertFalse(score_wrong["hit"])

    def test_unverified_labels_rejected(self):
        # Chapter-only label with no pdf_page or specific locator
        chapter_only_book = {"source_id": "book", "chapter": "1"}
        self.assertFalse(is_valid_gold_target(chapter_only_book))

        case_with_vague_label = {
            "id": "c-vague",
            "expected": [chapter_only_book],
        }
        hit = {
            "source_id": "book",
            "metadata": {"chapter": "1", "pdf_page": 15},
            "content": "Introduction to AI",
        }
        score = score_case([hit], case_with_vague_label)
        self.assertFalse(score["hit"])
        self.assertTrue(score.get("unverified_label"))

    def test_unverified_alternative_invalidates_case(self):
        case_with_vague_alt = {
            "id": "c-vague-alt",
            "expected": [
                {
                    "source_id": "book",
                    "chapter": "1",
                    "pdf_page": 15,
                    "quotes": ["AI"],
                    "alternatives": [
                        {"source_id": "book", "chapter": "1"}  # vague chapter-only alternative
                    ],
                }
            ],
        }
        hit = {
            "source_id": "book",
            "metadata": {"chapter": "1", "pdf_page": 15},
            "content": "AI is artificial intelligence",
        }
        score = score_case([hit], case_with_vague_alt)
        self.assertFalse(score["hit"])
        self.assertTrue(score.get("unverified_label"))

    def test_empty_set_fails(self):
        case = {
            "expected": [{"source_id": "book", "pdf_page": 15, "quotes": ["AI"]}],
        }
        self.assertFalse(score_case([], case)["hit"])
        self.assertFalse(score_case([], {"expected": []})["hit"])

    def test_fake_citation_unresolvable(self):
        store = MockStore(
            valid_chunks={"real_chk": {"chunk_id": "real_chk", "source_id": "book"}},
            valid_sources={"book": {"source_id": "book"}},
        )
        real_hit = {
            "chunk_id": "real_chk",
            "source_id": "book",
            "citation": "UDL p. 15",
        }
        fake_hit = {
            "chunk_id": "fake_chk_999",
            "source_id": "book",
            "citation": "UDL p. 15",
        }
        self.assertTrue(resolve_citation(real_hit, store=store))
        self.assertFalse(resolve_citation(fake_hit, store=store))


class TestGoldBenchmarkFidelity(unittest.TestCase):
    """Directly verifies all 100 cases in resources/evaluation.json against original UDL sources."""

    @classmethod
    def setUpClass(cls):
        suite_path = PROJECT_ROOT / "resources/evaluation.json"
        with open(suite_path, "r", encoding="utf-8") as f:
            cls.suite = json.load(f)
        cls.cases = cls.suite["cases"]
        pdf_path = PROJECT_ROOT / "data/raw/UnderstandingDeepLearning.pdf"
        if not pdf_path.is_file():
            raise unittest.SkipTest("Source-fidelity audit requires downloaded pinned book and notebooks")
        cls.pdf_doc = fitz.open(pdf_path)
        cls.nb_root = PROJECT_ROOT / "data/raw/udlbook_notebooks"

    @classmethod
    def tearDownClass(cls):
        cls.pdf_doc.close()

    def test_case_count_and_splits(self):
        self.assertEqual(len(self.cases), 100)
        dev_cases = [c for c in self.cases if c["split"] == "development"]
        heldout_cases = [c for c in self.cases if c["split"] == "heldout"]
        self.assertEqual(len(dev_cases), 60)
        self.assertEqual(len(heldout_cases), 40)

        # Check unsupported negatives exist in both splits
        unsupp_dev = [c for c in dev_cases if c.get("unsupported")]
        unsupp_heldout = [c for c in heldout_cases if c.get("unsupported")]
        self.assertEqual(len(unsupp_dev), 6)
        self.assertEqual(len(unsupp_heldout), 4)

    def test_category_distributions(self):
        counts = {}
        for c in self.cases:
            cat = c["category"]
            counts[cat] = counts.get(cat, 0) + 1
        expected_counts = {
            "theory": 30,
            "equation": 10,
            "figure": 10,
            "problem": 10,
            "notebook_code": 20,
            "mixed": 10,
            "unsupported": 10,
        }
        self.assertEqual(counts, expected_counts)

    def test_all_21_chapters_covered_in_theory(self):
        theory_cases = [c for c in self.cases if c["category"] == "theory"]
        chapters_covered = set()
        for c in theory_cases:
            for g in c["expected"]:
                targets = []
                if isinstance(g, dict) and "alternatives" in g:
                    targets.extend(g["alternatives"])
                    if "source_id" in g:
                        targets.append(g)
                elif isinstance(g, list):
                    targets.extend(g)
                else:
                    targets.append(g)
                for target in targets:
                    if target.get("source_id") == "book" and "chapter" in target:
                        chapters_covered.add(str(target["chapter"]))
        for ch_num in range(1, 22):
            self.assertIn(str(ch_num), chapters_covered, f"Chapter {ch_num} missing from theory coverage")

    def test_all_gold_labels_grounded_in_original_files(self):
        errors = []
        for c in self.cases:
            cid = c["id"]
            if c.get("unsupported"):
                self.assertEqual(c.get("expected", []), [], f"{cid}: unsupported case must have empty expected")
                continue

            for g_idx, g in self.TestCase_validate_group(c, cid, errors):
                pass
        self.assertEqual(errors, [], f"Encountered ground-truth label validation errors: {errors}")

    def TestCase_validate_group(self, c, cid, errors):
        for g_idx, group in enumerate(c.get("expected", [])):
            targets = []
            if isinstance(group, dict) and "alternatives" in group:
                if "source_id" in group:
                    targets.append({k: v for k, v in group.items() if k != "alternatives"})
                targets.extend(group["alternatives"])
            elif isinstance(group, list):
                targets.extend(group)
            elif isinstance(group, dict):
                targets.append(group)
            else:
                errors.append(f"{cid} group {g_idx}: invalid group structure {type(group)}")
                continue

            for t_idx, target in enumerate(targets):
                sid = target.get("source_id")
                if not sid:
                    errors.append(f"{cid} group {g_idx} target {t_idx}: missing source_id")
                    continue
                if sid == "book":
                    pages = list(target.get("pdf_pages") or [])
                    if target.get("pdf_page") is not None:
                        pages.append(target["pdf_page"])
                    if not pages or any(not isinstance(p, int) or not (1 <= p <= len(self.pdf_doc)) for p in pages):
                        errors.append(f"{cid} group {g_idx} target {t_idx}: invalid pdf_pages {pages}")
                        continue
                    txt = normalize_text("\n".join(self.pdf_doc[p - 1].get_text("text") for p in pages))
                    for q in target.get("quotes", []):
                        if normalize_text(q) not in txt:
                            errors.append(f"{cid} group {g_idx} target {t_idx}: quote '{q}' not on pages {pages}")
                    if "figure_number" in target:
                        fnum = str(target["figure_number"]).strip()
                        if f"figure {fnum}".casefold() not in txt:
                            errors.append(f"{cid} group {g_idx} target {t_idx}: figure {fnum} not on pages {pages}")
                    if "equation_number" in target:
                        eqnum = str(target["equation_number"]).strip()
                        if f"({eqnum})".casefold() not in txt and f"equation {eqnum}".casefold() not in txt:
                            errors.append(f"{cid} group {g_idx} target {t_idx}: equation {eqnum} not on pages {pages}")
                    if "problem_number" in target:
                        pnum = str(target["problem_number"]).strip()
                        if f"problem {pnum}".casefold() not in txt:
                            errors.append(f"{cid} group {g_idx} target {t_idx}: problem {pnum} not on pages {pages}")
                elif sid.startswith("Notebooks/"):
                    nb_path = self.nb_root / sid
                    if not nb_path.exists():
                        errors.append(f"{cid} group {g_idx} target {t_idx}: notebook {sid} not found")
                        continue
                    with open(nb_path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                    cells = nb.get("cells", [])
                    c_idx = target.get("cell_index")
                    if c_idx is None or not (0 <= c_idx < len(cells)):
                        errors.append(f"{cid} group {g_idx} target {t_idx}: invalid cell_index {c_idx}")
                        continue
                    cell_src = "".join(cells[c_idx].get("source", []))
                    for q in target.get("quotes", []):
                        if normalize_text(q) not in normalize_text(cell_src):
                            errors.append(f"{cid} group {g_idx} target {t_idx}: quote '{q}' not in cell {c_idx} of {sid}")
                    if "symbol" in target:
                        sym = target["symbol"]
                        if sym not in cell_src:
                            errors.append(f"{cid} group {g_idx} target {t_idx}: symbol '{sym}' not in cell {c_idx} of {sid}")
        return []


if __name__ == "__main__":
    unittest.main()
