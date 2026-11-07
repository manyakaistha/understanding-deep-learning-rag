"""Comprehensive extraction-only tests for `src/corpus.py`, `resources/book_layout.json`, and `src/parsers/*`."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from PIL import Image

from src.contracts import (
    BOOK_SHA256,
    NOTEBOOK_COMMIT,
    PROJECT_ROOT,
    UDLException,
)
from src.corpus import (
    _find_chapter_notebooks,
    _partition_code_cell_intervals,
    ensure_sources,
    extract_corpus,
)


class CorpusExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (PROJECT_ROOT / "data/raw/UnderstandingDeepLearning.pdf").is_file():
            raise unittest.SkipTest("Real-source extraction tests require: python -m src.cli build")
        cls.ensure_info = ensure_sources(PROJECT_ROOT, offline=True)
        cls.corpus = extract_corpus(PROJECT_ROOT, render_assets=True)
        cls.sources_by_id = {s["source_id"]: s for s in cls.corpus.sources}
        cls.chunks_by_id = {c["chunk_id"]: c for c in cls.corpus.chunks}

    def test_full_width_relu_equation_includes_body_and_tag(self) -> None:
        equation = self.chunks_by_id["book_eq_4_18"]
        meta = equation["metadata"]
        self.assertEqual(meta["pdf_page"], 68)
        self.assertLess(meta["bbox"][1], 472)
        self.assertGreater(meta["bbox"][3], 509)
        self.assertGreater(len(equation["raw_content"]), 80)
        self.assertIn("ReLU", equation["raw_content"])
        self.assertIn("(4.18)", equation["raw_content"])
        self.assertEqual(meta["review_status"], "pending")
        with Image.open(PROJECT_ROOT / meta["asset_path"]) as crop:
            self.assertGreater(crop.height, 85)
        # Formula glyphs must not also be emitted as corrupted book prose.
        prose = [c for c in self.corpus.chunks if c["content_type"] == "text"
                 and c["metadata"].get("pdf_page") == 68]
        self.assertFalse(any("ReLU[β" in c["raw_content"] for c in prose))

    def test_adjacent_matrix_equation_keeps_rows_below_number(self) -> None:
        first = self.chunks_by_id["book_eq_4_11"]
        second = self.chunks_by_id["book_eq_4_12"]
        self.assertEqual(first["metadata"]["pdf_page"], 62)
        self.assertGreater(first["metadata"]["bbox"][3], 581)
        self.assertGreater(second["metadata"]["bbox"][1], 576)
        self.assertIn("θ30", first["raw_content"])
        self.assertNotIn("θ30", second["raw_content"])

    def test_vertical_ellipsis_keeps_multiline_definition_together(self) -> None:
        equation = self.chunks_by_id["book_eq_4_15"]
        self.assertLess(equation["metadata"]["bbox"][1], 365)
        self.assertGreater(equation["metadata"]["bbox"][3], 458)
        for glyph in ("h1", "h2", "h3", "hK", "..."):
            self.assertIn(glyph, equation["raw_content"])

    def test_wide_derivation_constraint_table_and_problem_math_are_complete(self) -> None:
        derivation = self.chunks_by_id["book_eq_8_3"]
        self.assertLess(derivation["metadata"]["bbox"][1], 549)
        self.assertIn("Ey", derivation["raw_content"])
        batchnorm = self.chunks_by_id["book_eq_11_10"]
        self.assertGreater(batchnorm["metadata"]["bbox"][3], 360)
        self.assertLess(batchnorm["metadata"]["bbox"][3], 365)
        self.assertIn("δ", batchnorm["raw_content"])
        constraints = self.chunks_by_id["book_eq_15_11"]
        self.assertLess(constraints["metadata"]["bbox"][1], 577)
        self.assertGreater(constraints["metadata"]["bbox"][3], 619)
        self.assertIn("non-negative masses", constraints["raw_content"])

    def test_fraction_fragments_belong_to_their_numbered_display(self) -> None:
        derivation = self.chunks_by_id["book_eq_18_33"]
        self.assertLess(derivation["metadata"]["bbox"][1], 274)
        self.assertGreater(derivation["metadata"]["bbox"][3], 447)
        self.assertLess(self.chunks_by_id["book_eq_18_38"]["metadata"]["bbox"][1], 330)
        self.assertLess(self.chunks_by_id["book_eq_C_29"]["metadata"]["bbox"][1], 463)
        fragments = [c for c in self.corpus.chunks if c["content_type"] == "equation"
                     and not c["metadata"].get("equation_number")
                     and c["metadata"].get("pdf_page") in (52, 376, 377, 456, 473)]
        self.assertEqual(fragments, [])

    def test_sources_only_pinned_book_and_69_chapter_notebooks(self) -> None:
        self.assertEqual(len(self.corpus.sources), 70)
        self.assertIn("book", self.sources_by_id)
        book_src = self.sources_by_id["book"]
        self.assertEqual(book_src["sha256"], BOOK_SHA256)
        self.assertEqual(book_src["path"], "data/raw/UnderstandingDeepLearning.pdf")
        self.assertIn("CC-BY-NC-ND", book_src["metadata"]["license"])

        nb_sources = [s for s in self.corpus.sources if s["source_type"] == "notebook"]
        self.assertEqual(len(nb_sources), 69)
        for src in nb_sources:
            self.assertTrue(src["source_id"].startswith("Notebooks/Chap"))
            self.assertTrue(src["source_id"].endswith(".ipynb"))
            self.assertEqual(src["revision"], NOTEBOOK_COMMIT)
            self.assertEqual(src["metadata"]["license"], "MIT")
            full_path = PROJECT_ROOT / src["path"]
            self.assertTrue(full_path.is_file(), f"Missing notebook file: {full_path}")

        all_paths = " ".join(s["path"] for s in self.corpus.sources)
        for forbidden in (
            "UDL_Equations.tex",
            "UDL_Answer_Booklet",
            "UDL_Errata",
            "Blogs/",
            "Slides/",
            "CM20315",
        ):
            self.assertNotIn(forbidden, all_paths)

    def test_ensure_sources_offline_and_checksum_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            with self.assertRaises(UDLException) as ctx:
                ensure_sources(tmp_root, offline=True)
            self.assertEqual(ctx.exception.code, "source_missing")

            raw_dir = tmp_root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            bad_pdf = raw_dir / "UnderstandingDeepLearning.pdf"
            bad_pdf.write_bytes(b"corrupted pdf bytes")

            with self.assertRaises(UDLException) as ctx2:
                ensure_sources(tmp_root, offline=True)
            self.assertEqual(ctx2.exception.code, "source_checksum_mismatch")

            with patch(
                "urllib.request.urlopen",
                return_value=io.BytesIO(b"wrong downloaded bytes"),
            ):
                with self.assertRaises(UDLException) as ctx3:
                    ensure_sources(tmp_root, offline=False)
                self.assertEqual(ctx3.exception.code, "source_checksum_mismatch")
            part_files = list(raw_dir.glob("*.part"))
            self.assertEqual(part_files, [])

    def test_all_69_notebooks_have_zero_dropped_lines_and_valid_links(self) -> None:
        nb_repo_dir = PROJECT_ROOT / "data" / "raw" / "udlbook_notebooks"
        nb_files = _find_chapter_notebooks(nb_repo_dir)
        self.assertEqual(len(nb_files), 69)

        nb_chunks_by_source: dict[str, list[dict]] = defaultdict(list)
        for ch in self.corpus.chunks:
            if ch["source_type"] == "notebook":
                nb_chunks_by_source[ch["source_id"]].append(ch)

        for nb_file in nb_files:
            rel_id = nb_file.relative_to(nb_repo_dir).as_posix()
            chunks = nb_chunks_by_source[rel_id]
            self.assertTrue(len(chunks) > 0, f"No chunks emitted for {rel_id}")

            for idx, ch in enumerate(chunks):
                expected_prev = chunks[idx - 1]["chunk_id"] if idx > 0 else None
                expected_next = (
                    chunks[idx + 1]["chunk_id"] if idx < len(chunks) - 1 else None
                )
                self.assertEqual(ch["metadata"]["previous_id"], expected_prev)
                self.assertEqual(ch["metadata"]["next_id"], expected_next)
                if expected_prev:
                    self.assertIn(expected_prev, self.chunks_by_id)
                if expected_next:
                    self.assertIn(expected_next, self.chunks_by_id)

            by_cell: dict[int, list[dict]] = defaultdict(list)
            for ch in chunks:
                by_cell[int(ch["metadata"]["cell_index"])].append(ch)

            with open(nb_file, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            for cell_idx, cell in enumerate(nb.cells):
                raw_src = (
                    "".join(cell.source)
                    if isinstance(cell.source, list)
                    else str(cell.source)
                )
                if not raw_src.strip():
                    continue
                self.assertIn(
                    cell_idx,
                    by_cell,
                    f"Dropped non-empty cell {cell_idx} in {rel_id}",
                )
                cell_chunks = sorted(
                    by_cell[cell_idx], key=lambda c: int(c["metadata"]["start_line"])
                )
                expected_lines = raw_src.splitlines()
                n_lines = max(1, len(expected_lines))

                self.assertEqual(cell_chunks[0]["metadata"]["start_line"], 1)
                self.assertEqual(cell_chunks[-1]["metadata"]["end_line"], n_lines)
                for k in range(1, len(cell_chunks)):
                    self.assertEqual(
                        cell_chunks[k]["metadata"]["start_line"],
                        cell_chunks[k - 1]["metadata"]["end_line"] + 1,
                        f"Non-contiguous line intervals in {rel_id} cell {cell_idx}",
                    )

                # Verify exact raw byte/character fidelity including trailing newlines
                reconstructed_raw = "".join(c["raw_content"] for c in cell_chunks)
                self.assertEqual(
                    reconstructed_raw,
                    raw_src,
                    f"Raw character mismatch in {rel_id} cell {cell_idx}",
                )
                reconstructed_lines = [
                    ln for c in cell_chunks for ln in c["raw_content"].splitlines()
                ]
                self.assertEqual(
                    reconstructed_lines,
                    expected_lines,
                    f"Line mismatch in {rel_id} cell {cell_idx}",
                )
                for c in cell_chunks:
                    self.assertNotIn("data:image/", c["content"])
                    if c.get("embedding_text"):
                        self.assertNotIn("data:image/", c["embedding_text"])
                    if "data:image/" not in c["raw_content"]:
                        self.assertEqual(c["content"], c["raw_content"])
                    else:
                        self.assertIn("[Embedded ", c["content"])
                        asset_rel = c["metadata"].get("asset_path")
                        self.assertTrue(
                            asset_rel and (PROJECT_ROOT / asset_rel).is_file(),
                            f"Missing extracted notebook image asset for {c['chunk_id']}",
                        )

    def test_per_definition_metadata_decorators_magics_and_prior_context(self) -> None:
        sample_code = (
            "%matplotlib inline\n"
            "# Header comment for first function\n"
            "@my_decorator\n"
            "def first_func(x, y):\n"
            '    """First docstring."""\n'
            "    return x + y\n"
            "\n"
            "# Top-level script lines between functions\n"
            "val = first_func(1, 2)\n"
            "\n"
            "# Header comment for second function\n"
            "def second_func(z):\n"
            '    """Second docstring."""\n'
            "    # TODO: Replace this line\n"
            "    return first_func(z, 1)\n"
        )
        intervals = _partition_code_cell_intervals(sample_code)
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0]["name"], "first_func")
        self.assertEqual(intervals[0]["signature"], "def first_func(x, y):")
        self.assertEqual(intervals[0]["docstring"], "First docstring.")
        self.assertIn("@my_decorator", intervals[0]["raw_content"])
        self.assertIn("%matplotlib inline", intervals[0]["raw_content"])

        self.assertEqual(intervals[1]["kind"], "script")
        self.assertIn("val = first_func(1, 2)", intervals[1]["raw_content"])

        self.assertEqual(intervals[2]["name"], "second_func")
        self.assertEqual(intervals[2]["signature"], "def second_func(z):")
        self.assertEqual(intervals[2]["docstring"], "Second docstring.")
        self.assertEqual(intervals[2]["symbol_names"], ["second_func"])
        self.assertIn("first_func", intervals[2]["called_symbols"])

        self.assertEqual(
            "".join(iv["raw_content"] for iv in intervals),
            sample_code,
        )

        # Verify no chunk in the corpus has an unpartitioned >6000 char embedding input
        for ch in self.corpus.chunks:
            emb_txt = ch.get("embedding_text") or ch["content"]
            self.assertLess(
                len(emb_txt),
                6000,
                f"Chunk {ch['chunk_id']} embedding text is oversized ({len(emb_txt)} chars)",
            )

    def test_285_figures_and_2_algorithms_with_rendered_assets(self) -> None:
        layout_path = PROJECT_ROOT / "resources" / "book_layout.json"
        self.assertTrue(layout_path.is_file())
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        self.assertEqual(layout["pdf_sha256"], BOOK_SHA256)
        self.assertEqual(layout["figure_count"], 285)
        self.assertEqual(layout["algorithm_count"], 2)

        fig_chunks = [c for c in self.corpus.chunks if c["content_type"] == "figure"]
        alg_chunks = [c for c in self.corpus.chunks if c["content_type"] == "algorithm"]
        self.assertEqual(len(fig_chunks), 285)
        self.assertEqual(len(alg_chunks), 2)

        fig_numbers = {c["metadata"]["figure_number"] for c in fig_chunks}
        for missing_9 in (
            "3.7",
            "5.6",
            "6.8",
            "8.9",
            "11.5",
            "15.10",
            "19.6",
            "20.6",
            "20.9",
        ):
            self.assertIn(missing_9, fig_numbers, f"Missing figure {missing_9}")
        for app_fig in ("B.1", "B.2", "B.3", "C.1", "C.2", "C.3", "C.4", "C.5", "C.6"):
            self.assertIn(app_fig, fig_numbers, f"Missing appendix figure {app_fig}")

        alg_numbers = {c["metadata"]["algorithm_number"]: c for c in alg_chunks}
        self.assertEqual(set(alg_numbers.keys()), {"18.1", "18.2"})
        for _anum, ach in alg_numbers.items():
            self.assertEqual(ach["metadata"]["pdf_page"], 378)
            self.assertEqual(ach["metadata"]["book_page"], 364)
            asset_file = PROJECT_ROOT / ach["metadata"]["asset_path"]
            self.assertTrue(asset_file.is_file())

        for fch in fig_chunks:
            bbox = fch["metadata"]["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            self.assertGreaterEqual(bbox[1], 121.0, f"Header capture in {fch['chunk_id']}")
            self.assertLessEqual(bbox[3], 704.0, f"Footer capture in {fch['chunk_id']}")
            self.assertGreaterEqual(w, 80.0, f"Narrow sliver in {fch['chunk_id']}")
            self.assertGreaterEqual(h, 18.0, f"Short sliver in {fch['chunk_id']}")

            asset_path = PROJECT_ROOT / fch["metadata"]["asset_path"]
            self.assertTrue(asset_path.is_file(), f"Missing crop {asset_path}")
            with Image.open(asset_path) as im:
                self.assertGreater(im.size[0], 100)
                self.assertGreater(im.size[1], 30)

    def test_197_problems_complete_across_pages_with_no_false_mentions(self) -> None:
        exercises = [c for c in self.corpus.chunks if c["content_type"] == "exercise"]
        self.assertEqual(len(exercises), 197)
        prob_nums = [c["metadata"]["problem_number"] for c in exercises]
        self.assertEqual(len(set(prob_nums)), 197, "Duplicate problem numbers found")

        for ex in exercises:
            self.assertTrue(
                ex["content"].startswith("Problem "),
                f"False problem chunk does not start with Problem heading: {ex['content'][:60]}",
            )
            self.assertEqual(ex["metadata"]["section"], "Problems")

        by_prob = {c["metadata"]["problem_number"]: c for c in exercises}
        self.assertEqual(by_prob["17.1"]["metadata"]["pdf_page"], 361)
        self.assertEqual(by_prob["17.1"]["metadata"]["end_pdf_page"], 362)
        self.assertIn("equation 17.4", by_prob["17.1"]["content"])
        self.assertEqual(by_prob["17.5"]["metadata"]["pdf_page"], 362)
        self.assertEqual(by_prob["17.5"]["metadata"]["end_pdf_page"], 362)
        self.assertIn("REINFORCE", by_prob["17.5"]["content"])
        self.assertEqual(
            by_prob["16.7"]["metadata"]["referenced_equations"], ["16.31", "16.32"]
        )
        self.assertEqual(
            by_prob["16.9"]["metadata"]["referenced_equations"], ["16.33"]
        )
        self.assertIn("where b =", by_prob["16.9"]["content"])

    def test_display_equations_inventory_crops_and_pending_review_status(self) -> None:
        eq_chunks = [c for c in self.corpus.chunks if c["content_type"] == "equation"]
        numbered = [
            c for c in eq_chunks if c["metadata"].get("equation_number") is not None
        ]
        unnumbered = [
            c for c in eq_chunks if c["metadata"].get("equation_number") is None
        ]
        self.assertEqual(len(numbered), 496)
        # Earlier candidates included denominator rows, matrix tails and inline
        # paragraph glyphs. Full-page inspection leaves these four displays.
        self.assertEqual(len(unnumbered), 4)
        self.assertEqual({c["metadata"]["pdf_page"] for c in unnumbered}, {74, 299, 452, 455})
        self.assertEqual(
            len({c["metadata"]["equation_number"] for c in numbered}),
            496,
            "Duplicate numbered equations found",
        )

        for eq in eq_chunks:
            meta = eq["metadata"]
            self.assertEqual(meta["review_status"], "pending")
            asset_file = PROJECT_ROOT / meta["asset_path"]
            self.assertTrue(asset_file.is_file(), f"Missing equation crop {asset_file}")
            actual_sha = hashlib.sha256(asset_file.read_bytes()).hexdigest()
            self.assertEqual(meta["crop_sha256"], actual_sha)
            self.assertTrue(bool(meta.get("raw_glyphs")))

    def test_prose_chunks_exclude_headers_footers_and_separate_margin_notes(self) -> None:
        prose_chunks = [c for c in self.corpus.chunks if c["content_type"] == "text"]
        self.assertGreater(len(prose_chunks), 400)
        for pch in prose_chunks:
            txt = pch["content"]
            self.assertNotIn("Draft: please send errata", txt)
            self.assertNotIn("This work is subject to a Creative Commons", txt)
            pno = pch["metadata"]["pdf_page"]
            if pno >= 15:
                self.assertEqual(pch["metadata"]["book_page"], pno - 14)


if __name__ == "__main__":
    unittest.main()
