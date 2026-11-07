"""Authoritative UDL corpus extraction, source verification, layout manifest, and asset rendering.

Owns:
- `ensure_sources(project_root=PROJECT_ROOT, offline=False)`: deterministic checksum-verified
  and commit-pinned source verification/downloading without implicit floating updates.
- `extract_corpus(project_root=PROJECT_ROOT, render_assets=True) -> CorpusResult`:
  complete extraction of the pinned Understanding Deep Learning book PDF and 69 chapter
  notebooks (`Notebooks/Chap*/*.ipynb`) into `CorpusResult(sources, chunks, report)`.
- Versioned layout & bounding-box manifest at `resources/book_layout.json` keyed by `BOOK_SHA256`.
- Java-free rendering of all 285 figures, 2 algorithms (18.1, 18.2), and all display equations
  into `data/assets/`.
"""

from __future__ import annotations

import ast
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unicodedata
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fitz  # PyMuPDF
import nbformat

from src.contracts import (
    BOOK_SHA256,
    BOOK_URL,
    CORPUS_VERSION,
    NOTEBOOK_COMMIT,
    NOTEBOOK_REPO,
    PROJECT_ROOT,
    CorpusResult,
    UDLException,
)

LAYOUT_MANIFEST_VERSION = "udl-layout-v1"

BOOK_LICENSE_METADATA: Dict[str, Any] = {
    "license": "CC-BY-NC-ND 4.0",
    "license_notice": "This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.",
    "author": "Simon J.D. Prince",
    "publisher": "MIT Press",
    "edition": "v5.0.2 (2025-05-29)",
    "url": BOOK_URL,
}

NOTEBOOK_LICENSE_METADATA: Dict[str, Any] = {
    "license": "MIT",
    "license_notice": "MIT License - Copyright (c) 2023 Simon J.D. Prince",
    "repo": NOTEBOOK_REPO,
    "commit": NOTEBOOK_COMMIT,
}

# Six figures in the PDF are LaTeX boxed/table environments rather than external Form XObjects.
# Their exact coordinates come directly from the horizontal/vertical rules in `page.get_drawings()`.
BOXED_FIGURE_GEOMETRY: Dict[str, Tuple[int, Tuple[float, float, float, float]]] = {
    "1.6": (22, (133.8, 313.9, 522.1, 412.7)),
    "1.8": (23, (94.9, 138.4, 483.3, 254.5)),
    "5.11": (84, (148.3, 137.6, 500.8, 447.5)),
    "7.8": (126, (133.8, 137.2, 515.3, 663.4)),
    "14.3": (286, (142.9, 137.6, 506.2, 230.5)),
    "21.2": (438, (133.8, 138.6, 521.7, 196.5)),
}

# Two boxed algorithms on PDF page 378 (book page 364), bounded by horizontal rules in `page.get_drawings()`.
ALGORITHM_GEOMETRY: Dict[str, Tuple[int, Tuple[float, float, float, float]]] = {
    "18.1": (378, (133.77, 140.55, 515.34, 277.52)),
    "18.2": (378, (133.77, 297.83, 515.34, 419.92)),
}

# Pinned v5.0.2 geometry exception: the full-width ReLU identity starts near
# the prose margin and its tag sits on a lower baseline. The normal centered
# math heuristic otherwise retains only the tag/denominator row. Coordinates
# were checked against the original PDF, whose SHA is enforced by extraction.
DISPLAY_MATH_REGIONS = {
    68: [(133.77, 467.57, 515.34, 513.35)],
    # Wide expected-error derivation and a constraint table containing prose.
    137: [(94.91, 544.20, 476.48, 634.25)],
    298: [(133.77, 572.69, 515.34, 623.82)],
    376: [(133.77, 269.86, 515.34, 451.66)],
    377: [(94.91, 325.75, 476.48, 375.07)],
    387: [(94.91, 145.0, 480.0, 188.83), (94.91, 271.0, 480.0, 319.0)],
    473: [(94.91, 454.77, 476.48, 502.67)],
}

# Raised/lowered glyphs in these prose paragraphs are not separate displays.
INLINE_MATH_FRAGMENTS = {52: [(657.69, 671.97)], 456: [(498.97, 520.11)]}

# End of a numbered display whose following prose has a raised summation
# sign: a baseline-only lookahead would mistake that sign for a fifth row.
DISPLAY_MATH_BOTTOMS = {(220, "11.10"): 360.64}

ROMAN_PAGES: Dict[int, str] = {
    5: "v",
    6: "vi",
    7: "vii",
    8: "viii",
    9: "ix",
    10: "x",
    11: "xi",
    12: "xii",
    13: "xiii",
    14: "xiv",
}

MATH_FONT_PREFIXES = ("CMMI", "CMSY", "CMEX", "CMR", "CMBX", "MSBM", "MSAM", "LMMath")
ALLOWED_MATH_WORDS = {
    "where",
    "and",
    "or",
    "for",
    "if",
    "then",
    "else",
    "otherwise",
    "with",
    "such",
    "that",
    "subject",
    "to",
    "given",
    "by",
    "as",
    "in",
    "on",
    "at",
    "from",
    "over",
    "under",
    "when",
    "constant",
    "const",
    "even",
    "odd",
    "true",
    "false",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "ln",
    "min",
    "max",
    "argmin",
    "argmax",
    "softmax",
    "relu",
    "sig",
    "tanh",
    "diag",
    "tr",
    "trace",
    "det",
    "cov",
    "var",
    "std",
    "sign",
    "sgn",
    "kl",
    "elbo",
    "pr",
    "prob",
    "norm",
    "dist",
    "loss",
    "error",
    "update",
    "layer",
    "attention",
    "self",
    "multihead",
    "sa",
    "mhsa",
    "mlp",
    "gnn",
    "conv",
    "batchnorm",
    "layernorm",
    "dropout",
    "target",
    "mean",
    "of",
    "predicted",
    "reconstruction",
    "term",
    "prior",
    "likelihood",
    "posterior",
    "quality",
    "coverage",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_prose_text(text: str) -> str:
    """Normalize Unicode NFC, strip control/NUL chars, repair hyphenated linebreaks, collapse spaces."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    # Repair hyphenations across line breaks
    text = re.sub(r"([A-Za-z]{2,})-\s*[\r\n]+\s*([a-z][A-Za-z]*)", r"\1\2", text)
    # Repair common suffix hyphenations when lines were joined by space
    suffixes = (
        r"(?:tion|tions|sion|sions|ing|ed|er|ers|est|ment|ments|ness|ity|ities|"
        r"al|als|ive|ives|ous|able|ible|ize|izes|ized|ising|izing|ly|ally|ward|wards|"
        r"ance|ence|ant|ent|ry|ries|ty|ties|ic|ics|ical|ically|ate|ates|ated|ating)"
    )
    text = re.sub(rf"\b([A-Za-z]{{3,}})-\s+({suffixes})\b", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _book_page_for_pdf_page(pdf_page: int) -> Optional[Union[int, str]]:
    """Return printed book page number (1-based int for main text, Roman for front matter)."""
    if pdf_page >= 15:
        return pdf_page - 14
    return ROMAN_PAGES.get(pdf_page)


def _find_chapter_notebooks(notebooks_root: Path) -> List[Path]:
    """Return sorted list of the 69 canonical `Notebooks/Chap*/*.ipynb` files."""
    nb_dir = notebooks_root / "Notebooks"
    if not nb_dir.is_dir():
        return []
    found: List[Path] = []
    for chap_dir in sorted(nb_dir.glob("Chap*")):
        if not chap_dir.is_dir():
            continue
        for nb_file in sorted(chap_dir.glob("*.ipynb")):
            if ".ipynb_checkpoints" not in str(nb_file) and nb_file.is_file():
                found.append(nb_file)
    return found


def ensure_sources(
    project_root: Union[str, Path] = PROJECT_ROOT,
    offline: bool = False,
) -> Dict[str, Any]:
    """Verify or deterministically download the pinned book PDF and 69 chapter notebooks.

    - Never performs an implicit floating update when valid pinned sources exist.
    - Uses atomic temporary files and SHA-256 / commit verification when downloading.
    - Raises `UDLException` if `offline=True` and sources are missing or corrupted.
    """
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = raw_dir / "UnderstandingDeepLearning.pdf"
    nb_repo_dir = raw_dir / "udlbook_notebooks"

    # 1. Verify or download book PDF
    pdf_ok = False
    if pdf_path.is_file():
        actual_sha = _sha256_file(pdf_path)
        if actual_sha == BOOK_SHA256:
            pdf_ok = True
        elif offline:
            raise UDLException(
                "source_checksum_mismatch",
                f"Book PDF at {pdf_path} has SHA-256 {actual_sha}, expected {BOOK_SHA256}.",
                retryable=False,
            )

    if not pdf_ok:
        if offline:
            raise UDLException(
                "source_missing",
                f"Book PDF is missing at {pdf_path} and offline=True.",
                retryable=False,
            )
        fd, tmp_name = tempfile.mkstemp(prefix="udl_book_", suffix=".pdf.part", dir=str(raw_dir))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            req = urllib.request.Request(BOOK_URL, headers={"User-Agent": "UDL-Tutor/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=65536)
            dl_sha = _sha256_file(tmp_path)
            if dl_sha != BOOK_SHA256:
                raise UDLException(
                    "source_checksum_mismatch",
                    f"Downloaded book PDF SHA-256 ({dl_sha}) does not match pinned {BOOK_SHA256}.",
                    retryable=False,
                )
            os.replace(str(tmp_path), str(pdf_path))
        except UDLException:
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise UDLException(
                "source_download_failed",
                f"Failed to download book PDF from {BOOK_URL}: {exc}",
                retryable=True,
            ) from exc

    # 2. Verify or clone pinned notebook repository
    notebooks = _find_chapter_notebooks(nb_repo_dir)
    nb_ok = len(notebooks) == 69
    git_dir = nb_repo_dir / ".git"
    if nb_ok and git_dir.exists():
        try:
            proc = subprocess.run(
                ["git", "-C", str(nb_repo_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            head_commit = proc.stdout.strip()
            if head_commit != NOTEBOOK_COMMIT:
                if offline:
                    raise UDLException(
                        "source_commit_mismatch",
                        f"Notebooks repo HEAD is {head_commit}, expected {NOTEBOOK_COMMIT}.",
                        retryable=False,
                    )
                nb_ok = False
        except UDLException:
            raise
        except Exception:
            pass

    if not nb_ok:
        if offline:
            raise UDLException(
                "source_missing",
                f"Pinned 69 chapter notebooks are missing or incomplete under {nb_repo_dir} and offline=True.",
                retryable=False,
            )
        tmp_repo = Path(tempfile.mkdtemp(prefix="udl_notebooks_", dir=str(raw_dir)))
        try:
            subprocess.run(
                ["git", "init", str(tmp_repo)],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(tmp_repo), "remote", "add", "origin", NOTEBOOK_REPO],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(tmp_repo), "fetch", "--depth", "1", "origin", NOTEBOOK_COMMIT],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(tmp_repo), "checkout", "--detach", "FETCH_HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            fetched_nbs = _find_chapter_notebooks(tmp_repo)
            if len(fetched_nbs) != 69:
                raise UDLException(
                    "source_incomplete",
                    f"Pinned commit {NOTEBOOK_COMMIT} contained {len(fetched_nbs)} notebooks, expected 69.",
                    retryable=False,
                )
            if nb_repo_dir.exists():
                backup_dir = raw_dir / f"udlbook_notebooks_backup_{os.getpid()}"
                os.replace(str(nb_repo_dir), str(backup_dir))
                try:
                    os.replace(str(tmp_repo), str(nb_repo_dir))
                finally:
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                os.replace(str(tmp_repo), str(nb_repo_dir))
            notebooks = _find_chapter_notebooks(nb_repo_dir)
        except UDLException:
            shutil.rmtree(tmp_repo, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(tmp_repo, ignore_errors=True)
            raise UDLException(
                "source_download_failed",
                f"Failed to fetch pinned notebooks commit {NOTEBOOK_COMMIT}: {exc}",
                retryable=True,
            ) from exc

    return {
        "book_path": str(pdf_path.relative_to(root)),
        "book_sha256": BOOK_SHA256,
        "notebooks_dir": str((nb_repo_dir / "Notebooks").relative_to(root)),
        "notebook_commit": NOTEBOOK_COMMIT,
        "notebook_count": len(notebooks),
    }


# ---------------------------------------------------------------------------
# PDF Geometry, Layout Manifest, Figures, Algorithms, Equations, Problems, Prose
# ---------------------------------------------------------------------------


def _extract_page_form_xobject_rects(
    page: fitz.Page,
) -> List[Tuple[str, int, Tuple[float, float, float, float]]]:
    """Parse actual PDF page content stream `cm` transformation matrices before `/FmX Do`."""
    height = float(page.rect.height)
    xobjs = {
        name: (xref, bbox)
        for xref, name, parent, bbox in page.get_xobjects()
        if parent == 0
    }
    if not xobjs:
        return []
    stream = page.read_contents().decode("latin1", errors="ignore")
    pattern = re.compile(
        r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+cm\b"
        r"(?:(?!cm\b).){0,120}?/(Fm\d+)\s+Do",
        re.DOTALL,
    )
    rects: List[Tuple[str, int, Tuple[float, float, float, float]]] = []
    for match in pattern.finditer(stream):
        a, _b, _c, d, e, f = [float(match.group(i)) for i in range(1, 7)]
        name = match.group(7)
        if name in xobjs:
            xref, (bx0, by0, bx1, by1) = xobjs[name]
            x0 = e + a * bx0
            x1 = e + a * bx1
            y0 = height - (f + d * by1)
            y1 = height - (f + d * by0)
            bbox = (
                round(min(x0, x1), 2),
                round(min(y0, y1), 2),
                round(max(x0, x1), 2),
                round(max(y0, y1), 2),
            )
            rects.append((name, int(xref), bbox))
    rects.sort(key=lambda item: (item[2][1], item[2][0]))
    return rects


def _bbox_union(boxes: Sequence[Sequence[float]]) -> List[float]:
    valid = [b for b in boxes if b and len(b) == 4]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(float(b[0]) for b in valid), 2),
        round(min(float(b[1]) for b in valid), 2),
        round(max(float(b[2]) for b in valid), 2),
        round(max(float(b[3]) for b in valid), 2),
    ]


def _is_outer_margin_span(pno: int, span_bbox: Sequence[float], font: str, size: float) -> bool:
    """Return True if a span is a 7.97pt outer-margin cross-reference note."""
    sx0, _sy0, sx1, _sy1 = span_bbox
    if size > 8.5 or "Bold" in font:
        return False
    if pno % 2 == 0:
        return float(sx1) < 120.0
    return float(sx0) > 485.0


def _is_display_math_visual_row(
    row: Dict[str, Any],
    row_text: str,
    left_margin: float,
    right_margin: float,
) -> Tuple[bool, Optional[str]]:
    """Determine if a visual row is part of a display equation and return `(is_math, eq_number)`."""
    rx0, _ry0, rx1, _ry1 = row["bbox"]
    if row.get("pinned_inline_math"):
        return False, None
    eq_tag_m = re.search(r"\(((\d+|[ABC])\.\d+)\)\s*$", row_text)
    if eq_tag_m and abs(rx1 - right_margin) < 15.0:
        return True, eq_tag_m.group(1)

    if row.get("pinned_display_math"):
        return True, None

    # Vertical ellipsis uses an ordinary Roman font in the PDF. Keep it in
    # a display band so a six-line definition such as (4.15) is not split.
    dots_only = re.sub(r"[\uf8e0-\uf8ff]", "", row_text).strip()
    if re.fullmatch(r"[.\s⋮⋱…]+", dots_only) and len(dots_only) >= 3 and rx0 > left_margin + 22.0:
        return True, None

    if row.get("is_footnote"):
        return False, None

    spans = row["spans"]
    if any("LMSans10-Bold" in s["font"] and float(s["size"]) >= 9.5 for s in spans):
        return False, None

    total_chars = 0
    math_font_chars = 0
    roman_words: List[str] = []
    for s in spans:
        t = s["text"].strip()
        if not t:
            continue
        f = s["font"]
        n_c = len(re.sub(r"\s+", "", t))
        total_chars += n_c
        if any(f.startswith(pref) for pref in MATH_FONT_PREFIXES):
            math_font_chars += n_c
        elif "LMRoman" in f or "LMSans" in f:
            for w in re.findall(r"[A-Za-z]{4,}", t):
                if w.lower() not in ALLOWED_MATH_WORDS:
                    roman_words.append(w)

    if total_chars == 0 or math_font_chars == 0:
        return False, None
    if rx0 <= left_margin + 22.0:
        # Long display formulas can start at the body margin. Short inline
        # definitions and prose introducing a formula remain ordinary text.
        if (rx1 - rx0 < (right_margin - left_margin) * 0.5
                or math_font_chars / total_chars < 0.70
                or re.match(r"^(?:where|so|and|then|that|such|given|for)\b", row_text, re.IGNORECASE)):
            return False, None
    # Reject indented bullet/caption prose lines that have multiple English words
    if len(roman_words) >= 2:
        return False, None
    if math_font_chars / max(1, total_chars) < 0.45 and len(roman_words) >= 1:
        return False, None
    return True, None


def _build_page_structure_map(
    doc: fitz.Document,
) -> Tuple[Dict[int, Dict[str, Optional[str]]], List[Dict[str, Any]]]:
    """Build per-page chapter/title structure map from the PDF TOC and headings."""
    toc = doc.get_toc()
    toc_entries: List[Dict[str, Any]] = []
    chap_counter = 0
    app_letters = ["A", "B", "C"]
    app_idx = 0

    l1_starts: List[Tuple[int, Optional[str], str]] = []
    for level, title, page_1based in toc:
        clean_title = title.strip()
        if level == 1:
            if clean_title in ("Preface", "Acknowledgements", "Bibliography", "Index"):
                chap_id = None
            elif clean_title in ("Notation", "Mathematics", "Probability"):
                chap_id = app_letters[app_idx] if app_idx < len(app_letters) else None
                app_idx += 1
            else:
                chap_counter += 1
                chap_id = str(chap_counter)
            l1_starts.append((int(page_1based), chap_id, clean_title))
        toc_entries.append(
            {"level": int(level), "title": clean_title, "pdf_page": int(page_1based)}
        )

    page_map: Dict[int, Dict[str, Optional[str]]] = {}
    for pno in range(1, len(doc) + 1):
        active_chap = None
        active_title = "Front Matter"
        for start_page, chap_id, title in l1_starts:
            if pno >= start_page:
                active_chap = chap_id
                active_title = title
            else:
                break
        page_map[pno] = {
            "chapter": active_chap,
            "chapter_title": active_title,
        }
    return page_map, toc_entries


def _render_clip_png(
    page: fitz.Page,
    bbox: Sequence[float],
    dest_path: Path,
    *,
    pad: float = 2.0,
    dpi: int = 150,
) -> str:
    """Render a precise PDF clip to PNG without capturing headers or footers."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    clip_rect = fitz.Rect(
        max(0.0, x0 - pad),
        max(121.0, y0 - pad),
        min(float(page.rect.width), x1 + pad),
        min(704.0, y1 + pad),
    )
    pix = page.get_pixmap(clip=clip_rect, dpi=dpi, alpha=False)
    png_bytes = pix.tobytes("png")
    dest_path.write_bytes(png_bytes)
    return hashlib.sha256(png_bytes).hexdigest()


def _extract_book_pdf(
    pdf_path: Path,
    project_root: Path,
    *,
    render_assets: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Extract all figures (285), algorithms (2), exercises (197), equations, and prose from the book PDF."""
    doc = fitz.open(str(pdf_path))
    try:
        page_struct, toc_entries = _build_page_structure_map(doc)
        assets_root = project_root / "data" / "assets"
        figures_dir = assets_root / "figures"
        algorithms_dir = assets_root / "algorithms"
        equations_dir = assets_root / "equations"

        if render_assets:
            figures_dir.mkdir(parents=True, exist_ok=True)
            algorithms_dir.mkdir(parents=True, exist_ok=True)
            equations_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: Locate all 285 Figure captions, 2 Algorithm captions, and section headings across the book
        figure_records: List[Dict[str, Any]] = []
        algorithm_records: List[Dict[str, Any]] = []
        page_exclusion_zones: Dict[int, List[Tuple[float, float, float, float]]] = {}

        cur_section: Optional[str] = None

        for pno in range(1, len(doc) + 1):
            page = doc[pno - 1]
            chap = page_struct[pno]["chapter"]
            d = page.get_text("dict")
            page_caps: List[Dict[str, Any]] = []

            for b in d.get("blocks", []):
                if "lines" not in b:
                    continue
                bx0, by0, bx1, by1 = [float(v) for v in b["bbox"]]
                if by0 < 120.0 or by1 > 704.0:
                    continue

                lines = b["lines"]
                first_spans = [s for l in lines for s in l["spans"] if s["text"].strip()]
                if not first_spans:
                    continue
                s0 = first_spans[0]
                font0 = s0["font"]
                size0 = round(float(s0["size"]), 2)
                first_line_txt = "".join(s["text"] for s in lines[0]["spans"]).strip()
                full_block_txt = _normalize_prose_text(
                    "\n".join(
                        "".join(
                            s["text"]
                            for s in l["spans"]
                            if not _is_outer_margin_span(
                                pno, s["bbox"], s["font"], float(s["size"])
                            )
                        )
                        for l in lines
                    )
                )

                if "LMSans10-Bold" in font0 and size0 in (17.22, 20.66):
                    cur_section = None
                elif "LMSans10-Bold" in font0 and abs(size0 - 11.95) < 0.3:
                    sec_m = re.match(r"^((?:\d+|[ABC])\.\d+)\s+(.*)$", full_block_txt)
                    if sec_m:
                        cur_section = sec_m.group(1)
                    elif full_block_txt in ("Problems", "Notes", "Summary"):
                        cur_section = full_block_txt

                if "Bold" in font0 and re.match(
                    r"^Figure\s+(?:\d+|[BC])\.\d+\b", first_line_txt
                ):
                    fig_m = re.match(r"^Figure\s+((?:\d+|[BC])\.\d+)\b", first_line_txt)
                    if fig_m:
                        fig_num = fig_m.group(1)
                        cap_bbox = (
                            round(bx0, 2),
                            round(by0, 2),
                            round(bx1, 2),
                            round(by1, 2),
                        )
                        page_caps.append(
                            {
                                "figure_number": fig_num,
                                "pdf_page": pno,
                                "book_page": _book_page_for_pdf_page(pno),
                                "chapter": chap
                                or (
                                    fig_num.split(".")[0]
                                    if fig_num[0].isalpha()
                                    else None
                                ),
                                "section": cur_section,
                                "caption": full_block_txt,
                                "caption_bbox": cap_bbox,
                            }
                        )

                if "Bold" in font0 and re.match(r"^Algorithm\s+\d+\.\d+", first_line_txt):
                    alg_m = re.match(r"^Algorithm\s+(\d+\.\d+)", first_line_txt)
                    if alg_m:
                        alg_num = alg_m.group(1)
                        _, alg_bbox = ALGORITHM_GEOMETRY.get(
                            alg_num,
                            (
                                pno,
                                (
                                    round(bx0, 2),
                                    round(by0, 2),
                                    round(bx1, 2),
                                    round(by1 + 120.0, 2),
                                ),
                            ),
                        )
                        alg_lines: List[str] = []
                        for ab in d.get("blocks", []):
                            if "lines" not in ab:
                                continue
                            _abx0, aby0, _abx1, aby1 = ab["bbox"]
                            if aby0 >= alg_bbox[1] - 2.0 and aby1 <= alg_bbox[3] + 2.0:
                                for al in ab["lines"]:
                                    lt = "".join(s["text"] for s in al["spans"]).strip()
                                    if lt:
                                        alg_lines.append(_normalize_prose_text(lt))
                        alg_text = "\n".join(alg_lines)
                        slug = alg_num.replace(".", "_")
                        rel_asset = f"data/assets/algorithms/alg_{slug}.png"
                        crop_sha = None
                        if render_assets:
                            crop_sha = _render_clip_png(
                                page, alg_bbox, project_root / rel_asset
                            )
                        elif (project_root / rel_asset).is_file():
                            crop_sha = _sha256_file(project_root / rel_asset)

                        algorithm_records.append(
                            {
                                "algorithm_number": alg_num,
                                "pdf_page": pno,
                                "book_page": _book_page_for_pdf_page(pno),
                                "chapter": chap or alg_num.split(".")[0],
                                "section": cur_section or "18.6",
                                "title": first_line_txt,
                                "content": alg_text,
                                "bbox": list(alg_bbox),
                                "caption_bbox": [
                                    round(bx0, 2),
                                    round(by0, 2),
                                    round(bx1, 2),
                                    round(by1, 2),
                                ],
                                "asset_path": rel_asset,
                                "crop_sha256": crop_sha,
                                "source_geometry": "drawing_rules",
                            }
                        )
                        page_exclusion_zones.setdefault(pno, []).append(alg_bbox)

            if page_caps:
                page_caps.sort(
                    key=lambda c: (c["caption_bbox"][1], c["caption_bbox"][0])
                )
                xobj_rects = _extract_page_form_xobject_rects(page)
                xobj_idx = 0
                for cap_info in page_caps:
                    fig_num = cap_info["figure_number"]
                    if fig_num in BOXED_FIGURE_GEOMETRY:
                        _, graphic_bbox = BOXED_FIGURE_GEOMETRY[fig_num]
                        geom_kind = "drawing_box"
                    elif xobj_idx < len(xobj_rects):
                        xname, xref, graphic_bbox = xobj_rects[xobj_idx]
                        xobj_idx += 1
                        geom_kind = f"xobject:{xname}:{xref}"
                    else:
                        graphic_bbox = cap_info["caption_bbox"]
                        geom_kind = "caption_fallback"

                    cap_bbox = list(cap_info["caption_bbox"])
                    caption_text = cap_info["caption"]
                    for b2 in d.get("blocks", []):
                        if "lines" not in b2:
                            continue
                        b2x0, b2y0, b2x1, b2y1 = [float(v) for v in b2["bbox"]]
                        if (
                            0.0 <= b2y0 - cap_bbox[3] <= 12.0
                            and abs(b2x0 - cap_bbox[0]) <= 6.0
                            and abs(b2x1 - cap_bbox[2]) <= 24.0
                        ):
                            extra_txt = _normalize_prose_text(
                                " ".join(
                                    "".join(s["text"] for s in l["spans"])
                                    for l in b2["lines"]
                                )
                            )
                            if extra_txt and not re.match(
                                r"^(Figure|Problem|Chapter|\d+\.\d+)", extra_txt
                            ):
                                caption_text = _normalize_prose_text(
                                    caption_text + " " + extra_txt
                                )
                                cap_bbox = _bbox_union(
                                    [cap_bbox, [b2x0, b2y0, b2x1, b2y1]]
                                )

                    union_bbox = _bbox_union([graphic_bbox, cap_bbox])
                    slug = fig_num.replace(".", "_")
                    rel_asset = f"data/assets/figures/fig_{slug}.png"
                    crop_sha = None
                    if render_assets:
                        crop_sha = _render_clip_png(
                            page, graphic_bbox, project_root / rel_asset
                        )
                    elif (project_root / rel_asset).is_file():
                        crop_sha = _sha256_file(project_root / rel_asset)

                    fig_rec = {
                        "figure_number": fig_num,
                        "pdf_page": pno,
                        "book_page": cap_info["book_page"],
                        "chapter": cap_info["chapter"] or fig_num.split(".")[0],
                        "section": cap_info["section"],
                        "caption": caption_text,
                        "bbox": [round(float(v), 2) for v in graphic_bbox],
                        "caption_bbox": [round(float(v), 2) for v in cap_bbox],
                        "union_bbox": union_bbox,
                        "asset_path": rel_asset,
                        "crop_sha256": crop_sha,
                        "source_geometry": geom_kind,
                    }
                    figure_records.append(fig_rec)
                    page_exclusion_zones.setdefault(pno, []).append(tuple(graphic_bbox))
                    page_exclusion_zones.setdefault(pno, []).append(tuple(cap_bbox))

        # Pass 2: Extract Problems, Display Equations (numbered + unnumbered), and Prose chunks
        prose_chunks: List[Dict[str, Any]] = []
        exercise_chunks: List[Dict[str, Any]] = []
        equation_chunks: List[Dict[str, Any]] = []

        active_section: Optional[str] = None
        active_subsection: Optional[str] = None
        in_problems_section = False
        current_problem: Optional[Dict[str, Any]] = None
        pending_prose: Optional[Dict[str, Any]] = None
        eq_seq = 0

        def flush_problem() -> None:
            nonlocal current_problem
            if current_problem is not None:
                text = _normalize_prose_text(" ".join(current_problem["parts"]))
                prob_num = current_problem["problem_number"]
                pno = current_problem["pdf_page"]
                chap = current_problem["chapter"]
                bbox = _bbox_union(current_problem["bboxes"])
                cid = f"book_problem_{prob_num.replace('.', '_')}"
                exercise_chunks.append(
                    {
                        "chunk_id": cid,
                        "source_id": "book",
                        "source_type": "book",
                        "content_type": "exercise",
                        "content": text,
                        "raw_content": "\n".join(current_problem["raw_parts"]).strip(),
                        "metadata": {
                            "chapter": chap,
                            "section": "Problems",
                            "pdf_page": pno,
                            "book_page": _book_page_for_pdf_page(pno),
                            "end_pdf_page": current_problem["end_pdf_page"],
                            "bbox": bbox,
                            "problem_number": prob_num,
                            "question_id": prob_num,
                            "has_asterisk": current_problem["has_asterisk"],
                            "referenced_equations": sorted(
                                set(current_problem["equations"])
                            ),
                            "margin_notes": sorted(set(current_problem["margin_notes"])),
                        },
                    }
                )
                current_problem = None

        def flush_prose() -> None:
            nonlocal pending_prose
            if pending_prose is not None:
                text = _normalize_prose_text(" ".join(pending_prose["parts"]))
                if text:
                    pno = pending_prose["pdf_page"]
                    chap = pending_prose["chapter"]
                    sec = pending_prose["section"]
                    bbox = _bbox_union(pending_prose["bboxes"])
                    seq_idx = len(prose_chunks) + 1
                    cid = f"book_text_{pno:03d}_{seq_idx:04d}"
                    prose_chunks.append(
                        {
                            "chunk_id": cid,
                            "source_id": "book",
                            "source_type": "book",
                            "content_type": "text",
                            "content": text,
                            "raw_content": "\n".join(
                                pending_prose["raw_parts"]
                            ).strip(),
                            "metadata": {
                                "chapter": chap,
                                "chapter_title": pending_prose["chapter_title"],
                                "section": sec,
                                "subsection": pending_prose["subsection"],
                                "pdf_page": pno,
                                "book_page": _book_page_for_pdf_page(pno),
                                "end_pdf_page": pending_prose["end_pdf_page"],
                                "bbox": bbox,
                                "margin_notes": sorted(
                                    set(pending_prose["margin_notes"])
                                ),
                            },
                        }
                    )
                pending_prose = None

        for pno in range(5, 477):
            page = doc[pno - 1]
            chap = page_struct[pno]["chapter"]
            chap_title = page_struct[pno]["chapter_title"]
            left_margin = 94.91 if pno % 2 == 1 else 133.77
            right_margin = 476.48 if pno % 2 == 1 else 515.34
            excl_zones = page_exclusion_zones.get(pno, [])

            footnote_y = 704.0
            for dr in page.get_drawings():
                r = dr["rect"]
                if (
                    abs(r.y1 - r.y0) < 1.5
                    and abs(r.x0 - left_margin) < 6.0
                    and 100.0 < (r.x1 - r.x0) < 200.0
                    and r.y0 > 500.0
                ):
                    footnote_y = min(footnote_y, float(r.y0))

            d = page.get_text("dict")
            page_margin_notes: List[str] = []

            raw_lines: List[Dict[str, Any]] = []
            for b_idx, b in enumerate(d.get("blocks", [])):
                if "lines" not in b:
                    continue
                for l in b["lines"]:
                    lx0, ly0, lx1, ly1 = [float(v) for v in l["bbox"]]
                    if ly0 < 120.0 or ly1 > 704.0:
                        continue
                    cx, cy = (lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0
                    if any(
                        ex0 - 1.5 <= cx <= ex1 + 1.5 and ey0 - 1.5 <= cy <= ey1 + 1.5
                        for ex0, ey0, ex1, ey1 in excl_zones
                    ):
                        continue

                    main_spans = []
                    for s in l["spans"]:
                        if not s["text"].strip():
                            continue
                        if _is_outer_margin_span(
                            pno, s["bbox"], s["font"], float(s["size"])
                        ):
                            page_margin_notes.append(s["text"].strip())
                        else:
                            main_spans.append(s)

                    if not main_spans:
                        continue
                    sx0 = min(float(s["bbox"][0]) for s in main_spans)
                    sy0 = min(float(s["bbox"][1]) for s in main_spans)
                    sx1 = max(float(s["bbox"][2]) for s in main_spans)
                    sy1 = max(float(s["bbox"][3]) for s in main_spans)
                    line_text = "".join(s["text"] for s in main_spans).strip()
                    if not line_text:
                        continue
                    if re.search(
                        r"Draft:\s*please send errata|Creative Commons CC-BY-NC-ND",
                        line_text,
                    ):
                        continue
                    raw_lines.append(
                        {
                            "block_idx": b_idx,
                            "bbox": [sx0, sy0, sx1, sy1],
                            "spans": main_spans,
                            "text": line_text,
                            "is_footnote": sy0 >= footnote_y - 1.0,
                        }
                    )

            raw_lines.sort(key=lambda item: (round(item["bbox"][1], 1), item["bbox"][0]))

            visual_rows: List[Dict[str, Any]] = []
            for rl in raw_lines:
                rx0, ry0, rx1, ry1 = rl["bbox"]
                rcy = (ry0 + ry1) / 2.0
                if (
                    visual_rows
                    and rl["is_footnote"] == visual_rows[-1]["is_footnote"]
                    and abs(rcy - visual_rows[-1]["cy"]) <= 5.5
                ):
                    prev = visual_rows[-1]
                    prev["bbox"] = _bbox_union([prev["bbox"], [rx0, ry0, rx1, ry1]])
                    prev["spans"].extend(rl["spans"])
                    prev["lines"].append(rl)
                else:
                    visual_rows.append(
                        {
                            "bbox": [rx0, ry0, rx1, ry1],
                            "cy": rcy,
                            "spans": list(rl["spans"]),
                            "lines": [rl],
                            "is_footnote": rl["is_footnote"],
                        }
                    )

            last_prose_snippet = ""
            for visual_row in visual_rows:
                vx0, vy0, vx1, vy1 = visual_row["bbox"]
                visual_row["pinned_display_math"] = any(
                    vx0 >= x0 - 1.0 and vx1 <= x1 + 1.0
                    and vy0 >= y0 and vy1 <= y1
                    for x0, y0, x1, y1 in DISPLAY_MATH_REGIONS.get(pno, [])
                )
                visual_row["pinned_inline_math"] = any(
                    vy0 >= y0 and vy1 <= y1
                    for y0, y1 in INLINE_MATH_FRAGMENTS.get(pno, [])
                )
            idx_row = 0
            while idx_row < len(visual_rows):
                row = visual_rows[idx_row]
                rx0, ry0, rx1, ry1 = row["bbox"]
                spans = sorted(
                    row["spans"],
                    key=lambda s: (round(float(s["bbox"][0]), 1), float(s["bbox"][1])),
                )
                row_text = " ".join(
                    "".join(s["text"] for s in l["spans"]).strip()
                    for l in sorted(row["lines"], key=lambda l: l["bbox"][0])
                ).strip()
                s0 = spans[0]
                font0 = s0["font"]
                size0 = round(float(s0["size"]), 2)

                # 1. Check for Chapter / Appendix / Section / Subsection headings
                if "LMSans10-Bold" in font0 and size0 >= 10.5:
                    if size0 in (17.22, 20.66):
                        flush_problem()
                        flush_prose()
                        in_problems_section = False
                        active_section = None
                        active_subsection = None
                        idx_row += 1
                        continue
                    if abs(size0 - 11.95) < 0.35:
                        flush_problem()
                        flush_prose()
                        if row_text == "Problems":
                            in_problems_section = True
                            active_section = "Problems"
                            active_subsection = None
                        else:
                            in_problems_section = False
                            sec_m = re.match(r"^((?:\d+|[ABC])\.\d+)\s+(.*)$", row_text)
                            if sec_m:
                                active_section = sec_m.group(1)
                                active_subsection = sec_m.group(2).strip()
                            else:
                                active_section = row_text
                                active_subsection = None
                        idx_row += 1
                        continue
                    if abs(size0 - 10.91) < 0.35:
                        flush_prose()
                        sub_m = re.match(
                            r"^((?:\d+|[ABC])\.\d+\.\d+)\s+(.*)$", row_text
                        )
                        if sub_m:
                            active_section = ".".join(sub_m.group(1).split(".")[:2])
                            active_subsection = (
                                f"{sub_m.group(1)} {sub_m.group(2).strip()}"
                            )
                        else:
                            active_subsection = row_text
                        idx_row += 1
                        continue

                # 2. Inside Problems section
                if in_problems_section:
                    prob_m = (
                        re.match(r"^Problem\s+(\d+\.\d+)(\*?)\b", row_text)
                        if "Bold" in font0
                        else None
                    )
                    if prob_m:
                        flush_problem()
                        prob_num = prob_m.group(1)
                        has_star = bool(prob_m.group(2))
                        current_problem = {
                            "problem_number": prob_num,
                            "has_asterisk": has_star,
                            "chapter": chap or prob_num.split(".")[0],
                            "pdf_page": pno,
                            "end_pdf_page": pno,
                            "parts": [row_text],
                            "raw_parts": [row_text],
                            "bboxes": [[rx0, ry0, rx1, ry1]],
                            "equations": [],
                            "margin_notes": list(page_margin_notes),
                        }
                    elif current_problem is not None:
                        current_problem["parts"].append(row_text)
                        current_problem["raw_parts"].append(row_text)
                        current_problem["bboxes"].append([rx0, ry0, rx1, ry1])
                        current_problem["end_pdf_page"] = pno

                    eq_tag_m = re.search(r"\(((\d+|[ABC])\.\d+)\)\s*$", row_text)
                    if eq_tag_m and abs(rx1 - right_margin) < 15.0:
                        eq_num = eq_tag_m.group(1)
                        if current_problem is not None:
                            current_problem["equations"].append(eq_num)
                        eq_boxes = [[rx0, ry0, rx1, ry1]]
                        eq_glyph_rows = [row_text]
                        back_idx = idx_row - 1
                        while back_idx >= 0:
                            prev_r = visual_rows[back_idx]
                            prx0, pry0, prx1, pry1 = prev_r["bbox"]
                            ptxt = " ".join(
                                "".join(s["text"] for s in l["spans"]).strip()
                                for l in sorted(prev_r["lines"], key=lambda l: l["bbox"][0])
                            ).strip()
                            if (
                                (prx0 > left_margin + 20.0 or prev_r.get("pinned_display_math"))
                                and (ry0 - pry1) < 24.0
                                and not re.search(r"\(((\d+|[ABC])\.\d+)\)\s*$", ptxt)
                            ):
                                eq_boxes.insert(0, [prx0, pry0, prx1, pry1])
                                eq_glyph_rows.insert(0, ptxt)
                                ry0 = pry0
                                back_idx -= 1
                            else:
                                break
                        # Tags can sit in the middle of a multi-row display.
                        # Keep every following mathematical row up to prose or
                        # the next tag, including the fourth batch-norm row.
                        forward_idx = idx_row + 1
                        while forward_idx < len(visual_rows):
                            nxt_r = visual_rows[forward_idx]
                            nrx0, nry0, nrx1, nry1 = nxt_r["bbox"]
                            if nry0 > DISPLAY_MATH_BOTTOMS.get((pno, eq_num), float("inf")):
                                break
                            ntxt = " ".join(
                                "".join(s["text"] for s in l["spans"]).strip()
                                for l in sorted(nxt_r["lines"], key=lambda l: l["bbox"][0])
                            ).strip()
                            is_m, n_tag = _is_display_math_visual_row(
                                nxt_r, ntxt, left_margin, right_margin
                            )
                            if (is_m and n_tag is None and (nry0 - ry1) <= 16.0
                                    and not re.match(r"^where\b", ntxt, re.IGNORECASE)):
                                eq_boxes.append([nrx0, nry0, nrx1, nry1])
                                eq_glyph_rows.append(ntxt)
                                ry1 = max(ry1, nry1)
                                forward_idx += 1
                            else:
                                break

                        eq_bbox = _bbox_union(eq_boxes)
                        eq_bbox_full = [
                            round(left_margin, 2),
                            round(max(121.0, eq_bbox[1] - 4.0), 2),
                            round(right_margin, 2),
                            round(min(702.0, eq_bbox[3] + 4.0), 2),
                        ]
                        eq_seq += 1
                        slug = eq_num.replace(".", "_")
                        rel_asset = f"data/assets/equations/eq_{eq_seq:04d}_{slug}.png"
                        crop_sha = None
                        if render_assets:
                            crop_sha = _render_clip_png(
                                page, eq_bbox_full, project_root / rel_asset
                            )
                        elif (project_root / rel_asset).is_file():
                            crop_sha = _sha256_file(project_root / rel_asset)
                        raw_glyphs = "\n".join(eq_glyph_rows).strip()
                        ctx_before = (
                            _normalize_prose_text(
                                " ".join(current_problem["parts"][:-1])
                            )[-240:]
                            if current_problem
                            else ""
                        )
                        equation_chunks.append(
                            {
                                "chunk_id": f"book_eq_{slug}",
                                "source_id": "book",
                                "source_type": "book",
                                "content_type": "equation",
                                "content": f"Equation ({eq_num}): {raw_glyphs}",
                                "raw_content": raw_glyphs,
                                "metadata": {
                                    "chapter": chap or eq_num.split(".")[0],
                                    "section": "Problems",
                                    "pdf_page": pno,
                                    "book_page": _book_page_for_pdf_page(pno),
                                    "bbox": eq_bbox_full,
                                    "equation_number": eq_num,
                                    "problem_number": (
                                        current_problem["problem_number"]
                                        if current_problem
                                        else None
                                    ),
                                    "asset_path": rel_asset,
                                    "crop_sha256": crop_sha,
                                    "review_status": "pending",
                                    "raw_glyphs": raw_glyphs,
                                    "context_before": ctx_before,
                                },
                            }
                        )
                    idx_row += 1
                    continue

                # 3. Outside Problems: check if this row starts a Display Math Band
                is_math, first_tag = _is_display_math_visual_row(
                    row, row_text, left_margin, right_margin
                )
                if is_math:
                    # Collect the entire contiguous Display Math Band between prose rows
                    math_band: List[Tuple[Dict[str, Any], str, Optional[str]]] = [
                        (row, row_text, first_tag)
                    ]
                    j = idx_row + 1
                    while j < len(visual_rows):
                        nxt = visual_rows[j]
                        _nx0, ny0, _nx1, _ny1 = nxt["bbox"]
                        if ny0 - math_band[-1][0]["bbox"][3] > 24.0:
                            break
                        ntxt = " ".join(
                            "".join(s["text"] for s in l["spans"]).strip()
                            for l in sorted(nxt["lines"], key=lambda l: l["bbox"][0])
                        ).strip()
                        nxt_is_math, nxt_tag = _is_display_math_visual_row(
                            nxt, ntxt, left_margin, right_margin
                        )
                        if nxt_is_math:
                            math_band.append((nxt, ntxt, nxt_tag))
                            j += 1
                        else:
                            break

                    # Partition `math_band` into equation units based on equation tags `(X.Y)` in the band:
                    # - 0 tags in band -> 1 unnumbered display equation covering the entire band
                    # - 1 tag in band -> 1 numbered display equation covering the entire band (numerator + tag row + denominator/underbrace)
                    # - k >= 2 tags: divide between tag baselines, keeping rows
                    #   below each tag (matrix bottoms/fraction denominators)
                    #   with that equation rather than the following equation.
                    tag_indices = [
                        idx_b for idx_b, item in enumerate(math_band) if item[2] is not None
                    ]
                    units: List[Tuple[Optional[str], List[Tuple[Dict[str, Any], str]]]] = []
                    if not tag_indices:
                        units.append((None, [(r_item, t_item) for r_item, t_item, _ in math_band]))
                    elif len(tag_indices) == 1:
                        only_tag = math_band[tag_indices[0]][2]
                        units.append(
                            (only_tag, [(r_item, t_item) for r_item, t_item, _ in math_band])
                        )
                    else:
                        start_b = 0
                        for t_pos_idx, tag_b_idx in enumerate(tag_indices):
                            eq_tag_val = math_band[tag_b_idx][2]
                            end_b = len(math_band)
                            if t_pos_idx < len(tag_indices) - 1:
                                next_tag_idx = tag_indices[t_pos_idx + 1]
                                midpoint = (math_band[tag_b_idx][0]["cy"]
                                            + math_band[next_tag_idx][0]["cy"]) / 2.0
                                end_b = tag_b_idx + 1
                                while end_b < next_tag_idx and math_band[end_b][0]["cy"] <= midpoint:
                                    end_b += 1
                            units.append(
                                (
                                    eq_tag_val,
                                    [
                                        (r_item, t_item)
                                        for r_item, t_item, _ in math_band[start_b:end_b]
                                    ],
                                )
                            )
                            start_b = end_b

                    ctx_after = ""
                    if j < len(visual_rows):
                        ctx_after = _normalize_prose_text(
                            " ".join(
                                "".join(s["text"] for s in l["spans"]).strip()
                                for l in sorted(
                                    visual_rows[j]["lines"], key=lambda l: l["bbox"][0]
                                )
                            )
                        )[:240]

                    ctx_before = (
                        _normalize_prose_text(" ".join(pending_prose["parts"]))[-240:]
                        if pending_prose
                        else last_prose_snippet[-240:]
                    )

                    for eq_num, unit_items in units:
                        raw_bbox = _bbox_union([r_item["bbox"] for r_item, _ in unit_items])
                        eq_bbox_full = [
                            round(left_margin, 2),
                            round(max(121.0, raw_bbox[1] - 4.0), 2),
                            round(right_margin, 2),
                            round(min(702.0, raw_bbox[3] + 4.0), 2),
                        ]
                        raw_glyphs = "\n".join(t_item for _, t_item in unit_items).strip()
                        eq_seq += 1
                        if eq_num is not None:
                            slug = eq_num.replace(".", "_")
                            cid = f"book_eq_{slug}"
                            rel_asset = f"data/assets/equations/eq_{eq_seq:04d}_{slug}.png"
                            label_str = f"Equation ({eq_num})"
                        else:
                            slug = f"p{pno:03d}_{eq_seq:04d}"
                            cid = f"book_eq_unnumbered_{slug}"
                            rel_asset = (
                                f"data/assets/equations/eq_{eq_seq:04d}_unnumbered_p{pno:03d}.png"
                            )
                            label_str = (
                                f"Display equation on page {_book_page_for_pdf_page(pno) or pno}"
                            )

                        crop_sha = None
                        if render_assets:
                            crop_sha = _render_clip_png(
                                page, eq_bbox_full, project_root / rel_asset
                            )
                        elif (project_root / rel_asset).is_file():
                            crop_sha = _sha256_file(project_root / rel_asset)

                        content_str = f"{label_str}: {raw_glyphs}"
                        if ctx_before:
                            content_str += f"\nContext: {ctx_before}"

                        equation_chunks.append(
                            {
                                "chunk_id": cid,
                                "source_id": "book",
                                "source_type": "book",
                                "content_type": "equation",
                                "content": content_str,
                                "raw_content": raw_glyphs,
                                "metadata": {
                                    "chapter": chap
                                    or (eq_num.split(".")[0] if eq_num else None),
                                    "section": active_section,
                                    "subsection": active_subsection,
                                    "pdf_page": pno,
                                    "book_page": _book_page_for_pdf_page(pno),
                                    "bbox": eq_bbox_full,
                                    "equation_number": eq_num,
                                    "asset_path": rel_asset,
                                    "crop_sha256": crop_sha,
                                    "review_status": "pending",
                                    "raw_glyphs": raw_glyphs,
                                    "context_before": ctx_before,
                                    "context_after": ctx_after,
                                },
                            }
                        )
                        if pending_prose is not None:
                            pending_prose["parts"].append(f"[{label_str}: {raw_glyphs}]")
                            pending_prose["raw_parts"].append(raw_glyphs)
                            pending_prose["bboxes"].append(raw_bbox)
                            pending_prose["end_pdf_page"] = pno

                    idx_row = j
                    continue

                # 4. Regular prose row
                if pending_prose is None:
                    pending_prose = {
                        "chapter": chap,
                        "chapter_title": chap_title,
                        "section": active_section,
                        "subsection": active_subsection,
                        "pdf_page": pno,
                        "end_pdf_page": pno,
                        "parts": [row_text],
                        "raw_parts": [row_text],
                        "bboxes": [[rx0, ry0, rx1, ry1]],
                        "margin_notes": list(page_margin_notes),
                    }
                else:
                    prev_y1 = pending_prose["bboxes"][-1][3]
                    same_page = pending_prose["end_pdf_page"] == pno
                    cur_words = sum(len(p.split()) for p in pending_prose["parts"])
                    prev_text = pending_prose["parts"][-1].strip()
                    ends_sentence = bool(re.search(r"[.?!:;\u201d\"]\s*$", prev_text))
                    is_para_gap = same_page and (ry0 - prev_y1 > 14.5)

                    if (
                        (is_para_gap and cur_words >= 120 and ends_sentence)
                        or (cur_words >= 280 and ends_sentence)
                        or cur_words >= 380
                    ):
                        last_prose_snippet = _normalize_prose_text(
                            " ".join(pending_prose["parts"])
                        )
                        flush_prose()
                        pending_prose = {
                            "chapter": chap,
                            "chapter_title": chap_title,
                            "section": active_section,
                            "subsection": active_subsection,
                            "pdf_page": pno,
                            "end_pdf_page": pno,
                            "parts": [row_text],
                            "raw_parts": [row_text],
                            "bboxes": [[rx0, ry0, rx1, ry1]],
                            "margin_notes": list(page_margin_notes),
                        }
                    else:
                        pending_prose["parts"].append(row_text)
                        pending_prose["raw_parts"].append(row_text)
                        pending_prose["bboxes"].append([rx0, ry0, rx1, ry1])
                        pending_prose["end_pdf_page"] = pno
                        pending_prose["margin_notes"].extend(page_margin_notes)

                idx_row += 1

            if pending_prose is not None:
                last_txt = pending_prose["parts"][-1].strip()
                cur_words = sum(len(p.split()) for p in pending_prose["parts"])
                if (cur_words >= 140 and re.search(r"[.?!]\s*$", last_txt)) or cur_words >= 260:
                    last_prose_snippet = _normalize_prose_text(
                        " ".join(pending_prose["parts"])
                    )
                    flush_prose()

        flush_problem()
        flush_prose()

        figure_chunks: List[Dict[str, Any]] = []
        for fig in figure_records:
            fnum = fig["figure_number"]
            pno = fig["pdf_page"]
            slug = fnum.replace(".", "_")
            cid = f"book_figure_{slug}"
            caption = fig["caption"]
            figure_chunks.append(
                {
                    "chunk_id": cid,
                    "source_id": "book",
                    "source_type": "book",
                    "content_type": "figure",
                    "content": caption,
                    "raw_content": caption,
                    "metadata": {
                        "chapter": fig["chapter"],
                        "section": fig["section"],
                        "pdf_page": pno,
                        "book_page": fig["book_page"],
                        "bbox": fig["bbox"],
                        "caption_bbox": fig["caption_bbox"],
                        "union_bbox": fig["union_bbox"],
                        "figure_number": fnum,
                        "asset_path": fig["asset_path"],
                        "crop_sha256": fig["crop_sha256"],
                        "source_geometry": fig["source_geometry"],
                    },
                }
            )

        algorithm_chunks: List[Dict[str, Any]] = []
        for alg in algorithm_records:
            anum = alg["algorithm_number"]
            pno = alg["pdf_page"]
            slug = anum.replace(".", "_")
            cid = f"book_algorithm_{slug}"
            algorithm_chunks.append(
                {
                    "chunk_id": cid,
                    "source_id": "book",
                    "source_type": "book",
                    "content_type": "algorithm",
                    "content": alg["content"],
                    "raw_content": alg["content"],
                    "metadata": {
                        "chapter": alg["chapter"],
                        "section": alg["section"],
                        "pdf_page": pno,
                        "book_page": alg["book_page"],
                        "bbox": alg["bbox"],
                        "caption_bbox": alg["caption_bbox"],
                        "algorithm_number": anum,
                        "asset_path": alg["asset_path"],
                        "crop_sha256": alg["crop_sha256"],
                        "source_geometry": alg["source_geometry"],
                    },
                }
            )

        all_book_chunks = (
            prose_chunks
            + exercise_chunks
            + figure_chunks
            + algorithm_chunks
            + equation_chunks
        )
        all_book_chunks.sort(
            key=lambda c: (
                int(c["metadata"].get("pdf_page") or 0),
                float((c["metadata"].get("bbox") or [0, 0, 0, 0])[1]),
                float((c["metadata"].get("bbox") or [0, 0, 0, 0])[0]),
                c["chunk_id"],
            )
        )
        for i, ch in enumerate(all_book_chunks):
            source_page = int(ch["metadata"].get("pdf_page") or 0)
            if source_page in page_struct:
                ch["metadata"]["chapter_title"] = page_struct[source_page]["chapter_title"]
            ch["metadata"]["previous_id"] = (
                all_book_chunks[i - 1]["chunk_id"] if i > 0 else None
            )
            ch["metadata"]["next_id"] = (
                all_book_chunks[i + 1]["chunk_id"]
                if i < len(all_book_chunks) - 1
                else None
            )

        layout_manifest = {
            "manifest_version": LAYOUT_MANIFEST_VERSION,
            "corpus_version": CORPUS_VERSION,
            "pdf_sha256": BOOK_SHA256,
            "pdf_pages": len(doc),
            "book_page_offset": 14,
            "toc": toc_entries,
            "figure_count": len(figure_records),
            "algorithm_count": len(algorithm_records),
            "figures": figure_records,
            "algorithms": algorithm_records,
        }

        book_stats = {
            "total_book_chunks": len(all_book_chunks),
            "prose_chunks": len(prose_chunks),
            "exercise_chunks": len(exercise_chunks),
            "figure_chunks": len(figure_chunks),
            "algorithm_chunks": len(algorithm_chunks),
            "equation_chunks": len(equation_chunks),
            "numbered_equations": sum(
                1
                for e in equation_chunks
                if e["metadata"].get("equation_number") is not None
            ),
            "unnumbered_equations": sum(
                1
                for e in equation_chunks
                if e["metadata"].get("equation_number") is None
            ),
        }
        return all_book_chunks, layout_manifest, book_stats
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Lossless Notebook Extraction (69 chapter notebooks, zero dropped lines/cells)
# ---------------------------------------------------------------------------


def _sanitize_code_for_ast(code: str) -> str:
    """Replace Jupyter magic (`%...`) and shell (`!...`) lines with same-line-count comments for `ast.parse`."""
    out_lines: List[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out_lines.append("# " + line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _safe_ast_parse(code: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(_sanitize_code_for_ast(code))
        except IndentationError:
            return ast.parse(_sanitize_code_for_ast(textwrap.dedent(code)))


def _extract_node_metadata(
    node_code: str,
    *,
    parent_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract per-definition or per-interval AST symbols, calls, imports, signature, and docstring."""
    defined_symbols: List[str] = []
    symbol_names: List[str] = []
    called_symbols: List[str] = []
    imports: List[str] = []
    signature: Optional[str] = None
    docstring: Optional[str] = None

    if parent_class:
        symbol_names.append(parent_class)
        defined_symbols.append(f"class:{parent_class}")

    try:
        tree = _safe_ast_parse(node_code)

        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                name = getattr(n, "name", "")
                if name:
                    symbol_names.append(name)
                    defined_symbols.append(f"{kind}:{name}")
                if signature is None:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in n.args.args]
                        signature = f"def {n.name}({', '.join(args)}):"
                    elif isinstance(n, ast.ClassDef):
                        signature = f"class {n.name}:"
                    docstring = ast.get_docstring(n)

        if not symbol_names:
            for n in tree.body:
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if isinstance(target, ast.Name) and target.id:
                            defined_symbols.append(f"variable:{target.id}")
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    if n.target.id:
                        defined_symbols.append(f"variable:{n.target.id}")

        for walk_node in ast.walk(tree):
            if isinstance(walk_node, ast.Import):
                for alias in walk_node.names:
                    imports.append(alias.name)
            elif isinstance(walk_node, ast.ImportFrom):
                mod = walk_node.module or ""
                for alias in walk_node.names:
                    imports.append(f"{mod}.{alias.name}" if mod else alias.name)
            elif isinstance(walk_node, ast.Call):
                if isinstance(walk_node.func, ast.Name):
                    called_symbols.append(walk_node.func.id)
                elif isinstance(walk_node.func, ast.Attribute):
                    called_symbols.append(walk_node.func.attr)
    except Exception:
        pass

    def _dedup(seq: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in seq:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return {
        "symbol_names": _dedup(symbol_names),
        "defined_symbols": _dedup(defined_symbols),
        "called_symbols": _dedup(called_symbols),
        "imports": _dedup(imports),
        "signature": signature,
        "docstring": docstring,
    }


def _partition_code_cell_intervals(code: str) -> List[Dict[str, Any]]:
    """Partition a code cell's lines 1..N into contiguous, lossless 1-based intervals `[start_line, end_line]`.

    - Preserves exact raw source byte/character fidelity including trailing newlines (`splitlines(keepends=True)`).
    - Partitions top-level function/class definitions (`def` / `class`).
    - Further partitions oversized classes (>3000 chars) at method boundaries and oversized script blocks
      (>2200 chars) at top-level AST statement boundaries so no single code chunk overflows the embedding context.
    """
    lines = code.splitlines()
    lines_keepends = code.splitlines(keepends=True)
    n_lines = max(1, len(lines_keepends))

    def _slice_raw(s_line: int, e_line: int) -> str:
        if not lines_keepends:
            return code
        return "".join(lines_keepends[s_line - 1 : e_line])

    def _span_chars(s_line: int, e_line: int) -> int:
        if not lines_keepends:
            return len(code)
        return sum(
            len(lines_keepends[k])
            for k in range(max(0, s_line - 1), min(len(lines_keepends), e_line))
        )

    try:
        tree = _safe_ast_parse(code)
    except Exception:
        meta = _extract_node_metadata(code)
        return [
            {
                "kind": "script",
                "name": None,
                "start_line": 1,
                "end_line": n_lines,
                "raw_content": code,
                **meta,
            }
        ]

    raw_blocks: List[Dict[str, Any]] = []
    for node in tree.body:
        node_start = min(
            [int(node.lineno)]
            + [int(dec.lineno) for dec in getattr(node, "decorator_list", [])]
        )
        node_end = int(getattr(node, "end_lineno", node.lineno))

        if isinstance(node, ast.ClassDef):
            cls_name = getattr(node, "name", None)
            cls_methods = [
                m
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if _span_chars(node_start, node_end) > 3000 and len(cls_methods) >= 2:
                # Partition large class at method boundaries (~2800 chars per group)
                group_start = node_start
                group_end = int(
                    getattr(cls_methods[0], "end_lineno", cls_methods[0].lineno)
                )
                group_name = f"{cls_name}_{cls_methods[0].name}" if cls_name else cls_methods[0].name
                for m in cls_methods[1:]:
                    m_start = min(
                        [int(m.lineno)]
                        + [int(dec.lineno) for dec in getattr(m, "decorator_list", [])]
                    )
                    m_end = int(getattr(m, "end_lineno", m.lineno))
                    if _span_chars(group_start, m_end) <= 2800:
                        group_end = m_end
                    else:
                        raw_blocks.append(
                            {
                                "kind": "def",
                                "name": group_name,
                                "parent_class": cls_name,
                                "ast_start": group_start,
                                "ast_end": group_end,
                            }
                        )
                        group_start = m_start
                        group_end = m_end
                        group_name = f"{cls_name}_{m.name}" if cls_name else m.name
                raw_blocks.append(
                    {
                        "kind": "def",
                        "name": group_name,
                        "parent_class": cls_name,
                        "ast_start": group_start,
                        "ast_end": max(group_end, node_end),
                    }
                )
            else:
                raw_blocks.append(
                    {
                        "kind": "def",
                        "name": cls_name,
                        "parent_class": None,
                        "ast_start": node_start,
                        "ast_end": node_end,
                    }
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raw_blocks.append(
                {
                    "kind": "def",
                    "name": getattr(node, "name", None),
                    "parent_class": None,
                    "ast_start": node_start,
                    "ast_end": node_end,
                }
            )
        else:
            stmt_name: Optional[str] = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                stmt_name = node.targets[0].id

            if raw_blocks and raw_blocks[-1]["kind"] == "script":
                prev_start = raw_blocks[-1]["ast_start"]
                if _span_chars(prev_start, node_end) <= 2200:
                    raw_blocks[-1]["ast_end"] = max(raw_blocks[-1]["ast_end"], node_end)
                else:
                    raw_blocks.append(
                        {
                            "kind": "script",
                            "name": stmt_name,
                            "parent_class": None,
                            "ast_start": node_start,
                            "ast_end": node_end,
                        }
                    )
            else:
                raw_blocks.append(
                    {
                        "kind": "script",
                        "name": stmt_name,
                        "parent_class": None,
                        "ast_start": node_start,
                        "ast_end": node_end,
                    }
                )

    if len(raw_blocks) <= 1:
        b0 = (
            raw_blocks[0]
            if raw_blocks
            else {"kind": "script", "name": None, "parent_class": None}
        )
        block_name = b0["name"] if b0["kind"] == "def" else None
        meta = _extract_node_metadata(code, parent_class=b0.get("parent_class"))
        return [
            {
                "kind": b0["kind"],
                "name": block_name,
                "start_line": 1,
                "end_line": n_lines,
                "raw_content": _slice_raw(1, n_lines),
                **meta,
            }
        ]

    boundaries: List[Tuple[int, int]] = []
    cur_start = 1
    for i in range(len(raw_blocks)):
        if i == len(raw_blocks) - 1:
            boundaries.append((cur_start, n_lines))
        else:
            prev_ast_end = raw_blocks[i]["ast_end"]
            next_ast_start = raw_blocks[i + 1]["ast_start"]
            split_before = next_ast_start
            while split_before - 1 > prev_ast_end:
                candidate_line = lines[split_before - 2].strip()
                if candidate_line.startswith("#"):
                    split_before -= 1
                else:
                    break
            end_i = max(cur_start, split_before - 1)
            boundaries.append((cur_start, end_i))
            cur_start = end_i + 1

    intervals: List[Dict[str, Any]] = []
    for blk, (s_line, e_line) in zip(raw_blocks, boundaries):
        slice_code = _slice_raw(s_line, e_line)
        meta = _extract_node_metadata(
            slice_code, parent_class=blk.get("parent_class")
        )
        intervals.append(
            {
                "kind": blk["kind"],
                "name": blk["name"],
                "start_line": s_line,
                "end_line": e_line,
                "raw_content": slice_code,
                **meta,
            }
        )
    return intervals


def _extract_markdown_headers(
    md: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract clean H1, H2, H3 headings from a markdown cell."""

    def _clean(h: str) -> str:
        t = re.sub(r"<[^>]+>", "", h)
        t = re.sub(r"[*_`]+", "", t)
        return re.sub(r"\s+", " ", t).strip()

    h1 = re.findall(r"^\s*#\s+(.+?)\s*$", md, flags=re.MULTILINE)
    h2 = re.findall(r"^\s*##\s+(.+?)\s*$", md, flags=re.MULTILINE)
    h3 = re.findall(r"^\s*###\s+(.+?)\s*$", md, flags=re.MULTILINE)
    h1_html = re.findall(r"<h1[^>]*>(.+?)</h1>", md, flags=re.IGNORECASE | re.DOTALL)
    h2_html = re.findall(r"<h2[^>]*>(.+?)</h2>", md, flags=re.IGNORECASE | re.DOTALL)
    h3_html = re.findall(r"<h3[^>]*>(.+?)</h3>", md, flags=re.IGNORECASE | re.DOTALL)

    h1_all = h1 + h1_html
    h2_all = h2 + h2_html
    h3_all = h3 + h3_html
    return (
        _clean(h1_all[-1]) if h1_all else None,
        _clean(h2_all[-1]) if h2_all else None,
        _clean(h3_all[-1]) if h3_all else None,
    )


def _sanitize_markdown_base64(
    md: str,
    nb_slug: str,
    cell_idx: int,
    project_root: Path,
    *,
    render_assets: bool = True,
) -> Tuple[str, List[Dict[str, str]]]:
    """Strip inline `data:image/...;base64,...` payloads from normalized markdown retrieval text.

    Preserves `raw_content` verbatim on the caller side, saves decoded image assets under
    `data/assets/notebooks/`, and replaces the inline tag in `content` with concise alt text.
    """
    if "base64," not in md and "data:image" not in md:
        return md, []

    saved_assets: List[Dict[str, str]] = []
    img_counter = 0

    def _save_b64(mime_sub: str, b64_payload: str, alt_label: str) -> str:
        nonlocal img_counter
        img_counter += 1
        sub_lower = mime_sub.lower()
        if "svg" in sub_lower:
            ext = "svg"
            fmt_label = "SVG"
        elif "png" in sub_lower:
            ext = "png"
            fmt_label = "PNG"
        elif "jpeg" in sub_lower or "jpg" in sub_lower:
            ext = "jpg"
            fmt_label = "JPEG"
        else:
            ext = "bin"
            fmt_label = mime_sub.upper()

        rel_asset = f"data/assets/notebooks/nb_{nb_slug}_c{cell_idx:02d}_img{img_counter}.{ext}"
        dest = project_root / rel_asset
        asset_sha = ""
        try:
            raw_bytes = base64.b64decode(re.sub(r"\s+", "", b64_payload))
            asset_sha = hashlib.sha256(raw_bytes).hexdigest()
            if render_assets:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw_bytes)
        except Exception:
            pass
        saved_assets.append(
            {
                "asset_path": rel_asset,
                "crop_sha256": asset_sha,
                "format": fmt_label,
                "alt": alt_label,
            }
        )
        if alt_label:
            return f"[Embedded {fmt_label} image: {alt_label}]"
        return f"[Embedded {fmt_label} image]"

    # 1. HTML <img ... src="data:image/...;base64,..." ...>
    html_img_re = re.compile(
        r"<img\b([^>]*?)src=[\"']data:image/([^;\"']+);base64,([A-Za-z0-9+/=\s]+)[\"']([^>]*)>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _repl_html(match: re.Match[str]) -> str:
        attrs = (match.group(1) or "") + " " + (match.group(4) or "")
        mime_sub = match.group(2)
        b64_data = match.group(3)
        alt_m = re.search(r"alt=[\"']([^\"']*)[\"']", attrs, flags=re.IGNORECASE)
        alt_txt = alt_m.group(1).strip() if alt_m else ""
        return _save_b64(mime_sub, b64_data, alt_txt)

    cleaned = html_img_re.sub(_repl_html, md)

    # 2. Markdown ![alt](data:image/...;base64,...)
    md_img_re = re.compile(
        r"!\[([^\]]*)\]\(\s*data:image/([^;)]+);base64,([A-Za-z0-9+/=\s]+)\)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _repl_md(match: re.Match[str]) -> str:
        alt_txt = (match.group(1) or "").strip()
        mime_sub = match.group(2)
        b64_data = match.group(3)
        return _save_b64(mime_sub, b64_data, alt_txt)

    cleaned = md_img_re.sub(_repl_md, cleaned)
    return cleaned, saved_assets


def _is_student_template(text: str) -> bool:
    """Return True if a notebook cell/interval contains unfinished student assignment TODO markers."""
    return bool(
        re.search(
            r"\bTODO\b|\bTo\s+Do\b|Replace\s+this\s+line|YOUR_CODE_HERE",
            text,
            flags=re.IGNORECASE,
        )
    )


def _extract_single_notebook(
    nb_path: Path,
    notebooks_root: Path,
    project_root: Path,
    *,
    render_assets: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract source record and lossless cell/interval chunks from a single Jupyter notebook."""
    rel_source_id = nb_path.relative_to(notebooks_root).as_posix()
    rel_project_path = nb_path.relative_to(project_root).as_posix()
    file_sha256 = _sha256_file(nb_path)

    chap_m = re.search(r"Chap0*(\d+)", rel_source_id, flags=re.IGNORECASE)
    chapter_str = str(int(chap_m.group(1))) if chap_m else None
    nb_slug = nb_path.stem
    human_title = nb_slug.replace("_", " ").replace("-", " ")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None

    accumulated_import_lines: List[str] = []
    prior_symbol_defs: Dict[str, str] = {}

    nb_chunks: List[Dict[str, Any]] = []

    for cell_idx, cell in enumerate(nb.cells):
        raw_cell_source = (
            "".join(cell.source)
            if isinstance(cell.source, list)
            else str(cell.source)
        )
        if not raw_cell_source.strip():
            continue

        cell_lines = raw_cell_source.splitlines()
        n_lines = max(1, len(cell_lines))

        if cell.cell_type == "markdown":
            h1, h2, h3 = _extract_markdown_headers(raw_cell_source)
            if h1:
                section_h1, section_h2, section_h3 = h1, None, None
            if h2:
                section_h2, section_h3 = h2, None
            if h3:
                section_h3 = h3

            active_sec = section_h3 or section_h2 or section_h1 or human_title
            cid = f"nb_{nb_slug}_c{cell_idx:02d}_L1_{n_lines}"
            is_tmpl = _is_student_template(raw_cell_source)
            clean_md, embedded_assets = _sanitize_markdown_base64(
                raw_cell_source,
                nb_slug,
                cell_idx,
                project_root,
                render_assets=render_assets,
            )

            md_meta: Dict[str, Any] = {
                "chapter": chapter_str,
                "chapter_id": chapter_str,
                "section": active_sec,
                "section_h1": section_h1,
                "section_h2": section_h2,
                "section_h3": section_h3,
                "notebook_path": rel_source_id,
                "notebook_slug": nb_slug,
                "notebook_title": section_h1 or human_title,
                "cell_index": cell_idx,
                "start_line": 1,
                "end_line": n_lines,
                "cell_line_count": n_lines,
                "symbol_names": [],
                "is_template": is_tmpl,
            }
            if embedded_assets:
                md_meta["asset_path"] = embedded_assets[0]["asset_path"]
                md_meta["crop_sha256"] = embedded_assets[0]["crop_sha256"]
                md_meta["embedded_assets"] = embedded_assets

            nb_chunks.append(
                {
                    "chunk_id": cid,
                    "source_id": rel_source_id,
                    "source_type": "notebook",
                    "content_type": "markdown",
                    "content": clean_md,
                    "raw_content": raw_cell_source,
                    "metadata": md_meta,
                }
            )

        elif cell.cell_type == "code":
            intervals = _partition_code_cell_intervals(raw_cell_source)
            active_sec = section_h3 or section_h2 or section_h1 or human_title
            prior_imports_snapshot = list(accumulated_import_lines)

            for iv_idx, iv in enumerate(intervals):
                s_line = int(iv["start_line"])
                e_line = int(iv["end_line"])
                iv_raw = iv["raw_content"]
                syms = iv["symbol_names"]
                calls = iv["called_symbols"]
                imps = iv["imports"]
                sig = iv["signature"]
                doc_str = iv["docstring"]
                is_tmpl = _is_student_template(iv_raw)

                dep_defs = [
                    prior_symbol_defs[c_sym]
                    for c_sym in calls
                    if c_sym in prior_symbol_defs and c_sym not in syms
                ]

                name_suffix = (
                    f"_{iv['name']}"
                    if iv.get("name")
                    else (f"_part{iv_idx + 1}" if len(intervals) > 1 else "")
                )
                cid = f"nb_{nb_slug}_c{cell_idx:02d}_L{s_line}_{e_line}{name_suffix}"

                emb_parts = [
                    f"Notebook {rel_source_id} (Chapter {chapter_str}) | Section: {active_sec}",
                ]
                if syms:
                    emb_parts.append(f"Symbols: {', '.join(syms)}")
                if sig:
                    emb_parts.append(f"Signature: {sig}")
                if prior_imports_snapshot and not imps:
                    emb_parts.append(
                        "Imports: " + "; ".join(prior_imports_snapshot[:8])
                    )
                emb_parts.append(iv_raw)

                nb_chunks.append(
                    {
                        "chunk_id": cid,
                        "source_id": rel_source_id,
                        "source_type": "notebook",
                        "content_type": "code",
                        "content": iv_raw,
                        "raw_content": iv_raw,
                        "embedding_text": "\n".join(emb_parts),
                        "metadata": {
                            "chapter": chapter_str,
                            "chapter_id": chapter_str,
                            "section": active_sec,
                            "section_h1": section_h1,
                            "section_h2": section_h2,
                            "section_h3": section_h3,
                            "notebook_path": rel_source_id,
                            "notebook_slug": nb_slug,
                            "notebook_title": section_h1 or human_title,
                            "cell_index": cell_idx,
                            "start_line": s_line,
                            "end_line": e_line,
                            "cell_line_count": n_lines,
                            "interval_kind": iv["kind"],
                            "symbol_names": syms,
                            "defined_symbols": iv["defined_symbols"],
                            "called_symbols": calls,
                            "imports": imps,
                            "signature": sig,
                            "docstring": doc_str,
                            "prior_imports": prior_imports_snapshot,
                            "dependency_context": dep_defs,
                            "is_template": is_tmpl,
                        },
                    }
                )

                for sym_name in syms:
                    if sym_name:
                        summary_line = sig or f"def {sym_name}(...)"
                        if doc_str:
                            summary_line += f"  # {doc_str.splitlines()[0]}"
                        prior_symbol_defs[sym_name] = summary_line

            for ln in cell_lines:
                s_ln = ln.strip()
                if (
                    s_ln.startswith(("import ", "from "))
                    and s_ln not in accumulated_import_lines
                ):
                    accumulated_import_lines.append(s_ln)

    for i, ch in enumerate(nb_chunks):
        ch["metadata"]["previous_id"] = nb_chunks[i - 1]["chunk_id"] if i > 0 else None
        ch["metadata"]["next_id"] = (
            nb_chunks[i + 1]["chunk_id"] if i < len(nb_chunks) - 1 else None
        )

    source_record = {
        "source_id": rel_source_id,
        "source_type": "notebook",
        "path": rel_project_path,
        "sha256": file_sha256,
        "revision": NOTEBOOK_COMMIT,
        "title": section_h1 or human_title,
        "metadata": {
            **NOTEBOOK_LICENSE_METADATA,
            "chapter": chapter_str,
            "notebook_slug": nb_slug,
            "cell_count": len(nb.cells),
            "non_empty_cell_count": sum(
                1
                for c in nb.cells
                if (
                    "".join(c.source)
                    if isinstance(c.source, list)
                    else str(c.source)
                ).strip()
            ),
            "chunk_count": len(nb_chunks),
        },
    }
    return source_record, nb_chunks


def extract_corpus(
    project_root: Union[str, Path] = PROJECT_ROOT,
    render_assets: bool = True,
) -> CorpusResult:
    """Extract the full pinned UDL corpus (book PDF + 69 chapter notebooks) into `CorpusResult`."""
    root = Path(project_root).resolve()
    ensure_sources(root, offline=True)

    pdf_path = root / "data" / "raw" / "UnderstandingDeepLearning.pdf"
    nb_repo_dir = root / "data" / "raw" / "udlbook_notebooks"
    nb_files = _find_chapter_notebooks(nb_repo_dir)
    if len(nb_files) != 69:
        raise UDLException(
            "source_incomplete",
            f"Expected 69 chapter notebooks under {nb_repo_dir / 'Notebooks'}, found {len(nb_files)}.",
            retryable=False,
        )

    book_source = {
        "source_id": "book",
        "source_type": "book",
        "path": pdf_path.relative_to(root).as_posix(),
        "sha256": BOOK_SHA256,
        "revision": CORPUS_VERSION,
        "title": "Understanding Deep Learning",
        "metadata": {
            **BOOK_LICENSE_METADATA,
            "pdf_pages": 541,
        },
    }

    book_chunks, layout_manifest, book_stats = _extract_book_pdf(
        pdf_path, root, render_assets=render_assets
    )

    layout_path = root / "resources" / "book_layout.json"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(
        json.dumps(layout_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sources: List[Dict[str, Any]] = [book_source]
    notebook_chunks: List[Dict[str, Any]] = []
    for nb_file in nb_files:
        src_rec, ch_list = _extract_single_notebook(
            nb_file, nb_repo_dir, root, render_assets=render_assets
        )
        sources.append(src_rec)
        notebook_chunks.extend(ch_list)

    all_chunks = book_chunks + notebook_chunks
    type_counts = Counter(c["content_type"] for c in all_chunks)

    report: Dict[str, Any] = {
        "corpus_version": CORPUS_VERSION,
        "book_sha256": BOOK_SHA256,
        "notebook_commit": NOTEBOOK_COMMIT,
        "total_sources": len(sources),
        "book_sources": 1,
        "notebook_sources": len(nb_files),
        "total_chunks": len(all_chunks),
        "chunks_by_content_type": dict(type_counts),
        "book_stats": book_stats,
        "notebook_stats": {
            "total_notebook_chunks": len(notebook_chunks),
            "markdown_chunks": type_counts.get("markdown", 0),
            "code_chunks": type_counts.get("code", 0),
            "template_chunks": sum(
                1 for c in notebook_chunks if c["metadata"].get("is_template")
            ),
        },
        "render_assets": bool(render_assets),
    }

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "corpus_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    equation_inventory = [
        {
            "chunk_id": c["chunk_id"],
            "equation_number": c["metadata"].get("equation_number"),
            "pdf_page": c["metadata"].get("pdf_page"),
            "book_page": c["metadata"].get("book_page"),
            "chapter": c["metadata"].get("chapter"),
            "section": c["metadata"].get("section"),
            "bbox": c["metadata"].get("bbox"),
            "asset_path": c["metadata"].get("asset_path"),
            "crop_sha256": c["metadata"].get("crop_sha256"),
            "review_status": c["metadata"].get("review_status"),
            "raw_glyphs": c["metadata"].get("raw_glyphs"),
            "context_before": c["metadata"].get("context_before"),
            "context_after": c["metadata"].get("context_after"),
        }
        for c in book_chunks
        if c["content_type"] == "equation"
    ]
    (data_dir / "equation_review_manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": CORPUS_VERSION,
                "book_sha256": BOOK_SHA256,
                "total_equations": len(equation_inventory),
                "numbered_equations": book_stats["numbered_equations"],
                "unnumbered_equations": book_stats["unnumbered_equations"],
                "equations": equation_inventory,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return CorpusResult(sources=sources, chunks=all_chunks, report=report)
