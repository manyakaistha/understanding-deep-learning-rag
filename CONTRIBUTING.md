# Contributing to Understanding Deep Learning RAG

Thank you for your interest in contributing to **Understanding Deep Learning RAG** (`understanding-deep-learning-rag`). This repository provides a local-first study tutor with handwritten hybrid retrieval (lexical BM25 and optional dense embeddings) and SQLite-backed provenance.

Please review these guidelines before submitting issues or pull requests.

---

## 1. Development Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic Python virtual environment and dependency management.

- **Prerequisites**: Python 3.10–3.13 (3.12 recommended) and `uv`. For running frontend contract tests, Node.js 22 is required.
- **Clone the repository**:
  ```bash
  git clone https://github.com/manyakaistha/understanding-deep-learning-rag.git
  cd understanding-deep-learning-rag
  ```
- **Install default dependencies**:
  ```bash
  uv sync --locked
  ```
  This creates a virtual environment and installs the core runtime and test dependencies (FastAPI, PyMuPDF, Pillow, nbformat, NumPy, uvicorn, etc.) without downloading heavy machine learning frameworks.

---

## 2. Optional Semantic Dependencies

Dense vector embedding functionality (e.g., PyTorch and Hugging Face Transformers) is entirely optional. The default pipeline runs in lexical-only mode.

To install semantic dense embedding dependencies:
```bash
uv sync --locked --extra semantic
```

When semantic dependencies are omitted, all lexical search, chunking, web serving, and unit test contracts remain fully functional.

---

## 3. Running Tests

Before submitting a pull request, ensure all tests pass locally:

- **Python unit and contract tests**:
  ```bash
  uv run python -B -m unittest discover -s tests -v
  ```
  *Note:* `tests/test_corpus.py` automatically skips the real-source extraction test class when raw book/notebook data is not present (`data/raw/`), preserving fixture coverage across local environments and CI without downloading external assets.

- **Lockfile consistency check**:
  ```bash
  uv lock --check
  ```

- **Frontend tests & syntax verification**:
  ```bash
  node src/static/tests/run_tests.js
  node --check src/static/app.js
  ```

---

## 4. Repository & Data Hygiene

To respect copyright, privacy, and repository cleanliness:

- **Never commit book PDFs or raw extracted datasets**: Do not add files under `data/` or any copies of Simon J.D. Prince's *Understanding Deep Learning* PDF to version control.
- **Never commit model weights or checkpoints**: Models should be fetched dynamically or cached locally outside git tracking.
- **Never commit secrets or API keys**: Keep LLM provider tokens, environment files (`.env`), and personal credentials out of git commits.
- **Clean diffs**: Do not introduce unnecessary tooling, telemetry, or secret scanners with paid/external requirements.

---

## 5. Evidence-Based Retrieval Improvements

Retrieval and ranking adjustments must be evidence-based:

- Ground proposed changes in evaluation benchmarks (e.g., `resources/evaluation.json`).
- Changes to tokenization, BM25 scoring, reciprocal rank fusion (RRF), or chunk windowing must provide measurable evidence of improved retrieval quality without causing regressions in citation fidelity or source provenance.
- All retrieved passages and citations must map strictly to authoritative chunks stored in SQLite.

---

## 6. Honest Beta & Preserving Release Gates

This project is in active beta (`0.1.0b1`):

- **Do not weaken release gates**: Do not relax test assertions, widen thresholds, or disable checks merely to pass CI or bypass failed edge cases.
- **Security & architectural invariants**:
  - The static frontend (`src/static/app.js`) strictly prohibits client-side persistence (`localStorage`, `sessionStorage`, or cookies).
  - All rendered markup must be sanitized via DOMPurify with KaTeX untrusted mode (`trust: false`).
  - All API routes must remain same-origin relative endpoints.

---

## 7. Sources and Licenses

Contributors must respect upstream intellectual property and project licensing:

- **Project Code**: Distributed under the [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.html) (`AGPL-3.0-or-later`).
- **Book Content**: *Understanding Deep Learning* by Simon J.D. Prince is copyright Simon J.D. Prince, distributed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Source book materials are not redistributed in this repository.
- **Accompanying Notebooks**: Notebook exercises derived from the [udlbook](https://github.com/udlbook/udlbook) project are licensed under the MIT License.
