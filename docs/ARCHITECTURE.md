# Architecture

The CLI and browser share `Runtime`, `SQLiteStore`, `HybridRetriever`, and `TutorService`. There is no independent web-only retrieval path.

## 1. Acquire and extract

`corpus.py` verifies the v5.0.2 book with SHA-256 and the official notebook repository at a fixed commit. Downloads are explicit build operations. Only `Notebooks/Chap*` contributes notebook evidence.

PDF extraction preserves page and region provenance, repairs prose reading order, separates exercises and display mathematics, and renders source crops through PyMuPDF. Notebook extraction uses AST boundaries where possible, preserves lossless cell spans, and distinguishes definitions, setup statements, Markdown, imports, and unfinished templates. It never executes cells.

The generated layout manifest is stored locally and ignored by Git. Mathematical transcription reviews bind source, crop, and transcription hashes. Changed bytes invalidate approval. The public release does not ship the development transcription collection; pending mathematics stays outside trusted generation context.

## 2. Stage and activate in SQLite

`store.py` persists sources, chunks, lexical postings, float32 vector BLOBs, embedding profiles, mathematical reviews, and build history. WAL and explicit write coordination support local use. Builds stage changes and activate only after indexing succeeds. A failed build preserves the preceding active snapshot.

Stable identities distinguish source occurrences even when content is identical. Reuse requires matching embedding input and profile fingerprints. Immutable local source snapshots preserve historical citation targets. The store exposes backup and recovery operations; a live database must be backed up through SQLite's backup API, not copied while its WAL is changing.

## 3. Retrieve without a framework

`search.py` implements tokenization, BM25, exact cosine similarity, and reciprocal rank fusion. NumPy performs in-memory numerical work; SQLite remains the persistent vector store. The BM25 index is also handwritten rather than delegated to an FTS ranking extension.

Explicit equation/figure/problem references and source-defined symbols receive dedicated checks. Compound book-and-notebook questions can use independently scoped evidence pools. Chapter, modality, notebook, source relevance, and bounded source continuations influence selection. Original BM25/cosine evidence drives abstention; RRF scores are not probabilities.

`dense.py` uses direct Transformers/PyTorch inference with a pinned `Qwen/Qwen3-Embedding-0.6B` revision, the documented query instruction, last-valid-token pooling, and L2 normalization. Long documents use overlapping windows and aggregation. Oversized queries fail explicitly. Microbatch splitting and CPU fallback preserve the declared numerical profile; inference is serialized inside each embedder instance.

## 4. Assemble and answer

`tutor.py` assembles complete evidence units inside a character budget and includes bounded history. Source excerpts and prior messages are treated as data, not instructions. The selected direct HTTP adapter calls Groq, Gemini, an OpenAI-compatible endpoint, or native Ollama.

Returned source labels must resolve to the current included context. One citation-repair attempt is allowed; a second failure produces an abstention. Derived solutions and generated code are labeled. This is label validation, not a semantic proof of every claim.

## 5. Inspect through the UI

`web.py` serves a localhost-only FastAPI application with same-origin endpoints, allowlisted source/asset access, redacted errors, and SSE generation. The buildless frontend uses local copies of DOMPurify, Marked, KaTeX, and Prism. Browser keys and history remain in memory; localStorage, sessionStorage, and cookies are not used for persistence.

Public routes include `/api/status`, `/api/search`, `/api/ask`, chunk detail, original sources, crops, and review operations. The server is a local single-process study application, not a hosted multi-user service with authentication or tenant isolation.
