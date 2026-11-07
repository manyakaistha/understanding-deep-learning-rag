# Sources and distribution boundaries

This independent application accompanies Simon J.D. Prince's *Understanding Deep Learning*, published by MIT Press. It does not replace the book and is not affiliated with its author or publisher.

## Pinned learning sources

- Book: **v5.0.2 (2025-05-29)**, obtained from the [official book project](https://udlbook.github.io/udlbook/).
- Book SHA-256: `9c0ea2e831e662a687d267dd9bb2d50ea743e58271e90b96a4fac4d585288e04`.
- Official notebooks: [udlbook/udlbook](https://github.com/udlbook/udlbook), commit `44a3e5f6786d57a3362d1480c9f6ca09f495aaa3`.
- Notebook scope: the 69 `Notebooks/Chap*` notebooks. Supplements and unrelated coursework are not retrieval sources.

Acquisition happens only when the user explicitly runs build/update. Sources are verified before use; a changed edition requires explicit parser/layout validation. Keep the downloaded source licenses with local data and follow their terms.

## What is public

The public repository contains the application engine, source-aware benchmark, tests, documentation, and frontend assets. It starts from a clean release history, separate from the private development archive.

It does **not** distribute the book PDF, rendered book pages/figures, extracted corpus, notebook checkout, model weights, SQLite index, private experiments, old source archives, credentials, or the development mathematical transcription bundle. The generated `resources/book_layout.json` and optional local `resources/book_math_reviews.json` are ignored.

A fresh build still extracts source regions and displays their original crops. Mathematical regions have pending review status. They are excluded from trusted answer context until a local transcription is compared with the source image and approved. This deliberately prevents a raw PDF glyph extraction from being presented as verified mathematics.

```bash
uv run python -m src.cli review --pending
uv run python -m src.cli review --chunk-id CHUNK_ID \
  --transcription-file equation.txt --reviewer YOUR_NAME --decision verified
```

Only approve after comparing the exact crop. Review changes invalidate affected vectors; rebuild before relying on semantic search. An index can be usable while corpus or retrieval release gates remain incomplete.

## Separate terms

| Material | Terms |
| --- | --- |
| Original application code | AGPL-3.0-or-later; see root `LICENSE`. |
| Book text and original figures | CC BY-NC-ND 4.0; copyright notice identifies MIT Press. The app does not relicense them. |
| Official chapter notebook code | MIT; Copyright 2023 Simon Prince. |
| Short book quotations in benchmark assertions | Retain the book attribution and original source terms; they are evidence anchors, not application code. |
| Screenshots | Application interface plus MIT-licensed notebook excerpts. Captured from a locally indexed development build. No book pages or figure crops are included in the published screenshots. |
| Frontend vendors and fonts | Their respective upstream licenses; see `NOTICE.md` and `docs/third-party/`. |
| Python dependencies / model weights | Their own upstream licenses. PyMuPDF uses AGPL or a commercial license; embedding weights are downloaded separately. |

The mathematical review data is not relicensed into a more permissive license. It is omitted from this release. No claim of permission to redistribute adapted book material is made.
