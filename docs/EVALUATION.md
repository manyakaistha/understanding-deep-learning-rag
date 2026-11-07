# Evaluation: evidence over a headline score

This beta separates extraction, retrieval, citation identity, and answer quality. Passing one does not establish the others.

## Reference experiment

The source-aware suite has 100 questions: 90 supported and 10 unsupported. The development split contains 54 supported and 6 negative cases; the broader split contains 36 supported and 4 negatives. Questions cover theory, equations, figures, code, exercises, and mixed book/notebook requests.

A supported Hit@5 requires all declared evidence groups. Targets have exact PDF pages or notebook cells plus source anchors. Multiple valid alternatives can be represented, but a chapter-title match is insufficient. Citation resolution checks that a retrieved chunk and source belong to the reported build.

| Locally reviewed reference build | Development | Broader regression split |
| --- | ---: | ---: |
| Supported Hit@5 | 51/54 (94.44%) | 27/36 (75.00%) |
| Unsupported abstention | 6/6 (100%) | 4/4 (100%) |
| Citation resolution | 100% | 100% |

Broader category Hit@5: theory 50%, figures 50%, mixed 75%, equations 100%, notebook code 100%, exercises 100%. The full validation thresholds require overall supported Hit@5 ≥90%, every category ≥80%, negative abstention ≥90%, and citation resolution 100% with actual dense use.

**The release gate has not passed.** These are reference-build results with local mathematical reviews and vectors. Those artifacts are not distributed, so the numbers are not a claim about a fresh public checkout with pending math. The broader questions were inspected during debugging; subsequent runs are repeated regression checks, not an untouched estimate of generalization. No production-accuracy claim is made.

Reference benchmark SHA-256: `33145a730b34134a1b48e9318742e70e02b86c89adf190db53091940550edeff`.

Machine-readable aggregated evidence: [reference-results.json](evaluation/reference-results.json).

## Reproduce locally

```bash
uv sync --locked --extra semantic
uv run --extra semantic python -m src.cli build --semantic
uv run --extra semantic python -m src.cli evaluate --mode hybrid --split development \
  --output data/evaluation/development.json
uv run --extra semantic python -m src.cli evaluate --mode hybrid --split heldout \
  --output data/evaluation/release.json
uv run python -m src.cli status
```

Expect different results when mathematical regions remain pending. Locally review the exact source crops before approving transcriptions; never promote a generated draft just to raise a score. Keep failed results and use new independently authored cases for future generalization checks.

## Test layers

- **Offline fixtures:** tokenization, BM25/cosine/fusion, source scoping, incomplete evidence, vector fingerprints, OOM fallback, SQLite staging, snapshots, API behavior, and provider/citation contracts. No keys, book, or model weights required.
- **Real-source suites:** extraction geometry, lossless notebook coverage, and gold-source fidelity. These skip when the pinned data is absent and run after `build` downloads it.
- **Optional real encoder:** set `UDL_RUN_REAL_EMBEDDER_CHECK=1` with semantic dependencies and cached weights to run the CPU model check.
- **Frontend:** local vendor files, DOM contracts, sanitization and storage invariants, SSE parsing, and math/code preservation.
- **Live generation:** local Ollama `qwen3:8b` was manually checked. Groq, Gemini, and compatible adapters have fixture coverage, not a claim of live remote verification.

Citation-label repair catches invalid identifiers. It cannot determine semantic entailment of every sentence. The final answer model can still over-explain, omit details, or attach a valid citation to an unsupported claim. Always inspect source evidence.
