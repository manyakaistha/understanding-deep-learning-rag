"""Unit and integration tests for `src/search.py`."""

from pathlib import Path
import tempfile
import unittest

from src.contracts import CorpusResult, SearchRequest, UDLException
from src.dense import compute_profile_fingerprint, l2_normalize
from src.search import (
    AbstentionConfig,
    HybridRetriever,
    build_context_block,
    embedding_text,
    expand_neighbors,
    extract_query_anchors,
    get_chunk_neighbors,
    locator,
    rrf_fuse,
    tokenize,
)
from src.store import SQLiteStore


class _ControlledEmbedder:
    """Controlled test embedder for deterministic cosine and RRF unit testing."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.query_map = {}
        self.query_calls = []

    @property
    def profile(self):
        return {
            "backend": "torch+transformers",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "controlled_rev",
            "configured_revision": "controlled_rev",
            "dimension": self.dimension,
            "pooling": "last_valid_token",
            "normalize": "l2",
            "dtype": "float32",
            "model_dtype": "float32",
            "max_length": 4096,
            "overflow_policy": "sliding_window",
            "window_overlap": 128,
            "window_aggregation": "l2_normalized_mean",
            "query_overflow_policy": "reject",
            "query_instruction": "Given a web search query, retrieve relevant passages that answer the query",
            "query_prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            "document_prompt": "",
        }

    @property
    def fingerprint(self) -> str:
        return compute_profile_fingerprint(self.profile)

    def _vec_for_text(self, text: str):
        lower = text.lower()
        if "equation 7.12" in lower or "jacobian" in lower:
            return l2_normalize([0.0, 0.0, 0.1, 1.0])
        if "figure 18.4" in lower or "diffusion" in lower:
            return l2_normalize([0.0, 0.1, 1.0, 0.0])
        if "gradient_descent" in lower or "step_size" in lower:
            return l2_normalize([0.1, 1.0, 0.0, 0.0])
        if "problem 9.1" in lower or "regularization" in lower or "sigma" in lower:
            return l2_normalize([0.3, 0.8, 0.4, 0.1])
        if "problem 7.1" in lower or "three-layer" in lower:
            return l2_normalize([0.85, 0.45, 0.0, 0.0])
        if "backpropagation" in lower or "chain rule" in lower:
            return l2_normalize([1.0, 0.1, 0.0, 0.0])
        return l2_normalize([0.25, 0.25, 0.25, 0.25])

    def encode_documents(self, texts):
        return [self._vec_for_text(t) for t in texts]

    def encode_queries(self, texts):
        out = []
        for t in texts:
            self.query_calls.append(t)
            if t in self.query_map:
                out.append(l2_normalize(self.query_map[t]))
            else:
                out.append(self._vec_for_text(t))
        return out


def _build_fixture_corpus() -> CorpusResult:
    sources = [
        {
            "source_id": "book",
            "source_type": "pdf",
            "path": "data/raw/UnderstandingDeepLearning.pdf",
            "sha256": "book_sha",
            "revision": "v5.0.2",
            "title": "Understanding Deep Learning",
            "metadata": {},
        },
        {
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "path": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "sha256": "nb_sha",
            "revision": "44a3e5f",
            "title": "6.2 Gradient Descent",
            "metadata": {"chapter": "6"},
        },
    ]

    chunks = [
        {
            "chunk_id": "ch7_text_1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "text",
            "content": "Backpropagation applies the chain rule to compute derivatives of the loss with respect to weights.",
            "raw_content": "Backpropagation applies the chain rule to compute derivatives of the loss with respect to weights.",
            "metadata": {
                "chapter": "7",
                "section": "7.2",
                "pdf_page": 116,
                "book_page": 104,
                "next_id": "ch7_eq_7_12",
            },
        },
        {
            "chunk_id": "ch7_eq_7_12",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "equation",
            "content": r"Equation 7.12: \frac{\partial L}{\partial h_k} = \Omega_k^T \frac{\partial L}{\partial f_k} Jacobian matrix product.",
            "raw_content": r"Equation 7.12: \frac{\partial L}{\partial h_k} = \Omega_k^T \frac{\partial L}{\partial f_k} Jacobian matrix product.",
            "metadata": {
                "chapter": "7",
                "section": "7.2",
                "pdf_page": 117,
                "equation_number": "7.12",
                "crop_sha256": "crop_eq_7_12",
                "review_status": "pending",
                "previous_id": "ch7_text_1",
                "next_id": "ch7_prob_7_1",
            },
        },
        {
            "chunk_id": "ch7_prob_7_1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "eoc_question",
            "content": "Problem 7.1 Derive the backpropagation update for a three-layer network.",
            "raw_content": "Problem 7.1 Derive the backpropagation update for a three-layer network.",
            "metadata": {
                "chapter": "7",
                "pdf_page": 125,
                "problem_number": "7.1",
                "previous_id": "ch7_eq_7_12",
            },
        },
        {
            "chunk_id": "ch9_prob_9_1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "eoc_question",
            "content": r"Problem 9.1 Show that a normal prior with variance \sigma_\phi^2 on parameters \phi induces L2 regularization.",
            "raw_content": r"Problem 9.1 Show that a normal prior with variance \sigma_\phi^2 on parameters \phi induces L2 regularization.",
            "metadata": {
                "chapter": "9",
                "pdf_page": 162,
                "problem_number": "9.1",
            },
        },
        {
            "chunk_id": "ch18_fig_18_4",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "figure",
            "content": "Figure 18.4 Forward diffusion process gradually adds Gaussian noise to images.",
            "raw_content": "Figure 18.4 Forward diffusion process gradually adds Gaussian noise to images.",
            "metadata": {
                "chapter": "18",
                "pdf_page": 366,
                "figure_number": "18.4",
                "crop_sha256": "crop_fig_18_4",
                "review_status": "verified",
            },
        },
        {
            "chunk_id": "nb6_code_0",
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "content_type": "code",
            "content": "def gradient_descent(start, step_size, n_steps):\n    return start - step_size * n_steps",
            "raw_content": "def gradient_descent(start, step_size, n_steps):\n    return start - step_size * n_steps",
            "metadata": {
                "chapter": "6",
                "cell_index": 0,
                "start_line": 1,
                "end_line": 2,
                "symbol_names": ["gradient_descent"],
                "is_template": True,
            },
        },
    ]
    return CorpusResult(sources=sources, chunks=chunks, report={})


class SearchTests(unittest.TestCase):
    def test_named_compounds_require_one_source_phrase(self):
        chunks = {"a": {"content": "Quantum effects are discussed separately from electrical circuits."},
                  "b": {"content": "The chain rule computes derivatives."}}
        retriever = HybridRetriever(None)
        self.assertIsNotNone(retriever._check_named_requirements(
            "Explain the behavior of Quantum Circuits.", list(chunks), chunks))
        self.assertIsNone(retriever._check_named_requirements(
            "Explain the Chain Rule.", list(chunks), chunks))

    def test_teaching_passages_precede_background_and_drawing_prefers_definition(self):
        retriever = HybridRetriever(None)
        chunks = {
            "notes": {"source_id": "book", "content_type": "text", "content": "Historical references for gradients.",
                      "metadata": {"section": "Notes", "chapter": "7"}},
            "main": {"source_id": "book", "content_type": "text", "content": "The chain rule computes gradients.",
                     "metadata": {"section": "7.3", "chapter": "7"}},
            "caller": {"source_id": "demo.ipynb", "content_type": "code", "content": "draw_model(data)", "metadata": {}},
            "definition": {"source_id": "demo.ipynb", "content_type": "code", "content": "def draw_model(data):\n    plot(data)", "metadata": {}}
        }
        query = "Explain how gradients are computed."
        ranking = retriever._prioritize_requested_evidence(query, [("notes", .9), ("main", .8)],
            chunks, extract_query_anchors(query), "all")
        self.assertEqual(ranking[0][0], "main")
        query = "Where is the model drawn in the notebook?"
        ranking = retriever._prioritize_requested_evidence(query, [("caller", .9), ("definition", .8)],
            chunks, extract_query_anchors(query), "all")
        self.assertEqual(ranking[0][0], "definition")

    def test_ordered_phrase_evidence_distinguishes_equal_word_bags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            for cid, content in (("a_reversed", "Silver anchors prevent cobalt drift."),
                                 ("z_ordered", "Cobalt anchors prevent silver drift.")):
                corpus.chunks.append({"chunk_id": cid, "source_id": "book", "source_type": "pdf",
                    "content_type": "text", "content": content, "metadata": {"chapter": "4"}})
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus)
            result = HybridRetriever(store).search(SearchRequest(
                "How do cobalt anchors prevent drift?", mode="lexical", chapter="4", top_k=1))
            self.assertFalse(result.abstained, result.reason)
            self.assertEqual(result.hits[0]["chunk_id"], "z_ordered")

    def test_anchors_do_not_license_absent_named_items_or_unreported_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedder = _ControlledEmbedder()
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(_build_fixture_corpus(), embedder=embedder)
            retriever = HybridRetriever(store, embedder)
            for query in (
                "For Equation 7.12 give the measured runtime on Hyperion900 hardware.",
                "What exact throughput does `gradient_descent` achieve in benchmark measurements?",
                "Explain how UnindexedAdapter changes the backpropagation chain rule.",
            ):
                embedder.query_map[query] = [1.0, .1, 0.0, 0.0]
                result = retriever.search(SearchRequest(query))
                self.assertTrue(result.abstained, query)
                self.assertEqual(result.hits, [])
            direct = retriever.search(SearchRequest("Where is `gradient_descent` defined?"))
            self.assertFalse(direct.abstained, direct.reason)
            self.assertIsNotNone(retriever._check_unsupported_external_query(
                "What does the author blog say about backpropagation?", ["mention"],
                {"mention": {"source_id": "book", "content": "The author blog is mentioned in these references."}}))

    def test_explicit_representation_survives_ownership_and_figures_use_source_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            corpus.chunks.extend([
                {"chunk_id": "diagram_owner", "source_id": "book", "source_type": "pdf",
                 "content_type": "text", "content": "Figure 4.8 depicts repeated averaging that suppresses sensor fluctuations.",
                 "metadata": {"chapter": "4", "pdf_page": 50}},
                {"chunk_id": "actual_diagram", "source_id": "book", "source_type": "pdf",
                 "content_type": "figure", "content": "Figure 4.8. Three panels of a toy scene.",
                 "metadata": {"chapter": "4", "pdf_page": 50, "figure_number": "4.8"}},
                {"chunk_id": "distractor_diagram", "source_id": "book", "source_type": "pdf",
                 "content_type": "figure", "content": "Figure 4.9. Sensor measurements.",
                 "metadata": {"chapter": "4", "pdf_page": 51, "figure_number": "4.9"}},
                {"chunk_id": "distant_reference", "source_id": "book", "source_type": "pdf",
                 "content_type": "text", "content": "Figure 4.9 was discussed earlier. Repeated averaging suppresses fluctuations.",
                 "metadata": {"chapter": "10", "pdf_page": 150}},
            ])
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus)
            retriever = HybridRetriever(store)
            result = retriever.search(SearchRequest(
                "Which figure depicts repeated averaging that suppresses sensor fluctuations?", mode="lexical", top_k=1))
            self.assertFalse(result.abstained, result.reason)
            self.assertEqual(result.hits[0]["chunk_id"], "actual_diagram")
            index = store._get_search_index(store.status()["build_id"])
            chunks = index["chunks_by_id"]
            for modality, cid in (("figure", "actual_diagram"), ("equation", "ch7_eq_7_12")):
                chosen = retriever._select_source_passages("What is its functional form?", [(cid, .9)],
                    chunks, list(chunks), 2, modality=modality)
                self.assertEqual(chosen, [(cid, .9)])

    def test_borrowed_context_uses_its_actual_length_and_inflections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            passage = "The coefficients initialize the weights. " + "background " * 300 + " Equation 4.8."
            corpus.chunks.extend([
                {"chunk_id": "long_owner", "source_id": "book", "source_type": "pdf",
                 "content_type": "text", "content": passage,
                 "metadata": {"chapter": "4", "next_id": "short_display"}},
                {"chunk_id": "short_display", "source_id": "book", "source_type": "pdf",
                 "content_type": "equation", "content": "y=x",
                 "metadata": {"chapter": "4", "equation_number": "4.8", "previous_id": "long_owner"}},
            ])
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            status = store.build(corpus)
            retriever = HybridRetriever(store)
            index = store._get_search_index(status["build_id"])
            ids = list(index["chunks_by_id"])
            tokens = tokenize("initializing coefficient weights")
            contextual = retriever._contextual_lexical_terms(tokens, ids, index)
            ranking, matched = retriever._score_lexical(tokens, ids, contextual, {}, {})
            scores = dict(ranking)
            self.assertIn("initializing", matched["short_display"])
            self.assertIn("coefficient", matched["short_display"])
            self.assertLessEqual(scores["short_display"], scores["long_owner"])

    def test_source_passage_selection_preserves_owners_and_bounded_continuations(self):
        retriever = HybridRetriever(None)
        chunks = {
            "narrative": {"chunk_id": "narrative", "source_id": "book", "content_type": "text",
                "content": "The first computation is Equation 4.8; its related result is Equation 4.9.",
                "metadata": {"chapter": "4", "pdf_page": 50, "section": "4.2", "subsection": "Example", "next_id": "factor_a"}},
            "factor_a": {"chunk_id": "factor_a", "source_id": "book", "content_type": "equation",
                "content": "a=x", "metadata": {"chapter": "4", "pdf_page": 50, "equation_number": "4.8"}},
            "factor_b": {"chunk_id": "factor_b", "source_id": "book", "content_type": "equation",
                "content": "b=x", "metadata": {"chapter": "4", "pdf_page": 50, "equation_number": "4.9"}},
            "independent": {"chunk_id": "independent", "source_id": "book", "content_type": "text",
                "content": "A complementary explanation.", "metadata": {"chapter": "4", "pdf_page": 51}},
        }
        selected = retriever._select_source_passages("What is the functional form of the computation?",
            [("factor_a", .9), ("factor_b", .8), ("independent", .7)], chunks, list(chunks), 2)
        self.assertEqual([cid for cid, _ in selected[:2]], ["narrative", "independent"])
        # Exact requests keep the independently cited numbered display itself.
        exact = retriever._select_source_passages("Explain Equation 4.8", [("factor_a", .9)], chunks, list(chunks), 2)
        self.assertEqual(exact, [("factor_a", .9)])
        chunks["narrative"]["metadata"]["next_id"] = "continuation"
        chunks["continuation"] = {"chunk_id": "continuation", "source_id": "book", "content_type": "text",
            "content": "The reason follows in this separate source paragraph.",
            "metadata": {"chapter": "4", "section": "4.2", "subsection": "Example"}}
        selected = retriever._select_source_passages("Why does this computation work?",
            [("narrative", .9), ("independent", .8)], chunks, list(chunks), 2)
        self.assertEqual([cid for cid, _ in selected[:2]], ["narrative", "continuation"])
        chunks["continuation"]["metadata"]["subsection"] = "Unrelated section"
        selected = retriever._select_source_passages("Why does this computation work?",
            [("narrative", .9), ("independent", .8)], chunks, list(chunks), 2)
        self.assertEqual([cid for cid, _ in selected[:2]], ["narrative", "independent"])
        # A setup announcement retains its actual assignment cell, rather than
        # filling both slots with later uses of the same variables.
        chunks["announcement"] = {"chunk_id": "announcement", "source_id": "demo.ipynb",
            "content_type": "markdown", "content": "Create the coefficients.", "metadata": {"next_id": "assignment"}}
        chunks["assignment"] = {"chunk_id": "assignment", "source_id": "demo.ipynb",
            "content_type": "code", "content": "coefficients = [1, 2]",
            "metadata": {"defined_symbols": ["variable:coefficients"]}}
        selected = retriever._select_source_passages("Where are the coefficients introduced in the notebook?",
            [("announcement", .9), ("independent", .8)], chunks, list(chunks), 2)
        self.assertEqual([cid for cid, _ in selected[:2]], ["announcement", "assignment"])

    def test_unknown_paraphrase_word_needs_joint_semantic_and_subject_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            embedder = _ControlledEmbedder()
            store.build(_build_fixture_corpus(), embedder=embedder)
            retriever = HybridRetriever(store, embedder)
            query = "Explain backpropagation chain rule while preserving derivatives."
            for mode in ("hybrid", "semantic"):
                result = retriever.search(SearchRequest(query, mode=mode))
                self.assertFalse(result.abstained, result.reason)
                self.assertEqual(result.hits[0]["chunk_id"], "ch7_text_1")
                unsupported = retriever.search(SearchRequest(
                    "Explain backpropagation chain rule while measuring quantum circuits.", mode=mode))
                self.assertTrue(unsupported.abstained)
            # A high cosine alone cannot rescue a query with no known subject.
            unrelated = "Why do chloroplasts photosynthesize?"
            embedder.query_map[unrelated] = [1.0, 0.1, 0.0, 0.0]
            self.assertTrue(retriever.search(SearchRequest(unrelated)).abstained)
            # Nor can lexical support in a different, semantically weak hit.
            abstained, _ = retriever._should_abstain(
                effective_mode="hybrid", query_tokens=["momentum", "dampen"],
                content_terms=["momentum", "dampen"], core_subject_terms=["momentum", "dampen"],
                item_boosts={}, top_chunk_ids=["semantic", "lexical"],
                bm25_map={"lexical": 2.0}, semantic_map={"semantic": .9, "lexical": .1},
                matched_terms_by_chunk={"lexical": {"momentum"}}, df_map={"momentum": 1})
            self.assertTrue(abstained)

    def test_definition_language_is_conceptual_but_form_and_derivative_request_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            for cid, kind, text in (
                ("fairness_prose", "text", "Mathematical definitions of fairness evaluate bias in machine learning models."),
                ("fairness_math", "equation", "Fairness bias learning probability formula."),
                ("shallow_prose", "text", "A shallow neural network has one hidden layer and ReLU activations."),
                ("shallow_math", "equation", "Shallow neural network with one hidden layer and ReLU activations: y = f[x]."),
                ("composition_prose", "text", "The formal definition of the composition of network layers in deep neural networks."),
                ("composition_math", "equation", "The composition of network layers in deep neural networks is formally expressed as f(x)."),
            ):
                corpus.chunks.append({"chunk_id": cid, "source_id": "book", "source_type": "pdf",
                    "content_type": kind, "content": text, "metadata": {"chapter": "3"}})
            embedder = _ControlledEmbedder()
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus, embedder=embedder)
            retriever = HybridRetriever(store, embedder)
            for query, expected in (
                ("What mathematical definitions of fairness evaluate bias in machine learning models?", "fairness_prose"),
                ("What is the functional form of a shallow neural network with one hidden layer and ReLU activations?", "shallow_math"),
                ("What is the derivative expression for a Jacobian matrix product?", "ch7_eq_7_12"),
                ("How is the composition of network layers formally expressed in deep neural networks?", "composition_math"),
                ("What is the formal definition of the composition of network layers in deep neural networks?", "composition_prose"),
            ):
                result = retriever.search(SearchRequest(query, top_k=1))
                self.assertFalse(result.abstained, result.reason)
                self.assertEqual(result.hits[0]["chunk_id"], expected)

    def test_unquoted_source_defined_callable_precedes_mentions_and_obeys_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            for ref, name in (("3.1", "ReLU"), ("7.1", "fn")):
                chapter, section = ref.split(".")
                source = f"Notebooks/Chap{int(chapter):02d}/{chapter}_{section}_Example.ipynb"
                corpus.sources.append({"source_id": source, "source_type": "notebook", "path": source,
                    "sha256": name, "metadata": {"chapter": chapter}})
                corpus.chunks.extend([
                    {"chunk_id": name + "_definition", "source_id": source, "source_type": "notebook",
                     "content_type": "code", "content": f"def {name}(x):\n    return x",
                     "metadata": {"chapter": chapter, "cell_index": 2,
                         "defined_symbols": ["function:" + name], "symbol_names": [name]}},
                    {"chunk_id": name + "_mention", "source_id": source, "source_type": "notebook",
                     "content_type": "code", "content": f"# def {name}(x): function implementation\nx = {name}(1)",
                     "metadata": {"chapter": chapter, "cell_index": 1}},
                ])
            embedder = _ControlledEmbedder()
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus, embedder=embedder)
            retriever = HybridRetriever(store, embedder)
            for ref, name in (("3.1", "ReLU"), ("7.1", "fn")):
                result = retriever.search(SearchRequest(
                    f"Where is the {name} function implemented in notebook {ref}?", top_k=2))
                self.assertFalse(result.abstained, result.reason)
                self.assertEqual(result.hits[0]["chunk_id"], name + "_definition")
                self.assertTrue(all(h["source_id"] == result.hits[0]["source_id"] for h in result.hits))
            bare = retriever.search(SearchRequest("ReLU", modality="code", chapter="3", top_k=1))
            self.assertEqual(bare.hits[0]["chunk_id"], "ReLU_definition")
            for missing in ("Where is fn implemented in notebook 7.99?", "Explain backpropagation on page 9999"):
                self.assertTrue(retriever.search(SearchRequest(missing)).abstained)

    def test_compound_question_embeds_separate_source_scoped_queries_and_interleaves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            corpus.chunks.append({
                "chunk_id": "momentum_equation", "source_id": "book", "source_type": "pdf",
                "content_type": "equation", "content": "Momentum update accumulates gradients to dampen oscillations.",
                "embedding_text": "Jacobian", "metadata": {"chapter": "6", "equation_number": "6.9"}})
            # A same-named definition in a different chapter must not fill the
            # 'corresponding notebook' facet.
            other = "Notebooks/Chap09/9_1_Other.ipynb"
            corpus.sources.append({"source_id": other, "source_type": "notebook", "path": other,
                "sha256": "other", "metadata": {"chapter": "9"}})
            corpus.chunks.append({"chunk_id": "other_definition", "source_id": other,
                "source_type": "notebook", "content_type": "code",
                "content": "def gradient_descent(start):\n    return start", "metadata": {"chapter": "9"}})
            book_query = "Explain the mathematical formulation of momentum in chapter 6 of the book"
            code_query = "where is gradient_descent implemented in the corresponding notebook"
            query = book_query + ", and " + code_query + "?"
            embedder = _ControlledEmbedder()
            embedder.query_map[book_query] = [0.0, 0.0, .1, 1.0]
            embedder.query_map[code_query] = [.1, 1.0, 0.0, 0.0]
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus, embedder=embedder)
            retriever = HybridRetriever(store, embedder)
            result = retriever.search(SearchRequest(query, top_k=4))
            self.assertFalse(result.abstained, result.reason)
            self.assertEqual([h["chunk_id"] for h in result.hits[:2]], ["momentum_equation", "nb6_code_0"])
            self.assertLessEqual(len(result.hits), 4)
            self.assertNotIn("other_definition", [h["chunk_id"] for h in result.hits])
            self.assertEqual(embedder.query_calls, [book_query, code_query])
            self.assertEqual(result.hits[0]["retrieval_query"], book_query)
            self.assertEqual(result.hits[1]["retrieval_query"], code_query)
            functional = "What is the functional form of the momentum update in chapter 6"
            embedder.query_map[functional] = [0.0, 0.0, .1, 1.0]
            implicit_book = retriever.search(SearchRequest(
                functional + ", and where is gradient_descent implemented in notebook 6.2?", top_k=2))
            self.assertFalse(implicit_book.abstained, implicit_book.reason)
            self.assertEqual([h["chunk_id"] for h in implicit_book.hits], ["momentum_equation", "nb6_code_0"])
            self.assertEqual(retriever._source_subqueries(
                "Where is the derivative expression computed in notebook 6.2?", "all",
                {c["chunk_id"]: c for c in corpus.chunks}), [])
            source_map = {c["chunk_id"]: c for c in corpus.chunks}
            for wording in ("Explain momentum and its assignment code",
                            "How is momentum defined in the book and implemented in the notebook?"):
                plans = retriever._source_subqueries(wording, "all", source_map)
                self.assertEqual([plan[0] for plan in plans], ["book", "notebook"])
                self.assertIn("momentum", plans[1][1])
            self.assertEqual(retriever._source_subqueries("Explain this function's code", "all", source_map), [])
            concept_code_q = "What explains the double descent curve in chapter 8, and where is weights_init defined in notebook 8.3?"
            concept_plans = retriever._source_subqueries(concept_code_q, "all", source_map)
            self.assertEqual([p[0] for p in concept_plans], ["book", "notebook"])
            self.assertEqual(concept_plans[0][1], "What explains the double descent curve in chapter 8")
            self.assertEqual(concept_plans[0][2], ["8"])
            self.assertEqual(concept_plans[1][1], "where is weights_init defined in notebook 8.3")
            self.assertEqual(concept_plans[1][3], ["8.3"])
            # Explicitly different chapter/notebook references override shared scope.
            cross = retriever.search(SearchRequest(
                "Explain Equation 7.12 in the book, and where is gradient_descent implemented in notebook 6.2?", top_k=2))
            self.assertFalse(cross.abstained, cross.reason)
            self.assertEqual([h["chunk_id"] for h in cross.hits], ["ch7_eq_7_12", "nb6_code_0"])
            missing = retriever.search(SearchRequest(
                "Explain Equation 7.12 in the book, and where is `absent_fn` implemented in notebook 6.2?"))
            self.assertTrue(missing.abstained)
            self.assertEqual(missing.hits, [])

    def test_tokenize_locator_embedding_text_and_rrf_helpers(self):
        self.assertEqual(
            tokenize("def gradient_descent(SelfAttention_x1):"),
            ["def", "gradient", "descent", "self", "attention", "x1"],
        )

        # Verify natural-language "class defined" and LaTeX/math subscripts ("sigma_phi^2", "sigma_phi squared")
        # are not misclassified as missing code symbols
        q_anc = extract_query_anchors(
            "Where is the TrueDataDistribution class defined for variance sigma_phi^2 in notebook 6.4 on page 141?"
        )
        self.assertEqual(q_anc["explicit_symbol"], [])
        self.assertEqual(q_anc["symbol"], [])
        self.assertEqual(q_anc["notebook"], ["6.4"])
        self.assertEqual(q_anc["page"], ["141"])

        q_prob_9_1 = extract_query_anchors(
            "Problem 9.1: show why a normal prior with variance sigma_phi squared induces L2 regularization."
        )
        self.assertEqual(q_prob_9_1["problem"], ["9.1"])
        self.assertEqual(q_prob_9_1["explicit_symbol"], [])
        self.assertEqual(q_prob_9_1["symbol"], [])

        book_chunk = {
            "chunk_id": "c1",
            "source_id": "book",
            "source_type": "pdf",
            "content_type": "figure",
            "content": "Caption text.",
            "metadata": {
                "chapter": "18",
                "pdf_page": 366,
                "book_page": 354,
                "figure_number": "18.4",
            },
        }
        loc_book = locator(book_chunk)
        self.assertIn("PDF page 366", loc_book)
        self.assertIn("book p. 354", loc_book)
        self.assertIn("Figure 18.4", loc_book)

        nb_chunk = {
            "chunk_id": "c2",
            "source_id": "Notebooks/Chap06/6_2_Gradient_Descent.ipynb",
            "source_type": "notebook",
            "content_type": "code",
            "content": "x = 1",
            "metadata": {"cell_index": 0, "start_line": 3, "end_line": 7},
        }
        loc_nb = locator(nb_chunk)
        self.assertIn("Notebooks/Chap06/6_2_Gradient_Descent.ipynb", loc_nb)
        self.assertIn("cell 0", loc_nb)
        self.assertIn("lines 3-7", loc_nb)

        self.assertEqual(
            embedding_text({"content": "Plain text"}), "Plain text"
        )
        self.assertEqual(
            embedding_text(
                {"content": "Plain text", "embedding_text": "Explicit override"}
            ),
            "Explicit override",
        )
        enriched = embedding_text(book_chunk)
        self.assertIn("Figure 18.4", enriched)
        self.assertIn("Caption text.", enriched)
        context_rich = embedding_text({"content_type": "equation", "content": "y=x",
                                       "metadata": {"chapter_title": "Fitting models", "subsection": "Momentum",
                                                    "context_before": "The update accumulates previous gradients."}})
        self.assertIn("Fitting models", context_rich)
        self.assertIn("Momentum", context_rich)
        self.assertIn("previous gradients", context_rich)

        # Deterministic tie-breaking: when two items tie on RRF score and best_rank,
        # the item ranked #1 in the first (primary/lexical) list wins regardless of chunk_id order!
        fused = rrf_fuse([("z_first", 10.0), ("a_second", 5.0)], [("a_second", 0.9), ("z_first", 0.8)], k=60)
        self.assertEqual(fused[0][0], "z_first")
        self.assertEqual(fused[1][0], "a_second")
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62, places=8)
        self.assertAlmostEqual(fused[1][1], 1 / 61 + 1 / 62, places=8)

    def test_lexical_hybrid_fallback_and_semantic_only_error_when_dense_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            store.build(_build_fixture_corpus(), embedder=None)

            retriever = HybridRetriever(store, embedder=None)

            # Default SearchRequest has mode='hybrid', top_k=6, modality='all'.
            # Without dense embeddings, hybrid falls back to lexical with a warning.
            res = retriever.search(SearchRequest(query="backpropagation chain rule"))
            self.assertFalse(res.abstained)
            self.assertEqual(res.mode, "lexical")
            self.assertTrue(len(res.warnings) >= 1)
            self.assertIn("falling back to lexical", res.warnings[0])
            self.assertEqual(res.hits[0]["chunk_id"], "ch7_text_1")
            self.assertEqual(res.hits[0]["citation"], "S1")
            self.assertIsNone(res.hits[0]["semantic_score"])
            self.assertGreater(res.hits[0]["bm25_score"], 0.0)

            # Semantic-only mode must raise a useful UDLException when dense is missing
            with self.assertRaises(UDLException) as ctx:
                retriever.search(
                    SearchRequest(query="backpropagation", mode="semantic")
                )
            self.assertEqual(ctx.exception.code, "dense_missing")

    def test_exact_anchors_scope_filters_and_nonexistent_reference_abstention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            embedder = _ControlledEmbedder(dimension=4)
            store.build(_build_fixture_corpus(), embedder=embedder)

            retriever = HybridRetriever(store, embedder=embedder)

            # Exact figure anchor
            fig_res = retriever.search(SearchRequest(query="Explain Figure 18.4"))
            self.assertFalse(fig_res.abstained)
            self.assertEqual(fig_res.hits[0]["chunk_id"], "ch18_fig_18_4")
            self.assertEqual(len(fig_res.hits), 1)

            # Exact equation anchor
            eq_res = retriever.search(SearchRequest(query="What is Equation 7.12?"))
            self.assertFalse(eq_res.abstained)
            self.assertEqual(eq_res.hits[0]["chunk_id"], "ch7_eq_7_12")

            # Exact problem anchor
            prob_res = retriever.search(SearchRequest(query="Problem 7.1"))
            self.assertFalse(prob_res.abstained)
            self.assertEqual(prob_res.hits[0]["chunk_id"], "ch7_prob_7_1")

            # Regression: Problem 9.1 query with math subscript "sigma_phi squared" or
            # non-code underscore notation must NOT hard-abstain as a missing code symbol
            prob_9_1_res = retriever.search(
                SearchRequest(
                    query="Problem 9.1: show why a normal prior with variance sigma_phi squared induces L2 regularization."
                )
            )
            self.assertFalse(prob_9_1_res.abstained)
            self.assertEqual(prob_9_1_res.hits[0]["chunk_id"], "ch9_prob_9_1")

            prose_underscore_res = retriever.search(
                SearchRequest(
                    query="Problem 9.1: explain how prior_variance induces L2 regularization."
                )
            )
            self.assertFalse(prose_underscore_res.abstained)
            self.assertEqual(prose_underscore_res.hits[0]["chunk_id"], "ch9_prob_9_1")

            # Exact symbol / identifier component lookup
            sym_res = retriever.search(
                SearchRequest(query="gradient_descent", modality="code")
            )
            self.assertFalse(sym_res.abstained)
            self.assertEqual(sym_res.hits[0]["chunk_id"], "nb6_code_0")

            # Chapter filter
            chap6_res = retriever.search(
                SearchRequest(query="gradient", chapter="Chap06")
            )
            self.assertFalse(chap6_res.abstained)
            self.assertTrue(
                all(
                    h["source_id"].startswith("Notebooks/Chap06")
                    for h in chap6_res.hits
                )
            )

            # Nonexistent exact figure, equation, problem, chapter, code-intent symbol, and excluded external source lookups MUST abstain
            for bad_query in (
                "Explain Figure 18.99 on diffusion noise",
                "What is Equation 7.99 in backpropagation?",
                "Solve Problem 7.99 on backpropagation",
                "Show `nonexistent_optimizer_step` in gradient descent",
                "Where is the nonexistent_helper_fn function implemented in the notebooks?",
                "What does chapter 99 say about backpropagation?",
                "What does the answer booklet say for every end-of-chapter problem?",
                "Summarize the errata published after the pinned edition.",
            ):
                bad_res = retriever.search(SearchRequest(query=bad_query))
                self.assertTrue(
                    bad_res.abstained,
                    f"Expected abstention for nonexistent/unsupported reference: {bad_query}",
                )
                self.assertEqual(bad_res.hits, [])
                self.assertIsNotNone(bad_res.reason)

            # Nonexistent unquoted symbol when modality='code' also abstains
            bad_code_mod = retriever.search(
                SearchRequest(query="nonexistent_helper_fn", modality="code")
            )
            self.assertTrue(bad_code_mod.abstained)
            self.assertEqual(bad_code_mod.hits, [])

    def test_direct_equation_precedes_code_that_only_mentions_its_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _build_fixture_corpus()
            corpus.chunks.append({
                "chunk_id": "nb_mention", "source_id": corpus.sources[1]["source_id"],
                "source_type": "notebook", "content_type": "code",
                "content": "# Equation 7.12 is discussed here\ngradient_descent()",
                "embedding_text": "gradient_descent step_size",
                "metadata": {"chapter": "6", "cell_index": 1, "start_line": 1, "end_line": 2},
            })
            embedder = _ControlledEmbedder()
            query = "Explain Equation 7.12 in gradient_descent()"
            embedder.query_map[query] = [0.1, 1.0, 0.0, 0.0]
            store = SQLiteStore(root / "index.sqlite3", project_root=root)
            store.build(corpus, embedder=embedder)
            result = HybridRetriever(store, embedder).search(SearchRequest(query, top_k=2))
            self.assertFalse(result.abstained)
            self.assertEqual(result.hits[0]["chunk_id"], "ch7_eq_7_12")
            self.assertIn("nb_mention", [h["chunk_id"] for h in result.hits])

    def test_abstention_uses_original_scores_not_rrf_and_expands_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteStore(root / "data" / "udl_rag.sqlite3", project_root=root)
            embedder = _ControlledEmbedder(dimension=4)
            # Configure a realistic unrelated cosine score (~0.378) for an out-of-domain query
            embedder.query_map["What is photosynthesis in chloroplasts?"] = [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
            store.build(_build_fixture_corpus(), embedder=embedder)

            retriever = HybridRetriever(
                store,
                embedder=embedder,
                abstention=AbstentionConfig(
                    min_bm25_score=0.25,
                    min_semantic_score=0.42,
                    strong_semantic_score=0.62,
                ),
            )

            # Out-of-domain query has stopwords ("What", "is", "in") in the corpus and gets a top-1 RRF rank,
            # but must abstain because non-stopword lexical overlap is 0 and cosine similarity is below threshold.
            ood_res = retriever.search(
                SearchRequest(
                    query="What is photosynthesis in chloroplasts?",
                    mode="hybrid",
                )
            )
            self.assertTrue(ood_res.abstained)
            self.assertEqual(ood_res.hits, [])

            # Out-of-domain query that shares generic ML terms ("backpropagation", "gradients")
            # with the corpus MUST still abstain when its core subject ("quantum", "circuits") is OOV.
            quantum_res = retriever.search(
                SearchRequest(
                    query="Explain the quantum backpropagation algorithm used to compute derivatives across quantum circuits.",
                    mode="lexical",
                )
            )
            self.assertTrue(quantum_res.abstained)
            self.assertEqual(quantum_res.hits, [])

            # Valid hybrid search preserves original bm25_score, semantic_score, and RRF retrieval_score
            good_res = retriever.search(
                SearchRequest(query="backpropagation chain rule", mode="hybrid")
            )
            self.assertFalse(good_res.abstained)
            top_hit = good_res.hits[0]
            self.assertEqual(top_hit["chunk_id"], "ch7_text_1")
            self.assertGreater(top_hit["bm25_score"], 0.5)
            self.assertIsNotNone(top_hit["semantic_score"])
            self.assertGreater(top_hit["semantic_score"], 0.8)
            self.assertAlmostEqual(
                top_hit["retrieval_score"], 1 / 61 + 1 / 61, places=6
            )

            # Test source neighbor expansion helpers
            neighbors = get_chunk_neighbors(store, "ch7_eq_7_12", window=1)
            self.assertEqual(
                [c["chunk_id"] for c in neighbors["previous"]], ["ch7_text_1"]
            )
            self.assertEqual(
                [c["chunk_id"] for c in neighbors["next"]], ["ch7_prob_7_1"]
            )

            expanded = expand_neighbors(store, [top_hit], window=1)
            self.assertEqual(
                [c["chunk_id"] for c in expanded],
                ["ch7_text_1", "ch7_eq_7_12"],
            )
            ctx_block = build_context_block(good_res.hits)
            self.assertIn("[S1]", ctx_block)
            self.assertIn("PDF page 116", ctx_block)


if __name__ == "__main__":
    unittest.main()
