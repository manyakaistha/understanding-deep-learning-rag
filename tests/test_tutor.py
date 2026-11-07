import unittest
from unittest.mock import Mock, patch

from src.contracts import SearchRequest, SearchResult
from src.providers import ProviderConfig
from src.tutor import TutorService, check_citations, context_from_hits


def hit(content="Weights follow gradients.", **kw):
    return dict(chunk_id="a", source_id="book", content_type="text", content=content,
                metadata={}, locator="Book, PDF p. 42", **kw)


class TutorTests(unittest.TestCase):
    def test_context_never_slices_source_and_skips_unreviewed_math(self):
        context, sources = context_from_hits([hit("x" * 1000), dict(hit(), content_type="equation"), hit("short")], 100)
        self.assertEqual(len(sources), 1)
        self.assertIn("short", context)
        self.assertNotIn("xxx", context)

    def test_unknown_citations_are_not_sources(self):
        cited, invalid = check_citations("Claim [S1]. Fabrication [S99].", [hit()])
        self.assertEqual(invalid, ["S99"])
        self.assertEqual(len(cited), 1)

    def test_abstention_makes_no_provider_call(self):
        service = TutorService(Mock(search=Mock(return_value=SearchResult(abstained=True))))
        with patch("src.tutor.stream_chat") as generate:
            result = service.answer(SearchRequest("not in book"), ProviderConfig("groq", ""))
        generate.assert_not_called()
        self.assertTrue(result["abstained"])

    def test_invalid_answer_repaired_once_and_rejected(self):
        service = TutorService(Mock(search=Mock(return_value=SearchResult(hits=[hit()]))))
        with patch("src.tutor.stream_chat", return_value=iter(["False [S99]."])), patch("src.tutor.complete_chat", return_value="Still false [S88].") as repair:
            result = service.answer(SearchRequest("weights"), ProviderConfig("ollama", "test"))
        self.assertEqual(repair.call_count, 1)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["sources"], [])

    def test_source_map_contains_only_actually_cited_evidence(self):
        second = dict(hit("Other source"), chunk_id="b")
        service = TutorService(Mock(search=Mock(return_value=SearchResult(hits=[hit(), second]))))
        with patch("src.tutor.stream_chat", return_value=iter(["Supported [S2]."])):
            result = service.answer(SearchRequest("weights"), ProviderConfig("ollama", "test"))
        self.assertFalse(result["abstained"])
        self.assertEqual([s["chunk_id"] for s in result["sources"]], ["b"])

    def test_total_prompt_budget_includes_history_and_question(self):
        service = TutorService(Mock(search=Mock(return_value=SearchResult(hits=[hit("evidence " * 2000)]))))
        history = [{"role": "user", "content": "history " * 2000} for _ in range(6)]
        captured = []
        def generate(messages, config, client=None):
            captured.extend(messages)
            return iter(["Supported [S1]."])
        with patch("src.tutor.stream_chat", side_effect=generate):
            service.answer(SearchRequest("question " * 500), ProviderConfig("ollama", "test"), history)
        self.assertTrue(captured)
        self.assertLessEqual(sum(len(m["content"]) for m in captured), 32000)

    def test_generated_worked_solution_is_always_labelled(self):
        service = TutorService(Mock(search=Mock(return_value=SearchResult(hits=[hit()]))))
        with patch("src.tutor.stream_chat", return_value=iter(["Apply the chain rule [S1]."])):
            result = service.answer(SearchRequest("Solve this exercise"), ProviderConfig("ollama", "test"))
        self.assertTrue(result["answer"].startswith("**Derived worked solution**"))

    def test_generated_notebook_code_is_labelled_without_relabelling_explanations(self):
        service = TutorService(Mock(search=Mock(return_value=SearchResult(hits=[hit()]))))
        draft = "```python\ndef step(x):\n    return x\n```\nBased on [S1]."
        with patch("src.tutor.stream_chat", side_effect=lambda *a, **k: iter([draft])):
            generated = service.answer(SearchRequest("Complete the notebook function"), ProviderConfig("ollama", "test"))
            implementation = service.answer(SearchRequest("Implement gradient_descent_step"), ProviderConfig("ollama", "test"))
            explanation = service.answer(SearchRequest("Explain this notebook function"), ProviderConfig("ollama", "test"))
        self.assertTrue(generated["answer"].startswith("**Generated code"))
        self.assertIn("not executed", generated["answer"])
        self.assertTrue(implementation["answer"].startswith("**Generated code"))
        self.assertEqual(explanation["answer"], draft)

    def test_context_budget_cannot_silently_drop_a_required_evidence_facet(self):
        book = hit("x" * 25000)
        code = dict(hit("def step(x): return x"), chunk_id="b", source_id="notebook", content_type="code")
        result = SearchResult(hits=[book, code], required_evidence_groups=[["a"], ["b"]])
        service = TutorService(Mock(search=Mock(return_value=result)))
        with patch("src.tutor.stream_chat") as generate:
            answer = service.answer(SearchRequest("Explain the theory and its code"), ProviderConfig("ollama", "test"))
        generate.assert_not_called()
        self.assertTrue(answer["abstained"])
        self.assertIn("every part", answer["answer"])
        self.assertEqual(answer["sources"], [])
        result.hits[0] = hit("The update follows the negative gradient.")
        with patch("src.tutor.stream_chat", return_value=iter(["The theory [S1] supports the code [S2]."])) as generate:
            answer = service.answer(SearchRequest("Explain the theory and its code"), ProviderConfig("ollama", "test"))
        generate.assert_called_once()
        self.assertFalse(answer["abstained"])


if __name__ == "__main__":
    unittest.main()
