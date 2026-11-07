import json
import unittest

import httpx

from src.contracts import UDLException
from src.providers import ProviderConfig, _events, complete_chat


class ProviderTests(unittest.TestCase):
    def test_shared_wire_contract_and_usage_only_event(self):
        seen = []

        def handle(request):
            seen.append(request)
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"λ "}}]}\n\n'
                                  'data: {"choices":[],"usage":{"total_tokens":3}}\n\n'
                                  'data: {"choices":[{"delta":{"content":"works"}}]}\n\n'
                                  'data: [DONE]\n\n', headers={"content-type": "text/event-stream"})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            for provider in ("groq", "gemini", "openai_compat"):
                config = ProviderConfig(provider, "test-model", api_key="not-a-real-key")
                self.assertEqual(complete_chat([{"role": "user", "content": "question"}], config, client), "λ works")
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(r.headers["authorization"] == "Bearer not-a-real-key" for r in seen))
        self.assertTrue(all(json.loads(r.content)["stream"] for r in seen))
        self.assertTrue(all("not-a-real-key" not in str(r.url) for r in seen))

    def test_errors_are_actionable_and_redacted(self):
        for status, code in ((401, "provider_auth"), (429, "provider_rate_limit"), (503, "provider_response")):
            with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, text="SECRET"))) as client:
                with self.assertRaises(UDLException) as ctx:
                    complete_chat([], ProviderConfig("groq", "model", api_key="SECRET"), client)
            self.assertEqual(ctx.exception.code, code)
            self.assertNotIn("SECRET", str(ctx.exception))

    def test_no_environment_key_leak_to_custom_endpoint(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"OPENAI_API_KEY": "SECRET"}):
            with self.assertRaises(UDLException) as ctx:
                ProviderConfig("openai_compat", "model", base_url="https://example.com/v1").connection()
        self.assertEqual(ctx.exception.code, "key_required")

    def test_multiline_and_unterminated_sse(self):
        self.assertEqual(list(_events(["data: one", "data: two", "", ": heartbeat", "data: three"])), ["one\ntwo", "three"])

    def test_remote_cleartext_rejected(self):
        with self.assertRaises(UDLException):
            ProviderConfig("openai_compat", "model", api_key="x", base_url="http://example.com/v1").connection()

    def test_ollama_context_is_explicit_and_stream_is_ndjson(self):
        seen = []
        def handle(request):
            seen.append(request)
            return httpx.Response(200, text='{"message":{"content":"Local answer"},"done":false}\n{"message":{"content":""},"done":true,"done_reason":"stop"}\n')
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            answer = complete_chat([], ProviderConfig("ollama", "local-model"), client)
        self.assertEqual(answer, "Local answer")
        self.assertTrue(str(seen[0].url).endswith("/api/chat"))
        self.assertEqual(json.loads(seen[0].content)["options"]["num_ctx"], 16384)
        self.assertEqual(json.loads(seen[0].content)["options"]["temperature"], 0)
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            complete_chat([], ProviderConfig("ollama", "qwen3:8b"), client)
        self.assertIs(json.loads(seen[-1].content)["think"], False)

    def test_truncated_model_output_is_not_reported_as_complete(self):
        def handle(request):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}\n\n')
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(UDLException) as ctx:
                complete_chat([], ProviderConfig("groq", "model", api_key="key"), client)
        self.assertEqual(ctx.exception.code, "answer_truncated")


if __name__ == "__main__":
    unittest.main()
