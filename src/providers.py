"""Direct, small Chat Completions HTTP adapters. No agent or provider SDK."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import httpx

from src.contracts import UDLException

PRESETS = {
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "gemini": {"label": "Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY"},
    "openai_compat": {"label": "OpenAI-compatible", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "ollama": {"label": "Ollama (local)", "base_url": "http://127.0.0.1:11434/v1", "key_env": None},
}


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = None
    max_tokens: int = 2048
    timeout: float = 120.0

    def connection(self):
        if self.provider not in PRESETS:
            raise UDLException("provider_invalid", "Choose groq, gemini, openai_compat, or ollama.")
        if not self.model or not self.model.strip() or len(self.model) > 200:
            raise UDLException("model_required", "Enter an explicit model ID available to your provider account.")
        preset = PRESETS[self.provider]
        base = (self.base_url or preset["base_url"]).rstrip("/")
        parsed = urlparse(base)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise UDLException("endpoint_invalid", "Use an API base URL without credentials, query parameters, or fragments.")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise UDLException("endpoint_invalid", "Remote API endpoints require HTTPS; local loopback endpoints may use HTTP.")
        if self.provider in {"groq", "gemini"} and base != preset["base_url"]:
            raise UDLException("endpoint_invalid", "Use OpenAI-compatible mode to configure a custom endpoint and key.")
        key = (self.api_key or "").strip()
        # Never attach an environment credential to an arbitrary custom host.
        if not key and base == preset["base_url"] and preset["key_env"]:
            key = os.environ.get(preset["key_env"], "").strip()
            if self.provider == "gemini" and not key:
                key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key and self.provider != "ollama":
            raise UDLException("key_required", "Supply your API key for this provider. Keys are not saved.")
        return base, key


def public_presets():
    return [dict(id=name, label=p["label"], base_url=p["base_url"],
                 configured=bool((p["key_env"] and os.environ.get(p["key_env"])) or
                                 (name == "gemini" and os.environ.get("GOOGLE_API_KEY"))))
            for name, p in PRESETS.items()]


def _provider_error(status):
    if status in (401, 403):
        return UDLException("provider_auth", "The provider rejected the key or model access. Check your provider settings.")
    if status == 429:
        return UDLException("provider_rate_limit", "The provider's rate or quota limit was reached. Wait or choose another model.", True)
    if status == 404:
        return UDLException("provider_model", "The model or endpoint was not found. Check the exact model ID and base URL.")
    return UDLException("provider_response", f"The answer provider returned HTTP {status}. Check model compatibility and try again.", status >= 500)


def _events(lines):
    """SSE data events, including multiline events and a final unterminated event."""
    data = []
    for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def stream_chat(messages: List[Dict], config: ProviderConfig,
                client: Optional[httpx.Client] = None) -> Iterator[str]:
    base, key = config.connection()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = {"model": config.model.strip(), "messages": messages, "stream": True,
               "max_tokens": max(16, min(int(config.max_tokens), 8192))}
    local_native = config.provider == "ollama"
    endpoint = base + "/chat/completions"
    if local_native:
        # The compatible endpoint cannot configure Ollama's context window.
        # Native per-request options avoid silent truncation at the server's
        # smaller default, without modifying a student's installed model.
        endpoint = (base[:-3] if base.endswith("/v1") else base) + "/api/chat"
        payload = {"model": config.model.strip(), "messages": messages, "stream": True,
                   "options": {"num_ctx": 16384, "num_predict": payload["max_tokens"], "temperature": 0}}
        if config.model.strip().lower().startswith("qwen3"):
            # Use Qwen3's supported direct-answer mode; hidden reasoning would
            # otherwise consume the student's bounded answer budget.
            payload["think"] = False
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(config.timeout, connect=10.0), follow_redirects=False)
    emitted = False
    try:
        with client.stream("POST", endpoint, headers=headers, json=payload) as response:
            if response.status_code >= 300:
                raise _provider_error(response.status_code)
            events = response.iter_lines() if local_native else _events(response.iter_lines())
            for data in events:
                if not data.strip():
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except (ValueError, TypeError):
                    raise UDLException("provider_format", "The provider returned an invalid streaming response.") from None
                if event.get("error"):
                    raise UDLException("provider_response", "The provider reported an error while generating the answer.", True)
                if local_native:
                    text = (event.get("message") or {}).get("content")
                    if isinstance(text, str) and text:
                        emitted = True
                        yield text
                    if event.get("done_reason") == "length":
                        raise UDLException("answer_truncated", "The model reached the answer-length limit. Increase --max-tokens or ask a narrower question.")
                    if event.get("done"):
                        break
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue  # Optional usage-only event.
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if isinstance(text, str) and text:
                    emitted = True
                    yield text
                if choices[0].get("finish_reason") == "length":
                    raise UDLException("answer_truncated", "The model reached the answer-length limit. Increase --max-tokens or ask a narrower question.")
            if not emitted:
                raise UDLException("provider_empty", "The provider returned no answer text. Try another model.", True)
    except UDLException:
        raise
    except httpx.TimeoutException:
        raise UDLException("provider_timeout", "The provider timed out. Retry or select a faster model.", True) from None
    except httpx.HTTPError:
        raise UDLException("provider_unavailable", "Could not connect to the answer provider. Check the endpoint and local model server.", True) from None
    finally:
        if owned:
            client.close()


def complete_chat(messages, config, client=None):
    return "".join(stream_chat(messages, config, client=client))
