"""LLM integration (Person 3, Task 1). Provider is chosen in .env, so it can be swapped
without changing any other code."""
import re

import httpx

from .config import Settings
from .prompts import NO_ANSWER_TOKEN


class LLMError(Exception):
    """The LLM provider could not be reached or returned an error."""


class BaseLLM:
    def generate(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, s: Settings):
        self.s = s

    def generate(self, system, messages):
        try:
            r = httpx.post(
                self.URL,
                headers={"x-api-key": self.s.llm_api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": self.s.llm_model, "max_tokens": self.s.llm_max_tokens,
                      "temperature": self.s.llm_temperature, "system": system, "messages": messages},
                timeout=self.s.llm_timeout,
            )
            r.raise_for_status()
            blocks = r.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


class CohereLLM(BaseLLM):
    """Cohere Chat API v2 (langchain-cohere is already in Person 2's requirements, so the team
    may already have a Cohere key)."""
    URL = "https://api.cohere.com/v2/chat"

    def __init__(self, s: Settings):
        self.s = s

    def generate(self, system, messages):
        try:
            r = httpx.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.s.llm_api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.s.llm_model,
                      "messages": [{"role": "system", "content": system}, *messages],
                      "temperature": self.s.llm_temperature, "max_tokens": self.s.llm_max_tokens},
                timeout=self.s.llm_timeout,
            )
            r.raise_for_status()
            parts = r.json()["message"]["content"]
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"Cohere request failed: {exc}") from exc


class OpenAICompatibleLLM(BaseLLM):
    """Works with OpenAI, Groq, Google Gemini (OpenAI endpoint), OpenRouter and local Ollama."""

    def __init__(self, s: Settings):
        self.s = s
        self.url = (s.llm_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def generate(self, system, messages):
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"
        try:
            r = httpx.post(
                self.url, headers=headers, timeout=self.s.llm_timeout,
                json={"model": self.s.llm_model,
                      "messages": [{"role": "system", "content": system}, *messages],
                      "temperature": self.s.llm_temperature, "max_tokens": self.s.llm_max_tokens},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc


class MockLLM(BaseLLM):
    """Offline stand-in so the backend runs and can be tested without an API key or cost.
    It answers with the first sentences of source [1]. Never use it for the final evaluation."""

    def generate(self, system, messages):
        last = messages[-1]["content"]
        if "<university_info>" not in last:  # query-rewrite call
            follow = re.search(r"Follow-up question:\s*(.+)$", last, re.S)
            prev = re.findall(r"^Student: (.+)$", last, re.M)
            q = follow.group(1).strip() if follow else last
            return f"{prev[-1]} {q}" if prev else q
        m = re.search(r"\[1\] Source: [^\n]*\n(?:Section: [^\n]*\n)?(.+?)(?:\n\n\[2\] Source:|\n</university_info>)", last, re.S)
        if not m:
            return NO_ANSWER_TOKEN
        sentences = re.split(r"(?<=[.!?])\s+", m.group(1).strip())
        return " ".join(sentences[:2]) + " [1]"


def get_llm(s: Settings) -> BaseLLM:
    if s.llm_provider == "mock":
        return MockLLM()
    if not s.llm_model:
        raise ValueError("LLM_MODEL must be set in .env")
    if s.llm_provider in ("anthropic", "cohere") and not s.llm_api_key:
        raise ValueError(f"LLM_API_KEY must be set in .env for the {s.llm_provider} provider")
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s)
    if s.llm_provider == "cohere":
        return CohereLLM(s)
    if s.llm_provider == "openai":
        return OpenAICompatibleLLM(s)
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
