"""Chatbot logic: connects retrieval (Person 2) to the LLM (Person 3, Tasks 3-5).

Pipeline for each question:
  1. Personal problem needing staff?  -> escalate with the right contact (no LLM call)
  2. Follow-up question?               -> rewrite it as a standalone question for retrieval
  3. Retrieve chunks, drop weak ones   -> nothing relevant? fallback + contact (no LLM guessing)
  4. LLM answers from retrieved chunks -> says INSUFFICIENT_INFO? fallback + contact
  5. Return answer + the sources the LLM actually cited
"""
import logging
import re
import time
from dataclasses import dataclass, field

from .contacts import ContactDirectory, needs_human
from .llm import BaseLLM, LLMError
from .prompts import (NO_ANSWER_TOKEN, REWRITE_SYSTEM_PROMPT, SYSTEM_PROMPT,
                      build_rewrite_message, build_user_message)
from .retriever import Chunk, Retriever
from .sessions import SessionStore

log = logging.getLogger("chatbot")


@dataclass
class ChatResult:
    answer: str
    status: str                      # answered | clarify | no_info | escalated
    sources: list[dict] = field(default_factory=list)
    contact: dict | None = None
    response_time_ms: int = 0


class Chatbot:
    def __init__(self, retriever: Retriever, llm: BaseLLM, contacts: ContactDirectory,
                 sessions: SessionStore, top_k: int = 4, min_relevance: float = 0.3):
        self.retriever, self.llm, self.contacts, self.sessions = retriever, llm, contacts, sessions
        self.top_k, self.min_relevance = top_k, min_relevance

    def ask(self, question: str, session_id: str) -> ChatResult:
        start = time.perf_counter()
        question = question.strip()
        history = self.sessions.get(session_id)

        if needs_human(question):
            result = self._escalate(question)
        else:
            query = self._standalone_question(question, history)
            chunks = [c for c in self.retriever.search(query, self.top_k)
                      if c.score >= self.min_relevance]
            if not chunks:
                result = self._fallback(query)
            else:
                result = self._answer(question, query, chunks, history)

        self.sessions.append(session_id, question, result.answer)
        result.response_time_ms = int((time.perf_counter() - start) * 1000)
        log.info("status=%s sources=%d time_ms=%d", result.status, len(result.sources),
                 result.response_time_ms)
        return result

    # --- steps -----------------------------------------------------------------------------
    def _standalone_question(self, question: str, history: list[dict]) -> str:
        """'What about international students?' -> 'What are the admission requirements for
        Computer Science for international students?' so retrieval finds the right chunks."""
        if not history:
            return question
        try:
            rewritten = self.llm.generate(
                REWRITE_SYSTEM_PROMPT,
                [{"role": "user", "content": build_rewrite_message(history, question)}])
            return rewritten.strip() or question
        except LLMError:
            last_user = next(m["content"] for m in reversed(history) if m["role"] == "user")
            return f"{last_user} {question}"

    def _answer(self, question, query, chunks: list[Chunk], history) -> ChatResult:
        messages = history + [{"role": "user", "content": build_user_message(question, chunks)}]
        raw = self.llm.generate(SYSTEM_PROMPT, messages)  # LLMError handled by the API layer

        if not raw or NO_ANSWER_TOKEN in raw:
            return self._fallback(query, category_hint=chunks[0].category)

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw) if 1 <= int(n) <= len(chunks)})
        if not cited:
            if raw.rstrip().endswith("?"):        # clarifying question, e.g. "Which fees do you mean?"
                return ChatResult(raw, "clarify")
            cited = [1]                           # uncited answer: attribute to the best chunk
        sources = [{"id": n, "title": chunks[n - 1].title, "url": chunks[n - 1].url} for n in cited]
        return ChatResult(raw, "answered", sources=sources)

    def _fallback(self, text: str, category_hint: str | None = None) -> ChatResult:
        contact = self.contacts.route(text, category_hint)
        answer = ("I couldn't find reliable information about this in the University of Mauritius "
                  f"resources available to me. Please contact the {ContactDirectory.describe(contact)} "
                  "for further assistance.")
        return ChatResult(answer, "no_info", contact=contact)

    def _escalate(self, question: str) -> ChatResult:
        contact = self.contacts.route(question)
        answer = ("This needs to be handled by a member of university staff, so I won't try to "
                  f"resolve it here. Please contact the {ContactDirectory.describe(contact)}.")
        return ChatResult(answer, "escalated", contact=contact)
