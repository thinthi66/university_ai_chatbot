"""Backend API (Person 3, Task 6). Run:  uvicorn app.main:app --reload
Interactive docs for Person 4: http://127.0.0.1:8000/docs"""
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .chatbot import Chatbot
from .config import load_settings
from .contacts import ContactDirectory
from .llm import LLMError, get_llm
from .retriever import build_retriever
from .sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = load_settings()


def build_chatbot(s=settings) -> Chatbot:
    return Chatbot(
        retriever=build_retriever(s),   # Person 2's Chroma vector store (or keyword fallback)
        llm=get_llm(s),
        contacts=ContactDirectory(s.contacts_path),
        sessions=SessionStore(s.history_turns, s.session_ttl_seconds),
        top_k=s.top_k,
        min_relevance=s.min_relevance,
    )


app = FastAPI(title="University of Mauritius AI Assistant API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])
app.state.bot = build_chatbot()


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = Field(None, max_length=64)

    @field_validator("question")
    @classmethod
    def check_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > settings.max_question_chars:
            raise ValueError(f"Question is too long (max {settings.max_question_chars} characters).")
        return v


class Source(BaseModel):
    id: int
    title: str
    url: str


class Contact(BaseModel):
    department: str
    email: str | None = None
    phone: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    status: str
    sources: list[Source]
    contact: Contact | None
    session_id: str
    response_time_ms: int


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Plain `def` (not async): FastAPI runs each request in a worker thread,
    # so several students can be served at the same time (FR11).
    sid = req.session_id or uuid.uuid4().hex
    try:
        r = app.state.bot.ask(req.question, sid)
    except LLMError:
        logging.getLogger("api").exception("LLM failure")
        # Generic message only: never expose provider errors, keys or internals (Security NFR).
        raise HTTPException(503, "The AI service is temporarily unavailable. Please try again shortly.")
    return ChatResponse(answer=r.answer, status=r.status, sources=r.sources, contact=r.contact,
                        session_id=sid, response_time_ms=r.response_time_ms)


@app.delete("/session/{session_id}")
def reset_session(session_id: str):
    app.state.bot.sessions.clear(session_id)
    return {"cleared": True}


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider, "retriever": settings.retriever}
