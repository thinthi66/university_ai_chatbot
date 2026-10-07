"""Backend tests mapped to the evaluation categories in the project plan (section 7), run on
Person 1's real knowledge base. Run from backend/:  pytest -v"""
import shutil

import pytest
from fastapi.testclient import TestClient

from app.chatbot import Chatbot
from app.config import load_settings
from app.contacts import ContactDirectory
from app.llm import BaseLLM, LLMError, MockLLM
from app.main import app
from app.retriever import REGULATIONS_URL, KeywordRetriever, SourceIndex
from app.sessions import SessionStore

S = load_settings()
client = TestClient(app)
if not S.knowledge_base_dir.is_dir():
    pytest.skip(f"knowledge_base/ not found at {S.knowledge_base_dir}", allow_module_level=True)


def make_bot(llm: BaseLLM) -> Chatbot:
    return Chatbot(KeywordRetriever(SourceIndex(S.knowledge_base_dir)), llm,
                   ContactDirectory(S.contacts_path), SessionStore(), top_k=4, min_relevance=0.3)


class FixedLLM(BaseLLM):
    def __init__(self, reply): self.reply = reply
    def generate(self, system, messages): return self.reply


class BrokenLLM(BaseLLM):
    def generate(self, system, messages): raise LLMError("down")


def ask(question, session_id=None):
    body = {"question": question} | ({"session_id": session_id} if session_id else {})
    return client.post("/chat", json=body).json()


# --- Knowledge base integration -------------------------------------------------------------
def test_source_index_maps_every_document_to_a_url():
    idx = SourceIndex(S.knowledge_base_dir)
    assert len(idx.by_title) >= 15
    assert idx.by_key["fees/tuition_fees"]["url"] == "https://www.uom.ac.mu/fees"
    assert idx.by_key["current_students/module_registration"]["url"] == REGULATIONS_URL


# A. Direct question
def test_direct_question():
    r = ask("What are the general entry requirements for undergraduate degree programmes?")
    assert r["status"] == "answered"
    assert "Entry Requirements" in r["sources"][0]["title"]
    assert r["sources"][0]["url"].startswith("https://")


# C. Multi-part question
def test_multi_part_question():
    r = ask("What is the application fee and the application deadline for international students?")
    assert r["status"] == "answered" and r["sources"]


# D. Follow-up question uses the conversation (same session_id)
def test_follow_up_question():
    first = ask("How do I submit an application?")
    r = ask("What about international students?", first["session_id"])
    assert r["status"] == "answered"
    assert any("International" in s["title"] for s in r["sources"])


# E. Ambiguous question -> clarifying question, no fake source attached
def test_clarifying_question():
    r = make_bot(FixedLLM("Do you mean tuition fees or the application fee?")).ask("How much are the fees?", "e")
    assert r.status == "clarify" and r.sources == []


# F. Out-of-scope question -> LLM not called, fallback with contact
def test_out_of_scope():
    r = ask("What is the weather tomorrow?")
    assert r["status"] == "no_info" and r["contact"]["phone"] == "(230) 403 7400"


# G. Retrieved text does not contain the answer (knowledge_base/library.md has no opening hours)
def test_no_information_in_knowledge_base():
    r = make_bot(FixedLLM("INSUFFICIENT_INFO")).ask("What are the library opening hours?", "g")
    assert r.status == "no_info"
    assert r.contact["department"] == "Library" and r.contact["email"] == "library@uom.ac.mu"


# H. Personal problem -> escalate to the right office with real contact details
def test_human_escalation_registration():
    r = ask("I have a problem with my registration. Who should I contact?")
    assert r["status"] == "escalated"
    assert r["contact"]["email"] == "admission@uom.ac.mu"
    assert "Registry" in r["contact"]["department"]


def test_human_escalation_visa():
    r = ask("My visa application has a problem, who should I contact?")
    assert r["status"] == "escalated" and r["contact"]["department"] == "International Office"


# Contact routing falls back to the knowledge-base category of the best chunk
def test_category_routing():
    contacts = ContactDirectory(S.contacts_path)
    assert contacts.route("hello", "campus_resources/library")["department"] == "Library"
    assert contacts.route("hello", "current_students/examinations")["email"] == "admission@uom.ac.mu"


# Error handling for Person 4
def test_empty_question_rejected():
    assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_llm_down_returns_503_without_leaking_details():
    original = app.state.bot.llm
    app.state.bot.llm = BrokenLLM()
    try:
        resp = client.post("/chat", json={"question": "How do I register for modules?"})
        assert resp.status_code == 503 and "down" not in resp.text
    finally:
        app.state.bot.llm = original


# --- Person 2's real Chroma vector store ----------------------------------------------------
@pytest.mark.filterwarnings("ignore:Relevance scores")
def test_chroma_retriever_reads_person2_vector_store(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    pytest.importorskip("langchain_chroma")
    from langchain_core.embeddings import Embeddings
    from app.retriever import ChromaRetriever

    if not S.chroma_dir.is_dir():
        pytest.skip("db/chroma_db not built yet")
    db = tmp_path / "chroma_db"
    shutil.copytree(S.chroma_dir, db)   # work on a copy so the team's DB is never modified

    # The embedding model can't be downloaded offline, so reuse a vector already stored for
    # the Module Registration document as the "question" vector.
    col = chromadb.PersistentClient(path=str(db)).get_collection(S.chroma_collection)
    got = col.get(where={"Header 1": "Module Registration — University of Mauritius"},
                  include=["embeddings"], limit=1)
    vec = list(got["embeddings"][0])

    class StoredVector(Embeddings):
        def embed_documents(self, texts): return [vec for _ in texts]
        def embed_query(self, text): return vec

    r = ChromaRetriever(db, S.chroma_collection, S.embedding_model,
                        SourceIndex(S.knowledge_base_dir), embedding_function=StoredVector())
    top = r.search("How do I register for modules?", k=4)[0]
    assert top.title == "Module Registration — University of Mauritius"
    assert top.category == "current_students/module_registration"
    assert top.url == REGULATIONS_URL
    assert 0.99 <= top.score <= 1.0
