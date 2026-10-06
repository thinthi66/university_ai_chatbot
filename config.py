"""Central configuration. Secrets come from environment variables (.env), never from code.

Default paths assume this `backend/` folder sits in the root of the team repository:
    <repo>/knowledge_base/                     <- Person 1
    <repo>/db/chroma_db/                       <- Person 2 (built by ingestion_pipeline.py)
    <repo>/retrieval_augmented_generation_rag/ <- Person 2
    <repo>/backend/                            <- Person 3 (this folder)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")


@dataclass
class Settings:
    # --- LLM (Person 3) ---
    llm_provider: str = "mock"      # mock | cohere | anthropic | openai (any OpenAI-compatible API)
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout: float = 30.0
    llm_max_tokens: int = 600
    llm_temperature: float = 0.1
    # --- Retrieval (must match Person 2's ingestion_pipeline.py) ---
    retriever: str = "chroma"       # chroma (Person 2's vector DB) | keyword (offline fallback)
    chroma_dir: Path = REPO_ROOT / "db" / "chroma_db"
    chroma_collection: str = "langchain"   # LangChain's default name, used by Chroma.from_documents
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_base_dir: Path = REPO_ROOT / "knowledge_base"
    top_k: int = 4
    min_relevance: float = 0.35     # cosine relevance (0-1); tune with Person 2 on real questions
    # --- Chat behaviour ---
    max_question_chars: int = 1000
    history_turns: int = 3
    session_ttl_seconds: int = 1800
    contacts_path: Path = BACKEND_DIR / "data" / "contacts.json"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])


def _path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


def load_settings() -> Settings:
    e = os.getenv
    d = Settings()
    return Settings(
        llm_provider=e("LLM_PROVIDER", d.llm_provider).lower(),
        llm_model=e("LLM_MODEL", ""),
        llm_api_key=e("LLM_API_KEY", ""),
        llm_base_url=e("LLM_BASE_URL", ""),
        llm_timeout=float(e("LLM_TIMEOUT", "30")),
        llm_max_tokens=int(e("LLM_MAX_TOKENS", "600")),
        llm_temperature=float(e("LLM_TEMPERATURE", "0.1")),
        retriever=e("RETRIEVER", d.retriever).lower(),
        chroma_dir=_path("CHROMA_DIR", d.chroma_dir),
        chroma_collection=e("CHROMA_COLLECTION", d.chroma_collection),
        embedding_model=e("EMBEDDING_MODEL", d.embedding_model),
        knowledge_base_dir=_path("KNOWLEDGE_BASE_DIR", d.knowledge_base_dir),
        top_k=int(e("TOP_K", "4")),
        min_relevance=float(e("MIN_RELEVANCE", str(d.min_relevance))),
        max_question_chars=int(e("MAX_QUESTION_CHARS", "1000")),
        history_turns=int(e("HISTORY_TURNS", "3")),
        session_ttl_seconds=int(e("SESSION_TTL_SECONDS", "1800")),
        allowed_origins=[o.strip() for o in e(
            "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500").split(",") if o.strip()],
    )
