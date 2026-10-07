"""Setup checker for the UoM chatbot backend.
Run from the backend/ folder:   python check_setup.py
It tests each part in order and prints what to fix. Paste the output to your team if stuck."""
import importlib
import os
import sys
import time

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = []


def report(status, msg, fix=None):
    print(f"{status} {msg}")
    if fix:
        print(f"       -> {fix}")
    if status == FAIL:
        problems.append(msg)


print("=== 1. Python and folder ===")
v = sys.version_info
if v < (3, 10):
    report(FAIL, f"Python {v.major}.{v.minor} is too old", "Install Python 3.10, 3.11 or 3.12 and recreate the venv")
    sys.exit(1)
report(OK, f"Python {v.major}.{v.minor}.{v.micro}")
if not os.path.isdir("app"):
    report(FAIL, "Not running from the backend/ folder", "cd backend   then run this again")
    sys.exit(1)
report(OK, "Running from backend/")

print("\n=== 2. Packages ===")
for mod, pip_name in [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("httpx", "httpx"),
                      ("dotenv", "python-dotenv"), ("pydantic", "pydantic"), ("chromadb", "chromadb"),
                      ("langchain_chroma", "langchain-chroma"), ("langchain_huggingface", "langchain-huggingface"),
                      ("sentence_transformers", "sentence-transformers")]:
    try:
        m = importlib.import_module(mod)
        report(OK, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        report(FAIL, f"{mod} not importable ({e.__class__.__name__}: {e})",
               f"pip install -r requirements.txt   (or: pip install {pip_name})")
import pydantic
if not pydantic.VERSION.startswith("2"):
    report(FAIL, f"pydantic {pydantic.VERSION} is installed, version 2 is needed", "pip install -U \"pydantic>=2.6\"")
if problems:
    print("\nFix the packages above first, then run this again.")
    sys.exit(1)

print("\n=== 3. Settings (.env) ===")
from app.config import load_settings
s = load_settings()
if not os.path.isfile(".env"):
    report(WARN, ".env not found, so defaults are used (mock LLM)",
           "copy .env.example .env  (Windows)   or   cp .env.example .env  (Mac/Linux)")
else:
    report(OK, ".env found")
masked = (s.llm_api_key[:4] + "..." + s.llm_api_key[-4:]) if len(s.llm_api_key) > 8 else ("(empty)" if not s.llm_api_key else "(set)")
print(f"       LLM_PROVIDER={s.llm_provider}  LLM_MODEL={s.llm_model or '(empty)'}  LLM_API_KEY={masked}")
print(f"       RETRIEVER={s.retriever}  MIN_RELEVANCE={s.min_relevance}")
if s.llm_provider == "mock":
    report(WARN, "LLM_PROVIDER=mock: answers will only copy the top passage, not real answers",
           "set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env")

print("\n=== 4. Knowledge base (Person 1) ===")
from app.retriever import SourceIndex
idx = SourceIndex(s.knowledge_base_dir)
if not idx.by_key:
    report(FAIL, f"No .md files found in {s.knowledge_base_dir}",
           "backend/ must sit in the repo root next to knowledge_base/, or set KNOWLEDGE_BASE_DIR in .env")
else:
    report(OK, f"{len(idx.by_key)} documents in {s.knowledge_base_dir}")

print("\n=== 5. Vector store and embeddings (Person 2) ===")
retriever = None
if s.retriever == "keyword":
    report(WARN, "RETRIEVER=keyword (offline test mode), Person 2's ChromaDB is not used", "set RETRIEVER=chroma in .env")
from app.retriever import ChromaRetriever, KeywordRetriever
if not s.chroma_dir.is_dir():
    report(FAIL, f"Vector store not found at {s.chroma_dir}",
           "from the repo root run: python retrieval_augmented_generation_rag/ingestion_pipeline.py")
else:
    try:
        import chromadb
        col = chromadb.PersistentClient(path=str(s.chroma_dir)).get_collection(s.chroma_collection)
        n = col.count()
        report(OK, f"Collection '{s.chroma_collection}' has {n} chunks")
        if n > 71 and n % 71 == 0:
            report(WARN, f"{n} chunks = {n // 71} copies of the 71 expected: ingestion was run {n // 71} times",
                   "delete db/chroma_db and run the ingestion pipeline once")
    except Exception as e:
        report(FAIL, f"Cannot open collection '{s.chroma_collection}': {e}", "check CHROMA_DIR / CHROMA_COLLECTION, or re-run ingestion")
    try:
        t = time.time()
        print("       loading embedding model (first time downloads ~90 MB)...")
        retriever = ChromaRetriever(s.chroma_dir, s.chroma_collection, s.embedding_model, idx)
        hits = retriever.search("How do I register for modules?", s.top_k)
        report(OK, f"Embedding model + search work ({time.time() - t:.1f}s)")
        for h in hits:
            mark = "kept" if h.score >= s.min_relevance else "BELOW THRESHOLD"
            print(f"       {h.score:.3f} {mark:15} {h.title}")
        if hits and all(h.score < s.min_relevance for h in hits):
            report(WARN, "All results are below MIN_RELEVANCE, so every question will get 'no_info'",
                   "lower MIN_RELEVANCE in .env (e.g. 0.25)")
    except Exception as e:
        report(FAIL, f"Search failed: {e.__class__.__name__}: {e}",
               "check internet access to huggingface.co and that sentence-transformers is installed")
if retriever is None or s.retriever == "keyword":
    retriever = KeywordRetriever(idx)

print("\n=== 6. LLM ===")
from app.llm import LLMError, get_llm
llm = None
try:
    llm = get_llm(s)
    t = time.time()
    reply = llm.generate("Reply with the single word OK.", [{"role": "user", "content": "Test"}])
    report(OK, f"{s.llm_provider} replied in {time.time() - t:.1f}s: {reply[:60]!r}")
except ValueError as e:
    report(FAIL, str(e), "fill in .env")
except LLMError as e:
    msg = str(e)
    fix = "check LLM_API_KEY"
    if "401" in msg or "403" in msg: fix = "API key is wrong, expired or for a different provider"
    elif "404" in msg or "model" in msg.lower(): fix = "LLM_MODEL name is wrong for this provider; copy it exactly from the provider's docs"
    elif "429" in msg: fix = "rate limit / free-tier quota reached; wait or use another key"
    elif "Connect" in msg or "Timeout" in msg: fix = "no internet connection, or wrong LLM_BASE_URL"
    report(FAIL, msg, fix)

print("\n=== 7. Full pipeline ===")
if llm is not None and not problems:
    from app.chatbot import Chatbot
    from app.contacts import ContactDirectory
    from app.sessions import SessionStore
    bot = Chatbot(retriever, llm, ContactDirectory(s.contacts_path), SessionStore(), s.top_k, s.min_relevance)
    for q in ["When is the deadline to register for modules?", "What is the weather tomorrow?"]:
        try:
            r = bot.ask(q, "check")
            report(OK, f"{q!r} -> {r.status} ({r.response_time_ms} ms)")
            print("       " + r.answer[:200].replace("\n", " "))
        except Exception as e:
            report(FAIL, f"{q!r} -> {e.__class__.__name__}: {e}")
else:
    print("       skipped until the problems above are fixed")

print("\n=== Result ===")
print("All checks passed. Start the server: uvicorn app.main:app --reload" if not problems
      else f"{len(problems)} problem(s) found. Fix the [FAIL] lines above, top to bottom.")
