# University of Mauritius AI Assistant: LLM & Backend (Person 3)

This `backend/` folder goes in the **root of the team repository**, next to Person 1's
`knowledge_base/` and Person 2's `retrieval_augmented_generation_rag/` and `db/`:

```
university_ai_chatbot/
├── knowledge_base/                       Person 1: cleaned Markdown documents
├── retrieval_augmented_generation_rag/   Person 2: ingestion_pipeline.py
├── db/chroma_db/                         Person 2: vector store (built by the pipeline)
└── backend/                              Person 3: this folder
```

## Run it
```bash
# 1. From the repo root, build the vector store once (Person 2's pipeline)
python retrieval_augmented_generation_rag/ingestion_pipeline.py

# 2. Start the backend
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # set LLM_PROVIDER / LLM_MODEL / LLM_API_KEY
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs  ·  Tests: `pytest -v` (offline, no API key needed)

**Not working?** Run `python check_setup.py` from `backend/`. It checks Python, packages, `.env`,
the knowledge base, the vector store, the embedding model and the LLM in order, and prints the
fix for each failure.

The first start downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB).

## How it connects to the team's work
| From | What the backend uses | Where |
|---|---|---|
| Person 1 | `knowledge_base/*.md`: document titles and official source URLs | `app/retriever.py` (`SourceIndex`) |
| Person 1 | Verified contact details (`contacts/`, `international_students/contacts.md`, `campus_resources/`) | `data/contacts.json` |
| Person 2 | `db/chroma_db`, collection `langchain`, `all-MiniLM-L6-v2`, cosine space | `app/retriever.py` (`ChromaRetriever`) |
| Person 4 | `POST /chat` API contract (below) | `app/main.py` |

**Source mapping.** Person 2's chunks keep only their Markdown headers (`Header 1/2/3`), not
the file path. `SourceIndex` therefore matches each chunk's `Header 1` (the document title) to the
file in `knowledge_base/`, and reads the URL from that file's `**Source:**` line. If Person 2
later stores `metadata["source"]`, it is used automatically.

## How a question is handled
1. **Personal problem?** -> reply with the right office's contact (no LLM call).
2. **Follow-up?** -> LLM rewrites it as a standalone question for retrieval.
3. **Retrieve** top `TOP_K` chunks from Chroma; drop chunks below `MIN_RELEVANCE`.
   Nothing left -> fallback + contact. **The LLM is never asked to guess.**
4. **Generate** from the numbered chunks only, citing [1], [2]. `INSUFFICIENT_INFO` -> fallback.
5. **Return** answer, cited sources (title + official URL), status and response time.

## API contract (for Person 4)
`POST /chat` with `{ "question": "...", "session_id": "optional" }` returns:
```json
{
  "answer": "...",
  "status": "answered | clarify | no_info | escalated",
  "sources": [{ "id": 1, "title": "Module Registration — University of Mauritius", "url": "https://..." }],
  "contact": { "department": "International Office", "email": "iso@uom.ac.mu", "phone": "+230 403 7810", "url": "..." },
  "session_id": "send back with the next question",
  "response_time_ms": 850
}
```
Errors: **422** empty/too-long question · **503** AI service unavailable.
`DELETE /session/{id}` resets a conversation · `GET /health` shows provider and retriever.

## Files
| File | Purpose |
|---|---|
| `app/llm.py` | LLM providers: Cohere, Anthropic, OpenAI-compatible (Gemini, Groq, Ollama), mock |
| `app/prompts.py` | System prompt, numbered context with source URLs, follow-up rewrite prompt |
| `app/retriever.py` | Chroma adapter, source mapping, offline keyword fallback |
| `app/chatbot.py` | Pipeline, fallback and citation handling |
| `app/contacts.py`, `data/contacts.json` | Escalation detection and department routing |
| `app/sessions.py` | Per-user conversation memory |
| `app/main.py` | REST API |
| `tests/` | 13 tests, including one against Person 2's real `chroma_db` |
| `check_setup.py` | Step-by-step setup checker with suggested fixes |

## Security
Keys only in `.env` (git-ignored); provider errors logged server-side, never returned; CORS limited
to `ALLOWED_ORIGINS`; question length capped; prompt-injection rule in the system prompt.
