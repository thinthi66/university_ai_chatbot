"""Connects the backend to Person 2's retrieval system and Person 1's knowledge base.

ChromaRetriever opens the vector store created by Person 2's ingestion_pipeline.py
(db/chroma_db, collection "langchain", all-MiniLM-L6-v2 embeddings, cosine space) and
returns the best-matching chunks with a relevance score from 0 to 1.

Chunks in the vector store only carry their Markdown headers ("Header 1", "Header 2", ...),
not the file they came from. SourceIndex scans Person 1's knowledge_base/ folder and maps each
document title (Header 1) back to its file, category and official source URL, so the backend
can still show students where an answer came from.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

UOM_HOME = "https://www.uom.ac.mu"
# From Person 1's sources.md: used when a file cites "University Regulations" without a link.
REGULATIONS_URL = ("https://www.uom.ac.mu/index.php/study-at-uom/current-students/"
                   "regulations/undergraduate-postgraduate")
_URL = re.compile(r"https?://[^\s)|>*]+")


@dataclass
class Chunk:
    text: str
    title: str        # document title shown to the student (Header 1 of the .md file)
    url: str          # official page the document was built from
    category: str     # file key, e.g. "current_students/module_registration"
    score: float = 0.0


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...


# --------------------------------------------------------------------------------------------
class SourceIndex:
    """Title -> (category, url) for every Markdown file in the knowledge base."""

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.by_title: dict[str, dict] = {}
        self.by_key: dict[str, dict] = {}
        if not self.kb_dir.is_dir():
            return
        for f in sorted(self.kb_dir.rglob("*.md")):
            text = f.read_text(encoding="utf-8")
            first = next((l for l in text.splitlines() if l.startswith("# ")), None)
            key = f.relative_to(self.kb_dir).with_suffix("").as_posix()
            info = {"title": first[2:].strip() if first else f.stem, "category": key,
                    "url": self._source_url(text)}
            self.by_key[key] = info
            self.by_title[info["title"]] = info

    @staticmethod
    def _source_url(text: str) -> str:
        lines = text.splitlines()
        cites_regulations = False
        for i, line in enumerate(lines):          # check every "Source" line in the file
            if "source" in line.lower():
                for l in lines[i:i + 4]:            # URL on the Source line or just below it
                    m = _URL.search(l)
                    if m:
                        return m.group(0).rstrip(".,")
                cites_regulations |= "regulation" in line.lower()
        return REGULATIONS_URL if cites_regulations else UOM_HOME

    def lookup(self, metadata: dict) -> dict:
        # If Person 2 keeps the file path in metadata["source"], use it directly.
        src = metadata.get("source")
        if src:
            p = Path(src).as_posix()
            key = p.split("knowledge_base/", 1)[-1].rsplit(".", 1)[0]
            if key in self.by_key:
                return self.by_key[key]
        title = metadata.get("Header 1", "")
        return self.by_title.get(title, {"title": title or "University of Mauritius",
                                         "category": "", "url": UOM_HOME})


def _chunk_text(content: str, metadata: dict) -> str:
    """The header splitter removes headings from the chunk text; put the section path back
    so the LLM knows what the passage is about (e.g. 'Important Deadlines (typical pattern)')."""
    section = " > ".join(metadata[h] for h in ("Header 2", "Header 3") if metadata.get(h))
    return f"Section: {section}\n{content.strip()}" if section else content.strip()


# --------------------------------------------------------------------------------------------
class ChromaRetriever:
    """Semantic search over Person 2's Chroma vector store."""

    def __init__(self, chroma_dir: Path, collection: str, embedding_model: str,
                 sources: SourceIndex, embedding_function=None):
        from langchain_chroma import Chroma  # imported here so keyword/offline mode needs no ML libs

        if not Path(chroma_dir).is_dir():
            raise FileNotFoundError(
                f"Vector store not found at {chroma_dir}. Run Person 2's ingestion_pipeline.py "
                "from the repository root first, or set CHROMA_DIR in .env.")
        if embedding_function is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store = Chroma(collection_name=collection, persist_directory=str(chroma_dir),
                            embedding_function=embedding_function)
        self.sources = sources

    def search(self, query: str, k: int) -> list[Chunk]:
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        chunks, seen = [], set()
        for doc, score in results:
            if doc.page_content in seen:        # ignore duplicates if ingestion ran twice
                continue
            seen.add(doc.page_content)
            info = self.sources.lookup(doc.metadata)
            chunks.append(Chunk(_chunk_text(doc.page_content, doc.metadata), info["title"],
                                info["url"], info["category"], round(min(max(float(score), 0.0), 1.0), 3)))
        return chunks


# --------------------------------------------------------------------------------------------
_STOP = set("""a an the and or of to in on for at by with from about is are was be can could do does
did i me my we you your it this that what whats which who when where how much many any there will
would should need tell please get want im university mauritius uom""".split())


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w.rstrip("s") if len(w) > 3 else w for w in words if w not in _STOP}


class KeywordRetriever:
    """Offline fallback: keyword overlap over the same knowledge_base/ files, split by Markdown
    headers like Person 2's pipeline. Used for automated tests and when the vector store or
    embedding model is unavailable. Not intended for the final evaluation."""

    def __init__(self, sources: SourceIndex):
        self.sources = sources
        self.chunks: list[tuple[Chunk, set[str]]] = []
        for key, info in sources.by_key.items():
            text = (sources.kb_dir / f"{key}.md").read_text(encoding="utf-8")
            for section, body in self._sections(text):
                if body.strip():
                    meta = {"Header 2": section} if section else {}
                    c = Chunk(_chunk_text(body, meta), info["title"], info["url"], info["category"])
                    self.chunks.append((c, _tokens(info["title"] + " " + section + " " + body)))

    @staticmethod
    def _sections(text: str):
        section, buf = "", []
        for line in text.splitlines():
            if line.startswith("## "):
                yield section, "\n".join(buf)
                section, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line)
        yield section, "\n".join(buf)

    def search(self, query: str, k: int) -> list[Chunk]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for c, toks in self.chunks:
            s = len(q & toks) / len(q)
            if s > 0:
                scored.append(Chunk(c.text, c.title, c.url, c.category, round(s, 3)))
        return sorted(scored, key=lambda c: c.score, reverse=True)[:k]


def build_retriever(s) -> Retriever:
    sources = SourceIndex(s.knowledge_base_dir)
    if s.retriever == "keyword":
        return KeywordRetriever(sources)
    if s.retriever == "chroma":
        return ChromaRetriever(s.chroma_dir, s.chroma_collection, s.embedding_model, sources)
    raise ValueError(f"Unknown RETRIEVER: {s.retriever}")
