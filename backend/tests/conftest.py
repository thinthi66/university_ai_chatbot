"""Tests run offline: keyword retrieval over the real knowledge_base/ and the mock LLM.
The Chroma adapter is tested separately against Person 2's real vector store."""
import os

os.environ["RETRIEVER"] = "keyword"
os.environ["LLM_PROVIDER"] = "mock"
