from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def similarity_search (vector_store, query:str):
    results = vector_store.similarity_search_with_score(query = query,
                                             k = 5
                                             )
    
    print(f"User Query: {query}")
    for doc, score in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("Score:", score)

    return results

def max_marginal_relevance_search(vector_store, query:str):
    results = vector_store.max_marginal_relevance_search(query, k=5, fetch_k=10)

    for i, doc in enumerate(results):
        print(f"\n{'='*60}")
        print(f"RESULT {i+1}")
        print(f"METADATA: {doc.metadata}")
        print(f"CONTENT:\n{doc.page_content}")

def main():
    print("---- RAG Document Retrieval ---- \n")

    persistent_vector_store_dir = Path("db/chroma_db")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(embedding_function = embedding_model,
                          persist_directory = persistent_vector_store_dir,
                          collection_metadata={"hnsw:space": "cosine"} #the algorithm the database will use to retrieve similar results
                          )
    in_query = "What are the entry requirements for undergraduate programmes?"
    similarity_search_results = max_marginal_relevance_search(vector_store, in_query)

    print("Retrieval Complete!")
    return similarity_search_results


if __name__ == "__main__":
    main()
