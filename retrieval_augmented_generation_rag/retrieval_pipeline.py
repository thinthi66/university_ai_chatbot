from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def similarity_search (vector_store, query:str):
    results = vector_store.similarity_search_with_score(query=query, k=5)
    
    print(f"User Query: {query}")
    for i, (doc, score) in enumerate(results):
        print(f"RESULT {i+1}")
        print(f"Content: {doc.page_content}")
        print("Score:", score)
        print(f"SOURCE: {doc.metadata.get('source')}, SECTION: {doc.metadata.get('Header 1')} > {doc.metadata.get('Header 2')}")

    return results

def max_marginal_relevance_search(vector_store, query:str):
    results = vector_store.max_marginal_relevance_search(query, k=5, fetch_k=10)

    print(f"User Query: {query}")
    for i, doc in enumerate(results):
        print(f"RESULT {i+1}")
        print(f"CONTENT: {doc.page_content}")
        print(f"SOURCE: {doc.metadata.get('source')}, SECTION: {doc.metadata.get('Header 1')} > {doc.metadata.get('Header 2')}")
        

def main():
    print("---- RAG Document Retrieval ---- \n")

    persistent_vector_store_dir = Path("db/chroma_db")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(embedding_function = embedding_model,
                          persist_directory = persistent_vector_store_dir,
                          collection_metadata={"hnsw:space": "cosine"} #the algorithm the database will use to retrieve similar results
                          )
    
    #Q1: "What are the entry requirements for undergraduate programmes?"
    #Q2: "Are there any visa requirements?"
    #Q3: "Are there any module registration requirements?"

    in_query = "Are there any module registration requirements?"
    similarity_search_results = similarity_search(vector_store, in_query)

    print("Retrieval Complete!")
    return similarity_search_results


if __name__ == "__main__":
    main()
