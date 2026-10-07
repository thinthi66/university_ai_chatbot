from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_files(folder_name:str):
    folder_path = Path(folder_name)

    document_loader = {".txt": (TextLoader, {"encoding": "utf-8"}),
                       ".pdf": (PyPDFLoader, {}),
                       ".md": (TextLoader, {"encoding": "utf-8"})
                       }

    documents = []

    try:
        for extension, (loader_cls, loader_kwargs) in document_loader.items():
            loader = DirectoryLoader(
                path= folder_path,
                glob=f"**/*{extension}",
                loader_cls = loader_cls,
                loader_kwargs= loader_kwargs,
                recursive=True              #Allows to open and read subdirectories
            )

            documents.extend(loader.load())

        #Files presence check
        if len(documents) == 0:
            raise FileNotFoundError(f"Error: No files found in {folder_path}. Please add or create files.")
        else:
            print(f"---- Documents loaded from {folder_path} ---- \n")

            for i, doc in enumerate(documents):  # Show first 2 documents
                print(f"\nDocument {i+1}:")
                print(f"  Source: {doc.metadata['source']}")
                print(f"  Content length: {len(doc.page_content)} characters")
                print(f"  Content preview: {doc.page_content[:100]}...")
                print(f"  metadata: {doc.metadata}")

    except FileNotFoundError:
        print(f"Error: The directory {folder_path} does not exist. Please create directory and add files.")

    except NotADirectoryError:
        print(f"Error: The path '{folder_path}' is a file, not a directory.")

    except Exception as e:
        print(f"An unexpected error occurred while accessing the directory: {e}")
        
    return documents

def chunk_files(documents, chunk_size = 1000, chunk_overlap = 50):
    
    headers_to_split_on = [("#", "Header 1"),
                           ("##", "Header 2"),
                           ("###", "Header 3")
                           ]
    chunks_collection = []

    for doc in documents:
        if ".md" in doc.metadata["source"].lower():
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            text_chunks = text_splitter.split_documents(header_splits)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            text_chunks = text_splitter.split_documents([doc])
        
        chunks_collection.extend(text_chunks)

    if chunks_collection:
        for i, chunk in enumerate(chunks_collection):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)
            
    print("---- Files split into chunks  ---- \n")
    return chunks_collection

def create_vector_store(chunks, persistent_vector_store_dir="db/chroma_db"):

    if len(chunks) == 0:
        print("ERROR: No chunks were created!")
        return None

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(documents = chunks,
                                         embedding = embedding_model,
                                         persist_directory = persistent_vector_store_dir,
                                         collection_metadata={"hnsw:space": "cosine"} #the algorithm the database will use to retrieve similar results
                                         )

    print(f"Vector database store created and saved to {persistent_vector_store_dir}")
    return vector_store

def bm25_retriever (chunks):
    
    if len(chunks) == 0:
        print("ERROR: No chunks were created!")
        return None

    bm25_retriever = BM25Retriever.from_documents(chunks)
    return bm25_retriever

def max_marginal_relevance_search_retrieval (vector_store, query:str):
    results = vector_store.max_marginal_relevance_search(query, k=5, fetch_k=10)
    return results

def reciprocal_rank_fusion(mmr_results, bm25_results, k=60):
    rrf_scores = {}

    for rank, doc in enumerate(mmr_results, start=1):
        doc_id = doc.page_content
        rrf_score = 1/(k + rank)

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
        
        rrf_scores[doc_id]+= rrf_score

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.page_content
        rrf_score = 1/(k + rank)

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
                
        rrf_scores[doc_id] += rrf_score
        
    sorted_results = sorted(rrf_scores, key=rrf_scores.get)

    return sorted_results
    
def main():
    print("---- RAG Document Ingestion ---- \n")

    folder_name = "knowledge_base"
    persistent_vector_store_dir = Path("db/chroma_db")

    #1. Load documents from their directory
    docs = load_files(folder_name)

    #2. Split documents into chunks
    chunks = chunk_files(docs)

    #3. Check if vector store already exists
    if persistent_vector_store_dir.is_file():

        #If it exists load documents from it
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma( persist_directory=persistent_vector_store_dir,
                              embedding_function=embedding_model, 
                              collection_metadata={"hnsw:space": "cosine"}
                              )
    else:
        #If not, create vector store
        vector_store = create_vector_store(chunks, persistent_vector_store_dir)

    #4.Load from bm25_retriever_store
    bm25_retriever_store = bm25_retriever(chunks)

    print("Ingestion Complete!")
    print("---- RAG Document Retrieval ---- \n")

    #5. Input user query
    user_query = input("Please input user query:")

    #6. Retrieve results matching user query
    mmr_search_results = max_marginal_relevance_search_retrieval(vector_store, user_query)
    bm25_results = bm25_retriever_store.invoke(user_query)

    print("\n--- Max Marginal Relevance Search Results ---")
    for doc in mmr_search_results:
        print(doc.metadata)
        print(doc.page_content[:200])

    print("\n--- BM25 Results ---")
    for doc in bm25_results:
        print(doc.metadata)
        print(doc.page_content[:200])

    ranked_results = reciprocal_rank_fusion (mmr_search_results, bm25_results)
    
    print("\n--- Reciprocal Rank Fusion Results ---")
    for i, ranked_result in enumerate(ranked_results):
        print(f"--- Result {i+1} ---")
        print(f"{ranked_result}\n")

    print("Retrieval Complete!")
    return ranked_results

if __name__ == "__main__":
    main()
