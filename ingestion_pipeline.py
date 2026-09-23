from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_files(folder_name:str):
    folder_path = Path(folder_name)

    document_loader = {"txt": (TextLoader, {"encoding": "utf-8"}),
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
            print(f"---- Documents loaded from {folder_path}... ---- \n")

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

def chunk_files(documents, chunk_size = 512, chunk_overlap = 0):
    #chunk_size defines the number of characters in each chunk. chunk_overlap defines the number of overlapping characters to maintain context.
    #Consider using Recursive or Sentence-based because 
    #Breaks down long documents into smaller chunks

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
            #Consider RecursiveCharacterTextSplitter() 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            #split_text() is different from split_documents(). It takes in separator and text as parameters.
            text_chunks = text_splitter.split_documents([doc])
        
        chunks_collection.extend(text_chunks)

    if chunks_collection:
        for i, chunk in enumerate(text_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

    print("---- Files split into chunks  ---- \n")

    return chunks_collection

def create_vector_store(chunks, persistent_vector_store_dir="db/chroma_db"):

    print(f"Number of chunks: {len(chunks)}")

    if len(chunks) == 0:
        print("ERROR: No chunks were created!")
        return None

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(documents = chunks,
                                         embedding = embedding_model,
                                         persist_directory = persistent_vector_store_dir,
                                         collection_metadata={"hnsw:space": "cosine"} #the algorithm the database will use to retrieve similar results
                                         )

    print(f"Vector database store created and saved to {persistent_vector_store_dir}")

    return vector_store

def main():
    print("---- RAG Document Ingestion ---- \n")

    folder_name = "knowledge_base"
    persistent_vector_store_dir = Path("db/chroma_db")


    #1. Load documents from their directory
    docs = load_files(folder_name)

    #2. Split documents into chunks
    chunks = chunk_files(docs)

    #3. Create vector store
    vector_store = create_vector_store(chunks, persistent_vector_store_dir)

    print("Ingestion Complete!")
    return vector_store

if __name__ == "__main__":
    main()
