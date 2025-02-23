import os
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
client = chromadb.Client()

# Define an embedding function (using default sentence transformer)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get a collection
collection = client.create_collection(name="csharp_codebase", embedding_function=embedding_function)

# Path to your C# codebase
codebase_path = "sample_docs/example1"  

# Function to read C# files
def read_cs_files(directory):
    documents = []
    ids = []
    metadatas = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cs"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    ids.append(file_path)  # Unique ID for each file
                    metadatas.append({"filename": file, "path": file_path})
    
    return documents, ids, metadatas

# Index the codebase
def index_codebase():
    print("Reading C# files...")
    documents, ids, metadatas = read_cs_files(codebase_path)
    
    print(f"Indexing {len(documents)} files...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print("Indexing complete!")

# Query the index (example)
def query_index(query_text):
    results = collection.query(
        query_texts=[query_text],
        n_results=3  # Return top 3 matches
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"Result {i+1}:")
        print(f"File: {meta['filename']}")
        print(f"Path: {meta['path']}")
        print(f"Content snippet: {doc[:200]}...\n")

if __name__ == "__main__":
    index_codebase()
    # Example query
    query_index("How is dependency injection implemented?")