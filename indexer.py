import os
import chromadb
from chromadb.utils import embedding_functions
import re

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Function to chunk by size
def chunk_by_size(content, chunk_size=1000):
    """Chunk content by character size."""
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Function to chunk by lines
def chunk_by_lines(content, lines_per_chunk=50):
    """Chunk content by number of lines."""
    lines = content.splitlines()
    return ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

# Function to chunk by function (C# method detection)
def chunk_by_function(content):
    """Chunk content by C# function/method definitions."""
    # Simple regex to detect method signatures (public/private/protected, etc.)
    pattern = r'(?:(?:public|private|protected|internal|static|async)\s+)?(?:\w+\s+)?\w+\s*\([^)]*\)\s*{'
    matches = list(re.finditer(pattern, content))
    if not matches:
        return [content]  # Fallback to whole content if no functions found
    
    chunks = []
    start = 0
    for match in matches:
        end = match.start()
        if start < end:
            chunks.append(content[start:end].strip())
        chunks.append(content[end:match.end()].strip())  # Include the method signature
        start = match.end()
    
    # Add remaining content after last function
    if start < len(content):
        chunks.append(content[start:].strip())
    
    return chunks

# Function to chunk by class
def chunk_by_class(content):
    """Chunk content by C# class definitions."""
    # Regex to detect class declarations
    pattern = r'(?:(?:public|private|protected|internal)\s+)?class\s+\w+\s*(?::\s*\w+)?\s*{'
    matches = list(re.finditer(pattern, content))
    if not matches:
        return [content]  # Fallback to whole content if no classes found
    
    chunks = []
    start = 0
    for match in matches:
        end = match.start()
        if start < end:
            chunks.append(content[start:end].strip())
        chunks.append(content[end:match.end()].strip())  # Include the class declaration
        start = match.end()
    
    # Add remaining content after last class
    if start < len(content):
        chunks.append(content[start:].strip())
    
    return chunks

# Function to read and chunk C# files
def read_cs_files(directory, chunk_type="size", chunk_size=1000, lines_per_chunk=50):
    documents = []
    ids = []
    metadatas = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cs", ".csproj", ".xml")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Apply chunking based on file type
                    if file.endswith(".cs"):
                        if chunk_type == "size":
                            chunks = chunk_by_size(content, chunk_size)
                        elif chunk_type == "function":
                            chunks = chunk_by_function(content)
                        elif chunk_type == "class":
                            chunks = chunk_by_class(content)
                        else:
                            chunks = [content]
                    else:  # .csproj and .xml
                        if chunk_type == "size":
                            chunks = chunk_by_size(content, chunk_size)
                        elif chunk_type == "lines":
                            chunks = chunk_by_lines(content, lines_per_chunk)
                        else:
                            chunks = [content]
                    
                    for i, chunk in enumerate(chunks):
                        if chunk:
                            chunk_id = f"{file_path}_chunk_{i}"
                            documents.append(chunk)
                            ids.append(chunk_id)
                            metadatas.append({
                                "filename": file,
                                "path": file_path,
                                "chunk_index": i,
                                "chunk_type": chunk_type if file.endswith(".cs") else ("size" if chunk_type == "size" else ("lines" if chunk_type == "lines" else "none")),
                                "file_type": "cs" if file.endswith(".cs") else ("csproj" if file.endswith(".csproj") else "xml")
                            })
    
    return documents, ids, metadatas

# Main indexing function
def index_codebase(
    codebase_path,
    chunk_type="size",
    chunk_size=1000,
    lines_per_chunk=50,
    use_persistent=False,
    persistent_path="./chroma_db"
):
    if use_persistent:
        client = chromadb.PersistentClient(path=persistent_path)
        print(f"Using persistent client at {persistent_path}")
    else:
        client = chromadb.Client()
        print("Using in-memory client")
    
    collection = client.get_or_create_collection(
        name="csharp_codebase",
        embedding_function=embedding_function
    )
    
    print(f"Reading and chunking files with chunk_type='{chunk_type}'...")
    documents, ids, metadatas = read_cs_files(codebase_path, chunk_type, chunk_size, lines_per_chunk)
    
    print(f"Indexing {len(documents)} chunks...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print("Indexing complete!")
    
    return collection

# Query the index
def query_index(collection, query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"Result {i+1}:")
        print(f"File: {meta['filename']}")
        print(f"Path: {meta['path']}")
        print(f"Chunk Index: {meta['chunk_index']} (Type: {meta['chunk_type']})")
        print(f"Content snippet: {doc[:200]}...\n")

if __name__ == "__main__":
    # Configuration
    codebase_path = "sample_docs/example1"  # Replace with your actual path
    chunk_type = "class"  # Options: "size", "function", "class"
    chunk_size = 1000  # Only used if chunk_type is "size"
    lines_per_chunk = 50  # Used for "lines", e.g., 1500 lines / 50 = 30 chunks
    use_persistent = True  # Set to False for in-memory
    persistent_path = "./chroma_db"  # Directory for persistent storage
    
    # Index the codebase
    collection = index_codebase(
        codebase_path=codebase_path,
        chunk_type=chunk_type,
        chunk_size=chunk_size,
        lines_per_chunk=lines_per_chunk,
        use_persistent=use_persistent,
        persistent_path=persistent_path
    )
    
    query_index(collection, "What types does DataAgent allow?")