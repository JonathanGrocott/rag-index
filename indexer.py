import os
import chromadb
from chromadb.utils import embedding_functions
import re
from xml.etree import ElementTree as ET
import PyPDF2

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Character-based chunking (fallback)
def chunk_by_size(content, chunk_size=1000):
    """Chunk content by character size."""
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Chunk .cs files by class
def chunk_by_class(file, content, chunk_size=1000):
    """Chunk content by C# class definitions."""
    # Updated regex to handle attributes and more flexible spacing
    pattern = r'(?:(?:public|private|protected|internal|static|abstract|sealed)?\s+)?(?:\[.*?\]\s*)?class\s+\w+\s*(?::\s*[\w<>, ]+)?\s*{'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if not matches:
        print(f"{file}: No class definitions found, indexing by chunk size.")
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    chunks = []
    start = 0
    for match in matches:
        end = match.start()
        if start < end:
            # Add preamble or content before the class
            preamble = content[start:end].strip()
            if preamble:
                chunks.append(preamble)
        # Add the class itself (from start of declaration to end of match)
        class_start = match.start()
        # Find the end of the class by counting braces
        brace_count = 0
        i = match.end() - 1  # Start at the opening brace
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    chunks.append(content[class_start:i + 1].strip())
                    start = i + 1
                    break
            i += 1
        else:
            # If no closing brace found, take rest of content
            chunks.append(content[class_start:].strip())
            start = len(content)
    
    # Add any remaining content after the last class
    if start < len(content):
        remaining = content[start:].strip()
        if remaining:
            chunks.append(remaining)
    
    return chunks

# Chunk .xml files by top-level elements
def chunk_by_xml_elements(file, content, chunk_size):
    try:
        root = ET.fromstring(content)
        chunks = []
        for child in root:
            # Special handling for userEvent elements
            if child.tag == "agentEventManager":
                for event in child.findall(".//event"):
                    chunk = ET.tostring(event, encoding="unicode", method="xml").strip()
                    if chunk:
                        chunks.append(chunk)
            else:
                chunk = ET.tostring(child, encoding="unicode", method="xml").strip()
                if chunk:
                    chunks.append(chunk)
        return chunks or [content]
    except ET.ParseError as e:
        print(f"{file}: Failed to parse XML content due to {e}, indexing by chunk size.")
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Read and chunk PDF files
def read_pdf_files(file_path, chunk_size=1000):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        return chunk_by_size(content, chunk_size)

# Read and chunk files
def read_cs_files(directory, cs_chunk_type="class", xml_chunk_type="elements", chunk_size=1000):
    documents = []
    ids = []
    metadatas = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cs", ".csproj", ".xml", ".pdf")):
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    chunks = read_pdf_files(file_path, chunk_size)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        if file.endswith(".cs"):
                            if cs_chunk_type == "class":
                                chunks = chunk_by_class(file, content, chunk_size)
                            elif cs_chunk_type == "size":
                                chunks = chunk_by_size(content, chunk_size)
                            else:
                                chunks = [content]
                        elif file.endswith(".xml"):
                            if xml_chunk_type == "elements":
                                chunks = chunk_by_xml_elements(file, content, chunk_size)
                            elif xml_chunk_type == "size":
                                chunks = chunk_by_size(content, chunk_size)
                            else:
                                chunks = [content]
                        else:  # .csproj
                            if xml_chunk_type == "size":
                                chunks = chunk_by_size(content, chunk_size)
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
                            "chunk_type": cs_chunk_type if file.endswith(".cs") else xml_chunk_type,
                            "file_type": "cs" if file.endswith(".cs") else ("csproj" if file.endswith(".csproj") else "xml")
                        })
    
    return documents, ids, metadatas

# Index codebase
def index_codebase(
    codebase_path,
    cs_chunk_type="class",
    xml_chunk_type="elements",
    chunk_size=1000,
    use_persistent=False,
    persistent_path="./chroma_db",
    batch_size=500  # Add batch size parameter
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
    
    print(f"Reading and chunking files: .cs with '{cs_chunk_type}', .xml/.csproj with '{xml_chunk_type}'...")
    documents, ids, metadatas = read_cs_files(codebase_path, cs_chunk_type, xml_chunk_type, chunk_size)
    
    print(f"Total chunks to index: {len(documents)}")
    
    # Split into batches and add incrementally
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        print(f"Indexing batch {i // batch_size + 1}: {len(batch_docs)} chunks...")
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas
        )
    
    print("Indexing complete!")
    return collection

# Query the index with chunk size in lines
def query_index(collection, query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        # Calculate chunk size in lines
        line_count = len(doc.splitlines())
        
        print(f"Result {i+1}:")
        print(f"File: {meta['filename']}")
        print(f"Path: {meta['path']}")
        print(f"Chunk Index: {meta['chunk_index']} (Type: {meta['chunk_type']})")
        print(f"File Type: {meta['file_type']}")
        print(f"Chunk Size: {line_count} lines")
        print(f"Content snippet: {doc[:200]}...\n")

if __name__ == "__main__":
    codebase_path = "sample_docs/example1"  # Change to your codebase directory
    cs_chunk_type = "class"
    xml_chunk_type = "elements"
    chunk_size = 1000
    use_persistent = True
    persistent_path = "./chroma_db"
    
    collection = index_codebase(
        codebase_path=codebase_path,
        cs_chunk_type=cs_chunk_type,
        xml_chunk_type=xml_chunk_type,
        chunk_size=chunk_size,
        use_persistent=use_persistent,
        persistent_path=persistent_path,
        batch_size=500  # Set batch size here
    )
    
    query_index(collection, "What substitution parameter should i use to retreive the element name?")