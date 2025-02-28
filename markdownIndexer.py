#
# WARNING: This requires pytorch and tesseract to be installed on the system. I am using PyTorch on NVIDIA GPU.
# It is incredibly slow on CPUs to process PDFs with this library.
#
import os
import chromadb
import torch
from chromadb.utils import embedding_functions
import re
from xml.etree import ElementTree as ET
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from dotenv import load_dotenv

# Load environment variables from .env file (if needed for other keys)
load_dotenv()

# Load model dict for Marker (PDF conversion)
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Character-based chunking (fallback)
def chunk_by_size(content, chunk_size=1000):
    """Chunk content by character size."""
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Paragraph-based chunking for text files
def chunk_by_paragraph(content, max_chunk_size=1000):
    """Chunk content by paragraphs, falling back to size if too long."""
    paragraphs = content.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if para:
            if len(para) <= max_chunk_size:
                chunks.append(para)
            else:
                sub_chunks = chunk_by_size(para, max_chunk_size)
                chunks.extend(sub_chunks)
    if not chunks:
        chunks = [content.strip()] if content.strip() else [""]
    return chunks

# Convert .cs content to Markdown with error handling
def cs_to_markdown(file, content):
    """Convert C# code to Markdown with code block formatting."""
    print(f"Converting '{file}' to Markdown")
    try:
        return f"```csharp\n{content}\n```"
    except Exception as e:
        print(f"Error converting {file} to Markdown: {e}")
        return f"```csharp\n[Error: Malformed C# content]\n{content}\n```"

# Convert .xml content to Markdown with error handling
def xml_to_markdown(file, content):
    """Convert XML to Markdown with code block formatting."""
    print(f"Converting '{file}' to Markdown")
    try:
        xml_root = ET.fromstring(content)
        ET.indent(xml_root)
        pretty_xml = ET.tostring(xml_root, encoding="unicode", method="xml")
        return f"```xml\n{pretty_xml}\n```"
    except ET.ParseError as e:
        print(f"Warning: {file} has invalid XML, treating as plain text: {e}")
        return f"```xml\n[Error: Invalid XML]\n{content}\n```"
    except Exception as e:
        print(f"Error converting {file} to Markdown: {e}")
        return f"```xml\n[Error: Malformed content]\n{content}\n```"

# Convert .csproj (XML-based) to Markdown with error handling
def csproj_to_markdown(file, content):
    """Convert .csproj (XML) to Markdown with code block formatting."""
    return xml_to_markdown(file, content)

# Convert .txt to Markdown with error handling
def txt_to_markdown(file, content):
    """Convert plain text to Markdown (minimal formatting)."""
    print(f"Converting '{file}' to Markdown")
    try:
        return content
    except Exception as e:
        print(f"Error converting {file} to Markdown: {e}")
        return f"[Error: Malformed text]\n{content}"

# Parse PDFs with Marker
def chunk_pdf_files(file_path, chunk_size=1000):
    file_name = os.path.basename(file_path)
    print(f"Chunking: '{file_name}' with Marker (full path: {file_path})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        rendered = converter(file_path)
        markdown_text, _, _ = text_from_rendered(rendered)
        return chunk_by_size(markdown_text, chunk_size)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return [""]

# Read and chunk files, converting all to Markdown first
def read_files(directory, cs_chunk_type="class", xml_chunk_type="elements", pdf_chunk_type="pdf", txt_chunk_type="paragraph", chunk_size=1000):
    documents = []
    ids = []
    metadatas = []
    
    print(f"Starting traversal of directory: {directory}")
    for root, dirs, files in os.walk(directory):
        print(f"Entering directory: {root} (subdirs: {dirs}, files: {files})")
        for file in files:
            if file.endswith((".cs", ".csproj", ".xml", ".pdf", ".txt")):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Read and convert to Markdown based on file type
                if file.endswith((".cs", ".csproj", ".xml", ".txt")):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                        content = f"[Error: Unable to read file]\n"
                    
                    if file.endswith(".cs"):
                        markdown_content = cs_to_markdown(file, content)
                    elif file.endswith(".csproj"):
                        markdown_content = csproj_to_markdown(file, content)
                    elif file.endswith(".xml"):
                        markdown_content = xml_to_markdown(file, content)
                    elif file.endswith(".txt"):
                        markdown_content = txt_to_markdown(file, content)
                elif file.endswith(".pdf"):
                    chunks = chunk_pdf_files(file_path, chunk_size)
                    for i, chunk in enumerate(chunks):
                        if chunk:
                            chunk_id = f"{file_path}_chunk_{i}"
                            documents.append(chunk)
                            ids.append(chunk_id)
                            metadatas.append({
                                "source_name": file,
                                "source_url": file_path,
                                "chunk_index": i,
                                "chunk_type": pdf_chunk_type,
                                "file_type": "pdf"
                            })
                    continue
                
                # Chunk the Markdown content based on type
                if file.endswith(".cs"):
                    if cs_chunk_type == "class":
                        pattern = r'(?:(?:public|private|protected|internal|static|abstract|sealed)?\s+)?(?:\[.*?\]\s*)?class\s+\w+\s*(?::\s*[\w<>, ]+)?\s*{'
                        matches = list(re.finditer(pattern, content, re.DOTALL))
                        if not matches:
                            print(f"{file}: No class definitions found, indexing by chunk size.")
                            chunks = chunk_by_size(markdown_content, chunk_size)
                        else:
                            chunks = []
                            start = 0
                            for match in matches:
                                end = match.start()
                                if start < end:
                                    preamble = content[start:end].strip()
                                    if preamble:
                                        chunks.append(f"```csharp\n{preamble}\n```")
                                class_start = match.start()
                                brace_count = 0
                                i = match.end() - 1
                                while i < len(content):
                                    if content[i] == '{':
                                        brace_count += 1
                                    elif content[i] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            class_chunk = content[class_start:i + 1].strip()
                                            chunks.append(f"```csharp\n{class_chunk}\n```")
                                            start = i + 1
                                            break
                                    i += 1
                                else:
                                    class_chunk = content[class_start:].strip()
                                    chunks.append(f"```csharp\n{class_chunk}\n```")
                                    start = len(content)
                            if start < len(content):
                                remaining = content[start:].strip()
                                if remaining:
                                    chunks.append(f"```csharp\n{remaining}\n```")
                    elif cs_chunk_type == "size":
                        chunks = chunk_by_size(markdown_content, chunk_size)
                    else:
                        chunks = [markdown_content]
                elif file.endswith(".xml") or file.endswith(".csproj"):
                    if xml_chunk_type == "elements":
                        try:
                            xml_root = ET.fromstring(content)
                            chunks = []
                            for child in xml_root:
                                if child.tag == "agentEventManager":
                                    for event in child.findall(".//event"):
                                        chunk = ET.tostring(event, encoding="unicode", method="xml").strip()
                                        if chunk:
                                            chunks.append(f"```xml\n{chunk}\n```")
                                elif child.tag == "opcDaMonitor":
                                    for event in child.findall(".//connection"):
                                        chunk = ET.tostring(event, encoding="unicode", method="xml").strip()
                                        if chunk:
                                            chunks.append(f"```xml\n{chunk}\n```")
                                else:
                                    chunk = ET.tostring(child, encoding="unicode", method="xml").strip()
                                    if chunk:
                                        chunks.append(f"```xml\n{chunk}\n```")
                            if not chunks:
                                chunks = [markdown_content]
                        except ET.ParseError:
                            chunks = chunk_by_size(markdown_content, chunk_size)
                    elif xml_chunk_type == "size":
                        chunks = chunk_by_size(markdown_content, chunk_size)
                    else:
                        chunks = [markdown_content]
                else:  # .txt
                    if txt_chunk_type == "paragraph":
                        chunks = chunk_by_paragraph(markdown_content, chunk_size)
                    elif txt_chunk_type == "size":
                        chunks = chunk_by_size(markdown_content, chunk_size)
                    else:
                        chunks = [markdown_content]
                
                # Index the chunks
                for i, chunk in enumerate(chunks):
                    if chunk:
                        chunk_id = f"{file_path}_chunk_{i}"
                        documents.append(chunk)
                        ids.append(chunk_id)
                        metadatas.append({
                            "source_name": file,
                            "source_url": file_path,
                            "chunk_index": i,
                            "chunk_type": cs_chunk_type if file.endswith(".cs") else (xml_chunk_type if file.endswith((".xml", ".csproj")) else (pdf_chunk_type if file.endswith(".pdf") else txt_chunk_type)),
                            "file_type": "cs" if file.endswith(".cs") else ("csproj" if file.endswith(".csproj") else ("pdf" if file.endswith(".pdf") else ("xml" if file.endswith(".xml") else "txt")))
                        })
    
    return documents, ids, metadatas

# Index codebase
def index_codebase(
    codebase_path,
    cs_chunk_type="class",
    xml_chunk_type="elements",
    pdf_chunk_type="pdf",
    txt_chunk_type="paragraph",
    chunk_size=1000,
    use_persistent=False,
    persistent_path="./chroma_db",
    batch_size=160
):
    if use_persistent:
        client = chromadb.PersistentClient(path=persistent_path)
        print(f"Using persistent client at {persistent_path}")
    else:
        client = chromadb.Client()
        print("Using in-memory client")
    
    collection_name = "DEFAULT_COLLECTION"
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Using existing collection.")
        return collection
    except Exception:
        print(f"Collection '{collection_name}' does not exist. Creating a new collection.")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    print(f"Reading and chunking files: cs with '{cs_chunk_type}', .xml/.csproj with '{xml_chunk_type}', .pdf with '{pdf_chunk_type}', .txt with '{txt_chunk_type}'...")
    documents, ids, metadatas = read_files(
        codebase_path, cs_chunk_type, xml_chunk_type, pdf_chunk_type, txt_chunk_type, chunk_size
    )
    
    print(f"Total chunks to index: {len(documents)}")
    
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

# Query the index
def query_index(collection, query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        line_count = len(doc.splitlines())
        
        print(f"\n")
        print(f"Query: {query_text}")
        print(f"Result {i+1}:")
        print(f"File: {meta['source_name']}")
        print(f"Path: {meta['source_url']}")  # Full path including subdirs
        print(f"Chunk Index: {meta['chunk_index']} (Type: {meta['chunk_type']})")
        print(f"File Type: {meta['file_type']}")
        print(f"Chunk Size: {line_count} lines")
        print(f"Content snippet: {doc}...\n")

if __name__ == "__main__":
    codebase_path = "sample_docs/example1"
    cs_chunk_type = "class"
    xml_chunk_type = "elements"
    pdf_chunk_type = "pdf"
    txt_chunk_type = "paragraph"
    chunk_size = 1000
    use_persistent = True
    persistent_path = "./chroma_db"
    collection_name = "DEFAULT_COLLECTION"
    
    collection = index_codebase(
        codebase_path=codebase_path,
        cs_chunk_type=cs_chunk_type,
        xml_chunk_type=xml_chunk_type,
        pdf_chunk_type=pdf_chunk_type,
        txt_chunk_type=txt_chunk_type,
        chunk_size=chunk_size,
        use_persistent=use_persistent,
        persistent_path=persistent_path,
        batch_size=160
    )
    
    query_index(collection, "In PI System Explorer what is attribute configuration string syntax?")
   