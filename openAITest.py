import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI  

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 

# Initialize ChromaDB client and load collection
def load_collection(persistent_path="./chroma_db"):
    client = chromadb.PersistentClient(path=persistent_path)
    collection = client.get_collection(name="DEFAULT_COLLECTION")
    return collection

# Retrieve chunks from ChromaDB
def retrieve_chunks(collection, query, n_results=5, filename_filter=None):
    where_clause = {"filename": filename_filter} if filename_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return chunks, metadatas

# Query OpenAI LLM with retrieved chunks
def query_openai_llm(query, chunks):
    context = "\n\n".join(chunks)
    prompt = (
        "You are an expert in IIoT, AVEVA PI System, and C#."
        "Based on the following code and configuration excerpts, provide a detailed and accurate answer to the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are an expert in IIoT, AVEVA PI System, and C#."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Adjust for longer responses 
        temperature=0.7  
    )
    return response.choices[0].message.content  

# Test the RAG system
def test_rag(query, collection, n_results=5, filename_filter=None):
    print(f"Query: {query}")
    chunks, metadatas = retrieve_chunks(collection, query, n_results, filename_filter)
    
    # Print retrieved chunks for validation
    print("\nRetrieved Chunks:")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        line_count = len(chunk.splitlines())
        print(f"Chunk {i+1}:")
        print(f"File: {meta['source_name']}")
        print(f"Path: {meta['source_url']}")
        print(f"Chunk Size: {line_count} lines")
        print(f"Content snippet: {chunk}...")
    
    # Get LLM response
    answer = query_openai_llm(query, chunks)
    print(f"\nOpenAI LLM Answer:\n{answer}")

if __name__ == "__main__":
    # Load indexed collection
    collection = load_collection("./chroma_db")
    
    # Test queries
    test_queries = [
        "In PI System Explorer how do you reference the sibling element?"
    ]
    
    for query in test_queries:
        # Adjust filename_filter as needed
        #filename_filter = "Example.xml" if "LP_ProcessStart" in query else None
        test_rag(query, collection, n_results=5, filename_filter=None)
        print("\n" + "="*50 + "\n")