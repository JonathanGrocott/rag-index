# RAG Index with OpenAI

This project creates a Retrieval-Augmented Generation (RAG) system using ChromaDB and OpenAI's LLM. Its specialized for a C# codebase that indexes `.cs`, `.csproj`, and `.xml` files into a vector database and allows querying the indexed data with natural language questions, leveraging OpenAI for detailed responses.

## Files
- **`indexer.py`**: Indexes a directory of C# code and XML config files into a persistent ChromaDB instance.
- **`openAITest.py`**: Retrieves relevant chunks from the ChromaDB index and queries OpenAI’s LLM for answers.

## Features
- Indexes by chunk size or refined for C# files by class and XML config files by special tags or top-level elements elsewhere.
- Supports persistent storage of the index in ChromaDB.
- Integrates with OpenAI’s GPT models for natural language answers based on retrieved code/config snippets.

## Prerequisites
- Python 3.8+
- Dependencies:
  ```bash
  pip install chromadb sentence-transformers openai
