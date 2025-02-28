# RAG-Index

A Retrieval-Augmented Generation (RAG) indexing and querying system for processing and analyzing files. This project supports indexing various file types (`.cs`, `.csproj`, `.xml`, `.pdf`, `.txt`) into a ChromaDB vector store and querying them using OpenAI's LLM, with options for different PDF parsing methods (Marker, LlamaParse, PyPDF2).

## Features

- **File Indexing**: Recursively processes files in a directory, including subdirectories, and indexes them into ChromaDB.
- **Multiple Parsing Options**:
  - `index-marker.py`: Uses Marker (GPU-accelerated with PyTorch) to convert PDFs to Markdown, with smart chunking for `.txt` (paragraphs) and structured chunking for `.cs` (classes) and `.xml` (elements).
  - `index-llama.py`: Uses LlamaParse (cloud-based) for PDF-to-Markdown conversion, with basic chunking for other files.
  - `index-pypdf2.py`: Uses PyPDF2 for local PDF text extraction, with similar chunking strategies.
  - `index-marker-no-md.py`: Uses Marker for PDFs without Markdown conversion for other files.
- **Querying**: Retrieves relevant chunks from ChromaDB and queries OpenAI's GPT-3.5-turbo for detailed answers (`query-openai.py`).
- **Custom Chunking**:
  - `.cs`: By class definitions or size.
  - `.xml`/`.csproj`: By XML elements or size.
  - `.pdf`: By character size (Marker/LlamaParse adds Markdown structure).
  - `.txt`: By paragraphs or size (Marker version).
- **Error Handling**: Robust handling for malformed files with fallbacks.
- **GPU Support**: Marker leverages NVIDIA GPU via PyTorch for faster PDF processing.

## Prerequisites

- **Python 3.9+**
- **Dependencies**:
  - `chromadb`
  - `sentence-transformers` (for embeddings)
  - `python-dotenv` (for `.env` file)
  - Marker version: `torch`, `marker-pdf`, `tesseract-ocr`
  - LlamaParse version: `llama-index-core`, `llama-index-readers-llama-parse`
  - PyPDF2 version: `PyPDF2`
  - OpenAI version: `openai`
- **Environment**:
  - For Marker: NVIDIA GPU with CUDA and Tesseract installed.
  - For LlamaParse: Llama Cloud API key in `.env`.
  - For OpenAI: OpenAI API key in `.env`.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/rag-index.git
   cd rag-index

2. **Create a Virtual Environment**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows

3. **Install Dependencies**:

  **For Marker**:
  ```bash
  pip install -r requirements-marker.txt
  ```

  **For LlamaParse**:
  ```bash
  pip install -r requirements-llama.txt
  ```

  **For PyPDF2**:
  ```bash
  pip install -r requirements-pypdf2.txt
  ```

  **For OpenAI**:
  ```bash
  pip install openai
  ```
   
4. **Set Up Environment Variables**:
    - Create a `.env` file in the root directory.
    - Add the following variables:
      `LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key  # For LlamaParse `
      `OPENAI_API_KEY=your_openai_api_key           # For OpenAI querying `

5. **Install Tesseract** (for Marker):
    - Windows: Download from Tesseract GitHub and add to PATH.
    - Linux: sudo apt install tesseract-ocr
    - Mac: brew install tesseract

## Usage

**Indexing Files**

  - Run one of the indexing scripts to process files in sample_docs/example1 and its subdirectories:

    - Marker (Recommended, GPU-accelerated):
      ```bash
      python index-marker.py

    Converts all files to Markdown, uses paragraph chunking for .txt.

  - LlamaParse (Cloud-based):
    ```bash
    python index-llama.py
    ```
    PDFs to Markdown via LlamaParse, basic chunking for others.

  - PyPDF2 (Local, lightweight):
    ```bash
    python index-pypdf2.py
    ```
    Basic text extraction for PDFs.

  - Marker without Markdown:
    ```bash
    python index-marker-no-md.py
    ``` 
    PDFs to Markdown, raw text for others.

* Output is stored in ./chroma_db as a persistent ChromaDB collection named DEFAULT_COLLECTION.

## Querying with OpenAI

  - Run the querying script to retrieve and answer questions:
    ```bash
    python query-openai.py
    ```
  - Example output:

    Query: In PI System Explorer how do you reference the sibling element?

    Retrieved Chunks:
    Chunk 1:
    File: example.xml
    Path: sample_docs/example1/subdir/example.xml
    Chunk Size: 10 lines
    Content snippet: ```xml
    <element>...</element>
    ```
    OpenAI LLM Answer:
    In PI System Explorer, to reference a sibling element...

    Customize `test_queries` in `query-openai.py` for different questions.

## Project Structure

- `LICENSE`: MIT License.
- `index-marker.py`: GPU-accelerated indexing with Marker and Markdown conversion.
- `index-llama.py`: Cloud-based PDF parsing with LlamaParse.
- `index-pypdf2.py`: Local PDF parsing with PyPDF2.
- `index-marker-no-md.py`: Marker for PDFs without Markdown for other files.
- `query-openai.py`: RAG querying with OpenAI LLM.
- `.env`: Environment variables (not tracked).
- `sample_docs/example1/`: Directory for test files (not included).

## Contributing

Feel free to fork, submit issues, or PRs. Ensure changes maintain compatibility with ChromaDB and the specified file types.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

© 2025 Jonathan Grocott