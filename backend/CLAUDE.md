# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STEP-gpt is a RAG (Retrieval-Augmented Generation) application for querying medical textbook content (First Aid for USMLE Step 1). The backend is a FastAPI application that uses ChromaDB for vector storage, HuggingFace embeddings, and Google's Gemini AI for both image description during ingestion and answering user queries.

## Architecture

### Two-Phase System

1. **Ingestion Phase** (`app/ingest.py`): Processes the PDF textbook once to populate the vector database
   - Extracts text and images from PDF using PyMuPDF (fitz)
   - Chunks text using RecursiveCharacterTextSplitter (2000 chars, 200 overlap)
   - Describes images concurrently using Gemini 2.5 Flash (5 workers)
   - Embeds all content using sentence-transformers/all-MiniLM-L6-v2
   - Stores in ChromaDB at `./chroma_db`

2. **API Phase** (`app/main.py`): Serves queries via FastAPI endpoints
   - `/chat`: Answers questions with citations from the textbook
   - `/notes`: Generates structured study notes on topics
   - Both use retrieval with k=5 (chat) or k=10 (notes) from ChromaDB
   - Gemini 2.5 Flash generates responses with mandatory citations

### Key Design Patterns

**Contextualized Content**: All chunks (text and image descriptions) are prefixed with metadata:
```
[Section: {chapter} | Page {page_num}]

{actual content}
```

This allows the LLM to provide accurate page/chapter citations in responses.

**Metadata Structure**: Each document has:
- `page`: Page number (int)
- `chapter`: Chapter name (str)
- `source`: "textbook"
- `type`: "text" or "figure"

**Concurrent Image Processing**: Uses ThreadPoolExecutor with 5 workers to describe images in parallel during ingestion, significantly reducing processing time.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the FastAPI server (from backend directory)
uvicorn app.main:app --reload

# The API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Ingestion
```bash
# Run the ingestion script to populate ChromaDB
python -m app.ingest

# This will:
# - Clear existing chroma_db directory
# - Extract ~800 pages of text and images
# - Describe images using Gemini API (concurrent)
# - Embed everything and store in ChromaDB
# - Takes significant time due to image processing
```

### Diagnostic Tools
```bash
# Check retrieval quality and database contents
python diagnose.py

# Test available Gemini embedding models
python test.py
```

## Environment Variables

Required in `.env` file:
- `GEMINI_API_KEY`: Google Gemini API key for image description and chat responses

## Important Constants

**In `app/ingest.py`:**
- `PDF_PATH`: Hardcoded path to the textbook PDF (currently `/Users/akhil/Downloads/STEP-textbook.pdf`)
- `CHROMA_PATH`: `./chroma_db`
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `SKIP_CHAPTERS`: Set of chapter names to exclude during ingestion (index, TOC, acknowledgments, etc.)

**In `app/main.py`:**
- `CHROMA_PATH`: `./chroma_db` (must match ingest.py)
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2` (must match ingest.py)
- Model: `gemini-2.5-flash` with temperature 0.2

## Working with Paths

When modifying ingestion:
- The PDF path in `app/ingest.py:18` is hardcoded and user-specific
- To support different users, this should be parameterized or use environment variables
- The ChromaDB path is relative (`./chroma_db`) and should work across environments

## Testing Queries

The `diagnose.py` script includes sample test queries:
- "What is chapter 1 about?"
- "What are the key topics covered in the first chapter?"
- "Explain the mechanism of action of beta blockers"
- "What are gram positive bacteria?"

Use this as a reference for typical medical queries the system should handle.

## CORS Configuration

The API allows all origins (`allow_origins=["*"]`) in `app/main.py:20`. This is suitable for development but should be restricted for production deployments.

## Response Format

**Chat endpoint** returns:
```json
{
  "answer": "string with inline citations [Page X, Chapter Y]",
  "sources": [
    {
      "page": 42,
      "chapter": "Chapter Name",
      "snippet": "first 200 chars..."
    }
  ]
}
```

**Notes endpoint** returns:
```json
{
  "notes": "structured markdown notes with citations"
}
```

## Deployment Notes

The frontend exists in `../frontend` as a Next.js application. When making changes to the API:
- Ensure endpoint contracts remain compatible
- The frontend expects the exact JSON structure shown above
- API runs on localhost:8000 by default; frontend likely configures this via environment variables
