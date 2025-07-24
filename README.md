# Knowledge Base Aware Question Answering System

A Retrieval Augmented Generation (RAG) system that answers questions based on a collection of PDF documents with metadata. The system extracts text from PDFs, creates embeddings, stores them in a vector database, and uses a local LLM to generate answers grounded in the provided documents.

## Features

- **Document Processing**: Extracts text from PDF files and associates them with metadata
- **Vector Search**: Uses semantic similarity search to find relevant document chunks
- **RAG Pipeline**: Combines retrieval with local LLM generation for accurate answers
- **Web Interface**: Clean, modern web UI for asking questions and viewing results
- **Metadata Integration**: Leverages document metadata for enhanced search and source attribution
- **Local LLM**: Uses Ollama for privacy-friendly, local answer generation

## Project Structure

```
.
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # PDF files and metadata
│   └── metadata.jsonl
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # PDF processing and text chunking
│   └── rag_system.py       # RAG pipeline and vector store
├── templates/
│   └── index.html          # Web interface
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py
└── vector_store/           # ChromaDB persistent storage
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or pip
- Ollama (for LLM functionality)

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd isi-market-task

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Install and Setup Ollama (Optional but recommended)

For full functionality including answer generation:

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull a model (e.g., Llama 3.2)
ollama pull llama3.2
```

Without Ollama, the system will still work for document search but won't generate answers.

## Usage

### 1. Process Documents

Before using the system, you need to process the PDF documents and create embeddings:

```bash
# Activate virtual environment
source .venv/bin/activate

# Process all documents (this may take several minutes)
python -c "from src.rag_system import RAGSystem; rag = RAGSystem(); rag.process_and_embed_all_documents()"
```

### 2. Start the Web Application

**Option 1: Smart Launcher (Recommended)**
```bash
# This will automatically check if the database is ready and start the app
python start_app.py
```

**Option 2: Manual Start**
```bash
# Start the Flask web server directly
python app.py
```

The application will be available at `http://localhost:5000`

### 3. Using the Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Check the system status indicator (should be green when ready)
3. Enter your question in the text area
4. Click "Ask Question" to get an answer with sources

### 4. API Usage

The system also provides REST API endpoints:

#### Check System Status
```bash
curl http://localhost:5000/api/status
```

#### Ask a Question
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GDP growth in the US?"}'
```

#### Search Documents (without answer generation)
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "economic growth", "k": 3}'
```

## System Components

### Data Processing (`src/data_processing.py`)

- Loads metadata from `data/metadata.jsonl`
- Extracts text from PDF files using `pdfplumber`
- Chunks text while preserving metadata associations
- Handles 500 PDF documents with various content types

### RAG System (`src/rag_system.py`)

- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB for persistent embedding storage
- **Retrieval**: Semantic similarity search with metadata filtering
- **Generation**: Local LLM via Ollama for answer generation

### Web Application (`app.py`)

- Flask-based web server with REST API
- Status monitoring and health checks
- Error handling and logging
- Modern, responsive web interface

## Configuration

### Environment Variables

You can customize the system using environment variables:

```bash
# Vector store configuration
export VECTOR_STORE_DIR="custom_vector_store"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Ollama configuration
export OLLAMA_HOST="localhost"
export OLLAMA_PORT="11434"
export OLLAMA_MODEL="llama3.2"
```

### Chunk Size Configuration

Modify chunk settings in `src/data_processing.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust chunk size
    chunk_overlap=200,  # Adjust overlap
    length_function=len,
)
```

## Testing

Run the test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests
pytest tests/ -v
```

## Performance Considerations

- **Initial Processing**: Processing 500 PDFs takes 5-15 minutes depending on hardware
- **Memory Usage**: Expect 2-4GB RAM usage during processing
- **Storage**: Vector store requires ~100-500MB for the full dataset
- **Query Speed**: Typical response time is 1-3 seconds per question

## Troubleshooting

### Common Issues

1. **"No documents in vector store"**
   - Run the document processing step first
   - Check that PDF files exist in the `data/` directory

2. **"Ollama service not available"**
   - Install and start Ollama service
   - Pull a compatible model (e.g., `ollama pull llama3.2`)

3. **PDF processing errors**
   - Some PDFs may be corrupted or password-protected
   - Check logs for specific error messages

4. **Out of memory during processing**
   - Reduce batch size in processing
   - Process documents in smaller chunks

### Logs

The system uses Python logging. To see detailed logs:

```bash
export PYTHONPATH="."
python app.py
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/status` | GET | System status |
| `/api/ask` | POST | Ask question |
| `/api/search` | POST | Search documents |
| `/api/process_documents` | POST | Process all documents |

### Request/Response Formats

#### `/api/ask`

**Request:**
```json
{
  "question": "What is GDP growth?",
  "k": 5,
  "filters": {"country_codes": ["US"]}
}
```

**Response:**
```json
{
  "answer": "Based on the provided context...",
  "sources": [
    {
      "title": "GDP Report Q3 2024",
      "date": "2024-10-15",
      "similarity_score": 0.85
    }
  ],
  "question": "What is GDP growth?"
}
```
