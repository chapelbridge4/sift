# Brain-Inspired RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system inspired by human brain architecture, featuring modular cognitive components that work together to provide intelligent document retrieval and question answering.

## Overview

This system implements a brain-inspired architecture for RAG, with specialized modules that mirror cognitive functions:

- **Hippocampus**: Memory formation, indexing, and retrieval
- **Amygdala**: Importance scoring and emotional salience
- **Prefrontal Cortex**: Executive control and reasoning
- **Working Memory**: Contextual buffer for conversations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Prefrontal Cortex (PFC)                         │
│         Executive Control & Orchestration                    │
└─┬──────────────┬──────────────┬──────────────┬─────────────┘
  │              │              │              │
  │              │              │              │
┌─▼────────┐ ┌──▼───────┐ ┌───▼────────┐ ┌───▼──────────┐
│Hippocampus│ │ Amygdala │ │Working     │ │LLM Service   │
│           │ │          │ │Memory      │ │(Ollama)      │
│Memory     │ │Importance│ │Context     │ │              │
│Storage &  │ │Scoring   │ │Buffer      │ │llama3.2:3b   │
│Retrieval  │ │          │ │            │ │              │
└─┬─────────┘ └──────────┘ └────────────┘ └──────────────┘
  │
┌─▼──────────────────────────────────────────────────────────┐
│                  Qdrant Vector Database                      │
│         Hybrid Search: Dense + Sparse (BM25)                 │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Hybrid Search**: Combines dense semantic vectors with sparse BM25 for optimal retrieval
- **Multi-Format Support**: Parse PDF, TXT, DOC, DOCX, XLS, XLSX documents
- **Async Processing**: Fully async architecture for high performance
- **Batch Processing**: Efficient document processing with configurable batch sizes
- **Conversation Context**: Maintains conversation history via Working Memory
- **Importance Ranking**: Scores documents by relevance, recency, and importance
- **Brain-Inspired Modules**: Modular architecture inspired by cognitive neuroscience

## Technology Stack

- **Framework**: FastAPI (async)
- **Vector Database**: Qdrant (Docker)
- **LLM**: Ollama with llama3.2:3b
- **Embeddings**:
  - Dense: sentence-transformers/all-MiniLM-L6-v2
  - Sparse: Qdrant/bm25 via FastEmbed
- **Document Parsing**: PyPDF2, PyMuPDF, python-docx, openpyxl
- **Search Fusion**: RRF (Reciprocal Rank Fusion) or DBSF (Distribution-Based Score Fusion)

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Docker** (for Qdrant)
3. **Ollama** with llama3.2:3b model

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Brain_rag
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Qdrant (Docker)**
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

5. **Verify Ollama is running**
```bash
ollama list
# Should show llama3.2:3b
```

6. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

7. **Create logs directory**
```bash
mkdir -p logs
```

## Usage

### Start the API Server

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the main module
python -m app.main
```

The API will be available at `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Create Collection
```bash
curl -X POST http://localhost:8000/build_collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "description": "My document collection"
  }'
```

#### 3. Upload Documents
```bash
curl -X POST http://localhost:8000/upload_files \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "file_paths": [
      "/path/to/document1.pdf",
      "/path/to/document2.txt"
    ],
    "batch_size": 32
  }'
```

#### 4. Query with RAG
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents",
    "query": "What is the main topic of the documents?",
    "top_k": 10,
    "fusion_method": "rrf",
    "use_llm": true
  }'
```

#### 5. List Collections
```bash
curl http://localhost:8000/collections
```

#### 6. Delete Collection
```bash
curl -X DELETE http://localhost:8000/delete_collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_documents"
  }'
```

## Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant host | localhost |
| `QDRANT_PORT` | Qdrant HTTP port | 6333 |
| `OLLAMA_HOST` | Ollama API host | http://localhost:11434 |
| `OLLAMA_MODEL` | LLM model name | llama3.2:3b |
| `CHUNK_SIZE` | Document chunk size | 512 |
| `CHUNK_OVERLAP` | Chunk overlap size | 128 |
| `BATCH_SIZE` | Processing batch size | 32 |
| `HYBRID_FUSION_METHOD` | Fusion method (rrf/dbsf) | rrf |
| `TOP_K_RESULTS` | Default top-k results | 10 |

### Brain Module Weights

Configure how the Amygdala ranks document importance:

- `RECENCY_WEIGHT` (0.3): Weight for document recency
- `RELEVANCE_WEIGHT` (0.5): Weight for search relevance
- `IMPORTANCE_WEIGHT` (0.2): Weight for base importance score

## Brain Modules Explained

### Hippocampus
**Function**: Long-term memory storage and retrieval

- Indexes documents into Qdrant with hybrid embeddings
- Performs vector search (dense + sparse)
- Manages memory spaces (collections)

### Amygdala
**Function**: Emotional importance and salience

- Scores retrieved documents by importance
- Applies recency bias (recent = more salient)
- Ranks results for optimal context

### Prefrontal Cortex
**Function**: Executive control and reasoning

- Orchestrates the full RAG pipeline
- Integrates information from all modules
- Generates final response using LLM

### Working Memory
**Function**: Temporary context buffer

- Maintains conversation history
- Provides context window for queries
- Manages short-term state

## Performance Optimizations

1. **Async Operations**: All I/O operations are async
2. **Batch Processing**: Documents processed in configurable batches
3. **Thread Pools**: CPU-intensive parsing uses thread pools
4. **Hybrid Search**: Optimal retrieval with dense + sparse vectors
5. **Connection Pooling**: Reuses Qdrant gRPC connections
6. **Lazy Loading**: Models loaded on first use

## Supported Document Formats

- **PDF**: Using PyMuPDF (primary) and PyPDF2 (fallback)
- **TXT**: Auto-encoding detection with chardet
- **DOC/DOCX**: Full text and table extraction
- **XLS/XLSX**: All sheets and cells

## Development

### Project Structure

```
Brain_rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── brain/                  # Brain modules
│   │   ├── __init__.py
│   │   ├── hippocampus.py     # Memory & retrieval
│   │   ├── amygdala.py        # Importance scoring
│   │   ├── prefrontal_cortex.py  # Executive control
│   │   └── working_memory.py  # Context buffer
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_parser.py # Multi-format parsing
│   │   ├── embeddings.py      # Dense + sparse embeddings
│   │   ├── qdrant_service.py  # Qdrant client
│   │   └── llm_service.py     # Ollama client
│   └── utils/
│       ├── __init__.py
│       └── async_helpers.py   # Async utilities
├── logs/                       # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with pytest (install pytest first if needed)
pip install pytest pytest-asyncio
pytest tests/
```

## Troubleshooting

### Qdrant Connection Issues
- Verify Qdrant is running: `curl http://localhost:6333`
- Check Docker container: `docker ps | grep qdrant`

### Ollama Issues
- Verify Ollama is running: `ollama list`
- Check model is available: `ollama run llama3.2:3b`

### Memory Issues
- Reduce `BATCH_SIZE` in .env
- Reduce `CHUNK_SIZE` for smaller documents
- Limit `MAX_WORKERS` for thread pool

### Slow Performance
- Increase `BATCH_SIZE` if you have enough RAM
- Use SSD storage for Qdrant
- Enable gRPC for Qdrant connections (already enabled)

## License

MIT License

## Credits

Inspired by cognitive neuroscience research on:
- Hippocampal memory consolidation
- Amygdala emotional processing
- Prefrontal cortex executive function
- Working memory systems

Built with ❤️ using FastAPI, Qdrant, and Ollama.
