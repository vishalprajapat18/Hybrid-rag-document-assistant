# Document Intelligence RAG Assistant

A document question-answering system built with **FastAPI, Qdrant, BM25, Cross-Encoder Reranking, Sentence Transformers, Groq, and Streamlit**.

The application allows users to upload PDF documents and ask natural-language questions grounded in their contents. It combines semantic vector search with keyword-based BM25 retrieval and cross-encoder reranking to improve retrieval quality before sending context to the LLM.

---

## Why I Built This

Basic RAG systems often rely only on vector similarity search. While semantic retrieval works well for conceptually similar text, it can miss exact terminology, names, identifiers, or keyword-heavy queries.

This project implements a more complete retrieval pipeline:

**Vector Search + BM25 → Hybrid Retrieval → Cross-Encoder Reranking → LLM Generation**

The goal was to build the complete lifecycle of a RAG application rather than only calling an LLM API.

---

## Architecture

```text
                        ┌─────────────────────┐
                        │    Streamlit UI     │
                        └──────────┬──────────┘
                                   │
                          HTTP / REST API
                                   │
                        ┌──────────▼──────────┐
                        │      FastAPI        │
                        └──────────┬──────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
          Document Ingestion                    User Query
                 │                                   │
          PDF Text Extraction                        │
                 │                                   │
          Text Preprocessing                         │
                 │                                   │
              Chunking                               │
                 │                                   │
        ┌────────┴────────┐                          │
        │                 │                          │
   Embeddings           BM25                         │
        │                 │                          │
     Qdrant         Keyword Index                    │
        │                 │                          │
        └────────┬────────┘                          │
                 │                                   │
                 └────────── Hybrid Retrieval ◄──────┘
                                   │
                          Cross-Encoder Reranker
                                   │
                            Top Context Chunks
                                   │
                             Prompt Builder
                                   │
                               Groq LLM
                                   │
                                Answer
```

---

## Key Features

- **PDF ingestion** through a FastAPI upload endpoint
- **Text extraction and preprocessing**
- **Recursive text chunking** for document segmentation
- **Local sentence-transformer embeddings**
- **Qdrant vector search** for semantic retrieval
- **BM25 keyword search** for lexical retrieval
- **Hybrid retrieval** combining semantic and keyword results
- **Cross-encoder reranking** to improve final context selection
- **Groq LLM integration** for grounded answer generation
- **Conversation history** for follow-up questions
- **Streaming response endpoint**
- **Document metadata and UUID-based identification**
- **Document listing and deletion endpoints**
- **Pydantic request/response validation**
- **Logging and API error handling**
- **Health-check endpoint**
- **Basic retrieval evaluation**
- **Streamlit web interface**
- **Dockerized FastAPI backend**

---

## RAG Pipeline

### 1. Document Ingestion

When a PDF is uploaded:

```text
PDF
 ↓
Text Extraction
 ↓
Text Cleaning
 ↓
Chunking
 ↓
 ┌───────────────┬────────────────┐
 ↓               ↓
Embeddings       BM25 Index
 ↓
Qdrant
```

Each chunk is stored with metadata such as the document ID, filename, and page number.

### 2. Retrieval

For every user question, the system performs two retrieval strategies:

**Semantic retrieval**

```text
Question
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Qdrant Similarity Search
```

**Keyword retrieval**

```text
Question
 ↓
Tokenization
 ↓
BM25
 ↓
Keyword Matches
```

The results are merged and duplicates are removed.

### 3. Reranking

Hybrid retrieval provides candidate chunks.

A cross-encoder then evaluates the question and each candidate together:

```text
Question + Candidate Chunks
            ↓
      Cross-Encoder
            ↓
       Relevance Scores
            ↓
        Ranked Context
```

This provides a stronger relevance signal before context is passed to the LLM.

### 4. Generation

The highest-ranked chunks are inserted into a controlled prompt together with conversation history.

```text
Retrieved Context
       +
Conversation History
       +
User Question
       ↓
Prompt Builder
       ↓
Groq LLM
       ↓
Grounded Answer
```

The prompt instructs the model to answer using the retrieved document context and indicate when the answer cannot be found.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Validation | Pydantic |
| Frontend | Streamlit |
| PDF Processing | PyPDF |
| Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Vector Database | Qdrant |
| Keyword Retrieval | BM25 |
| Reranking | Cross-Encoder |
| LLM | Groq API |
| Containerization | Docker |

---

## Project Structure

```text
.
├── app/
│   ├── bm25_db.py
│   ├── chunker.py
│   ├── config.py
│   ├── embedding.py
│   ├── evaluation.py
│   ├── hybrid.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── models.py
│   ├── pdf_loader.py
│   ├── preprocessing.py
│   ├── prompt_builder.py
│   ├── qdrant_db.py
│   └── reranker.py
│
├── frontend/
│   └── streamlit_app.py
│
├── main.py
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
```

---

## API Endpoints

### Upload Document

```http
POST /documents
```

Uploads and indexes a PDF document.

### List Documents

```http
GET /documents
```

Returns documents registered during the current application session.

### Delete Document

```http
DELETE /documents/{document_id}
```

Removes a document from the application's indexes.

### Chat

```http
POST /chat
```

Example request:

```json
{
  "question": "What is the role of quercetin?",
  "history": []
}
```

Example response:

```json
{
  "answer": "..."
}
```

### Streaming Chat

```http
POST /chat/stream
```

Streams generated text to the client.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Then provide your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI interactive documentation:

```text
http://localhost:8000/docs
```

### 6. Start the Streamlit frontend

In another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The interface will normally be available at:

```text
http://localhost:8501
```

---

## Docker

Build the backend image:

```bash
docker build -t rag-api .
```

Run it while injecting environment variables at runtime:

```bash
docker run --env-file .env -p 8000:8000 rag-api
```

The FastAPI service is then exposed on port `8000`.

The Docker image packages the Python runtime, application code, and required dependencies while keeping secrets outside the image.

---

## Engineering Decisions

### Why Hybrid Retrieval?

Vector search captures semantic similarity, while BM25 performs well for exact terminology and keyword matches.

Combining both improves candidate retrieval across different query types.

### Why Reranking?

Initial retrieval is optimized for efficiently finding candidates.

The cross-encoder performs a more expensive but more precise comparison between the query and retrieved chunks, allowing the system to reorder candidates before generation.

### Why Separate FastAPI and Streamlit?

Streamlit is used only as the presentation layer.

The RAG pipeline remains behind FastAPI endpoints:

```text
Streamlit → HTTP → FastAPI → RAG Pipeline
```

This keeps UI concerns separate from retrieval and generation logic and allows other clients to consume the same API.

### Why Docker?

Containerization provides a reproducible runtime containing the required Python environment and application dependencies.

It also exposed environment-specific assumptions during development, such as relying on a Qdrant collection that already existed locally. The application now ensures the required collection exists during initialization.

---

## Current Limitations

This repository is designed as a portfolio-scale RAG application rather than a fully distributed production system.

Current limitations include:

- Qdrant is currently used in local filesystem mode.
- BM25 state is maintained in application memory.
- The lightweight document registry is also maintained in memory.
- Container-local data is not intended as durable production storage.
- Authentication and multi-user isolation are not implemented.
- Retrieval is not currently filtered to an explicitly selected document.

For a production deployment, these components could be replaced with persistent external services and user/document-level access controls.

---

## Future Improvements

Potential extensions include:

- Managed Qdrant or standalone Qdrant service
- Persistent document metadata database
- Object storage for uploaded documents
- Document-specific retrieval filters
- Automated retrieval and generation evaluation
- Authentication and multi-user document isolation
- Observability and tracing
- Cloud deployment and CI/CD

---

## What I Learned

Building this project provided hands-on experience with:

- designing an end-to-end RAG pipeline
- semantic and lexical information retrieval
- embedding generation and vector databases
- hybrid retrieval and reranking
- LLM prompt construction
- REST API design with FastAPI
- request validation with Pydantic
- frontend/backend separation
- application logging and error handling
- environment-variable based secret management
- Docker containerization
- debugging differences between local and containerized environments

---

## Author

**Vishal Prajapat**

B.Tech, Delhi Technological University

Interested in Generative AI, RAG systems, backend engineering, and applied AI.