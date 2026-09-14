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
- **Hybrid retrieval** combining semantic and keyword results (20 candidates from each)
- **Cross-encoder reranking** to improve final context selection
- **Groq LLM integration** for grounded answer generation
- **Page-cited answers** – every chunk keeps its filename and page number through retrieval, and the model is asked to cite the pages it used
- **Conversation history** for follow-up questions
- **Streaming response endpoint**
- **Document metadata and UUID-based identification**
- **Document listing and deletion endpoints**
- **Pydantic request/response validation**
- **Logging and API error handling**
- **Health-check endpoint**
- **Retrieval evaluation script** with reproducible numbers (see [Retrieval Evaluation](#retrieval-evaluation))
- **Streamlit web interface**
- **Dockerized** – API and UI run from one image with a single command

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

Tokenization splits on anything that is not a letter or digit (`re.findall(r"\w+", ...)`). This matters more than it looks: with a plain `.split()`, the question *"What is quercetin?"* produced the token `quercetin?`, which never matched `quercetin` in the document, so keyword retrieval silently failed for almost every real question (see the evaluation below).

Each source returns its top 20 candidates. The results are merged, duplicates are removed, and each chunk keeps its `filename` and `page` metadata.

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

Each context passage is labelled with its source, e.g. `[thesis.pdf, page 47]`. The prompt instructs the model to answer using only the retrieved context, to mention the page numbers it used, and to say so when the answer cannot be found.

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
│   ├── bm25_db.py          # BM25 index + tokenizer
│   ├── chunker.py
│   ├── config.py
│   ├── embedding.py
│   ├── groq_client.py
│   ├── hybrid.py           # vector + BM25 -> rerank
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
├── data/
│   └── nist_ai_rmf_1.0.pdf # public document used by evaluate.py
│
├── main.py
├── evaluate.py             # retrieval evaluation
├── eval_questions.json     # 30 questions with answer pages/keywords
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
  "answer": "Quercetin is a flavonoid that acts as both reducing and stabilizing agent ... (page 23, page 30)"
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

Build the image:

```bash
docker build -t rag-app .
```

Run it while injecting environment variables at runtime:

```bash
docker run --env-file .env -p 7860:7860 rag-app
```

The container starts the FastAPI backend in the background and serves the Streamlit UI on port `7860` (see `start.sh`), then open http://localhost:7860. The image installs the CPU-only PyTorch wheel and downloads the embedding and reranking models at build time, so startup is fast and no secrets are baked in. The same image can be deployed to any container host (Cloud Run, a VPS, Hugging Face Spaces) without changes; `start.sh` honours the `PORT` variable those platforms set.

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

## Retrieval Evaluation

Claims like "hybrid is better than vector-only" are easy to make and rarely measured, so the repository includes a small evaluation:

```bash
python evaluate.py
```

The script indexes the [NIST AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) (48 pages, public domain, included in `data/`) and asks 30 questions from `eval_questions.json`. Half are exact-term lookups (`"What is MEASURE 2.7 about?"`) where keyword search should do well, half are paraphrased (`"When should development of an AI system stop?"`) where semantic search should do well. Each question lists the page(s) that contain the answer and a phrase from it; a retrieved chunk counts as a hit only if it is from one of those pages **and** contains the phrase.

Results (top 5 chunks, 286 chunks in the index):

| Retrieval | Hit@5 all | Hit@5 keyword | Hit@5 semantic | MRR@5 |
|---|---|---|---|---|
| BM25, old `.split()` tokenizer | 67% | 67% | 67% | 0.49 |
| BM25, fixed tokenizer | 83% | 87% | 80% | 0.73 |
| Vector only | 87% | 87% | 87% | 0.63 |
| **Hybrid + rerank (what the app uses)** | **100%** | **100%** | **100%** | **0.90** |

*Hit@5*: share of questions with a relevant chunk in the top 5. *MRR@5*: mean of 1/rank of the first relevant chunk, so 0.90 means the answer is usually the first chunk.

Two things this showed me:

- **The tokenizer bug was real and large.** Replacing `.split()` with a regex moved BM25 from 67% to 83% on the same questions – one line of code.
- **Vector and keyword search fail on different questions.** Neither alone reaches 90%, but fusing both and letting the cross-encoder pick gets every question, and puts the right chunk first far more often (MRR 0.63 → 0.90).

The question set is small and single-document, so these numbers are indicative rather than a benchmark. Generation quality (is the *answer* correct, not just the retrieved chunk) is not measured yet.

---

## Current Limitations

This repository is designed as a portfolio-scale RAG application rather than a fully distributed production system.

Current limitations include:

- Qdrant is currently used in local filesystem mode, which allows only one process to open the storage folder at a time (a second server instance fails with a lock error).
- BM25 state is maintained in application memory.
- The lightweight document registry is also maintained in memory.
- Container-local data is not intended as durable production storage; uploads and the index inside a container are lost when it is recreated.
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
- Generation evaluation (answer correctness and faithfulness, not only retrieval)
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
- measuring retrieval quality instead of assuming it (Hit@k, MRR)
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