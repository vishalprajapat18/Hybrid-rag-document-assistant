
import sys
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI,UploadFile, File, HTTPException
from app.models import ChatRequest, ChatResponse
from app.prompt_builder import build_prompt
from app.llm import LLM
from fastapi.responses import StreamingResponse
from app.hybrid import hybrid_search
from uuid import uuid4
from app.ingestion import ingest_document
from app.qdrant_db import delete_document
from app.bm25_db import remove_document as remove_bm25_document
from app.qdrant_db import ensure_collection
import shutil
import os
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


app = FastAPI()
ensure_collection()
llm = LLM()
documents=[]

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/")
def home():
    return {"message": "RAG API Running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger.info(f"Question received: {request.question}")

    try:
        # 1. Retrieve relevant chunks (vector + BM25 -> rerank)
        results = hybrid_search(request.question)

        # 2. Build grounded prompt
        prompt = build_prompt(request.question, results, request.history)

        # 3. Generate answer
        answer, latency = llm.complete(prompt)
        logger.info(f"LLM latency: {latency:.2f}s")

        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"Chat request failed: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer"
        )

@app.post("/chat/stream")
def stream_chat(request:ChatRequest):


      results = hybrid_search(request.question)

      prompt = build_prompt(request.question, results,request.history)
      return StreamingResponse(
           llm.stream(prompt),
           media_type="text/plain"
      )

@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    logger.info(f"Uploading document: {file.filename}")

    os.makedirs("uploads", exist_ok=True)

    document_id = str(uuid4())

    file_path = f"uploads/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_document(
            pdf_path=file_path,
            document_id=document_id,
            filename=file.filename
        )

    except Exception as e:
        logger.exception(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process document"
        )

    documents.append(
    {
        "document_id": document_id,
        "filename": file.filename
    }
)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "uploaded and indexed"
    }

@app.get("/documents")
def list_documents():

    return {
        "documents": documents
    }



@app.delete("/documents/{document_id}")
def remove_document(document_id: str):

    global documents

    documents = [
        doc
        for doc in documents
        if doc["document_id"] != document_id
    ]

    delete_document(document_id)

    remove_bm25_document(document_id)

    return {
        "document_id": document_id,
        "status": "deleted"
    }