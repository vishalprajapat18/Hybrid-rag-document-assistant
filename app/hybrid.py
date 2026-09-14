from app.embedding import creating_embedding
from app.qdrant_db import search_chunks
from app.bm25_db import search as bm25_search
from app.reranker import rerank


def hybrid_search(question, candidates=20, document_id=None):

    # Vector search - pull a wide candidate pool, the reranker picks the best few
    query_vector = creating_embedding(question)
    vector_results = search_chunks(query_vector, limit=candidates, document_id=document_id)

    # BM25 search
    bm25_results = bm25_search(question, k=candidates, document_id=document_id)

    # Merge both, keeping page/filename with each chunk so answers can cite them
    merged = []
    seen = set()

    for r in vector_results:
        chunk = r.payload
        if chunk["text"] not in seen:
            merged.append(chunk)
            seen.add(chunk["text"])

    for chunk in bm25_results:
        if chunk["text"] not in seen:
            merged.append(chunk)
            seen.add(chunk["text"])

    ranked = rerank(question, merged)
    return ranked
