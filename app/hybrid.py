from app.embedding import creating_embedding
from app.qdrant_db import search_chunks
from app.bm25_db import search as bm25_search
from app.reranker import rerank
def hybrid_search(question):

    # Vector search
    query_vector = creating_embedding(question)
    vector_results = search_chunks(query_vector, limit=3)

    # BM25 search
    bm25_results = bm25_search(question, k=3)

    # Merge both
    merged = []
    seen = set()



    for r in vector_results:
        text = r.payload["text"]
        merged.append(text)
        seen.add(text)

        
    for text in bm25_results:
        if text not in seen:
            merged.append(text)
            seen.add(text)

    ranked = rerank(question, merged)
    return ranked