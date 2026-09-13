from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(question: str, chunks: list[dict], top_k=5):
    # chunks are dicts with text/page/filename/document_id (see bm25_db and qdrant_db payloads)
    pairs = [[question, chunk["text"]] for chunk in chunks]

    # predicting relevance scores
    scores = model.predict(pairs)

    scored = list(zip(scores, chunks))

    # highest scored first
    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored[:top_k]]
