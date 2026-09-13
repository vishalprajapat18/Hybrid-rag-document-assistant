from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(question:str,chunks:list[str],top_k=5):
    #creating question,chunk pairs
    pairs=[[question,chunk] for chunk in chunks]

    #predicting relevance scores
    scores= model.predict(pairs)

    scored = list(zip(scores,chunks))

    #highest scored first
    scored.sort(key=lambda x:x[0],reverse=True)

    #return only text
    return[chunk for _,chunk in scored[:top_k]]
