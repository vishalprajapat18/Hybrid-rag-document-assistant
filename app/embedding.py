from sentence_transformers import SentenceTransformer
# Load once when the application starts
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def creating_embedding(text: str) -> list[float]:
    """
    Convert one text (a question) into a vector.
    """
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def creating_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert many chunks into vectors in one batched call.
    Much faster than calling creating_embedding() in a loop: the model
    processes 32 chunks at a time instead of paying the per-call overhead 300 times.
    """
    if not texts:
        return []
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
    return vectors.tolist()
