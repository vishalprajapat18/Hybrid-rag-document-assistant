from sentence_transformers import SentenceTransformer
# Load once when the application starts
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def creating_embedding(text:str)-> list[float]:
    """
    Convert one text chunk into a vector.
    """
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()