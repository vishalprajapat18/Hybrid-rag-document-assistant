from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from uuid import uuid4
client = QdrantClient(path="./qdrant_data")

COLLECTION_NAME = "thesis"

def ensure_collection():

    collections = client.get_collections().collections

    collection_names = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

def insert_chunks(
    chunks,
    embeddings,
    page,
    document_id,
    filename
):

    points = []

    for chunk, vector in zip(chunks, embeddings):
        #We don't need idx anymore because UUID generates the point ID

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                 "filename": filename,
                 "page": page,
                 "text": chunk
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def search_chunks(query_vector, limit=3):

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )

    return response.points


#And every chunk has its own unique Qdrant ID:

#THESIS.pdf
# ├─ chunk → UUID #1
# ├─ chunk → UUID #2
# └─ chunk → UUID #3

#resume.pdf
# ├─ chunk → UUID #4
# ├─ chunk → UUID #5
# └─ chunk → UUID #6

def delete_document(document_id: str):

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                ) ] )
    )