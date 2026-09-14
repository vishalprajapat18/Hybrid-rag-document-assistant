from app.pdf_loader import load_pdf
from app.chunker import create_chunks
from app.embedding import creating_embeddings
from app.qdrant_db import insert_chunks
from app.bm25_db import add_chunks


def ingest_document(
    pdf_path: str,
    document_id: str,
    filename: str
):

    pages = load_pdf(pdf_path)   # pages are already cleaned by load_pdf

    for page in pages:

        # 1. Split the page into chunks
        chunks = create_chunks(page["text"])

        if not chunks:
            continue

        # 2. Send SAME chunks to BM25
        add_chunks(
            chunks=chunks,
            document_id=document_id,
            filename=filename,
            page=page["page"]
        )

        # 3. Embed all chunks of the page in ONE batched call
        #    (one call per chunk made a 48-page PDF take ~2 minutes)
        embeddings = creating_embeddings(chunks)

        # 4. Store them in Qdrant
        insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            page=page["page"],
            document_id=document_id,
            filename=filename
        )
