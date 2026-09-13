from app.pdf_loader import load_pdf
from app.preprocessing import clean_text
from app.chunker import create_chunks
from app.embedding import creating_embedding
from app.qdrant_db import insert_chunks
from app.bm25_db import add_chunks




def ingest_document(
    pdf_path: str,
    document_id: str,
    filename: str
):

    pages = load_pdf(pdf_path)

    for page in pages:

        # 1. Get text from this PDF page
        text = page["text"]

        # 2. Clean it
        cleaned_text = clean_text(text)

        # 3. Split into chunks
        chunks = create_chunks(cleaned_text)

        # 4. Send SAME chunks to BM25
        add_chunks(
         chunks=chunks,
         document_id=document_id,
         filename=filename,
         page=page["page"]
        )

        # 5. Create embeddings for SAME chunks
        embeddings = [
            creating_embedding(chunk)
            for chunk in chunks
        ]

        # 6. Store them in Qdrant
        insert_chunks(
              chunks=chunks,
               embeddings=embeddings,
               page=page["page"],
               document_id=document_id,
              filename=filename
        )