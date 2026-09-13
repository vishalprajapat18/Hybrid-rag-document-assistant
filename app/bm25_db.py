#from rank_bm25 import BM25Okapi

##from app.pdf_loader import load_pdf
##from app.preprocessing import clean_text
#from app.chunker import create_chunks

# ---------- Load thesis once ----------
#pages = load_pdf("data/biology.pdf")
#chunks = create_chunks(pages)

#texts = [c["text"] for c in chunks]

# ---------- Build BM25 index ----------
#tokenized = [text.lower().split() for text in texts]
#bm25 = BM25Okapi(tokenized)

# ---------- Search function ----------
#def search(query, k=3):

 #   tokens = query.lower().split()

  #  docs = bm25.get_top_n(
   #     tokens,
    #    texts,
     #   n=k
    #)

    #return docs


from rank_bm25 import BM25Okapi

#from app.pdf_loader import load_pdf
#from app.chunker import create_chunks

#---------- Load PDF (pages are already cleaned by load_pdf)
#pages = load_pdf("data/biology.pdf")  # now its job is like give 
#------------me chnuks i will index them we will not manully add `file name
# -----------we will connect this to ingestion upto chunks


# Convert list of pages -> one string
#text = " ".join(page["text"] for page in pages)

# Chunk the string
#chunks = create_chunks(text)

#texts = chunks

#tokenized = [t.lower().split() for t in texts] #list comprehension
#bm25 = BM25Okapi(tokenized)

#def search(query, k=3):
#    tokens = query.lower().split()
#    return bm25.get_top_n(tokens, texts, n=k)


from rank_bm25 import BM25Okapi


all_chunks = []

bm25 = None


def add_chunks(
    chunks: list[str],
    document_id: str,
    filename: str,
    page: int
):

    global bm25

    for chunk in chunks:

        all_chunks.append(
            {
                "text": chunk,
                "document_id": document_id,
                "filename": filename,
                "page": page
            }
        )

    tokenized_chunks = [
        item["text"].lower().split()
        for item in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)


def search(query: str, k: int = 3):

    if bm25 is None:
        return []

    query_tokens = query.lower().split()

    texts = [
        item["text"]
        for item in all_chunks
    ]

    top_texts = bm25.get_top_n(
        query_tokens,
        texts,
        n=k
    )

    return top_texts



def remove_document(document_id: str):

    global all_chunks
    global bm25

    all_chunks = [
        item
        for item in all_chunks
        if item["document_id"] != document_id
    ]

    if not all_chunks:
        bm25 = None
        return

    tokenized_chunks = [
        item["text"].lower().split()
        for item in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)