import re
from rank_bm25 import BM25Okapi


all_chunks = []

bm25 = None


def tokenize(text: str) -> list[str]:
    # Split on anything that is not a letter/digit, so "quercetin?" becomes "quercetin".
    # Plain .split() kept the punctuation attached and the question never matched the document.
    return re.findall(r"\w+", text.lower())


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
        tokenize(item["text"])
        for item in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)


def search(query: str, k: int = 3):

    if bm25 is None:
        return []

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    # Highest score first; skip chunks that share no words with the query
    ranked = sorted(
        range(len(all_chunks)),
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        all_chunks[i]
        for i in ranked[:k]
        if scores[i] > 0
    ]


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
        tokenize(item["text"])
        for item in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
