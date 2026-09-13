"""
Retrieval evaluation.

Indexes one PDF, asks a fixed set of questions (eval_questions.json) and checks
whether the top-5 retrieved chunks contain the answer. Four retrieval setups are
compared so the numbers in the README can be reproduced with:

    python evaluate.py

A retrieved chunk counts as a hit when it comes from one of the answer's pages
AND contains one of the answer keywords.

Metrics:
  Hit@5  - share of questions with at least one relevant chunk in the top 5
  MRR@5  - mean of 1/rank of the first relevant chunk (0 if none in top 5)
"""

import json

import numpy as np
from rank_bm25 import BM25Okapi

from app.bm25_db import tokenize
from app.chunker import create_chunks
from app.embedding import creating_embedding
from app.pdf_loader import load_pdf
from app.reranker import rerank

PDF = "data/nist_ai_rmf_1.0.pdf"
QUESTIONS = "eval_questions.json"
TOP_K = 5
CANDIDATES = 20  # same candidate pool size as app/hybrid.py


def naive_tokenize(text):
    # The tokenizer the project used before the fix: punctuation stays attached to words
    return text.lower().split()


def build_chunks(pdf_path):
    chunks = []
    for page in load_pdf(pdf_path):
        for text in create_chunks(page["text"]):
            chunks.append({"text": text, "page": page["page"], "filename": pdf_path})
    return chunks


def is_relevant(chunk, question):
    on_answer_page = chunk["page"] in question["pages"]
    has_answer = any(kw.lower() in chunk["text"].lower() for kw in question["keywords"])
    return on_answer_page and has_answer


# ---- retrieval setups -------------------------------------------------------

def bm25_search(bm25, chunks, tokenizer, question, k):
    scores = bm25.get_scores(tokenizer(question))
    order = np.argsort(scores)[::-1]
    return [chunks[i] for i in order[:k] if scores[i] > 0]


def vector_search(vectors, chunks, question, k):
    query = np.array(creating_embedding(question))
    scores = vectors @ query  # embeddings are normalized, so dot product == cosine similarity
    order = np.argsort(scores)[::-1]
    return [chunks[i] for i in order[:k]]


def hybrid_search(bm25, vectors, chunks, question, k):
    # Mirrors app/hybrid.py: wide pool from both sources, dedupe, cross-encoder rerank
    merged, seen = [], set()
    candidates = vector_search(vectors, chunks, question, CANDIDATES) + bm25_search(
        bm25, chunks, tokenize, question, CANDIDATES
    )
    for chunk in candidates:
        if chunk["text"] not in seen:
            merged.append(chunk)
            seen.add(chunk["text"])
    return rerank(question, merged, top_k=k)


# ---- scoring ----------------------------------------------------------------

def score(questions, search):
    hits, mrr = 0, 0.0
    for question in questions:
        for rank, chunk in enumerate(search(question["question"]), start=1):
            if is_relevant(chunk, question):
                hits += 1
                mrr += 1 / rank
                break
    return hits / len(questions), mrr / len(questions)


def main():
    questions = json.load(open(QUESTIONS, encoding="utf-8"))
    chunks = build_chunks(PDF)
    print(f"{len(chunks)} chunks from {PDF}, {len(questions)} questions")

    # Sanity check the question set: every question must have at least one relevant chunk
    for q in questions:
        if not any(is_relevant(c, q) for c in chunks):
            print(f"WARNING: no relevant chunk for: {q['question']}")

    print("Embedding chunks...")
    vectors = np.array([creating_embedding(c["text"]) for c in chunks])
    bm25_old = BM25Okapi([naive_tokenize(c["text"]) for c in chunks])
    bm25_new = BM25Okapi([tokenize(c["text"]) for c in chunks])

    setups = {
        "BM25, old .split() tokenizer": lambda q: bm25_search(bm25_old, chunks, naive_tokenize, q, TOP_K),
        "BM25, fixed tokenizer": lambda q: bm25_search(bm25_new, chunks, tokenize, q, TOP_K),
        "Vector only": lambda q: vector_search(vectors, chunks, q, TOP_K),
        "Hybrid + rerank (what the app uses)": lambda q: hybrid_search(bm25_new, vectors, chunks, q, TOP_K),
    }

    keyword_qs = [q for q in questions if q["type"] == "keyword"]
    semantic_qs = [q for q in questions if q["type"] == "semantic"]

    print()
    print(f"| Retrieval | Hit@{TOP_K} all | Hit@{TOP_K} keyword | Hit@{TOP_K} semantic | MRR@{TOP_K} |")
    print("|---|---|---|---|---|")
    for name, search in setups.items():
        hit_all, mrr = score(questions, search)
        hit_kw, _ = score(keyword_qs, search)
        hit_sem, _ = score(semantic_qs, search)
        print(f"| {name} | {hit_all:.0%} | {hit_kw:.0%} | {hit_sem:.0%} | {mrr:.2f} |")


if __name__ == "__main__":
    main()
