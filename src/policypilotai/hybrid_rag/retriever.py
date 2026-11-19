
import re

from policypilotai.hybrid_rag.vector_store import search_vector_store
from policypilotai.hybrid_rag.keyword_store import search_bm25
from policypilotai.hybrid_rag.reranker import rerank


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def hybrid_retrieve(query: str, top_k=5, alpha=0.8):

    if not query or not query.strip():
        return []

    candidate_k = 10

    # Step 1: Vector retrieval
    vector_results = search_vector_store(
        query, top_k=candidate_k
    )

    # Step 2: BM25 retrieval
    bm25_results = search_bm25(
        query, top_k=candidate_k
    )

    # Step 3: Combine results using RRF
    hybrid_scores = {}
    original_texts = {}

    k = 60

    for rank, doc in enumerate(vector_results, start=1):

        text = doc.page_content
        key = normalize_text(text)

        original_texts[key] = text

        hybrid_scores[key] = hybrid_scores.get(key, 0) + (
            alpha / (k + rank)
        )

    for rank, (score, text) in enumerate(bm25_results, start=1):

        key = normalize_text(text)

        original_texts.setdefault(key, text)

        hybrid_scores[key] = hybrid_scores.get(key, 0) + (
            (1 - alpha) / (k + rank)
        )

    candidates = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    candidate_texts = [
        original_texts[key]
        for key, score in candidates
    ]

    # Step 4: Rerank using your existing cross-encoder
    final_results = rerank(
        query=query,
        chunks=candidate_texts,
        top_k=top_k
    )

    return final_results
