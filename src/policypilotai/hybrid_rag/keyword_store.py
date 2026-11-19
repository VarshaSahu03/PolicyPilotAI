
import os
import re
import pickle
from typing import List
from rank_bm25 import BM25Okapi


INDEX_FILE = "./data/bm25_index.pkl"

# Common words that do not help identify relevant documents
STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "to", "for",
    "in", "on", "and", "or", "how", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should",
    "we", "our", "you", "your", "it", "this", "that"
}


def tokenize(text: str) -> List[str]:
    # Extract words and ignore punctuation and whitespace
    words = re.findall(r"[a-zA-Z]+", text.lower())

    # Remove common words
    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def build_bm25(documents):

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    index_data = {
        "bm25": bm25,
        "documents": documents
    }

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index saved with {len(documents)} documents")

    return index_data


def get_bm25_index(chunks=None):

    if chunks is None:
        try:
            with open(INDEX_FILE, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    if not chunks:
        return None

    documents = [
        chunk.page_content if hasattr(chunk, "page_content") else chunk
        for chunk in chunks
    ]

    return build_bm25(documents)


def rebuild_bm25_index():

    # Read existing documents from the old BM25 index
    with open(INDEX_FILE, "rb") as f:
        old_data = pickle.load(f)

    documents = old_data["documents"]

    # Build a fresh index using improved tokenization
    return build_bm25(documents)


def search_bm25(query: str, top_k=5):

    index_data = get_bm25_index()

    if index_data is None:
        return []

    bm25 = index_data["bm25"]
    documents = index_data["documents"]

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        (scores[i], documents[i])
        for i in top_indices
        if scores[i] > 0
    ]
