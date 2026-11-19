
from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_reranker():

    print("Loading reranker model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Reranker model loaded successfully.")

    return tokenizer, model


def rerank(query, chunks, top_k=5):

    if not chunks:
        return []

    # Load model only when reranking is first required
    tokenizer, model = load_reranker()

    pairs = [[query, chunk] for chunk in chunks]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.inference_mode():

        scores = model(**inputs).logits.squeeze(-1)

    ranked = sorted(
        zip(scores.tolist(), chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        text for score, text in ranked[:top_k]
    ]
