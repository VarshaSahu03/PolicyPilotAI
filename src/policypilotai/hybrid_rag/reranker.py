
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()


def rerank(query, chunks, top_k=5):

    if not chunks:
        return []

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
