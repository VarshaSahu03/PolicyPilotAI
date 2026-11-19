
import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# --------------------------------------
# DATABASE PATH
# --------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(
    os.getenv("POLICYPILOT_DATA_DIR", PROJECT_ROOT / "data")
)

CHROMA_DIR = DATA_DIR / "chroma_db"


# --------------------------------------
# EMBEDDING MODEL
# --------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------
# VECTOR STORE
# --------------------------------------

def get_vector_store(
    chunks: Optional[List[Document]] = None,
    collection_name="hr_policies"
):
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    if chunks:
        vector_store.add_documents(chunks)

    return vector_store


# --------------------------------------
# VECTOR SEARCH
# --------------------------------------

def search_vector_store(query: str, top_k=5):
    vector_store = get_vector_store([])
    return vector_store.similarity_search(query, k=top_k)
