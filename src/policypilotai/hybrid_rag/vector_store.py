
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def get_vector_store(chunks: List[Document] = None, collection_name="hr_policies"):
    """
    Creates or loads ChromaDB with HR policy chunks.
    Uses HuggingFace embeddings.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./data/chroma_db"
    )

    if chunks:
        vector_store.add_documents(chunks)

    return vector_store


def search_vector_store(query: str, top_k=5):
    vector_store = get_vector_store([])
    return vector_store.similarity_search(query, k=top_k)
