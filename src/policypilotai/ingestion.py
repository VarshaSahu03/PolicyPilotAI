
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from policypilotai.hybrid_rag.chunking import chunk_text
from policypilotai.hybrid_rag.vector_store import get_vector_store
from policypilotai.hybrid_rag.keyword_store import get_bm25_index


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POLICY_FOLDER = PROJECT_ROOT / "data" / "policies"


# ==========================================
# PDF INGESTION
# ==========================================

def ingestion_of_pdfs(folder_path=None, batch_size=100):

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if folder_path is None:
        folder_path = DEFAULT_POLICY_FOLDER

    folder_path = Path(folder_path).resolve()

    print("\n========== PDF INGESTION START ==========")
    print(f"Policy folder: {folder_path}")

    # --------------------------------------
    # CHECK FOLDER
    # --------------------------------------

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Policy folder does not exist: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {folder_path}"
        )

    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {folder_path}"
        )

    print(f"PDF files found: {len(pdf_files)}")

    all_chunks = []

    # --------------------------------------
    # LOAD AND CHUNK PDF DOCUMENTS
    # --------------------------------------

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:

            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                print(f"WARNING: No pages extracted from {pdf_path.name}")
                continue

            print(f"Pages loaded: {len(documents)}")

            for doc in documents:

                page_text = doc.page_content.strip()

                if not page_text:
                    continue

                chunks = chunk_text(page_text)

                for chunk in chunks:

                    if not isinstance(chunk, str):
                        continue

                    chunk = chunk.strip()

                    if not chunk:
                        continue

                    all_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                "source": pdf_path.name
                            }
                        )
                    )

        except Exception as e:

            print(
                f"ERROR processing {pdf_path.name}: {e}"
            )

    # --------------------------------------
    # VALIDATE CHUNKS
    # --------------------------------------

    if not all_chunks:

        raise RuntimeError(
            "No chunks were created from the PDF files. "
            "Check PDF text extraction and chunk_text()."
        )

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # --------------------------------------
    # STORE IN CHROMA DB
    # --------------------------------------

    print("\nStoring chunks in ChromaDB...")

    for start in range(0, len(all_chunks), batch_size):

        batch = all_chunks[start:start + batch_size]

        get_vector_store(batch)

        print(
            f"Stored batch "
            f"{start // batch_size + 1}: {len(batch)} chunks"
        )

    # --------------------------------------
    # BUILD BM25 INDEX
    # --------------------------------------

    print("\nBuilding BM25 index...")

    get_bm25_index(all_chunks)

    print("\n========== INGESTION COMPLETED ==========")

    return all_chunks
