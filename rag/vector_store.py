
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"


# ==========================================================
# CREATE OR LOAD VECTOR STORE
# ==========================================================

def create_vector_store(pages, embeddings):
    """
    Create or load a persistent Chroma vector store.

    Supports multiple PDFs:
    - Creates the database if it does not exist.
    - Loads the existing database if available.
    - Adds pages from PDFs that are not indexed yet.
    - Avoids indexing the same PDF repeatedly.

    Note: If a PDF is modified after indexing, delete its old
    chunks and re-index it to include the changes.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Split the supplied pages into chunks.
    chunks = text_splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created. Check your PDF files."
        )

    print(f"\nReceived {len(pages)} PDF pages.")
    print(f"Created {len(chunks)} document chunks.")

    # ------------------------------------------------------
    # LOAD EXISTING DATABASE
    # ------------------------------------------------------

    if PERSIST_DIRECTORY.exists():
        print("\nExisting Chroma database found.")

        vector_store = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings,
        )

        # Find which PDF filenames are already indexed.
        stored_data = vector_store.get(
            include=["metadatas"]
        )

        existing_sources = {
            Path(metadata["source"]).name
            for metadata in (stored_data.get("metadatas") or [])
            if metadata and metadata.get("source")
        }

        # Select chunks belonging to PDFs not indexed yet.
        new_chunks = []

        for chunk in chunks:
            source = chunk.metadata.get("source")

            if not source:
                # Keep chunks without source metadata rather
                # than silently discarding their content.
                new_chunks.append(chunk)
                continue

            filename = Path(source).name

            if filename not in existing_sources:
                new_chunks.append(chunk)

        if new_chunks:
            print(
                f"Adding {len(new_chunks)} chunks "
                "from new PDFs..."
            )

            vector_store.add_documents(new_chunks)

            print("New PDF content indexed successfully.")

        else:
            print(
                "No new PDF files found. "
                "Using the existing database."
            )

        return vector_store

    # ------------------------------------------------------
    # CREATE NEW DATABASE
    # ------------------------------------------------------

    print("\nCreating a new Chroma database...")

    PERSIST_DIRECTORY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    print("Chroma database created successfully.")

    return vector_store