"""Vector store module — embeds documents and persists vectors to ChromaDB."""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "documents")


def _is_persist_enabled() -> bool:
    """Read CHROMA_PERSIST env var. Default 'true'."""
    return os.getenv("CHROMA_PERSIST", "true").lower() in ("true", "1", "yes")


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Initialize embedding model (runs locally, no API key needed).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def create_vector_store(
    chunks: list[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    persist: bool | None = None,
) -> Chroma:
    """Create a ChromaDB vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        persist_directory: Path to persist the database (ignored if persist=False).
        collection_name: Name of the ChromaDB collection.
        persist: If True, save to disk. If False, in-memory only.
                 If None, read from CHROMA_PERSIST env var (default true).

    Returns:
        Chroma vector store instance ready for queries.
    """
    embeddings = get_embedding_model()
    if persist is None:
        persist = _is_persist_enabled()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory if persist else None,
        collection_name=collection_name,
    )


def load_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Load an existing vector store from disk.

    Args:
        persist_directory: Path where the database is persisted.
        collection_name: Name of the collection to load.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If persist_directory does not exist.
    """
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            "Process documents with persist=True first."
        )

    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
