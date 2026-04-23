"""Vector store module — embedding và lưu trữ vectors vào ChromaDB."""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "documents")


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
) -> Chroma:
    """Create a ChromaDB vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        persist_directory: Path to persist the database.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance ready for queries.
    """
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vector_store


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
    """
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
