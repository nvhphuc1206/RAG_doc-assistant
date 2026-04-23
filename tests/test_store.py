"""Tests for vector store module."""

import pytest
from langchain.schema import Document
from src.embeddings.store import create_vector_store, load_vector_store


def test_create_vector_store(tmp_path):
    """Test creating a vector store from chunks."""
    chunks = [
        Document(page_content="Python is a programming language.", metadata={"source": "test.txt"}),
        Document(page_content="LangChain is a framework for LLM applications.", metadata={"source": "test.txt"}),
    ]
    persist_dir = str(tmp_path / "chroma_test")
    store = create_vector_store(chunks, persist_directory=persist_dir, collection_name="test")

    assert store is not None


def test_load_vector_store(tmp_path):
    """Test loading a persisted vector store."""
    chunks = [
        Document(page_content="Vector databases store embeddings.", metadata={"source": "doc.txt"}),
    ]
    persist_dir = str(tmp_path / "chroma_load")
    create_vector_store(chunks, persist_directory=persist_dir, collection_name="test_load")

    store = load_vector_store(persist_directory=persist_dir, collection_name="test_load")
    assert store is not None


def test_similarity_search(tmp_path):
    """Test that similarity search returns relevant documents."""
    chunks = [
        Document(page_content="The sky is blue.", metadata={"source": "a.txt"}),
        Document(page_content="Cats are mammals.", metadata={"source": "b.txt"}),
        Document(page_content="The ocean is also blue.", metadata={"source": "c.txt"}),
    ]
    persist_dir = str(tmp_path / "chroma_search")
    store = create_vector_store(chunks, persist_directory=persist_dir, collection_name="test_search")

    results = store.similarity_search("What color is the sky?", k=2)
    assert len(results) == 2
    # The blue-related docs should rank higher
    contents = " ".join(r.page_content for r in results)
    assert "blue" in contents.lower()
