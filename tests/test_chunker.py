"""Tests for text chunking module."""

from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents


def test_chunk_short_document():
    """Short documents should produce at least 1 chunk."""
    docs = [Document(page_content="Short text.", metadata={"source": "test"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_long_document():
    """Long documents should produce multiple chunks."""
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_preserves_metadata():
    """Chunking should preserve original metadata."""
    docs = [Document(
        page_content="A" * 2000,
        metadata={"source": "report.pdf", "page": 1},
    )]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"
        assert chunk.metadata["page"] == 1


def test_total_chunks_metadata():
    """Each chunk should have total_chunks metadata set correctly."""
    docs = [Document(page_content="Word " * 500, metadata={"source": "test"})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    total = len(chunks)
    for chunk in chunks:
        assert chunk.metadata["total_chunks"] == total
