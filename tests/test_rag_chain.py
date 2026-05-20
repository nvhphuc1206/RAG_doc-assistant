"""Tests for RAG chain module."""

import pytest
from unittest.mock import MagicMock, patch
from src.generation.rag_chain import format_docs, ask
from langchain_core.documents import Document


def test_format_docs():
    """Test that format_docs returns a well-structured context string."""
    docs = [
        Document(
            page_content="LangChain helps build LLM apps.",
            metadata={"source": "guide.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="ChromaDB is a vector database.",
            metadata={"source": "db.txt", "page": "N/A", "chunk_index": 1},
        ),
    ]
    result = format_docs(docs)

    assert "Source 1" in result
    assert "guide.pdf" in result
    assert "LangChain" in result
    assert "ChromaDB" in result


def test_format_docs_missing_metadata():
    """format_docs should handle missing metadata gracefully."""
    docs = [Document(page_content="Some text.", metadata={})]
    result = format_docs(docs)

    assert "Unknown" in result
    assert "Some text." in result


def test_get_llm_unknown_provider():
    """get_llm should raise ValueError for unknown providers."""
    from src.generation.rag_chain import get_llm

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm(provider="unknown_provider")


def test_get_llm_error_message_lists_providers():
    """Error message should mention all supported providers."""
    from src.generation.rag_chain import get_llm

    with pytest.raises(ValueError) as exc_info:
        get_llm(provider="invalid")

    msg = str(exc_info.value)
    for provider in ("anthropic", "openai", "gemini", "groq", "ollama"):
        assert provider in msg
