"""Tests for document loader module."""

import pytest
from pathlib import Path
from src.ingestion.loader import load_document


def test_load_txt_file(tmp_path: Path):
    """Test loading a plain text file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, this is a test document.")

    docs = load_document(test_file)

    assert len(docs) >= 1
    assert "Hello" in docs[0].page_content
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["file_type"] == ".txt"


def test_unsupported_file_type(tmp_path: Path):
    """Test that unsupported file types raise ValueError."""
    test_file = tmp_path / "test.xyz"
    test_file.write_text("content")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(test_file)


def test_file_not_found():
    """Test that missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/file.pdf")


def test_load_md_file(tmp_path: Path):
    """Test loading a Markdown file."""
    test_file = tmp_path / "test.md"
    test_file.write_text("# Title\n\nSome markdown content.")

    docs = load_document(test_file)

    assert len(docs) >= 1
    assert docs[0].metadata["source"] == "test.md"
    assert docs[0].metadata["file_type"] == ".md"
