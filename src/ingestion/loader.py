"""Document loading module — hỗ trợ PDF, DOCX, TXT, Markdown."""

from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_document(file_path: str | Path) -> list[Document]:
    """Load a document and return a list of Document objects.

    Args:
        file_path: Path to the document file.

    Returns:
        List of Document objects with page_content and metadata.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in LOADER_MAP:
        supported = ", ".join(LOADER_MAP.keys())
        raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")

    loader = LOADER_MAP[ext](str(path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = ext

    return documents


def load_multiple_documents(file_paths: list[str | Path]) -> list[Document]:
    """Load multiple documents and merge into a single list.

    Args:
        file_paths: List of file paths.

    Returns:
        Combined list of Document objects from all files.
    """
    all_docs: list[Document] = []
    for fp in file_paths:
        docs = load_document(fp)
        all_docs.extend(docs)
    return all_docs
