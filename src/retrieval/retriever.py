"""Retrieval module — tìm kiếm vector similarity từ ChromaDB."""

from langchain_community.vectorstores import Chroma
from langchain.schema import Document


def get_retriever(vector_store: Chroma, k: int = 4, search_type: str = "similarity"):
    """Create a retriever from a vector store.

    Args:
        vector_store: ChromaDB vector store instance.
        k: Number of top documents to retrieve.
        search_type: 'similarity' or 'mmr' (Maximal Marginal Relevance).

    Returns:
        LangChain retriever object.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    k: int = 4,
) -> list[Document]:
    """Retrieve the top-k most relevant documents for a query.

    Args:
        vector_store: ChromaDB vector store instance.
        query: User query string.
        k: Number of documents to retrieve.

    Returns:
        List of relevant Document objects.
    """
    retriever = get_retriever(vector_store, k=k)
    return retriever.invoke(query)
