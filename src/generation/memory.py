"""Conversation memory — cho phép hỏi đáp nhiều lượt."""

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain


def create_conversational_chain(vector_store, k: int = 4, memory_window: int = 5):
    """Create a conversational RAG chain with memory.

    Args:
        vector_store: ChromaDB vector store.
        k: Number of documents to retrieve.
        memory_window: Number of recent exchanges to remember.

    Returns:
        ConversationalRetrievalChain with memory.
    """
    from src.generation.rag_chain import get_llm

    memory = ConversationBufferWindowMemory(
        k=memory_window,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        memory=memory,
        return_source_documents=True,
    )
    return chain
