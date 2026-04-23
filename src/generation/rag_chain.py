"""RAG chain module — kết hợp retrieval và generation."""

import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma


SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh. Trả lời câu hỏi DỰA TRÊN context
được cung cấp bên dưới. Tuân thủ các quy tắc sau:

1. Chỉ trả lời dựa trên thông tin trong context
2. Trích dẫn nguồn (tên file, trang) khi trả lời
3. Nếu không tìm thấy thông tin, nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
4. Trả lời bằng ngôn ngữ của câu hỏi
5. Trả lời ngắn gọn, đúng trọng tâm

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_llm(provider: str | None = None):
    """Initialize LLM based on provider setting.

    Args:
        provider: One of 'anthropic', 'openai', 'gemini', 'groq', 'ollama'.
                  Defaults to LLM_PROVIDER env var.

    Returns:
        LLM instance.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    model_override = os.getenv("LLM_MODEL")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_override or "claude-sonnet-4-6",
            temperature=0,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_override or "gpt-4o-mini",
            temperature=0,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_override or "gemini-2.0-flash",
            temperature=0,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_override or "llama-3.3-70b-versatile",
            temperature=0,
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model_override or "llama3")
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Supported: anthropic, openai, gemini, groq, ollama"
        )


def format_docs(docs) -> str:
    """Format retrieved documents into a context string with citations.

    Args:
        docs: List of retrieved Document objects.

    Returns:
        Formatted string with source annotations.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Nguồn {i} | File: {source} | Trang: {page} | Chunk: {chunk_idx}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def create_rag_chain(vector_store: Chroma, k: int = 4):
    """Create the full RAG chain: query → retrieve → generate.

    Args:
        vector_store: ChromaDB vector store instance.
        k: Number of top documents to retrieve.

    Returns:
        A runnable chain that accepts a question string.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
    return chain


def ask(vector_store: Chroma, question: str, k: int = 4) -> dict:
    """High-level function: ask a question and get answer + sources.

    Args:
        vector_store: ChromaDB vector store.
        question: User question in natural language.
        k: Number of documents to retrieve.

    Returns:
        Dict with 'answer' and 'sources' keys.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    sources = retriever.invoke(question)

    chain = create_rag_chain(vector_store, k=k)
    response = chain.invoke(question)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "...",
            }
            for doc in sources
        ],
    }
