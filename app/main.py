"""Streamlit UI for RAG Document Assistant."""

import streamlit as st
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.store import create_vector_store
from src.generation.rag_chain import ask

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

st.title("RAG Document Assistant")
st.caption("Upload tài liệu và hỏi đáp thông minh dựa trên nội dung")

with st.sidebar:
    st.header("Tài liệu")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Xử lý tài liệu", type="primary"):
        all_chunks = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix
            ) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            try:
                docs = load_document(tmp_path)
                chunks = chunk_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"Lỗi khi xử lý {file.name}: {e}")
                continue

            progress.progress((i + 1) / len(uploaded_files))

        if all_chunks:
            st.session_state["vector_store"] = create_vector_store(all_chunks)
            st.session_state["chunk_count"] = len(all_chunks)
            st.success(f"{len(all_chunks)} chunks từ {len(uploaded_files)} files!")

    if "chunk_count" in st.session_state:
        st.info(f"{st.session_state['chunk_count']} chunks trong DB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Nguồn trích dẫn"):
                for s in msg["sources"]:
                    st.caption(f"{s['file']} | Trang {s['page']}")
                    st.text(s["content_preview"])

if prompt := st.chat_input("Hỏi gì đó về tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "vector_store" not in st.session_state:
            st.warning("Vui lòng upload tài liệu trước!")
        else:
            with st.spinner("Đang tìm kiếm..."):
                result = ask(st.session_state["vector_store"], prompt)

            st.markdown(result["answer"])
            with st.expander("Nguồn trích dẫn"):
                for s in result["sources"]:
                    st.caption(f"{s['file']} | Trang {s['page']}")
                    st.text(s["content_preview"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
