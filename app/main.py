"""Streamlit UI for RAG Document Assistant."""

import streamlit as st
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv(override=True)

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.store import create_vector_store, load_vector_store
from src.generation.rag_chain import ask

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS — neutral palette, cleaner controls ----
st.markdown(
    """
    <style>
      :root {
        --surface-1: #161b22;
        --surface-2: #1f242c;
        --border: #30363d;
        --text-muted: #9ca3af;
        --text: #e6edf3;
      }

      /* Page padding */
      .main .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 980px; }

      /* Title spacing */
      h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.1rem; font-size: 1.9rem; }
      .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
      }

      /* Buttons — neutral, no red */
      .stButton > button {
        background: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.45rem 1rem;
        font-weight: 500;
        transition: all 0.15s ease;
        box-shadow: none;
      }
      .stButton > button:hover:not(:disabled) {
        background: #2d333b;
        border-color: #484f58;
        color: #fff;
      }
      .stButton > button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      /* Override the red "primary" type — make it filled but neutral */
      .stButton > button[kind="primary"] {
        background: #30363d;
        border-color: #484f58;
      }
      .stButton > button[kind="primary"]:hover {
        background: #3a414a;
      }

      /* Alerts — softer, no harsh green/red */
      div[data-testid="stAlert"] {
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
      }
      div[data-testid="stAlert"] svg { color: var(--text-muted); }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-1);
        border: 1px dashed var(--border);
        border-radius: 8px;
      }

      /* Chat messages — cleaner bubble */
      [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.75rem 0;
        border: none;
      }
      [data-testid="stChatMessageAvatar"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border);
      }

      /* Chat input */
      [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--surface-1);
      }

      /* Expander (sources) */
      [data-testid="stExpander"] {
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-top: 0.5rem;
      }

      /* Sidebar */
      [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid var(--border);
      }
      [data-testid="stSidebar"] h2 {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
      }

      /* Progress bar */
      [data-testid="stProgressBar"] > div > div {
        background: var(--text-muted);
      }

      /* Checkbox */
      [data-testid="stCheckbox"] label p { color: var(--text-muted); font-size: 0.9rem; }

      /* Caption text in sources */
      .src-meta { color: var(--text-muted); font-size: 0.8rem; font-family: monospace; }

      /* Success alert — dark muted green instead of harsh bright green */
      div[data-testid="stAlertContentSuccess"],
      div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
      .stSuccess, [class*="stAlertSuccess"] {
        background: #0f2419 !important;
        border: 1px solid #2d5a3d !important;
        color: #7fc89c !important;
      }
      div[data-testid="stAlertContentSuccess"] p,
      div[data-testid="stAlertContentSuccess"] svg { color: #7fc89c !important; }

      /* Status widget header */
      [data-testid="stStatusWidget"], [data-testid="stStatus"] {
        background: var(--surface-1) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
      }
      /* Status when complete (state=complete) */
      [data-testid="stStatus"][data-state="complete"] {
        border-color: #2d5a3d !important;
      }

      /* Hero feature cards — compact fit-in-one-screen */
      .hero-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin: 0.6rem 0 0.4rem 0;
      }
      .hero-card {
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
      }
      .hero-card .step {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        text-transform: uppercase;
      }
      .hero-card h4 {
        font-size: 0.88rem;
        margin: 0.15rem 0 0.2rem 0;
        color: var(--text);
        font-weight: 600;
      }
      .hero-card p {
        font-size: 0.76rem;
        color: var(--text-muted);
        line-height: 1.35;
        margin: 0;
      }

      .hero-section h3 {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.85rem 0 0.35rem 0;
      }

      .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-bottom: 0.5rem;
      }
      .example-chip {
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.76rem;
        color: var(--text-muted);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = "❔"
BOT_AVATAR = "🔍"

# ---- Header ----
st.title("RAG Document Assistant")
st.caption("Ask your documents with cited sources — Retrieval-Augmented Generation")

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    persist_db = st.checkbox(
        "Persist DB to disk (.chroma/)",
        value=False,
        help="Off = keep in RAM only, lost on app close. "
             "On = persist to disk so you can reload later.",
    )

    process_btn = st.button(
        "Process documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if uploaded_files and process_btn:
        all_chunks = []
        with st.status("Processing...", expanded=True) as status:
            for i, file in enumerate(uploaded_files):
                st.write(f"`[{i+1}/{len(uploaded_files)}]` Reading {file.name}")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file.name).suffix
                ) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                try:
                    docs = load_document(tmp_path)
                    chunks = chunk_documents(docs)
                    all_chunks.extend(chunks)
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ {len(chunks)} chunks")
                except Exception as e:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✗ Error: {e}")
                    continue

            if all_chunks:
                st.write(f"Embedding {len(all_chunks)} chunks...")
                st.session_state["vector_store"] = create_vector_store(
                    all_chunks, persist=persist_db
                )
                st.session_state["chunk_count"] = len(all_chunks)
                mode = "disk" if persist_db else "RAM"
                status.update(
                    label=f"Done · {len(all_chunks)} chunks · stored in {mode}",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(label="No chunks were created", state="error")

    if "chunk_count" in st.session_state:
        st.caption(f"**{st.session_state['chunk_count']}** chunks ready")

    st.divider()

    if st.button(
        "Load saved DB",
        disabled=not persist_db,
        use_container_width=True,
        help="Available only when 'Persist DB to disk' was enabled.",
    ):
        try:
            st.session_state["vector_store"] = load_vector_store()
            st.success("Loaded from disk")
        except FileNotFoundError as e:
            st.warning("No saved DB on disk yet")
        except Exception as e:
            st.warning(f"Cannot load: {e}")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Hero section (when no messages yet) ----
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-section">
          <div class="hero-grid">
            <div class="hero-card">
              <div class="step">Step 1</div>
              <h4>Upload documents</h4>
              <p>Supports PDF, DOCX, TXT, Markdown. Multiple files at once.</p>
            </div>
            <div class="hero-card">
              <div class="step">Step 2</div>
              <h4>Process & Embed</h4>
              <p>Auto-chunked and embedded locally with a multilingual model.</p>
            </div>
            <div class="hero-card">
              <div class="step">Step 3</div>
              <h4>Ask & Cite</h4>
              <p>Ask in natural language — answers come with source citations.</p>
            </div>
          </div>

          <h3>Key features</h3>
          <div class="hero-grid">
            <div class="hero-card">
              <h4>Multi-LLM</h4>
              <p>Anthropic Claude, OpenAI, Google Gemini, Groq, or Ollama (local).</p>
            </div>
            <div class="hero-card">
              <h4>Multilingual</h4>
              <p>Embeddings support 50+ languages including English and Vietnamese.</p>
            </div>
            <div class="hero-card">
              <h4>Source citations</h4>
              <p>Every answer includes file name, page number, and original text.</p>
            </div>
          </div>

          <h3>Try asking</h3>
          <div class="example-chips">
            <span class="example-chip">Summarize the main content</span>
            <span class="example-chip">Who is the author?</span>
            <span class="example-chip">What sections are included?</span>
            <span class="example-chip">Research methodology</span>
            <span class="example-chip">Key conclusions</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"<div class='src-meta'>{s['file']} · page {s['page']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.text(s["content_preview"])

# ---- Chat input ----
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if "vector_store" not in st.session_state:
            st.warning("Please upload documents first.")
        else:
            with st.spinner("Searching for an answer..."):
                result = ask(st.session_state["vector_store"], prompt, k=8)

            st.markdown(result["answer"])
            with st.expander(f"Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f"<div class='src-meta'>{s['file']} · page {s['page']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.text(s["content_preview"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
