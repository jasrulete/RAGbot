import streamlit as st
import tempfile
import os
from rag_engine import RAGEngine

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4A90D9;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .source-label {
        font-weight: 600;
        color: #4A90D9;
        margin-bottom: 4px;
    }
    .source-text {
        color: #555;
        line-height: 1.5;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ready { background: #d4edda; color: #155724; }
    .badge-empty { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}

# ── RAG Engine (cached so it persists across reruns) ───────────────────────
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 DocChat")
    st.caption("Upload a document, then ask anything about it.")
    st.divider()

    # Status
    if st.session_state.doc_loaded:
        info = st.session_state.doc_info
        st.markdown('<span class="status-badge badge-ready">✅ Document Ready</span>', unsafe_allow_html=True)
        st.caption(f"**Source:** {info.get('source', 'Unknown')}")
        st.caption(f"**Pages/Sections:** {info.get('pages', '?')} → {info.get('chunks', '?')} chunks")
    else:
        st.markdown('<span class="status-badge badge-empty">⚠️ No Document Loaded</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Load a Document")

    tab_pdf, tab_url = st.tabs(["📄 PDF", "🌐 URL"])

    # ── PDF Tab ────────────────────────────────────────────────────────────
    with tab_pdf:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            label_visibility="collapsed",
        )
        process_pdf = st.button("Process PDF", use_container_width=True, type="primary")

        if uploaded_file and process_pdf:
            with st.spinner("Reading and indexing PDF..."):
                # Save to temp file (PyPDFLoader needs a path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    docs, metadata = engine.load_pdf(tmp_path)
                    num_chunks = engine.process_documents(docs, metadata)
                    metadata["chunks"] = num_chunks

                    st.session_state.doc_loaded = True
                    st.session_state.doc_info = {**metadata, "source": uploaded_file.name}
                    st.session_state.messages = []  # Clear old chat
                    st.success(f"✅ Indexed {metadata['pages']} pages → {num_chunks} chunks")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(tmp_path)

    # ── URL Tab ────────────────────────────────────────────────────────────
    with tab_url:
        url_input = st.text_input(
            "Paste a URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
        )
        process_url = st.button("Process URL", use_container_width=True, type="primary")

        if url_input and process_url:
            with st.spinner("Fetching and indexing page..."):
                try:
                    docs, metadata = engine.load_url(url_input)
                    num_chunks = engine.process_documents(docs, metadata)
                    metadata["chunks"] = num_chunks

                    st.session_state.doc_loaded = True
                    st.session_state.doc_info = {**metadata}
                    st.session_state.messages = []
                    st.success(f"✅ Indexed {metadata['pages']} section(s) → {num_chunks} chunks")
                except Exception as e:
                    st.error(f"Error fetching URL: {e}")

    st.divider()
    if st.button("🗑️ Clear & Start Over", use_container_width=True):
        engine.clear()
        st.session_state.doc_loaded = False
        st.session_state.doc_info = {}
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Tip:** Try asking:\n- *'Summarize this document'*\n- *'What does it say about X?'*\n- *'List the key points'*")

# ── Main Chat Area ─────────────────────────────────────────────────────────
st.header("Chat with Your Document")

if not st.session_state.doc_loaded:
    st.info("👈 Upload a PDF or enter a URL in the sidebar to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s) used"):
                for src in msg["sources"]:
                    page = src.metadata.get("page_display", "?")
                    source_label = f"Page {page}" if page != "web" else src.metadata.get("source_url", "Web")
                    snippet = src.page_content[:350].replace("\n", " ").strip()
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-label">📍 {source_label}</div>'
                        f'<div class="source-text">{snippet}…</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer…"):
            answer, sources = engine.query(prompt)
        st.markdown(answer)

        if sources:
            with st.expander(f"📄 {len(sources)} source(s) used"):
                for src in sources:
                    page = src.metadata.get("page_display", "?")
                    source_label = f"Page {page}" if page != "web" else src.metadata.get("source_url", "Web")
                    snippet = src.page_content[:350].replace("\n", " ").strip()
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-label">📍 {source_label}</div>'
                        f'<div class="source-text">{snippet}…</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })