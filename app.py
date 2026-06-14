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
    /* Source citation cards */
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

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ready { background: #d4edda; color: #155724; }
    .badge-empty { background: #fff3cd; color: #856404; }

    /* Make the chat input always visible at the bottom */
    .stChatFloatingInputContainer { bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def render_sources(sources: list):
    """
    Render a list of source Document objects as styled citation cards.
    Single function used everywhere — change once, updates everywhere.
    """
    for src in sources:
        page = src.metadata.get("page_display", "?")
        if page == "web":
            source_label = src.metadata.get("source_url", "Web source")
        else:
            source_label = f"Page {page}"

        snippet = src.page_content[:350].replace("\n", " ").strip()
        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-label">📍 {source_label}</div>'
            f'<div class="source-text">{snippet}…</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}


# ── RAG Engine ─────────────────────────────────────────────────────────────
# Cached so the embedding model (~420MB for all-mpnet-base-v2) is only loaded
# once per server session, not on every Streamlit rerun.
@st.cache_resource(show_spinner=False)
def get_engine():
    return RAGEngine()

# Show a friendly message the very first time the model downloads
if "engine_ready" not in st.session_state:
    with st.spinner("⚙️ Loading embedding model — this only happens once…"):
        engine = get_engine()
    st.session_state.engine_ready = True
else:
    engine = get_engine()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 DocChat")
    st.caption("Upload a document or paste a URL, then ask anything about it.")
    st.divider()

    # ── Document status ────────────────────────────────────────────────────
    if st.session_state.doc_loaded:
        info = st.session_state.doc_info
        st.markdown(
            '<span class="status-badge badge-ready">✅ Document Ready</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"**Source:** {info.get('source', 'Unknown')}")
        st.caption(
            f"**Pages/Sections:** {info.get('pages', '?')} → {info.get('chunks', '?')} chunks"
        )
    else:
        st.markdown(
            '<span class="status-badge badge-empty">⚠️ No Document Loaded</span>',
            unsafe_allow_html=True,
        )

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
            with st.spinner("Reading and indexing PDF…"):
                # PyPDFLoader requires a file path, not a file object
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    docs, metadata = engine.load_pdf(tmp_path)
                    num_chunks = engine.process_documents(docs, metadata)
                    metadata["chunks"] = num_chunks

                    st.session_state.doc_loaded = True
                    st.session_state.doc_info = {**metadata, "source": uploaded_file.name}
                    st.session_state.messages = []  # Clear old chat on new doc
                    st.success(f"✅ Indexed {metadata['pages']} pages → {num_chunks} chunks")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(tmp_path)  # Always clean up the temp file

    # ── URL Tab ────────────────────────────────────────────────────────────
    with tab_url:
        url_input = st.text_input(
            "Paste a URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
        )
        process_url = st.button("Process URL", use_container_width=True, type="primary")

        if url_input and process_url:
            # Client-side validation before hitting the network
            if not url_input.startswith(("http://", "https://")):
                st.error("Please include https:// at the start of the URL.")
            else:
                with st.spinner("Fetching and indexing page…"):
                    try:
                        docs, metadata = engine.load_url(url_input)
                        num_chunks = engine.process_documents(docs, metadata)
                        metadata["chunks"] = num_chunks

                        st.session_state.doc_loaded = True
                        st.session_state.doc_info = {**metadata}
                        st.session_state.messages = []
                        st.success(
                            f"✅ Indexed {metadata['pages']} section(s) → {num_chunks} chunks"
                        )
                    except ValueError as e:
                        # Friendly errors from our loader (bad URL, no text, etc.)
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    st.divider()

    if st.button("🗑️ Clear & Start Over", use_container_width=True):
        engine.clear()
        st.session_state.doc_loaded = False
        st.session_state.doc_info = {}
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**Try asking:**\n"
        "- *'Summarize this document'*\n"
        "- *'What does it say about X?'*\n"
        "- *'List the key points on page 3'*"
    )


# ── Main Chat Area ─────────────────────────────────────────────────────────
st.header("Chat with Your Document")

if not st.session_state.doc_loaded:
    st.info("👈 Upload a PDF or enter a URL in the sidebar to get started.")
    st.stop()

# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s) used"):
                render_sources(msg["sources"])  # ← single shared function

# ── Handle new user input ──────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document…"):
            token_gen, sources = engine.stream(prompt)

        full_answer = st.write_stream(token_gen)

        if sources:
            with st.expander(f"📄 {len(sources)} source(s) used"):
                render_sources(sources)

    # Persist the completed message (including sources) in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })