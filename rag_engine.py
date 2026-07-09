import os
from queue import Queue
from threading import Thread
from typing import List, Tuple, Optional, Iterator

import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# ── Prompt ─────────────────────────────────────────────────────────────────
# ChatPromptTemplate properly separates system instructions from the human
# turn, which chat models (like Groq's Llama) respond to significantly better
# than a single merged PromptTemplate string.
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise research assistant. Answer questions ONLY using the provided context.

Rules:
- After each factual claim, cite your source inline:
    • For PDFs: [Page X]
    • For web pages: [Source: URL]
- If the answer spans multiple pages, cite all relevant pages.
- If the context does not contain the answer, say exactly:
  "I couldn't find this information in the document."
- Never invent or infer information outside the context.
- Be concise but complete.

Context:
{context}

Chat History:
{chat_history}""",
    ),
    ("human", "{question}"),
])


class StreamingCallbackHandler(BaseCallbackHandler):
    """Collects LLM tokens into a thread-safe Queue for real-time streaming."""

    def __init__(self):
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs) -> None:
        self.queue.put(None)


class RAGEngine:
    def __init__(self):
        # ── Embedding model ────────────────────────────────────────────────
        # all-mpnet-base-v2 has a 512-token limit (vs 256 for all-MiniLM-L6-v2),
        # which matches our chunk_size much better and produces higher-quality
        # embeddings. Still fully free and runs on CPU.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── LLM ───────────────────────────────────────────────────────────
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
            streaming=True,  # Required for token-by-token streaming in the UI
        )

        # ── Text splitter ─────────────────────────────────────────────────
        # chunk_size=800 chars keeps chunks well within the 512-token embedding
        # limit. chunk_overlap=100 prevents answers from being cut at boundaries.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore: Optional[Chroma] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.source_name: str = ""
        self.doc_metadata: dict = {}

    # ── Loaders ────────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> Tuple[List[Document], dict]:
        """Load a PDF and return (documents, metadata)."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Normalize to 1-indexed page numbers for human-readable citations
        for doc in documents:
            if "page" in doc.metadata:
                doc.metadata["page_display"] = doc.metadata["page"] + 1

        metadata = {
            "type": "pdf",
            "pages": len(documents),
            "source": os.path.basename(file_path),
        }
        return documents, metadata

    def load_url(self, url: str) -> Tuple[List[Document], dict]:
        """
        Scrape a URL using trafilatura, which strips nav/footer boilerplate
        and handles many sites that WebBaseLoader fails on.
        """
        # Validate URL format before hitting the network
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(f"Could not reach the URL: {url}")

        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_comments=False,
            no_fallback=False,
        )
        if not text or len(text.strip()) < 100:
            raise ValueError(
                "Could not extract readable text from that URL. "
                "The page may be JavaScript-rendered or behind a login wall."
            )

        doc = Document(
            page_content=text,
            metadata={
                "source_url": url,
                "page_display": "web",
                "source": url,
            },
        )
        metadata = {"type": "url", "pages": 1, "source": url}
        return [doc], metadata

    # ── Processing ─────────────────────────────────────────────────────────

    def process_documents(self, documents: List[Document], metadata: dict) -> int:
        """Chunk → embed → store → build retrieval chain. Returns chunk count."""
        self.doc_metadata = metadata
        self.source_name = metadata["source"]

        chunks = self.text_splitter.split_documents(documents)

        # Always rebuild the vectorstore fresh so a new document fully replaces
        # the previous one without stale chunks leaking through.
        if self.vectorstore:
            self.vectorstore.delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="rag_docs",
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",               # Diverse results, not just nearest
                search_kwargs={"k": 5, "fetch_k": 12},
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CHAT_PROMPT},
            verbose=False,
        )

        return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Run a question through the RAG chain. Returns (answer, sources)."""
        if not self.chain:
            return (
                "⚠️ No document loaded. Please upload a PDF or enter a URL first.",
                [],
            )

        result = self.chain.invoke({"question": question})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        return answer, self._deduplicate_sources(sources)

    def stream(self, question: str) -> Tuple[Iterator[str], List[Document]]:
        """
        Returns a generator that yields answer tokens in real time via
        LangChain callbacks plus the source documents used.

        Usage in Streamlit:
            tokens, sources = engine.stream(question)
            st.write_stream(tokens)
        """
        if not self.chain:
            def _err():
                yield "⚠️ No document loaded. Please upload a PDF or enter a URL first."
            return _err(), []

        handler = StreamingCallbackHandler()

        # Retrieve source docs up front so they're available immediately
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 12},
        )
        raw_sources = retriever.invoke(question)
        unique_sources = self._deduplicate_sources(raw_sources)

        def _token_generator():
            # Run chain.invoke in a background thread so the callback handler
            # can feed tokens into the queue as the LLM produces them.
            thread = Thread(target=lambda: self.chain.invoke(
                {"question": question},
                callbacks=[handler],
            ))
            thread.start()

            while True:
                token = handler.queue.get()
                if token is None:
                    break
                yield token

            thread.join()

        return _token_generator(), unique_sources

    # ── Helpers ────────────────────────────────────────────────────────────

    def _deduplicate_sources(self, sources: List[Document]) -> List[Document]:
        """Remove duplicate chunks that reference the same page/URL."""
        seen: set = set()
        unique: List[Document] = []
        for s in sources:
            key = (
                s.metadata.get("page_display"),
                s.metadata.get("source", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def is_ready(self) -> bool:
        return self.chain is not None

    def clear(self):
        """Fully reset the engine so a new document can be loaded."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None
        self.chain = None
        self.source_name = ""
        self.doc_metadata = {}