import os
import tempfile
from typing import List, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = PromptTemplate(
    template="""You are a precise research assistant. Answer questions ONLY using the provided context.

Rules:
- Always cite your source at the end of each claim using [Page X] for PDFs or [Source: URL] for web pages.
- If the answer spans multiple pages, cite all of them.
- If the context does not contain the answer, say: "I couldn't find this information in the document."
- Do NOT make up information outside the context.
- Be concise but thorough.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (with citations):""",
    input_variables=["context", "chat_history", "question"],
)


class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore: Optional[Chroma] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.source_name: str = ""
        self.doc_metadata: dict = {}

    # ── Loaders ────────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> Tuple[List[Document], dict]:
        """Load a PDF and return documents + metadata."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        metadata = {
            "type": "pdf",
            "pages": len(documents),
            "source": os.path.basename(file_path),
        }
        # Normalize page numbers to 1-indexed in metadata
        for doc in documents:
            if "page" in doc.metadata:
                doc.metadata["page_display"] = doc.metadata["page"] + 1
        return documents, metadata

    def load_url(self, url: str) -> Tuple[List[Document], dict]:
        """Load a URL and return documents + metadata."""
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"features": "html.parser"},
        )
        documents = loader.load()
        # Tag every chunk with the URL as source
        for doc in documents:
            doc.metadata["source_url"] = url
            doc.metadata["page_display"] = "web"
        metadata = {
            "type": "url",
            "pages": len(documents),
            "source": url,
        }
        return documents, metadata

    # ── Processing ─────────────────────────────────────────────────────────

    def process_documents(self, documents: List[Document], metadata: dict) -> int:
        """Chunk documents, embed them, and build the retrieval chain."""
        self.doc_metadata = metadata
        self.source_name = metadata["source"]

        chunks = self.text_splitter.split_documents(documents)

        # Rebuild vectorstore fresh for each new document
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
                search_type="mmr",           # Max Marginal Relevance → diverse results
                search_kwargs={"k": 5, "fetch_k": 10},
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
            verbose=False,
        )

        return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Run a question through the RAG chain."""
        if not self.chain:
            return "⚠️ No document loaded. Please upload a PDF or enter a URL first.", []

        result = self.chain.invoke({"question": question})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        # Deduplicate sources by page
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.metadata.get("page_display"), s.metadata.get("source", ""))
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return answer, unique_sources

    # ── Utils ──────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self.chain is not None

    def clear(self):
        """Reset the engine for a new document."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None
        self.chain = None
        self.source_name = ""
        self.doc_metadata = {}