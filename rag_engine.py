import os
import tempfile
from typing import List, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


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
        )
        self.vectorstore: Optional[Chroma] = None
        self.chain = None
        self.chat_history: List = []
        self.source_name: str = ""
        self.doc_metadata: dict = {}

    # ── Loaders ────────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> Tuple[List[Document], dict]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
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
        loader = WebBaseLoader(web_paths=[url], bs_kwargs={"features": "lxml"})
        documents = loader.load()
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
        self.doc_metadata = metadata
        self.source_name = metadata["source"]
        self.chat_history = []

        chunks = self.text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="rag_docs",
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10},
        )

        # Step 1: Rephrase the question given chat history
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("human", (
                "Given the conversation above, rephrase the follow-up question "
                "into a standalone question that can be understood without the history."
            )),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, rephrase_prompt
        )

        # Step 2: Answer using retrieved context
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise research assistant. Answer ONLY using the context below.\n\n"
                "Rules:\n"
                "- Cite your source after each claim: [Page X] for PDFs, [Source: URL] for web.\n"
                "- If the answer isn't in the context, say: "
                "'I couldn't find this information in the document.'\n"
                "- Do NOT make up information.\n\n"
                "Context:\n{context}"
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(self.llm, answer_prompt)

        self.chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> Tuple[str, List[Document]]:
        if not self.chain:
            return "⚠️ No document loaded. Please upload a PDF or enter a URL first.", []

        result = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })

        answer = result["answer"]
        sources = result.get("context", [])

        # Append to chat history for next turn
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

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
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.source_name = ""
        self.doc_metadata = {}