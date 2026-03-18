
"""
RAG pipeline implemented as a LangGraph state machine.

Graph flow:
  START → retrieve → grade_documents → generate → END
                            ↓ (no relevant docs)
                      transform_query → retrieve (retry)
"""

from __future__ import annotations

import os
from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

import config


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int


# ─────────────────────────────────────────────
# RAGGraph class
# ─────────────────────────────────────────────

class RAGGraph:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
        self.llm = ChatAnthropic(
            model=config.LLM_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.graph = self._build_graph()

    # ── Graph nodes ────────────────────────────

    def _retrieve(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state["question"])
        return {**state, "documents": docs}

    def _grade_documents(self, state: RAGState) -> RAGState:
        """Keep only documents that are relevant to the question."""
        question = state["question"]
        relevant = []
        for doc in state["documents"]:
            prompt = [
                SystemMessage(
                    "You are a relevance grader. "
                    "Reply with only 'yes' if the document is relevant to the question, "
                    "or 'no' if it is not."
                ),
                HumanMessage(
                    f"Question: {question}\n\nDocument:\n{doc.page_content}"
                ),
            ]
            result = self.llm.invoke(prompt)
            if "yes" in result.content.lower():
                relevant.append(doc)
        return {**state, "documents": relevant}

    def _generate(self, state: RAGState) -> RAGState:
        context = "\n\n".join(d.page_content for d in state["documents"])
        prompt = [
            SystemMessage(
                "You are a helpful assistant. "
                "Answer the question using ONLY the provided context. "
                "If the context does not contain enough information, say so clearly."
            ),
            HumanMessage(
                f"Context:\n{context}\n\nQuestion: {state['question']}"
            ),
        ]
        result = self.llm.invoke(prompt)
        return {**state, "generation": result.content}

    def _transform_query(self, state: RAGState) -> RAGState:
        """Rewrite the query to improve retrieval."""
        prompt = [
            SystemMessage(
                "You are a query rewriter. "
                "Rewrite the question to be more specific and improve document retrieval. "
                "Reply with only the rewritten question."
            ),
            HumanMessage(f"Original question: {state['question']}"),
        ]
        result = self.llm.invoke(prompt)
        return {
            **state,
            "question": result.content.strip(),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _no_docs_answer(self, state: RAGState) -> RAGState:
        return {
            **state,
            "generation": (
                "I could not find relevant documents to answer your question. "
                "Please ingest documents first using the `ingest_documents` tool."
            ),
        }

    # ── Conditional routing ─────────────────────

    def _route_after_grading(
        self, state: RAGState
    ) -> Literal["generate", "transform_query", "no_docs"]:
        if state["documents"]:
            return "generate"
        if state.get("retry_count", 0) >= config.MAX_RETRIES:
            return "no_docs"
        return "transform_query"

    # ── Build graph ─────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(RAGState)

        g.add_node("retrieve", self._retrieve)
        g.add_node("grade_documents", self._grade_documents)
        g.add_node("generate", self._generate)
        g.add_node("transform_query", self._transform_query)
        g.add_node("no_docs", self._no_docs_answer)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate": "generate",
                "transform_query": "transform_query",
                "no_docs": "no_docs",
            },
        )
        g.add_edge("transform_query", "retrieve")
        g.add_edge("generate", END)
        g.add_edge("no_docs", END)

        return g.compile()

    # ── Public API ──────────────────────────────

    def query(self, question: str) -> dict:
        """Run the full RAG pipeline and return the answer + sources."""
        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "generation": "",
            "retry_count": 0,
        }
        final_state = self.graph.invoke(initial_state)
        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "snippet": d.page_content[:200],
            }
            for d in final_state.get("documents", [])
        ]
        return {
            "answer": final_state["generation"],
            "sources": sources,
            "final_question": final_state["question"],
        }

    def ingest(self, path: str, glob: str = "**/*.*") -> dict:
        """
        Load documents from `path` (file or directory), split, embed, and store.
        Supported formats: .txt, .md, .pdf
        """
        path = os.path.abspath(path)
        docs: List[Document] = []

        if os.path.isfile(path):
            docs = self._load_file(path)
        elif os.path.isdir(path):
            loader = DirectoryLoader(
                path,
                glob=glob,
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                silent_errors=True,
            )
            docs = loader.load()
            # Also load PDFs if present
            pdf_loader = DirectoryLoader(
                path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=True,
            )
            docs += pdf_loader.load()
        else:
            return {"status": "error", "message": f"Path not found: {path}"}

        if not docs:
            return {"status": "error", "message": "No documents found at path."}

        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

        return {
            "status": "success",
            "documents_loaded": len(docs),
            "chunks_indexed": len(chunks),
            "path": path,
        }

    def _load_file(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext in (".md", ".markdown"):
            return UnstructuredMarkdownLoader(path).load()
        return TextLoader(path, autodetect_encoding=True).load()

    def list_documents(self) -> dict:
        """Return metadata of all indexed documents."""
        collection = self.vectorstore._collection
        result = collection.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in result["metadatas"]})
        return {
            "total_chunks": len(result["metadatas"]),
            "unique_sources": len(sources),
            "sources": sorted(sources),
        }

    def clear_index(self) -> dict:
        """Delete all documents from the vector store."""
        collection = self.vectorstore._collection
        count_before = collection.count()
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
        return {
            "status": "success",
            "chunks_deleted": count_before,
        }
