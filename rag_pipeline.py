# rag_pipeline.py
# RAG pipeline using FAISS + Google Gemini Embeddings

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore(kb_path: str = "knowledge_base.md") -> FAISS:
    """
    Load the knowledge base markdown file, split into chunks,
    embed using Gemini, and store in a FAISS vectorstore.
    """
    # Load the knowledge base
    loader = TextLoader(kb_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embed using Gemini
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_context(vectorstore: FAISS, query: str, k: int = 3) -> str:
    """
    Retrieve top-k relevant chunks from the vectorstore for a given query.
    Returns a concatenated string of the relevant chunks.
    """
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
