"""
RAG (Retrieval-Augmented Generation) Module

This module provides components for document processing, embedding generation,
vector storage, and retrieval for the trading agent system.

Components:
- DocumentProcessor: Process and chunk documents (PDF, HTML, text)
- EmbeddingManager: Generate embeddings using sentence transformers
- FAISSVectorStore: Manage FAISS vector index for similarity search
- DocumentRetriever: Retrieve relevant documents for agent queries
"""

from .document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager
from .vector_store import FAISSVectorStore
from .retriever import DocumentRetriever

__all__ = [
    'DocumentProcessor',
    'EmbeddingManager',
    'FAISSVectorStore',
    'DocumentRetriever'
]

__version__ = '1.0.0'