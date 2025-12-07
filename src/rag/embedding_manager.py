"""
Embedding Manager for RAG

Handles generation of embeddings using sentence transformers.
"""

import numpy as np
from typing import List, Union
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate and manage embeddings for documents and queries"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 32, 
                       show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True
        )
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            numpy array embedding (shape: [embedding_dim])
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        
        return embedding.astype('float32')
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts
            
        Returns:
            numpy array of embeddings
        """
        return self.embed_documents(texts, show_progress=False)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize if not already normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2
        
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         document_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar documents to query
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Array of document embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute similarities
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.dimension,
            'max_seq_length': self.model.max_seq_length,
        }