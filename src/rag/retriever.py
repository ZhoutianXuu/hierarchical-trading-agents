"""
FAISS Vector Store for RAG

Manages FAISS index for efficient similarity search.
"""

import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
import json

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manage FAISS index for similarity search"""
    
    def __init__(self, dimension: int = 384, index_type: str = "IVF"):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ("Flat" or "IVF")
        """
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
    
    def build_index(self, embeddings: np.ndarray, documents: List[str], 
                   metadata: List[Dict], n_clusters: int = 100):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of embeddings
            documents: List of document texts
            metadata: List of metadata dicts
            n_clusters: Number of clusters for IVF index
        """
        n_embeddings = len(embeddings)
        logger.info(f"Building FAISS index for {n_embeddings} embeddings...")
        
        if self.index_type == "IVF" and n_embeddings >= 1000:
            # Use IVF index for large datasets
            n_clusters = min(n_clusters, n_embeddings // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dimension, 
                n_clusters,
                faiss.METRIC_INNER_PRODUCT
            )
            
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            self.index.train(embeddings)
            logger.info("Index training complete")
        else:
            # Use flat index for small datasets or if specified
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Using flat index (exact search)")
        
        # Add embeddings to index
        logger.info("Adding embeddings to index...")
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = metadata
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document_text, similarity_score, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Validate index
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, directory: str):
        """
        Save index and metadata to disk
        
        Args:
            directory: Directory to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to {directory}/index.faiss")
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save documents
        logger.info(f"Saving documents to {directory}/documents.pkl")
        with open(directory / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        logger.info(f"Saving metadata to {directory}/metadata.pkl")
        with open(directory / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save index info
        info = {
            'num_documents': len(self.documents),
            'embedding_dimension': self.dimension,
            'index_type': self.index_type,
            'total_vectors': int(self.index.ntotal)
        }
        
        with open(directory / "index_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Vector store saved to {directory}")
    
    def load(self, directory: str):
        """
        Load index and metadata from disk
        
        Args:
            directory: Directory to load from
        """
        directory = Path(directory)
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {directory}/index.faiss")
        self.index = faiss.read_index(str(directory / "index.faiss"))
        
        # Load documents
        logger.info(f"Loading documents from {directory}/documents.pkl")
        with open(directory / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load metadata
        logger.info(f"Loading metadata from {directory}/metadata.pkl")
        with open(directory / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load info
        with open(directory / "index_info.json", 'r') as f:
            info = json.load(f)
            self.dimension = info['embedding_dimension']
            self.index_type = info.get('index_type', 'unknown')
        
        logger.info(f"Loaded vector store with {len(self.documents)} documents")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': int(self.index.ntotal) if self.index else 0,
            'total_documents': len(self.documents),
            'embedding_dimension': self.dimension,
            'index_type': self.index_type
        }
    
    def delete(self, indices: List[int]):
        """
        Delete documents from index (not supported by all FAISS indices)
        
        Args:
            indices: List of document indices to delete
        """
        raise NotImplementedError("Document deletion not supported by current index type")