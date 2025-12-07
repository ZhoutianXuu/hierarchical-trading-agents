"""
Build RAG Vector Store from Documents

This script processes financial documents (PDFs, HTML, text) and builds
a FAISS vector store for retrieval-augmented generation.

Usage:
    # Build index from all documents in a directory
    python scripts/build_rag_index.py --documents data/documents --output rag_store/faiss_index
    
    # Build with custom settings
    python scripts/build_rag_index.py --documents data/documents \
        --output rag_store/faiss_index --chunk-size 1000 --chunk-overlap 100
    
    # Build from specific document types
    python scripts/build_rag_index.py --documents data/documents/sec_filings \
        --output rag_store/sec_index --file-types pdf
"""

import argparse
import logging
from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm
import numpy as np

# Document processing
import PyPDF2
from bs4 import BeautifulSoup

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents and chunk them"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, filepath: Path) -> List[Dict]:
        """Extract text from PDF and chunk it"""
        try:
            chunks = []
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    full_text += text + "\n\n"
                
                # Chunk the text
                text_chunks = self._chunk_text(full_text)
                
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(filepath),
                            'type': 'pdf',
                            'chunk_id': i,
                            'total_chunks': len(text_chunks)
                        }
                    })
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing PDF {filepath}: {e}")
            return []
    
    def process_html(self, filepath: Path) -> List[Dict]:
        """Extract text from HTML documents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks_text if chunk)
                
                # Chunk the text
                text_chunks = self._chunk_text(text)
                
                chunks = []
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(filepath),
                            'type': 'html',
                            'chunk_id': i,
                            'total_chunks': len(text_chunks)
                        }
                    })
                
                return chunks
        
        except Exception as e:
            logger.error(f"Error processing HTML {filepath}: {e}")
            return []
    
    def process_text(self, filepath: Path) -> List[Dict]:
        """Process plain text files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Chunk the text
                text_chunks = self._chunk_text(text)
                
                chunks = []
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(filepath),
                            'type': 'text',
                            'chunk_id': i,
                            'total_chunks': len(text_chunks)
                        }
                    })
                
                return chunks
        
        except Exception as e:
            logger.error(f"Error processing text {filepath}: {e}")
            return []
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # Simple character-based chunking
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence ending
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim != -1:
                        end = start + last_delim + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks


class RAGIndexBuilder:
    """Build FAISS index for RAG"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for documents"""
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        return embeddings.astype('float32')
    
    def build_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
        """Build FAISS index"""
        n_embeddings = len(embeddings)
        logger.info(f"Building FAISS index for {n_embeddings} embeddings...")
        
        if n_embeddings < 1000:
            # Use simple flat index for small datasets
            index = faiss.IndexFlatIP(self.dimension)
            logger.info("Using IndexFlatIP (exact search)")
        else:
            # Use IVF index for larger datasets
            n_clusters = min(100, n_embeddings // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
            
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            index.train(embeddings)
            logger.info("Using IndexIVFFlat (approximate search)")
        
        logger.info("Adding embeddings to index...")
        index.add(embeddings)
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def save_index(self, index, documents: List[str], metadata: List[Dict], output_dir: Path):
        """Save index and associated data"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to {output_dir}/index.faiss")
        if hasattr(index, 'index'):  # GPU index
            faiss.write_index(faiss.index_gpu_to_cpu(index), str(output_dir / "index.faiss"))
        else:
            faiss.write_index(index, str(output_dir / "index.faiss"))
        
        # Save documents
        logger.info(f"Saving documents to {output_dir}/documents.pkl")
        with open(output_dir / "documents.pkl", 'wb') as f:
            pickle.dump(documents, f)
        
        # Save metadata
        logger.info(f"Saving metadata to {output_dir}/metadata.pkl")
        with open(output_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save index info
        info = {
            'num_documents': len(documents),
            'embedding_dimension': self.dimension,
            'embedding_model': self.embedding_model.model_name,
            'index_type': type(index).__name__
        }
        
        with open(output_dir / "index_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Index saved successfully to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Build RAG vector store from documents')
    parser.add_argument('--documents', required=True,
                       help='Directory containing documents to index')
    parser.add_argument('--output', required=True,
                       help='Output directory for FAISS index')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Size of text chunks (default: 500)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Overlap between chunks (default: 50)')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Sentence transformer model for embeddings')
    parser.add_argument('--file-types', nargs='+', default=['pdf', 'html', 'txt'],
                       help='File types to process (default: pdf html txt)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for FAISS index (if available)')
    
    args = parser.parse_args()
    
    # Validate paths
    doc_dir = Path(args.documents)
    if not doc_dir.exists():
        logger.error(f"Document directory not found: {doc_dir}")
        return
    
    output_dir = Path(args.output)
    
    # Initialize processor and builder
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    builder = RAGIndexBuilder(embedding_model=args.embedding_model)
    
    # Process documents
    logger.info(f"Processing documents from {doc_dir}")
    all_chunks = []
    all_metadata = []
    
    # Collect all files
    files_to_process = []
    for file_type in args.file_types:
        files_to_process.extend(doc_dir.rglob(f'*.{file_type}'))
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for filepath in tqdm(files_to_process, desc="Processing documents"):
        if filepath.suffix == '.pdf':
            chunks = processor.process_pdf(filepath)
        elif filepath.suffix in ['.html', '.htm']:
            chunks = processor.process_html(filepath)
        elif filepath.suffix == '.txt':
            chunks = processor.process_text(filepath)
        else:
            continue
        
        for chunk in chunks:
            all_chunks.append(chunk['text'])
            all_metadata.append(chunk['metadata'])
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    if len(all_chunks) == 0:
        logger.error("No chunks created. Check document directory and file types.")
        return
    
    # Generate embeddings
    embeddings = builder.embed_documents(all_chunks)
    
    # Build index
    index = builder.build_index(embeddings, use_gpu=args.use_gpu)
    
    # Save everything
    builder.save_index(index, all_chunks, all_metadata, output_dir)
    
    logger.info("✓ RAG index build complete!")
    logger.info(f"✓ Index saved to: {output_dir}")
    logger.info(f"✓ Total documents: {len(all_chunks)}")
    logger.info(f"✓ Ready to use for retrieval!")


if __name__ == '__main__':
    main()