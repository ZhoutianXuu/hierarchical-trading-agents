"""
Document Processor for RAG

Handles processing of various document types (PDF, HTML, text) and
chunks them for embedding and indexing.
"""

import os
from typing import List, Dict, Tuple
from pathlib import Path
import logging

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG indexing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check for optional dependencies
        if PyPDF2 is None:
            logger.warning("PyPDF2 not installed. PDF processing will be unavailable.")
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup not installed. HTML processing will be unavailable.")
    
    def process_pdf(self, filepath: str) -> List[Dict]:
        """
        Extract text from PDF and chunk it
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List of chunks with metadata
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        chunks = []
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    full_text += text + "\n\n"
                
                # Chunk the text
                text_chunks = self.chunk_text(full_text)
                
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(filepath),
                            'type': 'pdf',
                            'chunk_id': i,
                            'total_chunks': len(text_chunks),
                            'total_pages': len(reader.pages)
                        }
                    })
            
            logger.info(f"Processed PDF {filepath}: {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing PDF {filepath}: {e}")
            return []
    
    def process_html(self, filepath: str) -> List[Dict]:
        """
        Extract text from HTML documents
        
        Args:
            filepath: Path to HTML file
            
        Returns:
            List of chunks with metadata
        """
        if BeautifulSoup is None:
            raise ImportError("BeautifulSoup is required for HTML processing. Install with: pip install beautifulsoup4")
        
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
                text_chunks = self.chunk_text(text)
                
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
                
                logger.info(f"Processed HTML {filepath}: {len(chunks)} chunks")
                return chunks
        
        except Exception as e:
            logger.error(f"Error processing HTML {filepath}: {e}")
            return []
    
    def process_text(self, filepath: str) -> List[Dict]:
        """
        Process plain text files
        
        Args:
            filepath: Path to text file
            
        Returns:
            List of chunks with metadata
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Chunk the text
                text_chunks = self.chunk_text(text)
                
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
                
                logger.info(f"Processed text {filepath}: {len(chunks)} chunks")
                return chunks
        
        except Exception as e:
            logger.error(f"Error processing text {filepath}: {e}")
            return []
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence ending
                for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim != -1:
                        end = start + last_delim + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_directory(self, directory: str, file_types: List[str] = None) -> List[Dict]:
        """
        Process all documents in a directory
        
        Args:
            directory: Path to directory
            file_types: List of file extensions to process (e.g., ['pdf', 'txt'])
            
        Returns:
            List of all chunks from all documents
        """
        if file_types is None:
            file_types = ['pdf', 'html', 'htm', 'txt']
        
        all_chunks = []
        directory = Path(directory)
        
        for file_type in file_types:
            for filepath in directory.rglob(f'*.{file_type}'):
                if file_type == 'pdf':
                    chunks = self.process_pdf(str(filepath))
                elif file_type in ['html', 'htm']:
                    chunks = self.process_html(str(filepath))
                elif file_type == 'txt':
                    chunks = self.process_text(str(filepath))
                else:
                    continue
                
                all_chunks.extend(chunks)
        
        logger.info(f"Processed directory {directory}: {len(all_chunks)} total chunks")
        return all_chunks