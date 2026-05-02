# app/ingestion.py
"""
Data ingestion module to build FAISS vector database from markdown files.
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib

import numpy as np
import pandas as pd
try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DataIngestion:
    """Ingest markdown files into FAISS vector database."""
    
    def __init__(self, data_dir: str = "data", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ingestion system.
        
        Args:
            data_dir: Root directory containing data folders (claude, hackerrank, visa)
            embedding_model: Sentence transformer model name
        """
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadata = []
        
    def _load_embedder(self):
        """Lazy load the embedding model."""
        if self.embedder is None and SentenceTransformer:
            self.embedder = SentenceTransformer(self.embedding_model)
        elif self.embedder is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        return self.embedder
    
    def _load_documents(self) -> List[Tuple[str, str]]:
        """
        Load all markdown files from data directory.
        
        Returns:
            List of (file_path, content) tuples
        """
        documents = []
        
        for md_file in self.data_dir.rglob("*.md"):
            # Skip index files at root of each product folder
            if md_file.name == "index.md":
                continue
                
            try:
                content = md_file.read_text(encoding="utf-8")
                # Get relative path for categorization
                rel_path = md_file.relative_to(self.data_dir)
                parts = rel_path.parts
                
                # Extract metadata from folder structure
                # data/claude/claude-code/... → company: claude, product: claude-code
                # data/hackerrank/chakra/... → company: hackerrank, product: chakra
                # data/visa/support/... → company: visa, product: support
                
                company = parts[0] if len(parts) > 0 else "unknown"
                product = parts[1] if len(parts) > 1 else "unknown"
                
                # Extract topic from filename (remove numeric ID prefix)
                filename = md_file.name
                topic = filename
                if "-" in filename:
                    # Try to extract topic after numeric ID
                    name_without_ext = filename.rsplit(".", 1)[0] if "." in filename else filename
                    parts_name = name_without_ext.split("-", 1)
                    if len(parts_name) > 1 and parts_name[0].isdigit():
                        topic = parts_name[1]
                    else:
                        topic = name_without_ext
                
                documents.append({
                    "path": str(md_file),
                    "content": content,
                    "company": company,
                    "product_area": product,
                    "filename": md_file.name,
                    "topic": topic
                })
            except Exception as e:
                print(f"Warning: Could not read {md_file}: {e}")
                
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If single paragraph is too large, split by sentences
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk, but keep some overlap
                if overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-overlap:] + "\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para
                
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        embedder = self._load_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=True)
        return embeddings.astype("float32")
    
    def build_index(self, chunk_size: int = 500, overlap: int = 50, 
                    force_rebuild: bool = False) -> dict:
        """
        Build FAISS index from all documents.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            force_rebuild: If True, rebuild even if index exists
            
        Returns:
            Statistics about the indexed documents
        """
        if faiss is None:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
        
        # Load documents
        documents = self._load_documents()
        if not documents:
            return {"error": "No documents found"}
        
        # Chunk all documents
        all_chunks = []
        chunk_metadata = []
        
        for doc in documents:
            chunks = self._chunk_text(doc["content"], chunk_size, overlap)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "path": doc["path"],
                    "company": doc["company"],
                    "product_area": doc["product_area"],
                    "filename": doc["filename"],
                    "topic": doc["topic"]
                })
        
        if not all_chunks:
            return {"error": "No text chunks created"}
        
        # Create embeddings
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._create_embeddings(all_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Store metadata
        self.documents = all_chunks
        self.metadata = chunk_metadata
        
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "dimension": dimension,
            "companies": list(set(d["company"] for d in chunk_metadata)),
            "product_areas": list(set(d["product_area"] for d in chunk_metadata))
        }
        
        return stats
    
    def save_index(self, index_path: str = "data/index.faiss", 
                   metadata_path: str = "data/metadata.json"):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")
            
        # Create directory if needed
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, indent=2)
            
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str = "data/index.faiss",
                   metadata_path: str = "data/metadata.json"):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
        """
        if faiss is None:
            raise ImportError("faiss-cpu not installed")
            
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "r",  encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search the vector database.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        if self.index is None:
            raise ValueError("No index loaded. Call build_index() or load_index() first.")
            
        # Create query embedding
        embedder = self._load_embedder()
        query_embedding = embedder.encode([query]).astype("float32")
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(dist)
                })
                
        return results


def ingest_data(data_dir: str = "data", output_dir: str = "data") -> dict:
    """
    Main function to ingest all data files into FAISS.
    
    Args:
        data_dir: Source data directory
        output_dir: Output directory for index files
        
    Returns:
        Statistics about the indexed data
    """
    ingestion = DataIngestion(data_dir=data_dir)
    
    # Build index
    print("Building FAISS index...")
    stats = ingestion.build_index()
    print(f"Indexed {stats.get('total_chunks', 0)} chunks from {stats.get('total_documents', 0)} documents")
    
    # Save index
    index_path = os.path.join(output_dir, "index.faiss")
    metadata_path = os.path.join(output_dir, "metadata.json")
    ingestion.save_index(index_path, metadata_path)
    
    return stats


if __name__ == "__main__":
    stats = ingest_data()
    print("\nIngestion complete!")
    print(f"Product areas indexed: {stats.get('product_areas', [])}")