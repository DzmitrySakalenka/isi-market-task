"""
Data processing module for the knowledge base aware QA system.

This module handles:
1. Loading and parsing metadata from metadata.jsonl
2. Extracting text from PDF files
3. Chunking text while preserving metadata associations
4. Preparing data for vector store ingestion
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data acquisition and preprocessing for the QA system."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing PDF files and metadata.jsonl
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "metadata.jsonl"
        self.metadata_dict = {}
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_metadata(self) -> Dict[str, Dict]:
        """
        Load metadata from metadata.jsonl file.
        
        Returns:
            Dictionary mapping UUID to metadata
        """
        logger.info(f"Loading metadata from {self.metadata_file}")
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        metadata_dict = {}
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    metadata = json.loads(line.strip())
                    uuid = metadata.get('uuid')
                    if uuid:
                        metadata_dict[uuid] = metadata
                    else:
                        logger.warning(f"Missing UUID in line {line_num}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in line {line_num}: {e}")
        
        logger.info(f"Loaded metadata for {len(metadata_dict)} documents")
        self.metadata_dict = metadata_dict
        return metadata_dict
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get list of all PDF files in the data directory.
        
        Returns:
            List of PDF file paths
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def extract_uuid_from_filename(self, pdf_path: Path) -> Optional[str]:
        """
        Extract UUID from PDF filename (assuming filename is the UUID).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            UUID string or None if not found
        """
        # Remove .pdf extension to get UUID
        uuid = pdf_path.stem
        return uuid if uuid in self.metadata_dict else None
    
    def process_pdf_with_metadata(self, pdf_path: Path) -> Tuple[str, Dict, str]:
        """
        Process a single PDF file and associate it with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (text, metadata, uuid)
        """
        uuid = self.extract_uuid_from_filename(pdf_path)
        if not uuid:
            logger.warning(f"No metadata found for {pdf_path.name}")
            return "", {}, ""
        
        metadata = self.metadata_dict.get(uuid, {})
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
        
        return text, metadata, uuid
    
    def chunk_text_with_metadata(self, text: str, metadata: Dict, uuid: str) -> List[Dict]:
        """
        Split text into chunks while preserving metadata association.
        
        Args:
            text: Text to be chunked
            metadata: Metadata associated with the text
            uuid: UUID of the source document
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text.strip():
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'text': chunk,
                'metadata': {
                    **metadata,  # Include all original metadata
                    'chunk_id': f"{uuid}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source_uuid': uuid
                }
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def process_all_documents(self) -> List[Dict]:
        """
        Process all PDF documents in the data directory.
        
        Returns:
            List of chunk documents with text and metadata
        """
        # Load metadata first
        self.load_metadata()
        
        # Get all PDF files
        pdf_files = self.get_pdf_files()
        
        all_chunks = []
        processed_count = 0
        
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}")
            
            # Extract text and get metadata
            text, metadata, uuid = self.process_pdf_with_metadata(pdf_path)
            
            if text and uuid:
                # Chunk the text
                chunks = self.chunk_text_with_metadata(text, metadata, uuid)
                all_chunks.extend(chunks)
                processed_count += 1
                logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
            else:
                logger.warning(f"Skipped {pdf_path.name} - no text or metadata")
        
        logger.info(f"Processed {processed_count} documents, created {len(all_chunks)} total chunks")
        return all_chunks
    
    def get_document_stats(self) -> Dict:
        """
        Get statistics about the processed documents.
        
        Returns:
            Dictionary with statistics
        """
        pdf_files = self.get_pdf_files()
        metadata_count = len(self.metadata_dict) if self.metadata_dict else 0
        
        return {
            'total_pdf_files': len(pdf_files),
            'total_metadata_entries': metadata_count,
            'data_directory': str(self.data_dir)
        }


def main():
    """Main function for testing the data processing functionality."""
    processor = DataProcessor()
    
    # Get stats
    stats = processor.get_document_stats()
    print("Document Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Process a few documents for testing
    processor.load_metadata()
    pdf_files = processor.get_pdf_files()
    
    if pdf_files:
        # Test with first PDF
        test_pdf = pdf_files[0]
        print(f"\nTesting with: {test_pdf.name}")
        
        text, metadata, uuid = processor.process_pdf_with_metadata(test_pdf)
        print(f"Extracted {len(text)} characters")
        print(f"Metadata: {metadata.get('title', 'No title')}")
        
        chunks = processor.chunk_text_with_metadata(text, metadata, uuid)
        print(f"Created {len(chunks)} chunks")
        
        if chunks:
            print(f"First chunk preview: {chunks[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
