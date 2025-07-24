#!/usr/bin/env python3
"""
Simple, reliable document processing script.
Processes documents one by one with detailed progress tracking.
"""

import time
import sys
from pathlib import Path

def process_documents_simple():
    """Process documents one by one with progress tracking."""
    
    print("Simple Document Processing")
    print("=" * 50)
    
    try:
        from src.rag_system import RAGSystem
        from src.data_processing import DataProcessor
        
        # Initialize components
        print("Initializing components...")
        rag = RAGSystem()
        processor = DataProcessor()
        
        # Load metadata
        print("Loading metadata...")
        processor.load_metadata()
        print(f"Loaded {len(processor.metadata_dict)} metadata entries")
        
        # Get PDF files
        pdf_files = processor.get_pdf_files()
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found!")
            return False
        
        # Check initial database state
        initial_stats = rag.get_collection_stats()
        initial_count = initial_stats.get('document_count', 0)
        print(f"Initial database count: {initial_count}")
        
        # Process documents one by one
        successful = 0
        failed = 0
        total_chunks = 0
        
        # Process only first 20 documents for testing
        test_files = pdf_files[:20]
        
        print(f"\nProcessing {len(test_files)} documents...")
        print("Progress: ", end="", flush=True)
        
        for i, pdf_path in enumerate(test_files):
            try:
                # Process single document
                text, metadata, uuid = processor.process_pdf_with_metadata(pdf_path)
                
                if text and uuid:
                    # Create chunks
                    chunks = processor.chunk_text_with_metadata(text, metadata, uuid)
                    
                    if chunks:
                        # Embed chunks immediately
                        success = rag.embed_documents(chunks)
                        
                        if success:
                            successful += 1
                            total_chunks += len(chunks)
                            print("✓", end="", flush=True)
                        else:
                            failed += 1
                            print("✗", end="", flush=True)
                    else:
                        failed += 1
                        print("○", end="", flush=True)  # No chunks
                else:
                    failed += 1
                    print("×", end="", flush=True)  # No text/uuid
                    
            except Exception as e:
                failed += 1
                print("!", end="", flush=True)  # Exception
                
            # Progress update every 5 documents
            if (i + 1) % 5 == 0:
                current_stats = rag.get_collection_stats()
                current_count = current_stats.get('document_count', 0)
                print(f" [{i+1}/{len(test_files)}] DB: {current_count}", end="", flush=True)
        
        print()  # New line after progress
        
        # Final statistics
        final_stats = rag.get_collection_stats()
        final_count = final_stats.get('document_count', 0)
        
        print(f"\nProcessing Results:")
        print(f"  Successful documents: {successful}")
        print(f"  Failed documents: {failed}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Final database count: {final_count}")
        print(f"  Documents added: {final_count - initial_count}")
        
        if final_count > initial_count:
            print(f"\nSUCCESS! Database now has {final_count} documents!")
            
            # Test search functionality
            print("\nTesting search...")
            results = rag.similarity_search("GDP economic growth", k=3)
            print(f"Found {len(results)} relevant documents:")
            
            for i, result in enumerate(results, 1):
                title = result['metadata'].get('title', 'Unknown')
                score = 1 - result['distance']
                print(f"  {i}. {title} (similarity: {score:.3f})")
            
            print(f"\nSystem is ready! Run: python app.py")
            return True
        else:
            print(f"\nNo documents were added to the database")
            return False
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    
    # Check if database already has documents
    try:
        from src.rag_system import RAGSystem
        rag = RAGSystem()
        stats = rag.get_collection_stats()
        current_count = stats.get('document_count', 0)
        
        if current_count > 0:
            print(f"Database already has {current_count} documents!")
            print("System is ready! Run: python app.py")
            return True
    except:
        pass
    
    # Run processing
    return process_documents_simple()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 