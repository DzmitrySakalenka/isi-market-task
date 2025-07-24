#!/usr/bin/env python3
"""
Diagnostic script to test the complete document processing pipeline step by step.
"""

import sys
import traceback

def test_step_by_step():
    print("🔍 Step-by-step diagnostic...")
    
    try:
        # Step 1: Test imports
        print("\n1. Testing imports...")
        from src.rag_system import RAGSystem
        from src.data_processing import DataProcessor
        print("Imports successful")
        
        # Step 2: Test initialization
        print("\n2. Testing initialization...")
        rag = RAGSystem()
        processor = DataProcessor()
        print("Components initialized")
        
        # Step 3: Test metadata loading
        print("\n3. Testing metadata loading...")
        processor.load_metadata()
        print(f"Metadata loaded: {len(processor.metadata_dict)} entries")
        
        # Step 4: Test PDF file discovery
        print("\n4. Testing PDF file discovery...")
        pdf_files = processor.get_pdf_files()
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found!")
            return False
        
        # Step 5: Test single document processing
        print("\n5. Testing single document processing...")
        test_file = pdf_files[0]
        print(f"Testing with: {test_file.name}")
        
        text, metadata, uuid = processor.process_pdf_with_metadata(test_file)
        print(f"Text length: {len(text)} characters")
        print(f"UUID: {uuid}")
        print(f"Title: {metadata.get('title', 'No title')}")
        
        if not text or not uuid:
            print("Failed to extract text or UUID")
            return False
        
        # Step 6: Test text chunking
        print("\n6. Testing text chunking...")
        chunks = processor.chunk_text_with_metadata(text, metadata, uuid)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            print("No chunks created")
            return False
        
        # Show sample chunk
        sample_chunk = chunks[0]
        print(f"Sample chunk text: {sample_chunk['text'][:100]}...")
        print(f"Sample metadata keys: {list(sample_chunk['metadata'].keys())}")
        
        # Step 7: Test embedding
        print("\n7. Testing embedding...")
        
        # Check current database state
        initial_stats = rag.get_collection_stats()
        print(f"Initial document count: {initial_stats.get('document_count', 0)}")
        
        # Try to embed the chunks
        success = rag.embed_documents(chunks)
        print(f"Embedding success: {success}")
        
        if success:
            final_stats = rag.get_collection_stats()
            final_count = final_stats.get('document_count', 0)
            print(f"Final document count: {final_count}")
            
            if final_count > 0:
                print("\nSUCCESS! Database write is working!")
                
                # Test search
                print("\n8. Testing search...")
                results = rag.similarity_search("economic growth", k=2)
                print(f"Search results: {len(results)}")
                for i, result in enumerate(results):
                    title = result['metadata'].get('title', 'Unknown')
                    score = 1 - result['distance']
                    print(f"  {i+1}. {title} (score: {score:.3f})")
                
                return True
            else:
                print("Documents not persisted in database")
                return False
        else:
            print("Embedding failed")
            return False
            
    except Exception as e:
        print(f"\nError in step-by-step test: {e}")
        traceback.print_exc()
        return False

def run_batch_processing():
    print("\nRunning batch processing...")
    
    try:
        from src.rag_system import RAGSystem
        
        rag = RAGSystem()
        
        # Check initial state
        stats = rag.get_collection_stats()
        print(f"Initial documents: {stats.get('document_count', 0)}")
        
        # Run processing with small batch
        print("Starting batch processing (batch_size=10)...")
        success = rag.process_and_embed_all_documents(batch_size=10)
        
        if success:
            final_stats = rag.get_collection_stats()
            final_count = final_stats.get('document_count', 0)
            print(f"Processing complete! Final count: {final_count}")
            return True
        else:
            print("Batch processing failed")
            return False
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        traceback.print_exc()
        return False

def main():
    print("🩺 Knowledge Base QA System Diagnostics")
    print("=" * 50)
    
    # First test step by step
    step_success = test_step_by_step()
    
    if step_success:
        print("\n" + "="*50)
        print("Single document test passed!")
        print("The system components are working correctly.")
        
        # Ask about running full batch processing
        print("\nThe database write functionality is confirmed to work.")
        print("You can now run the full processing:")
        print("  python -c \"from src.rag_system import RAGSystem; rag = RAGSystem(); rag.process_and_embed_all_documents()\"")
        
    else:
        print("\n" + "="*50)
        print("Diagnostic test failed!")
        print("There's an issue with the basic functionality.")
        
    return step_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 