"""
RAG (Retrieval Augmented Generation) system for the knowledge base QA.

This module handles:
1. Embedding generation using sentence-transformers
2. Vector store setup and management with ChromaDB
3. Document retrieval based on semantic similarity
4. Answer generation using local LLM via Ollama
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests

from .data_processing import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval Augmented Generation system for document-based QA."""
    
    def __init__(
        self, 
        vector_store_dir: str = "vector_store",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents"
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_dir: Directory for persistent vector storage
            embedding_model_name: Name of the sentence-transformer model
            collection_name: Name of the ChromaDB collection
        """
        self.vector_store_dir = Path(vector_store_dir)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Ensure vector store directory exists
        self.vector_store_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_vector_store()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB client and collection."""
        logger.info("Initializing ChromaDB vector store")
        try:
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_store_dir)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Vector store initialized. Collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the current collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "vector_store_dir": str(self.vector_store_dir)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def embed_documents(self, documents: List[Dict]) -> bool:
        """
        Embed documents and store them in the vector database.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents to embed")
            return False
        
        logger.info(f"Embedding {len(documents)} documents")
        
        try:
            # Extract texts for embedding
            texts = [doc['text'] for doc in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Prepare data for ChromaDB
            ids = [doc['metadata']['chunk_id'] for doc in documents]
            
            # Convert metadata to ChromaDB-compatible format
            metadatas = []
            for doc in documents:
                metadata = doc['metadata'].copy()
                
                # Convert lists to comma-separated strings for ChromaDB
                if 'industries' in metadata and isinstance(metadata['industries'], list):
                    metadata['industries'] = ', '.join(metadata['industries']) if metadata['industries'] else ''
                
                if 'country_codes' in metadata and isinstance(metadata['country_codes'], list):
                    metadata['country_codes'] = ', '.join(metadata['country_codes']) if metadata['country_codes'] else ''
                
                # Ensure all values are scalar types (str, int, float, bool, None)
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ', '.join(str(v) for v in value) if value else ''
                    elif value is not None and not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Add to vector store
            logger.info("Storing embeddings in vector database...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully embedded {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform similarity search to find relevant documents.
        
        Args:
            query: Query string
            k: Number of top results to return
            filters: Optional metadata filters
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Prepare ChromaDB query
            query_params = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add filters if provided
            if filters:
                # Convert filters to ChromaDB format
                where_clause = {}
                for key, value in filters.items():
                    if key in ["country_codes", "industries"] and isinstance(value, list):
                        # Handle array filters - since we store as comma-separated strings,
                        # we need to use contains filtering
                        if len(value) == 1:
                            where_clause[key] = {"$contains": value[0]}
                        else:
                            # For multiple values, we'll use the first one
                            where_clause[key] = {"$contains": value[0]}
                    elif key in ["country_codes", "industries"] and isinstance(value, str):
                        where_clause[key] = {"$contains": value}
                    else:
                        where_clause[key] = value
                
                if where_clause:
                    query_params["where"] = where_clause
            
            # Perform search
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def query_ollama(self, prompt: str, model: str = "llama3.2") -> str:
        """
        Query Ollama LLM for answer generation.
        
        Args:
            prompt: Formatted prompt with context and question
            model: Ollama model name
            
        Returns:
            Generated answer
        """
        if not self.check_ollama_connection():
            return "Error: Ollama service is not available. Please start Ollama and ensure a model is installed."
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error querying LLM: {str(e)}"
    
    def create_rag_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """
        Create a RAG prompt with context and question.
        
        Args:
            question: User's question
            context_docs: Retrieved relevant documents
            
        Returns:
            Formatted prompt
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc['metadata']
            title = metadata.get('title', 'Unknown Document')
            date = metadata.get('date', 'Unknown Date')
            
            context_part = f"Document {i}: {title} ({date})\n{doc['text']}\n"
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based only on the provided context. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer: Based on the provided context,"""
        
        return prompt
    
    def answer_question(
        self, 
        question: str, 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            Dictionary with answer and source information
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self.similarity_search(question, k=k, filters=filters)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "error": "No relevant documents found"
            }
        
        # Step 2: Create RAG prompt
        prompt = self.create_rag_prompt(question, relevant_docs)
        
        # Step 3: Generate answer using Ollama
        answer = self.query_ollama(prompt)
        
        # Step 4: Prepare sources information
        sources = []
        for doc in relevant_docs:
            metadata = doc['metadata']
            
            # Convert comma-separated strings back to lists for display
            country_codes = metadata.get('country_codes', '')
            if isinstance(country_codes, str) and country_codes:
                country_codes = [c.strip() for c in country_codes.split(',') if c.strip()]
            elif not country_codes:
                country_codes = []
            
            industries = metadata.get('industries', '')
            if isinstance(industries, str) and industries:
                industries = [i.strip() for i in industries.split(',') if i.strip()]
            elif not industries:
                industries = []
            
            source = {
                "title": metadata.get('title', 'Unknown'),
                "date": metadata.get('date', 'Unknown'),
                "country_codes": country_codes,
                "industries": industries,
                "similarity_score": 1 - doc['distance']  # Convert distance to similarity
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }
    
    def process_and_embed_all_documents(self, data_dir: str = "data", batch_size: int = 50) -> bool:
        """
        Process all documents and embed them in the vector store in batches.
        
        Args:
            data_dir: Directory containing PDF files and metadata
            batch_size: Number of documents to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting document processing and embedding pipeline")
        
        try:
            # Initialize data processor
            processor = DataProcessor(data_dir)
            
            # Load metadata first
            processor.load_metadata()
            logger.info(f"Loaded metadata for {len(processor.metadata_dict)} documents")
            
            # Get PDF files
            pdf_files = processor.get_pdf_files()
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            if not pdf_files:
                logger.error("No PDF files found")
                return False
            
            # Process in batches
            total_chunks = 0
            successful_docs = 0
            failed_docs = 0
            
            for i in range(0, len(pdf_files), batch_size):
                batch_files = pdf_files[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(pdf_files) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
                
                batch_chunks = []
                
                # Process each PDF in the batch
                for pdf_path in batch_files:
                    try:
                        text, metadata, uuid = processor.process_pdf_with_metadata(pdf_path)
                        
                        if text and uuid:
                            chunks = processor.chunk_text_with_metadata(text, metadata, uuid)
                            batch_chunks.extend(chunks)
                            successful_docs += 1
                        else:
                            failed_docs += 1
                            logger.warning(f"Failed to process {pdf_path.name}")
                            
                    except Exception as e:
                        failed_docs += 1
                        logger.error(f"Error processing {pdf_path.name}: {e}")
                
                # Embed the batch
                if batch_chunks:
                    logger.info(f"Embedding {len(batch_chunks)} chunks from batch {batch_num}")
                    success = self.embed_documents(batch_chunks)
                    if success:
                        total_chunks += len(batch_chunks)
                        logger.info(f"Successfully embedded batch {batch_num}")
                        
                        # Verify the documents were actually added
                        stats = self.get_collection_stats()
                        current_count = stats.get('document_count', 0)
                        logger.info(f"Vector store now has {current_count} documents")
                        
                    else:
                        logger.error(f"Failed to embed batch {batch_num}")
                        return False
                else:
                    logger.warning(f"No valid chunks in batch {batch_num}")
                
                # Clear memory between batches
                del batch_chunks
            
            # Final stats
            final_stats = self.get_collection_stats()
            final_count = final_stats.get('document_count', 0)
            
            logger.info(f"Processing complete!")
            logger.info(f"Successful documents: {successful_docs}")
            logger.info(f"Failed documents: {failed_docs}")
            logger.info(f"Total chunks processed: {total_chunks}")
            logger.info(f"Final vector store count: {final_count}")
            
            return final_count > 0
            
        except Exception as e:
            logger.error(f"Fatal error in document processing: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for testing the RAG system."""
    rag = RAGSystem()
    
    # Get collection stats
    stats = rag.get_collection_stats()
    print("Vector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check if collection is empty
    if stats.get("document_count", 0) == 0:
        print("\nCollection is empty. Would you like to process documents? (This may take a while)")
        print("Run: python -c \"from src.rag_system import RAGSystem; rag = RAGSystem(); rag.process_and_embed_all_documents()\"")
    else:
        # Test similarity search
        test_query = "What is GDP growth?"
        print(f"\nTesting similarity search with query: '{test_query}'")
        results = rag.similarity_search(test_query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Title: {result['metadata'].get('title', 'Unknown')}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Text preview: {result['text'][:100]}...")
        
        # Test full RAG pipeline if Ollama is available
        if rag.check_ollama_connection():
            print(f"\nTesting full RAG pipeline...")
            answer_result = rag.answer_question(test_query)
            print(f"Answer: {answer_result['answer']}")
        else:
            print("\nOllama not available. Install and run Ollama to test answer generation.")


if __name__ == "__main__":
    main() 