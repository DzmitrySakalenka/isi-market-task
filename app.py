"""
Flask web application for the Knowledge Base Aware Question Answering System.

This application provides:
1. A web interface for users to input questions
2. API endpoints for question processing
3. Integration with the RAG system for answer generation
"""

import logging
import os
from flask import Flask, render_template, request, jsonify
from pathlib import Path

from src.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG system
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system."""
    global rag_system
    try:
        rag_system = RAGSystem()
        stats = rag_system.get_collection_stats()
        logger.info(f"RAG system initialized. Documents in vector store: {stats.get('document_count', 0)}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API endpoint to check system status."""
    if not rag_system:
        return jsonify({
            "status": "error",
            "message": "RAG system not initialized"
        }), 500
    
    try:
        stats = rag_system.get_collection_stats()
        ollama_available = rag_system.check_ollama_connection()
        
        return jsonify({
            "status": "ok",
            "vector_store": stats,
            "ollama_available": ollama_available,
            "message": "System is ready" if stats.get('document_count', 0) > 0 else "No documents in vector store"
        })
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to process questions and return answers."""
    if not rag_system:
        return jsonify({
            "error": "RAG system not initialized"
        }), 500
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body"
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        # Get optional parameters
        k = data.get('k', 5)  # Number of documents to retrieve
        filters = data.get('filters', None)  # Metadata filters
        
        # Validate k parameter
        if not isinstance(k, int) or k < 1 or k > 20:
            k = 5
        
        logger.info(f"Processing question: {question}")
        
        # Process question through RAG system
        result = rag_system.answer_question(question, k=k, filters=filters)
        
        # Return response
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """API endpoint for similarity search without answer generation."""
    if not rag_system:
        return jsonify({
            "error": "RAG system not initialized"
        }), 500
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Get optional parameters
        k = data.get('k', 5)
        filters = data.get('filters', None)
        
        # Validate k parameter
        if not isinstance(k, int) or k < 1 or k > 20:
            k = 5
        
        logger.info(f"Performing similarity search: {query}")
        
        # Perform similarity search
        results = rag_system.similarity_search(query, k=k, filters=filters)
        
        # Format response
        formatted_results = []
        for result in results:
            metadata = result['metadata']
            
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
            
            formatted_result = {
                "text": result['text'],
                "metadata": {
                    "title": metadata.get('title', 'Unknown'),
                    "date": metadata.get('date', 'Unknown'),
                    "country_codes": country_codes,
                    "industries": industries
                },
                "similarity_score": 1 - result['distance']
            }
            formatted_results.append(formatted_result)
        
        return jsonify({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/process_documents', methods=['POST'])
def process_documents():
    """API endpoint to process and embed all documents."""
    if not rag_system:
        return jsonify({
            "error": "RAG system not initialized"
        }), 500
    
    try:
        logger.info("Starting document processing and embedding")
        
        # This is a long-running operation
        success = rag_system.process_and_embed_all_documents()
        
        if success:
            stats = rag_system.get_collection_stats()
            return jsonify({
                "status": "success",
                "message": "Documents processed and embedded successfully",
                "document_count": stats.get('document_count', 0)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to process documents"
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

def create_directories():
    """Create necessary directories."""
    directories = ['templates', 'static', 'vector_store']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """Main function to run the Flask application."""
    # Create necessary directories
    create_directories()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    if not initialize_rag_system():
        logger.error("Failed to initialize RAG system. Exiting.")
        return
    
    # Run Flask app
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 