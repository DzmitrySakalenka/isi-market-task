"""
Unit tests for the data processing module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.data_processing import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor("test_data")
        assert processor.data_dir == Path("test_data")
        assert processor.metadata_file == Path("test_data/metadata.jsonl")
        assert processor.text_splitter is not None
    
    def test_load_metadata_success(self, tmp_path):
        """Test successful metadata loading."""
        # Create test metadata file
        metadata_file = tmp_path / "metadata.jsonl"
        test_data = [
            {"uuid": "test-uuid-1", "title": "Test Document 1", "date": "2024-01-01"},
            {"uuid": "test-uuid-2", "title": "Test Document 2", "date": "2024-01-02"}
        ]
        
        with open(metadata_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        processor = DataProcessor(str(tmp_path))
        metadata = processor.load_metadata()
        
        assert len(metadata) == 2
        assert "test-uuid-1" in metadata
        assert metadata["test-uuid-1"]["title"] == "Test Document 1"
        assert "test-uuid-2" in metadata
        assert metadata["test-uuid-2"]["title"] == "Test Document 2"
    
    def test_load_metadata_file_not_found(self, tmp_path):
        """Test metadata loading when file doesn't exist."""
        processor = DataProcessor(str(tmp_path))
        
        with pytest.raises(FileNotFoundError):
            processor.load_metadata()
    
    def test_load_metadata_invalid_json(self, tmp_path):
        """Test metadata loading with invalid JSON."""
        metadata_file = tmp_path / "metadata.jsonl"
        
        with open(metadata_file, 'w') as f:
            f.write('{"uuid": "test-1", "title": "Valid"}\n')
            f.write('invalid json line\n')
            f.write('{"uuid": "test-2", "title": "Also Valid"}\n')
        
        processor = DataProcessor(str(tmp_path))
        metadata = processor.load_metadata()
        
        # Should load valid entries and skip invalid ones
        assert len(metadata) == 2
        assert "test-1" in metadata
        assert "test-2" in metadata
    
    @patch('pdfplumber.open')
    def test_extract_text_from_pdf_success(self, mock_pdfplumber):
        """Test successful PDF text extraction."""
        # Mock PDF pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        processor = DataProcessor()
        text = processor.extract_text_from_pdf(Path("test.pdf"))
        
        assert text == "Page 1 content\nPage 2 content"
    
    @patch('pdfplumber.open')
    def test_extract_text_from_pdf_empty_pages(self, mock_pdfplumber):
        """Test PDF text extraction with empty pages."""
        mock_page = Mock()
        mock_page.extract_text.return_value = None
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        processor = DataProcessor()
        text = processor.extract_text_from_pdf(Path("test.pdf"))
        
        assert text == ""
    
    @patch('pdfplumber.open')
    def test_extract_text_from_pdf_exception(self, mock_pdfplumber):
        """Test PDF text extraction with exception."""
        mock_pdfplumber.side_effect = Exception("PDF error")
        
        processor = DataProcessor()
        text = processor.extract_text_from_pdf(Path("test.pdf"))
        
        assert text == ""
    
    def test_extract_uuid_from_filename(self):
        """Test UUID extraction from filename."""
        processor = DataProcessor()
        processor.metadata_dict = {"test-uuid": {"title": "Test"}}
        
        # Valid UUID
        uuid = processor.extract_uuid_from_filename(Path("test-uuid.pdf"))
        assert uuid == "test-uuid"
        
        # Invalid UUID (not in metadata)
        uuid = processor.extract_uuid_from_filename(Path("invalid-uuid.pdf"))
        assert uuid is None
    
    def test_chunk_text_with_metadata(self):
        """Test text chunking with metadata preservation."""
        processor = DataProcessor()
        
        text = "This is a test document. " * 100  # Long text to ensure chunking
        metadata = {"title": "Test Document", "date": "2024-01-01"}
        uuid = "test-uuid"
        
        chunks = processor.chunk_text_with_metadata(text, metadata, uuid)
        
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["title"] == "Test Document"
            assert chunk["metadata"]["chunk_id"] == f"test-uuid_chunk_{i}"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["source_uuid"] == "test-uuid"
    
    def test_chunk_text_empty(self):
        """Test chunking with empty text."""
        processor = DataProcessor()
        
        chunks = processor.chunk_text_with_metadata("", {}, "test-uuid")
        assert chunks == []
        
        chunks = processor.chunk_text_with_metadata("   ", {}, "test-uuid")
        assert chunks == []
    
    def test_get_document_stats(self, tmp_path):
        """Test document statistics."""
        # Create some test files
        (tmp_path / "test1.pdf").touch()
        (tmp_path / "test2.pdf").touch()
        (tmp_path / "not_pdf.txt").touch()
        
        processor = DataProcessor(str(tmp_path))
        stats = processor.get_document_stats()
        
        assert stats["total_pdf_files"] == 2
        assert stats["data_directory"] == str(tmp_path)


@pytest.fixture
def sample_processor():
    """Fixture providing a DataProcessor instance."""
    return DataProcessor("test_data") 