"""Unit tests for ingest module."""

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import pytest
import pyarrow as pa

from synthetic_data_kit.core.ingest import determine_parser, process_file
from synthetic_data_kit.parsers.image_parser import ImageParser


@pytest.mark.unit
def test_determine_parser_image_file(tmp_path):
    """Test that determine_parser correctly identifies image files."""
    from synthetic_data_kit.parsers.pdf_parser import PDFParser
    from synthetic_data_kit.parsers.txt_parser import TXTParser
    
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    config = {}
    
    # Test image file detection
    parser = determine_parser(str(image_path), config, multimodal=False)
    assert isinstance(parser, ImageParser)
    
    # Test image file with multimodal flag
    parser_multimodal = determine_parser(str(image_path), config, multimodal=True)
    assert isinstance(parser_multimodal, ImageParser)


@pytest.mark.unit
def test_determine_parser_all_image_extensions(tmp_path):
    """Test that all supported image extensions are detected."""
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    config = {}
    
    for ext in ImageParser.SUPPORTED_EXTENSIONS:
        image_path = tmp_path / f"test{ext}"
        image_path.write_bytes(image_bytes)
        
        parser = determine_parser(str(image_path), config, multimodal=False)
        assert isinstance(parser, ImageParser), f"Failed to detect {ext} extension"


@pytest.mark.unit
def test_determine_parser_image_vs_other_formats(tmp_path):
    """Test that image files are correctly distinguished from other formats."""
    from synthetic_data_kit.parsers.txt_parser import TXTParser
    from synthetic_data_kit.parsers.pdf_parser import PDFParser
    
    config = {}
    
    # Create a text file
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("Sample text")
    
    # Create an image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    # Text file should use TXTParser
    txt_parser = determine_parser(str(txt_path), config, multimodal=False)
    assert isinstance(txt_parser, TXTParser)
    
    # Image file should use ImageParser
    img_parser = determine_parser(str(image_path), config, multimodal=False)
    assert isinstance(img_parser, ImageParser)


@pytest.mark.unit
def test_process_file_image_creates_multimodal_schema(tmp_path):
    """Test that processing an image file creates a Lance dataset with multimodal schema."""
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Process the image file
    output_path = process_file(
        str(image_path),
        output_dir=str(output_dir),
        config=config,
        multimodal=False  # Should still use multimodal schema due to image detection
    )
    
    # Verify output exists
    assert os.path.exists(output_path)
    
    # Load and verify Lance dataset schema
    import lance
    dataset = lance.dataset(output_path)
    schema = dataset.schema
    
    # Should have both text and image fields (multimodal schema)
    assert "text" in schema.names
    assert "image" in schema.names
    assert schema.field("image").type == pa.binary()


@pytest.mark.unit
def test_process_file_image_base64_to_binary_conversion(tmp_path):
    """Test that ImageParser's base64 output is converted to binary for Lance storage."""
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Process the image file
    output_path = process_file(
        str(image_path),
        output_dir=str(output_dir),
        config=config,
        multimodal=False
    )
    
    # Load the Lance dataset and verify image data is binary
    import lance
    dataset = lance.dataset(output_path)
    table = dataset.to_table()
    
    # Get the first row
    first_row = table[0]
    image_data = first_row["image"].as_py()
    
    # Verify it's binary (bytes), not a base64 string
    assert isinstance(image_data, bytes), "Image data should be binary bytes, not base64 string"
    assert image_data == image_bytes, "Image bytes should match original"


@pytest.mark.unit
def test_process_file_image_multimodal_flag(tmp_path):
    """Test that processing image with multimodal flag works correctly."""
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Process with multimodal flag
    output_path = process_file(
        str(image_path),
        output_dir=str(output_dir),
        config=config,
        multimodal=True
    )
    
    # Verify output exists and has correct schema
    assert os.path.exists(output_path)
    
    import lance
    dataset = lance.dataset(output_path)
    schema = dataset.schema
    
    assert "text" in schema.names
    assert "image" in schema.names


@pytest.mark.unit
def test_process_file_image_empty_text_field(tmp_path):
    """Test that image files result in empty text field."""
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "test.png"
    image_path.write_bytes(image_bytes)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Process the image file
    output_path = process_file(
        str(image_path),
        output_dir=str(output_dir),
        config=config,
        multimodal=False
    )
    
    # Load and verify text field is empty
    import lance
    dataset = lance.dataset(output_path)
    table = dataset.to_table()
    
    first_row = table[0]
    text_data = first_row["text"].as_py()
    
    assert text_data == "", "Text field should be empty for image-only content"


@pytest.mark.unit
def test_process_file_image_different_formats(tmp_path):
    """Test processing different image formats."""
    # Create a minimal valid image for different formats
    # Using a simple 1x1 PNG that works for all formats
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Test a few common formats
    test_formats = [".png", ".jpg", ".jpeg"]
    
    for ext in test_formats:
        image_path = tmp_path / f"test{ext}"
        image_path.write_bytes(image_bytes)
        
        # Process the image file
        output_path = process_file(
            str(image_path),
            output_dir=str(output_dir),
            output_name=f"test{ext}",
            config=config,
            multimodal=False
        )
        
        # Verify output exists
        assert os.path.exists(output_path), f"Output should exist for {ext}"
        
        # Verify it's a Lance dataset
        import lance
        dataset = lance.dataset(output_path)
        assert dataset.count_rows() > 0, f"Dataset should have rows for {ext}"


@pytest.mark.unit
def test_process_file_image_custom_output_name(tmp_path):
    """Test processing image with custom output name."""
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = tmp_path / "original_image.png"
    image_path.write_bytes(image_bytes)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {}
    
    # Process with custom output name
    output_path = process_file(
        str(image_path),
        output_dir=str(output_dir),
        output_name="custom_name",
        config=config,
        multimodal=False
    )
    
    # Verify output has custom name
    assert "custom_name.lance" in output_path
    assert os.path.exists(output_path)


@pytest.mark.unit
def test_process_file_image_nonexistent_file():
    """Test that processing nonexistent image file raises FileNotFoundError."""
    config = {}
    
    with pytest.raises(FileNotFoundError):
        process_file(
            "/nonexistent/path/image.png",
            output_dir="/tmp",
            config=config,
            multimodal=False
        )


@pytest.mark.unit
def test_process_file_image_unsupported_extension(tmp_path):
    """Test that unsupported image extension raises ValueError."""
    config = {}
    
    # Create a file with unsupported extension
    fake_file = tmp_path / "test.xyz"
    fake_file.write_text("not an image")
    
    # Should raise ValueError when determining parser
    with pytest.raises(ValueError, match="Unsupported file extension"):
        determine_parser(str(fake_file), config, multimodal=False)

