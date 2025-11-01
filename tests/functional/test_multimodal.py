import os
import shutil
import subprocess
import pytest
import requests
import lance

# URL of a sample PDF with images for testing
PDF_URL = "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
PDF_FILENAME = "sample_multimodal.pdf"
OUTPUT_DIR = "test_output"

@pytest.fixture(scope="module")
def setup_module():
    """Download the test PDF and create the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    response = requests.get(PDF_URL)
    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    yield

    # Teardown: remove the created directory and its contents
    shutil.rmtree(OUTPUT_DIR)

def run_cli_command(command):
    """Helper function to run a CLI command and return the output."""
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result

def test_ingest_pdf_default(setup_module):
    """Test default PDF ingestion (text only)."""
    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    output_lance_path = os.path.join(OUTPUT_DIR, "sample_multimodal.lance")

    # Run the ingest command
    run_cli_command([
        "synthetic-data-kit", "ingest", pdf_path, "--output-dir", OUTPUT_DIR
    ])

    # Verify the output
    assert os.path.exists(output_lance_path)

    # Check the contents of the Lance dataset
    dataset = lance.dataset(output_lance_path)
    assert dataset.count_rows() > 0

    # Verify schema and data
    schema = dataset.schema
    assert "text" in schema.names
    assert "image" not in schema.names

    table = dataset.to_table()
    text_column = table.column("text")
    assert all(text is not None and len(text.as_py()) > 0 for text in text_column)

def test_ingest_pdf_multimodal(setup_module):
    """Test multimodal PDF ingestion (text and images)."""
    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    output_lance_path = os.path.join(OUTPUT_DIR, "sample_multimodal.lance")

    # Clean up previous run if necessary
    if os.path.exists(output_lance_path):
        shutil.rmtree(output_lance_path)

    # Run the ingest command with the --multimodal flag
    run_cli_command([
        "synthetic-data-kit", "ingest", pdf_path, "--output-dir", OUTPUT_DIR, "--multimodal"
    ])

    # Verify the output
    assert os.path.exists(output_lance_path)

    # Check the contents of the Lance dataset
    dataset = lance.dataset(output_lance_path)
    assert dataset.count_rows() > 0

    # Verify schema and data
    schema = dataset.schema
    assert "text" in schema.names
    assert "image" in schema.names

    table = dataset.to_table()
    text_column = table.column("text")
    image_column = table.column("image")

    # Check that text and image data is not null where expected
    assert all(text is not None for text in text_column)

    # At least one image should be present in a multimodal PDF
    assert any(image is not None for image in image_column)


def test_ingest_image_file(setup_module):
    """Test ingestion of a single image file."""
    import base64
    
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = os.path.join(OUTPUT_DIR, "test_image.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    
    output_lance_path = os.path.join(OUTPUT_DIR, "test_image.lance")
    
    # Clean up if exists
    if os.path.exists(output_lance_path):
        import shutil
        shutil.rmtree(output_lance_path)
    
    # Run the ingest command
    run_cli_command([
        "synthetic-data-kit", "ingest", image_path, "--output-dir", OUTPUT_DIR
    ])
    
    # Verify the output
    assert os.path.exists(output_lance_path)
    
    # Check the contents of the Lance dataset
    dataset = lance.dataset(output_lance_path)
    assert dataset.count_rows() > 0
    
    # Verify schema and data
    schema = dataset.schema
    assert "text" in schema.names
    assert "image" in schema.names  # Should have multimodal schema
    
    table = dataset.to_table()
    text_column = table.column("text")
    image_column = table.column("image")
    
    # Verify text is empty and image is present
    assert text_column[0].as_py() == ""
    assert image_column[0].as_py() is not None
    
    # Verify image data is binary (not base64 string)
    image_data = image_column[0].as_py()
    assert isinstance(image_data, bytes), "Image should be stored as binary"


def test_ingest_image_directory(setup_module):
    """Test ingestion of a directory containing image files."""
    import base64
    
    # Create a subdirectory for images
    images_dir = os.path.join(OUTPUT_DIR, "test_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create multiple test image files
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    
    image_files = ["image1.png", "image2.jpg", "image3.jpeg"]
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
    
    # Run the ingest command on directory
    run_cli_command([
        "synthetic-data-kit", "ingest", images_dir, "--output-dir", OUTPUT_DIR, "--verbose"
    ])
    
    # Verify outputs exist
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        output_lance_path = os.path.join(OUTPUT_DIR, f"{base_name}.lance")
        assert os.path.exists(output_lance_path), f"Output should exist for {img_file}"
        
        # Verify schema
        dataset = lance.dataset(output_lance_path)
        schema = dataset.schema
        assert "text" in schema.names
        assert "image" in schema.names


def test_ingest_image_multimodal_flag(setup_module):
    """Test that --multimodal flag works with image files."""
    import base64
    
    # Create a test image file
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    image_path = os.path.join(OUTPUT_DIR, "test_image_multimodal.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    
    output_lance_path = os.path.join(OUTPUT_DIR, "test_image_multimodal.lance")
    
    # Clean up if exists
    if os.path.exists(output_lance_path):
        import shutil
        shutil.rmtree(output_lance_path)
    
    # Run with --multimodal flag
    run_cli_command([
        "synthetic-data-kit", "ingest", image_path, "--output-dir", OUTPUT_DIR, "--multimodal"
    ])
    
    # Verify the output
    assert os.path.exists(output_lance_path)
    
    # Verify schema has multimodal fields
    dataset = lance.dataset(output_lance_path)
    schema = dataset.schema
    assert "text" in schema.names
    assert "image" in schema.names


def test_ingest_image_preview_mode(setup_module):
    """Test preview mode with image files."""
    import base64
    
    # Create test images directory
    images_dir = os.path.join(OUTPUT_DIR, "preview_images")
    os.makedirs(images_dir, exist_ok=True)
    
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAiMBgUdQW6cAAAAASUVORK5CYII="
    )
    image_bytes = base64.b64decode(png_base64)
    
    image_files = ["preview1.png", "preview2.jpg"]
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
    
    # Run preview mode
    result = run_cli_command([
        "synthetic-data-kit", "ingest", images_dir, "--preview"
    ])
    
    # Verify preview output mentions image files
    assert "preview1.png" in result.stdout or "preview2.jpg" in result.stdout or "Supported files" in result.stdout
