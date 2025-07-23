import pytest
import os
import tempfile
from PIL import Image
import io
import base64


@pytest.fixture
def sample_image_path():
    """Create a sample PNG image for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a simple 100x100 red image
        image = Image.new('RGB', (100, 100), color='red')
        image.save(f.name, 'PNG')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_image_data():
    """Create sample image data as base64."""
    image = Image.new('RGB', (100, 100), color='blue')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


@pytest.fixture
def large_image_path():
    """Create a large image for compression testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a 3000x3000 image (larger than max_dimension)
        image = Image.new('RGB', (3000, 3000), color='green')
        image.save(f.name, 'PNG')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock API key for testing."""
    monkeypatch.setenv("INTERNVL_API_KEY", "test_api_key_12345")
    return "test_api_key_12345"