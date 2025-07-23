import pytest
import os
import tempfile
from PIL import Image
from src.image_mcp.image_handler import ImageHandler


class TestImageHandler:
    
    @pytest.fixture
    def handler(self):
        return ImageHandler()
    
    def test_is_url(self, handler):
        """Test URL detection."""
        assert handler._is_url("https://example.com/image.jpg") == True
        assert handler._is_url("http://example.com/image.png") == True
        assert handler._is_url("/local/path/image.jpg") == False
        assert handler._is_url("image.png") == False
        assert handler._is_url("not_a_url") == False
    
    def test_read_local_image(self, handler, sample_image_path):
        """Test reading local image file."""
        image_data = handler._read_local_image(sample_image_path)
        assert isinstance(image_data, bytes)
        assert len(image_data) > 0
    
    def test_read_nonexistent_image(self, handler):
        """Test reading non-existent image file."""
        with pytest.raises(FileNotFoundError):
            handler._read_local_image("/nonexistent/path/image.jpg")
    
    def test_get_mime_type(self, handler):
        """Test MIME type detection."""
        assert handler._get_mime_type("PNG") == "image/png"
        assert handler._get_mime_type("JPEG") == "image/jpeg"
        assert handler._get_mime_type("JPG") == "image/jpeg"
        assert handler._get_mime_type("WEBP") == "image/webp"
        assert handler._get_mime_type("GIF") == "image/gif"
        assert handler._get_mime_type("BMP") == "image/bmp"
        assert handler._get_mime_type("UNKNOWN") == "image/jpeg"  # Default
    
    def test_process_image_data_small(self, handler, sample_image_path):
        """Test processing small image that doesn't need compression."""
        with open(sample_image_path, 'rb') as f:
            image_data = f.read()
        
        result = handler._process_image_data(image_data)
        assert result.startswith("data:image/png;base64,")
        
        # Decode and verify it's still a valid image
        import base64
        base64_data = result.split(",")[1]
        decoded_data = base64.b64decode(base64_data)
        
        # Should be able to open the decoded image
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(decoded_data)
            temp_file.flush()
            
            test_image = Image.open(temp_file.name)
            assert test_image.format in ['PNG', 'JPEG']
        
        os.unlink(temp_file.name)
    
    def test_compress_image(self, handler, large_image_path):
        """Test image compression for large images."""
        with open(large_image_path, 'rb') as f:
            original_data = f.read()
        
        compressed_data = handler._compress_image(original_data)
        
        # Compressed data should be smaller
        assert len(compressed_data) < len(original_data)
        
        # Should still be a valid image
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(compressed_data)
            temp_file.flush()
            
            test_image = Image.open(temp_file.name)
            
            # Should be resized to within max_dimension
            assert max(test_image.size) <= handler.max_dimension
        
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_process_single_image_local(self, handler, sample_image_path):
        """Test processing a single local image."""
        result = await handler._process_single_image(sample_image_path)
        assert result.startswith("data:image/png;base64,")
    
    @pytest.mark.asyncio
    async def test_process_images_multiple(self, handler, sample_image_path):
        """Test processing multiple images."""
        # Create a second test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image = Image.new('RGB', (50, 50), color='yellow')
            image.save(f.name, 'PNG')
            second_image_path = f.name
        
        try:
            sources = [sample_image_path, second_image_path]
            results = await handler.process_images(sources)
            
            assert len(results) == 2
            assert all(result.startswith("data:image/png;base64,") for result in results)
        finally:
            os.unlink(second_image_path)
    
    @pytest.mark.asyncio
    async def test_process_images_too_many(self, handler, sample_image_path):
        """Test processing too many images should raise error."""
        sources = [sample_image_path] * (handler.max_images + 1)
        
        with pytest.raises(ValueError, match="Maximum .* images allowed"):
            await handler.process_images(sources)
    
    @pytest.mark.asyncio
    async def test_process_invalid_source(self, handler):
        """Test processing invalid image source."""
        with pytest.raises(ValueError, match="Invalid image source"):
            await handler._process_single_image("invalid_source_type")
    
    def test_process_invalid_image_data(self, handler):
        """Test processing invalid image data."""
        invalid_data = b"This is not image data"
        
        with pytest.raises(ValueError, match="Invalid image data"):
            handler._process_image_data(invalid_data)