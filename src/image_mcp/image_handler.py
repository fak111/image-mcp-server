import os
import base64
import io
from typing import List, Optional, Union
from urllib.parse import urlparse
import httpx
from PIL import Image
import subprocess
from .config import settings


class ImageHandler:
    SUPPORTED_FORMATS = {'PNG', 'JPEG', 'JPG', 'WEBP', 'GIF', 'BMP'}
    
    def __init__(self):
        self.max_size_mb = settings.max_image_size_mb
        self.compression_threshold_mb = settings.compression_threshold_mb
        self.max_dimension = settings.max_image_dimension
        self.max_images = settings.max_images_per_request

    async def process_images(self, image_sources: List[str]) -> List[str]:
        """Process multiple images from various sources and return base64 encoded strings."""
        if len(image_sources) > self.max_images:
            raise ValueError(f"Maximum {self.max_images} images allowed per request")
        
        processed_images = []
        for source in image_sources:
            image_data = await self._process_single_image(source)
            processed_images.append(image_data)
        
        return processed_images

    async def _process_single_image(self, source: str) -> str:
        """Process a single image from various sources."""
        # Determine source type and get image data
        if source.lower() == "clipboard":
            image_data = self._get_clipboard_image()
        elif self._is_url(source):
            image_data = await self._download_image(source)
        elif os.path.isfile(source):
            image_data = self._read_local_image(source)
        else:
            raise ValueError(f"Invalid image source: {source}")
        
        # Validate and process the image
        return self._process_image_data(image_data)

    def _is_url(self, source: str) -> bool:
        """Check if the source is a valid URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def _download_image(self, url: str) -> bytes:
        """Download image from URL."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image: {content_type}")
            
            return response.content

    def _read_local_image(self, file_path: str) -> bytes:
        """Read image from local file system."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            raise ValueError(f"Image file too large: {file_size_mb:.1f}MB > {self.max_size_mb}MB")
        
        with open(file_path, 'rb') as f:
            return f.read()

    def _get_clipboard_image(self) -> bytes:
        """Get image from system clipboard."""
        try:
            # Try to get image from clipboard using PIL
            from PIL import ImageGrab
            image = ImageGrab.grabclipboard()
            if image is None:
                raise ValueError("No image found in clipboard")
            
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        except ImportError:
            # Fallback for systems without ImageGrab
            try:
                # macOS
                result = subprocess.run(['pbpaste'], capture_output=True)
                if result.returncode == 0 and result.stdout:
                    return result.stdout
            except FileNotFoundError:
                pass
            
            raise ValueError("Clipboard image access not supported on this system")

    def _process_image_data(self, image_data: bytes) -> str:
        """Process image data: validate, compress if needed, and encode to base64."""
        # Validate image format
        try:
            image = Image.open(io.BytesIO(image_data))
            format_name = image.format
            if format_name not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {format_name}")
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

        # Check if compression is needed
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > self.compression_threshold_mb:
            image_data = self._compress_image(image_data)

        # Get MIME type
        mime_type = self._get_mime_type(image.format)
        
        # Encode to base64
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_data}"

    def _compress_image(self, image_data: bytes) -> bytes:
        """Compress image to reduce file size."""
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if too large
        if max(image.size) > self.max_dimension:
            ratio = self.max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save with compression
        buffer = io.BytesIO()
        if image.format == 'PNG':
            image.save(buffer, format='PNG', optimize=True)
        else:
            # Convert to JPEG for better compression
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert transparency to white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            image.save(buffer, format='JPEG', quality=85, optimize=True)
        
        return buffer.getvalue()

    def _get_mime_type(self, format_name: str) -> str:
        """Get MIME type from image format."""
        mime_types = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'WEBP': 'image/webp',
            'GIF': 'image/gif',
            'BMP': 'image/bmp'
        }
        return mime_types.get(format_name, 'image/jpeg')