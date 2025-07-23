import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.image_mcp.api_client import (
    InternVLClient,
    APIError,
    APIAuthError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    APINetworkError
)


class TestInternVLClient:
    
    @pytest.fixture
    def client(self, mock_api_key):
        return InternVLClient()
    
    def test_client_initialization(self, client, mock_api_key):
        """Test client initialization with API key."""
        assert client.api_key == mock_api_key
        assert client.endpoint == "https://chat.intern-ai.org.cn/api/v1/chat/completions"
        assert client.default_model == "internvl3-latest"
    
    @pytest.mark.asyncio
    async def test_stream_completion_success(self, client):
        """Test successful streaming completion."""
        # Mock response data
        mock_responses = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            'data: [DONE]\n'
        ]
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = AsyncMock(return_value=mock_responses)
        
        mock_client = MagicMock()
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            messages = [{"role": "user", "content": "Test message"}]
            chunks = []
            
            async for chunk in client.stream_completion(messages):
                chunks.append(chunk)
            
            assert chunks == ["Hello", " world"]
    
    @pytest.mark.asyncio
    async def test_stream_completion_auth_error(self, client):
        """Test authentication error handling."""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.stream.return_value.__aenter__.side_effect = \
                httpx.HTTPStatusError("Unauthorized", request=None, response=mock_response)
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(APIAuthError, match="Invalid API key"):
                async for _ in client.stream_completion(messages):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_completion_rate_limit_error(self, client):
        """Test rate limit error handling."""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.stream.return_value.__aenter__.side_effect = \
                httpx.HTTPStatusError("Rate limited", request=None, response=mock_response)
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(APIRateLimitError, match="Rate limit exceeded"):
                async for _ in client.stream_completion(messages):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_completion_server_error(self, client):
        """Test server error handling."""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.stream.return_value.__aenter__.side_effect = \
                httpx.HTTPStatusError("Server error", request=None, response=mock_response)
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(APIServerError, match="Server error: 500"):
                async for _ in client.stream_completion(messages):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_completion_timeout_error(self, client):
        """Test timeout error handling."""
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_client.stream.return_value.__aenter__.side_effect = httpx.TimeoutException("Timeout")
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(APITimeoutError, match="Request timeout"):
                async for _ in client.stream_completion(messages):
                    pass
    
    @pytest.mark.asyncio
    async def test_analyze_image(self, client, sample_image_data):
        """Test image analysis."""
        mock_responses = ['data: {"choices": [{"delta": {"content": "This is a test image"}}]}\n']
        
        with patch.object(client, 'stream_completion', return_value=iter(["This is a test image"])):
            image_data = [f"data:image/png;base64,{sample_image_data}"]
            prompt = "Describe this image"
            
            chunks = []
            async for chunk in client.analyze_image(image_data, prompt):
                chunks.append(chunk)
            
            assert chunks == ["This is a test image"]
    
    @pytest.mark.asyncio
    async def test_describe_image(self, client, sample_image_data):
        """Test image description."""
        with patch.object(client, 'analyze_image', return_value=iter(["A blue square image"])):
            image_data = [f"data:image/png;base64,{sample_image_data}"]
            
            chunks = []
            async for chunk in client.describe_image(image_data, detail_level="normal"):
                chunks.append(chunk)
            
            assert chunks == ["A blue square image"]
    
    @pytest.mark.asyncio
    async def test_extract_text(self, client, sample_image_data):
        """Test text extraction."""
        with patch.object(client, 'analyze_image', return_value=iter(["No text found"])):
            image_data = [f"data:image/png;base64,{sample_image_data}"]
            
            chunks = []
            async for chunk in client.extract_text(image_data):
                chunks.append(chunk)
            
            assert chunks == ["No text found"]
    
    @pytest.mark.asyncio
    async def test_compare_images(self, client, sample_image_data):
        """Test image comparison."""
        with patch.object(client, 'analyze_image', return_value=iter(["Images are similar"])):
            image_data = [
                f"data:image/png;base64,{sample_image_data}",
                f"data:image/png;base64,{sample_image_data}"
            ]
            
            chunks = []
            async for chunk in client.compare_images(image_data):
                chunks.append(chunk)
            
            assert chunks == ["Images are similar"]
    
    @pytest.mark.asyncio
    async def test_compare_images_insufficient(self, client):
        """Test image comparison with insufficient images."""
        with pytest.raises(ValueError, match="At least 2 images required"):
            async for _ in client.compare_images(["single_image"]):
                pass
    
    @pytest.mark.asyncio
    async def test_identify_objects(self, client, sample_image_data):
        """Test object identification."""
        with patch.object(client, 'analyze_image', return_value=iter(["Found: rectangular shape"])):
            image_data = [f"data:image/png;base64,{sample_image_data}"]
            
            chunks = []
            async for chunk in client.identify_objects(image_data):
                chunks.append(chunk)
            
            assert chunks == ["Found: rectangular shape"]