import json
import httpx
from typing import List, Dict, Any, AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings


class InternVLClient:
    def __init__(self):
        self.api_key = settings.internvl_api_key
        self.endpoint = settings.internvl_api_endpoint
        self.default_model = settings.default_model
        self.default_temperature = settings.default_temperature
        self.default_top_p = settings.default_top_p
        self.max_tokens = settings.max_tokens

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3)
    )
    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None
    ) -> AsyncGenerator[str, None]:
        """Stream completion from InternVL API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        json_data = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "top_p": top_p or self.default_top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream(
                    "POST", 
                    self.endpoint, 
                    headers=headers, 
                    json=json_data
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    continue
                                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise APIAuthError("Invalid API key")
                elif e.response.status_code == 403:
                    raise APIAuthError("API access forbidden")
                elif e.response.status_code == 429:
                    raise APIRateLimitError("Rate limit exceeded")
                elif e.response.status_code >= 500:
                    raise APIServerError(f"Server error: {e.response.status_code}")
                else:
                    raise APIError(f"HTTP error: {e.response.status_code}")
            except httpx.TimeoutException:
                raise APITimeoutError("Request timeout")
            except httpx.RequestError as e:
                raise APINetworkError(f"Network error: {str(e)}")

    async def analyze_image(
        self,
        image_data: List[str],
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Analyze images with a text prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add images to the message content
        for img_data in image_data:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": img_data}
            })
        
        async for chunk in self.stream_completion(messages, **kwargs):
            yield chunk

    async def describe_image(
        self,
        image_data: List[str],
        detail_level: str = "normal",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate detailed description of images."""
        detail_prompts = {
            "brief": "Briefly describe what you see in this image.",
            "normal": "Describe this image in detail, including objects, people, actions, colors, and setting.",
            "detailed": "Provide a comprehensive description of this image, including all visible elements, their relationships, emotions (if people are present), lighting, composition, and any text or symbols visible."
        }
        
        prompt = detail_prompts.get(detail_level, detail_prompts["normal"])
        async for chunk in self.analyze_image(image_data, prompt, **kwargs):
            yield chunk

    async def extract_text(
        self,
        image_data: List[str],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Extract text from images using OCR capabilities."""
        prompt = "Extract all visible text from this image. Present the text exactly as it appears, maintaining any formatting or structure when possible. If no text is visible, respond with 'No text found'."
        
        async for chunk in self.analyze_image(image_data, prompt, **kwargs):
            yield chunk

    async def compare_images(
        self,
        image_data: List[str],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Compare multiple images and identify similarities and differences."""
        if len(image_data) < 2:
            raise ValueError("At least 2 images required for comparison")
        
        prompt = "Compare these images and identify the similarities and differences. Focus on objects, people, settings, colors, composition, and any other notable elements."
        
        async for chunk in self.analyze_image(image_data, prompt, **kwargs):
            yield chunk

    async def identify_objects(
        self,
        image_data: List[str],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Identify and list objects in the images."""
        prompt = "Identify and list all the main objects, people, and elements visible in this image. For each item, provide its approximate location or position in the image if relevant."
        
        async for chunk in self.analyze_image(image_data, prompt, **kwargs):
            yield chunk


class APIError(Exception):
    """Base API error."""
    pass


class APIAuthError(APIError):
    """Authentication error."""
    pass


class APIRateLimitError(APIError):
    """Rate limit error."""
    pass


class APIServerError(APIError):
    """Server error."""
    pass


class APITimeoutError(APIError):
    """Timeout error."""
    pass


class APINetworkError(APIError):
    """Network error."""
    pass